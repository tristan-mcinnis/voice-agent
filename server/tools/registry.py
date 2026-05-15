"""Tool registry — declarative tool classes, handler wiring, schema export.

Adding a tool: write a `BaseTool` subclass, decorate with `@REGISTRY.register`,
set `name` / `description` / `parameters` / `required` / optional `speak_text`
/ `category`, and implement `execute(**kwargs)`. The registry wires it onto the
LLM automatically via `register_handlers(llm)`.

The sync `execute` runs in `asyncio.to_thread` — every tool does blocking I/O
(screen capture, webcam read, file ops, network calls) and must not run on the
asyncio event loop or it starves the Soniox STT/TTS WebSocket heartbeats.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import TTSSpeakFrame


class BaseTool:
    """Declarative tool: name + description + JSON-schema params + execute().

    Subclasses override `execute(**kwargs)`; the engine takes care of
    threading, error handling, spoken filler, and session logging.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    speak_text: Optional[str] = None
    # Coarse grouping for the auto-generated capability summary in the system
    # prompt. Pick one of: input, files, browser, vision, system, web, demo.
    category: str = "misc"
    # Optional usage guidance injected into the system prompt. The tool author
    # owns this — PromptBuilder collects it from the registry rather than
    # hard-coding tool-specific prose. Set to None for no guidance.
    guidance: Optional[str] = None

    def execute(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> FunctionSchema:
        return FunctionSchema(
            name=cls.name,
            description=cls.description,
            properties=cls.parameters,
            required=cls.required,
        )


class ToolRegistry:
    """Holds tool instances, hands out schemas, registers handlers on an LLM."""

    # Stable category order in the capability summary — read top-to-bottom so
    # related tools cluster naturally for the LLM.
    _CATEGORY_ORDER = (
        "input", "files", "browser", "vision", "system", "web", "agents", "demo", "misc",
    )

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool_cls: type[BaseTool]) -> type[BaseTool]:
        if not tool_cls.name:
            raise ValueError(f"{tool_cls.__name__} must set `name`")
        if tool_cls.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool_cls.name}")
        self._tools[tool_cls.name] = tool_cls()
        return tool_cls

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def schemas(self) -> list[FunctionSchema]:
        return [t.to_schema() for t in self.all()]

    def by_category(self) -> dict[str, list[BaseTool]]:
        groups: dict[str, list[BaseTool]] = {}
        for tool in self.all():
            groups.setdefault(tool.category, []).append(tool)
        # Stable order: known categories first, then anything unexpected.
        ordered: dict[str, list[BaseTool]] = {}
        for cat in self._CATEGORY_ORDER:
            if cat in groups:
                ordered[cat] = groups.pop(cat)
        ordered.update(groups)
        return ordered

    def capabilities_summary(self) -> str:
        """One-line-per-category tool inventory for the system prompt.

        Rendered by ``PromptBuilder`` into the ``# Available Tools`` layer
        so the LLM knows what's available without the prompt hard-coding
        tool names.
        """
        lines: list[str] = []
        for cat, tools in self.by_category().items():
            names = ", ".join(t.name for t in tools)
            lines.append(f"- {cat}: {names}")
        return "\n".join(lines)

    def register_handlers(self, llm, session_log=None) -> list[FunctionSchema]:
        """Register every tool on the given LLM. Returns the FunctionSchemas.

        Pass `session_log` (a `session_log.SessionLog`) to record tool calls,
        results, and errors as kebab-case events alongside user/bot speech.
        """
        for tool in self.all():
            llm.register_function(tool.name, _make_handler(tool, session_log=session_log))
        return self.schemas()


REGISTRY = ToolRegistry()


def _make_handler(tool: BaseTool, session_log=None) -> Callable:
    """Wrap a tool's `execute` as a Pipecat function-call handler.

    The sync `execute` runs inside `asyncio.to_thread`, which is critical:
    every tool here does blocking I/O (screen capture, webcam read, network
    calls). Running them on the asyncio event loop directly starves the
    Soniox STT/TTS WebSockets of heartbeats and the connections drop after
    ~30s with `Error: 408 _receive_messages - Request timeout`.

    Result-shaping rule: dict results pass through unchanged (the LLM sees
    them as structured data); any other result is wrapped in `{"result": value}`.

    Argument filtering: the LLM occasionally hallucinates parameter names. We
    drop unknown keys against the tool's declared `parameters` schema instead
    of catching `TypeError`, which used to silently mask real bugs in
    `execute()` by retrying with zero args.
    """
    known_params = set(tool.parameters.keys())

    async def handler(params):
        raw_args = params.arguments or {}
        args = {k: v for k, v in raw_args.items() if k in known_params}
        dropped = set(raw_args) - known_params
        if dropped:
            logger.warning(
                f"tool {tool.name}: dropping unknown arg(s) {sorted(dropped)} "
                f"(declared: {sorted(known_params)})"
            )
        if session_log is not None:
            session_log.event("tool-called", name=tool.name, args=args)

        if tool.speak_text:
            await params.llm.push_frame(TTSSpeakFrame(tool.speak_text))
        try:
            result = await asyncio.to_thread(tool.execute, **args)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if session_log is not None:
                session_log.event("tool-error", name=tool.name, error=err)
            await params.result_callback({"result": f"Tool {tool.name} failed: {err}"})
            return

        if session_log is not None:
            session_log.event("tool-result", name=tool.name, result=result)
        if isinstance(result, dict):
            await params.result_callback(result)
        else:
            await params.result_callback({"result": result})

    return handler
