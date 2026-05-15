"""External-agent tools — consult other LLMs and spawn coding agents from voice.

The provider catalog lives in ``config.yaml`` under ``agents:``. The dynamic
factory in this module walks that catalog at registration time and emits one
``BaseTool`` per entry, so adding a new model or CLI is a YAML-only change.

Three tiers, all registered on the same ``REGISTRY`` and grouped under the
``agents`` category in the capability summary:

Tier 1 — sync "ask another model" (one tool per ``agents.consult.<key>``):
    ask_<key>              OpenAI-compatible chat call, returns text.

Tier 2 — fire-and-forget coding agents (one tool per ``agents.coding.<key>``):
    spawn_<key>            forks a CLI subprocess, returns a ``task_id``.

Tier 3 — Hermes peer (static, single instance — there is only one Hermes):
    hermes_chat            sync chat call.
    hermes_spawn           async task POST.

Task management (shared across every spawn tool):
    list_agent_tasks, get_agent_task, cancel_agent_task.

Missing api keys / missing binaries are non-fatal: the tool stays registered
(so the LLM knows it exists) and returns a useful error string when called.
That mirrors the vision-chain graceful-degradation pattern.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

from config import get_config
from tools.agent_runner import (
    TaskStore,
    cancel as runner_cancel,
    reconcile_status,
    spawn as runner_spawn,
)
from tools.registry import BaseTool, REGISTRY


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _agents_cfg() -> Any:
    """Return ``Config.agents`` or ``None`` so tools degrade rather than crash."""
    try:
        return get_config().agents
    except (RuntimeError, AttributeError):
        return None


def _openai_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    extra_body: Optional[dict] = None,
) -> str:
    """Minimal OpenAI-compatible chat completion. Returns the assistant text.

    Reasoning-model handling: some providers return the final answer in
    ``message.content``, others in ``message.reasoning_content``. Try both.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if extra_body:
        payload.update(extra_body)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = (data.get("choices") or [{}])[0].get("message", {})
    return msg.get("content") or msg.get("reasoning_content") or ""


def _consult(provider_cfg, prompt: str) -> str:
    """Tier 1 dispatch — calls the named consult provider."""
    if provider_cfg is None:
        return "Consult unavailable: this provider is not configured."
    api_key = ""
    if provider_cfg.api_key_env:
        api_key = os.getenv(provider_cfg.api_key_env, "")
        if not api_key:
            return (
                f"Consult unavailable: {provider_cfg.api_key_env} is not set "
                f"in environment."
            )
    try:
        return _openai_chat(
            base_url=provider_cfg.base_url,
            api_key=api_key,
            model=provider_cfg.model,
            prompt=prompt,
            timeout=provider_cfg.timeout,
            extra_body=provider_cfg.extra,
        ).strip() or "(empty response)"
    except httpx.HTTPError as exc:
        return f"Consult failed ({provider_cfg.name}): {exc}"


def _binary_path(bin_name: str) -> Optional[str]:
    """Resolve a binary by absolute path or PATH lookup. ``None`` if missing."""
    if not bin_name:
        return None
    if os.path.sep in bin_name and os.path.isfile(bin_name):
        return bin_name
    return shutil.which(bin_name)


def _default_cwd(cwd: Optional[str]) -> str:
    """Normalise the ``cwd:`` arg into a real directory. Defaults to project root."""
    if cwd:
        p = Path(cwd).expanduser()
    else:
        p = Path(__file__).resolve().parent.parent.parent
    if not p.exists() or not p.is_dir():
        return str(Path.cwd())
    return str(p)


_TITLE_RE = re.compile(r"[_-]+")


def _humanise(key: str) -> str:
    """``deepseek_reasoner`` → ``Deepseek Reasoner`` for default speak/description text."""
    return _TITLE_RE.sub(" ", key).strip().title()


# ---------------------------------------------------------------------------
# Tier 1 — Consult: one dynamic tool per `agents.consult.<key>`
# ---------------------------------------------------------------------------


def _make_consult_tool(key: str, provider_cfg) -> type[BaseTool]:
    """Build a ``BaseTool`` subclass for one consult provider."""
    friendly = provider_cfg.name or _humanise(key)
    description = provider_cfg.description or (
        f"Ask {friendly} ({provider_cfg.model}) for a second opinion. "
        f"Different model from the voice path — useful for fact checks, "
        f"alternative phrasings, or cross-model validation. Blocks until "
        f"the response returns; provide full context in the prompt."
    )
    speak_text = provider_cfg.speak_text or f"Let me ask {friendly}."
    tool_name = f"ask_{key}"

    class _ConsultTool(BaseTool):
        name = tool_name
        category = "agents"
        description = ""  # placeholder so dataclass-style assignment below works
        parameters = {
            "prompt": {
                "type": "string",
                "description": (
                    "The full question or task. Include necessary context — "
                    f"{friendly} has no access to the current conversation."
                ),
            },
        }
        required = ["prompt"]

        def execute(self, prompt: str) -> str:
            cfg = _agents_cfg()
            if cfg is None or key not in cfg.consult:
                return f"{tool_name} unavailable: not configured in config.yaml."
            return _consult(cfg.consult[key], prompt)

    _ConsultTool.__name__ = f"AskTool_{key}"
    _ConsultTool.description = description
    _ConsultTool.speak_text = speak_text
    return _ConsultTool


# ---------------------------------------------------------------------------
# Tier 2 — Coding-agent spawn: one dynamic tool per `agents.coding.<key>`
# ---------------------------------------------------------------------------


def _spawn_coding_agent(*, key: str, friendly: str, task: str, cwd: Optional[str]) -> dict:
    """Resolve the coding-agent config, check binary + concurrency, spawn."""
    cfg = _agents_cfg()
    if cfg is None or key not in cfg.coding:
        return {"error": f"spawn_{key} unavailable: not configured in config.yaml."}
    agent_cfg = cfg.coding[key]
    bin_path = _binary_path(agent_cfg.bin)
    if bin_path is None:
        return {
            "error": (
                f"spawn_{key} unavailable: `{agent_cfg.bin}` not on PATH. "
                f"Install it and try again."
            )
        }
    argv = (
        [bin_path, *agent_cfg.default_args, task]
        if agent_cfg.task_as_arg
        else [bin_path, *agent_cfg.default_args]
    )
    stdin = None if agent_cfg.task_as_arg else task

    if cfg.max_concurrent > 0:
        running = sum(
            1 for m in TaskStore().list_meta() if reconcile_status(m).status == "running"
        )
        if running >= cfg.max_concurrent:
            return {
                "error": (
                    f"spawn_{key} refused: {running}/{cfg.max_concurrent} concurrent "
                    f"agents already running. Cancel one with cancel_agent_task first."
                )
            }

    meta = runner_spawn(
        agent=key,
        argv=argv,
        cwd=_default_cwd(cwd),
        task=task,
        stdin_text=stdin,
        extra={"friendly": friendly, "bin": bin_path},
    )
    if meta.status == "failed":
        return {"task_id": meta.task_id, "status": meta.status, "error": meta.error}
    return {
        "task_id": meta.task_id,
        "agent": meta.agent,
        "status": meta.status,
        "pid": meta.pid,
        "cwd": meta.cwd,
        "note": (
            f"{friendly} is running. "
            f"Call get_agent_task with task_id={meta.task_id!r} to check progress."
        ),
    }


def _make_spawn_tool(key: str, agent_cfg) -> type[BaseTool]:
    """Build a ``BaseTool`` subclass for one coding-agent spawn target."""
    friendly = agent_cfg.friendly or _humanise(key)
    description = agent_cfg.description or (
        f"Spawn {friendly} (`{agent_cfg.bin}`) on a coding task in the background. "
        f"Returns immediately with a task_id; the agent runs for "
        f"30 seconds to several minutes. Check progress with get_agent_task."
    )
    speak_text = agent_cfg.speak_text or f"Kicking off {friendly}."
    tool_name = f"spawn_{key}"

    class _SpawnTool(BaseTool):
        name = tool_name
        category = "agents"
        description = ""
        parameters = {
            "task": {
                "type": "string",
                "description": f"Plain-English coding task — what you want {friendly} to do.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the agent. Defaults to the project root.",
            },
        }
        required = ["task"]

        def execute(self, task: str, cwd: Optional[str] = None) -> dict:
            return _spawn_coding_agent(key=key, friendly=friendly, task=task, cwd=cwd)

    _SpawnTool.__name__ = f"SpawnTool_{key}"
    _SpawnTool.description = description
    _SpawnTool.speak_text = speak_text
    return _SpawnTool


# ---------------------------------------------------------------------------
# Dynamic registration — call from `tools.register_all()` after config loads.
# ---------------------------------------------------------------------------


def register_dynamic_tools() -> None:
    """Walk ``config.agents.consult`` and ``.coding`` and register one tool each.

    Idempotent: re-registering a tool with the same name raises in the
    registry, so this should only be called once at startup.
    """
    cfg = _agents_cfg()
    if cfg is None:
        logger.info("external_agents: no `agents` block in config.yaml — skipping")
        return
    consult_count = 0
    for key, provider in cfg.consult.items():
        REGISTRY.register(_make_consult_tool(key, provider))
        consult_count += 1
    spawn_count = 0
    for key, agent_cfg in cfg.coding.items():
        REGISTRY.register(_make_spawn_tool(key, agent_cfg))
        spawn_count += 1
    logger.info(
        f"external_agents: registered {consult_count} consult tool(s), "
        f"{spawn_count} spawn tool(s)"
    )


# ---------------------------------------------------------------------------
# Task management — static, shared across every spawn target
# ---------------------------------------------------------------------------


def _meta_to_brief(meta) -> dict:
    """Compact view of a task — what the LLM needs without flooding the context."""
    duration = None
    if meta.finished_at and meta.started_at:
        duration = round(meta.finished_at - meta.started_at, 1)
    elif meta.started_at:
        import time
        duration = round(time.time() - meta.started_at, 1)
    return {
        "task_id": meta.task_id,
        "agent": meta.agent,
        "status": meta.status,
        "task": meta.task[:200] + ("…" if len(meta.task) > 200 else ""),
        "started_at": meta.started_at,
        "duration_s": duration,
        "exit_code": meta.exit_code,
    }


@REGISTRY.register
class ListAgentTasksTool(BaseTool):
    name = "list_agent_tasks"
    category = "agents"
    description = (
        "List spawned background agent tasks. Use to see what's running, "
        "what's finished, and what failed."
    )
    parameters = {
        "status": {
            "type": "string",
            "enum": ["all", "running", "done", "failed", "cancelled", "orphaned"],
            "description": "Filter by status. Default 'all'.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum tasks to return (most recent first). Default 10.",
        },
    }
    required = []

    def execute(self, status: str = "all", limit: int = 10) -> dict:
        store = TaskStore()
        metas = [reconcile_status(m) for m in store.list_meta()]
        metas.sort(key=lambda m: m.started_at, reverse=True)
        if status != "all":
            metas = [m for m in metas if m.status == status]
        metas = metas[: max(1, limit)]
        return {"tasks": [_meta_to_brief(m) for m in metas]}


@REGISTRY.register
class GetAgentTaskTool(BaseTool):
    name = "get_agent_task"
    category = "agents"
    description = (
        "Fetch the status, last output lines, and exit code of a spawned "
        "agent task. Use this to find out whether a background agent has "
        "finished and what it produced."
    )
    parameters = {
        "task_id": {
            "type": "string",
            "description": "The task_id returned by a previous spawn_* call.",
        },
        "tail_lines": {
            "type": "integer",
            "description": "How many trailing lines of stdout to return. Default 40.",
        },
        "include_stderr": {
            "type": "boolean",
            "description": "Also include stderr tail. Default false.",
        },
    }
    required = ["task_id"]

    def execute(
        self,
        task_id: str,
        tail_lines: int = 40,
        include_stderr: bool = False,
    ) -> dict:
        store = TaskStore()
        try:
            meta = reconcile_status(store.read_meta(task_id))
        except FileNotFoundError:
            return {"error": f"no task with id {task_id!r}"}
        out = _meta_to_brief(meta)
        out["stdout_tail"] = store.tail(task_id, "stdout", tail_lines)
        if include_stderr:
            out["stderr_tail"] = store.tail(task_id, "stderr", tail_lines)
        if meta.error:
            out["error"] = meta.error
        return out


@REGISTRY.register
class CancelAgentTaskTool(BaseTool):
    name = "cancel_agent_task"
    category = "agents"
    description = (
        "Send SIGTERM to a running background agent task. No-op for tasks "
        "that have already finished."
    )
    parameters = {
        "task_id": {
            "type": "string",
            "description": "The task_id of the task to cancel.",
        },
    }
    required = ["task_id"]

    def execute(self, task_id: str) -> dict:
        try:
            meta = runner_cancel(task_id)
        except FileNotFoundError:
            return {"error": f"no task with id {task_id!r}"}
        return _meta_to_brief(meta)


# ---------------------------------------------------------------------------
# Tier 3 — Hermes peer (one instance; static class still cheapest)
# ---------------------------------------------------------------------------


@REGISTRY.register
class HermesChatTool(BaseTool):
    name = "hermes_chat"
    category = "agents"
    speak_text = "Asking Hermes."
    description = (
        "Send a synchronous chat message to your local Hermes agent. Hermes "
        "has its own memory, skills, and tools — use this to delegate to it "
        "for tasks where its capabilities outweigh the voice bot's. Blocks "
        "until Hermes responds."
    )
    parameters = {
        "prompt": {
            "type": "string",
            "description": "What to send to Hermes. Provide full context — Hermes "
            "does not see the voice conversation.",
        },
        "session_id": {
            "type": "string",
            "description": "Optional Hermes session id to keep the conversation "
            "threaded. Omit to start a new thread.",
        },
    }
    required = ["prompt"]

    def execute(self, prompt: str, session_id: Optional[str] = None) -> str:
        cfg = _agents_cfg()
        if cfg is None or cfg.hermes is None:
            return "hermes_chat unavailable: `agents.hermes` block missing from config.yaml."
        h = cfg.hermes
        api_key = os.getenv(h.api_key_env, "") if h.api_key_env else ""
        try:
            extra = {"session_id": session_id} if session_id else None
            return _openai_chat(
                base_url=h.base_url,
                api_key=api_key,
                model=h.model,
                prompt=prompt,
                timeout=h.timeout,
                extra_body=extra,
            ).strip() or "(empty response from Hermes)"
        except httpx.HTTPError as exc:
            return f"Hermes chat failed: {exc}"


@REGISTRY.register
class HermesSpawnTool(BaseTool):
    name = "hermes_spawn"
    category = "agents"
    speak_text = "Spawning a Hermes task."
    description = (
        "Spawn a long-running task on Hermes via its task endpoint. Returns "
        "the Hermes task id immediately. Use for jobs Hermes is best at: "
        "research, multi-step planning, anything that needs its memory."
    )
    parameters = {
        "task": {
            "type": "string",
            "description": "The task description for Hermes.",
        },
    }
    required = ["task"]

    def execute(self, task: str) -> dict:
        cfg = _agents_cfg()
        if cfg is None or cfg.hermes is None:
            return {"error": "hermes_spawn unavailable: `agents.hermes` not configured."}
        h = cfg.hermes
        api_key = os.getenv(h.api_key_env, "") if h.api_key_env else ""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = httpx.post(
                f"{h.base_url.rstrip('/')}{h.spawn_path}",
                headers=headers,
                json={"task": task},
                timeout=h.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            return {"error": f"hermes_spawn failed: {exc}"}
        return {
            "hermes_task_id": data.get("id") or data.get("task_id"),
            "status": data.get("status", "submitted"),
            "raw": data,
        }
