"""MCP server adapter — register every ``config.mcp_servers`` entry as tools.

Reads ``config.mcp_servers`` from ``config.yaml``. For each enabled entry:

1. Spawn the server via :class:`tools.mcp_client.MCPStdioClient` and run the
   JSON-RPC handshake.
2. Call ``tools/list`` to discover exposed tools.
3. For each discovered tool (respecting ``include`` / ``exclude``), build a
   :class:`BaseTool` subclass on the fly and register it under
   ``"{server_name}__{tool_name}"``.

The double-underscore separator keeps registered names inside the OpenAI
function-name regex (``[a-zA-Z0-9_-]{1,64}``) while staying visually
distinct from the per-server tool names. Names longer than 64 chars are
truncated with a deterministic hash suffix so collisions stay rare.

Connection failures are non-fatal: missing binaries, handshake errors,
schema problems, and duplicate names all just log a warning. The bot
stays usable; the server's tools are simply absent from the registry.

Spawn happens during ``tools.register_all()`` — the same call site that
imports every other tool module. Subprocesses are kept alive for the
process lifetime and shut down lazily on interpreter exit via ``atexit``.
"""

from __future__ import annotations

import atexit
import hashlib
from typing import Any, Optional

from loguru import logger

from tools.mcp_client import MCPClientError, MCPStdioClient, summarise_mcp_result
from tools.registry import REGISTRY, BaseTool


# Track every successfully-started client so ``mcp_status`` can introspect
# the live state and ``atexit`` can shut them down cleanly.
_clients: dict[str, MCPStdioClient] = {}
# Last attempted server config + failure reason, used by ``mcp_status``.
_last_errors: dict[str, str] = {}


# OpenAI function-name regex: ``^[a-zA-Z0-9_-]{1,64}$``. We slightly
# over-cap at 60 to leave room for a hash suffix on collisions.
_MAX_NAME_LEN = 64
_HASH_SUFFIX_LEN = 6


def _normalise_name(server: str, tool: str) -> str:
    """Build a safe registry name from server + tool.

    - Replaces any non-allowed character with ``_``.
    - Joins with ``__``.
    - If too long, truncates and appends a deterministic 6-char hex hash so
      different long names don't collide.
    """
    def safe(part: str) -> str:
        return "".join(c if (c.isalnum() or c in "_-") else "_" for c in part)

    full = f"{safe(server)}__{safe(tool)}"
    if len(full) <= _MAX_NAME_LEN:
        return full
    digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_LEN]
    keep = _MAX_NAME_LEN - _HASH_SUFFIX_LEN - 1  # 1 for '_'
    return f"{full[:keep]}_{digest}"


def _extract_schema(mcp_tool: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Pull ``properties`` + ``required`` from an MCP tool's inputSchema.

    Tolerates servers that omit ``inputSchema`` entirely (treat as zero-arg)
    or that put the schema directly at top-level without an ``object`` wrapper.
    """
    schema = mcp_tool.get("inputSchema") or {}
    if not isinstance(schema, dict):
        return {}, []
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []
    return properties, [r for r in required if isinstance(r, str)]


def _build_tool_class(
    *,
    registry_name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    category: str,
    mcp_tool_name: str,
    client: MCPStdioClient,
) -> type[BaseTool]:
    """Construct a ``BaseTool`` subclass for one MCP-exposed tool.

    The ``execute`` closes over ``client`` + ``mcp_tool_name`` so the
    registry call site doesn't need to know the originating server.
    """

    class _MCPTool(BaseTool):
        pass

    _MCPTool.name = registry_name
    _MCPTool.description = description or f"MCP tool {mcp_tool_name}"
    _MCPTool.parameters = properties
    _MCPTool.required = required
    _MCPTool.category = category

    def execute(self: BaseTool, **kwargs: Any) -> Any:
        try:
            result = client.call_tool(mcp_tool_name, kwargs)
        except MCPClientError as exc:
            return f"MCP call {mcp_tool_name!r} failed: {exc}"
        text = summarise_mcp_result(
            result, default=f"{mcp_tool_name} completed."
        )
        return text

    _MCPTool.execute = execute  # type: ignore[assignment]
    _MCPTool.__name__ = f"MCPTool_{registry_name}"
    return _MCPTool


def _spawn_client(server_cfg) -> Optional[MCPStdioClient]:
    """Spawn one MCP server. Returns ``None`` on any error (already logged)."""
    argv = [server_cfg.command, *server_cfg.args]
    client = MCPStdioClient(
        argv=argv,
        env=dict(server_cfg.env),
        timeout=server_cfg.timeout,
        log_name=f"mcp/{server_cfg.name}",
    )
    try:
        client.start()
    except MCPClientError as exc:
        _last_errors[server_cfg.name] = str(exc)
        logger.warning(f"mcp/{server_cfg.name}: failed to start — {exc}")
        return None
    except Exception as exc:
        _last_errors[server_cfg.name] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            f"mcp/{server_cfg.name}: crashed during start — "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    return client


def _register_one_server(server_cfg) -> None:
    """Spawn one server and register each of its tools."""
    if not server_cfg.enabled:
        logger.info(f"mcp/{server_cfg.name}: skipped (enabled=false)")
        return
    if server_cfg.name in _clients:
        logger.debug(f"mcp/{server_cfg.name}: already registered, skipping")
        return

    client = _spawn_client(server_cfg)
    if client is None:
        return
    _clients[server_cfg.name] = client

    include = set(server_cfg.include)
    exclude = set(server_cfg.exclude)
    registered = 0
    for mcp_tool in client.list_tools():
        mcp_name = mcp_tool.get("name")
        if not mcp_name:
            continue
        if include and mcp_name not in include:
            continue
        if mcp_name in exclude:
            continue
        registry_name = _normalise_name(server_cfg.name, mcp_name)
        properties, required = _extract_schema(mcp_tool)
        try:
            cls = _build_tool_class(
                registry_name=registry_name,
                description=mcp_tool.get("description") or "",
                properties=properties,
                required=required,
                category=server_cfg.category,
                mcp_tool_name=mcp_name,
                client=client,
            )
            REGISTRY.register(cls)
            registered += 1
        except ValueError as exc:
            # Duplicate registration or empty name — log and move on.
            logger.warning(
                f"mcp/{server_cfg.name}: cannot register {mcp_name!r} as "
                f"{registry_name!r}: {exc}"
            )
        except Exception as exc:
            logger.warning(
                f"mcp/{server_cfg.name}: error registering {mcp_name!r}: "
                f"{type(exc).__name__}: {exc}"
            )

    logger.info(
        f"mcp/{server_cfg.name}: {registered} tool(s) registered "
        f"(of {len(client.list_tools())} exposed)"
    )


def register_mcp_servers() -> None:
    """Spawn every enabled ``config.mcp_servers`` entry and register its tools.

    Called from ``tools.register_all()`` after the static tool modules have
    been imported. Safe to call multiple times — already-started servers
    are skipped.
    """
    try:
        from config import get_config
        servers = get_config().mcp_servers
    except RuntimeError:
        # Config not initialised — happens in tests that import the package
        # without calling init_config. Just no-op.
        logger.debug("mcp_tools: config not initialised, skipping")
        return
    except Exception as exc:
        logger.warning(f"mcp_tools: cannot read config — {exc}")
        return

    for server_cfg in servers:
        _register_one_server(server_cfg)


def get_live_clients() -> dict[str, MCPStdioClient]:
    """Snapshot of currently-running MCP clients keyed by config name."""
    return dict(_clients)


def get_last_errors() -> dict[str, str]:
    """Failure reasons for servers that couldn't be started this process."""
    return dict(_last_errors)


def _shutdown_all() -> None:
    for name, client in list(_clients.items()):
        try:
            client.shutdown()
        except Exception:
            pass
        _clients.pop(name, None)


atexit.register(_shutdown_all)


# ---------------------------------------------------------------------------
# Diagnostic tool — handshake / inventory introspection from a live session.
# ---------------------------------------------------------------------------

@REGISTRY.register
class MCPStatusTool(BaseTool):
    name = "mcp_status"
    category = "system"
    description = (
        "Report which MCP servers are connected, which tools they expose, "
        "and any servers that failed to start. Use to debug "
        "config.mcp_servers entries before relying on their tools."
    )
    parameters = {
        "server": {
            "type": "string",
            "description": (
                "Optional server name. If set, show only that server's "
                "details (full tool list). Otherwise, show a one-line "
                "summary per server."
            ),
        },
    }

    def execute(self, server: str = "") -> str:
        try:
            from config import get_config
            configured = {s.name: s for s in get_config().mcp_servers}
        except Exception as exc:
            return f"mcp_status: config unavailable: {type(exc).__name__}: {exc}"

        if not configured:
            return "mcp_status: no servers configured (config.mcp_servers is empty)"

        if server:
            cfg = configured.get(server)
            if cfg is None:
                return (
                    f"mcp_status: no server named {server!r}. "
                    f"Configured: {sorted(configured)}"
                )
            return self._render_one(server, cfg)

        lines = ["mcp_status: configured servers"]
        for name, cfg in configured.items():
            if not cfg.enabled:
                lines.append(f"  - {name}: disabled")
                continue
            client = _clients.get(name)
            if client is None:
                err = _last_errors.get(name, "not started")
                lines.append(f"  - {name}: DOWN ({err})")
                continue
            alive = "alive" if client.is_alive() else "DEAD"
            lines.append(
                f"  - {name}: {alive}, {len(client.tool_names())} tool(s)"
            )
        return "\n".join(lines)

    def _render_one(self, name: str, cfg) -> str:
        lines = [
            f"mcp/{name}",
            f"  command: {cfg.command} {' '.join(cfg.args)}".rstrip(),
            f"  enabled: {cfg.enabled}",
            f"  category: {cfg.category}",
            f"  timeout: {cfg.timeout}s",
        ]
        if cfg.include:
            lines.append(f"  include: {cfg.include}")
        if cfg.exclude:
            lines.append(f"  exclude: {cfg.exclude}")
        client = _clients.get(name)
        if client is None:
            lines.append(f"  status: DOWN ({_last_errors.get(name, 'not started')})")
            return "\n".join(lines)
        info = client.server_info()
        lines.append(
            f"  status: {'alive' if client.is_alive() else 'DEAD'} "
            f"({info.get('name', '?')} {info.get('version', '')})".rstrip()
        )
        names = client.tool_names()
        lines.append(f"  tools ({len(names)}):")
        for n in names:
            registered = _normalise_name(name, n)
            mark = "✓" if registered in {t.name for t in REGISTRY.all()} else " "
            lines.append(f"    {mark} {n}  →  {registered}")
        return "\n".join(lines)
