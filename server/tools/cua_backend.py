"""cua-driver actuation backend (trycua/cua).

`cua-driver` is a Swift binary distributed via Homebrew that drives macOS
without stealing cursor/focus. It speaks MCP over stdio (newline-delimited
JSON-RPC 2.0). This module spawns it as a subprocess on first use and routes
click/type/key/scroll/move calls through it.

The MCP-over-stdio plumbing lives in ``tools.mcp_client``; this module is the
cua-specific wrapper (tool-name constants, string-formatted return values,
singleton accessor).

The whole file is opt-in via ``config.yaml`` (``computer_use.backend: cua``).
If the binary is missing or the handshake fails, callers in
``tools/computer_use.py`` log and fall back to pyautogui — the bot stays
usable.

----------------------------------------------------------------------------
HOW TO STRIP THIS INTEGRATION OUT
----------------------------------------------------------------------------
1. Delete this file.
2. In ``tools/actuation_backend.py``, delete the ``_try_cua`` helper and the
   ``inner = _try_cua() or PyAutoGUIBackend()`` line in ``_build_backend``.
3. In ``tools/computer_use.py``, remove the ``CuaStatusTool`` class.
4. In ``config.py``, drop ``ComputerUseConfig`` and the ``computer_use``
   field from ``Config`` + ``_parse_config``.
5. In ``config.example.yaml``, remove the ``computer_use:`` block.
6. In ``CLAUDE.md``, remove the "Computer-use backends" subsection.
----------------------------------------------------------------------------

Tool-name table
---------------
The MCP tool names cua-driver exposes are listed here as constants. They are
a best-guess based on the cua-driver README and need to be verified against
your installed version — call the ``cua_status`` tool from a session and
inspect the logged inventory. If a name is wrong, edit the constant; the
restart picks it up.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from tools.mcp_client import MCPClientError, MCPStdioClient, summarise_mcp_result


# Compat alias — used to be raised from this module directly. Keep the name
# importable so external callers don't break.
CuaBackendError = MCPClientError


# ---------------------------------------------------------------------------
# MCP tool name constants — adjust if cua-driver renames anything.
# Verified at startup against the live tools/list response; mismatches log
# a warning.
# ---------------------------------------------------------------------------
TOOL_CLICK = "click"
TOOL_DOUBLE_CLICK = "double_click"
TOOL_TYPE = "type_text"
TOOL_KEY = "press_key"
TOOL_SCROLL = "scroll"
TOOL_MOVE = "move_mouse"


# ---------------------------------------------------------------------------
# CuaBackend — what tools/actuation_backend.py calls into.
# ---------------------------------------------------------------------------

class CuaBackend:
    """High-level wrapper around the MCP client matching what the actuation
    tools need. Methods return strings the way the native tools do.
    """

    def __init__(self, client: MCPStdioClient) -> None:
        self._client = client

    def click(self, x: int, y: int, *, double: bool = False, button: str = "left") -> str:
        tool = TOOL_DOUBLE_CLICK if double else TOOL_CLICK
        args: dict[str, Any] = {"x": x, "y": y}
        # Only forward `button` for single clicks — cua's double_click typically
        # doesn't take it. Harmless either way.
        if not double:
            args["button"] = button
        result = self._client.call_tool(tool, args)
        return summarise_mcp_result(result, default=(
            f"{'Double-clicked' if double else 'Clicked'} ({x},{y}) via cua-driver"
        ))

    def type_text(self, text: str, *, interval: float = 0.02) -> str:
        args: dict[str, Any] = {"text": text}
        if interval:
            args["delay"] = interval  # cua may name this `delay` or `interval`
        result = self._client.call_tool(TOOL_TYPE, args)
        return summarise_mcp_result(
            result,
            default=f"Typed {len(text)} char(s) via cua-driver",
        )

    def press_key(self, keys: str) -> str:
        result = self._client.call_tool(TOOL_KEY, {"keys": keys})
        return summarise_mcp_result(result, default=f"Pressed {keys} via cua-driver")

    def scroll(self, amount: int, *, x: Optional[int] = None, y: Optional[int] = None) -> str:
        args: dict[str, Any] = {"amount": amount}
        if x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        result = self._client.call_tool(TOOL_SCROLL, args)
        return summarise_mcp_result(result, default=f"Scrolled {amount:+d} via cua-driver")

    def mouse_move(self, x: int, y: int, *, duration: float = 0.1) -> str:
        args = {"x": x, "y": y, "duration": duration}
        result = self._client.call_tool(TOOL_MOVE, args)
        return summarise_mcp_result(result, default=f"Moved to ({x},{y}) via cua-driver")

    # ---- diagnostics ----

    def tool_names(self) -> list[str]:
        return self._client.tool_names()


# ---------------------------------------------------------------------------
# Singleton accessor — lazy spawn, cached. Importing this module costs nothing.
# ---------------------------------------------------------------------------

_singleton: Optional[CuaBackend] = None
_singleton_lock = threading.Lock()
_singleton_failed: Optional[str] = None  # cached failure reason


def get_cua_backend() -> CuaBackend:
    """Return the lazily-started ``CuaBackend`` singleton.

    Reads ``config.computer_use`` for binary path + timeout. Raises
    ``CuaBackendError`` if cua-driver can't be started.
    """
    global _singleton, _singleton_failed
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        if _singleton_failed is not None:
            # Stay failed for the rest of the process — don't keep retrying
            # the subprocess spawn on every tool call.
            raise CuaBackendError(_singleton_failed)
        try:
            from config import get_config
            cfg = get_config().computer_use
            client = MCPStdioClient(
                argv=[cfg.cua_binary],
                timeout=cfg.cua_timeout,
                log_name="cua-driver",
            )
            client.start()
            _singleton = CuaBackend(client)
            return _singleton
        except CuaBackendError as exc:
            _singleton_failed = str(exc)
            raise
        except Exception as exc:
            _singleton_failed = f"{type(exc).__name__}: {exc}"
            raise CuaBackendError(_singleton_failed) from exc
