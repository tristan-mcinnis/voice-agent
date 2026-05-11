"""cua-driver actuation backend (trycua/cua).

`cua-driver` is a Swift binary distributed via Homebrew that drives macOS
without stealing cursor/focus. It speaks MCP over stdio (newline-delimited
JSON-RPC 2.0). This module spawns it as a subprocess on first use and routes
click/type/key/scroll/move calls through it.

The whole file is opt-in via ``config.yaml`` (``computer_use.backend: cua``).
If the binary is missing or the handshake fails, callers in
``tools/computer_use.py`` log and fall back to pyautogui — the bot stays
usable. Stdlib-only (no new pip deps).

----------------------------------------------------------------------------
HOW TO STRIP THIS INTEGRATION OUT
----------------------------------------------------------------------------
1. Delete this file.
2. In ``tools/computer_use.py``, delete the ``_try_cua_backend()`` helper
   and every ``cua = _try_cua_backend(); if cua: return cua.<method>(...)``
   block inside the actuation tools' ``execute()`` methods.
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

import json
import os
import shutil
import subprocess
import threading
from typing import Any, Optional

from loguru import logger


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


class CuaBackendError(RuntimeError):
    """Raised when cua-driver is unavailable or returns an error."""


# ---------------------------------------------------------------------------
# Minimal MCP-over-stdio client. JSON-RPC 2.0, one line per message.
# ---------------------------------------------------------------------------

class _MCPStdioClient:
    """Spawn a long-lived MCP server subprocess and call its tools.

    Single-request-at-a-time: every public call grabs ``_io_lock`` for the
    duration of send+recv. The bot calls tools sequentially from
    ``asyncio.to_thread`` so this is fine.
    """

    def __init__(self, argv: list[str], *, timeout: float = 10.0) -> None:
        self._argv = argv
        self._timeout = timeout
        self._proc: Optional[subprocess.Popen[str]] = None
        self._next_id = 1
        self._io_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._tools: dict[str, dict[str, Any]] = {}

    # ---- lifecycle ----

    def start(self) -> None:
        """Spawn the subprocess and complete the MCP handshake. Idempotent."""
        if self._proc and self._proc.poll() is None:
            return
        if shutil.which(self._argv[0]) is None and not os.path.isfile(self._argv[0]):
            raise CuaBackendError(
                f"{self._argv[0]} not found in PATH. "
                f"Install with: brew tap trycua/cua && brew install cua-driver"
            )
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise CuaBackendError(f"failed to spawn {self._argv[0]}: {exc}") from exc

        # Drain stderr in a background thread so the subprocess doesn't block
        # on a full pipe. Lines are forwarded to loguru at DEBUG level.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="cua-driver-stderr", daemon=True
        )
        self._stderr_thread.start()

        # Handshake.
        init = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "voice-agent", "version": "1.0"},
        })
        logger.info(
            "cua-driver: handshake ok ({})".format(
                init.get("serverInfo", {}).get("name", "?")
            )
        )
        self._notify("notifications/initialized", {})

        # Pull the live tool inventory so the diagnostic tool can show it.
        listed = self._rpc("tools/list", {})
        for tool in listed.get("tools", []):
            self._tools[tool["name"]] = tool
        logger.info(
            f"cua-driver: {len(self._tools)} tool(s) exposed — "
            + ", ".join(sorted(self._tools)) if self._tools else "cua-driver: no tools listed"
        )

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    # ---- IO ----

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, ""):
            if not line:
                break
            logger.debug(f"cua-driver: {line.rstrip()}")

    def _send(self, msg: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin or self._proc.poll() is not None:
            raise CuaBackendError("cua-driver subprocess is not running")
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CuaBackendError(f"cua-driver stdin closed: {exc}") from exc

    def _recv_response(self, expected_id: int) -> dict[str, Any]:
        """Read until we see a response matching ``expected_id``.

        Drops any notifications or unrelated messages along the way. Errors
        out if the subprocess closes stdout (driver crashed).
        """
        if not self._proc or not self._proc.stdout:
            raise CuaBackendError("cua-driver stdout is not available")
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise CuaBackendError(
                    "cua-driver closed stdout (subprocess likely exited)"
                )
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"cua-driver: bad JSON on stdout: {line.rstrip()!r}")
                continue
            if msg.get("id") == expected_id:
                return msg
            # Otherwise it's a notification or stale message — keep waiting.

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """One round-trip JSON-RPC call. Returns the ``result`` field."""
        with self._io_lock:
            req_id = self._next_id
            self._next_id += 1
            self._send({
                "jsonrpc": "2.0", "id": req_id,
                "method": method, "params": params,
            })
            resp = self._recv_response(req_id)
        if "error" in resp:
            err = resp["error"]
            raise CuaBackendError(
                f"{method} failed: {err.get('code')} {err.get('message')}"
            )
        return resp.get("result", {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        with self._io_lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # ---- public surface ----

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            logger.warning(
                f"cua-driver: calling tool {name!r} which wasn't in tools/list — "
                f"name may be wrong; cua exposes: {sorted(self._tools)}"
            )
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def tool_names(self) -> list[str]:
        return sorted(self._tools)


# ---------------------------------------------------------------------------
# CuaBackend — what tools/computer_use.py calls into.
# ---------------------------------------------------------------------------

class CuaBackend:
    """High-level wrapper around the MCP client matching what the actuation
    tools need. Methods return strings the way the native tools do.
    """

    def __init__(self, client: _MCPStdioClient) -> None:
        self._client = client

    def click(self, x: int, y: int, *, double: bool = False, button: str = "left") -> str:
        tool = TOOL_DOUBLE_CLICK if double else TOOL_CLICK
        args: dict[str, Any] = {"x": x, "y": y}
        # Only forward `button` for single clicks — cua's double_click typically
        # doesn't take it. Harmless either way.
        if not double:
            args["button"] = button
        result = self._client.call_tool(tool, args)
        return _summarise(result, default=(
            f"{'Double-clicked' if double else 'Clicked'} ({x},{y}) via cua-driver"
        ))

    def type_text(self, text: str, *, interval: float = 0.02) -> str:
        args = {"text": text}
        if interval:
            args["delay"] = interval  # cua may name this `delay` or `interval`
        result = self._client.call_tool(TOOL_TYPE, args)
        return _summarise(
            result,
            default=f"Typed {len(text)} char(s) via cua-driver",
        )

    def press_key(self, keys: str) -> str:
        result = self._client.call_tool(TOOL_KEY, {"keys": keys})
        return _summarise(result, default=f"Pressed {keys} via cua-driver")

    def scroll(self, amount: int, *, x: Optional[int] = None, y: Optional[int] = None) -> str:
        args: dict[str, Any] = {"amount": amount}
        if x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        result = self._client.call_tool(TOOL_SCROLL, args)
        return _summarise(result, default=f"Scrolled {amount:+d} via cua-driver")

    def mouse_move(self, x: int, y: int, *, duration: float = 0.1) -> str:
        args = {"x": x, "y": y, "duration": duration}
        result = self._client.call_tool(TOOL_MOVE, args)
        return _summarise(result, default=f"Moved to ({x},{y}) via cua-driver")

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
            client = _MCPStdioClient(
                argv=[cfg.cua_binary],
                timeout=cfg.cua_timeout,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(result: dict[str, Any], *, default: str) -> str:
    """Pull a human-readable summary out of an MCP tools/call result.

    MCP returns ``{"content": [{"type": "text", "text": "..."}], ...}``. If
    the server returns text, surface it; otherwise use ``default``.
    """
    content = result.get("content")
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            return joined
    if result.get("isError"):
        return f"cua-driver returned error: {result}"
    return default
