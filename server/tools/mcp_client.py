"""Generic MCP-over-stdio client. Stdlib-only.

Spawns an MCP server subprocess, performs the JSON-RPC 2.0 handshake, and
exposes ``tools/list`` and ``tools/call``. Used by two callers today:

  - ``tools.cua_backend`` — wraps cua-driver as the actuation backend.
  - ``tools.mcp_tools`` — registers every MCP server in ``config.mcp_servers``
    as a set of dynamic ``BaseTool`` subclasses.

Single-request-at-a-time: every public call grabs ``_io_lock`` for the
duration of send+recv. The bot calls tools sequentially from
``asyncio.to_thread`` so this is fine.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from typing import Any, Optional

from loguru import logger


class MCPClientError(RuntimeError):
    """Raised when an MCP server is unavailable or returns a JSON-RPC error."""


class MCPStdioClient:
    """Long-lived MCP server subprocess driver.

    Construct with the ``argv`` you'd type at a shell prompt. Call
    :meth:`start` to spawn + handshake + populate the tool inventory. After
    that, :meth:`call_tool` is safe from any thread.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        log_name: str = "mcp",
    ) -> None:
        self._argv = argv
        self._env = env or {}
        self._timeout = timeout
        self._log_name = log_name
        self._proc: Optional[subprocess.Popen[str]] = None
        self._next_id = 1
        self._io_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._tools: dict[str, dict[str, Any]] = {}
        self._server_info: dict[str, Any] = {}

    # ---- lifecycle ----

    def start(self) -> None:
        """Spawn the subprocess and complete the MCP handshake. Idempotent."""
        if self._proc and self._proc.poll() is None:
            return
        if not self._argv:
            raise MCPClientError("argv is empty")
        if shutil.which(self._argv[0]) is None and not os.path.isfile(self._argv[0]):
            raise MCPClientError(f"{self._argv[0]} not found in PATH")

        merged_env = {**os.environ, **self._env} if self._env else None
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=merged_env,
            )
        except OSError as exc:
            raise MCPClientError(f"failed to spawn {self._argv[0]}: {exc}") from exc

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name=f"{self._log_name}-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        init = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "voice-agent", "version": "1.0"},
        })
        self._server_info = init.get("serverInfo", {}) or {}
        logger.info(
            f"{self._log_name}: handshake ok "
            f"({self._server_info.get('name', '?')})"
        )
        self._notify("notifications/initialized", {})

        listed = self._rpc("tools/list", {})
        for tool in listed.get("tools", []):
            name = tool.get("name")
            if name:
                self._tools[name] = tool
        if self._tools:
            logger.info(
                f"{self._log_name}: {len(self._tools)} tool(s) — "
                + ", ".join(sorted(self._tools))
            )
        else:
            logger.info(f"{self._log_name}: no tools listed")

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
            logger.debug(f"{self._log_name}: {line.rstrip()}")

    def _send(self, msg: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin or self._proc.poll() is not None:
            raise MCPClientError(f"{self._log_name} subprocess is not running")
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPClientError(f"{self._log_name} stdin closed: {exc}") from exc

    def _recv_response(self, expected_id: int) -> dict[str, Any]:
        if not self._proc or not self._proc.stdout:
            raise MCPClientError(f"{self._log_name} stdout is not available")
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise MCPClientError(
                    f"{self._log_name} closed stdout (subprocess likely exited)"
                )
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    f"{self._log_name}: bad JSON on stdout: {line.rstrip()!r}"
                )
                continue
            if msg.get("id") == expected_id:
                return msg

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
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
            raise MCPClientError(
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
                f"{self._log_name}: calling tool {name!r} not in tools/list — "
                f"exposed: {sorted(self._tools)}"
            )
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the raw tool descriptors captured during ``tools/list``."""
        return list(self._tools.values())

    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def is_alive(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)


def summarise_mcp_result(result: dict[str, Any], *, default: str = "") -> str:
    """Pull a human-readable summary out of an MCP ``tools/call`` result.

    MCP returns ``{"content": [{"type": "text", "text": "..."}, ...], ...}``.
    If the server returned text blocks, surface their concatenation;
    otherwise fall back to ``default`` (or an error indicator when
    ``isError`` is set).
    """
    content = result.get("content")
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "\n".join(p for p in parts if p).strip()
        if joined:
            return joined
    if result.get("isError"):
        return f"error: {result}"
    return default
