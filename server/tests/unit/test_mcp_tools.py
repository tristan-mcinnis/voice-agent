"""Tests for MCP server registration into the tool registry.

Mocks ``MCPStdioClient`` so no subprocess is ever spawned — we verify the
registration flow, naming, schema mapping, and execute() dispatch.
"""

from __future__ import annotations

from typing import Any

import pytest

from config import MCPServerConfig
from tools import mcp_tools
from tools.mcp_client import MCPClientError
from tools.registry import REGISTRY, ToolRegistry


class FakeClient:
    """Stand-in for MCPStdioClient with a scripted tool inventory.

    Records every ``call_tool`` invocation so tests can assert on dispatch.
    """

    def __init__(self, tools: list[dict[str, Any]], *, start_raises: Exception | None = None) -> None:
        self._tools = tools
        self._start_raises = start_raises
        self._started = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_called = False

    def start(self) -> None:
        if self._start_raises:
            raise self._start_raises
        self._started = True

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def tool_names(self) -> list[str]:
        return sorted(t["name"] for t in self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": f"called {name}"}]}

    def shutdown(self) -> None:
        self.shutdown_called = True

    def is_alive(self) -> bool:
        return self._started and not self.shutdown_called

    def server_info(self) -> dict[str, Any]:
        return {"name": "fake-server", "version": "0.1"}


@pytest.fixture
def fresh_registry(monkeypatch):
    """Swap REGISTRY for an empty one for the duration of the test.

    Each MCP-tool registration calls ``REGISTRY.register`` at import time,
    which uses the singleton; rebinding the module-level name to a fresh
    instance keeps tests isolated.
    """
    original = mcp_tools.REGISTRY
    fresh = ToolRegistry()
    monkeypatch.setattr(mcp_tools, "REGISTRY", fresh)
    # Also clear bookkeeping dicts.
    monkeypatch.setattr(mcp_tools, "_clients", {})
    monkeypatch.setattr(mcp_tools, "_last_errors", {})
    yield fresh
    # No teardown needed — monkeypatch undoes both bindings.


def test_register_one_server_registers_each_tool(fresh_registry, monkeypatch):
    fake = FakeClient([
        {
            "name": "turn_on",
            "description": "Turn on a light",
            "inputSchema": {
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
        {
            "name": "list_devices",
            "description": "List all devices",
            "inputSchema": {},
        },
    ])
    monkeypatch.setattr(mcp_tools, "_spawn_client", lambda cfg: fake)

    cfg = MCPServerConfig(name="ha", command="ha-mcp", category="system")
    mcp_tools._register_one_server(cfg)

    names = {t.name for t in fresh_registry.all()}
    assert "ha__turn_on" in names
    assert "ha__list_devices" in names

    turn_on = fresh_registry.get("ha__turn_on")
    assert turn_on.description == "Turn on a light"
    assert turn_on.required == ["entity_id"]
    assert turn_on.category == "system"
    assert "entity_id" in turn_on.parameters


def test_execute_dispatches_to_underlying_client(fresh_registry, monkeypatch):
    fake = FakeClient([{
        "name": "say",
        "description": "say hello",
        "inputSchema": {"properties": {"msg": {"type": "string"}}},
    }])
    monkeypatch.setattr(mcp_tools, "_spawn_client", lambda cfg: fake)

    cfg = MCPServerConfig(name="demo", command="demo")
    mcp_tools._register_one_server(cfg)

    tool = fresh_registry.get("demo__say")
    result = tool.execute(msg="hi")
    assert result == "called say"
    assert fake.calls == [("say", {"msg": "hi"})]


def test_execute_returns_friendly_message_on_client_error(fresh_registry, monkeypatch):
    class FailingClient(FakeClient):
        def call_tool(self, name, arguments):
            raise MCPClientError("connection lost")

    fake = FailingClient([{"name": "x", "description": "", "inputSchema": {}}])
    monkeypatch.setattr(mcp_tools, "_spawn_client", lambda cfg: fake)

    mcp_tools._register_one_server(MCPServerConfig(name="d", command="d"))
    tool = fresh_registry.get("d__x")
    out = tool.execute()
    assert "MCP call" in out
    assert "connection lost" in out


def test_disabled_server_is_skipped(fresh_registry, monkeypatch):
    """enabled=false: no spawn, no registrations, no entry in _clients."""
    spawned = []
    monkeypatch.setattr(
        mcp_tools, "_spawn_client",
        lambda cfg: spawned.append(cfg) or FakeClient([]),
    )

    cfg = MCPServerConfig(name="off", command="x", enabled=False)
    mcp_tools._register_one_server(cfg)

    assert spawned == []
    assert fresh_registry.all() == []
    assert "off" not in mcp_tools.get_live_clients()


def test_spawn_failure_recorded_in_last_errors(fresh_registry, monkeypatch):
    monkeypatch.setattr(
        mcp_tools, "_spawn_client", lambda cfg: None,
    )
    # Pretend _spawn_client did its own logging and populated _last_errors.
    mcp_tools._last_errors["broken"] = "binary missing"

    mcp_tools._register_one_server(MCPServerConfig(name="broken", command="x"))

    assert fresh_registry.all() == []
    assert "broken" in mcp_tools.get_last_errors()


def test_include_filter(fresh_registry, monkeypatch):
    fake = FakeClient([
        {"name": "keep", "description": "", "inputSchema": {}},
        {"name": "drop", "description": "", "inputSchema": {}},
    ])
    monkeypatch.setattr(mcp_tools, "_spawn_client", lambda cfg: fake)

    cfg = MCPServerConfig(name="s", command="x", include=["keep"])
    mcp_tools._register_one_server(cfg)

    names = {t.name for t in fresh_registry.all()}
    assert names == {"s__keep"}


def test_exclude_filter_wins(fresh_registry, monkeypatch):
    fake = FakeClient([
        {"name": "keep", "description": "", "inputSchema": {}},
        {"name": "drop", "description": "", "inputSchema": {}},
    ])
    monkeypatch.setattr(mcp_tools, "_spawn_client", lambda cfg: fake)

    cfg = MCPServerConfig(
        name="s", command="x",
        include=["keep", "drop"], exclude=["drop"],
    )
    mcp_tools._register_one_server(cfg)

    names = {t.name for t in fresh_registry.all()}
    assert names == {"s__keep"}


def test_already_registered_server_is_skipped(fresh_registry, monkeypatch):
    fake1 = FakeClient([{"name": "t", "description": "", "inputSchema": {}}])
    fake2 = FakeClient([{"name": "t2", "description": "", "inputSchema": {}}])
    queue = [fake1, fake2]
    monkeypatch.setattr(mcp_tools, "_spawn_client", lambda cfg: queue.pop(0))

    cfg = MCPServerConfig(name="dup", command="x")
    mcp_tools._register_one_server(cfg)
    mcp_tools._register_one_server(cfg)  # second call must be a no-op

    names = {t.name for t in fresh_registry.all()}
    assert names == {"dup__t"}
    # fake2 was never consumed.
    assert queue == [fake2]


def test_register_mcp_servers_pulls_from_config(fresh_registry, monkeypatch):
    captured: list[MCPServerConfig] = []

    def fake_register(cfg):
        captured.append(cfg)

    monkeypatch.setattr(mcp_tools, "_register_one_server", fake_register)

    class FakeCfg:
        mcp_servers = [
            MCPServerConfig(name="a", command="a"),
            MCPServerConfig(name="b", command="b"),
        ]

    monkeypatch.setattr("config.get_config", lambda: FakeCfg())

    mcp_tools.register_mcp_servers()
    assert [c.name for c in captured] == ["a", "b"]


def test_register_mcp_servers_swallows_uninit_config(fresh_registry, monkeypatch):
    def boom():
        raise RuntimeError("Config not initialised.")

    monkeypatch.setattr("config.get_config", boom)
    # Must not raise.
    mcp_tools.register_mcp_servers()
    assert fresh_registry.all() == []
