"""Pure-function tests for the MCP client + summariser.

The transport layer (subprocess spawn + stdin/stdout JSON-RPC) is integration
territory and isn't exercised here. We verify the parts whose contract is
defined in this codebase: the result summariser, the schema-tolerant
introspection helpers, and the registry-name normaliser.
"""

from __future__ import annotations

import pytest

from tools import mcp_tools
from tools.mcp_client import summarise_mcp_result


class TestSummariseMCPResult:
    def test_joins_text_content_blocks(self):
        result = {
            "content": [
                {"type": "text", "text": "Light turned on."},
                {"type": "text", "text": "Brightness: 80%."},
            ],
        }
        assert summarise_mcp_result(result) == "Light turned on.\nBrightness: 80%."

    def test_ignores_non_text_blocks(self):
        result = {
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "ok"},
            ],
        }
        assert summarise_mcp_result(result) == "ok"

    def test_falls_back_to_default_when_no_text(self):
        result = {"content": []}
        assert summarise_mcp_result(result, default="done") == "done"

    def test_marks_error_when_no_text_and_isError(self):
        result = {"isError": True, "content": []}
        assert summarise_mcp_result(result, default="").startswith("error:")

    def test_text_wins_over_error_flag_when_present(self):
        """A server that returns isError=True plus a text block: the text is
        the human-readable explanation, so surface it."""
        result = {
            "isError": True,
            "content": [{"type": "text", "text": "device offline"}],
        }
        assert summarise_mcp_result(result) == "device offline"

    def test_handles_missing_content(self):
        assert summarise_mcp_result({}, default="(empty)") == "(empty)"

    def test_handles_non_list_content(self):
        # Server protocol violation: content is a string. Don't crash.
        assert summarise_mcp_result({"content": "oops"}, default="d") == "d"


class TestNormaliseName:
    def test_simple_concatenation(self):
        assert mcp_tools._normalise_name("ha", "turn_on") == "ha__turn_on"

    def test_replaces_invalid_characters(self):
        assert mcp_tools._normalise_name("my-server!", "do.it") == "my-server___do_it"

    def test_truncates_with_hash_suffix(self):
        long_server = "s" * 40
        long_tool = "t" * 40
        out = mcp_tools._normalise_name(long_server, long_tool)
        assert len(out) <= 64
        # Deterministic — same input twice yields same output.
        assert out == mcp_tools._normalise_name(long_server, long_tool)

    def test_different_long_names_get_different_hashes(self):
        a = mcp_tools._normalise_name("s" * 40, "tool_a" * 10)
        b = mcp_tools._normalise_name("s" * 40, "tool_b" * 10)
        assert a != b


class TestExtractSchema:
    def test_pulls_properties_and_required(self):
        tool = {
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x"],
            },
        }
        props, req = mcp_tools._extract_schema(tool)
        assert set(props.keys()) == {"x", "y"}
        assert req == ["x"]

    def test_handles_missing_input_schema(self):
        props, req = mcp_tools._extract_schema({})
        assert props == {}
        assert req == []

    def test_handles_non_dict_input_schema(self):
        # Some servers send a JSON-Schema reference URI as a string. Don't crash.
        props, req = mcp_tools._extract_schema({"inputSchema": "https://..."})
        assert props == {}
        assert req == []

    def test_drops_non_string_required(self):
        tool = {
            "inputSchema": {
                "properties": {"x": {}},
                "required": ["x", 42, None],
            },
        }
        _, req = mcp_tools._extract_schema(tool)
        assert req == ["x"]
