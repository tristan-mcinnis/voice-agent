"""Unit tests for grounding_vision._parse_coords — pure parsing, no model calls."""

from __future__ import annotations

import tools.grounding_vision as gv


class TestParseCoords:
    def test_raw_json(self):
        assert gv._parse_coords('{"x": 123, "y": 456}') == {"x": 123, "y": 456}

    def test_json_with_whitespace(self):
        assert gv._parse_coords('  {"x": 10, "y": 20}\n') == {"x": 10, "y": 20}

    def test_json_in_code_fence(self):
        assert gv._parse_coords('```json\n{"x": 5, "y": 6}\n```') == {"x": 5, "y": 6}

    def test_json_embedded_in_prose(self):
        text = 'Here are the coords: {"x": 100, "y": 200}, hope this helps.'
        assert gv._parse_coords(text) == {"x": 100, "y": 200}

    def test_negative_coords_means_not_found(self):
        assert gv._parse_coords('{"x": -1, "y": -1}') is None
        assert gv._parse_coords('{"x": -1, "y": 50}') is None

    def test_uitars_click_token(self):
        assert gv._parse_coords("<click>250, 300</click>") == {"x": 250, "y": 300}

    def test_os_atlas_box_token_returns_center(self):
        # box (100,100)-(200,200) → center (150, 150)
        assert gv._parse_coords("<box>100, 100, 200, 200</box>") == {"x": 150, "y": 150}

    def test_last_resort_pair(self):
        assert gv._parse_coords("The button is at 450, 600.") == {"x": 450, "y": 600}

    def test_empty_input(self):
        assert gv._parse_coords("") is None
        assert gv._parse_coords(None) is None  # type: ignore[arg-type]

    def test_garbage_input(self):
        assert gv._parse_coords("I don't know where it is.") is None
        assert gv._parse_coords("just 42") is None

    def test_picks_first_pair_when_multiple_present(self):
        # Real-world case: model says "tried 100, 200 but actually 300, 400".
        # We accept the first pair — agent can re-prompt if it's wrong.
        assert gv._parse_coords("around 100, 200 area") == {"x": 100, "y": 200}
