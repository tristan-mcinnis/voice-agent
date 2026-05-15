"""Unit tests for memory layer — entry loading, saving, add/replace/list actions."""

import os
import tempfile
from pathlib import Path

import pytest
from agent.memory_layer import (
    MemoryLayer,
    USER_CHAR_LIMIT,
    MEMORY_CHAR_LIMIT,
)
from agent.memory_layer import _load_entries, _save_entries, _total_chars, _limit_for


class TestLoadSaveEntries:
    def test_load_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            path = Path(f.name)
        try:
            assert _load_entries(path) == []
        finally:
            path.unlink()

    def test_load_single_entry(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Hello world\n")
            path = Path(f.name)
        try:
            assert _load_entries(path) == ["Hello world"]
        finally:
            path.unlink()

    def test_load_multiple_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Entry one\n§\nEntry two\n§\nEntry three\n")
            path = Path(f.name)
        try:
            assert _load_entries(path) == ["Entry one", "Entry two", "Entry three"]
        finally:
            path.unlink()

    def test_save_and_reload_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.md"
            entries = ["Fact A", "Fact B", "Fact C"]
            _save_entries(path, entries)
            assert _load_entries(path) == entries

    def test_total_chars(self):
        assert _total_chars(["abc", "de"]) == 5
        assert _total_chars([]) == 0

    def test_limit_for(self):
        assert _limit_for("user") == USER_CHAR_LIMIT
        assert _limit_for("memory") == MEMORY_CHAR_LIMIT


class TestMemoryLayerActions:
    """Integration-style tests on MemoryLayer with temp directories."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self._tmpdir = tempfile.TemporaryDirectory()
        monkeypatch.setitem(os.environ, "VOICE_AGENT_HOME", self._tmpdir.name)
        self.layer = MemoryLayer()
        yield
        self._tmpdir.cleanup()

    def test_list_empty(self):
        result = self.layer.list_entries("user")
        assert "is empty" in result["result"]
        assert result["chars_used"] == 0

    def test_list_empty_memory(self):
        result = self.layer.list_entries("memory")
        assert "is empty" in result["result"]

    def test_add_simple(self):
        result = self.layer.add("user", "User likes Python.")
        assert "Added" in result["result"]
        assert result["chars_used"] == len("User likes Python.")

        # Verify on disk.
        entries = _load_entries(Path(self._tmpdir.name) / "memories" / "USER.md")
        assert "User likes Python." in entries

    def test_add_then_list(self):
        self.layer.add("user", "Fact 1")
        self.layer.add("user", "Fact 2")
        result = self.layer.list_entries("user")
        assert "[0]" in result["result"]
        assert "[1]" in result["result"]
        assert result["chars_used"] > 0

    def test_add_near_duplicate_rejected(self):
        self.layer.add("user", "User uses VS Code.")
        result = self.layer.add("user", "uses VS Code")
        assert "Similar entry already exists" in result["result"]

    def test_add_empty_content_rejected(self):
        result = self.layer.add("user", "")
        assert "Cannot add empty" in result["result"]

    def test_replace_by_old_text(self):
        self.layer.add("memory", "Project uses Python 3.12.")
        result = self.layer.replace(
            "memory",
            content="Project uses Python 3.14.",
            old_text="Python 3.12",
        )
        assert "Replaced entry" in result["result"]
        assert result["replaced_index"] == 0

    def test_replace_nonexistent_old_text(self):
        self.layer.add("memory", "Foo bar baz.")
        result = self.layer.replace(
            "memory",
            content="New content",
            old_text="nonexistent",
        )
        assert "No entry" in result["result"]

    def test_replace_without_old_text_shows_list(self):
        self.layer.add("memory", "Some fact.")
        result = self.layer.replace("memory", content="New")
        assert "provide 'old_text'" in result["result"].lower()

    def test_replace_empty_content_rejected(self):
        self.layer.add("memory", "Some fact.")
        result = self.layer.replace("memory", content="", old_text="Some")
        assert "Cannot replace with empty" in result["result"]


class TestMemoryPressure:
    """Character limit enforcement tests."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self._tmpdir = tempfile.TemporaryDirectory()
        monkeypatch.setitem(os.environ, "VOICE_AGENT_HOME", self._tmpdir.name)
        self.layer = MemoryLayer()

        # Pre-fill with entries near the limit.
        # USER limit is 1375. Add enough to leave < 50 chars.
        big_entry = "x" * (USER_CHAR_LIMIT - 10)
        self.layer.add("user", big_entry)
        yield
        self._tmpdir.cleanup()

    def test_pressure_reported_when_over_limit(self):
        result = self.layer.add("user", "y" * 50)
        assert "Memory pressure" in result["result"]

    def test_add_under_limit_succeeds(self):
        result = self.layer.add("user", "hi")
        assert "Added" in result["result"]

    def test_replace_does_not_check_pressure(self):
        """Replace should always work since you're swapping, not growing."""
        # After pre-filling with x*1365, replace with x*1365 — same size.
        # Replace doesn't have a pressure check — only add does.
        self.layer.replace("user", content="z" * (USER_CHAR_LIMIT - 10), old_text="x" * 10)
        result = self.layer.list_entries("user")
        assert "z" * 50 in result["result"]


class TestAtomicWrite:
    """Verify temp-file + rename semantics: writes are all-or-nothing."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self._tmpdir = tempfile.TemporaryDirectory()
        monkeypatch.setenv("VOICE_AGENT_HOME", self._tmpdir.name)
        self.layer = MemoryLayer()
        yield
        self._tmpdir.cleanup()

    def test_no_temp_file_left_behind_after_success(self):
        self.layer.add("user", "first")
        self.layer.add("user", "second")
        # Temp files use leading-dot prefix. None should remain.
        leftovers = list(self.layer.base_path.glob(".*.tmp.*"))
        assert leftovers == []

    def test_write_failure_preserves_old_file(self, monkeypatch):
        # Establish a known-good state.
        self.layer.add("user", "original entry")
        path = self.layer.base_path / "USER.md"
        original = path.read_text()

        # Force the next save to fail mid-flight (after fsync, before rename).
        import agent.memory_layer as ml
        real_replace = ml.os.replace
        def boom(*args, **kwargs):
            raise OSError("simulated disk error")
        monkeypatch.setattr(ml.os, "replace", boom)

        with pytest.raises(OSError):
            self.layer.add("user", "doomed entry")

        # Old file content is untouched.
        assert path.read_text() == original
        # Restore and verify recovery works.
        monkeypatch.setattr(ml.os, "replace", real_replace)
        self.layer.add("user", "recovered")
        assert "recovered" in path.read_text()


class TestConcurrency:
    """Concurrent add/replace should never lose updates."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self._tmpdir = tempfile.TemporaryDirectory()
        monkeypatch.setenv("VOICE_AGENT_HOME", self._tmpdir.name)
        self.layer = MemoryLayer()
        yield
        self._tmpdir.cleanup()

    def test_parallel_adds_all_succeed(self):
        """20 threads each add a distinct entry — all 20 land."""
        from concurrent.futures import ThreadPoolExecutor

        def do_add(i: int) -> None:
            self.layer.add("memory", f"entry-{i:03d}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(do_add, range(20)))

        entries = _load_entries(self.layer.base_path / "MEMORY.md")
        # Every entry shows up exactly once.
        assert sorted(entries) == sorted(f"entry-{i:03d}" for i in range(20))
