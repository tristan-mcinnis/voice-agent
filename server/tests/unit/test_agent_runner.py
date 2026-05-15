"""Unit tests for tools.agent_runner — task store, tail, reconcile, spawn errors.

Real subprocesses are launched via the system `echo` / `false` / `sleep`
binaries so we exercise the spawn path end-to-end without depending on any
LLM SDK. Tests that require real signals are skipped on Windows.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from tools import agent_runner as ar


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestTailText:
    def test_returns_last_n_lines(self):
        text = "a\nb\nc\nd\ne"
        assert ar._tail_text(text, 2) == "d\ne"

    def test_returns_all_when_n_exceeds_length(self):
        text = "one\ntwo"
        assert ar._tail_text(text, 50) == "one\ntwo"

    def test_returns_empty_for_zero_or_negative(self):
        assert ar._tail_text("anything", 0) == ""
        assert ar._tail_text("anything", -5) == ""

    def test_handles_empty_input(self):
        assert ar._tail_text("", 5) == ""


class TestTaskMetaRoundtrip:
    def test_to_json_includes_required_fields(self):
        meta = ar.TaskMeta(
            task_id="abc", agent="claude_code", task="do thing",
            cwd="/tmp", argv=["claude", "-p", "do thing"], status="running",
        )
        data = json.loads(meta.to_json())
        assert data["task_id"] == "abc"
        assert data["status"] == "running"
        assert data["argv"] == ["claude", "-p", "do thing"]

    def test_from_json_tolerates_unknown_fields(self):
        """Future-version meta.json files shouldn't crash parsing."""
        blob = json.dumps({
            "task_id": "abc",
            "agent": "x",
            "task": "y",
            "cwd": "/",
            "argv": [],
            "status": "done",
            "exit_code": 0,
            "unknown_future_field": "ignore me",
            "started_at": 1.0,
            "finished_at": 2.0,
            "pid": 42,
            "extra": {},
            "error": None,
        })
        meta = ar.TaskMeta.from_json(blob)
        assert meta.task_id == "abc"
        assert meta.exit_code == 0


class TestTaskStore:
    def test_write_then_read(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        meta = ar.TaskMeta(
            task_id="t1", agent="a", task="t", cwd="/", argv=[], status="running",
        )
        store.write_meta(meta)
        loaded = store.read_meta("t1")
        assert loaded.task_id == "t1"
        assert loaded.status == "running"

    def test_read_missing_raises(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.read_meta("does-not-exist")

    def test_update_meta_patches_fields(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        meta = ar.TaskMeta(
            task_id="t1", agent="a", task="t", cwd="/", argv=[], status="running",
        )
        store.write_meta(meta)
        updated = store.update_meta("t1", status="done", exit_code=0)
        assert updated.status == "done"
        assert updated.exit_code == 0
        # And it persisted.
        assert store.read_meta("t1").status == "done"

    def test_list_meta_returns_all_directories(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        for tid in ("a", "b", "c"):
            store.write_meta(ar.TaskMeta(
                task_id=tid, agent="x", task="y", cwd="/", argv=[], status="done",
            ))
        names = {m.task_id for m in store.list_meta()}
        assert names == {"a", "b", "c"}

    def test_list_meta_skips_bad_json(self, tmp_path):
        """A corrupted task dir should not break listing of healthy ones."""
        store = ar.TaskStore(tmp_path)
        store.write_meta(ar.TaskMeta(
            task_id="ok", agent="x", task="y", cwd="/", argv=[], status="done",
        ))
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "meta.json").write_text("not json at all")
        names = {m.task_id for m in store.list_meta()}
        assert names == {"ok"}

    def test_tail_reads_log(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        meta = ar.TaskMeta(
            task_id="t1", agent="a", task="t", cwd="/", argv=[], status="done",
        )
        store.write_meta(meta)
        (tmp_path / "t1" / "stdout.log").write_text("x\ny\nz\n")
        assert store.tail("t1", "stdout", 2) == "y\nz"

    def test_tail_missing_returns_empty(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        store.write_meta(ar.TaskMeta(
            task_id="t1", agent="a", task="t", cwd="/", argv=[], status="done",
        ))
        assert store.tail("t1", "stdout") == ""

    def test_tail_rejects_invalid_stream(self, tmp_path):
        store = ar.TaskStore(tmp_path)
        with pytest.raises(ValueError):
            store.tail("t1", "garbage")


class TestReconcileStatus:
    def test_finished_task_passes_through(self, tmp_path):
        meta = ar.TaskMeta(
            task_id="t", agent="a", task="t", cwd="/", argv=[],
            status="done", exit_code=0,
        )
        assert ar.reconcile_status(meta).status == "done"

    def test_running_with_live_pid_stays_running(self, tmp_path, monkeypatch):
        meta = ar.TaskMeta(
            task_id="t", agent="a", task="t", cwd="/", argv=[],
            status="running", pid=12345,
        )
        monkeypatch.setattr(ar, "_pid_alive", lambda pid: True)
        assert ar.reconcile_status(meta).status == "running"

    def test_running_with_dead_pid_becomes_orphaned(self, tmp_path, monkeypatch):
        meta = ar.TaskMeta(
            task_id="t", agent="a", task="t", cwd="/", argv=[],
            status="running", pid=12345,
        )
        monkeypatch.setattr(ar, "_pid_alive", lambda pid: False)
        result = ar.reconcile_status(meta)
        assert result.status == "orphaned"
        assert result.finished_at is not None


# ---------------------------------------------------------------------------
# Spawn — exercised with real cheap subprocesses (echo, false, sleep).
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return ar.TaskStore(tmp_path)


class TestSpawnRealProcess:
    def test_spawns_and_reaps_successful_command(self, store, tmp_path):
        if shutil.which("echo") is None:
            pytest.skip("echo not on PATH")
        meta = ar.spawn(
            agent="echo-test",
            argv=["echo", "hello world"],
            cwd=str(tmp_path),
            task="say hi",
            store=store,
        )
        assert meta.task_id
        assert meta.pid is not None
        # Wait for the daemon reaper to finalise. echo exits fast.
        for _ in range(50):
            final = store.read_meta(meta.task_id)
            if final.status != "running":
                break
            time.sleep(0.05)
        else:
            pytest.fail("reaper didn't finalise within 2.5s")
        assert final.status == "done"
        assert final.exit_code == 0
        assert "hello world" in (tmp_path / meta.task_id / "stdout.log").read_text()

    def test_spawns_and_marks_failure_for_nonzero_exit(self, store, tmp_path):
        if shutil.which("false") is None:
            pytest.skip("false not on PATH")
        meta = ar.spawn(
            agent="false-test",
            argv=["false"],
            cwd=str(tmp_path),
            task="fail",
            store=store,
        )
        for _ in range(50):
            final = store.read_meta(meta.task_id)
            if final.status != "running":
                break
            time.sleep(0.05)
        assert final.status == "failed"
        assert final.exit_code == 1

    def test_missing_binary_marks_failed_immediately(self, store, tmp_path):
        meta = ar.spawn(
            agent="missing",
            argv=["/nonexistent/binary/does/not/exist"],
            cwd=str(tmp_path),
            task="x",
            store=store,
        )
        assert meta.status == "failed"
        assert meta.error
        assert meta.pid is None

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX-only")
    def test_cancel_sends_sigterm(self, store, tmp_path):
        if shutil.which("sleep") is None:
            pytest.skip("sleep not on PATH")
        meta = ar.spawn(
            agent="sleep-test",
            argv=["sleep", "30"],
            cwd=str(tmp_path),
            task="block",
            store=store,
        )
        assert meta.status == "running"
        cancelled = ar.cancel(meta.task_id, store=store)
        assert cancelled.status == "cancelled"
        # The reaper will run after SIGTERM; give it a beat to record exit_code.
        for _ in range(50):
            final = store.read_meta(meta.task_id)
            if final.exit_code is not None:
                break
            time.sleep(0.05)
        assert final.exit_code is not None
