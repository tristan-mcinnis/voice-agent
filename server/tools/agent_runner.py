"""External-agent runtime — spawn, track, and reap long-running agent processes.

This module is the engine that backs every `spawn_*` tool in
`tools/external_agents.py`. It's deliberately filesystem-first: each task lives
in its own directory under `.voice-agent/agent-tasks/<task_id>/` with three
files (`meta.json`, `stdout.log`, `stderr.log`), and tool calls operate on
those files instead of a long-lived in-memory registry. That means:

- `get_agent_task` / `list_agent_tasks` work after a bot restart.
- Task state survives crashes — you just lose the reaper thread and the
  task gets marked "orphaned" on next status check via PID liveness.
- No global mutable state to lock around (each task dir is the lock).

Tasks are *not* run with `start_new_session=True`: when the bot dies, agent
subprocesses die with it. That's intentional — voice sessions are
short-lived and we don't want zombie LLM processes burning tokens after
the user closed the lid. If you want detached background runs, that's a
separate feature.

Sync, not asyncio. Tool `execute()` methods already run in
`asyncio.to_thread`, so there's no benefit to async subprocess plumbing
here — and a lot of complexity avoided.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Task store — pure filesystem layout (no subprocess).
# Extracted so unit tests can exercise the on-disk format without spawning.
# ---------------------------------------------------------------------------


def _tasks_root() -> Path:
    """Resolve the task directory from `VOICE_AGENT_HOME` (or cwd fallback)."""
    home = os.getenv("VOICE_AGENT_HOME")
    if home:
        return Path(home) / "agent-tasks"
    return Path.cwd() / ".voice-agent" / "agent-tasks"


def _new_task_id() -> str:
    """Short, sortable task id. Sortable so `ls` lists oldest-first."""
    return f"{int(time.time())}-{secrets.token_hex(3)}"


@dataclass
class TaskMeta:
    """The on-disk shape of `meta.json`. Mirrors keys exactly."""

    task_id: str
    agent: str
    task: str
    cwd: str
    argv: list[str]
    status: str               # "running" | "done" | "failed" | "cancelled" | "orphaned"
    pid: Optional[int] = None
    started_at: float = 0.0
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    # Extra context the spawning tool wants to carry forward (model, args, etc).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TaskMeta":
        data = json.loads(text)
        # Tolerate extra unknown fields written by future versions.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class TaskStore:
    """Filesystem-backed task directory. All operations are atomic.

    The shape of `<root>/<task_id>/`:
        meta.json    — TaskMeta as JSON
        stdout.log   — child stdout (raw, line-buffered when possible)
        stderr.log   — child stderr
        summary.txt  — optional human summary written on completion

    Atomic writes via write-temp-then-rename so a crash mid-update can't
    corrupt meta.json.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _tasks_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir_for(self, task_id: str) -> Path:
        return self.root / task_id

    def write_meta(self, meta: TaskMeta) -> None:
        d = self.dir_for(meta.task_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "meta.json.tmp"
        tmp.write_text(meta.to_json())
        tmp.replace(d / "meta.json")

    def read_meta(self, task_id: str) -> TaskMeta:
        path = self.dir_for(task_id) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"no task with id {task_id!r}")
        return TaskMeta.from_json(path.read_text())

    def update_meta(self, task_id: str, **fields: Any) -> TaskMeta:
        meta = self.read_meta(task_id)
        for k, v in fields.items():
            setattr(meta, k, v)
        self.write_meta(meta)
        return meta

    def list_meta(self) -> list[TaskMeta]:
        out: list[TaskMeta] = []
        if not self.root.exists():
            return out
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.exists():
                continue
            try:
                out.append(TaskMeta.from_json(meta_path.read_text()))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(f"agent-runner: skipping {child.name}: {exc}")
        return out

    def tail(self, task_id: str, stream: str = "stdout", lines: int = 50) -> str:
        if stream not in ("stdout", "stderr"):
            raise ValueError(f"stream must be stdout|stderr, got {stream!r}")
        path = self.dir_for(task_id) / f"{stream}.log"
        if not path.exists():
            return ""
        return _tail_text(path.read_text(errors="replace"), lines)


def _tail_text(text: str, n: int) -> str:
    """Return the last `n` lines of `text`. Pure — easy to unit test."""
    if n <= 0:
        return ""
    parts = text.splitlines()
    return "\n".join(parts[-n:])


# ---------------------------------------------------------------------------
# Process liveness — read-only check that doesn't reap.
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Return True if `pid` exists. `os.kill(pid, 0)` is the POSIX idiom."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Some other user owns it — still alive, just not ours to signal.
        return True
    return True


def reconcile_status(meta: TaskMeta) -> TaskMeta:
    """If a task is marked `running` but its PID is dead, mark it `orphaned`.

    Called by `list_agent_tasks` and `get_agent_task` so stale state from a
    previous bot run gets cleaned up lazily — no background daemon needed.
    """
    if meta.status != "running":
        return meta
    if meta.pid is None or not _pid_alive(meta.pid):
        meta.status = "orphaned"
        meta.finished_at = meta.finished_at or time.time()
    return meta


# ---------------------------------------------------------------------------
# Spawn — the only function that actually starts a subprocess.
# ---------------------------------------------------------------------------


def spawn(
    *,
    agent: str,
    argv: list[str],
    cwd: str,
    task: str,
    stdin_text: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    store: Optional[TaskStore] = None,
) -> TaskMeta:
    """Start a subprocess, write initial meta, and return it.

    The caller (a `BaseTool.execute()`) gets the TaskMeta back immediately —
    typically within ~10ms of the syscall. A daemon thread watches the
    Popen object and updates `meta.json` on exit. If the bot dies before
    the child does, the daemon thread dies with it and the task's status
    stays `running` on disk; next `reconcile_status()` call marks it
    `orphaned`.

    `stdin_text` is for piping the prompt into agents that read it from
    stdin (e.g. `claude -p` reads the prompt as a positional arg, but
    `codex exec` accepts stdin in some modes — left to the caller).
    """
    store = store or TaskStore()
    task_id = _new_task_id()
    task_dir = store.dir_for(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = task_dir / "stdout.log"
    stderr_path = task_dir / "stderr.log"

    meta = TaskMeta(
        task_id=task_id,
        agent=agent,
        task=task,
        cwd=str(Path(cwd).expanduser().resolve()),
        argv=list(argv),
        status="running",
        started_at=time.time(),
        extra=dict(extra or {}),
    )
    store.write_meta(meta)

    try:
        stdout_fh = open(stdout_path, "w", buffering=1)
        stderr_fh = open(stderr_path, "w", buffering=1)
    except OSError as exc:
        meta.status = "failed"
        meta.error = f"could not open log files: {exc}"
        meta.finished_at = time.time()
        store.write_meta(meta)
        return meta

    try:
        proc = subprocess.Popen(
            argv,
            cwd=meta.cwd,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        stdout_fh.close()
        stderr_fh.close()
        meta.status = "failed"
        meta.error = f"{type(exc).__name__}: {exc}"
        meta.finished_at = time.time()
        store.write_meta(meta)
        logger.warning(f"agent-runner: failed to spawn {agent}: {exc}")
        return meta

    if stdin_text is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_text)
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            logger.warning(f"agent-runner: stdin write to {agent} failed: {exc}")

    meta.pid = proc.pid
    store.write_meta(meta)
    logger.info(
        f"agent-runner: spawned {agent} task={task_id} pid={proc.pid} "
        f"argv={shlex.join(argv)}"
    )

    threading.Thread(
        target=_reap,
        args=(proc, store, task_id, stdout_fh, stderr_fh),
        daemon=True,
        name=f"agent-reap-{task_id}",
    ).start()

    return meta


def _reap(
    proc: subprocess.Popen,
    store: TaskStore,
    task_id: str,
    stdout_fh,
    stderr_fh,
) -> None:
    """Wait on `proc` and finalise meta.json. Runs in a daemon thread."""
    try:
        rc = proc.wait()
    except Exception as exc:
        logger.warning(f"agent-runner: wait() failed for task {task_id}: {exc}")
        rc = -1
    finally:
        for fh in (stdout_fh, stderr_fh):
            try:
                fh.close()
            except Exception:
                pass

    try:
        meta = store.read_meta(task_id)
        # If the task was already cancelled, keep that status — `rc` will be
        # negative anyway from the signal.
        if meta.status not in ("cancelled", "failed"):
            meta.status = "done" if rc == 0 else "failed"
        meta.exit_code = rc
        meta.finished_at = time.time()
        store.write_meta(meta)
    except FileNotFoundError:
        return
    except OSError as exc:
        # The task dir was deleted out from under us (e.g. test teardown).
        # Reaper is best-effort — there's no one to report the failure to.
        logger.debug(f"agent-runner: reap finalize skipped for {task_id}: {exc}")


def cancel(task_id: str, *, store: Optional[TaskStore] = None) -> TaskMeta:
    """Send SIGTERM to a running task. Idempotent — finished tasks are no-ops."""
    store = store or TaskStore()
    meta = store.read_meta(task_id)
    if meta.status != "running" or meta.pid is None:
        return meta
    try:
        os.kill(meta.pid, signal.SIGTERM)
    except ProcessLookupError:
        # Already dead — let reconcile_status() mark it orphaned on next read.
        pass
    except PermissionError as exc:
        return store.update_meta(task_id, error=f"cancel: {exc}")
    return store.update_meta(task_id, status="cancelled")
