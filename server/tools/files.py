"""File-system tools — read, write, patch, move, copy, delete, list, search.

All functions are synchronous (blocking I/O); the registry runs them in
`asyncio.to_thread` so they don't starve the STT/TTS WebSocket heartbeats.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from tools.registry import REGISTRY, BaseTool


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    """Read a text file with line numbers and pagination.

    Output lines are formatted `LINE_NUM|CONTENT`. `offset` is 1-based.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {p}"
    if p.is_dir():
        return f"{p} is a directory, not a file. Use list_folder instead."
    if offset < 1:
        offset = 1
    if limit < 1:
        limit = 500
    try:
        with open(p, "r", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as exc:
        return f"Could not read {p}: {exc}"

    total = len(lines)
    end = min(offset - 1 + limit, total)
    chunk = lines[offset - 1:end]
    body = "".join(
        f"{offset + i}|{line.rstrip(chr(10))}\n" for i, line in enumerate(chunk)
    )
    header = f"File {p.name} (lines {offset}-{end} of {total}):\n"
    if end < total:
        body += f"...({total - end} more lines; raise offset to continue)\n"
    return header + body


def write_file(path: str, content: str) -> str:
    """Create or completely overwrite a file. Creates parent dirs as needed."""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        with open(p, "w") as fh:
            fh.write(content)
    except Exception as exc:
        return f"Could not write {p}: {exc}"
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {p} ({len(content)} bytes)."


def _fuzzy_find(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find `needle` in `haystack` exactly first, then with whitespace-tolerant match.

    Returns (start, end) on success, None on failure or ambiguity.
    """
    idx = haystack.find(needle)
    if idx >= 0:
        # Reject non-unique exact matches.
        if haystack.find(needle, idx + 1) >= 0:
            return None
        return idx, idx + len(needle)

    import re
    pattern = re.compile(
        r"\s*".join(re.escape(tok) for tok in needle.split()),
        re.MULTILINE,
    )
    matches = list(pattern.finditer(haystack))
    if len(matches) == 1:
        m = matches[0]
        return m.start(), m.end()
    return None


def patch_file(path: str, old_str: str, new_str: str) -> str:
    """Targeted edit using fuzzy match. Returns a unified diff of the change."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {p}"
    if p.is_dir():
        return f"{p} is a directory, not a file."
    try:
        original = p.read_text(errors="replace")
    except Exception as exc:
        return f"Could not read {p}: {exc}"

    span = _fuzzy_find(original, old_str)
    if span is None:
        return (
            f"Patch failed: `old_str` not found uniquely in {p}. "
            "Add more surrounding context to make it unique."
        )
    start, end = span
    updated = original[:start] + new_str + original[end:]

    try:
        p.write_text(updated)
    except Exception as exc:
        return f"Could not write {p}: {exc}"

    import difflib
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(p), tofile=str(p), n=3,
    )
    diff_text = "".join(diff)

    # Best-effort syntax check for Python files.
    syntax_msg = ""
    if p.suffix == ".py":
        try:
            compile(updated, str(p), "exec")
            syntax_msg = "\nSyntax check: OK."
        except SyntaxError as exc:
            syntax_msg = f"\nSyntax check FAILED: {exc}"

    return (
        f"Patched {p}.{syntax_msg}\n{diff_text}"
        if diff_text
        else f"Patched {p} (no textual diff)."
    )


def make_directory(path: str, parents: bool = True) -> str:
    """Create a directory. `parents=True` mirrors `mkdir -p`."""
    p = Path(path).expanduser()
    try:
        p.mkdir(parents=parents, exist_ok=True)
    except Exception as exc:
        return f"Could not create {p}: {exc}"
    return f"Directory ready: {p}"


def move_path(src: str, dst: str) -> str:
    """Move or rename a file or directory."""
    import shutil
    s = Path(src).expanduser()
    d = Path(dst).expanduser()
    if not s.exists():
        return f"Source not found: {s}"
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    except Exception as exc:
        return f"Move failed: {exc}"
    return f"Moved {s} → {d}"


def copy_path(src: str, dst: str) -> str:
    """Copy a file or directory (recursive for dirs)."""
    import shutil
    s = Path(src).expanduser()
    d = Path(dst).expanduser()
    if not s.exists():
        return f"Source not found: {s}"
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    except Exception as exc:
        return f"Copy failed: {exc}"
    return f"Copied {s} → {d}"


def delete_path(path: str, recursive: bool = False) -> str:
    """Delete a file or (with recursive=True) a directory tree."""
    import shutil
    p = Path(path).expanduser()
    if not p.exists():
        return f"Path not found: {p}"
    # Refuse to nuke obviously catastrophic targets even when asked.
    resolved = p.resolve()
    if resolved == Path.home() or str(resolved) in ("/", str(Path.home().parent)):
        return f"Refusing to delete {resolved}: too dangerous."
    try:
        if p.is_dir():
            if not recursive:
                return f"{p} is a directory; pass recursive=true to delete it."
            shutil.rmtree(p)
        else:
            p.unlink()
    except Exception as exc:
        return f"Delete failed: {exc}"
    return f"Deleted {p}"


def append_to_file(path: str, content: str) -> str:
    """Append text to a file. Creates the file (and parent dirs) if missing."""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(content)
    except Exception as exc:
        return f"Append failed: {exc}"
    return f"Appended {len(content)} bytes to {p}."


def file_info(path: str) -> dict:
    """Return existence, size, mtime, and type of a path."""
    p = Path(path).expanduser()
    if not p.exists():
        return {"exists": False, "path": str(p)}
    st = p.stat()
    return {
        "exists": True,
        "path": str(p),
        "is_dir": p.is_dir(),
        "is_file": p.is_file(),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_terminal_command(command: str, timeout: int = 30, cwd: str = "") -> dict:
    """Run a shell command via /bin/bash -lc. Captures stdout/stderr.

    Ephemeral: each call is a fresh subprocess (no persistent cwd between
    calls — pass `cwd` if you need to run from a specific directory).
    """
    if not command.strip():
        return {"result": "Empty command.", "exit_code": -1}
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            capture_output=True, text=True,
            timeout=max(1, int(timeout)),
            cwd=cwd or None,
        )
    except subprocess.TimeoutExpired:
        return {"result": f"Command timed out after {timeout}s.", "exit_code": 124}
    except Exception as exc:
        return {"result": f"Command failed to launch: {exc}", "exit_code": -1}

    out = (proc.stdout or "").rstrip()
    err = (proc.stderr or "").rstrip()
    parts = []
    if out:
        parts.append(
            out
            if len(out) <= 1500
            else out[:1500] + f"\n...(truncated, {len(out) - 1500} more chars)"
        )
    if err:
        parts.append(
            f"[stderr]\n{err if len(err) <= 800 else err[:800] + '...(truncated)'}"
        )
    if not parts:
        parts.append(f"(no output, exit={proc.returncode})")
    return {"result": "\n".join(parts), "exit_code": proc.returncode}


# ---------------------------------------------------------------------------
# Search / list
# ---------------------------------------------------------------------------

def search_files(target: str, query: str, path: str = ".", max_results: int = 50) -> str:
    """Search file contents or filenames. Uses ripgrep when available."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Path not found: {p}"

    if target == "content":
        rg = subprocess.run(
            [
                "rg", "--line-number", "--no-heading", "--color=never",
                "--max-count", "5", "-e", query, str(p),
            ],
            capture_output=True, text=True, timeout=15.0,
        )
        if rg.returncode not in (0, 1):
            return f"Search failed: {rg.stderr.strip() or 'rg error'}"
        lines = [l for l in rg.stdout.splitlines() if l]
        if not lines:
            return f"No matches for {query!r} in {p}."
        header = f"{len(lines)} match{'es' if len(lines) != 1 else ''} for {query!r}:"
        if len(lines) > max_results:
            return (
                header + "\n"
                + "\n".join(lines[:max_results])
                + f"\n...and {len(lines) - max_results} more."
            )
        return header + "\n" + "\n".join(lines)

    if target == "filename":
        try:
            matches = [str(m) for m in p.rglob(query)]
        except Exception as exc:
            return f"Filename search failed: {exc}"
        if not matches:
            return f"No files matching {query!r} under {p}."
        header = (
            f"{len(matches)} file{'s' if len(matches) != 1 else ''} matching {query!r}:"
        )
        if len(matches) > max_results:
            return (
                header + "\n"
                + "\n".join(matches[:max_results])
                + f"\n...and {len(matches) - max_results} more."
            )
        return header + "\n" + "\n".join(matches)

    return f"Unknown target {target!r}. Use 'content' or 'filename'."


_LIST_FOLDER_HARD_CAP = 30


def list_folder(path: str, recursive: bool = False, max_items: int = 25) -> str:
    """List the contents of a folder. `recursive` walks subfolders.

    Hard-capped at `_LIST_FOLDER_HARD_CAP` items regardless of caller request —
    voice responses go off the rails past ~30 names anyway.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Folder not found: {p}"
    if not p.is_dir():
        return f"{p} is not a folder."

    try:
        items = list(p.rglob("*")) if recursive else list(p.iterdir())
    except Exception as exc:
        return f"Could not list {p}: {exc}"

    # Folders first, then files, alphabetically.
    items.sort(key=lambda i: (not i.is_dir(), i.name.lower()))
    total = len(items)
    cap = min(max(1, max_items), _LIST_FOLDER_HARD_CAP)
    shown = items[:cap]

    lines = [f"Folder {p} contains {total} item{'s' if total != 1 else ''}:"]
    for item in shown:
        kind = "folder" if item.is_dir() else "file"
        rel = item.relative_to(p) if recursive else Path(item.name)
        lines.append(f"  {kind}: {rel}")
    if total > cap:
        lines.append(f"  ...and {total - cap} more.")
    return "\n".join(lines)


def read_finder_selection() -> str:
    """Return the POSIX paths currently selected in Finder."""
    import sys
    if sys.platform != "darwin":
        return f"Finder selection only works on macOS — current platform is {sys.platform}."
    import subprocess as sp

    script = '''
    tell application "Finder"
        set selectedItems to selection
        if (count of selectedItems) is 0 then
            return ""
        end if
        set output to ""
        repeat with itemRef in selectedItems
            set output to output & POSIX path of (itemRef as alias) & linefeed
        end repeat
        return output
    end tell
    '''
    try:
        result = sp.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10.0, check=True,
        )
        output = result.stdout.strip()
    except sp.TimeoutExpired:
        return "Finder didn't respond in time. Click on Finder once and try again."
    except Exception as exc:
        return f"Could not read Finder selection: {exc}"

    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if not paths:
        return "Nothing is selected in Finder."
    if len(paths) == 1:
        return f"Finder selection: {paths[0]}"
    return f"Finder selection ({len(paths)} items):\n" + "\n".join(paths)


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

@REGISTRY.register
class ReadFileTool(BaseTool):
    name = "read_file"
    category = "files"
    speak_text = "Opening that file."
    description = (
        "Read a text file with line numbers (LINE_NUM|CONTENT format) and "
        "pagination. Use offset/limit to page through large files."
    )
    parameters = {
        "path": {"type": "string", "description": "Absolute or ~-prefixed path to the file."},
        "offset": {"type": "integer", "description": "1-based line to start from. Default 1."},
        "limit": {"type": "integer", "description": "Max lines to return. Default 500."},
    }
    required = ["path"]

    def execute(self, path: str, offset: int = 1, limit: int = 500) -> str:
        return read_file(path, offset, limit)


@REGISTRY.register
class WriteFileTool(BaseTool):
    name = "write_file"
    category = "files"
    speak_text = "Writing that file."
    description = (
        "Create a new file or completely overwrite an existing one. Creates "
        "parent directories as needed. Use `patch` for targeted edits to large files."
    )
    parameters = {
        "path": {"type": "string", "description": "Where to save the file."},
        "content": {"type": "string", "description": "Full content to write."},
    }
    required = ["path", "content"]

    def execute(self, path: str, content: str) -> str:
        return write_file(path, content)


@REGISTRY.register
class PatchFileTool(BaseTool):
    name = "patch"
    category = "files"
    speak_text = "Applying that change."
    description = (
        "Apply a targeted edit to a file via fuzzy find-and-replace. The "
        "`old_str` must appear uniquely in the file; whitespace is tolerated. "
        "Returns a unified diff and (for .py files) a syntax-check result."
    )
    parameters = {
        "path": {"type": "string", "description": "Path to the file being edited."},
        "old_str": {
            "type": "string",
            "description": "Exact block to replace. Include enough context to be unique.",
        },
        "new_str": {"type": "string", "description": "Replacement block."},
    }
    required = ["path", "old_str", "new_str"]

    def execute(self, path: str, old_str: str, new_str: str) -> str:
        return patch_file(path, old_str, new_str)


@REGISTRY.register
class MakeDirectoryTool(BaseTool):
    name = "make_directory"
    category = "files"
    speak_text = "Creating that folder."
    description = "Create a directory (mkdir -p semantics by default)."
    parameters = {
        "path": {"type": "string", "description": "Directory path to create."},
        "parents": {"type": "boolean", "description": "Create missing parents. Default true."},
    }
    required = ["path"]

    def execute(self, path: str, parents: bool = True) -> str:
        return make_directory(path, parents)


@REGISTRY.register
class MovePathTool(BaseTool):
    name = "move_path"
    category = "files"
    speak_text = "Moving that."
    description = "Move or rename a file or directory."
    parameters = {
        "src": {"type": "string", "description": "Source path."},
        "dst": {"type": "string", "description": "Destination path."},
    }
    required = ["src", "dst"]

    def execute(self, src: str, dst: str) -> str:
        return move_path(src, dst)


@REGISTRY.register
class CopyPathTool(BaseTool):
    name = "copy_path"
    category = "files"
    speak_text = "Copying that."
    description = "Copy a file or directory (directories copied recursively)."
    parameters = {
        "src": {"type": "string", "description": "Source path."},
        "dst": {"type": "string", "description": "Destination path."},
    }
    required = ["src", "dst"]

    def execute(self, src: str, dst: str) -> str:
        return copy_path(src, dst)


@REGISTRY.register
class DeletePathTool(BaseTool):
    name = "delete_path"
    category = "files"
    speak_text = "Deleting that."
    description = (
        "Delete a file. For directories, pass recursive=true. "
        "Refuses to delete the home directory or root."
    )
    parameters = {
        "path": {"type": "string", "description": "Path to delete."},
        "recursive": {
            "type": "boolean",
            "description": "Required for directories. Default false.",
        },
    }
    required = ["path"]

    def execute(self, path: str, recursive: bool = False) -> str:
        return delete_path(path, recursive)


@REGISTRY.register
class AppendToFileTool(BaseTool):
    name = "append_to_file"
    category = "files"
    speak_text = "Appending."
    description = "Append text to a file. Creates the file (and parent dirs) if missing."
    parameters = {
        "path": {"type": "string", "description": "File path."},
        "content": {"type": "string", "description": "Text to append."},
    }
    required = ["path", "content"]

    def execute(self, path: str, content: str) -> str:
        return append_to_file(path, content)


@REGISTRY.register
class FileInfoTool(BaseTool):
    name = "file_info"
    category = "files"
    description = "Return existence, size, mtime, and type (file/dir) for a path."
    parameters = {
        "path": {"type": "string", "description": "Path to inspect."},
    }
    required = ["path"]

    def execute(self, path: str) -> dict:
        return file_info(path)


@REGISTRY.register
class RunTerminalCommandTool(BaseTool):
    name = "run_terminal_command"
    category = "system"
    speak_text = "Running that."
    description = (
        "Execute a shell command via bash and return stdout/stderr/exit code. "
        "Each call is a fresh subprocess — pass `cwd` to run from a specific "
        "directory. Use for git, build tools, one-liners. Output is truncated "
        "to keep voice responses tractable."
    )
    parameters = {
        "command": {"type": "string", "description": "The bash command to execute."},
        "timeout": {"type": "integer", "description": "Max seconds to wait. Default 30."},
        "cwd": {"type": "string", "description": "Working directory. Optional."},
    }
    required = ["command"]

    def execute(self, command: str, timeout: int = 30, cwd: str = "") -> dict:
        return run_terminal_command(command, timeout, cwd)


@REGISTRY.register
class SearchFilesTool(BaseTool):
    name = "search_files"
    category = "files"
    speak_text = "Searching."
    description = (
        "Search for a string inside file contents (target='content', uses ripgrep) "
        "or for filenames matching a glob (target='filename')."
    )
    parameters = {
        "target": {
            "type": "string",
            "enum": ["content", "filename"],
            "description": "Search inside file contents or for file names.",
        },
        "query": {
            "type": "string",
            "description": "Regex/string for content; glob for filename.",
        },
        "path": {"type": "string", "description": "Directory to search. Default '.'."},
    }
    required = ["target", "query"]

    def execute(self, target: str, query: str, path: str = ".") -> str:
        return search_files(target, query, path)


@REGISTRY.register
class ListFolderTool(BaseTool):
    name = "list_folder"
    category = "files"
    speak_text = "Listing that folder."
    description = (
        "List the contents of a folder (sorted, folders first). "
        "Set recursive=true to walk subfolders."
    )
    parameters = {
        "path": {"type": "string", "description": "Absolute or ~-prefixed folder path."},
        "recursive": {"type": "boolean", "description": "Walk subfolders. Default false."},
        "max_items": {"type": "integer", "description": "Cap items shown. Default 25."},
    }
    required = ["path"]

    def execute(self, path: str, recursive: bool = False, max_items: int = 25) -> str:
        return list_folder(path, recursive, max_items)


@REGISTRY.register
class ReadFinderSelectionTool(BaseTool):
    name = "read_finder_selection"
    category = "files"
    speak_text = "Checking Finder."
    description = (
        "Return the file or folder paths currently selected in Finder (macOS only). "
        "Use when the user says 'look at what I picked in Finder'."
    )

    def execute(self) -> str:
        return read_finder_selection()
