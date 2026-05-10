"""Voice-bot tools — multimodal function-call capabilities.

Importing this package registers all tools on the global REGISTRY.
The registry, base class, and handler wiring live in `registry.py`.
Domain-specific tool implementations live in:
  - files.py     — file system operations, shell, search
  - desktop.py   — macOS desktop automation (clipboard, browser, capture, terminal, system info)
  - web.py       — web search and external APIs
  - vision.py    — image description chain (no tool classes, called by desktop.py)

Backward-compat callables (e.g. ``tools.read_file(...)``) delegate to the
canonical ``BaseTool.execute()`` implementations via the REGISTRY.
"""

from tools.registry import BaseTool, REGISTRY, ToolRegistry  # noqa: F401

# Side-effect imports: each module registers its tools on REGISTRY at import time.
import tools.files     # noqa: F401
import tools.capture   # noqa: F401
import tools.desktop   # noqa: F401
import tools.computer_use  # noqa: F401
import tools.web       # noqa: F401
import tools.memory    # noqa: F401
import tools.search_history  # noqa: F401


# ---------------------------------------------------------------------------
# Backward-compat: module-level callables that delegate to REGISTRY.execute()
# ---------------------------------------------------------------------------

def _compat(name: str):
    """Return a callable that delegates to REGISTRY.get(name).execute()."""
    def wrapper(*args, **kwargs):
        return REGISTRY.get(name).execute(*args, **kwargs)
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


# -- files.py compat ----------------------------------------------------------
read_file = _compat("read_file")
write_file = _compat("write_file")
patch_file = _compat("patch")
make_directory = _compat("make_directory")
move_path = _compat("move_path")
copy_path = _compat("copy_path")
delete_path = _compat("delete_path")
append_to_file = _compat("append_to_file")
file_info = _compat("file_info")
run_terminal_command = _compat("run_terminal_command")
search_files = _compat("search_files")
list_folder = _compat("list_folder")
read_finder_selection = _compat("read_finder_selection")

# -- capture.py compat -------------------------------------------------------
take_screenshot = _compat("take_screenshot")
capture_webcam = _compat("capture_webcam")
capture_frontmost_window = _compat("capture_frontmost_window")
capture_screen_region = _compat("capture_screen_region")
capture_display = _compat("capture_display")

# -- desktop.py compat -------------------------------------------------------
read_clipboard = _compat("read_clipboard")
read_selected_text = _compat("read_selected_text")
read_focused_input = _compat("read_focused_input")
read_browser_url = _compat("read_browser_url")
read_browser_page_text = _compat("read_browser_page_text")
list_browser_tabs = _compat("list_browser_tabs")
get_frontmost_app = _compat("get_frontmost_app")
list_running_apps = _compat("list_running_apps")
read_terminal_output = _compat("read_terminal_output")

# -- computer_use.py compat --------------------------------------------------
list_ui_elements = _compat("list_ui_elements")
click_element = _compat("click_element")
click_at = _compat("click_at")
type_text = _compat("type_text")
press_key = _compat("press_key")
scroll = _compat("scroll")
mouse_move = _compat("mouse_move")

# -- web.py compat -----------------------------------------------------------
web_search = _compat("web_search")
get_current_weather = _compat("get_current_weather")

# -- memory.py compat --------------------------------------------------------
memory = _compat("memory")
patch_memory = _compat("memory")  # legacy name

# -- search_history.py compat ------------------------------------------------
search_history = _compat("search_history")
