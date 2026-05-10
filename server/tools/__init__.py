"""Voice-bot tools — multimodal function-call capabilities.

Importing this package registers all tools on the global REGISTRY.
The registry, base class, and handler wiring live in `registry.py`.
Domain-specific tool implementations live in:
  - files.py     — file system operations, shell, search
  - desktop.py   — macOS desktop automation (clipboard, browser, capture, terminal, system info)
  - web.py       — web search and external APIs
  - vision.py    — image description chain (no tool classes, called by desktop.py)

For backward compatibility, all module-level tool functions are re-exported
so `import tools; tools.read_file(...)` continues to work.
"""

from tools.registry import BaseTool, REGISTRY, ToolRegistry  # noqa: F401

# Re-export all tool helper functions for backward compatibility.
from tools.desktop import (  # noqa: F401
    capture_display,
    capture_frontmost_window,
    capture_screen_region,
    capture_webcam,
    get_frontmost_app,
    list_browser_tabs,
    list_running_apps,
    read_browser_page_text,
    read_browser_url,
    read_clipboard,
    read_focused_input,
    read_selected_text,
    read_terminal_output,
    take_screenshot,
)
from tools.files import (  # noqa: F401
    append_to_file,
    copy_path,
    delete_path,
    file_info,
    list_folder,
    make_directory,
    move_path,
    patch_file,
    read_file,
    read_finder_selection,
    run_terminal_command,
    search_files,
    write_file,
)
from tools.web import get_current_weather, web_search  # noqa: F401

# Side-effect imports: each module registers its tools on REGISTRY at import time.
import tools.files     # noqa: F401
import tools.desktop   # noqa: F401
import tools.web       # noqa: F401
