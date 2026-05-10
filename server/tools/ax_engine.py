"""AX engine — walk the macOS Accessibility tree to discover clickable UI elements.

Extracted from ``computer_use.py``. The AX tree walking (PyObjC CoreFoundation calls)
is dense, macOS-specific code that's difficult to test without a real app running.
Extracting it into its own deepened module lets callers test with a fake engine
and keeps the tool classes thin.

Usage::

    from tools.ax_engine import AXEngine, AXElement

    engine = AXEngine()
    proc_name, elements = engine.list_frontmost()
    for i, el in enumerate(elements):
        print(el.short(i))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Roles we consider interactive. AXStaticText included so the agent can also
# discover labels next to unlabeled icons (e.g. song titles in Spotify rows).
_INTERACTIVE_ROLES = {
    "AXButton", "AXMenuItem", "AXMenuButton", "AXPopUpButton",
    "AXTextField", "AXTextArea", "AXSearchField",
    "AXLink", "AXCheckBox", "AXRadioButton",
    "AXTabGroup", "AXTab", "AXRow", "AXCell",
    "AXImage", "AXStaticText",
}


@dataclass
class AXElement:
    """One interactive UI element with role, label, and screen position."""

    role: str
    label: str
    x: int
    y: int
    w: int
    h: int

    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def short(self, idx: int) -> str:
        label = self.label or "(unlabeled)"
        # Trim long labels — Spotify/Finder produce huge "description" strings.
        if len(label) > 80:
            label = label[:77] + "…"
        role = self.role.replace("AX", "")
        return f"[{idx}] {role:<10} {label}  @({self.x + self.w // 2},{self.y + self.h // 2})"


class AXEngine:
    """Walk the macOS Accessibility tree and return interactive elements.

    Uses the native macOS Accessibility C API through PyObjC — ~50x faster
    than AppleScript's "entire contents" on heavy apps like IDEs and Spotify.

    Bounded by ``max_elements`` / ``max_depth`` so a runaway tree (web-view
    content, IDE outline panes) can't hang the voice loop.

    The ``_LAST_ELEMENTS`` / ``_LAST_PROCESS`` cache is per-instance so
    ``list_ui_elements`` can populate it and ``click_element`` can resolve
    indices without re-walking the tree.
    """

    def __init__(self, max_elements: int = 600, max_depth: int = 25) -> None:
        self._max_elements = max_elements
        self._max_depth = max_depth
        # Per-instance cache: last listing keyed by process name.
        self.last_elements: dict[str, list[AXElement]] = {}
        self.last_process: list[str] = []  # stack to remember most recent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_frontmost(
        self, filter: str = "", max_items: int = 80
    ) -> tuple[str, list[AXElement], str]:
        """Walk the AX tree of the frontmost app. Returns (proc_name, elements, display_text).

        Args:
            filter: Optional case-insensitive substring to filter labels.
            max_items: Cap display rows for the LLM (full list still cached).

        Returns:
            (process_name, full_element_list, formatted_display_text)
        """
        proc, elements = self._ax_list()
        self.last_elements[proc] = elements
        if proc in self.last_process:
            self.last_process.remove(proc)
        self.last_process.append(proc)

        # Build the display text from the (possibly filtered) view.
        view = elements
        if filter:
            f = filter.lower()
            view = [e for e in elements if f in e.label.lower() or f in e.role.lower()]

        truncated = False
        if len(view) > max_items:
            view = view[:max_items]
            truncated = True

        # Indices are into the full (unfiltered) list so click_element is stable.
        idx_lookup = {id(e): i for i, e in enumerate(elements)}
        lines = [f"{proc}: {len(elements)} interactive elements"]
        if filter:
            lines[0] += f" ({len(view)} match filter {filter!r})"
        for e in view:
            lines.append(e.short(idx_lookup[id(e)]))
        if truncated:
            lines.append(f"... (truncated to {max_items}; refine with filter=)")

        return proc, elements, "\n".join(lines)

    def resolve(self, index: int) -> Optional[AXElement]:
        """Look up an element by index in the most recently listed process."""
        if not self.last_process:
            return None
        proc = self.last_process[-1]
        elems = self.last_elements.get(proc, [])
        if 0 <= index < len(elems):
            return elems[index]
        return None

    # ------------------------------------------------------------------
    # AX tree walking (PyObjC)
    # ------------------------------------------------------------------

    def _ax_list(self) -> tuple[str, list[AXElement]]:
        """Walk the AX tree of the frontmost app via AXUIElementRef.

        Returns (process_name, elements). Bounded by max_elements / max_depth.
        """
        from AppKit import NSWorkspace  # type: ignore
        from ApplicationServices import (  # type: ignore
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
            AXValueGetValue,
            kAXValueCGPointType,
            kAXValueCGSizeType,
        )

        ws = NSWorkspace.sharedWorkspace()
        front_app = ws.frontmostApplication()
        if front_app is None:
            return "(unknown)", []
        proc_name = str(front_app.localizedName() or "")
        pid = int(front_app.processIdentifier())

        app_ref = AXUIElementCreateApplication(pid)

        def _attr(ref, name):
            err, value = AXUIElementCopyAttributeValue(ref, name, None)
            if err:
                return None
            return value

        def _point(value) -> Optional[tuple[float, float]]:
            if value is None:
                return None
            success, pt = AXValueGetValue(value, kAXValueCGPointType, None)
            return (pt.x, pt.y) if success else None

        def _size(value) -> Optional[tuple[float, float]]:
            if value is None:
                return None
            success, sz = AXValueGetValue(value, kAXValueCGSizeType, None)
            return (sz.width, sz.height) if success else None

        def _label(ref) -> str:
            for attr in ("AXTitle", "AXValue", "AXDescription", "AXHelp", "AXLabel"):
                v = _attr(ref, attr)
                if v:
                    s = str(v).strip()
                    if s:
                        return s
            return ""

        elements: list[AXElement] = []

        def _walk(ref, depth: int) -> None:
            if len(elements) >= self._max_elements or depth > self._max_depth:
                return
            role = _attr(ref, "AXRole")
            if role and str(role) in _INTERACTIVE_ROLES:
                pos = _point(_attr(ref, "AXPosition"))
                sz = _size(_attr(ref, "AXSize"))
                if pos and sz and sz[0] > 0 and sz[1] > 0:
                    elements.append(AXElement(
                        role=str(role),
                        label=_label(ref),
                        x=int(pos[0]), y=int(pos[1]),
                        w=int(sz[0]), h=int(sz[1]),
                    ))
            children = _attr(ref, "AXChildren") or []
            for child in children:
                _walk(child, depth + 1)

        # Start from the focused / main window if available, else app root.
        window = _attr(app_ref, "AXFocusedWindow") or _attr(app_ref, "AXMainWindow")
        if window is None:
            windows = _attr(app_ref, "AXWindows") or []
            for w in windows:
                _walk(w, 0)
        else:
            _walk(window, 0)

        # Drop exact duplicates (AX often double-counts wrappers and their labels).
        seen: set[tuple] = set()
        unique: list[AXElement] = []
        for e in elements:
            key = (e.role, e.label, e.x, e.y, e.w, e.h)
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)
        return proc_name, unique
