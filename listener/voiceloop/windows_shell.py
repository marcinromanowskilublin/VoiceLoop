from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from typing import Any

SHELL_CLASSES = {"Progman", "WorkerW", "CabinetWClass", "ExploreWClass"}
EXPLORER_CLASSES = {"CabinetWClass", "ExploreWClass"}
PHONETIC_ALIASES = {
    "a lej aut": "a way out",
    "a lej ałt": "a way out",
    "a łej aut": "a way out",
    "a łej ałt": "a way out",
    "mortal szel": "mortal shell",
}

# "Aktywny folder" oznacza wyłącznie okno Eksploratora, w którym aktualnie
# znajduje się kursor myszy. Ten moduł nigdy nie posiłkuje się pulpitem ani
# oknem będącym na pierwszym planie jako niejawnym fallbackiem.
ACTIVE_FOLDER_FUZZY_MIN_SCORE = 0.35
ACTIVE_FOLDER_FUZZY_CONFIDENT_SCORE = 0.88
ACTIVE_FOLDER_FUZZY_CONFIDENT_MARGIN = 0.10
ACTIVE_FOLDER_CANDIDATE_MIN_SCORE = 0.45
ACTIVE_FOLDER_MAX_CANDIDATES = 3

WINDOW_LAYOUTS = (
    "left_half",
    "right_half",
    "left_third",
    "center_third",
    "right_third",
    "top_left_quarter",
    "bottom_left_quarter",
    "top_right_quarter",
    "bottom_right_quarter",
)


@dataclass(frozen=True)
class ShellItem:
    name: str
    rectangle: tuple[int, int, int, int]
    control_type: str
    root_class: str
    root_handle: int
    runtime_id: tuple[int, ...]
    automation_id: str = ""
    # Best-effort metadata for the active-folder palette. ``is_folder`` stays
    # ``None`` when it could not be determined (no automatic desktop/file
    # guessing is ever performed from this alone).
    is_folder: bool | None = None
    extension: str = ""

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control_type": self.control_type,
            "root_class": self.root_class,
            "root_handle": self.root_handle,
            "runtime_id": list(self.runtime_id),
            "automation_id": self.automation_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def _phonetic(value: str) -> str:
    normalized = _normalized(value)
    return PHONETIC_ALIASES.get(normalized, normalized)


def _extension_of(name: str) -> str:
    dot = name.rfind(".")
    if dot <= 0 or dot == len(name) - 1:
        return ""
    return name[dot + 1 :].strip().casefold()


def _first_letter(name: str) -> str:
    for character in name:
        if character.isalnum():
            return character.casefold()
    return ""


def match_shell_item(query: str, candidates: list[ShellItem]) -> ShellItem:
    wanted = _normalized(query)
    exact = [item for item in candidates if _normalized(item.name) == wanted]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError("Znalazłem kilka równorzędnych elementów. Nie wykonuję ruchu.")

    spoken = _phonetic(query)
    scored = sorted(
        (
            (
                SequenceMatcher(None, spoken, _normalized(item.name)).ratio(),
                item,
            )
            for item in candidates
        ),
        key=lambda pair: pair[0],
    )
    if not scored or scored[-1][0] < 0.88:
        raise RuntimeError("Nie znalazłem jednoznacznego widocznego elementu.")
    best_score, best = scored[-1]
    second_score = scored[-2][0] if len(scored) > 1 else 0.0
    if best_score - second_score < 0.10:
        raise RuntimeError("Znalazłem kilka podobnych elementów. Nie wykonuję ruchu.")
    return best


class WindowsShellLocator:
    def enumerate_items(self) -> list[ShellItem]:
        try:
            import win32gui
            from pywinauto import Desktop
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows UI Automation: {exc}") from exc

        foreground = int(win32gui.GetForegroundWindow() or 0)
        items: list[ShellItem] = []
        for root in Desktop(backend="uia").windows():
            handle = int(getattr(root, "handle", 0) or 0)
            class_name = str(getattr(root.element_info, "class_name", "") or "")
            if class_name not in SHELL_CLASSES:
                continue
            if class_name in EXPLORER_CLASSES and handle != foreground:
                continue
            try:
                if not root.is_visible():
                    continue
                descendants = root.descendants()
            except Exception:
                continue
            for wrapper in descendants:
                try:
                    if not wrapper.is_visible():
                        continue
                    info = wrapper.element_info
                    name = str(getattr(info, "name", "") or "").strip()
                    control_type = str(getattr(info, "control_type", "") or "")
                    rectangle = wrapper.rectangle()
                    coords = (
                        int(rectangle.left),
                        int(rectangle.top),
                        int(rectangle.right),
                        int(rectangle.bottom),
                    )
                    runtime_id = tuple(int(value) for value in (info.runtime_id or ()))
                    if (
                        not name
                        or control_type not in {"ListItem", "TreeItem", "DataItem"}
                        or coords[2] <= coords[0]
                        or coords[3] <= coords[1]
                        or not runtime_id
                    ):
                        continue
                    items.append(
                        ShellItem(
                            name=name,
                            rectangle=coords,
                            control_type=control_type,
                            root_class=class_name,
                            root_handle=handle,
                            runtime_id=runtime_id,
                            automation_id=str(getattr(info, "automation_id", "") or ""),
                        )
                    )
                except Exception:
                    continue
        return items

    def resolve(self, query: str) -> ShellItem:
        return match_shell_item(query, self.enumerate_items())

    def validate(self, expected: dict[str, Any]) -> tuple[ShellItem, Any]:
        candidates = self.enumerate_items()
        runtime_id = tuple(int(value) for value in expected.get("runtime_id", ()))
        matches = [
            item
            for item in candidates
            if item.runtime_id == runtime_id
            and item.root_handle == int(expected.get("root_handle", 0))
            and item.root_class == str(expected.get("root_class", ""))
            and item.control_type == str(expected.get("control_type", ""))
            and _normalized(item.name) == _normalized(str(expected.get("name", "")))
        ]
        if len(matches) != 1:
            raise RuntimeError("Element zmienił się lub zniknął po potwierdzeniu.")
        item = matches[0]
        return item, self._wrapper_for(item)

    @staticmethod
    def _wrapper_for(item: ShellItem) -> Any:
        from pywinauto import Desktop

        root = Desktop(backend="uia").window(handle=item.root_handle)
        matches = [
            wrapper
            for wrapper in root.descendants()
            if tuple(int(value) for value in (wrapper.element_info.runtime_id or ()))
            == item.runtime_id
        ]
        if len(matches) != 1:
            raise RuntimeError("Nie można ponownie powiązać elementu UIA.")
        return matches[0]

    def enumerate_active_folder_items(self, handle: int) -> list[ShellItem]:
        """Lists only the items of the Explorer window ``handle`` points to.

        This never falls back to the desktop or to any other window: the
        caller must first resolve ``handle`` via
        :func:`locate_active_folder_handle`, which already enforces that the
        window under the mouse cursor is a real Explorer window.
        """

        try:
            import win32gui
            from pywinauto import Desktop
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows UI Automation: {exc}") from exc

        class_name = str(win32gui.GetClassName(handle) or "")
        if class_name not in EXPLORER_CLASSES:
            raise RuntimeError("Wskazane okno nie jest Eksploratorem plików.")

        root = Desktop(backend="uia").window(handle=handle)
        try:
            if not root.is_visible():
                raise RuntimeError("Wskazane okno Eksploratora nie jest już widoczne.")
            descendants = root.descendants()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Nie mogę odczytać zawartości aktywnego folderu.") from exc

        items: list[ShellItem] = []
        for wrapper in descendants:
            try:
                if not wrapper.is_visible():
                    continue
                info = wrapper.element_info
                name = str(getattr(info, "name", "") or "").strip()
                control_type = str(getattr(info, "control_type", "") or "")
                rectangle = wrapper.rectangle()
                coords = (
                    int(rectangle.left),
                    int(rectangle.top),
                    int(rectangle.right),
                    int(rectangle.bottom),
                )
                runtime_id = tuple(int(value) for value in (info.runtime_id or ()))
                if (
                    not name
                    or control_type not in {"ListItem", "TreeItem", "DataItem"}
                    or coords[2] <= coords[0]
                    or coords[3] <= coords[1]
                    or not runtime_id
                ):
                    continue
                items.append(
                    ShellItem(
                        name=name,
                        rectangle=coords,
                        control_type=control_type,
                        root_class=class_name,
                        root_handle=handle,
                        runtime_id=runtime_id,
                        automation_id=str(getattr(info, "automation_id", "") or ""),
                        extension=_extension_of(name),
                    )
                )
            except Exception:
                continue
        return _attach_folder_metadata(handle, items)


def locate_active_folder_handle() -> int:
    """Finds the Explorer window currently under the mouse cursor.

    Raises ``RuntimeError`` (in Polish, matching the rest of this module) when
    there is no window under the cursor or it is not an Explorer window. The
    desktop and any other window are never picked automatically.
    """

    try:
        import win32api
        import win32con
        import win32gui
    except ImportError as exc:
        raise RuntimeError(
            f"Brak zależności Windows do wykrycia aktywnego folderu: {exc}"
        ) from exc

    cursor = win32api.GetCursorPos()
    child_hwnd = win32gui.WindowFromPoint(cursor)
    if not child_hwnd:
        raise RuntimeError("Nie znalazłem okna pod kursorem.")
    hwnd = int(win32gui.GetAncestor(child_hwnd, getattr(win32con, "GA_ROOT", 2)) or child_hwnd)
    class_name = str(win32gui.GetClassName(hwnd) or "")
    if class_name not in EXPLORER_CLASSES:
        raise RuntimeError(
            "Pod kursorem nie ma okna Eksploratora. Wskaż kursorem aktywny folder "
            "i powtórz polecenie."
        )
    return hwnd


def revalidate_active_folder_item(handle: int, expected: ShellItem) -> ShellItem:
    """Re-reads the active folder and confirms ``expected`` is still there."""

    items = WindowsShellLocator().enumerate_active_folder_items(handle)
    matches = [
        item
        for item in items
        if item.runtime_id == expected.runtime_id
        and item.root_handle == expected.root_handle
        and item.root_class == expected.root_class
        and item.control_type == expected.control_type
        and _normalized(item.name) == _normalized(expected.name)
    ]
    if len(matches) != 1:
        raise RuntimeError("Element zmienił się lub zniknął. Powtórz polecenie.")
    return matches[0]


def _attach_folder_metadata(handle: int, items: list[ShellItem]) -> list[ShellItem]:
    """Best-effort ``is_folder``/``extension`` lookup via Shell.Application.

    UIA alone does not reliably expose file-vs-folder information for shell
    list items, so this uses the read-only Shell COM namespace to look up
    metadata for the same window, matched by display name. Failures degrade
    gracefully: items are returned unchanged (``is_folder`` stays ``None``).
    """

    try:
        import win32com.client
    except ImportError:
        return items

    try:
        shell = win32com.client.Dispatch("Shell.Application")
        folder = None
        for window in shell.Windows():
            try:
                if int(window.HWND) == int(handle):
                    folder = window.Document.Folder
                    break
            except Exception:
                continue
        if folder is None:
            return items
        metadata: dict[str, tuple[bool, str]] = {}
        for shell_item in folder.Items():
            try:
                name = str(shell_item.Name or "")
                if not name:
                    continue
                is_folder = bool(shell_item.IsFolder)
                path = str(shell_item.Path or "")
                file_name = path.rsplit("\\", 1)[-1] if path else name
                extension = "" if is_folder else _extension_of(file_name)
                metadata[_normalized(name)] = (is_folder, extension)
            except Exception:
                continue
    except Exception:
        return items

    if not metadata:
        return items

    updated: list[ShellItem] = []
    for entry in items:
        info = metadata.get(_normalized(entry.name))
        if info is None:
            updated.append(entry)
            continue
        is_folder, extension = info
        updated.append(
            replace(entry, is_folder=is_folder, extension=extension or entry.extension)
        )
    return updated


def match_active_folder_item(
    query: str,
    candidates: list[ShellItem],
    *,
    kind: str | None = None,
) -> tuple[ShellItem | None, list[ShellItem]]:
    """Resolves ``query`` against items already verified in the active folder.

    Returns ``(item, [])`` for a confident single match, or ``(None,
    candidates)`` with 2-3 best candidates when the name is ambiguous. Raises
    ``RuntimeError`` only when nothing plausible was found at all. Never opens
    or selects anything by itself.
    """

    pool = candidates
    if kind == "folder":
        pool = [item for item in pool if item.is_folder is True]
    elif kind == "file":
        pool = [item for item in pool if item.is_folder is False]
    if kind in {"folder", "file"} and not pool:
        raise RuntimeError(
            "Nie mogę rozróżnić plików i folderów w tym oknie Eksploratora."
        )

    wanted = _normalized(query)
    exact = [item for item in pool if _normalized(item.name) == wanted]
    if len(exact) == 1:
        return exact[0], []
    if len(exact) > 1:
        return None, exact[:ACTIVE_FOLDER_MAX_CANDIDATES]

    spoken = _phonetic(query)
    scored = sorted(
        ((SequenceMatcher(None, spoken, _normalized(item.name)).ratio(), item) for item in pool),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not scored or scored[0][0] < ACTIVE_FOLDER_FUZZY_MIN_SCORE:
        raise RuntimeError("Nie znalazłem elementu o tej nazwie w aktywnym folderze.")

    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if (
        best_score >= ACTIVE_FOLDER_FUZZY_CONFIDENT_SCORE
        and best_score - second_score >= ACTIVE_FOLDER_FUZZY_CONFIDENT_MARGIN
    ):
        return best, []

    plausible = [
        item
        for score, item in scored[:ACTIVE_FOLDER_MAX_CANDIDATES]
        if score >= ACTIVE_FOLDER_CANDIDATE_MIN_SCORE
    ]
    return None, plausible or [best]


def filter_by_extension(items: list[ShellItem], extension: str) -> list[ShellItem]:
    wanted = extension.strip().lstrip(".").casefold()
    if not wanted:
        return []
    return [item for item in items if item.extension and item.extension == wanted]


def filter_by_first_letter(items: list[ShellItem], letter: str) -> list[ShellItem]:
    wanted = _first_letter(letter)
    if not wanted:
        return []
    return [item for item in items if _first_letter(item.name) == wanted]


def select_shell_items(handle: int, targets: list[ShellItem]) -> list[ShellItem]:
    """Selects ``targets`` in the Explorer window ``handle`` via UI Automation.

    Uses only the UIA ``SelectionItem`` pattern (``Select`` for the first item,
    ``AddToSelection`` for the rest). Never drags, never sends blind clicks,
    and re-resolves the live UIA element for each target by ``runtime_id``
    before selecting it.
    """

    if not targets:
        raise RuntimeError("Nie ma elementów do zaznaczenia.")
    try:
        from pywinauto import Desktop
    except ImportError as exc:
        raise RuntimeError(f"Brak Windows UI Automation: {exc}") from exc

    root = Desktop(backend="uia").window(handle=handle)
    try:
        descendants = root.descendants()
    except Exception as exc:
        raise RuntimeError("Nie mogę ponownie odczytać zawartości aktywnego folderu.") from exc

    by_runtime_id = {}
    for wrapper in descendants:
        try:
            runtime_id = tuple(int(value) for value in (wrapper.element_info.runtime_id or ()))
        except Exception:
            continue
        if runtime_id:
            by_runtime_id[runtime_id] = wrapper

    selected: list[ShellItem] = []
    for index, target in enumerate(targets):
        wrapper = by_runtime_id.get(target.runtime_id)
        if wrapper is None:
            raise RuntimeError(
                f"Element „{target.name}” zmienił się lub zniknął przed zaznaczeniem."
            )
        try:
            pattern = wrapper.iface_selection_item
        except Exception as exc:
            raise RuntimeError(
                "Element nie udostępnia zaznaczania przez UI Automation."
            ) from exc
        try:
            if index == 0:
                pattern.Select()
            else:
                pattern.AddToSelection()
        except Exception as exc:
            raise RuntimeError(
                f"Nie udało się zaznaczyć „{target.name}” przez UI Automation."
            ) from exc
        selected.append(target)
    return selected


def get_cursor_position() -> tuple[int, int]:
    try:
        import win32api
    except ImportError as exc:
        raise RuntimeError(f"Brak Win32 do odczytu pozycji kursora: {exc}") from exc
    x, y = win32api.GetCursorPos()
    return int(x), int(y)


def set_cursor_position(position: tuple[int, int]) -> None:
    try:
        import win32api
    except ImportError as exc:
        raise RuntimeError(f"Brak Win32 do ruchu kursora: {exc}") from exc
    win32api.SetCursorPos((int(position[0]), int(position[1])))


def monitor_work_area_at(position: tuple[int, int]) -> tuple[int, int, int, int]:
    try:
        import win32api
        import win32con
    except ImportError as exc:
        raise RuntimeError(f"Brak Win32 do odczytu monitora: {exc}") from exc
    monitor = win32api.MonitorFromPoint(
        (int(position[0]), int(position[1])),
        getattr(win32con, "MONITOR_DEFAULTTONEAREST", 2),
    )
    info = win32api.GetMonitorInfo(monitor)
    left, top, right, bottom = info["Work"]
    return int(left), int(top), int(right), int(bottom)


def monitor_work_area_for_window(hwnd: int) -> tuple[int, int, int, int]:
    try:
        import win32api
        import win32con
    except ImportError as exc:
        raise RuntimeError(f"Brak Win32 do odczytu monitora: {exc}") from exc
    monitor = win32api.MonitorFromWindow(hwnd, getattr(win32con, "MONITOR_DEFAULTTONEAREST", 2))
    info = win32api.GetMonitorInfo(monitor)
    left, top, right, bottom = info["Work"]
    return int(left), int(top), int(right), int(bottom)


def center_cursor() -> tuple[int, int]:
    current = get_cursor_position()
    left, top, right, bottom = monitor_work_area_at(current)
    center = ((left + right) // 2, (top + bottom) // 2)
    set_cursor_position(center)
    return center


def layout_rect(work_area: tuple[int, int, int, int], layout: str) -> tuple[int, int, int, int]:
    """Computes ``(x, y, width, height)`` for ``layout`` within ``work_area``.

    ``work_area`` must already be the work area (excluding the taskbar) of the
    monitor that hosts the window being moved. This never talks to the Windows
    11 Snap Layout menu; callers apply the rectangle directly.
    """

    left, top, right, bottom = work_area
    width = right - left
    height = bottom - top
    half_w = width // 2
    half_h = height // 2
    third_w = width // 3
    if layout == "left_half":
        return left, top, half_w, height
    if layout == "right_half":
        return left + half_w, top, width - half_w, height
    if layout == "left_third":
        return left, top, third_w, height
    if layout == "center_third":
        return left + third_w, top, third_w, height
    if layout == "right_third":
        return left + 2 * third_w, top, width - 2 * third_w, height
    if layout == "top_left_quarter":
        return left, top, half_w, half_h
    if layout == "bottom_left_quarter":
        return left, top + half_h, half_w, height - half_h
    if layout == "top_right_quarter":
        return left + half_w, top, width - half_w, half_h
    if layout == "bottom_right_quarter":
        return left + half_w, top + half_h, width - half_w, height - half_h
    raise ValueError(f"Nieznany układ okna: {layout}")


def hover_shell_item(expected: dict[str, Any]) -> ShellItem:
    item, _wrapper = WindowsShellLocator().validate(expected)
    left, top, right, bottom = item.rectangle
    try:
        import win32api
    except ImportError as exc:
        raise RuntimeError(f"Brak Win32 do ruchu kursora: {exc}") from exc
    win32api.SetCursorPos(((left + right) // 2, (top + bottom) // 2))
    return item


def open_shell_item(expected: dict[str, Any]) -> ShellItem:
    item, wrapper = WindowsShellLocator().validate(expected)
    try:
        wrapper.invoke()
        return item
    except Exception:
        try:
            wrapper.select()
            wrapper.set_focus()
            wrapper.type_keys("{ENTER}")
            return item
        except Exception as exc:
            raise RuntimeError("Element nie udostępnia bezpiecznego otwarcia przez UIA.") from exc
