from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

NOTEPAD_PROCESS_NAMES = frozenset({"notepad.exe"})
EDITOR_CONTROL_TYPES = frozenset({"Edit", "Document"})
SEARCH_MARKERS = (
    "search",
    "szukaj",
    "find",
    "znajdz",
    "znajdź",
    "omnibox",
)
MAX_NOTEPAD_CHARS = 20000


@dataclass(frozen=True)
class NotepadIdentity:
    process_id: int
    process_name: str
    hwnd: int
    window_title: str
    window_class: str
    control_type: str
    control_class: str
    automation_id: str
    runtime_id: str
    document_name: str

    def as_args(self) -> dict[str, Any]:
        return {
            "expected_process_id": self.process_id,
            "expected_hwnd": self.hwnd,
            "expected_process_name": self.process_name,
            "expected_window_class": self.window_class,
            "expected_window_title": self.window_title,
            "expected_automation_id": self.automation_id,
            "expected_runtime_id": self.runtime_id,
            "expected_document_name": self.document_name,
        }

    def mismatches(self, other: NotepadIdentity) -> list[str]:
        checks = (
            ("process_id", self.process_id, other.process_id),
            ("hwnd", self.hwnd, other.hwnd),
            ("process_name", self.process_name.casefold(), other.process_name.casefold()),
            ("window_class", self.window_class.casefold(), other.window_class.casefold()),
            ("automation_id", self.automation_id, other.automation_id),
            ("runtime_id", self.runtime_id, other.runtime_id),
            ("document_name", self.document_name, other.document_name),
        )
        return [name for name, left, right in checks if left != right]


@dataclass(frozen=True)
class NotepadSnapshot:
    identity: NotepadIdentity
    text: str
    control_count: int

    def public_data(self) -> dict[str, Any]:
        return {
            **self.identity.as_args(),
            "window_title": self.identity.window_title,
            "process_name": self.identity.process_name,
            "process_id": self.identity.process_id,
            "hwnd": self.identity.hwnd,
            "document_name": self.identity.document_name,
            "characters": len(self.text),
            "control_count": self.control_count,
            "target_source": "active_notepad",
        }


def is_notepad_process(process_name: str) -> bool:
    return (process_name or "").strip().casefold() in NOTEPAD_PROCESS_NAMES


def identity_from_args(args: dict[str, Any]) -> NotepadIdentity | None:
    process_id = int(args.get("expected_process_id") or 0)
    hwnd = int(args.get("expected_hwnd") or 0)
    process_name = str(args.get("expected_process_name") or "").strip()
    window_class = str(args.get("expected_window_class") or "").strip()
    if process_id <= 0 or hwnd <= 0 or not process_name or not window_class:
        return None
    return NotepadIdentity(
        process_id=process_id,
        process_name=process_name,
        hwnd=hwnd,
        window_title=str(args.get("expected_window_title") or "").strip(),
        window_class=window_class,
        control_type="",
        control_class="",
        automation_id=str(args.get("expected_automation_id") or ""),
        runtime_id=str(args.get("expected_runtime_id") or ""),
        document_name=str(args.get("expected_document_name") or "").strip(),
    )


def resolve_active_notepad() -> NotepadSnapshot:
    hwnd, window_title, window_class, process_id, process_name = _foreground_window()
    return _snapshot_for_window(hwnd, window_title, window_class, process_id, process_name)


def resolve_notepad_hwnd(hwnd: int) -> NotepadSnapshot:
    window_title, window_class, process_id, process_name = _window_identity(int(hwnd))
    return _snapshot_for_window(int(hwnd), window_title, window_class, process_id, process_name)


def _snapshot_for_window(
    hwnd: int,
    window_title: str,
    window_class: str,
    process_id: int,
    process_name: str,
) -> NotepadSnapshot:
    if not is_notepad_process(process_name):
        raise RuntimeError(
            "Aktywne okno nie jest Notatnikiem. Otwórz jedną notatkę i powtórz polecenie."
        )
    wrapper, control_count = _unique_editor(hwnd)
    text = _read_wrapper_text(wrapper)
    identity = NotepadIdentity(
        process_id=process_id,
        process_name=process_name,
        hwnd=hwnd,
        window_title=window_title,
        window_class=window_class,
        control_type=_control_type(wrapper),
        control_class=_class_name(wrapper),
        automation_id=_automation_id(wrapper),
        runtime_id=_runtime_id(wrapper),
        document_name=_document_name(window_title),
    )
    return NotepadSnapshot(identity=identity, text=text, control_count=control_count)


def read_active_notepad(expected: NotepadIdentity | None = None) -> NotepadSnapshot:
    snapshot = resolve_active_notepad()
    if expected is not None:
        _assert_same_document(expected, snapshot.identity, snapshot.text, None)
    return snapshot


def write_active_notepad(
    text: str,
    *,
    expected: NotepadIdentity | None = None,
    expected_source_text: str | None = None,
) -> tuple[str, dict[str, Any]]:
    if not str(text):
        raise ValueError("Tekst do wpisania jest pusty.")
    if len(text) > MAX_NOTEPAD_CHARS:
        raise ValueError("Tekst do wpisania jest zbyt długi.")

    before = resolve_active_notepad()
    if expected is not None or expected_source_text is not None:
        _assert_same_document(expected, before.identity, before.text, expected_source_text)
    return apply_notepad_write(before, text)


def apply_notepad_write(before: NotepadSnapshot, text: str) -> tuple[str, dict[str, Any]]:
    _set_wrapper_text(before, text)
    after = resolve_notepad_hwnd(before.identity.hwnd)
    mismatches = before.identity.mismatches(after.identity)
    verified = after.text == text and not mismatches
    data = {
        **after.public_data(),
        "expected_text": text,
        "actual_text": after.text,
        "verified": verified,
        "verification": "matched" if verified else "mismatch",
        "identity_mismatches": mismatches,
        "source_text_unchanged_before_write": True,
    }
    if not verified:
        return (
            "Zapis w Notatniku nie potwierdził się odczytem. Sprawdź treść ręcznie. "
            "Nie powtarzam zapisu.",
            data,
        )
    name = after.identity.document_name or after.identity.window_title or "Notatnik"
    return (
        f"Zastąpiłem treść w Notatniku „{name}” i sprawdziłem odczyt.",
        data,
    )


def _assert_same_document(
    expected: NotepadIdentity | None,
    current: NotepadIdentity,
    current_text: str,
    expected_source_text: str | None,
) -> None:
    if expected is not None:
        mismatches = expected.mismatches(current)
        if mismatches:
            raise RuntimeError(
                "Cel Notatnika zmienił się po przygotowaniu operacji. "
                f"Zmienione pola: {', '.join(mismatches)}."
            )
    if expected_source_text is not None and current_text != expected_source_text:
        raise RuntimeError(
            "Treść notatki zmieniła się po przygotowaniu operacji. "
            "Oczekująca zgoda nie obowiązuje."
        )


def _foreground_window() -> tuple[int, str, str, int, str]:
    try:
        import win32gui
    except ImportError as exc:
        raise RuntimeError(f"Brak zależności Windows do odczytu Notatnika: {exc}") from exc

    hwnd = int(win32gui.GetForegroundWindow() or 0)
    if hwnd <= 0:
        raise RuntimeError("Nie znaleziono aktywnego okna.")
    window_title, window_class, process_id, process_name = _window_identity(hwnd)
    return hwnd, window_title, window_class, process_id, process_name


def _window_identity(hwnd: int) -> tuple[str, str, int, str]:
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
    except ImportError as exc:
        raise RuntimeError(f"Brak zależności Windows do odczytu Notatnika: {exc}") from exc

    if not win32gui.IsWindow(hwnd):
        raise RuntimeError("Wskazane okno Notatnika nie jest już dostępne.")
    window_title = (win32gui.GetWindowText(hwnd) or "").strip()
    window_class = (win32gui.GetClassName(hwnd) or "").strip()
    _, process_id = win32process.GetWindowThreadProcessId(hwnd)
    process_id = int(process_id or 0)
    if process_id <= 0:
        raise RuntimeError("Nie mogę potwierdzić procesu aktywnego okna.")
    handle = win32api.OpenProcess(
        win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
        False,
        process_id,
    )
    try:
        process_name = os.path.basename(win32process.GetModuleFileNameEx(handle, 0))
    finally:
        win32api.CloseHandle(handle)
    if not process_name:
        raise RuntimeError("Nie mogę potwierdzić nazwy procesu aktywnego okna.")
    return window_title, window_class, process_id, process_name


def _unique_editor(hwnd: int) -> tuple[Any, int]:
    try:
        from pywinauto import Desktop
    except ImportError as exc:
        raise RuntimeError(f"Brak UI Automation do odczytu Notatnika: {exc}") from exc

    window = Desktop(backend="uia").window(handle=hwnd)
    editors: list[Any] = []
    for descendant in window.descendants():
        if not _is_visible_editor(descendant):
            continue
        editors.append(descendant)
    if not editors:
        raise RuntimeError("Nie znalazłem edytora w aktywnym Notatniku.")
    if len(editors) > 1:
        raise RuntimeError(
            "W Notatniku jest kilka kontrolek edytora. "
            "Zostaw jedną kartę i jedną notatkę, potem powtórz polecenie."
        )
    return editors[0], len(editors)


def _is_visible_editor(wrapper: Any) -> bool:
    info = getattr(wrapper, "element_info", None)
    control_type = str(getattr(info, "control_type", "") or "").strip()
    class_name = str(getattr(info, "class_name", "") or "").strip()
    name = str(getattr(info, "name", "") or "").strip()
    if control_type not in EDITOR_CONTROL_TYPES and not class_name.casefold().startswith("edit"):
        return False
    visible = getattr(info, "visible", True)
    if visible is False:
        return False
    combined = " ".join((name, str(getattr(info, "automation_id", "") or ""))).casefold()
    if any(marker in combined for marker in SEARCH_MARKERS):
        return False
    return True


def _read_wrapper_text(wrapper: Any) -> str:
    readers = (
        lambda: wrapper.get_value(),
        lambda: wrapper.iface_value.CurrentValue,
        lambda: _text_pattern_document(wrapper),
        lambda: wrapper.window_text(),
    )
    last_error: Exception | None = None
    for reader in readers:
        try:
            value = reader()
        except Exception as exc:  # noqa: BLE001 - UIA backends vary
            last_error = exc
            continue
        if value is None:
            continue
        text = str(value)
        if len(text) > MAX_NOTEPAD_CHARS:
            raise RuntimeError("Notatka jest zbyt długa do bezpiecznego odczytu.")
        return text
    raise RuntimeError(
        "Nie udało się odczytać tekstu aktywnego Notatnika."
        + (f" ({last_error})" if last_error else "")
    )


def _set_wrapper_text(snapshot: NotepadSnapshot, text: str) -> None:
    wrapper, _ = _unique_editor(snapshot.identity.hwnd)
    writers = (
        lambda: wrapper.set_edit_text(text),
        lambda: wrapper.iface_value.SetValue(text),
        lambda: _set_text_via_wm(snapshot.identity.hwnd, wrapper, text),
    )
    last_error: Exception | None = None
    for writer in writers:
        try:
            writer()
            return
        except Exception as exc:  # noqa: BLE001 - UIA backends vary
            last_error = exc
    raise RuntimeError(
        "Nie udało się programowo zapisać tekstu w Notatniku bez myszy."
        + (f" ({last_error})" if last_error else "")
    )


def _set_text_via_wm(root_hwnd: int, wrapper: Any, text: str) -> None:
    import win32con
    import win32gui

    handle = int(getattr(getattr(wrapper, "element_info", None), "handle", 0) or 0)
    if handle <= 0:
        handle = int(root_hwnd)
    if not win32gui.IsWindow(handle):
        raise RuntimeError("Kontrolka Notatnika nie jest już dostępna.")
    result = win32gui.SendMessage(handle, win32con.WM_SETTEXT, 0, text)
    if result == 0 and text:
        raise RuntimeError("WM_SETTEXT nie przyjął nowej treści.")


def _text_pattern_document(wrapper: Any) -> str:
    from pywinauto.uia_defines import IUIA

    element = getattr(getattr(wrapper, "element_info", None), "element", None)
    if element is None:
        raise RuntimeError("Brak elementu UIA.")
    uia = IUIA()
    client = uia.ui_automation_client
    unknown = element.GetCurrentPattern(client.UIA_TextPatternId)
    if not unknown:
        raise RuntimeError("Brak TextPattern.")
    pattern = unknown.QueryInterface(client.IUIAutomationTextPattern)
    document = pattern.DocumentRange
    if document is None:
        raise RuntimeError("Brak zakresu dokumentu.")
    return str(document.GetText(MAX_NOTEPAD_CHARS + 1) or "")


def focused_selection_text() -> str | None:
    try:
        from pywinauto.uia_defines import IUIA
    except ImportError:
        return None
    try:
        uia = IUIA()
        client = uia.ui_automation_client
        focused = uia.iuia.GetFocusedElement()
        if focused is None:
            return None
        unknown = focused.GetCurrentPattern(client.UIA_TextPatternId)
        if not unknown:
            return None
        pattern = unknown.QueryInterface(client.IUIAutomationTextPattern)
        selection = pattern.GetSelection()
        if not selection:
            return None
        texts: list[str] = []
        length = int(getattr(selection, "Length", 0) or 0)
        if length <= 0:
            try:
                texts.append(str(selection[0].GetText(MAX_NOTEPAD_CHARS) or ""))
            except Exception:
                return None
        else:
            for index in range(min(length, 4)):
                try:
                    texts.append(str(selection.GetElement(index).GetText(MAX_NOTEPAD_CHARS) or ""))
                except Exception:
                    continue
        combined = "".join(texts).strip()
        return combined or None
    except Exception:
        return None


def focused_text_target() -> dict[str, Any] | None:
    try:
        from pywinauto.controls.uiawrapper import UIAWrapper
        from pywinauto.uia_defines import IUIA
        from pywinauto.uia_element_info import UIAElementInfo
    except ImportError:
        return None
    try:
        focused = IUIA().iuia.GetFocusedElement()
        if focused is None:
            return None
        wrapper = UIAWrapper(UIAElementInfo(focused))
        info = wrapper.element_info
        control_type = str(getattr(info, "control_type", "") or "").strip()
        class_name = str(getattr(info, "class_name", "") or "").strip()
        is_editable = control_type in EDITOR_CONTROL_TYPES or class_name.casefold().startswith(
            "edit"
        )
        if not is_editable:
            return None
        return {
            "field_name": str(getattr(info, "name", "") or "").strip(),
            "control_type": control_type,
            "automation_id": str(getattr(info, "automation_id", "") or "").strip(),
            "class_name": class_name,
            "is_editable": True,
            "target_source": "focused_control",
        }
    except Exception:
        return None


def _document_name(window_title: str) -> str:
    title = (window_title or "").strip()
    for separator in (" — ", " – ", " - "):
        if separator in title:
            return title.split(separator, 1)[0].strip() or title
    return title


def _control_type(wrapper: Any) -> str:
    return str(getattr(getattr(wrapper, "element_info", None), "control_type", "") or "")


def _class_name(wrapper: Any) -> str:
    return str(getattr(getattr(wrapper, "element_info", None), "class_name", "") or "")


def _automation_id(wrapper: Any) -> str:
    return str(getattr(getattr(wrapper, "element_info", None), "automation_id", "") or "")


def _runtime_id(wrapper: Any) -> str:
    runtime = getattr(getattr(wrapper, "element_info", None), "runtime_id", None)
    if runtime is None:
        return ""
    if isinstance(runtime, (list, tuple)):
        return ",".join(str(part) for part in runtime)
    return str(runtime)
