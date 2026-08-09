from __future__ import annotations

import time
from pathlib import Path

from pywinauto import Desktop
from pywinauto.keyboard import send_keys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "voiceattack" / "VoiceLoop-profil.vap"
OUT = ROOT / "logs" / "voiceattack-import-attempt.txt"
lines: list[str] = []


def dump(prefix: str, win) -> None:
    lines.append(f"=== {prefix}: {win.window_text()!r} handle={win.handle} class={win.element_info.class_name} ===")
    try:
        for c in win.descendants():
            info = c.element_info
            name = (info.name or "").replace("\r", " ").replace("\n", " ")
            if not name and not info.automation_id:
                continue
            lines.append(
                f"{info.control_type}|{name}|{info.automation_id}|{info.class_name}"
            )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"DUMP_ERR|{exc}")


def main() -> int:
    desktop_uia = Desktop(backend="uia")
    desktop_win = Desktop(backend="win32")

    va = None
    for window in desktop_uia.windows():
        if (window.window_text() or "").strip() == "VoiceAttack":
            va = window
            break
    if va is None:
        OUT.write_text("NO_VA_WINDOW", encoding="utf-8")
        return 2

    va.set_focus()
    send_keys("{ESC}{ESC}")
    time.sleep(0.4)

    more = None
    for control in va.descendants():
        if control.element_info.automation_id == "btnMore":
            more = control
            break
    if more is None:
        OUT.write_text("NO_MORE", encoding="utf-8")
        return 3

    more.click_input()
    time.sleep(0.8)

    import_item = None
    for window in desktop_uia.windows():
        for control in window.descendants():
            info = control.element_info
            if info.control_type == "MenuItem" and (info.name or "").casefold() == "import profile":
                import_item = control
                break
        if import_item is not None:
            break

    if import_item is None:
        OUT.write_text("\n".join(lines + ["NO_IMPORT_MENU"]), encoding="utf-8")
        return 4

    lines.append("CLICK_IMPORT_MENU")
    import_item.click_input()
    time.sleep(2.0)

    dialog = None
    for _ in range(10):
        # Classic Open dialog
        for window in desktop_win.windows():
            title = (window.window_text() or "").casefold()
            cls = (window.element_info.class_name or "")
            lines.append(f"WIN32|{window.window_text()!r}|{window.handle}|{cls}")
            if cls == "#32770" and ("open" in title or "otwórz" in title or title == ""):
                dialog = window
                break
        if dialog is None:
            for window in desktop_uia.windows():
                title = (window.window_text() or "").casefold()
                cls = window.element_info.class_name or ""
                lines.append(f"UIA|{window.window_text()!r}|{window.handle}|{cls}")
                if "open" in title or "otwórz" in title or cls == "#32770":
                    dialog = window
                    break
        if dialog is not None:
            break
        time.sleep(0.5)

    if dialog is None:
        OUT.write_text("\n".join(lines + ["NO_IMPORT_DIALOG"]), encoding="utf-8")
        return 5

    dump("DIALOG", dialog)

    # Prefer typing into focused file dialog via keyboard
    try:
        dialog.set_focus()
    except Exception:
        pass
    time.sleep(0.2)
    send_keys("^l")  # focus address/filename area on modern dialogs
    time.sleep(0.2)
    send_keys("^a{BACKSPACE}" + str(PROFILE) + "{ENTER}", with_spaces=True)
    lines.append(f"PATH_ENTER|{PROFILE}")
    time.sleep(2.5)

    # Dump VA state
    for window in desktop_uia.windows():
        if (window.window_text() or "").strip() == "VoiceAttack":
            dump("VA_AFTER", window)
            for control in window.descendants():
                if control.element_info.automation_id == "cboProfile":
                    try:
                        control.expand()
                    except Exception:
                        control.click_input()
                    time.sleep(0.7)
                    dump("PROFILE_COMBO", window)
                    break
            break

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
