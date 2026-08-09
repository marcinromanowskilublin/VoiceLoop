from __future__ import annotations

import time
from pathlib import Path

from pywinauto import Desktop

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "voiceattack-register-attempt.txt"
lines: list[str] = []


def _load_registration_key() -> str:
    env_path = ROOT / "listener" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VOICEATTACK_REGISTRATION_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise SystemExit(
        "Brak VOICEATTACK_REGISTRATION_KEY w listener/.env — wpisz klucz lokalnie."
    )


KEY = _load_registration_key()


def dump(prefix: str, win) -> None:
    lines.append(f"=== {prefix}: {win.window_text()!r} handle={win.handle} ===")
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


def find_button(win, *needles: str):
    for control in win.descendants():
        info = control.element_info
        if info.control_type != "Button":
            continue
        blob = f"{info.name}|{info.automation_id}".casefold()
        if any(needle in blob for needle in needles):
            return control
    return None


def click_control(control) -> None:
    info = control.element_info
    lines.append(f"CLICK|{info.name}|{info.automation_id}")
    try:
        control.invoke()
    except Exception:
        control.click_input()


def main() -> int:
    desktop = Desktop(backend="uia")
    va = None
    for window in desktop.windows():
        title = (window.window_text() or "").strip()
        if title == "VoiceAttack":
            va = window
            break
    if va is None:
        OUT.write_text("NO_VA_WINDOW", encoding="utf-8")
        return 2

    va.set_focus()
    time.sleep(0.3)
    dump("MAIN_BEFORE", va)

    reg_btn = find_button(va, "btnregistration")
    if reg_btn is None:
        opt = find_button(va, "btnoptionsmall")
        if opt is None:
            OUT.write_text("\n".join(lines + ["NO_OPTIONS_BUTTON"]), encoding="utf-8")
            return 3
        click_control(opt)
        time.sleep(1.5)
        dump("MAIN_AFTER_OPTIONS", va)
        reg_btn = find_button(va, "btnregistration")

    if reg_btn is None:
        # Search all top-level windows for registration button
        for window in desktop.windows():
            reg_btn = find_button(window, "btnregistration")
            if reg_btn is not None:
                va = window
                break

    if reg_btn is None:
        OUT.write_text("\n".join(lines + ["NO_REGISTRATION_BUTTON"]), encoding="utf-8")
        return 5

    click_control(reg_btn)
    time.sleep(1.8)

    reg = None
    for window in desktop.windows():
        title = (window.window_text() or "").casefold()
        if "regist" in title or "license" in title:
            reg = window
            break
    if reg is None:
        # Owned/untitled dialogs often hold the key field
        for window in desktop.windows():
            if window.element_info.control_type != "Window":
                continue
            edits = [
                c
                for c in window.descendants()
                if c.element_info.control_type == "Edit"
            ]
            blob = "\n".join(
                f"{c.element_info.name}|{c.element_info.automation_id}"
                for c in window.descendants()
            ).casefold()
            if edits and ("registration" in blob or "key" in blob or "license" in blob):
                reg = window
                break

    if reg is None:
        OUT.write_text("\n".join(lines + ["NO_REGISTRATION_DIALOG"]), encoding="utf-8")
        return 6

    dump("REGDIALOG", reg)
    edits = [c for c in reg.descendants() if c.element_info.control_type == "Edit"]
    lines.append(f"EDITS|{len(edits)}")
    target = None
    for edit in edits:
        aid = (edit.element_info.automation_id or "").casefold()
        name = (edit.element_info.name or "").casefold()
        lines.append(f"EDIT|{name}|{aid}")
        if any(token in aid or token in name for token in ("key", "registration", "license")):
            target = edit
            break
    if target is None and edits:
        target = edits[0]
    if target is None:
        OUT.write_text("\n".join(lines + ["NO_KEY_FIELD"]), encoding="utf-8")
        return 7

    target.set_focus()
    try:
        target.type_keys("^a{BACKSPACE}" + KEY, with_spaces=True)
        lines.append("TYPE_KEYS_OK")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"TYPE_KEYS_ERR|{exc}")
        try:
            target.set_edit_text(KEY)
            lines.append("SET_TEXT_OK")
        except Exception as exc2:  # noqa: BLE001
            lines.append(f"SET_TEXT_ERR|{exc2}")
            OUT.write_text("\n".join(lines), encoding="utf-8")
            return 8

    confirm = find_button(reg, "btnok", "ok", "validate", "register", "submit", "apply")
    if confirm is not None:
        click_control(confirm)
        lines.append("CONFIRM_CLICKED|True")
    else:
        lines.append("CONFIRM_CLICKED|False")
    time.sleep(4.0)

    for window in desktop.windows():
        title = window.window_text() or ""
        low = title.casefold()
        if window.element_info.control_type != "Window":
            continue
        if any(
            token in low
            for token in (
                "voiceattack",
                "regist",
                "option",
                "success",
                "thank",
                "error",
                "invalid",
                "valid",
            )
        ) or title == "":
            dump(f"AFTER:{title}", window)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK lines={len(lines)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
