from __future__ import annotations

import time
from pathlib import Path

from pywinauto import Desktop


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "voiceattack-registration-menu.txt"


def main() -> None:
    desktop = Desktop(backend="uia")
    window = desktop.window(title="VoiceAttack")
    menu_button = window.child_window(
        auto_id="btnMenu", control_type="Button"
    ).wrapper_object()
    menu_button.invoke()
    time.sleep(0.7)

    lines: list[str] = []
    for candidate in desktop.windows():
        title = candidate.window_text()
        if "voiceattack" not in title.casefold() and candidate.element_info.control_type != "Menu":
            continue
        lines.append(
            f"WINDOW|{title}|{candidate.handle}|{candidate.element_info.control_type}"
        )
        for control in candidate.descendants():
            info = control.element_info
            name = (info.name or "").replace("\r", " ").replace("\n", " ")
            if name:
                lines.append(
                    f"{info.control_type}|{name}|{info.automation_id}|{info.class_name}"
                )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
