from __future__ import annotations

from pathlib import Path

from pywinauto import Desktop

OUT = Path(__file__).resolve().parents[1] / "logs" / "voiceattack-ui-now.txt"
lines: list[str] = []
desktop = Desktop(backend="uia")
for window in desktop.windows():
    title = window.window_text() or ""
    if "voiceattack" not in title.casefold() and window.element_info.control_type != "Menu":
        continue
    lines.append(f"WINDOW|{title!r}|{window.handle}|{window.element_info.class_name}")
    for control in window.descendants():
        info = control.element_info
        name = (info.name or "").replace("\r", " ").replace("\n", " ")
        if not name and not info.automation_id:
            continue
        lines.append(f"{info.control_type}|{name}|{info.automation_id}|{info.class_name}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
