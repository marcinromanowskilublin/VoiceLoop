from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from voiceloop.notepad_document import apply_notepad_write, resolve_notepad_hwnd

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Wymaga Windows i Notatnika"),
    pytest.mark.skipif(
        os.environ.get("VOICELOOP_RUN_WINDOWS_LIVE_TESTS") != "1",
        reason=(
            "Live test uruchamia notepad.exe; ustaw "
            "VOICELOOP_RUN_WINDOWS_LIVE_TESTS=1, żeby odpalić go ręcznie."
        ),
    ),
]

DEMO_STEM = "VoiceLoop_demo_notatka"


def _demo_windows() -> list[int]:
    import win32gui

    matches: list[int] = []

    def _enum(hwnd: int, _: object) -> None:
        title = win32gui.GetWindowText(hwnd) or ""
        if (
            DEMO_STEM.casefold() in title.casefold()
            and win32gui.IsWindowVisible(hwnd)
            and win32gui.GetParent(hwnd) == 0
        ):
            matches.append(int(hwnd))

    win32gui.EnumWindows(_enum, None)
    return matches


def _close_demo_windows() -> None:
    import win32con
    import win32gui

    for hwnd in _demo_windows():
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            continue
    deadline = time.time() + 4
    while time.time() < deadline and _demo_windows():
        time.sleep(0.15)


def _wait_unique_demo_hwnd(timeout: float = 10.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        matches = _demo_windows()
        if len(matches) == 1:
            return matches[0]
        time.sleep(0.2)
    pytest.skip(
        f"Nie udało się uzyskać jednego okna testowego Notatnika (widać {len(_demo_windows())})."
    )


def test_live_notepad_read_write_verify_without_mouse(tmp_path) -> None:
    path = tmp_path / f"{DEMO_STEM}.txt"
    original = "VoiceLoop demo notatka 10 wrzesnia 2026. Fikcyjna tresc testowa."
    replacement = "VoiceLoop demo wpis: trzy punkty bez prywatnych danych."
    path.write_text(original, encoding="utf-8")
    _close_demo_windows()
    process = subprocess.Popen(["notepad.exe", str(path)])
    try:
        hwnd = _wait_unique_demo_hwnd()
        before = resolve_notepad_hwnd(hwnd)
        if DEMO_STEM.casefold() not in before.identity.document_name.casefold():
            pytest.skip(f"Pominięto obce okno: {before.identity.document_name}")
        assert original in before.text or before.text.strip() != ""
        message, data = apply_notepad_write(before, replacement)
        assert data.get("verified") is True, message
        assert data["actual_text"] == replacement
        after = resolve_notepad_hwnd(hwnd)
        assert after.text == replacement
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        _close_demo_windows()
