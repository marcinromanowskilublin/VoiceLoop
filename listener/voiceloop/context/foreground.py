from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from .schema import ContextEventV1, ForegroundConfidence


@dataclass(frozen=True)
class ForegroundIdentity:
    hwnd: int
    process_id: int
    process_name: str
    window_title: str
    window_class: str


class ForegroundSampler:
    """Explicit Win32 observation; no polling loop is started automatically."""

    def capture(self, *, observed_at: datetime | None = None) -> ContextEventV1:
        timestamp = observed_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        identity = _foreground_identity()
        raw_id = (
            f"{timestamp.isoformat()}:{identity.hwnd}:"
            f"{identity.process_id}:{identity.window_title}"
        )
        source_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return ContextEventV1.from_observation(
            source="win32_foreground",
            source_id=source_id,
            started_at=timestamp,
            event_type="foreground_window",
            app_name=identity.process_name,
            process_name=identity.process_name,
            window_title=identity.window_title,
            is_foreground=True,
            foreground_confidence=ForegroundConfidence.OBSERVED,
            metadata={
                "hwnd": identity.hwnd,
                "process_id": identity.process_id,
                "window_class": identity.window_class,
            },
        )


def _foreground_identity() -> ForegroundIdentity:
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
    except ImportError as exc:
        raise RuntimeError(f"Brak zależności Win32 dla samplera foreground: {exc}") from exc

    hwnd = int(win32gui.GetForegroundWindow() or 0)
    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        raise RuntimeError("Nie znaleziono aktywnego okna.")
    _, process_id = win32process.GetWindowThreadProcessId(hwnd)
    process_id = int(process_id or 0)
    process_name = ""
    if process_id > 0:
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
            False,
            process_id,
        )
        try:
            process_name = os.path.basename(
                win32process.GetModuleFileNameEx(handle, 0)
            )
        finally:
            win32api.CloseHandle(handle)
    return ForegroundIdentity(
        hwnd=hwnd,
        process_id=process_id,
        process_name=process_name,
        window_title=(win32gui.GetWindowText(hwnd) or "").strip(),
        window_class=(win32gui.GetClassName(hwnd) or "").strip(),
    )
