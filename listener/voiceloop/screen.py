from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
from pathlib import Path

from .models import ScreenSnapshot


class ScreenContextService:
    def __init__(self, screenshots_dir: Path) -> None:
        self.screenshots_dir = screenshots_dir

    async def capture(self, request_id: str, include_controls: bool = True) -> ScreenSnapshot:
        return await asyncio.to_thread(self._capture_sync, request_id, include_controls)

    def _capture_sync(self, request_id: str, include_controls: bool) -> ScreenSnapshot:
        try:
            import win32con
            import win32gui
            import win32process
            from PIL import Image, ImageGrab
        except ImportError as exc:
            return ScreenSnapshot(window_title=f"screen dependencies unavailable: {exc}")

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd) or ""
        process_name = ""
        try:
            import win32api

            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                False,
                process_id,
            )
            process_name = Path(win32process.GetModuleFileNameEx(handle, 0)).name
            win32api.CloseHandle(handle)
        except Exception:
            process_name = ""

        image_path: str | None = None
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right > left and bottom > top:
                self.screenshots_dir.mkdir(parents=True, exist_ok=True)
                safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", request_id)
                destination = self.screenshots_dir / f"{safe_id}.jpg"
                image = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
                image.thumbnail((896, 672), Image.Resampling.LANCZOS)
                image.convert("RGB").save(destination, quality=80, optimize=True)
                image_path = str(destination)
        except Exception:
            image_path = None

        controls: list[dict[str, object]] = []
        if include_controls:
            try:
                from pywinauto import Desktop

                window = Desktop(backend="uia").window(handle=hwnd)
                for control in window.descendants()[:60]:
                    info = control.element_info
                    control_type = info.control_type or ""
                    name = info.name or ""
                    if control_type in {"Edit", "Document"}:
                        name = "[editable content hidden]"
                    rectangle = info.rectangle
                    controls.append(
                        {
                            "name": name[:120],
                            "control_type": control_type,
                            "automation_id": (info.automation_id or "")[:120],
                            "enabled": bool(info.enabled),
                            "rectangle": [
                                rectangle.left,
                                rectangle.top,
                                rectangle.right,
                                rectangle.bottom,
                            ],
                        }
                    )
            except Exception:
                controls = []

        return ScreenSnapshot(
            window_title=title[:500],
            process_name=process_name,
            image_path=image_path,
            controls=controls,
        )

    @staticmethod
    def image_data_url(snapshot: ScreenSnapshot) -> str | None:
        if not snapshot.image_path:
            return None
        path = Path(snapshot.image_path)
        if not path.exists():
            return None
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
