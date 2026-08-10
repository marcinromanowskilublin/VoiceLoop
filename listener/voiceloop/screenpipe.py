from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .settings import Settings


class ScreenpipeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScreenpipeContext:
    app_name: str
    window_name: str
    timestamp: str
    browser_url: str = ""


@dataclass(frozen=True)
class ScreenpipeTextItem:
    app_name: str
    window_name: str
    timestamp: str
    browser_url: str
    text: str
    content_type: str


@dataclass(frozen=True)
class ScreenpipeMeeting:
    id: int
    meeting_start: str
    meeting_end: str | None
    meeting_app: str
    title: str
    detection_source: str


@dataclass(frozen=True)
class ScreenpipeAudioChunk:
    chunk_id: str
    file_path: Path
    device_name: str
    device_type: str
    start_time: str
    end_time: str
    text: str


class ScreenpipeClient:
    """Read-only client for Screenpipe's local API.

    This client never downloads frames, audio, or input events. It reads only
    application/window metadata and small OCR snippets returned by `/search`.
    """

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.screenpipe_enabled
        self.base_url = settings.screenpipe_base_url.rstrip("/")
        self.timeout_seconds = settings.screenpipe_timeout_seconds
        self.recent_window_seconds = max(10, min(settings.screenpipe_recent_window_seconds, 600))
        self.history_limit = max(1, min(settings.screenpipe_history_limit, 50))
        self.lookback_days = max(1, min(settings.screenpipe_lookback_days, 90))
        self._token = settings.screenpipe_api_token

    def _headers(self) -> dict[str, str]:
        if self._token is None:
            return {}
        token = self._token.get_secret_value().strip()
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączony w konfiguracji"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/health", headers=self._headers())
        except httpx.HTTPError as exc:
            return False, f"niedostępny: {exc}"
        if response.status_code in {401, 403}:
            return False, "wymaga poprawnego SCREENPIPE_API_TOKEN"
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"
        return True, "lokalny API odpowiada"

    async def recent_context(self) -> ScreenpipeContext | None:
        """Return the latest app/window metadata, never screen-image bytes."""
        now = datetime.now(UTC)
        items = await self._search(
            content_type="accessibility",
            start=now - timedelta(seconds=self.recent_window_seconds),
            end=now,
            limit=5,
        )
        for item in items:
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            app_name = str(content.get("app_name") or "").strip()
            window_name = str(content.get("window_name") or "").strip()
            timestamp = str(content.get("timestamp") or "").strip()
            if app_name or window_name:
                return ScreenpipeContext(
                    app_name=app_name,
                    window_name=window_name,
                    timestamp=timestamp,
                    browser_url=str(content.get("browser_url") or "").strip(),
                )
        return None

    async def recent_activity(self, *, minutes: int = 30) -> list[ScreenpipeContext]:
        """Return distinct recent app/window pairs in chronological order."""
        minutes = max(1, min(minutes, self.lookback_days * 24 * 60))
        now = datetime.now(UTC)
        items = await self._search(
            content_type="accessibility",
            start=now - timedelta(minutes=minutes),
            end=now,
            limit=self.history_limit,
        )
        contexts: list[ScreenpipeContext] = []
        seen: set[tuple[str, str]] = set()
        for item in reversed(items):
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            app_name = str(content.get("app_name") or "").strip()
            window_name = str(content.get("window_name") or "").strip()
            key = (app_name, window_name)
            if not (app_name or window_name) or key in seen:
                continue
            seen.add(key)
            contexts.append(
                ScreenpipeContext(
                    app_name=app_name,
                    window_name=window_name,
                    timestamp=str(content.get("timestamp") or "").strip(),
                    browser_url=str(content.get("browser_url") or "").strip(),
                )
            )
        return contexts

    async def recent_text_activity(
        self,
        *,
        minutes: int = 30,
        limit: int = 24,
    ) -> list[ScreenpipeTextItem]:
        """Return recent local text/OCR/audio snippets without applying content filters."""
        minutes = max(1, min(minutes, self.lookback_days * 24 * 60))
        now = datetime.now(UTC)
        items = await self._search(
            content_type="all",
            start=now - timedelta(minutes=minutes),
            end=now,
            limit=max(1, min(limit, 50)),
        )
        results: list[ScreenpipeTextItem] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in reversed(items):
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            app_name = str(content.get("app_name") or "").strip()
            window_name = str(content.get("window_name") or "").strip()
            timestamp = str(
                content.get("timestamp")
                or content.get("start_time")
                or content.get("created_at")
                or ""
            ).strip()
            text = str(
                content.get("text")
                or content.get("transcription")
                or content.get("accessibility_text")
                or ""
            ).strip()
            browser_url = str(content.get("browser_url") or "").strip()
            key = (timestamp, app_name, window_name, text)
            if key in seen or not (app_name or window_name or text):
                continue
            seen.add(key)
            results.append(
                ScreenpipeTextItem(
                    app_name=app_name,
                    window_name=window_name,
                    timestamp=timestamp,
                    browser_url=browser_url,
                    text=text[:4000],
                    content_type=str(item.get("type") or "").strip(),
                )
            )
        return results

    async def contexts_between(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int = 50,
    ) -> list[ScreenpipeContext]:
        items = await self._search(
            content_type="accessibility",
            start=start,
            end=end,
            limit=limit,
        )
        contexts: list[ScreenpipeContext] = []
        for item in items:
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            contexts.append(
                ScreenpipeContext(
                    app_name=str(content.get("app_name") or "").strip(),
                    window_name=str(content.get("window_name") or "").strip(),
                    timestamp=str(content.get("timestamp") or "").strip(),
                    browser_url=str(content.get("browser_url") or "").strip(),
                )
            )
        return contexts

    async def has_youtube_context(self, *, start: datetime, end: datetime) -> bool:
        """Conservatively detect any YouTube evidence during a time range."""
        items = await self._search(
            content_type="all",
            start=start,
            end=end,
            limit=20,
            query="youtube",
        )
        for item in items:
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            searchable = " ".join(
                str(content.get(key) or "")
                for key in ("app_name", "window_name", "browser_url", "text")
            ).casefold()
            if "youtube" in searchable or "youtu.be" in searchable:
                return True
        return False

    async def meetings(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[ScreenpipeMeeting]:
        params: dict[str, str] = {"limit": str(max(1, min(limit, 500)))}
        if start is not None:
            params["start_time"] = start.isoformat()
        if end is not None:
            params["end_time"] = end.isoformat()
        payload = await self._get_json("/meetings", params=params)
        if not isinstance(payload, list):
            return []
        meetings: list[ScreenpipeMeeting] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                meeting_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            meetings.append(
                ScreenpipeMeeting(
                    id=meeting_id,
                    meeting_start=str(item.get("meeting_start") or ""),
                    meeting_end=(
                        str(item["meeting_end"]) if item.get("meeting_end") is not None else None
                    ),
                    meeting_app=str(item.get("meeting_app") or ""),
                    title=str(item.get("title") or ""),
                    detection_source=str(item.get("detection_source") or ""),
                )
            )
        return meetings

    async def audio_chunks(
        self,
        *,
        start: datetime,
        end: datetime,
        max_results: int = 500,
    ) -> list[ScreenpipeAudioChunk]:
        max_results = max(1, min(max_results, 2000))
        items: list[dict[str, Any]] = []
        offset = 0
        while len(items) < max_results:
            page_size = min(50, max_results - len(items))
            page = await self._search(
                content_type="audio",
                start=start,
                end=end,
                limit=page_size,
                offset=offset,
            )
            items.extend(page)
            if len(page) < page_size:
                break
            offset += len(page)

        chunks: list[ScreenpipeAudioChunk] = []
        seen: set[str] = set()
        for item in items:
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            path_text = str(content.get("file_path") or "").strip()
            start_time = str(content.get("start_time") or content.get("timestamp") or "").strip()
            chunk_id = str(content.get("chunk_id") or f"{path_text}:{start_time}").strip()
            if not path_text or not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append(
                ScreenpipeAudioChunk(
                    chunk_id=chunk_id,
                    file_path=Path(path_text),
                    device_name=str(content.get("device_name") or "").strip(),
                    device_type=str(content.get("device_type") or "").strip(),
                    start_time=start_time,
                    end_time=str(content.get("end_time") or "").strip(),
                    text=str(content.get("transcription") or content.get("text") or "").strip(),
                )
            )
        return chunks

    async def _search(
        self,
        *,
        content_type: str,
        start: datetime,
        end: datetime,
        limit: int,
        offset: int = 0,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            raise ScreenpipeError("Screenpipe jest wyłączony w konfiguracji.")
        params = {
            "content_type": content_type,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "limit": str(max(1, min(limit, 50))),
            "offset": str(max(0, offset)),
        }
        if query:
            params["q"] = query
        payload = await self._get_json("/search", params=params)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    async def _get_json(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        if not self.enabled:
            raise ScreenpipeError("Screenpipe jest wyłączony w konfiguracji.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise ScreenpipeError("Screenpipe nie odpowiada lokalnie.") from exc
        if response.status_code in {401, 403}:
            raise ScreenpipeError("Screenpipe wymaga poprawnego SCREENPIPE_API_TOKEN.")
        if response.status_code >= 400:
            raise ScreenpipeError(f"Screenpipe zwrócił HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ScreenpipeError("Screenpipe zwrócił nieprawidłową odpowiedź.") from exc
        return payload
