from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from .memory import MemoryStore
from .models import (
    TranscriptEnvelopeV1,
    TranscriptWordV1,
    normalize_transcript_text,
)
from .screenpipe import ScreenpipeAudioChunk, ScreenpipeClient, ScreenpipeError, ScreenpipeMeeting
from .screenpipe_audio_policy import DeepgramAudioPolicy
from .settings import Settings

LOGGER = logging.getLogger("voiceloop.screenpipe_deepgram")
_EPOCH_KEY = "screenpipe_deepgram_enabled_at"


class DeepgramFileError(RuntimeError):
    pass


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


class DeepgramFileTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.deepgram_api_key
        self.model = settings.deepgram_model
        self.language = settings.deepgram_language
        self.max_bytes = max(1, settings.screenpipe_deepgram_max_file_mb) * 1024 * 1024

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.api_key.get_secret_value().strip())

    async def transcribe(self, path: Path) -> str:
        envelope = await self.transcribe_envelope(path)
        return envelope.raw_text if envelope is not None else ""

    async def transcribe_envelope(self, path: Path) -> TranscriptEnvelopeV1 | None:
        if not self.available:
            raise DeepgramFileError("Brak klucza DEEPGRAM_API_KEY.")
        size = path.stat().st_size
        if size <= 0:
            return None
        if size > self.max_bytes:
            raise DeepgramFileError(f"Plik audio przekracza limit {self.max_bytes} bajtów.")

        media_type = {
            ".mp4": "audio/mp4",
            ".m4a": "audio/mp4",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
        }.get(path.suffix.casefold(), "application/octet-stream")
        body = await asyncio.to_thread(path.read_bytes)
        headers = {
            "Authorization": f"Token {self.api_key.get_secret_value().strip()}",
            "Content-Type": media_type,
        }
        params = {
            "model": self.model,
            "language": self.language,
            "smart_format": "true",
            "punctuate": "true",
            "diarize": "true",
            "utterances": "true",
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    headers=headers,
                    params=params,
                    content=body,
                )
                response.raise_for_status()
                payload = response.json()
            return envelope_from_deepgram_payload(
                payload,
                language=self.language,
                model=self.model,
            )
        except httpx.HTTPStatusError as exc:
            raise DeepgramFileError(
                f"Deepgram zwrócił HTTP {exc.response.status_code}."
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise DeepgramFileError("Deepgram nie zwrócił poprawnej transkrypcji.") from exc


def envelope_from_deepgram_payload(
    payload: dict,
    *,
    language: str,
    model: str,
) -> TranscriptEnvelopeV1 | None:
    alternative = payload["results"]["channels"][0]["alternatives"][0]
    transcript = str(alternative.get("transcript") or "").strip()
    if not transcript:
        return None
    words = tuple(
        TranscriptWordV1(
            word=str(item.get("word") or item.get("punctuated_word") or "").strip(),
            punctuated_word=(
                str(item["punctuated_word"]).strip()
                if item.get("punctuated_word")
                else None
            ),
            start_seconds=float(item.get("start") or 0.0),
            end_seconds=float(item.get("end") or item.get("start") or 0.0),
            confidence=(
                float(item["confidence"]) if item.get("confidence") is not None else None
            ),
            speaker_id=(int(item["speaker"]) if item.get("speaker") is not None else None),
        )
        for item in alternative.get("words") or []
        if isinstance(item, dict)
        and str(item.get("word") or item.get("punctuated_word") or "").strip()
    )
    confidences = [word.confidence for word in words if word.confidence is not None]
    alternative_confidence = alternative.get("confidence")
    confidence_mean = (
        sum(confidences) / len(confidences)
        if confidences
        else (
            float(alternative_confidence)
            if alternative_confidence is not None
            else None
        )
    )
    speaker_ids = tuple(
        sorted(
            {
                word.speaker_id
                for word in words
                if word.speaker_id is not None
            }
        )
    )
    starts = [word.start_seconds for word in words]
    ends = [word.end_seconds for word in words]
    return TranscriptEnvelopeV1(
        raw_text=transcript,
        normalized_text=normalize_transcript_text(transcript),
        language=language,
        confidence_mean=confidence_mean,
        confidence_min=min(confidences) if confidences else confidence_mean,
        words=words,
        started_at_seconds=min(starts) if starts else None,
        ended_at_seconds=max(ends) if ends else None,
        speaker_ids=speaker_ids,
        is_final=True,
        speech_final=True,
        model=model,
    )


class ScreenpipeMeetingTranscriber:
    """Transcribe completed Screenpipe meetings, never general media."""

    def __init__(
        self,
        settings: Settings,
        screenpipe: ScreenpipeClient,
        memory: MemoryStore,
        file_transcriber: DeepgramFileTranscriber | None = None,
    ) -> None:
        self.settings = settings
        self.screenpipe = screenpipe
        self.memory = memory
        self.policy = DeepgramAudioPolicy(settings)
        self.file_transcriber = file_transcriber or DeepgramFileTranscriber(settings)
        self.poll_seconds = max(10, settings.screenpipe_deepgram_poll_seconds)
        self.grace_seconds = max(30, settings.screenpipe_deepgram_meeting_grace_seconds)
        self.data_root = (Path.home() / ".screenpipe" / "data").resolve()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._last_status = "nie uruchomiono"

    async def start(self) -> None:
        if not self.settings.screenpipe_deepgram_enabled:
            self._last_status = "wyłączony w konfiguracji"
            return
        if not self.file_transcriber.available:
            self._last_status = "brak DEEPGRAM_API_KEY"
            return
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="screenpipe-meeting-transcriber")
        self._last_status = "uruchomiony"

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def health(self) -> tuple[bool, str]:
        running = bool(self._task and not self._task.done())
        return running, self._last_status

    async def _run(self) -> None:
        backoff_seconds = float(self.poll_seconds)
        while not self._stop.is_set():
            try:
                processed = await self.process_once()
                self._last_status = (
                    f"działa; przetworzono spotkań: {processed}"
                    if processed
                    else "działa; oczekuje na zakończone spotkanie"
                )
                backoff_seconds = float(self.poll_seconds)
            except asyncio.CancelledError:
                raise
            except ScreenpipeError as exc:
                self._last_status = f"Screenpipe niedostępny: {exc}"
                LOGGER.warning(
                    "Selective Screenpipe transcription paused: %s",
                    exc,
                )
                backoff_seconds = min(max(backoff_seconds * 2, self.poll_seconds), 300.0)
            except Exception as exc:
                self._last_status = f"błąd: {type(exc).__name__}"
                LOGGER.exception("Selective Screenpipe transcription failed")
                backoff_seconds = min(max(backoff_seconds * 2, self.poll_seconds), 300.0)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff_seconds)
            except TimeoutError:
                pass

    async def process_once(self) -> int:
        now = datetime.now(UTC)
        epoch_text = await self.memory.get_state(_EPOCH_KEY)
        if epoch_text is None:
            await self.memory.set_state(_EPOCH_KEY, now.isoformat())
            await self.memory.prune_screenpipe_transcripts(
                retention_days=self.settings.screenpipe_lookback_days
            )
            return 0
        epoch = _parse_timestamp(epoch_text)
        meetings = await self.screenpipe.meetings(
            start=now - timedelta(days=max(1, self.settings.screenpipe_lookback_days)),
            end=now,
            limit=500,
        )
        processed = 0
        for meeting in meetings:
            if not meeting.meeting_end:
                continue
            meeting_end = _parse_timestamp(meeting.meeting_end)
            if meeting_end < epoch:
                continue
            if (now - meeting_end).total_seconds() < self.grace_seconds:
                continue
            if await self.memory.has_screenpipe_meeting_job(meeting.id):
                continue
            if await self._process_meeting(meeting):
                processed += 1
        await self.memory.prune_screenpipe_transcripts(
            retention_days=self.settings.screenpipe_lookback_days
        )
        return processed

    async def _process_meeting(self, meeting: ScreenpipeMeeting) -> bool:
        start = _parse_timestamp(meeting.meeting_start)
        end = _parse_timestamp(meeting.meeting_end or meeting.meeting_start)
        if await self.screenpipe.has_youtube_context(start=start, end=end):
            await self.memory.mark_screenpipe_meeting_job(
                meeting.id,
                status="blocked",
                reason="youtube_blocked",
            )
            return True
        contexts = await self.screenpipe.contexts_between(start=start, end=end, limit=100)
        decisions = [
            self.policy.decide(
                app_name=meeting.meeting_app,
                window_name=f"{meeting.title} {context.window_name}",
                browser_url=context.browser_url,
                meeting_detected=True,
                microphone_active=True,
            )
            for context in contexts
        ]
        decision = self.policy.decide(
            app_name=meeting.meeting_app,
            window_name=meeting.title,
            meeting_detected=True,
            microphone_active=True,
        )
        youtube = next(
            (item for item in decisions if item.reason == "youtube_blocked"),
            None,
        )
        if youtube is not None or not decision.allowed:
            reason = youtube.reason if youtube is not None else decision.reason
            await self.memory.mark_screenpipe_meeting_job(
                meeting.id,
                status="blocked",
                reason=reason,
            )
            return True

        chunks = await self.screenpipe.audio_chunks(start=start, end=end)
        saved = 0
        for chunk in chunks:
            if await self.memory.has_screenpipe_transcript(chunk.chunk_id):
                continue
            if chunk.text:
                text = chunk.text
                source = "screenpipe"
            else:
                path = self._safe_audio_path(chunk)
                if path is None:
                    continue
                text = await self.file_transcriber.transcribe(path)
                source = "deepgram"
            await self.memory.save_screenpipe_transcript(
                chunk_id=chunk.chunk_id,
                meeting_id=meeting.id,
                device_name=chunk.device_name,
                device_type=chunk.device_type,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=text,
                source=source,
            )
            saved += 1

        await self.memory.mark_screenpipe_meeting_job(
            meeting.id,
            status="completed",
            reason=f"transcript_chunks={saved}",
        )
        return True

    def _safe_audio_path(self, chunk: ScreenpipeAudioChunk) -> Path | None:
        candidate = chunk.file_path
        if not candidate.is_absolute():
            candidate = self.data_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.data_root)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if resolved.suffix.casefold() not in {".mp4", ".m4a", ".wav", ".webm"}:
            return None
        return resolved
