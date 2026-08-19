from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .audio_capture import CapturedAudioChunk, MeetingChannelCapture
from .events import EventBus
from .hume_emotion import HumeEmotionClient, HumeEmotionError
from .memory import (
    MeetingSession,
    MeetingTranscriptSegment,
    MemoryStore,
)
from .models import TranscriptEnvelopeV1, TranscriptWordV1
from .screenpipe import ScreenpipeAudioChunk, ScreenpipeClient, ScreenpipeError
from .screenpipe_deepgram import DeepgramFileError, DeepgramFileTranscriber
from .settings import Settings

LOGGER = logging.getLogger("voiceloop.meeting_recorder")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _channel_for_chunk(chunk: ScreenpipeAudioChunk) -> str:
    value = f"{chunk.device_type} {chunk.device_name}".casefold()
    if "input" in value or "microphone" in value or "mikrofon" in value:
        return "input"
    if "output" in value or "speaker" in value or "głoś" in value:
        return "output"
    return "unknown"


def _speaker_label(channel: str, speaker_id: int | None) -> str:
    if channel == "input":
        return "Ty"
    if channel == "output":
        if speaker_id in {None, 0}:
            return "Rozmówca"
        return f"Rozmówca {speaker_id + 1}"
    if speaker_id is None:
        return "Nieznany kanał"
    return f"Mówca {speaker_id + 1}"


def _word_text(word: TranscriptWordV1) -> str:
    return (word.punctuated_word or word.word).strip()


def _diarized_turns(
    envelope: TranscriptEnvelopeV1,
    *,
    channel: str,
) -> list[tuple[int | None, str, float, float]]:
    if channel == "input" or not envelope.words:
        start = envelope.started_at_seconds or 0.0
        end = envelope.ended_at_seconds
        if end is None:
            end = start
        return [(None, envelope.raw_text.strip(), start, end)]

    turns: list[tuple[int | None, str, float, float]] = []
    current_speaker: int | None = None
    current_words: list[str] = []
    current_start = 0.0
    current_end = 0.0
    for word in envelope.words:
        if current_words and word.speaker_id != current_speaker:
            turns.append(
                (
                    current_speaker,
                    " ".join(current_words).strip(),
                    current_start,
                    current_end,
                )
            )
            current_words = []
        if not current_words:
            current_speaker = word.speaker_id
            current_start = word.start_seconds
        current_words.append(_word_text(word))
        current_end = word.end_seconds
    if current_words:
        turns.append(
            (
                current_speaker,
                " ".join(current_words).strip(),
                current_start,
                current_end,
            )
        )
    return [turn for turn in turns if turn[1]]


class MeetingRecorder:
    """Local meeting mode: silent assistant, durable transcript and raw audio.

    The microphone is transcribed live by the streaming Deepgram listener.
    Deepgram microphone PCM and the default Windows output loopback are archived
    directly. Screenpipe files are imported as a redundant local source.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        memory: MemoryStore,
        events: EventBus,
        screenpipe: ScreenpipeClient,
        file_transcriber: DeepgramFileTranscriber | None = None,
        emotion_analyzer: HumeEmotionClient | None = None,
        channel_capture: MeetingChannelCapture | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.events = events
        self.screenpipe = screenpipe
        self.file_transcriber = file_transcriber or DeepgramFileTranscriber(settings)
        self.emotion_analyzer = emotion_analyzer or HumeEmotionClient(settings)
        self.audio_root = (settings.data_dir / "meetings").resolve()
        self.screenpipe_data_root = (Path.home() / ".screenpipe" / "data").resolve()
        self.poll_seconds = max(3, settings.meeting_recording_poll_seconds)
        self.finalize_seconds = max(5, settings.meeting_recording_finalize_seconds)
        self.archive_audio = settings.meeting_recording_archive_audio
        self._active: MeetingSession | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._finalize_tasks: set[asyncio.Task[None]] = set()
        self._emotion_tasks: set[asyncio.Task[None]] = set()
        self._processed_chunks: set[str] = set()
        self._transcribed_sources: set[str] = set()
        self._lock = asyncio.Lock()
        self.channel_capture = (
            channel_capture
            if channel_capture is not None
            else MeetingChannelCapture(
                settings=settings,
                on_chunk=self._handle_direct_chunk,
            )
        )

    @property
    def active(self) -> bool:
        return self._active is not None and self._active.status == "active"

    def health(self) -> tuple[bool, str]:
        if self._active is not None:
            capture_ok, capture_detail = self.channel_capture.health()
            return (
                capture_ok,
                f"nagrywanie: {self._active.session_id}; {capture_detail}",
            )
        running_finalizers = sum(not task.done() for task in self._finalize_tasks)
        if running_finalizers:
            return True, f"finalizowanie nagrań: {running_finalizers}"
        return True, "gotowy"

    async def start(self, *, title: str = "") -> dict[str, object]:
        async with self._lock:
            if self._active is not None:
                return await self.session_payload(self._active)

            stale = await self.memory.active_meeting_session()
            if stale is not None:
                await self.memory.update_meeting_session(
                    stale.session_id,
                    status="interrupted",
                    ended_at=datetime.now(UTC),
                )

            started_at = datetime.now(UTC)
            session_id = (
                f"meeting-{started_at.strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid4().hex[:8]}"
            )
            audio_dir = self.audio_root / session_id / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            session = await self.memory.create_meeting_session(
                session_id=session_id,
                started_at=started_at,
                audio_dir=audio_dir,
                title=title,
            )
            self._active = session
            self._processed_chunks.clear()
            self._transcribed_sources.clear()
            await self.channel_capture.start(
                session_id=session_id,
                audio_dir=audio_dir,
            )
            self._poll_task = asyncio.create_task(
                self._poll_loop(session),
                name=f"meeting-audio-{session_id}",
            )

        payload = await self.session_payload(session)
        await self.events.publish("meeting.started", payload)
        return payload

    async def stop(self) -> dict[str, object]:
        async with self._lock:
            session = self._active
            if session is None:
                latest = (await self.memory.list_meeting_sessions(limit=1))
                if not latest:
                    return {"status": "idle", "active": False}
                return await self.session_payload(latest[0])

            poll_task = self._poll_task
            self._poll_task = None
            if poll_task is not None:
                poll_task.cancel()
                await asyncio.gather(poll_task, return_exceptions=True)
            await self.channel_capture.stop()
            self._active = None

            ended_at = datetime.now(UTC)
            updated = await self.memory.update_meeting_session(
                session.session_id,
                status="finalizing",
                ended_at=ended_at,
            )
            assert updated is not None
            finalizer = asyncio.create_task(
                self._finalize(updated),
                name=f"meeting-finalize-{session.session_id}",
            )
            self._finalize_tasks.add(finalizer)
            finalizer.add_done_callback(self._finalize_tasks.discard)

        payload = await self.session_payload(updated)
        await self.events.publish("meeting.stopped", payload)
        return payload

    async def close(self) -> None:
        poll_task = self._poll_task
        self._poll_task = None
        if poll_task is not None:
            poll_task.cancel()
            await asyncio.gather(poll_task, return_exceptions=True)
        await self.channel_capture.stop()
        tasks = tuple(self._finalize_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._finalize_tasks.clear()
        emotion_tasks = tuple(self._emotion_tasks)
        for task in emotion_tasks:
            task.cancel()
        if emotion_tasks:
            await asyncio.gather(*emotion_tasks, return_exceptions=True)
        self._emotion_tasks.clear()
        session = self._active
        self._active = None
        if session is not None:
            await self.memory.update_meeting_session(
                session.session_id,
                status="interrupted",
                ended_at=datetime.now(UTC),
            )

    def feed_microphone_audio(self, data: bytes) -> None:
        self.channel_capture.feed_microphone_audio(data)

    async def record_live(self, envelope: TranscriptEnvelopeV1) -> bool:
        session = self._active
        if session is None:
            return False
        end_time = envelope.created_at.astimezone(UTC)
        duration = 0.0
        if (
            envelope.started_at_seconds is not None
            and envelope.ended_at_seconds is not None
        ):
            duration = max(
                0.0,
                envelope.ended_at_seconds - envelope.started_at_seconds,
            )
        start_time = end_time - timedelta(seconds=duration)
        inserted = await self.memory.save_meeting_transcript_segment(
            session_id=session.session_id,
            segment_key=f"live:{envelope.segment_id}",
            channel="input",
            speaker_label="Ty",
            speaker_id=None,
            device_name=str(self.settings.microphone_device or "mikrofon"),
            start_time=start_time,
            end_time=end_time,
            text=envelope.raw_text,
            transcript=envelope,
            source="deepgram_live",
        )
        if inserted:
            await self.events.publish(
                "meeting.transcript.final",
                {
                    "session_id": session.session_id,
                    "channel": "input",
                    "speaker_label": "Ty",
                    "speaker_id": None,
                    "device_name": str(self.settings.microphone_device or "mikrofon"),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "text": envelope.raw_text,
                    "source": "deepgram_live",
                },
            )
        return inserted

    async def current_payload(self) -> dict[str, object]:
        session = self._active
        if session is None:
            active = await self.memory.active_meeting_session()
            if active is not None:
                session = active
        if session is None:
            sessions = await self.memory.list_meeting_sessions(limit=1)
            if not sessions:
                return {"status": "idle", "active": False, "segments": [], "audio": []}
            session = sessions[0]
        return await self.session_payload(session)

    async def get_payload(self, session_id: str) -> dict[str, object] | None:
        session = await self.memory.get_meeting_session(session_id)
        if session is None:
            return None
        return await self.session_payload(session)

    async def list_payloads(self, *, limit: int = 20) -> list[dict[str, object]]:
        sessions = await self.memory.list_meeting_sessions(limit=limit)
        return [await self.session_payload(session, include_details=False) for session in sessions]

    async def session_payload(
        self,
        session: MeetingSession,
        *,
        include_details: bool = True,
    ) -> dict[str, object]:
        segments = (
            await self.memory.list_meeting_transcript_segments(session.session_id)
            if include_details
            else []
        )
        audio = (
            await self.memory.list_meeting_audio_files(session.session_id)
            if include_details
            else []
        )
        return {
            "session_id": session.session_id,
            "status": session.status,
            "active": bool(
                self._active is not None
                and self._active.session_id == session.session_id
            ),
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "title": session.title,
            "audio_dir": session.audio_dir,
            "segment_count": len(segments) if include_details else None,
            "audio_file_count": len(audio) if include_details else None,
            "segments": [self._segment_payload(item) for item in segments],
            "audio": [
                {
                    "id": item.id,
                    "chunk_id": item.chunk_id,
                    "channel": item.channel,
                    "device_name": item.device_name,
                    "start_time": item.start_time.isoformat(),
                    "end_time": item.end_time.isoformat(),
                    "archived_path": item.archived_path,
                }
                for item in audio
            ],
        }

    async def _poll_loop(self, session: MeetingSession) -> None:
        while (
            self._active is not None
            and self._active.session_id == session.session_id
        ):
            try:
                await self._process_screenpipe_chunks(
                    session,
                    range_end=datetime.now(UTC),
                )
            except asyncio.CancelledError:
                raise
            except ScreenpipeError as exc:
                LOGGER.warning("Meeting Screenpipe poll paused: %s", exc)
            except Exception:
                LOGGER.exception("Meeting audio poll failed")
            await asyncio.sleep(self.poll_seconds)

    async def _finalize(self, session: MeetingSession) -> None:
        assert session.ended_at is not None
        deadline = asyncio.get_running_loop().time() + self.finalize_seconds
        try:
            while True:
                await self._process_screenpipe_chunks(
                    session,
                    range_end=session.ended_at + timedelta(seconds=2),
                )
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(min(self.poll_seconds, 5))
            completed = await self.memory.update_meeting_session(
                session.session_id,
                status="completed",
                ended_at=session.ended_at,
            )
            if completed is not None:
                await self.events.publish(
                    "meeting.completed",
                    await self.session_payload(completed),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Meeting finalization failed")
            await self.memory.update_meeting_session(
                session.session_id,
                status="error",
                ended_at=session.ended_at,
            )

    async def _process_screenpipe_chunks(
        self,
        session: MeetingSession,
        *,
        range_end: datetime,
    ) -> None:
        range_start = session.started_at - timedelta(seconds=2)
        try:
            api_chunks = await self.screenpipe.audio_chunks(
                start=range_start,
                end=range_end,
                max_results=2000,
            )
        except ScreenpipeError as exc:
            LOGGER.warning("Screenpipe audio API unavailable, using files: %s", exc)
            api_chunks = []
        file_chunks = await asyncio.to_thread(
            self._scan_screenpipe_audio_files,
            range_start,
            range_end,
        )
        chunks_by_path: dict[str, ScreenpipeAudioChunk] = {}
        for chunk in [*api_chunks, *file_chunks]:
            path = chunk.file_path
            if not path.is_absolute():
                path = self.screenpipe_data_root / path
            chunks_by_path[str(path.resolve())] = chunk
        chunks = list(chunks_by_path.values())
        for chunk in sorted(chunks, key=lambda item: item.start_time):
            process_key = f"{session.session_id}:{chunk.chunk_id}"
            if process_key in self._processed_chunks:
                continue
            try:
                processed = await self._process_chunk(session, chunk)
            except (DeepgramFileError, OSError) as exc:
                LOGGER.warning(
                    "Meeting chunk %s is not ready: %s",
                    chunk.chunk_id,
                    exc,
                )
                continue
            if processed:
                self._processed_chunks.add(process_key)

    def _scan_screenpipe_audio_files(
        self,
        range_start: datetime,
        range_end: datetime,
    ) -> list[ScreenpipeAudioChunk]:
        root = self.screenpipe_data_root
        if not root.is_dir():
            return []
        chunks: list[ScreenpipeAudioChunk] = []
        now = datetime.now(UTC)
        for path in root.iterdir():
            if not path.is_file() or path.suffix.casefold() not in {
                ".mp4",
                ".m4a",
                ".wav",
                ".webm",
            }:
                continue
            match = re.match(
                r"^(?P<device>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}_"
                r"\d{2}-\d{2}-\d{2})$",
                path.stem,
            )
            if match is None:
                continue
            try:
                started_at = datetime.strptime(
                    match.group("stamp"),
                    "%Y-%m-%d_%H-%M-%S",
                ).replace(tzinfo=UTC)
                stat = path.stat()
            except (OSError, ValueError):
                continue
            ended_at = max(
                started_at
                + timedelta(seconds=self.settings.meeting_recording_audio_chunk_seconds),
                datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            )
            if ended_at < range_start or started_at > range_end:
                continue
            if now - datetime.fromtimestamp(stat.st_mtime, tz=UTC) < timedelta(seconds=1):
                continue
            device_name = match.group("device")
            device_type = (
                "input"
                if device_name.casefold().endswith("(input)")
                else (
                    "output"
                    if device_name.casefold().endswith("(output)")
                    else "unknown"
                )
            )
            chunks.append(
                ScreenpipeAudioChunk(
                    chunk_id=f"file:{path.name}",
                    file_path=path,
                    device_name=device_name,
                    device_type=device_type,
                    start_time=started_at.isoformat(),
                    end_time=ended_at.isoformat(),
                    text="",
                    start_offset_seconds=0.0,
                    end_offset_seconds=max(
                        0.0,
                        (ended_at - started_at).total_seconds(),
                    ),
                )
            )
        return chunks

    async def _process_chunk(
        self,
        session: MeetingSession,
        chunk: ScreenpipeAudioChunk,
    ) -> bool:
        source_path = self._safe_audio_path(chunk)
        if source_path is None:
            return False
        channel = _channel_for_chunk(chunk)
        archived_path = source_path
        if self.archive_audio:
            archived_path = await self._archive_chunk(
                session=session,
                chunk=chunk,
                source_path=source_path,
                channel=channel,
            )
        return await self._persist_audio_chunk(
            session=session,
            chunk=chunk,
            source_path=source_path,
            archived_path=archived_path,
            channel=channel,
            source_kind="screenpipe",
        )

    async def _handle_direct_chunk(self, captured: CapturedAudioChunk) -> None:
        session = self._active
        if session is None:
            return
        chunk = ScreenpipeAudioChunk(
            chunk_id=captured.chunk_id,
            file_path=captured.file_path,
            device_name=captured.device_name,
            device_type=captured.channel,
            start_time=captured.started_at.isoformat(),
            end_time=captured.ended_at.isoformat(),
            text="",
        )
        await self._persist_audio_chunk(
            session=session,
            chunk=chunk,
            source_path=captured.file_path.resolve(),
            archived_path=captured.file_path.resolve(),
            channel=captured.channel,
            source_kind="direct",
        )

    async def _persist_audio_chunk(
        self,
        *,
        session: MeetingSession,
        chunk: ScreenpipeAudioChunk,
        source_path: Path,
        archived_path: Path,
        channel: str,
        source_kind: str,
    ) -> bool:
        start_time = _parse_timestamp(chunk.start_time)
        end_time = _parse_timestamp(chunk.end_time)
        await self.memory.save_meeting_audio_file(
            session_id=session.session_id,
            chunk_id=chunk.chunk_id,
            channel=channel,
            device_name=chunk.device_name,
            start_time=start_time,
            end_time=end_time,
            source_path=source_path,
            archived_path=archived_path,
        )
        await self.events.publish(
            "meeting.audio.archived",
            {
                "session_id": session.session_id,
                "channel": channel,
                "device_name": chunk.device_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "archived_path": str(archived_path),
                "source": source_kind,
            },
        )

        # The microphone has an immediate, durable live transcript. Reprocessing
        # it would duplicate every utterance in the visible transcript.
        if channel == "input":
            self._schedule_emotion_annotation(
                session_id=session.session_id,
                channel=channel,
                audio_path=archived_path,
                file_start=start_time,
                file_end=end_time,
                targets=None,
            )
            return True

        source_key = f"{session.session_id}:{source_path}"
        if source_key in self._transcribed_sources:
            return True
        envelope = await self.file_transcriber.transcribe_envelope(source_path)
        if envelope is None:
            self._transcribed_sources.add(source_key)
            return True
        turns = _diarized_turns(envelope, channel=channel)
        file_start = start_time - timedelta(seconds=chunk.start_offset_seconds)
        source_digest = hashlib.sha256(str(source_path).encode()).hexdigest()[:20]
        emotion_targets: list[dict[str, object]] = []
        for index, (speaker_id, text, relative_start, relative_end) in enumerate(turns):
            absolute_start = file_start + timedelta(seconds=relative_start)
            absolute_end = file_start + timedelta(seconds=relative_end)
            label = _speaker_label(channel, speaker_id)
            segment_key = (
                f"{source_kind}:{session.session_id}:{source_digest}:"
                f"{index}:{speaker_id}"
            )
            transcript_source = (
                "direct_output_deepgram"
                if source_kind == "direct"
                else "screenpipe_deepgram"
            )
            inserted = await self.memory.save_meeting_transcript_segment(
                session_id=session.session_id,
                segment_key=segment_key,
                channel=channel,
                speaker_label=label,
                speaker_id=speaker_id,
                device_name=chunk.device_name,
                start_time=absolute_start,
                end_time=absolute_end,
                text=text,
                transcript=envelope,
                source=transcript_source,
            )
            if inserted:
                emotion_targets.append(
                    {
                        "segment_key": segment_key,
                        "begin_seconds": relative_start,
                        "end_seconds": relative_end,
                    }
                )
                await self.events.publish(
                    "meeting.transcript.final",
                    {
                        "session_id": session.session_id,
                        "channel": channel,
                        "speaker_label": label,
                        "speaker_id": speaker_id,
                        "device_name": chunk.device_name,
                        "start_time": absolute_start.isoformat(),
                        "end_time": absolute_end.isoformat(),
                        "text": text,
                        "source": transcript_source,
                    },
                )
        self._schedule_emotion_annotation(
            session_id=session.session_id,
            channel=channel,
            audio_path=archived_path,
            file_start=file_start,
            file_end=end_time,
            targets=emotion_targets,
        )
        self._transcribed_sources.add(source_key)
        return True

    def _schedule_emotion_annotation(
        self,
        *,
        session_id: str,
        channel: str,
        audio_path: Path,
        file_start: datetime,
        file_end: datetime,
        targets: list[dict[str, object]] | None,
    ) -> None:
        if not self.emotion_analyzer.enabled:
            return
        if targets == []:
            return
        task = asyncio.create_task(
            self._annotate_audio_emotions(
                session_id=session_id,
                channel=channel,
                audio_path=audio_path,
                file_start=file_start,
                file_end=file_end,
                targets=targets,
            ),
            name=f"hume-emotions-{session_id}",
        )
        self._emotion_tasks.add(task)
        task.add_done_callback(self._emotion_tasks.discard)

    async def _annotate_audio_emotions(
        self,
        *,
        session_id: str,
        channel: str,
        audio_path: Path,
        file_start: datetime,
        file_end: datetime,
        targets: list[dict[str, object]] | None,
    ) -> None:
        try:
            windows = await self.emotion_analyzer.analyze_file(audio_path)
        except (HumeEmotionError, OSError) as exc:
            LOGGER.warning("Hume emotion analysis skipped for %s: %s", audio_path, exc)
            return
        if not windows:
            return

        resolved_targets = targets
        if resolved_targets is None:
            resolved_targets = await self._emotion_targets_for_audio_range(
                session_id=session_id,
                channel=channel,
                file_start=file_start,
                file_end=file_end,
            )

        for target in resolved_targets:
            segment_key = str(target.get("segment_key") or "")
            if not segment_key:
                continue
            try:
                begin_seconds = float(target.get("begin_seconds") or 0.0)
                end_seconds = float(target.get("end_seconds") or begin_seconds)
            except (TypeError, ValueError):
                continue
            emotions = self.emotion_analyzer.emotions_for_interval(
                windows,
                begin_seconds=begin_seconds,
                end_seconds=max(end_seconds, begin_seconds),
            )
            if not emotions:
                continue
            updated = await self.memory.save_meeting_segment_emotions(
                segment_key=segment_key,
                emotions=emotions,
            )
            if updated:
                await self.events.publish(
                    "meeting.segment.emotions",
                    {
                        "session_id": session_id,
                        "segment_key": segment_key,
                        "emotions": emotions,
                        "source": "hume_prosody",
                    },
                )

    async def _emotion_targets_for_audio_range(
        self,
        *,
        session_id: str,
        channel: str,
        file_start: datetime,
        file_end: datetime,
    ) -> list[dict[str, object]]:
        segments = await self.memory.list_meeting_transcript_segments(session_id)
        targets: list[dict[str, object]] = []
        for segment in segments:
            if segment.channel != channel:
                continue
            if segment.end_time < file_start or segment.start_time > file_end:
                continue
            targets.append(
                {
                    "segment_key": segment.segment_key,
                    "begin_seconds": max(
                        0.0,
                        (segment.start_time - file_start).total_seconds(),
                    ),
                    "end_seconds": max(
                        0.0,
                        (segment.end_time - file_start).total_seconds(),
                    ),
                }
            )
        return targets

    async def _archive_chunk(
        self,
        *,
        session: MeetingSession,
        chunk: ScreenpipeAudioChunk,
        source_path: Path,
        channel: str,
    ) -> Path:
        audio_dir = Path(session.audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
        device = re.sub(r"[^A-Za-z0-9._-]+", "-", chunk.device_name).strip("-")
        device = device[:60] or "device"
        target = audio_dir / f"{channel}-{device}-{digest}{source_path.suffix.lower()}"
        if not target.exists():
            await asyncio.to_thread(shutil.copy2, source_path, target)
        return target

    def _safe_audio_path(self, chunk: ScreenpipeAudioChunk) -> Path | None:
        candidate = chunk.file_path
        if not candidate.is_absolute():
            candidate = self.screenpipe_data_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.screenpipe_data_root)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if resolved.suffix.casefold() not in {".mp4", ".m4a", ".wav", ".webm"}:
            return None
        return resolved

    @staticmethod
    def _segment_payload(item: MeetingTranscriptSegment) -> dict[str, object]:
        return {
            "id": item.id,
            "segment_key": item.segment_key,
            "channel": item.channel,
            "speaker_label": item.speaker_label,
            "speaker_id": item.speaker_id,
            "device_name": item.device_name,
            "start_time": item.start_time.isoformat(),
            "end_time": item.end_time.isoformat(),
            "text": item.text,
            "emotions": list(item.emotions),
            "source": item.source,
        }
