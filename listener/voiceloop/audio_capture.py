from __future__ import annotations

import asyncio
import logging
import threading
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundcard as sc

from .settings import Settings

LOGGER = logging.getLogger("voiceloop.audio_capture")


@dataclass(frozen=True)
class CapturedAudioChunk:
    chunk_id: str
    channel: str
    device_name: str
    file_path: Path
    started_at: datetime
    ended_at: datetime


AudioChunkCallback = Callable[[CapturedAudioChunk], Awaitable[None]]


class MeetingChannelCapture:
    """Record microphone PCM and the default Windows speaker loopback separately."""

    def __init__(
        self,
        *,
        settings: Settings,
        on_chunk: AudioChunkCallback,
    ) -> None:
        self.on_chunk = on_chunk
        self.input_sample_rate = settings.sample_rate
        self.output_sample_rate = settings.meeting_recording_output_sample_rate
        self.chunk_seconds = max(5, settings.meeting_recording_audio_chunk_seconds)
        self.input_device_name = str(settings.microphone_device or "mikrofon Deepgram")
        self.active = False
        self.output_device_name = ""
        self.last_error: str | None = None
        self._session_id = ""
        self._audio_dir: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_queue: asyncio.Queue[bytes | None] | None = None
        self._output_queue: asyncio.Queue[CapturedAudioChunk | None] | None = None
        self._input_task: asyncio.Task[None] | None = None
        self._output_record_task: asyncio.Task[None] | None = None
        self._output_consumer_task: asyncio.Task[None] | None = None
        self._output_stop = threading.Event()

    async def start(self, *, session_id: str, audio_dir: Path) -> None:
        if self.active:
            return
        self.active = True
        self.last_error = None
        self.output_device_name = ""
        self._session_id = session_id
        self._audio_dir = audio_dir
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._loop = asyncio.get_running_loop()
        self._input_queue = asyncio.Queue(maxsize=400)
        self._output_queue = asyncio.Queue()
        self._output_stop.clear()
        self._input_task = asyncio.create_task(
            self._archive_input_loop(),
            name=f"meeting-input-archive-{session_id}",
        )
        self._output_consumer_task = asyncio.create_task(
            self._consume_output_chunks(),
            name=f"meeting-output-consumer-{session_id}",
        )
        self._output_record_task = asyncio.create_task(
            self._run_output_recorder(),
            name=f"meeting-output-loopback-{session_id}",
        )

    async def stop(self) -> None:
        if not self.active and not any(
            (self._input_task, self._output_record_task, self._output_consumer_task)
        ):
            return
        self.active = False
        self._output_stop.set()
        input_queue = self._input_queue
        if input_queue is not None:
            await input_queue.put(None)
        if self._input_task is not None:
            await asyncio.gather(self._input_task, return_exceptions=True)
        if self._output_record_task is not None:
            await asyncio.gather(self._output_record_task, return_exceptions=True)
        output_queue = self._output_queue
        if output_queue is not None:
            await output_queue.put(None)
        if self._output_consumer_task is not None:
            await asyncio.gather(self._output_consumer_task, return_exceptions=True)
        self._input_task = None
        self._output_record_task = None
        self._output_consumer_task = None
        self._input_queue = None
        self._output_queue = None
        self._loop = None

    def health(self) -> tuple[bool, str]:
        if self.last_error:
            return False, self.last_error
        if self.active:
            output = self.output_device_name or "uruchamianie WASAPI loopback"
            return True, f"input PCM + output {output}"
        return True, "gotowy"

    def feed_microphone_audio(self, data: bytes) -> None:
        loop = self._loop
        if not self.active or loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._enqueue_input, bytes(data))
        except RuntimeError:
            return

    def _enqueue_input(self, data: bytes) -> None:
        queue = self._input_queue
        if not self.active or queue is None:
            return
        if queue.full():
            queue.get_nowait()
            LOGGER.warning("Dropping oldest microphone archive block")
        queue.put_nowait(data)

    async def _archive_input_loop(self) -> None:
        queue = self._input_queue
        assert queue is not None
        bytes_per_second = self.input_sample_rate * 2
        target_bytes = bytes_per_second * self.chunk_seconds
        buffer = bytearray()
        started_at: datetime | None = None
        while True:
            data = await queue.get()
            if data is None:
                break
            if started_at is None:
                started_at = datetime.now(UTC)
            buffer.extend(data)
            while len(buffer) >= target_bytes:
                chunk = bytes(buffer[:target_bytes])
                del buffer[:target_bytes]
                assert started_at is not None
                ended_at = started_at + timedelta(seconds=self.chunk_seconds)
                await self._save_input_chunk(chunk, started_at, ended_at)
                started_at = ended_at
        if buffer and started_at is not None:
            duration = len(buffer) / bytes_per_second
            ended_at = started_at + timedelta(seconds=duration)
            await self._save_input_chunk(bytes(buffer), started_at, ended_at)

    async def _save_input_chunk(
        self,
        pcm: bytes,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        audio_dir = self._audio_dir
        assert audio_dir is not None
        name = (
            f"input-microphone-{started_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{uuid4().hex[:8]}.wav"
        )
        path = audio_dir / name
        await asyncio.to_thread(
            _write_wave_bytes,
            path,
            pcm,
            self.input_sample_rate,
            1,
        )
        await self.on_chunk(
            CapturedAudioChunk(
                chunk_id=f"direct-input:{name}",
                channel="input",
                device_name=self.input_device_name,
                file_path=path,
                started_at=started_at,
                ended_at=ended_at,
            )
        )

    async def _run_output_recorder(self) -> None:
        loop = self._loop
        assert loop is not None
        try:
            await asyncio.to_thread(self._record_output_loop, loop)
        except Exception as exc:
            self.last_error = f"WASAPI loopback: {exc}"
            LOGGER.exception("Default speaker loopback recording failed")

    def _record_output_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("brak domyślnego urządzenia output")
        microphone = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True,
        )
        if microphone is None:
            raise RuntimeError(f"brak loopback dla {speaker.name}")
        self.output_device_name = str(speaker.name)
        target_frames = self.output_sample_rate * self.chunk_seconds
        block_frames = min(4096, target_frames)
        frames: list[np.ndarray] = []
        frame_count = 0
        started_at = datetime.now(UTC)
        with microphone.recorder(
            samplerate=self.output_sample_rate,
            channels=2,
            blocksize=block_frames,
        ) as recorder:
            while not self._output_stop.is_set():
                requested = min(block_frames, target_frames - frame_count)
                data = recorder.record(numframes=requested)
                if data.size == 0:
                    continue
                frames.append(data)
                frame_count += len(data)
                if frame_count >= target_frames:
                    ended_at = started_at + timedelta(
                        seconds=frame_count / self.output_sample_rate
                    )
                    chunk = self._save_output_chunk(frames, started_at, ended_at)
                    loop.call_soon_threadsafe(self._enqueue_output, chunk)
                    frames = []
                    frame_count = 0
                    started_at = ended_at
        if frames and frame_count:
            ended_at = started_at + timedelta(
                seconds=frame_count / self.output_sample_rate
            )
            chunk = self._save_output_chunk(frames, started_at, ended_at)
            loop.call_soon_threadsafe(self._enqueue_output, chunk)

    def _save_output_chunk(
        self,
        frames: list[np.ndarray],
        started_at: datetime,
        ended_at: datetime,
    ) -> CapturedAudioChunk:
        audio_dir = self._audio_dir
        assert audio_dir is not None
        name = (
            f"output-loopback-{started_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{uuid4().hex[:8]}.wav"
        )
        path = audio_dir / name
        samples = np.concatenate(frames, axis=0)
        pcm = (
            np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max
        ).astype("<i2", copy=False)
        _write_wave_bytes(
            path,
            pcm.tobytes(),
            self.output_sample_rate,
            int(pcm.shape[1]),
        )
        return CapturedAudioChunk(
            chunk_id=f"direct-output:{name}",
            channel="output",
            device_name=self.output_device_name,
            file_path=path,
            started_at=started_at,
            ended_at=ended_at,
        )

    def _enqueue_output(self, chunk: CapturedAudioChunk) -> None:
        queue = self._output_queue
        if queue is not None:
            queue.put_nowait(chunk)

    async def _consume_output_chunks(self) -> None:
        queue = self._output_queue
        assert queue is not None
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            try:
                await self.on_chunk(chunk)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Could not persist captured output chunk")


def _write_wave_bytes(
    path: Path,
    pcm: bytes,
    sample_rate: int,
    channels: int,
) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
