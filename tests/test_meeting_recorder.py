import asyncio
import os
import stat
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voiceloop.meeting_recorder import MeetingRecorder, _secure_copy_audio_file
from voiceloop.memory import MemoryStore
from voiceloop.models import (
    TranscriptEnvelopeV1,
    TranscriptWordV1,
    normalize_transcript_text,
)
from voiceloop.screenpipe import ScreenpipeAudioChunk
from voiceloop.settings import Settings


class FakeChannelCapture:
    def __init__(self) -> None:
        self.active = False

    async def start(self, *, session_id: str, audio_dir) -> None:
        self.active = True

    async def stop(self) -> None:
        self.active = False

    def health(self) -> tuple[bool, str]:
        return True, "fake capture"

    def feed_microphone_audio(self, data: bytes) -> None:
        return None


async def _recorder(tmp_path, *, envelope=None, archive_audio=False):
    settings = Settings(
        voiceloop_data_dir=str(tmp_path / "data"),
        meeting_recording_poll_seconds=60,
        meeting_recording_finalize_seconds=5,
        meeting_recording_archive_audio=archive_audio,
    )
    memory = MemoryStore(settings.data_dir / "voice.db")
    await memory.initialize()
    screenpipe = SimpleNamespace(audio_chunks=AsyncMock(return_value=[]))
    transcriber = SimpleNamespace(transcribe_envelope=AsyncMock(return_value=envelope))
    events = SimpleNamespace(publish=AsyncMock())
    recorder = MeetingRecorder(
        settings=settings,
        memory=memory,
        events=events,
        screenpipe=screenpipe,
        file_transcriber=transcriber,
        channel_capture=FakeChannelCapture(),  # type: ignore[arg-type]
    )
    return recorder, memory, screenpipe, transcriber, events


async def _cancel_poll(recorder: MeetingRecorder) -> None:
    task = recorder._poll_task
    recorder._poll_task = None
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_raw_audio_archive_is_off_by_default(tmp_path) -> None:
    recorder, _, _, _, _ = await _recorder(tmp_path)

    assert recorder.archive_audio is False


async def test_live_microphone_transcript_is_durable_and_labeled_as_user(tmp_path) -> None:
    recorder, memory, _, _, events = await _recorder(tmp_path)
    session = await recorder.start(title="Rozmowa testowa")
    await _cancel_poll(recorder)
    envelope = TranscriptEnvelopeV1(
        raw_text="Dzień dobry.",
        normalized_text="Dzień dobry.",
        confidence_mean=0.96,
        confidence_min=0.96,
        words=(
            TranscriptWordV1(
                word="dzień",
                punctuated_word="Dzień",
                start_seconds=0.1,
                end_seconds=0.4,
                confidence=0.96,
                speaker_id=0,
            ),
            TranscriptWordV1(
                word="dobry",
                punctuated_word="dobry.",
                start_seconds=0.5,
                end_seconds=0.8,
                confidence=0.96,
                speaker_id=0,
            ),
        ),
        started_at_seconds=0.1,
        ended_at_seconds=0.8,
        speaker_ids=(0,),
        model="nova-3",
    )

    assert await recorder.record_live(envelope) is True
    assert await recorder.record_live(envelope) is False

    segments = await memory.list_meeting_transcript_segments(session["session_id"])
    assert len(segments) == 1
    assert segments[0].channel == "input"
    assert segments[0].speaker_label == "Ty"
    assert segments[0].text == "Dzień dobry."
    assert segments[0].transcript == envelope
    assert any(
        call.args[0] == "meeting.transcript.final"
        for call in events.publish.await_args_list
    )
    await recorder.close()


async def test_output_audio_is_archived_and_split_by_deepgram_speaker(tmp_path) -> None:
    text = "Dzień dobry. Proszę usiąść."
    envelope = TranscriptEnvelopeV1(
        raw_text=text,
        normalized_text=normalize_transcript_text(text),
        words=(
            TranscriptWordV1(
                word="dzień",
                punctuated_word="Dzień",
                start_seconds=0.0,
                end_seconds=0.3,
                speaker_id=0,
            ),
            TranscriptWordV1(
                word="dobry",
                punctuated_word="dobry.",
                start_seconds=0.3,
                end_seconds=0.7,
                speaker_id=0,
            ),
            TranscriptWordV1(
                word="proszę",
                punctuated_word="Proszę",
                start_seconds=0.8,
                end_seconds=1.1,
                speaker_id=1,
            ),
            TranscriptWordV1(
                word="usiąść",
                punctuated_word="usiąść.",
                start_seconds=1.1,
                end_seconds=1.5,
                speaker_id=1,
            ),
        ),
        started_at_seconds=0.0,
        ended_at_seconds=1.5,
        speaker_ids=(0, 1),
        model="nova-3",
    )
    recorder, memory, _, transcriber, _ = await _recorder(
        tmp_path,
        envelope=envelope,
        archive_audio=True,
    )
    screenpipe_root = tmp_path / "screenpipe" / "data"
    screenpipe_root.mkdir(parents=True)
    audio_path = screenpipe_root / "speaker-output.mp4"
    audio_path.write_bytes(b"complete audio")
    recorder.screenpipe_data_root = screenpipe_root.resolve()
    session_payload = await recorder.start()
    await _cancel_poll(recorder)
    session = recorder._active
    assert session is not None
    now = datetime.now(UTC)
    chunk = ScreenpipeAudioChunk(
        chunk_id="output-1",
        file_path=audio_path,
        device_name="Głośniki (output)",
        device_type="Output",
        start_time=now.isoformat(),
        end_time=now.isoformat(),
        text="",
    )

    assert await recorder._process_chunk(session, chunk) is True

    segments = await memory.list_meeting_transcript_segments(
        session_payload["session_id"]
    )
    assert [(item.channel, item.speaker_label, item.text) for item in segments] == [
        ("output", "Rozmówca", "Dzień dobry."),
        ("output", "Rozmówca 2", "Proszę usiąść."),
    ]
    audio = await memory.list_meeting_audio_files(session_payload["session_id"])
    assert len(audio) == 1
    assert audio[0].channel == "output"
    assert audio[0].archived_path != str(audio_path)
    assert (tmp_path / "data" / "meetings" / session.session_id / "audio").is_dir()
    assert transcriber.transcribe_envelope.await_count == 1
    await recorder.close()


async def test_failed_database_write_removes_new_archive_copy(tmp_path) -> None:
    recorder, memory, _, _, _ = await _recorder(tmp_path, archive_audio=True)
    screenpipe_root = tmp_path / "screenpipe" / "data"
    screenpipe_root.mkdir(parents=True)
    source = screenpipe_root / "speaker-output.wav"
    source.write_bytes(b"audio")
    recorder.screenpipe_data_root = screenpipe_root.resolve()
    await recorder.start()
    await _cancel_poll(recorder)
    session = recorder._active
    assert session is not None
    memory.save_meeting_audio_file = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("database unavailable")
    )
    now = datetime.now(UTC)
    chunk = ScreenpipeAudioChunk(
        chunk_id="output-failure",
        file_path=source,
        device_name="Głośniki",
        device_type="Output",
        start_time=now.isoformat(),
        end_time=now.isoformat(),
        text="",
    )

    try:
        await recorder._process_chunk(session, chunk)
    except RuntimeError:
        pass
    else:
        raise AssertionError("database failure should be propagated")

    archive_dir = tmp_path / "data" / "meetings" / session.session_id / "audio"
    assert list(archive_dir.iterdir()) == []
    assert source.exists()
    await recorder.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor race regression")
def test_archive_copy_reads_pinned_source_during_path_swap(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "screenpipe"
    target_root = tmp_path / "meetings"
    target_dir = target_root / "session" / "audio"
    source_root.mkdir()
    target_dir.mkdir(parents=True)
    source = source_root / "chunk.wav"
    original = source_root / "chunk-original.wav"
    outside = tmp_path / "outside.wav"
    target = target_dir / "archive.wav"
    source.write_bytes(b"safe audio")
    outside.write_bytes(b"private outside")
    real_fstat = os.fstat
    swapped = False

    def swap_after_source_open(descriptor):
        nonlocal swapped
        result = real_fstat(descriptor)
        if not swapped and stat.S_ISREG(result.st_mode):
            source.rename(original)
            source.symlink_to(outside)
            swapped = True
        return result

    monkeypatch.setattr(
        "voiceloop.meeting_recorder.os.fstat",
        swap_after_source_open,
    )

    created = _secure_copy_audio_file(
        source,
        source_root=source_root,
        target=target,
        target_root=target_root,
    )

    assert created is True
    assert swapped is True
    assert target.read_bytes() == b"safe audio"


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor race regression")
def test_archive_copy_never_overwrites_symlink_target(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "screenpipe"
    target_root = tmp_path / "meetings"
    target_dir = target_root / "session" / "audio"
    source_root.mkdir()
    target_dir.mkdir(parents=True)
    source = source_root / "chunk.wav"
    outside = tmp_path / "outside.wav"
    target = target_dir / "archive.wav"
    source.write_bytes(b"safe audio")
    outside.write_bytes(b"private outside")
    real_open = os.open
    planted = False

    def plant_symlink(path, flags, *args, **kwargs):
        nonlocal planted
        if (
            not planted
            and path == target.name
            and flags & os.O_CREAT
            and kwargs.get("dir_fd") is not None
        ):
            target.symlink_to(outside)
            planted = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("voiceloop.meeting_recorder.os.open", plant_symlink)

    with pytest.raises(OSError):
        _secure_copy_audio_file(
            source,
            source_root=source_root,
            target=target,
            target_root=target_root,
        )

    assert planted is True
    assert target.is_symlink()
    assert outside.read_bytes() == b"private outside"
