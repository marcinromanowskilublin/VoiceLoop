from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from voiceloop.memory import MemoryStore
from voiceloop.screenpipe import (
    ScreenpipeAudioChunk,
    ScreenpipeClient,
    ScreenpipeContext,
    ScreenpipeMeeting,
)
from voiceloop.screenpipe_deepgram import (
    ScreenpipeMeetingTranscriber,
    envelope_from_deepgram_payload,
)
from voiceloop.settings import Settings


class FakeFileTranscriber:
    available = True

    def __init__(self, text: str = "Rozmowa testowa.") -> None:
        self.transcribe = AsyncMock(return_value=text)


def test_file_payload_preserves_words_confidence_time_and_speakers() -> None:
    envelope = envelope_from_deepgram_payload(
        {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Otwórz VoiceLoop.",
                                "confidence": 0.90,
                                "words": [
                                    {
                                        "word": "otwórz",
                                        "punctuated_word": "Otwórz",
                                        "start": 0.1,
                                        "end": 0.5,
                                        "confidence": 0.95,
                                        "speaker": 0,
                                    },
                                    {
                                        "word": "voiceloop",
                                        "punctuated_word": "VoiceLoop.",
                                        "start": 0.6,
                                        "end": 1.2,
                                        "confidence": 0.85,
                                        "speaker": 0,
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        },
        language="pl",
        model="nova-3",
    )

    assert envelope is not None
    assert envelope.confidence_mean == pytest.approx(0.90)
    assert envelope.confidence_min == pytest.approx(0.85)
    assert envelope.started_at_seconds == pytest.approx(0.1)
    assert envelope.ended_at_seconds == pytest.approx(1.2)
    assert envelope.speaker_ids == (0,)


def meeting(*, title: str = "Google Meet") -> ScreenpipeMeeting:
    now = datetime.now(UTC)
    return ScreenpipeMeeting(
        id=7,
        meeting_start=(now - timedelta(minutes=5)).isoformat(),
        meeting_end=(now - timedelta(minutes=2)).isoformat(),
        meeting_app="chrome.exe",
        title=title,
        detection_source="automatic",
    )


@pytest.mark.asyncio
async def test_first_run_arms_without_backfilling(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.meetings = AsyncMock()
    worker = ScreenpipeMeetingTranscriber(
        settings,
        screenpipe,
        memory,
        FakeFileTranscriber(),
    )

    processed = await worker.process_once()

    assert processed == 0
    assert await memory.get_state("screenpipe_deepgram_enabled_at") is not None
    screenpipe.meetings.assert_not_awaited()


@pytest.mark.asyncio
async def test_youtube_context_is_never_sent_to_deepgram(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.has_youtube_context = AsyncMock(return_value=True)
    screenpipe.contexts_between = AsyncMock(
        return_value=[
            ScreenpipeContext(
                app_name="chrome.exe",
                window_name="Wykład - YouTube",
                timestamp=datetime.now(UTC).isoformat(),
                browser_url="https://www.youtube.com/watch?v=abc",
            )
        ]
    )
    screenpipe.audio_chunks = AsyncMock()
    transcriber = FakeFileTranscriber()
    worker = ScreenpipeMeetingTranscriber(settings, screenpipe, memory, transcriber)

    assert await worker._process_meeting(meeting()) is True

    assert await memory.has_screenpipe_meeting_job(7) is True
    screenpipe.audio_chunks.assert_not_awaited()
    transcriber.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_meeting_audio_is_transcribed_once(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    audio_path = tmp_path / "meeting.mp4"
    audio_path.write_bytes(b"test audio")
    screenpipe = ScreenpipeClient(settings)
    screenpipe.has_youtube_context = AsyncMock(return_value=False)
    screenpipe.contexts_between = AsyncMock(
        return_value=[
            ScreenpipeContext(
                app_name="chrome.exe",
                window_name="Google Meet",
                timestamp=datetime.now(UTC).isoformat(),
                browser_url="https://meet.google.com/abc-defg-hij",
            )
        ]
    )
    screenpipe.audio_chunks = AsyncMock(
        return_value=[
            ScreenpipeAudioChunk(
                chunk_id="chunk-1",
                file_path=audio_path,
                device_name="Microphone",
                device_type="Input",
                start_time=datetime.now(UTC).isoformat(),
                end_time=datetime.now(UTC).isoformat(),
                text="",
            )
        ]
    )
    transcriber = FakeFileTranscriber()
    worker = ScreenpipeMeetingTranscriber(settings, screenpipe, memory, transcriber)
    worker.data_root = tmp_path.resolve()

    assert await worker._process_meeting(meeting()) is True
    assert await worker._process_meeting(meeting()) is True

    transcriber.transcribe.assert_awaited_once_with(audio_path.resolve())
    assert await memory.has_screenpipe_transcript("chunk-1") is True
