import asyncio
import wave
from unittest.mock import AsyncMock

from voiceloop.audio_capture import MeetingChannelCapture
from voiceloop.settings import Settings


async def test_deepgram_microphone_pcm_is_archived_as_wave(tmp_path) -> None:
    callback = AsyncMock()
    capture = MeetingChannelCapture(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            sample_rate=100,
            meeting_recording_audio_chunk_seconds=5,
        ),
        on_chunk=callback,
    )
    capture._run_output_recorder = AsyncMock()  # type: ignore[method-assign]
    audio_dir = tmp_path / "audio"
    await capture.start(session_id="meeting-test", audio_dir=audio_dir)

    capture.feed_microphone_audio(b"\x01\x00" * 500)
    await asyncio.sleep(0)
    await capture.stop()

    callback.assert_awaited_once()
    chunk = callback.await_args.args[0]
    assert chunk.channel == "input"
    assert chunk.file_path.is_file()
    with wave.open(str(chunk.file_path), "rb") as recording:
        assert recording.getframerate() == 100
        assert recording.getnchannels() == 1
        assert recording.getnframes() == 500
