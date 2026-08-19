import asyncio
from unittest.mock import AsyncMock

import pytest

from voiceloop.deepgram import DeepgramListener
from voiceloop.events import EventBus
from voiceloop.models import TranscriptEnvelopeV1
from voiceloop.settings import Settings


def test_health_uses_configured_language(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        deepgram_model="nova-test",
        deepgram_language="en",
    )
    listener = DeepgramListener(
        settings=settings,
        events=EventBus(),
        on_final=AsyncMock(),
    )
    listener.connected = True

    assert listener.health() == (True, "connected (nova-test, en, diarization)")


def test_streaming_url_enables_current_diarization_model(tmp_path) -> None:
    listener = DeepgramListener(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            deepgram_diarization_enabled=True,
            deepgram_diarization_model="latest",
        ),
        events=EventBus(),
        on_final=AsyncMock(),
    )

    assert "diarize_model=latest" in listener._url()
    assert "diarize=true" not in listener._url()


async def test_start_once_sets_prefix_and_one_shot_mode(tmp_path) -> None:
    listener = DeepgramListener(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            deepgram_api_key="test-key",
        ),
        events=EventBus(),
        on_final=AsyncMock(),
    )
    listener._run_forever = AsyncMock()  # type: ignore[method-assign]

    await listener.start_once(prefix="Zapamiętaj", timeout_seconds=20)

    assert listener.running is True
    assert listener._one_shot is True
    assert listener._one_shot_prefix == "Zapamiętaj "
    assert listener._one_shot_timeout_seconds == 20

    await listener.stop()


async def test_concurrent_start_once_keeps_only_one_listener_task(tmp_path) -> None:
    tasks: list[asyncio.Task[None]] = []

    async def run_forever() -> None:
        current = asyncio.current_task()
        assert current is not None
        tasks.append(current)
        await asyncio.Event().wait()

    listener = DeepgramListener(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            deepgram_api_key="test-key",
        ),
        events=EventBus(),
        on_final=AsyncMock(),
    )
    listener._run_forever = run_forever  # type: ignore[method-assign]

    await asyncio.gather(
        listener.start_once(timeout_seconds=20),
        listener.start_once(timeout_seconds=30),
    )
    await asyncio.sleep(0)

    assert len([task for task in tasks if not task.done()]) == 1
    assert listener._task in tasks
    assert listener._task.done() is False
    assert listener._one_shot_timeout_seconds == 30

    await listener.stop()


async def test_start_replaces_active_one_shot_with_continuous_listener(tmp_path) -> None:
    tasks: list[asyncio.Task[None]] = []

    async def run_forever() -> None:
        current = asyncio.current_task()
        assert current is not None
        tasks.append(current)
        await asyncio.Event().wait()

    listener = DeepgramListener(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            deepgram_api_key="test-key",
        ),
        events=EventBus(),
        on_final=AsyncMock(),
    )
    listener._run_forever = run_forever  # type: ignore[method-assign]

    await listener.start_once(timeout_seconds=20)
    first_task = listener._task
    await asyncio.sleep(0)
    await listener.start()
    await asyncio.sleep(0)

    assert first_task is not None and first_task.done()
    assert listener._task is not first_task
    assert listener._task in tasks
    assert listener.running is True
    assert listener._one_shot is False

    await listener.stop()


def test_final_callback_runs_without_blocking_receiver(tmp_path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def on_final(
            text: str,
            *,
            confidence: float | None = None,
            speaker_ids: tuple[int, ...] = (),
        ) -> None:
            assert text == "stop"
            assert confidence is None
            assert speaker_ids == ()
            started.set()
            await release.wait()

        listener = DeepgramListener(
            settings=Settings(voiceloop_data_dir=str(tmp_path)),
            events=EventBus(),
            on_final=on_final,
        )

        callback = listener._dispatch_final("stop")
        await started.wait()

        assert callback.done() is False
        assert callback in listener._callback_tasks

        release.set()
        await callback
        await asyncio.sleep(0)

        assert listener._callback_tasks == set()

    asyncio.run(scenario())


def test_dispatch_final_passes_confidence_and_speakers(tmp_path) -> None:
    async def scenario() -> None:
        seen: list[tuple[str, float | None, tuple[int, ...]]] = []

        async def on_final(
            text: str,
            *,
            confidence: float | None = None,
            speaker_ids: tuple[int, ...] = (),
        ) -> None:
            seen.append((text, confidence, speaker_ids))

        listener = DeepgramListener(
            settings=Settings(voiceloop_data_dir=str(tmp_path)),
            events=EventBus(),
            on_final=on_final,
        )
        task = listener._dispatch_final(
            "zmien nazwe",
            confidence=0.91,
            speaker_ids=(0, 1),
        )
        await task
        assert seen == [("zmien nazwe", 0.91, (0, 1))]

    asyncio.run(scenario())


def test_final_callbacks_are_bounded_and_serialized(tmp_path) -> None:
    async def scenario() -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        seen: list[str] = []

        async def on_final(
            text: str,
            *,
            confidence: float | None = None,
            speaker_ids: tuple[int, ...] = (),
        ) -> None:
            seen.append(text)
            if text == "pierwszy":
                first_started.set()
                await release_first.wait()

        listener = DeepgramListener(
            settings=Settings(voiceloop_data_dir=str(tmp_path)),
            events=EventBus(),
            on_final=on_final,
        )

        worker = listener._dispatch_final("pierwszy")
        await first_started.wait()
        same_worker = listener._dispatch_final("drugi")
        assert same_worker is worker
        assert seen == ["pierwszy"]

        release_first.set()
        await worker
        assert seen == ["pierwszy", "drugi"]
        assert listener._final_queue.empty()

    asyncio.run(scenario())


def test_priority_final_bypasses_full_normal_callback_queue(tmp_path) -> None:
    async def scenario() -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        stop_seen = asyncio.Event()
        seen: list[str] = []

        async def on_final(
            text: str,
            *,
            confidence: float | None = None,
            speaker_ids: tuple[int, ...] = (),
        ) -> None:
            seen.append(text)
            if text == "zwykły-0":
                first_started.set()
                await release_first.wait()
            if text == "stop":
                stop_seen.set()

        listener = DeepgramListener(
            settings=Settings(voiceloop_data_dir=str(tmp_path)),
            events=EventBus(),
            on_final=on_final,
        )
        listener.set_priority_final_predicate(lambda text: text == "stop")

        normal_worker = listener._dispatch_final("zwykły-0")
        await first_started.wait()
        for index in range(1, 20):
            listener._dispatch_final(f"zwykły-{index}")

        priority_task = listener._dispatch_final("stop")
        assert priority_task is not normal_worker
        await asyncio.wait_for(stop_seen.wait(), timeout=1.0)
        assert "stop" in seen

        release_first.set()
        await asyncio.gather(normal_worker, priority_task)
        assert listener._final_queue.empty()

    asyncio.run(scenario())


def test_stop_cancels_pending_final_callbacks(tmp_path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def on_final(
            text: str,
            *,
            confidence: float | None = None,
            speaker_ids: tuple[int, ...] = (),
        ) -> None:
            started.set()
            await asyncio.Event().wait()

        listener = DeepgramListener(
            settings=Settings(
                voiceloop_data_dir=str(tmp_path),
                deepgram_api_key="test-key",
            ),
            events=EventBus(),
            on_final=on_final,
        )
        listener._run_forever = AsyncMock()  # type: ignore[method-assign]
        await listener.start_once()
        callback = listener._dispatch_final("hello", confidence=0.4)
        await started.wait()
        await listener.stop()
        result = await asyncio.gather(callback, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert listener._callback_tasks == set()

    asyncio.run(scenario())


def test_extract_speaker_ids_from_diarized_words() -> None:
    assert DeepgramListener._extract_speaker_ids(
        {
            "words": [
                {"word": "cześć", "speaker": 1},
                {"word": "hej", "speaker": 0},
                {"word": "ponownie", "speaker": 1},
            ]
        }
    ) == (0, 1)


def test_final_callback_can_receive_versioned_transcript_envelope(tmp_path) -> None:
    async def scenario() -> None:
        seen: list[TranscriptEnvelopeV1] = []

        async def on_final(
            text: str,
            *,
            confidence: float | None = None,
            speaker_ids: tuple[int, ...] = (),
            transcript: TranscriptEnvelopeV1 | None = None,
        ) -> None:
            assert text == "otwórz chrome"
            assert confidence == pytest.approx(0.91)
            assert speaker_ids == (0,)
            assert transcript is not None
            seen.append(transcript)

        listener = DeepgramListener(
            settings=Settings(voiceloop_data_dir=str(tmp_path)),
            events=EventBus(),
            on_final=on_final,
        )
        envelope = TranscriptEnvelopeV1.from_text(
            "otwórz chrome",
            confidence=0.91,
            speaker_ids=(0,),
            model="nova-3",
        )
        await listener._dispatch_final(
            envelope.normalized_text,
            confidence=envelope.confidence_mean,
            speaker_ids=envelope.speaker_ids,
            transcript=envelope,
        )

        assert seen == [envelope]

    asyncio.run(scenario())


def test_extract_transcript_words_preserves_timing_confidence_and_speaker() -> None:
    words = DeepgramListener._extract_transcript_words(
        {
            "words": [
                {
                    "word": "otwórz",
                    "punctuated_word": "Otwórz",
                    "start": 0.2,
                    "end": 0.7,
                    "confidence": 0.93,
                    "speaker": 1,
                }
            ]
        }
    )

    assert len(words) == 1
    assert words[0].word == "otwórz"
    assert words[0].punctuated_word == "Otwórz"
    assert words[0].start_seconds == pytest.approx(0.2)
    assert words[0].end_seconds == pytest.approx(0.7)
    assert words[0].confidence == pytest.approx(0.93)
    assert words[0].speaker_id == 1


async def test_conversation_transport_is_reused_without_second_task(tmp_path) -> None:
    tasks: list[asyncio.Task[None]] = []

    async def run_forever() -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.append(task)
        await asyncio.Event().wait()

    listener = DeepgramListener(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            deepgram_api_key="test-key",
        ),
        events=EventBus(),
        on_final=AsyncMock(),
    )
    listener._run_forever = run_forever  # type: ignore[method-assign]

    await listener.start_conversation()
    first_task = listener._task
    await asyncio.sleep(0)
    await listener.start_conversation()

    assert listener._task is first_task
    assert len(tasks) == 1
    assert listener._conversation_mode is True
    assert listener._one_shot is False
    await listener.stop()


def test_streaming_url_uses_configured_endpointing(tmp_path) -> None:
    listener = DeepgramListener(
        settings=Settings(
            voiceloop_data_dir=str(tmp_path),
            deepgram_endpointing_ms=450,
            deepgram_utterance_end_ms=1500,
        ),
        events=EventBus(),
        on_final=AsyncMock(),
    )

    assert "endpointing=450" in listener._url()
    assert "utterance_end_ms=1500" in listener._url()
