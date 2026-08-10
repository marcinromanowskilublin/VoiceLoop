from unittest.mock import AsyncMock

from voiceloop.deepgram import DeepgramListener
from voiceloop.events import EventBus
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

    assert listener.health() == (True, "connected (nova-test, en)")


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
