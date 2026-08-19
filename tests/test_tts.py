from unittest.mock import AsyncMock

import pytest

from voiceloop.tts import WindowsTTS


@pytest.mark.asyncio
async def test_tts_uses_azure_when_enabled_and_configured() -> None:
    tts = WindowsTTS(
        azure_enabled=True,
        azure_key="key",
        azure_region="westeurope",
    )
    tts._speak_azure = AsyncMock(return_value=None)  # type: ignore[method-assign]
    tts._speak_windows = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await tts.speak("test")

    tts._speak_azure.assert_awaited_once()  # type: ignore[attr-defined]
    tts._speak_windows.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_tts_falls_back_to_windows_when_azure_fails() -> None:
    tts = WindowsTTS(
        azure_enabled=True,
        azure_key="key",
        azure_region="westeurope",
    )
    tts._speak_azure = AsyncMock(side_effect=RuntimeError("azure down"))  # type: ignore[method-assign]
    tts._speak_windows = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await tts.speak("test")

    tts._speak_azure.assert_awaited_once()  # type: ignore[attr-defined]
    tts._speak_windows.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_azure_prefers_sdk_over_rest() -> None:
    tts = WindowsTTS(
        azure_enabled=True,
        azure_key="key",
        azure_region="westeurope",
    )
    tts._speak_azure_sdk = AsyncMock(return_value=None)  # type: ignore[method-assign]
    tts._speak_azure_rest = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await tts._speak_azure("test")

    tts._speak_azure_sdk.assert_awaited_once_with("test")  # type: ignore[attr-defined]
    tts._speak_azure_rest.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_azure_rest_is_used_when_sdk_fails() -> None:
    tts = WindowsTTS(
        azure_enabled=True,
        azure_key="key",
        azure_region="westeurope",
    )
    tts._speak_azure_sdk = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("sdk down")
    )
    tts._speak_azure_rest = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await tts._speak_azure("test")

    tts._speak_azure_sdk.assert_awaited_once_with("test")  # type: ignore[attr-defined]
    tts._speak_azure_rest.assert_awaited_once_with("test")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stop_uses_native_sdk_cancellation() -> None:
    class Completed:
        @staticmethod
        def get() -> None:
            return None

    class Synthesizer:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop_speaking_async(self) -> Completed:
            self.stop_calls += 1
            return Completed()

    tts = WindowsTTS()
    synthesizer = Synthesizer()
    tts._azure_synthesizer = synthesizer
    tts._azure_speaking = True

    await tts.stop()

    assert synthesizer.stop_calls == 1
    assert tts._azure_speaking is False


def test_azure_ssml_escapes_text_and_voice() -> None:
    tts = WindowsTTS(azure_voice="pl-PL-ZofiaNeural")

    ssml = tts._azure_ssml("Ala < Ola & Jan")

    assert "name='pl-PL-ZofiaNeural'" in ssml
    assert "rate='-20%'" in ssml
    assert "pitch='-5%'" in ssml
    assert "Ala &lt; Ola &amp; Jan" in ssml


def test_azure_ssml_does_not_silently_slice_long_text() -> None:
    tts = WindowsTTS()
    text = ("pełne zdanie. " * 300) + "Koniec odpowiedzi."

    ssml = tts._azure_ssml(text)

    assert "Koniec odpowiedzi." in ssml


def test_azure_timeouts_include_estimated_full_playback_duration() -> None:
    tts = WindowsTTS(
        azure_timeout_seconds=20.0,
        speaking_rate_percent=-20,
    )
    text = "x" * 1200

    sdk_timeout = tts._speech_timeout_seconds(text, rest_fallback=False)
    rest_timeout = tts._speech_timeout_seconds(text, rest_fallback=True)

    assert sdk_timeout >= 200.0
    assert rest_timeout >= sdk_timeout + 10.0


def test_tts_rate_percent_maps_to_azure_and_windows() -> None:
    tts = WindowsTTS(speaking_rate_percent=-20, speaking_pitch_percent=-5)

    assert tts._azure_rate() == "-20%"
    assert tts._azure_pitch() == "-5%"
    assert tts._windows_rate() == -2
