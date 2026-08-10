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
