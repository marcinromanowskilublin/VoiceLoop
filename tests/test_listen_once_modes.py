from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from voiceloop.app import LISTEN_ONCE_MODES, app


def test_cursor_mode_is_registered_with_expected_prompt() -> None:
    assert "cursor" in LISTEN_ONCE_MODES
    prompt, prefix = LISTEN_ONCE_MODES["cursor"]
    assert prompt == "Gdzie przesunąć kursor lub co zrobić w aktywnym folderze?"
    assert prefix == ""


@pytest.mark.asyncio
async def test_listen_once_cursor_mode_starts_deepgram_without_prefix() -> None:
    missing = object()
    previous_services = getattr(app.state, "services", missing)

    deepgram = SimpleNamespace(stop=AsyncMock(), start_once=AsyncMock())
    tts = SimpleNamespace(speak=AsyncMock())
    settings = SimpleNamespace(conversation_cooldown_ms=0)
    app.state.services = SimpleNamespace(
        token="expected-local-token",
        deepgram=deepgram,
        tts=tts,
        settings=settings,
    )
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/listening/once",
                params={"mode": "cursor"},
                headers={"X-VoiceLoop-Token": "expected-local-token"},
            )
    finally:
        if previous_services is missing:
            del app.state.services
        else:
            app.state.services = previous_services

    assert response.status_code == 200
    assert response.json() == {"status": "listening_once", "mode": "cursor"}
    deepgram.stop.assert_awaited_once()
    tts.speak.assert_awaited_once_with(
        "Gdzie przesunąć kursor lub co zrobić w aktywnym folderze?"
    )
    deepgram.start_once.assert_awaited_once_with(prefix="")


@pytest.mark.asyncio
async def test_listen_once_rejects_unknown_mode() -> None:
    missing = object()
    previous_services = getattr(app.state, "services", missing)
    app.state.services = SimpleNamespace(token="expected-local-token")
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/listening/once",
                params={"mode": "nieznany"},
                headers={"X-VoiceLoop-Token": "expected-local-token"},
            )
    finally:
        if previous_services is missing:
            del app.state.services
        else:
            app.state.services = previous_services

    assert response.status_code == 422
