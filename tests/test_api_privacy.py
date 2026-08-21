from types import SimpleNamespace

import httpx
import pytest

from voiceloop.app import app


@pytest.mark.asyncio
async def test_health_and_events_require_local_token() -> None:
    missing = object()
    previous_services = getattr(app.state, "services", missing)
    app.state.services = SimpleNamespace(token="expected-local-token")
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for path in ("/api/v1/health", "/api/v1/events"):
                response = await client.get(path)
                assert response.status_code == 401

                response = await client.get(
                    path,
                    headers={"X-VoiceLoop-Token": "wrong-token"},
                )
                assert response.status_code == 401
    finally:
        if previous_services is missing:
            del app.state.services
        else:
            app.state.services = previous_services
