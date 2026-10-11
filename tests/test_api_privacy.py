from types import SimpleNamespace

import httpx
import pytest

from voiceloop.app import app


@pytest.mark.asyncio
async def test_health_and_events_require_local_token() -> None:
    missing = object()
    previous_services = getattr(app.state, "services", missing)
    app.state.services = SimpleNamespace(
        token="expected-local-token",
        settings=SimpleNamespace(llm_primary="local"),
    )
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


@pytest.mark.asyncio
async def test_session_rejects_non_local_host_header() -> None:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/v1/session",
            headers={"Host": "attacker.example"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_session_rejects_cross_origin_even_for_local_host() -> None:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/v1/session",
            headers={"Origin": "https://attacker.example"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_session_accepts_same_local_origin() -> None:
    missing = object()
    previous_services = getattr(app.state, "services", missing)
    app.state.services = SimpleNamespace(
        token="expected-local-token",
        settings=SimpleNamespace(llm_primary="local"),
    )
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/api/v1/session",
                headers={"Origin": "http://testserver"},
            )
        assert response.status_code == 200
        assert response.json()["token"] == "expected-local-token"
    finally:
        if previous_services is missing:
            del app.state.services
        else:
            app.state.services = previous_services
