import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from voiceloop.app import (
    RetentionWorkerState,
    _configure_retention_rollout,
    _run_private_data_retention_worker,
    app,
)
from voiceloop.memory import MemoryStore
from voiceloop.settings import Settings


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
            base_url="http://localhost",
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
        base_url="http://localhost",
    ) as client:
        response = await client.get(
            "/api/v1/session",
            headers={"Host": "attacker.example"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_session_rejects_production_testserver_host() -> None:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        response = await client.get(
            "/api/v1/session",
            headers={
                "Host": "testserver:8765",
                "Origin": "http://testserver:8765",
            },
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_session_rejects_cross_origin_even_for_local_host() -> None:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        response = await client.get(
            "/api/v1/session",
            headers={"Origin": "https://attacker.example"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_session_rejects_different_local_origin_port() -> None:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        response = await client.get(
            "/api/v1/session",
            headers={"Origin": "http://localhost:9999"},
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
            base_url="http://localhost",
        ) as client:
            response = await client.get(
                "/api/v1/session",
                headers={"Origin": "http://localhost"},
            )
        assert response.status_code == 200
        assert response.json()["token"] == "expected-local-token"
    finally:
        if previous_services is missing:
            del app.state.services
        else:
            app.state.services = previous_services


@pytest.mark.asyncio
async def test_retention_worker_recovers_after_transient_failure(tmp_path) -> None:
    memory = SimpleNamespace(
        retention_policy_effective_at=datetime.now(UTC) - timedelta(days=1),
        prune_expired_private_data=AsyncMock(
            side_effect=[
                RuntimeError("temporary"),
                {"commands": 0, "dry_run": 0},
            ]
        ),
        reconcile_orphaned_meeting_audio=AsyncMock(
            return_value={
                "orphan_candidates": 0,
                "orphan_removed": 0,
                "orphan_failed": 0,
            }
        ),
    )
    qdrant = SimpleNamespace(prune_expired=AsyncMock(return_value=0))
    services = SimpleNamespace(
        memory=memory,
        qdrant=qdrant,
        qdrant_shadow=None,
        retention=RetentionWorkerState(
            rollout_started_at=datetime.now(UTC) - timedelta(days=1)
        ),
    )
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        private_data_rollout_grace_days=0,
    )
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await _run_private_data_retention_worker(
            services,  # type: ignore[arg-type]
            settings,
            sleep=sleep,
        )

    assert memory.prune_expired_private_data.await_count == 2
    memory.reconcile_orphaned_meeting_audio.assert_awaited_once()
    qdrant.prune_expired.assert_awaited_once()
    assert services.retention.running is False
    assert services.retention.consecutive_failures == 0
    assert services.retention.last_error_type is None
    assert services.retention.last_success_at is not None


@pytest.mark.asyncio
async def test_destructive_retention_grace_starts_at_each_opt_in(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    initial = datetime(2026, 10, 1, tzinfo=UTC)
    enabled_at = datetime(2026, 10, 11, tzinfo=UTC)
    reenabled_at = datetime(2026, 10, 21, tzinfo=UTC)
    disabled = Settings(
        voiceloop_data_dir=str(tmp_path),
        private_data_prune_enabled=False,
    )
    destructive = Settings(
        voiceloop_data_dir=str(tmp_path),
        private_data_prune_enabled=True,
        private_data_prune_dry_run=False,
    )

    await _configure_retention_rollout(memory, disabled, now=initial)
    first_anchor = await _configure_retention_rollout(
        memory,
        destructive,
        now=enabled_at,
    )
    restart_anchor = await _configure_retention_rollout(
        memory,
        destructive,
        now=enabled_at + timedelta(days=1),
    )
    await _configure_retention_rollout(
        memory,
        disabled,
        now=reenabled_at - timedelta(days=1),
    )
    second_anchor = await _configure_retention_rollout(
        memory,
        destructive,
        now=reenabled_at,
    )

    assert first_anchor == enabled_at
    assert restart_anchor == enabled_at
    assert second_anchor == reenabled_at
