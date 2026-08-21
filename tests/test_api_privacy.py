import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from voiceloop.app import app
from voiceloop.corpus.cli import build_parser
from voiceloop.settings import Settings


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


@pytest.mark.asyncio
async def test_untrusted_host_is_rejected() -> None:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://evil.example",
    ) as client:
        response = await client.get("/api/v1/session")

    assert response.status_code == 400


def test_sensitive_context_integrations_are_opt_in() -> None:
    settings = Settings(_env_file=None)

    assert settings.behavior_digest_enabled is False
    assert settings.screenpipe_enabled is False
    assert settings.screenpipe_deepgram_enabled is False
    assert settings.meeting_recording_archive_audio is False
    assert settings.screenpipe_vector_memory_enabled is False


def test_screenpipe_startup_preserves_user_privacy_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "start-screenpipe.ps1").read_text(encoding="utf-8")

    assert "Set-UnfilteredCaptureSettings" not in script
    assert "--use-pii-removal=false" not in script
    assert "--ignore-incognito-windows=false" not in script
    assert "--capture-on-keystroke" not in script
    assert "--capture-on-clipboard" not in script
    assert "'--use-pii-removal=true'" in script


def test_private_corpus_sources_require_explicit_paths() -> None:
    args = build_parser().parse_args(["inventory"])

    assert args.audio is None
    assert args.cursor_root is None


def test_generated_local_token_is_owner_only_on_posix(tmp_path: Path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path), _env_file=None)

    token = settings.ensure_local_token()

    assert token
    if os.name == "posix":
        assert (tmp_path / "voiceloop.token").stat().st_mode & 0o777 == 0o600
