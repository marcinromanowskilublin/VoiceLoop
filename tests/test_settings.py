from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from voiceloop.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / "listener" / ".env.example"


def _example_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def test_env_example_matches_stabilization_defaults(tmp_path) -> None:
    values = _example_values()
    settings = Settings(voiceloop_data_dir=str(tmp_path))

    assert values["GEMINI_MODEL"] == settings.gemini_model
    assert values["VECTOR_MEMORY_MIN_SCORE"] == str(settings.vector_memory_min_score)
    assert values["ROUTING_V2_EXECUTE"] == str(settings.routing_v2_execute).lower()
    assert (
        values["ROUTING_V2_EXECUTE_MIN_SCORE"]
        == f"{settings.routing_v2_execute_min_score:.2f}"
    )
    assert (
        values["ROUTING_V2_EXECUTE_MIN_MARGIN"]
        == f"{settings.routing_v2_execute_min_margin:.2f}"
    )
    assert values["CONVERSATION_IGNORE_MULTI_SPEAKER"] == str(
        settings.conversation_ignore_multi_speaker
    ).lower()
    assert (
        values["STT_MIN_ACTION_CONFIDENCE"]
        == f"{settings.stt_min_action_confidence:.2f}"
    )
    assert values["SCREENPIPE_VECTOR_MEMORY_ENABLED"] == "false"
    assert settings.screenpipe_vector_memory_enabled is False
    assert values["CONTEXT_TIMELINE_RECALL_ENABLED"] == "false"
    assert settings.context_timeline_recall_enabled is False
    assert int(values["CONTEXT_TIMELINE_BUCKET_MINUTES"]) == (
        settings.context_timeline_bucket_minutes
    )
    assert values["MEETING_RECORDING_ARCHIVE_AUDIO"] == "false"
    assert settings.meeting_recording_archive_audio is False
    assert int(values["MEETING_AUDIO_RETENTION_DAYS"]) == 7
    assert int(values["COMMAND_TRANSCRIPT_RETENTION_DAYS"]) == 30
    assert values["PRIVATE_DATA_PRUNE_ENABLED"] == "true"
    assert values["PRIVATE_DATA_PRUNE_DRY_RUN"] == "false"
    assert int(values["PRIVATE_DATA_ROLLOUT_GRACE_DAYS"]) == 7
    assert int(values["MEETING_AUDIO_ORPHAN_GRACE_SECONDS"]) == 300
    assert values["RETENTION_MIGRATION_ENABLED"] == "false"
    assert values["RETENTION_MIGRATION_DRY_RUN"] == "true"
    assert hasattr(settings, "voiceattack_registration_key")


def test_pytest_disables_settings_env_file() -> None:
    assert Settings.model_config.get("env_file") is None


def test_local_token_file_is_private_on_create_and_reuse(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))

    token = settings.ensure_local_token()
    token_path = tmp_path / "voiceloop.token"
    token_path.chmod(0o644)
    reused = settings.ensure_local_token()

    assert reused == token
    if os.name != "nt":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_local_token_reuse_rejects_symlink_without_touching_target(tmp_path) -> None:
    target = tmp_path / "known-token.txt"
    target.write_text("a" * 32, encoding="utf-8")
    target.chmod(0o644)
    (tmp_path / "voiceloop.token").symlink_to(target)

    with pytest.raises(RuntimeError, match="non-symlink"):
        Settings(voiceloop_data_dir=str(tmp_path)).ensure_local_token()

    assert target.read_text(encoding="utf-8") == "a" * 32
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_local_token_reuse_rejects_non_regular_path(tmp_path) -> None:
    (tmp_path / "voiceloop.token").mkdir()

    with pytest.raises(RuntimeError, match="regular"):
        Settings(voiceloop_data_dir=str(tmp_path)).ensure_local_token()


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL semantics")
def test_local_token_file_has_protected_current_user_only_dacl(tmp_path) -> None:
    import win32security

    token_path = tmp_path / "voiceloop.token"
    Settings(voiceloop_data_dir=str(tmp_path)).ensure_local_token()
    descriptor = win32security.GetNamedSecurityInfo(
        str(token_path),
        win32security.SE_FILE_OBJECT,
        (
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION
        ),
    )
    owner = descriptor.GetSecurityDescriptorOwner()
    dacl = descriptor.GetSecurityDescriptorDacl()
    control, _revision = descriptor.GetSecurityDescriptorControl()

    assert control & win32security.SE_DACL_PROTECTED
    assert dacl.GetAceCount() == 1
    assert win32security.EqualSid(dacl.GetAce(0)[2], owner)
