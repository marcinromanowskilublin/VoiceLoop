from __future__ import annotations

from pathlib import Path

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
    assert hasattr(settings, "voiceattack_registration_key")


def test_pytest_disables_settings_env_file() -> None:
    assert Settings.model_config.get("env_file") is None
