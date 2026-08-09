import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.memory import MemoryStore
from voiceloop.models import PlanStep, RiskLevel
from voiceloop.settings import Settings
from voiceloop.tts import WindowsTTS


def test_policy_cannot_lower_registered_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    step = PlanStep(action_id="run_uivision_macro", risk=RiskLevel.LOW)

    secured = registry.enforce_policy(step)

    assert secured.risk is RiskLevel.MEDIUM
    assert secured.confirmation_required is True


def test_unknown_action_is_rejected(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    with pytest.raises(ValueError, match="unknown action"):
        registry.enforce_policy(PlanStep(action_id="powershell_anything"))
