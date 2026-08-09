from voiceloop.models import CommandRequest, CommandSource
from voiceloop.router import deterministic_plan, normalize_text


def test_normalize_polish_text() -> None:
    assert normalize_text("  OTWÓRZ   Kalendarz! ") == "otworz   kalendarz"


def test_calendar_command_is_deterministic() -> None:
    request = CommandRequest(source=CommandSource.PANEL, text="Otwórz kalendarz")
    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "open_calendar"
    assert [step.action_id for step in plan.steps] == ["open_calendar"]


def test_note_extracts_content() -> None:
    request = CommandRequest(text="Zapisz notatkę kup mleko i chleb")
    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "create_note"
    assert plan.steps[0].args["text"] == "kup mleko i chleb"


def test_stop_has_no_executable_step() -> None:
    plan = deterministic_plan(CommandRequest(command_id="stop"))

    assert plan is not None
    assert plan.intent == "stop"
    assert plan.steps == []
