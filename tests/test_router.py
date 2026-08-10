import pytest

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


def test_short_test_alias_is_deterministic() -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text="test"))

    assert plan is not None
    assert plan.intent == "voice_test"
    assert plan.steps == []


def test_active_window_question_is_deterministic() -> None:
    request = CommandRequest(source=CommandSource.DEEPGRAM, text="Co jest teraz otwarte?")

    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "describe_active_window"
    assert [step.action_id for step in plan.steps] == ["describe_active_window"]
    assert plan.confirmation_required is False


def test_minimize_window_commands_are_deterministic() -> None:
    active = deterministic_plan(
        CommandRequest(source=CommandSource.VOICEATTACK, command_id="minimize_active_window")
    )
    all_windows = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Pokaż pulpit")
    )

    assert active is not None
    assert active.intent == "minimize_active_window"
    assert [step.action_id for step in active.steps] == ["minimize_active_window"]
    assert all_windows is not None
    assert all_windows.intent == "minimize_all_windows"
    assert [step.action_id for step in all_windows.steps] == ["minimize_all_windows"]


def test_recent_activity_question_extracts_hours() -> None:
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="Co robiłem przez ostatnie 2 godziny?",
    )

    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "describe_recent_activity"
    assert [step.action_id for step in plan.steps] == ["describe_recent_activity"]
    assert plan.steps[0].args == {"minutes": 120}
    assert plan.confirmation_required is False


def test_copy_commands_are_deterministic() -> None:
    selected = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Kopiuj zaznaczony tekst")
    )
    number = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Skopiuj numer pod kursorem")
    )
    sentence = deterministic_plan(
        CommandRequest(
            source=CommandSource.VOICEATTACK,
            command_id="copy_sentence_under_cursor",
        )
    )

    assert selected is not None
    assert selected.intent == "copy_selected_text"
    assert [step.action_id for step in selected.steps] == ["copy_selected_text"]

    assert number is not None
    assert number.intent == "copy_number_under_cursor"
    assert [step.action_id for step in number.steps] == ["copy_number_under_cursor"]

    assert sentence is not None
    assert sentence.intent == "copy_sentence_under_cursor"
    assert [step.action_id for step in sentence.steps] == ["copy_sentence_under_cursor"]


def test_web_search_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Wyszukaj w internecie najnowsze info o Python 3.13",
        )
    )

    assert plan is not None
    assert plan.intent == "search_web"
    assert [step.action_id for step in plan.steps] == ["search_web"]
    assert "Python 3.13" in str(plan.steps[0].args["query"])


def test_web_search_command_id_without_query_requires_clarification() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.VOICEATTACK, command_id="search_web")
    )

    assert plan is not None
    assert plan.intent == "search_web"
    assert plan.requires_clarification is True
    assert "Co mam wyszukać" in (plan.clarification_question or "")
    assert plan.steps == []


def test_api_endpoint_check_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text=(
                "venice przejrzyj w necie api od stripe i zobacz "
                "czy jest tam endpopint /v1/customers"
            ),
        )
    )

    assert plan is not None
    assert plan.intent == "search_web"
    assert [step.action_id for step in plan.steps] == ["search_web"]
    assert plan.steps[0].args["api_name"] == "stripe"
    assert plan.steps[0].args["endpoint"] == "/v1/customers"


def test_remember_last_source_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="zapamietaj ostatnie źródło 2",
        )
    )

    assert plan is not None
    assert plan.intent == "remember_last_source"
    assert [step.action_id for step in plan.steps] == ["remember_last_source"]
    assert plan.steps[0].args["index"] == 2


def test_chat_split_commands_are_deterministic() -> None:
    gpt = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Otwórz GPT")
    )
    gemini = deterministic_plan(
        CommandRequest(source=CommandSource.VOICEATTACK, command_id="open_gemini_chat")
    )

    assert gpt is not None
    assert gpt.intent == "open_gpt_chat"
    assert [step.action_id for step in gpt.steps] == ["open_gpt_chat"]
    assert gemini is not None
    assert gemini.intent == "open_gemini_chat"
    assert [step.action_id for step in gemini.steps] == ["open_gemini_chat"]


def test_text_target_check_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Czy to pasek adresu?")
    )

    assert plan is not None
    assert plan.intent == "describe_text_target"
    assert [step.action_id for step in plan.steps] == ["describe_text_target"]


def test_safe_paste_command_extracts_target_and_text() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Wpisz do gemini: przygotuj krótkie podsumowanie spotkania",
        )
    )

    assert plan is not None
    assert plan.intent == "paste_text_safe"
    assert [step.action_id for step in plan.steps] == ["paste_text_safe"]
    assert plan.steps[0].args["expected_window"] == "gemini"
    assert "podsumowanie spotkania" in str(plan.steps[0].args["text"])


@pytest.mark.parametrize(
    ("text", "expected_action_id"),
    [
        ("Uruchom kalendarz", "open_calendar"),
        ("Kalendarz", "open_calendar"),
        ("Odpal przeglądarkę", "open_browser"),
        ("Przeglądarka", "open_browser"),
        ("Otwórz chat gpt", "open_gpt_chat"),
        ("ChatGPT", "open_chat"),
        ("Otwórz Gemini", "open_gemini_chat"),
        ("Jakie mam teraz okno aktywne?", "describe_active_window"),
        ("Czy to dobre pole do pisania", "describe_text_target"),
        ("Aktywne okno", "describe_active_window"),
        ("Zwiń aktywne okno", "minimize_active_window"),
        ("Zwiń", "minimize_active_window"),
        ("Minimalizuj wszystko", "minimize_all_windows"),
        ("Pulpit", "minimize_all_windows"),
        ("Skopiuj zaznaczone teskty", "copy_selected_text"),
        ("Kopiuj zaznaczenie", "copy_selected_text"),
        ("Skopiuj telefon pod myszką", "copy_number_under_cursor"),
        ("Kopiuj numer", "copy_number_under_cursor"),
        ("Skopiuj tekst pod kursorem", "copy_sentence_under_cursor"),
        ("Kopiuj zdanie", "copy_sentence_under_cursor"),
        ("Aktywność", "describe_recent_activity"),
    ],
)
def test_paraphrases_are_deterministic(text: str, expected_action_id: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert [step.action_id for step in plan.steps] == [expected_action_id]


def test_recent_activity_paraphrase_keeps_time_parsing() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Podsumuj aktywność z ostatnich 2 godzin",
        )
    )

    assert plan is not None
    assert plan.intent == "describe_recent_activity"
    assert plan.steps[0].args == {"minutes": 120}


def test_stop_paraphrase_has_no_executable_step() -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text="Awaryjnie stop"))

    assert plan is not None
    assert plan.intent == "stop"
    assert plan.steps == []


def test_note_extracts_content() -> None:
    request = CommandRequest(text="Zapisz notatkę kup mleko i chleb")
    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "create_note"
    assert plan.steps[0].args["text"] == "kup mleko i chleb"


def test_recent_activity_command_id_is_deterministic() -> None:
    plan = deterministic_plan(CommandRequest(command_id="recent_activity"))

    assert plan is not None
    assert plan.intent == "describe_recent_activity"
    assert plan.steps[0].args == {"minutes": 30}


def test_remember_extracts_content_and_requires_policy_confirmation() -> None:
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="Zapamiętaj że wolę krótkie odpowiedzi",
    )

    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "remember"
    assert plan.steps[0].action_id == "remember"
    assert plan.steps[0].args == {
        "content": "wolę krótkie odpowiedzi",
        "kind": "fact",
    }
    assert "potwierdź" in plan.response_text


def test_stop_has_no_executable_step() -> None:
    plan = deterministic_plan(CommandRequest(command_id="stop"))

    assert plan is not None
    assert plan.intent == "stop"
    assert plan.steps == []
