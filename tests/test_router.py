import pytest

from voiceloop.models import CommandRequest, CommandSource, RiskLevel
from voiceloop.router import deterministic_plan, normalize_text


def test_normalize_polish_text() -> None:
    assert normalize_text("  OTWÓRZ   Kalendarz! ") == "otworz   kalendarz"


def test_calendar_command_is_deterministic() -> None:
    request = CommandRequest(source=CommandSource.PANEL, text="Otwórz kalendarz")
    plan = deterministic_plan(request)

    assert plan is not None
    assert plan.intent == "open_calendar"
    assert [step.action_id for step in plan.steps] == ["open_calendar"]


def test_this_pc_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.PANEL, text="Otwórz mój komputer.")
    )

    assert plan is not None
    assert plan.intent == "open_folder"
    assert [step.action_id for step in plan.steps] == ["open_folder"]
    assert plan.steps[0].args == {"folder_id": "this_pc"}


def test_whatsapp_stt_variant_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.PANEL, text="Otwórz Whatsap.")
    )

    assert plan is not None
    assert plan.intent == "open_app"
    assert [step.action_id for step in plan.steps] == ["open_app"]
    assert plan.steps[0].args == {"app_id": "whatsapp"}


def test_spaced_polish_domain_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.PANEL,
            text="Otwórz stronę devilpage. Pl",
        )
    )

    assert plan is not None
    assert plan.intent == "open_url"
    assert [step.action_id for step in plan.steps] == ["open_url"]
    assert plan.steps[0].args == {"url": "https://devilpage.pl"}


def test_unknown_app_is_not_open_app() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.PANEL, text="Otwórz Autodesk Key.")
    )

    assert plan is None or not plan.steps or plan.steps[0].action_id != "open_app"


@pytest.mark.parametrize(
    ("text", "action_id", "query"),
    [
        ("Najedź na A Way Out", "hover_shell_item", "A Way Out"),
        ("Przesuń kursor na Mortal Shell", "hover_shell_item", "Mortal Shell"),
        ("Otwórz Mortal Shell", "open_shell_item", "Mortal Shell"),
    ],
)
def test_shell_item_commands_are_deterministic(text, action_id, query) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.steps[0].action_id == action_id
    assert plan.steps[0].args == {"query": query}
    assert plan.steps[0].confirmation_required is (action_id == "open_shell_item")


def test_known_open_commands_keep_priority_over_shell_items() -> None:
    for text, expected in (
        ("Otwórz WhatsApp", "open_app"),
        ("Otwórz kalendarz", "open_calendar"),
        ("Otwórz Ten komputer", "open_folder"),
        ("Otwórz Gemini", "open_gemini_chat"),
    ):
        plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))
        assert plan is not None
        assert plan.steps[0].action_id == expected


@pytest.mark.parametrize(
    "text",
    (
        "Czy potrafisz otworzyć Mortal Shell?",
        "Jak mam powiedzieć, żeby otworzyć A Way Out?",
    ),
)
def test_shell_capability_questions_do_not_execute(text: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.intent == "list_capabilities"
    assert [step.action_id for step in plan.steps] == ["list_capabilities"]


@pytest.mark.parametrize(
    "text",
    (
        "Czy potrafisz otworzyć WhatsApp?",
        "Jak mam powiedzieć, żeby otworzyć Ten komputer?",
    ),
)
def test_capability_questions_describe_action_without_executing_it(text: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.intent == "list_capabilities"
    assert [step.action_id for step in plan.steps] == ["list_capabilities"]
    assert plan.steps[0].args == {"query": text}


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


def test_minimize_window_under_cursor_from_deepgram_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Zminimalizuj aplikację pod kursorem.",
        )
    )

    assert plan is not None
    assert plan.intent == "minimize_window_under_cursor"
    assert [step.action_id for step in plan.steps] == ["minimize_window_under_cursor"]


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
    email = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Skopiuj mi ten mail gdzie mam kursor do schowka",
        )
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

    assert email is not None
    assert email.intent == "copy_email_under_cursor"
    assert [step.action_id for step in email.steps] == ["copy_email_under_cursor"]

    assert sentence is not None
    assert sentence.intent == "copy_sentence_under_cursor"
    assert [step.action_id for step in sentence.steps] == ["copy_sentence_under_cursor"]


def test_cursor_text_selection_commands_are_deterministic() -> None:
    copied = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Skopiuj tekst pod kursorem")
    )
    sentence = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Zaznacz to zdanie pod kursorem")
    )
    paragraph = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Zaznacz akapit")
    )

    assert copied is not None
    assert [step.action_id for step in copied.steps] == ["copy_text_under_cursor"]
    assert sentence is not None
    assert [step.action_id for step in sentence.steps] == ["select_sentence_under_cursor"]
    assert paragraph is not None
    assert [step.action_id for step in paragraph.steps] == ["select_paragraph_under_cursor"]


def test_close_window_under_cursor_is_medium_risk() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Wyłącz aplikację którą wskazuję kursorem",
        )
    )

    assert plan is not None
    assert plan.intent == "close_window_under_cursor"
    assert [step.action_id for step in plan.steps] == ["close_window_under_cursor"]
    assert plan.steps[0].risk is RiskLevel.MEDIUM
    assert plan.steps[0].confirmation_required is True


@pytest.mark.parametrize(
    "text",
    [
        "Zamknij to okno",
        "Zamknij tę aplikację",
        "Wyłącz tę aplikację",
        "Zamknij wskazane okno",
        "Zamknij program który wskazuję",
    ],
)
def test_generic_close_is_not_window_under_cursor(text: str) -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text=text)
    )

    assert plan is None or not plan.steps or plan.steps[0].action_id != (
        "close_window_under_cursor"
    )


def test_generic_minimize_is_not_window_under_cursor() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Schowaj to okno")
    )

    assert plan is None or not plan.steps or plan.steps[0].action_id != (
        "minimize_window_under_cursor"
    )


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


@pytest.mark.parametrize(
    "text",
    [
        "Jaka jest pogoda",
        "Powiedz jak dziś akcje wybranej spółki",
        "Jak dzisiaj akcje wybranej spółki",
    ],
)
def test_current_info_questions_use_web_search(text: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.intent == "search_web"
    assert [step.action_id for step in plan.steps] == ["search_web"]
    assert text.split()[0] in str(plan.steps[0].args["query"])


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


def test_rename_under_cursor_is_deterministic_with_confirmation() -> None:
    plan = deterministic_plan(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Zmień nazwę pod kursorem na Raport Q2",
        )
    )
    assert plan is not None
    assert plan.intent == "rename_under_cursor"
    assert plan.steps[0].action_id == "rename_under_cursor"
    assert plan.steps[0].confirmation_required is True
    assert plan.steps[0].args["new_name"] == "raport q2"


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
        ("Otwórz mój komputer", "open_folder"),
        ("Otwórz Whatsap", "open_app"),
        ("Otwórz YouTube", "open_url"),
        ("Otwórz chat gpt", "open_gpt_chat"),
        ("ChatGPT", "open_chat"),
        ("Otwórz Gemini", "open_gemini_chat"),
        ("Jakie okno jest aktywne", "describe_active_window"),
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
        ("Skopiuj tekst pod kursorem", "copy_text_under_cursor"),
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


# ---------------------------------------------------------------------------
# Paleta sterowania kursorem i aktywnym folderem.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    (
        "Kursor na środek",
        "Wyśrodkuj kursor",
        "Przesuń kursor na środek ekranu",
    ),
)
def test_cursor_center_commands_are_deterministic(text: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.intent == "cursor_center"
    assert [step.action_id for step in plan.steps] == ["cursor_center"]
    assert plan.steps[0].confirmation_required is False
    assert plan.steps[0].risk is RiskLevel.LOW


@pytest.mark.parametrize(
    "text",
    (
        "Wróć kursorem",
        "Przywróć kursor",
        "Przywróć pozycję kursora",
    ),
)
def test_cursor_return_commands_are_deterministic(text: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.intent == "cursor_return"
    assert [step.action_id for step in plan.steps] == ["cursor_return"]
    assert plan.steps[0].confirmation_required is False


def test_stop_command_takes_priority_and_clears_steps() -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text="Stop"))

    assert plan is not None
    assert plan.intent == "stop"
    assert plan.steps == []


@pytest.mark.parametrize(
    ("text", "layout"),
    [
        ("Przesuń okno na lewą połowę", "left_half"),
        ("Przesuń okno na prawą połowę", "right_half"),
        ("Przesuń okno na lewą jedną trzecią", "left_third"),
        ("Przesuń okno na środkową jedną trzecią", "center_third"),
        ("Przesuń okno na prawą jedną trzecią", "right_third"),
        ("Przesuń okno na lewą górną ćwiartkę", "top_left_quarter"),
        ("Przesuń okno na lewą dolną ćwiartkę", "bottom_left_quarter"),
        ("Przesuń okno na prawą górną ćwiartkę", "top_right_quarter"),
        ("Przesuń okno na prawą dolną ćwiartkę", "bottom_right_quarter"),
    ],
)
def test_window_layout_commands_are_deterministic(text: str, layout: str) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.intent == "snap_window_layout"
    assert plan.steps[0].action_id == "snap_window_layout"
    assert plan.steps[0].args == {"layout": layout}
    assert plan.steps[0].confirmation_required is False


@pytest.mark.parametrize(
    ("text", "action_id", "query"),
    [
        ("Zaznacz folder Projekty", "select_shell_folder", "Projekty"),
        ("Zaznacz plik Raport Q2", "select_shell_file", "Raport Q2"),
    ],
)
def test_select_shell_target_commands_are_deterministic(text, action_id, query) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.steps[0].action_id == action_id
    assert plan.steps[0].args == {"query": query}
    assert plan.steps[0].confirmation_required is False


def test_select_by_extension_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Zaznacz wszystkie PDF")
    )

    assert plan is not None
    assert plan.steps[0].action_id == "select_shell_items_by_extension"
    assert plan.steps[0].args == {"extension": "PDF"}


def test_select_by_letter_command_is_deterministic() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Zaznacz wszystkie na literę A")
    )

    assert plan is not None
    assert plan.steps[0].action_id == "select_shell_items_by_letter"
    assert plan.steps[0].args == {"letter": "A"}


@pytest.mark.parametrize(
    ("text", "index"),
    [
        ("pierwszy", 1),
        ("drugi", 2),
        ("trzeci", 3),
    ],
)
def test_candidate_ordinal_commands_are_deterministic(text: str, index: int) -> None:
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.steps[0].action_id == "select_listed_candidate"
    assert plan.steps[0].args == {"index": index}
    assert plan.steps[0].confirmation_required is False


def test_open_shell_item_still_requires_confirmation_alongside_new_commands() -> None:
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Uruchom Mortal Shell")
    )

    assert plan is not None
    assert plan.steps[0].action_id == "open_shell_item"
    assert plan.steps[0].args == {"query": "Mortal Shell"}
    assert plan.steps[0].confirmation_required is True
    assert plan.steps[0].risk is RiskLevel.MEDIUM


def test_cursor_commands_take_priority_over_generic_hover() -> None:
    # "Kursor na środek" nie powinno zostać pomylone z "najedź na ..." itp.
    plan = deterministic_plan(
        CommandRequest(source=CommandSource.DEEPGRAM, text="Kursor na środek")
    )

    assert plan is not None
    assert plan.steps[0].action_id == "cursor_center"


@pytest.mark.parametrize(
    ("text", "action_id", "query"),
    [
        ("Przesuń kursor na folder wizyta", "hover_shell_item", "wizyta"),
        ("Najedź na plik Faktura", "hover_shell_item", "Faktura"),
        ("Otwórz folder wizyta", "open_shell_item", "wizyta"),
        ("Uruchom plik Raport Q2", "open_shell_item", "Raport Q2"),
    ],
)
def test_hover_and_open_strip_folder_plik_qualifier(text, action_id, query) -> None:
    # Naturalna fraza "na folder NAZWA"/"na plik NAZWA" musi trafić w samą nazwę,
    # żeby dopasowanie rozmyte nie odpadło z powodu dodatkowego słowa.
    plan = deterministic_plan(CommandRequest(source=CommandSource.DEEPGRAM, text=text))

    assert plan is not None
    assert plan.steps[0].action_id == action_id
    assert plan.steps[0].args == {"query": query}


def test_stop_has_no_executable_step() -> None:
    plan = deterministic_plan(CommandRequest(command_id="stop"))

    assert plan is not None
    assert plan.intent == "stop"
    assert plan.steps == []
