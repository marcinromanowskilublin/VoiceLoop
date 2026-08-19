from __future__ import annotations

import re
import unicodedata

from .models import (
    CommandPlan,
    CommandRequest,
    PlanStep,
    RiskLevel,
    SegmentationDecisionV1,
)


def normalize_text(value: str) -> str:
    normalized = value.casefold().replace("ł", "l")
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9ąćęłńóśźż:/._ -]+", " ", normalized).strip()


def deterministic_plan(request: CommandRequest) -> CommandPlan | None:
    raw = (request.command_id or request.text or "").strip()
    text = normalize_text(raw)
    if not text:
        return None

    if request.command_id is None:
        from .routing.segmenter import has_stop_subtask, segment_command

        segmentation = segment_command(raw)
        if has_stop_subtask(segmentation):
            return CommandPlan(
                request_id=request.request_id,
                intent="stop",
                response_text="Zatrzymuję.",
                confidence=1.0,
                steps=[],
                provider="deterministic",
            )
        if segmentation.decision in {
            SegmentationDecisionV1.COMPOUND,
            SegmentationDecisionV1.AMBIGUOUS,
        }:
            count = len(segmentation.subtasks)
            detail = (
                f"Rozpoznałam {count} części polecenia"
                if count > 1
                else "Polecenie ma niejednoznaczną strukturę"
            )
            return CommandPlan(
                request_id=request.request_id,
                intent="task",
                response_text=(
                    f"{detail}. Nie wykonuję tylko pierwszego fragmentu, "
                    "żeby nie pominąć pozostałych czynności."
                ),
                confidence=segmentation.confidence,
                requires_clarification=True,
                clarification_question=(
                    "Powtórz proszę każdą czynność osobno albo potwierdź pełny plan, "
                    "gdy Routing V2 go przedstawi."
                ),
                provider="compound_fast_path_guard",
            )

    if text in {"ping", "voice_test", "voice test", "test petli", "test glosu", "test"}:
        return CommandPlan(
            request_id=request.request_id,
            intent="voice_test",
            response_text="Pętla VoiceLoop działa.",
            confidence=1.0,
            steps=[],
            provider="deterministic",
        )

    if _is_capabilities_command(text):
        return _single_step(
            request,
            intent="list_capabilities",
            response_text="Sprawdzam komendy VoiceAttack i wszystkie własne akcje VoiceLoop.",
            action_id="list_capabilities",
            args={"query": request.text or ""},
        )

    if _is_open_calendar_command(text):
        return _single_step(
            request,
            intent="open_calendar",
            response_text="Otwieram kalendarz.",
            action_id="open_calendar",
        )

    if _is_open_folder_command(text):
        return _single_step(
            request,
            intent="open_folder",
            response_text="Otwieram Ten komputer.",
            action_id="open_folder",
            args={"folder_id": "this_pc"},
        )

    if _is_open_app_command(text):
        return _single_step(
            request,
            intent="open_app",
            response_text="Otwieram WhatsApp.",
            action_id="open_app",
            args={"app_id": "whatsapp"},
        )

    if _is_open_browser_command(text):
        return _single_step(
            request,
            intent="open_browser",
            response_text="Otwieram przeglądarkę.",
            action_id="open_browser",
        )

    open_url = _extract_open_url(raw, text)
    if open_url:
        return _single_step(
            request,
            intent="open_url",
            response_text="Otwieram stronę.",
            action_id="open_url",
            args={"url": open_url},
        )

    api_endpoint_check = _extract_api_endpoint_check(raw)
    if api_endpoint_check:
        api_name, endpoint = api_endpoint_check
        return _single_step(
            request,
            intent="search_web",
            response_text="Sprawdzam dokumentację API i wskazany endpoint.",
            action_id="search_web",
            args={
                "query": f"{api_name} API documentation {endpoint}",
                "limit": 5,
                "api_name": api_name,
                "endpoint": endpoint,
            },
        )

    current_info_query = _extract_current_info_query(raw)
    if current_info_query:
        return _single_step(
            request,
            intent="search_web",
            response_text="Sprawdzam aktualne informacje w internecie.",
            action_id="search_web",
            args={"query": current_info_query, "limit": 5},
        )

    search_query = _extract_web_search_query(raw)
    if search_query:
        return _single_step(
            request,
            intent="search_web",
            response_text="Sprawdzam internet i szukam informacji.",
            action_id="search_web",
            args={"query": search_query, "limit": 5},
        )
    if _is_search_web_command(text):
        return CommandPlan(
            request_id=request.request_id,
            intent="search_web",
            response_text="Co mam wyszukać w internecie?",
            confidence=1.0,
            requires_clarification=True,
            clarification_question="Co mam wyszukać w internecie?",
            provider="deterministic",
        )

    remember_source_index = _extract_remember_last_source_index(text)
    if remember_source_index is not None:
        return _single_step(
            request,
            intent="remember_last_source",
            response_text="Mogę zapamiętać ostatnie źródło. Powiedz potwierdź albo anuluj zadanie.",
            action_id="remember_last_source",
            args={"index": remember_source_index, "kind": "web_source"},
            risk=RiskLevel.MEDIUM,
        )

    if _is_open_gpt_chat_command(text):
        return _single_step(
            request,
            intent="open_gpt_chat",
            response_text="Otwieram ChatGPT.",
            action_id="open_gpt_chat",
        )

    if _is_open_gemini_chat_command(text):
        return _single_step(
            request,
            intent="open_gemini_chat",
            response_text="Otwieram Gemini.",
            action_id="open_gemini_chat",
        )

    if _is_open_chat_command(text):
        return _single_step(
            request,
            intent="open_chat",
            response_text="Otwieram czat.",
            action_id="open_chat",
        )

    if _is_describe_text_target_command(text):
        return _single_step(
            request,
            intent="describe_text_target",
            response_text="Sprawdzam, gdzie trafi wpisywany tekst.",
            action_id="describe_text_target",
        )

    if _is_active_window_command(text):
        return _single_step(
            request,
            intent="describe_active_window",
            response_text="Sprawdzam aktywne okno.",
            action_id="describe_active_window",
        )

    if _is_minimize_window_under_cursor_command(text):
        return _single_step(
            request,
            intent="minimize_window_under_cursor",
            response_text="Minimalizuję okno wskazywane kursorem.",
            action_id="minimize_window_under_cursor",
        )

    if _is_minimize_active_window_command(text):
        return _single_step(
            request,
            intent="minimize_active_window",
            response_text="Minimalizuję aktywne okno.",
            action_id="minimize_active_window",
        )

    if _is_minimize_all_windows_command(text):
        return _single_step(
            request,
            intent="minimize_all_windows",
            response_text="Minimalizuję wszystkie okna.",
            action_id="minimize_all_windows",
        )

    if _is_close_window_under_cursor_command(text):
        return _single_step(
            request,
            intent="close_window_under_cursor",
            response_text=(
                "Zamknąć okno aplikacji wskazywane kursorem? "
                "Powiedz potwierdź albo anuluj zadanie."
            ),
            action_id="close_window_under_cursor",
            risk=RiskLevel.MEDIUM,
        )

    if _is_copy_email_under_cursor_command(text):
        return _single_step(
            request,
            intent="copy_email_under_cursor",
            response_text="Kopiuję adres e-mail wskazywany kursorem.",
            action_id="copy_email_under_cursor",
        )

    if _is_copy_selected_text_command(text):
        return _single_step(
            request,
            intent="copy_selected_text",
            response_text="Kopiuję zaznaczony tekst.",
            action_id="copy_selected_text",
        )

    if _is_copy_number_under_cursor_command(text):
        return _single_step(
            request,
            intent="copy_number_under_cursor",
            response_text="Szukam numeru pod kursorem i kopiuję go.",
            action_id="copy_number_under_cursor",
        )

    if _is_copy_sentence_under_cursor_command(text):
        return _single_step(
            request,
            intent="copy_sentence_under_cursor",
            response_text="Kopiuję całe zdanie pod kursorem.",
            action_id="copy_sentence_under_cursor",
        )

    if _is_copy_text_under_cursor_command(text):
        return _single_step(
            request,
            intent="copy_text_under_cursor",
            response_text="Kopiuję tekst wskazywany kursorem.",
            action_id="copy_text_under_cursor",
        )

    if _is_select_sentence_under_cursor_command(text):
        return _single_step(
            request,
            intent="select_sentence_under_cursor",
            response_text="Zaznaczam zdanie pod kursorem.",
            action_id="select_sentence_under_cursor",
        )

    if _is_select_paragraph_under_cursor_command(text):
        return _single_step(
            request,
            intent="select_paragraph_under_cursor",
            response_text="Zaznaczam akapit pod kursorem.",
            action_id="select_paragraph_under_cursor",
        )

    rename_match = _extract_rename_under_cursor(text)
    if rename_match is not None:
        new_name, response = rename_match
        args = {"new_name": new_name} if new_name else {}
        return CommandPlan(
            request_id=request.request_id,
            intent="rename_under_cursor",
            response_text=response,
            confidence=1.0,
            steps=[
                PlanStep(
                    action_id="rename_under_cursor",
                    args=args,
                    risk=RiskLevel.MEDIUM,
                    confirmation_required=True,
                )
            ],
            provider="deterministic",
        )

    if _is_recent_activity_command(text):
        minutes = 30
        minute_match = re.search(r"\b(\d{1,5})\s*(?:minut|minuty|min)\b", text)
        hour_match = re.search(r"\b(\d{1,3})\s*(?:godzin|godziny|godzine|h)\b", text)
        if minute_match:
            minutes = int(minute_match.group(1))
        elif hour_match:
            minutes = int(hour_match.group(1)) * 60
        elif "godzin" in text or "godzine" in text:
            minutes = 60
        return _single_step(
            request,
            intent="describe_recent_activity",
            response_text="Sprawdzam ostatnią aktywność w Screenpipe.",
            action_id="describe_recent_activity",
            args={"minutes": max(1, min(minutes, 20160))},
        )

    if _is_stop_command(text):
        return CommandPlan(
            request_id=request.request_id,
            intent="stop",
            response_text="Zatrzymuję.",
            confidence=1.0,
            steps=[],
            provider="deterministic",
        )

    safe_paste = _safe_paste_step(request, raw)
    if safe_paste is not None:
        return safe_paste

    note_match = re.search(
        r"(?:zapisz|utworz|stworz|dodaj|zanotuj)(?: mi)? "
        r"(?:notatke|notatka)(?: o tresci|:)? (.+)",
        text,
    )
    if note_match:
        content = raw[-len(note_match.group(1)) :].strip()
        return _single_step(
            request,
            intent="create_note",
            response_text="Tworzę notatkę.",
            action_id="create_note",
            args={"text": content},
            risk=RiskLevel.MEDIUM,
        )

    if text in {
        "note",
        "notatka",
        "take a note",
        "new note",
        "zapisz notatke",
        "nowa notatka",
        "dodaj notatke",
        "utworz notatke",
        "zanotuj",
    }:
        return CommandPlan(
            request_id=request.request_id,
            intent="create_note",
            response_text="Co mam zapisać w notatce?",
            confidence=1.0,
            requires_clarification=True,
            clarification_question="Co mam zapisać w notatce?",
            provider="deterministic",
        )

    remember_match = re.search(
        r"(?:zapamietaj|pamietaj|zapamietaj to)(?: prosze)?(?: ze| to)?[: ]+(.+)",
        text,
    )
    if remember_match:
        content = raw[-len(remember_match.group(1)) :].strip()
        return _single_step(
            request,
            intent="remember",
            response_text=(
                "Mogę to zapamiętać. Powiedz potwierdź albo anuluj zadanie."
            ),
            action_id="remember",
            args={"content": content, "kind": "fact"},
            risk=RiskLevel.MEDIUM,
        )

    return None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _is_exact_alias(text: str, aliases: tuple[str, ...]) -> bool:
    return text in aliases


def _has_copy_verb(text: str) -> bool:
    return _contains_any(text, ("kopiuj", "skopiuj", "kopiowac", "copy"))


def _is_open_calendar_command(text: str) -> bool:
    return _is_exact_alias(
        text,
        ("open_calendar", "open calendar", "kalendarz", "calendar"),
    ) or _contains_any(
        text,
        (
            "otworz kalendarz",
            "uruchom kalendarz",
            "wlacz kalendarz",
            "pokaz kalendarz",
            "odpal kalendarz",
        ),
    )


def _is_open_browser_command(text: str) -> bool:
    return _is_exact_alias(
        text,
        ("open_browser", "open browser", "przegladarka", "browser", "chrome"),
    ) or _contains_any(
        text,
        (
            "otworz przegladark",
            "uruchom przegladark",
            "wlacz przegladark",
            "odpal przegladark",
            "launch browser",
            "otworz chrome",
            "uruchom chrome",
        ),
    )


def _is_open_folder_command(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:moj|ten)\s+komputer\b|\b(?:eksplorator\w*|explorer)\b",
            text,
        )
    )


def _is_open_app_command(text: str) -> bool:
    if not re.search(r"\b(?:whatsapp|whats\s*app|whatsap)\b", text):
        return False
    return _is_exact_alias(
        text,
        ("open_app", "whatsapp", "whatsap", "whats app"),
    ) or _contains_any(
        text,
        ("otworz", "uruchom", "wlacz", "odpal", "wejdz"),
    )


def _extract_open_url(raw: str, text: str) -> str | None:
    haystack = f"{raw} {text}"
    explicit = re.search(r"https?://[^\s<>'\"]+", haystack, flags=re.IGNORECASE)
    if explicit:
        url = explicit.group(0).rstrip(".,;:!?)]}")
    elif re.search(r"\b(?:youtube|jutub|jutiub)\w*\b", text):
        url = "https://www.youtube.com"
    else:
        bare_domain = re.search(
            r"(?<![\w.-])([a-z0-9-]+\s*\.\s*(?:pl|com|org|net|io|ai))\b",
            haystack,
            flags=re.IGNORECASE,
        )
        if not bare_domain:
            return None
        domain = re.sub(r"\s+", "", bare_domain.group(1)).lower()
        url = f"https://{domain}"
    if _is_exact_alias(text, ("youtube", "open_url")):
        return url
    if _contains_any(
        text,
        (
            "otworz",
            "uruchom",
            "wlacz",
            "odpal",
            "wejdz",
            "przejdz",
            "stron",
            "link",
            "adres",
        ),
    ):
        return url
    return None


def _is_search_web_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "search_web",
            "wyszukaj w internecie",
            "wyszukaj w necie",
            "sprawdz w internecie",
            "sprawdz w necie",
            "szukaj online",
            "przejrzyj w necie",
            "przeszukaj internet",
        ),
    ):
        return True
    return _contains_any(
        text,
        (
            "wyszukaj w internecie",
            "wyszukaj w necie",
            "sprawdz w internecie",
            "sprawdz w necie",
            "znajdz w internecie",
            "poszukaj w internecie",
            "przejrzyj w necie",
            "przeszukaj internet",
        ),
    )


def _is_remember_last_source_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "remember_last_source",
            "zapamietaj ostatnie zrodlo",
            "zapamietaj ostatni link",
            "zapamietaj ostatnie zrodla",
            "zapamietaj to zrodlo",
        ),
    ):
        return True
    if not _contains_any(text, ("zapamiet", "pamietaj")):
        return False
    if not _contains_any(text, ("zrodl", "link", "source")):
        return False
    return _contains_any(text, ("ostatn", "poprzedn", "last", "to zrodlo"))


def _extract_remember_last_source_index(text: str) -> int | None:
    if not _is_remember_last_source_command(text):
        return None
    digit_match = re.search(r"\b(\d{1,2})\b", text)
    if digit_match:
        return max(1, min(int(digit_match.group(1)), 20))
    ordinal_map = {
        "pierwsz": 1,
        "drug": 2,
        "trzec": 3,
        "czwart": 4,
        "piat": 5,
        "szost": 6,
        "siodm": 7,
        "osm": 8,
        "dziewiat": 9,
        "dziesiat": 10,
    }
    for marker, value in ordinal_map.items():
        if marker in text:
            return value
    return 1


def _extract_api_endpoint_check(raw: str) -> tuple[str, str] | None:
    compact = re.sub(r"\s+", " ", raw).strip()
    if not compact:
        return None

    patterns = (
        r"(?:api(?:\s+od|\s+dla)?\s+)(?P<api>[^,.;:!?]+?)\s+"
        r"(?:i\s+)?(?:zobacz|sprawdz|zweryfikuj|potwierdz|czy).*?"
        r"(?:endp\w*)\s+(?P<endpoint>[^\s,;.!?\"']+)",
        r"(?:endp\w*)\s+(?P<endpoint>[^\s,;.!?\"']+).*?"
        r"(?:api(?:\s+od|\s+dla)?\s+)(?P<api>[^,.;:!?]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            continue
        api_name = str(match.group("api") or "").strip().strip("\"'`")
        endpoint = str(match.group("endpoint") or "").strip().strip("\"'`")
        api_name = re.sub(r"\s+", " ", api_name)
        api_name = re.sub(
            r"\b(?:w necie|w internecie|w sieci|online)\b",
            "",
            api_name,
            flags=re.IGNORECASE,
        ).strip()
        if len(api_name) >= 2 and len(endpoint) >= 1:
            return api_name, endpoint
    return None


def _extract_web_search_query(raw: str) -> str | None:
    patterns = (
        r"^\s*(?:(?:venice|gemini|duck|ddg)\s+)?"
        r"(?:wyszukaj|szukaj|sprawdz|znajdz|poszukaj|przejrzyj|przeszukaj)"
        r"(?:\s+(?:w\s+internecie|w\s+necie|online))?\s*[:,-]?\s+(.+?)\s*$",
        r"^\s*(?:search|find|look up)\s+(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        query = match.group(1).strip().strip('"').strip("'")
        if query:
            return query[:400]
    return None


def _extract_current_info_query(raw: str) -> str | None:
    compact = re.sub(r"\s+", " ", raw).strip()
    if not compact:
        return None
    normalized = normalize_text(compact)
    if re.search(r"\b(jaka|jaki|jakie)\s+jest\s+pogoda\b", normalized):
        return compact[:400]
    if re.search(r"\b(pogoda|temperatura)\s+(dzis|dzisiaj|teraz|jutro)\b", normalized):
        return compact[:400]
    if re.search(
        r"\b(akcje|kurs|notowania|gielda|giełda)\b.*\b(dzis|dzisiaj|teraz)\b",
        normalized,
    ):
        return compact[:400]
    if re.search(
        r"\b(powiedz|sprawdz|jak|co)\b.*\b(akcje|kurs|notowania)\b",
        normalized,
    ):
        return compact[:400]
    return None


def _is_open_gpt_chat_command(text: str) -> bool:
    return _is_exact_alias(text, ("open_gpt_chat", "otworz gpt", "open gpt")) or _contains_any(
        text,
        (
            "otworz chat gpt",
            "uruchom chat gpt",
            "otworz gpt",
            "uruchom gpt",
        ),
    )


def _is_open_gemini_chat_command(text: str) -> bool:
    return _is_exact_alias(
        text,
        ("open_gemini_chat", "open gemini", "otworz gemini"),
    ) or _contains_any(
        text,
        (
            "otworz gemini",
            "uruchom gemini",
            "open gemini chat",
            "czat gemini",
        ),
    )


def _is_open_chat_command(text: str) -> bool:
    return _is_exact_alias(
        text,
        ("open_chat", "open chat", "chat", "czat", "chatgpt", "chat gpt"),
    ) or _contains_any(
        text,
        (
            "otworz czat",
            "otworz chat",
            "otworz chatgpt",
            "otworz chat gpt",
            "new chatgpt",
            "nowy chat gpt",
            "uruchom chatgpt",
            "uruchom chat gpt",
        ),
    )


def _is_describe_text_target_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "describe_text_target",
            "gdzie pisze",
            "gdzie teraz pisze",
            "sprawdz pole tekstowe",
            "czy to pasek adresu",
        ),
    ):
        return True
    return _contains_any(
        text,
        (
            "gdzie trafi tekst",
            "w jakie pole pisze",
            "czy moge tu pisac",
            "czy wpisze do paska adresu",
            "czy to dobre pole do pisania",
        ),
    )


def _is_active_window_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "active_window",
            "active window",
            "co jest otwarte",
            "jakie okno jest aktywne",
            "jakie okno jest otwarte",
            "jakie okno mam otwarte",
            "co mam otwarte",
            "co jest teraz otwarte",
            "aktywne okno",
            "okno aktywne",
        ),
    ):
        return True
    if _contains_any(
        text,
        (
            "opisz aktywne okno",
            "ktore okno jest aktywne",
            "na jakim oknie jestem",
            "jakie mam teraz okno",
            "podaj aktywne okno",
            "co jest teraz aktywne",
        ),
    ):
        return True
    return "aktywne okno" in text and _contains_any(text, ("co", "jakie", "podaj", "opisz"))


def _is_minimize_active_window_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "minimize_active_window",
            "minimize window",
            "zminimalizuj okno",
            "schowaj okno",
            "ukryj okno",
            "minimalizuj okno",
            "zwin",
            "zwin okno",
        ),
    ):
        return True
    return _contains_any(
        text,
        (
            "zwin okno",
            "zwin aktywne okno",
            "schowaj aktywne okno",
            "ukryj aktywne okno",
            "zmniejsz okno",
        ),
    )


def _is_minimize_window_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "minimize_window_under_cursor",
            "zminimalizuj okno pod kursorem",
            "zminimalizuj aplikacje pod kursorem",
            "minimalizuj okno pod kursorem",
            "minimalizuj aplikacje pod kursorem",
            "schowaj okno pod kursorem",
            "zwin okno pod kursorem",
        ),
    ):
        return True
    return (
        _contains_any(text, ("minimalizuj", "zminimalizuj", "schowaj", "ukryj", "zwin"))
        and _contains_any(text, ("okno", "aplikacj", "program"))
        and _contains_any(text, ("kursor", "myszk", "wskaz", "pod mysz", "to okno"))
    )


def _is_minimize_all_windows_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "minimize_all_windows",
            "minimize all",
            "zminimalizuj wszystkie",
            "zminimalizuj wszystkie okna",
            "schowaj wszystkie okna",
            "pokaz pulpit",
            "pokaz desktop",
            "show desktop",
            "pulpit",
            "desktop",
        ),
    ):
        return True
    return _contains_any(
        text,
        (
            "minimalizuj wszystko",
            "zwin wszystkie okna",
            "ukryj wszystkie okna",
            "schowaj wszystkie",
            "pokaz biurko",
        ),
    )


def _is_close_window_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "close_window_under_cursor",
            "zamknij okno pod kursorem",
            "zamknij aplikacje pod kursorem",
            "wylacz aplikacje pod kursorem",
            "zamknij program pod kursorem",
            "wylacz program pod kursorem",
            "zamknij wskazane okno",
            "wylacz wskazana aplikacje",
        ),
    ):
        return True
    return (
        _contains_any(text, ("zamknij", "wylacz", "zakoncz"))
        and _contains_any(text, ("okno", "aplikacj", "program"))
        and _contains_any(text, ("kursor", "myszk", "wskaz", "pod mysz", "to okno"))
    )


def _is_copy_selected_text_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "copy_selected_text",
            "kopiuj zaznaczenie",
            "skopiuj zaznaczenie",
            "zaznaczenie do schowka",
        ),
    ):
        return True
    if _contains_any(
        text,
        (
            "kopiuj zaznaczony tekst",
            "skopiuj zaznaczony tekst",
            "kopiuj zaznaczon",
            "copy selected text",
        ),
    ):
        return True
    return (
        _has_copy_verb(text)
        and "zaznacz" in text
        and _contains_any(text, ("tekst", "teskt", "teskty", "fragment"))
    )


def _is_copy_email_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "copy_email_under_cursor",
            "kopiuj email",
            "skopiuj email",
            "kopiuj mail",
            "skopiuj mail",
            "email do schowka",
        ),
    ):
        return True
    return (
        _has_copy_verb(text)
        and _contains_any(
            text,
            (
                "email",
                "e-mail",
                "mail",
                "adres poczty",
                "adres mailowy",
            ),
        )
        and _contains_any(
            text,
            (
                "kursor",
                "myszk",
                "wskaznik",
                "pod mysz",
                "gdzie mam kursor",
            ),
        )
    )


def _is_copy_number_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "copy_number_under_cursor",
            "kopiuj numer",
            "skopiuj numer",
            "numer do schowka",
        ),
    ):
        return True
    if _contains_any(
        text,
        (
            "kopiuj numer pod kursorem",
            "skopiuj numer pod kursorem",
            "kopiuj liczbe pod kursorem",
            "skopiuj liczbe pod kursorem",
            "copy number under cursor",
        ),
    ):
        return True
    return (
        _has_copy_verb(text)
        and _contains_any(text, ("numer", "liczb", "telefon", "nr "))
        and _contains_any(text, ("kursor", "myszk", "wskaznik", "pod mysz"))
    )


def _is_copy_sentence_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "copy_sentence_under_cursor",
            "kopiuj zdanie",
            "skopiuj zdanie",
            "zdanie do schowka",
        ),
    ):
        return True
    if _contains_any(
        text,
        (
            "kopiuj cale zdanie pod kursorem",
            "skopiuj cale zdanie pod kursorem",
            "skopiuj cale zdanie",
            "copy sentence under cursor",
        ),
    ):
        return True
    return (
        _has_copy_verb(text)
        and _contains_any(text, ("zdanie", "fraze", "linijke"))
        and _contains_any(text, ("kursor", "myszk", "wskaznik", "pod mysz"))
    )


def _is_copy_text_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "copy_text_under_cursor",
            "kopiuj tekst pod kursorem",
            "skopiuj tekst pod kursorem",
            "kopiuj pod kursorem",
            "skopiuj pod kursorem",
            "kopiuj spod kursora",
            "tekst pod kursorem do schowka",
        ),
    ):
        return True
    return (
        _has_copy_verb(text)
        and _contains_any(text, ("tekst", "fragment", "element", "to co"))
        and _contains_any(text, ("kursor", "myszk", "wskaznik", "pod mysz"))
    )


def _is_select_sentence_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "select_sentence_under_cursor",
            "zaznacz zdanie",
            "zaznacz zdanie pod kursorem",
            "wybierz zdanie pod kursorem",
            "zaznacz wskazane zdanie",
        ),
    ):
        return True
    return (
        _contains_any(text, ("zaznacz", "wybierz"))
        and _contains_any(text, ("zdanie", "fraze"))
        and _contains_any(text, ("kursor", "myszk", "wskaz", "to zdanie"))
    )


def _is_select_paragraph_under_cursor_command(text: str) -> bool:
    if _is_exact_alias(
        text,
        (
            "select_paragraph_under_cursor",
            "zaznacz akapit",
            "zaznacz akapit pod kursorem",
            "wybierz akapit pod kursorem",
            "zaznacz wskazany akapit",
        ),
    ):
        return True
    return (
        _contains_any(text, ("zaznacz", "wybierz"))
        and _contains_any(text, ("akapit", "paragraf"))
        and _contains_any(text, ("kursor", "myszk", "wskaz", "ten akapit"))
    )


def _is_capabilities_command(text: str) -> bool:
    return _is_exact_alias(
        text,
        (
            "list_capabilities",
            "co potrafisz",
            "co umiesz",
            "jakie masz komendy",
            "jakie znasz komendy",
            "jakie masz akcje",
            "pokaz mozliwosci",
            "wymien mozliwosci",
            "pomoc glosowa",
            "help",
        ),
    ) or (
        _contains_any(text, ("co", "jakie", "wymien", "pokaz"))
        and _contains_any(text, ("potrafisz", "umiesz", "komend", "akcj", "mozliw"))
    ) or (
        text.startswith(("czy potrafisz ", "czy umiesz "))
    ) or (
        _contains_any(text, ("jak mam powiedziec", "jak powiedziec", "co powiedziec"))
        and _contains_any(text, ("zeby", "abys", "polecen", "komend"))
    )


def _extract_rename_under_cursor(text: str) -> tuple[str | None, str] | None:
    if _is_exact_alias(
        text,
        (
            "rename_under_cursor",
            "zmien nazwe",
            "zmien nazwe pod kursorem",
            "przemianuj",
            "przemianuj pod kursorem",
            "zmien nazwe ikony",
            "rename under cursor",
        ),
    ):
        return None, "Rozpoczynam zmianę nazwy elementu pod kursorem."
    match = re.search(
        r"\b(?:zmien nazwe|przemianuj|nazwij)\b(?:\s+(?:pod kursorem|ikony|pliku|elementu))?"
        r"(?:\s+na)?\s+(.+)$",
        text,
    )
    if not match:
        if _contains_any(text, ("zmien nazwe", "przemianuj", "nazwij")) and _contains_any(
            text, ("kursor", "ikon", "plik", "element")
        ):
            return None, "Rozpoczynam zmianę nazwy elementu pod kursorem."
        return None
    new_name = match.group(1).strip(" .\"'")
    if not new_name or new_name in {"pod kursorem", "ikony", "pliku", "elementu"}:
        return None, "Rozpoczynam zmianę nazwy elementu pod kursorem."
    return new_name, f"Zmieniam nazwę pod kursorem na: {new_name}."


def _is_recent_activity_command(text: str) -> bool:
    if _is_exact_alias(text, ("recent_activity", "aktywnosc", "historia")):
        return True
    return _contains_any(
        text,
        (
            "co robilem",
            "co robilam",
            "co bylo na ekranie",
            "historia ekranu",
            "ostatnia aktywnosc",
            "ostatnie aktywnosci",
            "co ostatnio robilem",
            "co ostatnio robilam",
            "co sie dzialo",
            "pokaz historie",
            "podsumuj aktywnosc",
            "ostatnie okna",
        ),
    )


def _is_stop_command(text: str) -> bool:
    return text in {"stop", "stop now", "abort", "przerwij", "zatrzymaj"} or _contains_any(
        text,
        (
            "anuluj wszystko",
            "awaryjnie stop",
            "natychmiastowy stop",
            "panic stop",
            "stop wszystko",
        ),
    )


def _safe_paste_step(request: CommandRequest, raw: str) -> CommandPlan | None:
    if not request.text:
        return None
    match = re.match(
        r"^\s*(?:wpisz|napisz|wklej)(?:\s+to)?\s+do\s+"
        r"(chatgpt|gpt|gemini|cursora|cursor)\s*[:,-]?\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    target = normalize_text(match.group(1))
    content = match.group(2).strip()
    if not content:
        return None

    expected_window = {
        "chatgpt": "chatgpt",
        "gpt": "chatgpt",
        "gemini": "gemini",
        "cursor": "cursor",
        "cursora": "cursor",
    }.get(target, "")
    args: dict[str, object] = {"text": content}
    if expected_window:
        args["expected_window"] = expected_window
    return _single_step(
        request,
        intent="paste_text_safe",
        response_text="Wklejam tekst do wskazanego pola po weryfikacji celu.",
        action_id="paste_text_safe",
        args=args,
        risk=RiskLevel.MEDIUM,
    )


def _single_step(
    request: CommandRequest,
    *,
    intent: str,
    response_text: str,
    action_id: str,
    args: dict[str, object] | None = None,
    risk: RiskLevel = RiskLevel.LOW,
) -> CommandPlan:
    return CommandPlan(
        request_id=request.request_id,
        intent=intent,
        response_text=response_text,
        confidence=1.0,
        steps=[PlanStep(action_id=action_id, args=args or {}, risk=risk)],
        provider="deterministic",
    )
