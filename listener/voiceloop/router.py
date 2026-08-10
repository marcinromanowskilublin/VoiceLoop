from __future__ import annotations

import re
import unicodedata

from .models import CommandPlan, CommandRequest, PlanStep, RiskLevel


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

    if text in {"ping", "voice_test", "voice test", "test petli", "test glosu", "test"}:
        return CommandPlan(
            request_id=request.request_id,
            intent="voice_test",
            response_text="Pętla VoiceLoop działa.",
            confidence=1.0,
            steps=[],
            provider="deterministic",
        )

    if _is_open_calendar_command(text):
        return _single_step(
            request,
            intent="open_calendar",
            response_text="Otwieram kalendarz.",
            action_id="open_calendar",
        )

    if _is_open_browser_command(text):
        return _single_step(
            request,
            intent="open_browser",
            response_text="Otwieram przeglądarkę.",
            action_id="open_browser",
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
            "kopiuj caly tekst pod kursorem",
            "skopiuj cale zdanie",
            "copy sentence under cursor",
        ),
    ):
        return True
    return (
        _has_copy_verb(text)
        and _contains_any(text, ("zdanie", "fraze", "linijke", "linijke", "tekst"))
        and _contains_any(text, ("kursor", "myszk", "wskaznik", "pod mysz"))
    )


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
