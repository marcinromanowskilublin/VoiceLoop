from __future__ import annotations

import re
from dataclasses import dataclass

TAXONOMY_VERSION = "routing-taxonomy-v2"


@dataclass(frozen=True, slots=True)
class OperationTaxon:
    name: str
    document_label: str
    prompt: str
    synonyms: frozenset[str]
    document_pattern: re.Pattern[str] | None = None


@dataclass(frozen=True, slots=True)
class TargetTaxon:
    name: str
    document_label: str
    segment_pattern: re.Pattern[str] | None
    document_pattern: re.Pattern[str] | None = None


OPERATIONS: tuple[OperationTaxon, ...] = (
    OperationTaxon(
        "stop",
        "zatrzymać",
        "stop",
        frozenset({"stop", "przerwij", "zatrzymaj", "anuluj"}),
    ),
    OperationTaxon(
        "open",
        "otworzyć",
        "otwórz",
        frozenset(
            {
                "otworz",
                "otwieraj",
                "uruchom",
                "wlacz",
                "odpal",
                "wyswietl",
                "wejdz",
                "przejdz",
                "zacznij",
            }
        ),
        re.compile(r"\b(?:otworz|otwier\w*|wlacz)\b"),
    ),
    OperationTaxon(
        "close",
        "zamknąć",
        "zamknij",
        frozenset({"zamknij", "zamykaj", "zamkniecie", "wylacz", "zakoncz"}),
        re.compile(r"\b(?:zamknij|zamyk\w*)\b"),
    ),
    OperationTaxon(
        "minimize",
        "zminimalizować",
        "zminimalizuj",
        frozenset(
            {
                "minimalizuj",
                "zminimalizuj",
                "schowaj",
                "ukryj",
                "zwin",
                "odslon",
            }
        ),
        re.compile(r"\b(?:minimaliz\w*|zminimaliz\w*|schowaj)\b"),
    ),
    OperationTaxon(
        "copy",
        "skopiować",
        "skopiuj",
        frozenset({"kopiuj", "skopiuj", "przerzuc", "wez", "zlap"}),
        re.compile(r"\b(?:kopiuj|skopiuj|kopiow\w*)\b"),
    ),
    OperationTaxon(
        "select",
        "zaznaczyć",
        "zaznacz",
        frozenset({"zaznacz", "wybierz", "podswietl", "obejmij"}),
        re.compile(r"\b(?:zaznacz|wybierz|podswietl)\b"),
    ),
    OperationTaxon(
        "paste",
        "wkleić",
        "wklej",
        frozenset({"wklej", "wstaw", "wpisz", "napisz"}),
        re.compile(r"\b(?:wklej|wstaw|wpisz)\b"),
    ),
    OperationTaxon(
        "create_note",
        "zapisać",
        "zapisz",
        frozenset({"zapisz", "utworz", "stworz", "dodaj", "zanotuj"}),
        re.compile(r"\b(?:zapisz|zachowaj)\b"),
    ),
    OperationTaxon(
        "remember",
        "zapamiętać",
        "zapamiętaj",
        frozenset({"zapamietaj", "pamietaj", "zachowaj"}),
        re.compile(r"\b(?:zapamietaj|pamietaj)\b"),
    ),
    OperationTaxon(
        "search",
        "wyszukać",
        "wyszukaj",
        frozenset(
            {
                "wyszukaj",
                "szukaj",
                "znajdz",
                "poszukaj",
                "przejrzyj",
                "przeszukaj",
                "sprawdz",
            }
        ),
        re.compile(r"\b(?:wyszukaj|szukaj|znajdz|sprawdz)\b"),
    ),
    OperationTaxon(
        "describe",
        "opisać",
        "opisz",
        frozenset({"opisz", "podsumuj", "powiedz", "pokaz", "popatrz", "spojrz"}),
        re.compile(r"\b(?:opisz|podsumuj)\b"),
    ),
    OperationTaxon(
        "rename",
        "zmienić nazwę",
        "przemianuj",
        frozenset({"przemianuj", "nazwij"}),
        re.compile(r"\b(?:przemianuj|zmien\s+nazwe)\b"),
    ),
    OperationTaxon(
        "recall",
        "wyszukać w pamięci",
        "przypomnij",
        frozenset({"przypomnij"}),
    ),
    OperationTaxon(
        "run",
        "uruchomić",
        "uruchom",
        frozenset({"wykonaj"}),
        re.compile(r"\b(?:uruchom|odpal)\b"),
    ),
    OperationTaxon("send", "wysłać", "send", frozenset({"wyslij"})),
    OperationTaxon("list", "wymienić", "wymień", frozenset({"wymien"})),
    OperationTaxon("do", "zrobić", "do", frozenset({"zrob"})),
)

OPERATIONS_BY_NAME = {item.name: item for item in OPERATIONS}
OPERATION_WORDS: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (item.name, item.synonyms) for item in OPERATIONS
)
DOCUMENT_OPERATION_ORDER = (
    "open",
    "run",
    "close",
    "minimize",
    "copy",
    "select",
    "paste",
    "create_note",
    "remember",
    "search",
    "describe",
    "rename",
)


TARGETS: tuple[TargetTaxon, ...] = (
    TargetTaxon(
        "UI Vision",
        "UI.Vision",
        re.compile(r"\bui\s*vision\b"),
        re.compile(r"\bui\s*vision\b"),
    ),
    TargetTaxon(
        "YouTube",
        "YouTube",
        re.compile(r"\b(?:youtube|jutub|jutiub)\w*\b"),
        re.compile(r"\b(?:youtube|jutub|jutiub)\w*\b"),
    ),
    TargetTaxon(
        "WhatsApp",
        "WhatsApp",
        re.compile(r"\b(?:whatsapp|whats\s*app|whatsap)\b"),
        re.compile(r"\b(?:whatsapp|whats\s*app|whatsap)\b"),
    ),
    TargetTaxon(
        "this_pc",
        "Ten komputer",
        re.compile(
            r"\b(?:moj|ten)\s+komputer\b|"
            r"\b(?:eksplorator\w*|explorer)\b"
        ),
        re.compile(
            r"\b(?:moj|ten)\s+komputer\b|"
            r"\b(?:eksplorator\w*|explorer)\b"
        ),
    ),
    TargetTaxon(
        "ChatGPT",
        "ChatGPT",
        re.compile(r"\b(?:chat\s*gpt|chatgpt|gpt)\b"),
    ),
    TargetTaxon("Gemini", "Gemini", re.compile(r"\bgemini\w*\b")),
    TargetTaxon(
        "browser",
        "przeglądarka",
        re.compile(r"\b(?:chrom\w*|edge|firefox|brave|przegladark\w*)\b"),
        re.compile(r"\b(?:przegladark\w*|chrom\w*|edge|firefox|brave)\b"),
    ),
    TargetTaxon(
        "calendar",
        "kalendarz",
        re.compile(r"\b(?:kalendarz\w*|terminarz\w*|harmonogram\w*)\b"),
        re.compile(r"\bkalendarz\w*\b"),
    ),
    TargetTaxon(
        "window_under_cursor",
        "okno pod kursorem",
        re.compile(
            r"\b(?:okn\w*|aplikacj\w*|program\w*).*"
            r"\b(?:kursor\w*|mysz\w*|wskaz\w*|strzalk\w*)\b|"
            r"\b(?:kursor\w*|mysz\w*|wskaz\w*|strzalk\w*).*"
            r"\b(?:okn\w*|aplikacj\w*|program\w*)\b"
        ),
        re.compile(r"\b(?:pod|spod)\s+kursorem\b"),
    ),
    TargetTaxon(
        "active_window",
        "aktywne okno",
        re.compile(
            r"\b(?:aktywn\w*|aktualn\w*|biezac\w*)\s+okn\w*\b|"
            r"\bco\s+mam\s+otwarte\b|"
            r"\bjakie\s+okn\w*.*\botwart\w*\b"
        ),
        re.compile(r"\baktywn\w*\s+okn\w*\b"),
    ),
    TargetTaxon(
        "all_windows",
        "pulpit",
        re.compile(r"\b(?:wszystk\w*(?:\s+(?:okn\w*|program\w*))?|pulpit\w*|desktop)\b"),
        re.compile(r"\bpulpit\w*\b"),
    ),
    TargetTaxon(
        "email_under_cursor",
        "adres e-mail pod kursorem",
        re.compile(
            r"\b(?:e-?mail\w*|mail\w*|poczt\w*).*"
            r"\b(?:kursor\w*|mysz\w*|strzalk\w*)\b"
        ),
    ),
    TargetTaxon(
        "number_under_cursor",
        "numer pod kursorem",
        re.compile(
            r"\b(?:numer\w*|liczb\w*|telefon\w*|cyfr\w*).*"
            r"\b(?:kursor\w*|mysz\w*|strzalk\w*)\b"
        ),
    ),
    TargetTaxon(
        "sentence_under_cursor",
        "zdanie pod kursorem",
        re.compile(
            r"\b(?:zdani\w*|fraz\w*).*"
            r"\b(?:kursor\w*|mysz\w*|strzalk\w*)\b"
        ),
    ),
    TargetTaxon(
        "paragraph_under_cursor",
        "akapit pod kursorem",
        re.compile(
            r"\b(?:akapit\w*|paragraf\w*|blok\w*\s+tekst\w*).*"
            r"\b(?:kursor\w*|mysz\w*|strzalk\w*)\b"
        ),
    ),
    TargetTaxon(
        "text_under_cursor",
        "tekst pod kursorem",
        re.compile(
            r"\b(?:tekst\w*|fragment\w*|napis\w*|etykiet\w*).*"
            r"\b(?:kursor\w*|mysz\w*|strzalk\w*)\b"
        ),
    ),
    TargetTaxon(
        "selected_text",
        "zaznaczony tekst",
        re.compile(r"\b(?:zaznaczon\w*|zaznaczeni\w*|podswietlon\w*)\b"),
    ),
    TargetTaxon(
        "note",
        "notatka",
        re.compile(r"\b(?:notatk\w*|notatnik\w*|notes\w*)\b"),
        re.compile(r"\bnotatk\w*\b"),
    ),
    TargetTaxon(
        "memory",
        "pamięć",
        re.compile(r"\b(?:pamie\w*|zapisk\w*)\b"),
        re.compile(r"\bpamie\w*\b"),
    ),
    TargetTaxon(
        "web",
        "internet",
        re.compile(r"\b(?:internet\w*|sieci\w*|necie|online)\b"),
    ),
    TargetTaxon(
        "chat",
        "czat",
        re.compile(r"\b(?:czat\w*|chat\w*)\b"),
        re.compile(r"\b(?:czat\w*|chatgpt|gemini)\b"),
    ),
    TargetTaxon(
        "url",
        "adres internetowy",
        re.compile(
            r"\b(?:https?://|url\w*|link\w*|stron\w*)\b|"
            r"\b[a-z0-9-]+\s*\.\s*(?:pl|com|org|net|io|ai)\b"
        ),
        re.compile(r"\b(?:url|link\w*|stron\w*)\b"),
    ),
    TargetTaxon(
        "macro",
        "makro",
        re.compile(r"\b(?:makr\w*|automatyzacj\w*)\b"),
    ),
    TargetTaxon(
        "window",
        "okno",
        re.compile(r"\bokn\w*\b"),
        re.compile(r"\bokn\w*\b"),
    ),
    TargetTaxon(
        "browser_tab",
        "karta przeglądarki",
        None,
        re.compile(r"\bkart\w*\b"),
    ),
    TargetTaxon("text", "tekst", None, re.compile(r"\btekst\w*\b")),
    TargetTaxon(
        "email",
        "adres e-mail",
        None,
        re.compile(r"\b(?:e-?mail\w*|mail\w*)\b"),
    ),
    TargetTaxon("number", "numer", None, re.compile(r"\bnumer\w*\b")),
    TargetTaxon("sentence", "zdanie", None, re.compile(r"\bzdani\w*\b")),
    TargetTaxon(
        "paragraph",
        "akapit",
        None,
        re.compile(r"\b(?:akapit\w*|paragraf\w*)\b"),
    ),
)

TARGETS_BY_NAME = {item.name: item for item in TARGETS}
TARGET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (item.name, item.segment_pattern)
    for item in TARGETS
    if item.segment_pattern is not None
)
DOCUMENT_TARGET_ORDER = (
    "window",
    "active_window",
    "window_under_cursor",
    "browser",
    "YouTube",
    "WhatsApp",
    "this_pc",
    "browser_tab",
    "calendar",
    "chat",
    "text",
    "email",
    "number",
    "sentence",
    "paragraph",
    "note",
    "memory",
    "url",
    "UI Vision",
    "all_windows",
)

ACTION_OPERATIONS = {
    "open_calendar": "open",
    "open_browser": "open",
    "open_url": "open",
    "open_folder": "open",
    "open_app": "open",
    "open_chat": "open",
    "open_gpt_chat": "open",
    "open_gemini_chat": "open",
    "search_web": "search",
    "describe_active_window": "describe",
    "describe_recent_activity": "describe",
    "describe_text_target": "describe",
    "minimize_active_window": "minimize",
    "minimize_all_windows": "minimize",
    "minimize_window_under_cursor": "minimize",
    "close_window_under_cursor": "close",
    "copy_selected_text": "copy",
    "copy_text_under_cursor": "copy",
    "copy_email_under_cursor": "copy",
    "copy_number_under_cursor": "copy",
    "copy_sentence_under_cursor": "copy",
    "select_sentence_under_cursor": "select",
    "select_paragraph_under_cursor": "select",
    "rename_under_cursor": "rename",
    "paste_text_safe": "paste",
    "create_note": "create_note",
    "run_uivision_macro": "run",
    "remember": "remember",
    "remember_last_source": "remember",
    "recall": "recall",
}

ACTION_TARGETS: dict[str, frozenset[str]] = {
    "open_calendar": frozenset({"calendar"}),
    "open_browser": frozenset({"browser", "web"}),
    "open_url": frozenset({"YouTube", "url"}),
    "open_folder": frozenset({"this_pc"}),
    "open_app": frozenset({"WhatsApp"}),
    "open_chat": frozenset({"chat"}),
    "open_gpt_chat": frozenset({"ChatGPT"}),
    "open_gemini_chat": frozenset({"Gemini"}),
    "describe_active_window": frozenset({"active_window", "window"}),
    "minimize_active_window": frozenset({"active_window", "window"}),
    "minimize_all_windows": frozenset({"all_windows"}),
    "minimize_window_under_cursor": frozenset({"window_under_cursor"}),
    "close_window_under_cursor": frozenset({"window_under_cursor"}),
    "copy_selected_text": frozenset({"selected_text"}),
    "copy_text_under_cursor": frozenset({"text_under_cursor"}),
    "copy_email_under_cursor": frozenset({"email_under_cursor"}),
    "copy_number_under_cursor": frozenset({"number_under_cursor"}),
    "copy_sentence_under_cursor": frozenset({"sentence_under_cursor"}),
    "select_sentence_under_cursor": frozenset({"sentence_under_cursor"}),
    "select_paragraph_under_cursor": frozenset({"paragraph_under_cursor"}),
    "create_note": frozenset({"note"}),
    "run_uivision_macro": frozenset({"UI Vision", "macro"}),
    "recall": frozenset({"memory"}),
}

UNDER_CURSOR_ACTIONS = frozenset(
    {
        "close_window_under_cursor",
        "copy_email_under_cursor",
        "copy_number_under_cursor",
        "copy_sentence_under_cursor",
        "copy_text_under_cursor",
        "minimize_window_under_cursor",
        "rename_under_cursor",
        "select_paragraph_under_cursor",
        "select_sentence_under_cursor",
    }
)
CURSOR_MARKERS = ("kursor", "mysz", "strzalk", "wskaz", "pod nim", "pod nia")
OPEN_COORDINATION_TARGETS = frozenset(
    {
        "UI Vision",
        "YouTube",
        "ChatGPT",
        "Gemini",
        "WhatsApp",
        "browser",
        "calendar",
        "chat",
        "url",
        "this_pc",
    }
)

ACTION_TOKEN_LABELS = {
    "active": "aktywny",
    "all": "wszystkie",
    "app": "dozwolona aplikacja",
    "browser": "przeglądarka",
    "calendar": "kalendarz",
    "chat": "czat",
    "cursor": "kursor",
    "email": "adres e-mail",
    "folder": "Ten komputer",
    "gemini": "Gemini",
    "gpt": "ChatGPT",
    "last": "ostatni",
    "macro": "makro",
    "note": "notatka",
    "number": "numer",
    "paragraph": "akapit",
    "recent": "ostatnia aktywność",
    "safe": "bezpieczny cel tekstowy",
    "selected": "zaznaczony element",
    "sentence": "zdanie",
    "source": "źródło",
    "text": "tekst",
    "uivision": "UI.Vision",
    "under": "pod",
    "url": "adres internetowy",
    "web": "internet",
    "window": "okno",
    "windows": "okna",
}
ACTION_OPERATION_TOKEN_LABELS = {
    "close": "zamknąć",
    "copy": "skopiować",
    "create": "utworzyć",
    "describe": "opisać",
    "list": "wymienić",
    "minimize": "zminimalizować",
    "open": "otworzyć",
    "paste": "wkleić",
    "recall": "wyszukać w pamięci",
    "remember": "zapamiętać",
    "rename": "zmienić nazwę",
    "run": "uruchomić",
    "search": "wyszukać",
    "select": "zaznaczyć",
}


def operation_for_token(token: str) -> str | None:
    for operation, words in OPERATION_WORDS:
        if token in words:
            return operation
    return None


def operation_prompt(operation: str) -> str:
    item = OPERATIONS_BY_NAME.get(operation)
    return item.prompt if item is not None else operation


def operation_document_label(operation: str) -> str:
    item = OPERATIONS_BY_NAME.get(operation)
    return item.document_label if item is not None else operation


def action_operation_label(action_id: str) -> str:
    operation_token = action_id.split("_", 1)[0]
    return ACTION_OPERATION_TOKEN_LABELS.get(
        operation_token,
        operation_document_label(operation_token),
    )


def match_document_operation_labels(normalized_text: str) -> list[str]:
    return _unique(
        OPERATIONS_BY_NAME[name].document_label
        for name in DOCUMENT_OPERATION_ORDER
        if OPERATIONS_BY_NAME[name].document_pattern is not None
        and OPERATIONS_BY_NAME[name].document_pattern.search(normalized_text)
    )


def detect_target(normalized_text: str) -> str | None:
    for target, pattern in TARGET_PATTERNS:
        if pattern.search(normalized_text):
            return target
    return None


def target_document_label(target: str) -> str:
    item = TARGETS_BY_NAME.get(target)
    return item.document_label if item is not None else target


def match_document_target_labels(normalized_text: str) -> list[str]:
    return _unique(
        TARGETS_BY_NAME[name].document_label
        for name in DOCUMENT_TARGET_ORDER
        if TARGETS_BY_NAME[name].document_pattern is not None
        and TARGETS_BY_NAME[name].document_pattern.search(normalized_text)
    )


def action_target_labels(action_id: str) -> list[str]:
    return _unique(
        ACTION_TOKEN_LABELS.get(token, target_document_label(token))
        for token in action_id.split("_")[1:]
        if token
    )


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
