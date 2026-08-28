from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .manual_memory import ManualMemoryService
from .memory import MemoryStore
from .memory_vectorization import (
    MEMORY_VECTOR_NAMES,
    memory_query_documents,
    memory_query_weights,
)
from .models import (
    ActionResult,
    CommandPlan,
    MemoryCreate,
    MemoryItem,
    PlanStep,
    RiskLevel,
)
from .qdrant_memory import QdrantMemoryError, QdrantVectorStore
from .screenpipe import ScreenpipeClient, ScreenpipeError
from .settings import Settings
from .tts import WindowsTTS
from .web_search import WebSearchClient, WebSearchError

ActionHandler = Callable[[dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]
RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
LOGGER = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{4,}\d|\d+(?:[.,]\d+)?)(?!\w)")
EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}"
    r"(?![A-Za-z0-9._%+-])",
    re.IGNORECASE,
)
TEXT_CONTROL_TYPES = {"Edit", "Document"}
MAX_CURSOR_TEXT_CHARS = 8000
MAX_UIA_SELECTION_TEXT_CHARS = 10000
PROTECTED_CURSOR_WINDOW_CLASSES = {
    "Progman",
    "WorkerW",
    "Shell_TrayWnd",
    "Shell_SecondaryTrayWnd",
    "NotifyIconOverflowWindow",
}
BROWSER_PROCESS_NAMES = {
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "arc.exe",
}
ADDRESS_BAR_MARKERS = (
    "address and search",
    "address bar",
    "search or enter web address",
    "pasek adresu",
    "pole adresu",
    "adres i wyszukiwanie",
    "omnibox",
    "url",
    "location",
)
LAST_WEB_SOURCES_STATE_KEY = "search_web.last_sources.v1"
ALLOWED_FOLDERS = {
    "this_pc": "shell:MyComputerFolder",
}
ALLOWED_APPS = {
    "whatsapp": "whatsapp:",
}
VOICEATTACK_ACTION_ALIASES = {
    "active_window": "describe_active_window",
    "recent_activity": "describe_recent_activity",
}
CAPABILITY_LABELS = {
    "close_window_under_cursor": "zamykać okno pod kursorem",
    "copy_email_under_cursor": "kopiować e-mail pod kursorem",
    "copy_number_under_cursor": "kopiować numer pod kursorem",
    "copy_selected_text": "kopiować zaznaczony tekst",
    "copy_sentence_under_cursor": "kopiować zdanie pod kursorem",
    "copy_text_under_cursor": "kopiować tekst pod kursorem",
    "create_note": "tworzyć notatki",
    "describe_active_window": "opisywać aktywne okno",
    "describe_recent_activity": "opisywać ostatnią aktywność",
    "describe_text_target": "sprawdzać cel wpisywania",
    "list_capabilities": "wymieniać dostępne możliwości",
    "minimize_active_window": "minimalizować aktywne okno",
    "minimize_window_under_cursor": "minimalizować okno pod kursorem",
    "minimize_all_windows": "pokazywać pulpit",
    "open_app": "otwierać aplikację z allowlisty",
    "open_browser": "otwierać przeglądarkę",
    "open_calendar": "otwierać kalendarz",
    "open_chat": "otwierać czat",
    "open_folder": "otwierać Ten komputer",
    "open_gemini_chat": "otwierać Gemini",
    "open_gpt_chat": "otwierać ChatGPT",
    "open_url": "otwierać bezpieczny adres URL",
    "paste_text_safe": "bezpiecznie wklejać tekst",
    "recall": "przeszukiwać pamięć",
    "remember": "zapamiętywać informacje",
    "remember_last_source": "zapamiętywać ostatnie źródło",
    "rename_under_cursor": "zmieniać nazwę elementu pod kursorem",
    "run_uivision_macro": "uruchamiać dozwolone makra UI.Vision",
    "search_web": "wyszukiwać w internecie",
    "select_paragraph_under_cursor": "zaznaczać akapit pod kursorem",
    "select_sentence_under_cursor": "zaznaczać zdanie pod kursorem",
}


def _capability_category(action_id: str) -> str:
    if action_id.startswith(("open_", "run_")):
        return "aplikacje"
    if action_id.startswith(("copy_", "select_", "rename_", "minimize_", "close_")):
        return "okna i tekst"
    if action_id in {"describe_active_window", "describe_text_target"}:
        return "okna i tekst"
    if action_id in {"remember", "recall", "remember_last_source"}:
        return "pamięć"
    if action_id == "search_web":
        return "internet"
    if action_id == "describe_recent_activity":
        return "nagrania i aktywność"
    return "inne"


@dataclass(frozen=True)
class ActionSpec:
    id: str
    description: str
    args_schema: dict[str, Any]
    risk: RiskLevel
    confirmation_required: bool
    handler: ActionHandler
    execution_layer: int = 1
    routing_examples: tuple[str, ...] = ()
    category: str = ""
    spoken_name: str = ""
    positive_examples: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    result_reporting: str = "Odczytaj rzeczywisty ActionResult po wykonaniu."

    def public_definition(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "args_schema": self.args_schema,
            "risk": self.risk.value,
            "confirmation_required": self.confirmation_required,
            "execution_layer": self.execution_layer,
            "routing_examples": list(self.routing_examples),
            "category": self.category or _capability_category(self.id),
            "spoken_name": self.spoken_name or CAPABILITY_LABELS.get(
                self.id,
                self.description,
            ),
            "positive_examples": list(
                dict.fromkeys((*self.routing_examples, *self.positive_examples))
            ),
            "counterexamples": list(self.counterexamples),
            "requirements": list(self.requirements),
            "result_reporting": self.result_reporting,
        }


class ActionRegistry:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        tts: WindowsTTS,
        screenpipe: ScreenpipeClient | None = None,
        web_search: WebSearchClient | None = None,
        embeddings: OpenAICompatibleEmbeddingClient | None = None,
        qdrant: QdrantVectorStore | None = None,
        manual_memory: ManualMemoryService | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.tts = tts
        self.screenpipe = screenpipe or ScreenpipeClient(settings)
        self.web_search = web_search or WebSearchClient(settings)
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.manual_memory = manual_memory
        self._last_web_sources: dict[str, Any] | None = None
        self._current_process: asyncio.subprocess.Process | None = None
        self._specs: dict[str, ActionSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._register(
            ActionSpec(
                id="open_calendar",
                description="Otwiera lokalny kalendarz Windows.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_calendar,
                routing_examples=(
                    "otwórz kalendarz",
                    "pokaż kalendarz Windows",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_browser",
                description="Otwiera domyślną przeglądarkę.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_browser,
                routing_examples=(
                    "otwórz przeglądarkę",
                    "uruchom Chrome bez wskazanej strony",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_url",
                description="Otwiera jawny adres HTTP lub HTTPS.",
                args_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_url,
                routing_examples=(
                    "otwórz stronę internetową lub link w karcie przeglądarki",
                    "otwórz kartę z YouTube",
                    "otwórz devilpage.pl",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_folder",
                description="Otwiera Ten komputer w Eksploratorze. Nie przyjmuje dowolnej ścieżki.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "folder_id": {
                            "type": "string",
                            "enum": sorted(ALLOWED_FOLDERS),
                        }
                    },
                    "required": ["folder_id"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_folder,
                routing_examples=(
                    "otwórz mój komputer",
                    "otwórz ten komputer",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_app",
                description="Otwiera aplikację wyłącznie z allowlisty. Na start: WhatsApp.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "app_id": {
                            "type": "string",
                            "enum": sorted(ALLOWED_APPS),
                        }
                    },
                    "required": ["app_id"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_app,
                routing_examples=(
                    "otwórz WhatsApp",
                    "uruchom WhatsApp",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_chat",
                description="Otwiera stronę ChatGPT (alias historyczny).",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_chat,
                routing_examples=(
                    "otwórz czat",
                    "uruchom stronę czatu",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="search_web",
                description="Wyszukuje informacje w internecie i zwraca krótką listę wyników.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 2, "maxLength": 400},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                        "api_name": {"type": "string", "minLength": 2, "maxLength": 200},
                        "endpoint": {"type": "string", "minLength": 1, "maxLength": 300},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._search_web,
                routing_examples=(
                    "wyszukaj podany temat w internecie",
                    "sprawdź aktualne informacje online",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_gpt_chat",
                description="Otwiera stronę ChatGPT.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_gpt_chat,
                routing_examples=(
                    "otwórz ChatGPT",
                    "uruchom czat GPT",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="open_gemini_chat",
                description="Otwiera stronę Gemini.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_gemini_chat,
                routing_examples=(
                    "otwórz Gemini",
                    "uruchom czat Google Gemini",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="describe_active_window",
                description=(
                    "Odczytuje tytuł i nazwę procesu aktualnie aktywnego okna Windows. "
                    "Nie zapisuje danych, nie wykonuje kliknięć i nie tworzy zrzutu ekranu."
                ),
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._describe_active_window,
                routing_examples=(
                    "opisz aktywne okno",
                    "powiedz jaki program jest na wierzchu",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="minimize_active_window",
                description="Minimalizuje aktualnie aktywne okno Windows.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._minimize_active_window,
                routing_examples=(
                    "zminimalizuj aktywne okno",
                    "schowaj bieżący program",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="minimize_all_windows",
                description="Minimalizuje wszystkie otwarte okna i pokazuje pulpit.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._minimize_all_windows,
                routing_examples=(
                    "pokaż pulpit",
                    "zminimalizuj wszystkie okna",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="minimize_window_under_cursor",
                description="Minimalizuje okno aplikacji wskazywane kursorem.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._minimize_window_under_cursor,
                routing_examples=(
                    "zminimalizuj okno pod kursorem",
                    "schowaj aplikację wskazywaną myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="close_window_under_cursor",
                description=(
                    "Wysyła bezpieczne WM_CLOSE do okna aplikacji wskazywanego kursorem. "
                    "Nie zabija procesu i pozwala aplikacji wyświetlić pytanie o zapis."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "expected_hwnd": {"type": "integer", "minimum": 1},
                        "expected_process_id": {"type": "integer", "minimum": 1},
                        "expected_window_title": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                        "expected_process_name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                        "expected_class_name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                    },
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._close_window_under_cursor,
                routing_examples=(
                    "zamknij okno pod kursorem",
                    "wyślij WM_CLOSE do wskazanego okna",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="copy_selected_text",
                description="Kopiuje aktualnie zaznaczony tekst do schowka.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._copy_selected_text,
                routing_examples=(
                    "skopiuj zaznaczony tekst",
                    "kopiuj zaznaczenie do schowka",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="copy_text_under_cursor",
                description="Kopiuje dostępny tekst elementu wskazywanego kursorem.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._copy_text_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "skopiuj tekst pod kursorem",
                    "kopiuj napis wskazywany myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="copy_email_under_cursor",
                description=(
                    "Kopiuje pojedynczy adres e-mail wykryty w tekście elementu "
                    "wskazywanego kursorem."
                ),
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._copy_email_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "skopiuj e-mail pod kursorem",
                    "kopiuj adres poczty wskazywany myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="copy_number_under_cursor",
                description="Kopiuje numer wykryty w tekście pod kursorem do schowka.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._copy_number_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "skopiuj numer pod kursorem",
                    "kopiuj telefon wskazywany myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="copy_sentence_under_cursor",
                description="Kopiuje całe zdanie wykryte pod kursorem do schowka.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._copy_sentence_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "skopiuj zdanie pod kursorem",
                    "kopiuj całą frazę wskazywaną myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="select_sentence_under_cursor",
                description=(
                    "Zaznacza zdanie pod kursorem przez wzorzec tekstowy Windows UI Automation."
                ),
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._select_sentence_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "zaznacz zdanie pod kursorem",
                    "podświetl frazę wskazywaną myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="select_paragraph_under_cursor",
                description=(
                    "Zaznacza akapit pod kursorem przez wzorzec tekstowy Windows UI Automation."
                ),
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._select_paragraph_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "zaznacz akapit pod kursorem",
                    "podświetl paragraf wskazywany myszą",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="rename_under_cursor",
                description=(
                    "Rozpoczyna zmianę nazwy elementu pod kursorem (F2). "
                    "Opcjonalnie wkleja nową nazwę i zatwierdza Enterem."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "new_name": {"type": "string", "minLength": 1, "maxLength": 200},
                    },
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._rename_under_cursor,
                execution_layer=2,
                routing_examples=(
                    "zmień nazwę elementu pod kursorem",
                    "przemianuj wskazany plik",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="describe_text_target",
                description=(
                    "Sprawdza aktywne okno i pole pod kursorem, aby potwierdzić gdzie "
                    "trafi wpisywany tekst."
                ),
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._describe_text_target,
                execution_layer=2,
                routing_examples=(
                    "sprawdź cel wpisywania tekstu",
                    "czy to bezpieczne pole tekstowe",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="paste_text_safe",
                description=(
                    "Wkleja tekst do aktywnego pola tylko gdy cel wygląda bezpiecznie "
                    "(z blokadą paska adresu)."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1, "maxLength": 8000},
                        "expected_window": {"type": "string", "maxLength": 160},
                        "expected_app": {"type": "string", "maxLength": 80},
                        "allow_address_bar": {"type": "boolean"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._paste_text_safe,
                execution_layer=2,
                routing_examples=(
                    "wklej podany tekst do bezpiecznego pola",
                    "wpisz treść po sprawdzeniu aktywnego okna",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="describe_recent_activity",
                description=(
                    "Odczytuje z lokalnego Screenpipe listę aplikacji i okien z ostatniego "
                    "okresu. To akcja tylko do odczytu; nie pobiera obrazu ani audio."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "minutes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20160,
                        }
                    },
                    "required": ["minutes"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._describe_recent_activity,
                routing_examples=(
                    "opisz ostatnią aktywność",
                    "podsumuj co było na ekranie",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="create_note",
                description="Uruchamia UI.Vision i wpisuje przekazaną treść do Notatnika.",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=False,
                handler=self._create_note,
                execution_layer=3,
                routing_examples=(
                    "utwórz notatkę z podaną treścią",
                    "zapisz tekst w Notatniku",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="run_uivision_macro",
                description="Uruchamia istniejące, dozwolone makro UI.Vision.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "macro": {
                            "type": "string",
                            "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$",
                            "maxLength": 160,
                        },
                        "var1": {"type": "string"},
                        "var2": {"type": "string"},
                        "var3": {"type": "string"},
                    },
                    "required": ["macro"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._run_uivision_macro,
                execution_layer=3,
                routing_examples=(
                    "uruchom makro UI Vision o podanej nazwie",
                    "wykonaj dozwolone makro automatyzacji",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="remember",
                description="Zapisuje fakt lub preferencję w lokalnej pamięci.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "kind": {"type": "string"},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._remember,
                routing_examples=(
                    "zapamiętaj podaną informację",
                    "zapisz preferencję w lokalnej pamięci",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="remember_last_source",
                description=(
                    "Zapisuje w pamięci ostatnie źródło z wyszukiwania internetowego "
                    "lub sprawdzania dokumentacji API."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "minimum": 1, "maximum": 20},
                        "note": {"type": "string", "maxLength": 500},
                        "kind": {"type": "string", "maxLength": 50},
                    },
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._remember_last_source,
                routing_examples=(
                    "zapamiętaj ostatnie źródło z wyszukiwania",
                    "zapisz wybrany link z poprzednich wyników",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="recall",
                description="Wyszukuje pasujące wpisy lokalnej pamięci.",
                args_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._recall,
                routing_examples=(
                    "wyszukaj hasło w lokalnej pamięci",
                    "przypomnij zapisane informacje na dany temat",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="list_capabilities",
                description=(
                    "Wymienia osobno szybkie komendy VoiceAttack i własne akcje VoiceLoop."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 500},
                    },
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._list_capabilities,
                routing_examples=(
                    "co potrafisz",
                    "jakie masz możliwości",
                    "czy umiesz otwierać aplikacje",
                    "jak mam powiedzieć żeby otworzyć WhatsApp",
                ),
            )
        )
        self._register(
            ActionSpec(
                id="speak_text",
                description="Wypowiada krótki tekst po polsku przez lokalny głos Windows.",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._speak_text,
            )
        )

    def _register(self, spec: ActionSpec) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,79}", spec.id):
            raise ValueError(f"Nieprawidłowy action_id: {spec.id!r}")
        if spec.id in self._specs:
            raise ValueError(f"Duplikat action_id: {spec.id}")
        if spec.args_schema.get("type") != "object":
            raise ValueError(f"Akcja {spec.id} musi mieć obiektowy args_schema")
        if not spec.description.strip():
            raise ValueError(f"Akcja {spec.id} musi mieć opis")
        if spec.risk is RiskLevel.HIGH and not spec.confirmation_required:
            raise ValueError(f"Akcja wysokiego ryzyka {spec.id} wymaga potwierdzenia")
        self._specs[spec.id] = spec

    def definitions(self) -> list[dict[str, Any]]:
        voiceattack_actions = self.voiceattack_action_ids()
        definitions = []
        for spec in self._specs.values():
            definition = spec.public_definition()
            definition["available_in_voiceattack"] = spec.id in voiceattack_actions
            definitions.append(definition)
        return definitions

    def voiceattack_command_ids(self) -> set[str]:
        scripts_dir = self.settings.project_root / "scripts" / "va"
        command_ids: set[str] = set()
        if not scripts_dir.is_dir():
            return command_ids
        pattern = re.compile(r"-CommandId\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
        for path in scripts_dir.glob("*.vbs"):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            match = pattern.search(content)
            if match:
                command_ids.add(match.group(1))
        return command_ids

    def voiceattack_action_ids(self) -> set[str]:
        return {
            VOICEATTACK_ACTION_ALIASES.get(command_id, command_id)
            for command_id in self.voiceattack_command_ids()
            if VOICEATTACK_ACTION_ALIASES.get(command_id, command_id) in self._specs
        }

    def capability_catalog(self) -> dict[str, Any]:
        va_ids = self.voiceattack_action_ids()
        actions = [
            {
                **spec.public_definition(),
                "label": CAPABILITY_LABELS.get(spec.id, spec.description),
                "available_in_voiceattack": spec.id in va_ids,
            }
            for spec in self._specs.values()
            if spec.id not in {"speak_text", "list_capabilities"}
        ]
        return {
            "voiceattack_command_ids": sorted(self.voiceattack_command_ids()),
            "voiceattack_actions": [item for item in actions if item["available_in_voiceattack"]],
            "voiceloop_actions": actions,
            "native_only_actions": [
                item for item in actions if not item["available_in_voiceattack"]
            ],
        }

    def has_action(self, action_id: str) -> bool:
        return action_id in self._specs

    async def bind_execution_targets(self, plan: CommandPlan) -> CommandPlan:
        for step in plan.steps:
            if step.action_id != "close_window_under_cursor":
                continue
            info = await asyncio.to_thread(self._window_under_cursor_info_sync)
            step.args = {
                "expected_hwnd": int(info["hwnd"]),
                "expected_process_id": int(info["process_id"]),
                "expected_window_title": str(info["window_title"]),
                "expected_process_name": str(info.get("process_name") or ""),
                "expected_class_name": str(info["class_name"]),
            }
        return plan

    def enforce_policy(self, step: PlanStep) -> PlanStep:
        spec = self._specs.get(step.action_id)
        if spec is None:
            raise ValueError(f"unknown action: {step.action_id}")
        from .routing.validation import validate_arguments

        argument_errors = validate_arguments(step.args, spec.args_schema)
        if argument_errors:
            raise ValueError(f"invalid arguments for {step.action_id}: {argument_errors[0]}")
        if RISK_ORDER[spec.risk] > RISK_ORDER[step.risk]:
            step.risk = spec.risk
        step.confirmation_required = (
            step.confirmation_required or spec.confirmation_required or step.risk is RiskLevel.HIGH
        )
        return step

    async def execute(self, step: PlanStep) -> ActionResult:
        spec = self._specs.get(step.action_id)
        if spec is None:
            return ActionResult(
                action_id=step.action_id,
                success=False,
                message=f"Nieznana akcja: {step.action_id}",
            )
        from .routing.validation import validate_arguments

        argument_errors = validate_arguments(step.args, spec.args_schema)
        if argument_errors:
            return ActionResult(
                action_id=step.action_id,
                success=False,
                message=(f"Nieprawidłowe argumenty akcji {step.action_id}: {argument_errors[0]}"),
            )
        started = time.perf_counter()
        try:
            message, data = await spec.handler(step.args)
            success = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc)
            data = {}
            success = False
        duration_ms = int((time.perf_counter() - started) * 1000)
        return ActionResult(
            action_id=step.action_id,
            success=success,
            message=message,
            data=data,
            duration_ms=duration_ms,
        )

    async def stop(self) -> None:
        await self.tts.stop()
        process = self._current_process
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._current_process = None

    async def speak(self, text: str) -> None:
        await self.tts.speak(text)

    async def _open_calendar(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        await asyncio.to_thread(os.startfile, "outlookcal:")
        return "Otwarto kalendarz.", {}

    async def _open_browser(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        opened = await asyncio.to_thread(webbrowser.open, "about:blank", 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć przeglądarki.")
        return "Otwarto przeglądarkę.", {}

    async def _open_chat(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await self._open_gpt_chat(_)

    async def _open_gpt_chat(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        opened = await asyncio.to_thread(webbrowser.open, "https://chatgpt.com", 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć strony.")
        return "Otwarto ChatGPT.", {"url": "https://chatgpt.com"}

    async def _open_gemini_chat(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        opened = await asyncio.to_thread(webbrowser.open, "https://gemini.google.com/app", 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć strony.")
        return "Otwarto Gemini.", {"url": "https://gemini.google.com/app"}

    async def _search_web(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        query = str(args.get("query") or "").strip()
        if len(query) < 2:
            raise ValueError("Zapytanie do internetu jest zbyt krótkie.")
        api_name = str(args.get("api_name") or "").strip()
        endpoint = str(args.get("endpoint") or "").strip()
        try:
            requested_limit = int(args.get("limit", self.settings.web_search_max_results))
        except (TypeError, ValueError) as exc:
            raise ValueError("Parametr limit musi być liczbą całkowitą.") from exc
        limit = max(1, min(requested_limit, 10))

        if api_name and endpoint:
            try:
                inspection = await self.web_search.inspect_endpoint_in_documentation(
                    api_name=api_name,
                    endpoint=endpoint,
                    limit=limit,
                )
            except WebSearchError as exc:
                raise RuntimeError(str(exc)) from exc

            docs_access = inspection.get("documentation_access", {})
            docs_access_label = str(docs_access.get("availability_label") or "nieznana").strip()
            checked_count = len(inspection.get("checked_sources", []))
            matched_source = inspection.get("matched_source") or {}
            matched_url = str(matched_source.get("url") or "").strip()

            if inspection.get("endpoint_found"):
                location = matched_url or "w dokumentacji online"
                message = (
                    f"Sprawdziłem API {api_name}. Endpoint „{endpoint}” wygląda na "
                    f"dostępny ({location}). Dokumentacja: {docs_access_label}."
                )
            else:
                message = (
                    f"Sprawdziłem API {api_name}, ale nie potwierdziłem endpointu "
                    f"„{endpoint}” w {checked_count} źródłach. Dokumentacja: "
                    f"{docs_access_label}."
                )

            await self._store_last_web_sources(
                query=str(inspection.get("query") or query),
                results=inspection.get("results", []),
                endpoint_check={
                    "api_name": api_name,
                    "endpoint": endpoint,
                    "endpoint_found": bool(inspection.get("endpoint_found")),
                    "documentation_access": docs_access,
                },
            )

            return (
                message,
                {
                    "query": inspection.get("query", query),
                    "api_name": api_name,
                    "endpoint": endpoint,
                    "endpoint_check": inspection,
                    "results": inspection.get("results", []),
                },
            )

        try:
            results = await self.web_search.search(query, limit=limit)
        except WebSearchError as exc:
            raise RuntimeError(str(exc)) from exc
        serialized_results = [item.to_dict() for item in results]
        await self._store_last_web_sources(query=query, results=serialized_results)
        if not results:
            return (
                f"Nie znalazłem wyników dla: {query}.",
                {"query": query, "results": []},
            )
        preview = "; ".join(f"{index + 1}. {item.title}" for index, item in enumerate(results[:3]))
        message = f"Znalazłem {len(results)} wyników dla „{query}”: {preview}."
        return (
            message,
            {
                "query": query,
                "results": serialized_results,
            },
        )

    async def _describe_active_window(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        message, data = await asyncio.to_thread(self._describe_active_window_sync)
        if data["window_title"] or data["process_name"]:
            return message, data

        try:
            context = await self.screenpipe.recent_context()
        except ScreenpipeError:
            context = None
        if context is None:
            return message, data

        parts = ["Według ostatniego zapisu Screenpipe"]
        if context.window_name:
            parts.append(f"aktywne było okno: {context.window_name}")
        if context.app_name:
            parts.append(f"program: {context.app_name}")
        return (
            ". ".join(parts) + ".",
            {
                "window_title": context.window_name,
                "process_name": context.app_name,
                "source": "screenpipe",
                "timestamp": context.timestamp,
            },
        )

    async def _minimize_active_window(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        message, data = await asyncio.to_thread(self._minimize_active_window_sync)
        return message, data

    async def _minimize_all_windows(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        message, data = await asyncio.to_thread(self._minimize_all_windows_sync)
        return message, data

    async def _minimize_window_under_cursor(
        self,
        _: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._minimize_window_under_cursor_sync)

    async def _close_window_under_cursor(
        self,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._close_window_under_cursor_sync, args)

    async def _copy_selected_text(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._copy_selected_text_sync)

    async def _copy_text_under_cursor(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._copy_text_under_cursor_sync)

    async def _copy_email_under_cursor(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._copy_email_under_cursor_sync)

    async def _copy_number_under_cursor(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._copy_number_under_cursor_sync)

    async def _copy_sentence_under_cursor(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._copy_sentence_under_cursor_sync)

    async def _select_sentence_under_cursor(
        self,
        _: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._select_sentence_under_cursor_sync)

    async def _select_paragraph_under_cursor(
        self,
        _: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._select_paragraph_under_cursor_sync)

    async def _rename_under_cursor(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._rename_under_cursor_sync, args)

    async def _describe_text_target(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._describe_text_target_sync)

    async def _paste_text_safe(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(self._paste_text_safe_sync, args)

    async def _describe_recent_activity(
        self,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        try:
            requested_minutes = int(args.get("minutes", 30))
        except (TypeError, ValueError) as exc:
            raise ValueError("Parametr minutes musi być liczbą całkowitą.") from exc
        max_minutes = max(1, self.settings.screenpipe_lookback_days) * 24 * 60
        minutes = max(1, min(requested_minutes, max_minutes))
        contexts = await self.screenpipe.recent_activity(minutes=minutes)
        if not contexts:
            return (
                f"Screenpipe nie znalazł aktywności z ostatnich {minutes} minut.",
                {"minutes": minutes, "items": []},
            )

        spoken = contexts[-8:]
        labels = [
            " — ".join(part for part in (item.app_name, item.window_name) if part)
            for item in spoken
        ]
        message = (
            f"W ostatnich {minutes} minutach Screenpipe zapisał między innymi: "
            + "; ".join(label for label in labels if label)
            + "."
        )
        return (
            message,
            {
                "minutes": minutes,
                "items": [
                    {
                        "app_name": item.app_name,
                        "window_name": item.window_name,
                        "timestamp": item.timestamp,
                    }
                    for item in contexts
                ],
            },
        )

    @staticmethod
    def _describe_active_window_sync() -> tuple[str, dict[str, Any]]:
        try:
            import win32api
            import win32con
            import win32gui
            import win32process
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do odczytu aktywnego okna: {exc}") from exc

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            raise RuntimeError("Nie znaleziono aktywnego okna.")

        title = (win32gui.GetWindowText(hwnd) or "").strip()
        process_name = ""
        try:
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                False,
                process_id,
            )
            try:
                process_name = os.path.basename(win32process.GetModuleFileNameEx(handle, 0))
            finally:
                win32api.CloseHandle(handle)
        except Exception:
            process_name = ""

        if title and process_name:
            message = f"Aktywne okno: {title}. Program: {process_name}."
        elif title:
            message = f"Aktywne okno: {title}."
        elif process_name:
            message = f"Aktywny program: {process_name}."
        else:
            message = "Aktywne okno nie ma dostępnego tytułu ani nazwy procesu."
        return message, {"window_title": title, "process_name": process_name}

    @staticmethod
    def _minimize_active_window_sync() -> tuple[str, dict[str, Any]]:
        try:
            import win32con
            import win32gui
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do minimalizacji okna: {exc}") from exc

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            raise RuntimeError("Nie znaleziono aktywnego okna.")
        title = (win32gui.GetWindowText(hwnd) or "").strip()
        if win32gui.IsIconic(hwnd):
            message = (
                f"Okno „{title}” jest już zminimalizowane."
                if title
                else "Aktywne okno jest już zminimalizowane."
            )
            return message, {"window_title": title, "already_minimized": True}

        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        message = f"Zminimalizowałem okno „{title}”." if title else "Zminimalizowałem aktywne okno."
        return message, {"window_title": title, "already_minimized": False}

    @staticmethod
    def _minimize_all_windows_sync() -> tuple[str, dict[str, Any]]:
        try:
            import win32com.client
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do minimalizacji okien: {exc}") from exc

        shell = win32com.client.Dispatch("Shell.Application")
        shell.MinimizeAll()
        return "Zminimalizowałem wszystkie okna.", {"mode": "minimize_all"}

    @staticmethod
    def _minimize_window_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        try:
            import win32con
            import win32gui
        except ImportError as exc:
            raise RuntimeError(
                f"Brak zależności Windows do minimalizacji okna pod kursorem: {exc}"
            ) from exc

        info = ActionRegistry._window_under_cursor_info_sync()
        hwnd = int(info["hwnd"])
        title = str(info.get("window_title") or "")
        process_name = str(info.get("process_name") or "")
        label = title or process_name or "wskazane okno"
        if win32gui.IsIconic(hwnd):
            return (
                f"Okno „{label}” jest już zminimalizowane.",
                {**info, "already_minimized": True},
            )
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        return (
            f"Zminimalizowałem okno „{label}” wskazywane kursorem.",
            {**info, "already_minimized": False},
        )

    @staticmethod
    def _close_window_under_cursor_sync(
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        expected_hwnd = int(args.get("expected_hwnd") or 0)
        expected_process_id = int(args.get("expected_process_id") or 0)
        expected_class_name = str(args.get("expected_class_name") or "")
        expected_process_name = str(args.get("expected_process_name") or "")
        if (
            expected_hwnd <= 0
            or expected_process_id <= 0
            or not expected_class_name
            or not expected_process_name
        ):
            raise RuntimeError("Brak związanej tożsamości okna. Wydaj polecenie ponownie.")

        info = ActionRegistry._window_info_sync(expected_hwnd)
        if int(info["process_id"]) != expected_process_id:
            raise RuntimeError("Tożsamość wskazanego okna zmieniła się po potwierdzeniu.")
        if str(info["class_name"]).casefold() != expected_class_name.casefold():
            raise RuntimeError("Klasa wskazanego okna zmieniła się po potwierdzeniu.")
        current_process_name = str(info.get("process_name") or "")
        if (
            expected_process_name
            and current_process_name.casefold() != expected_process_name.casefold()
        ):
            raise RuntimeError("Proces wskazanego okna zmienił się po potwierdzeniu.")
        ActionRegistry._post_close_window_sync(expected_hwnd)
        title = str(info.get("window_title") or "")
        process_name = current_process_name
        label = title or process_name or "wskazane okno"
        return (
            f"Wysłałem prośbę o zamknięcie okna „{label}”.",
            {
                "hwnd": expected_hwnd,
                "process_id": expected_process_id,
                "window_title": title,
                "process_name": process_name,
                "class_name": str(info.get("class_name") or ""),
                "mode": "wm_close",
            },
        )

    @staticmethod
    def _window_under_cursor_info_sync() -> dict[str, Any]:
        try:
            import win32api
            import win32con
            import win32gui
        except ImportError as exc:
            raise RuntimeError(
                f"Brak zależności Windows do zamknięcia okna pod kursorem: {exc}"
            ) from exc

        cursor = win32api.GetCursorPos()
        child_hwnd = win32gui.WindowFromPoint(cursor)
        if not child_hwnd:
            raise RuntimeError("Nie znalazłem okna pod kursorem.")
        hwnd = win32gui.GetAncestor(child_hwnd, getattr(win32con, "GA_ROOT", 2)) or child_hwnd
        return ActionRegistry._window_info_sync(int(hwnd))

    @staticmethod
    def _window_info_sync(hwnd: int) -> dict[str, Any]:
        try:
            import win32api
            import win32con
            import win32gui
            import win32process
        except ImportError as exc:
            raise RuntimeError(
                f"Brak zależności Windows do sprawdzenia tożsamości okna: {exc}"
            ) from exc

        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            raise RuntimeError("Wskazane okno nie jest już dostępne.")

        class_name = (win32gui.GetClassName(hwnd) or "").strip()
        if class_name in PROTECTED_CURSOR_WINDOW_CLASSES:
            raise RuntimeError("Nie zamykam pulpitu, paska zadań ani powłoki systemowej.")

        title = (win32gui.GetWindowText(hwnd) or "").strip()
        process_name = ""
        process_id = 0
        try:
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                False,
                process_id,
            )
            try:
                process_name = os.path.basename(win32process.GetModuleFileNameEx(handle, 0))
            finally:
                win32api.CloseHandle(handle)
        except Exception:
            process_name = ""
        if process_id <= 0:
            raise RuntimeError("Nie mogę potwierdzić procesu wskazanego okna.")
        if not process_name:
            raise RuntimeError("Nie mogę potwierdzić nazwy procesu wskazanego okna.")
        if not title:
            raise RuntimeError(
                "Wskazane okno nie ma tytułu. Dla bezpieczeństwa nie wysyłam zamknięcia."
            )
        return {
            "hwnd": int(hwnd),
            "process_id": int(process_id),
            "window_title": title,
            "process_name": process_name,
            "class_name": class_name,
        }

    @staticmethod
    def _post_close_window_sync(hwnd: int) -> None:
        try:
            import win32con
            import win32gui
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do zamknięcia okna: {exc}") from exc
        if not win32gui.IsWindow(hwnd):
            raise RuntimeError("Wskazane okno nie jest już dostępne.")
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    @staticmethod
    def _describe_text_target_sync() -> tuple[str, dict[str, Any]]:
        info = ActionRegistry._text_target_info_sync()
        window_title = str(info.get("window_title") or "brak")
        field_name = str(info.get("field_name") or "brak nazwy")
        if info.get("looks_like_address_bar"):
            message = (
                f"Wykryłem pasek adresu lub wyszukiwania w oknie „{window_title}” "
                f"(pole: „{field_name}”). Nie wpisuję tam tekstu."
            )
        elif info.get("is_editable"):
            message = f"Pisanie wygląda bezpiecznie w oknie „{window_title}”, pole: „{field_name}”."
        else:
            message = (
                f"Nie wykryłem aktywnego pola tekstowego w oknie „{window_title}”. "
                "Najpierw kliknij docelowe pole wpisywania."
            )
        return message, info

    @staticmethod
    def _paste_text_safe_sync(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = str(args.get("text") or "")
        if not text.strip():
            raise ValueError("Tekst do wpisania jest pusty.")

        expected_window = str(args.get("expected_window") or "").strip().casefold()
        expected_app = str(args.get("expected_app") or "").strip().casefold()
        allow_address_bar = bool(args.get("allow_address_bar", False))
        info = ActionRegistry._text_target_info_sync()
        window_title = str(info.get("window_title") or "")
        process_name = str(info.get("process_name") or "")

        if expected_window and expected_window not in window_title.casefold():
            raise RuntimeError(
                f"Aktywne okno to „{window_title}”, a oczekiwane zawiera „{expected_window}”."
            )
        if (
            expected_app
            and expected_app not in process_name.casefold()
            and expected_app not in window_title.casefold()
        ):
            raise RuntimeError(
                f"Aktywna aplikacja to „{process_name or window_title}”, "
                f"a oczekiwana zawiera „{expected_app}”."
            )
        if not info.get("is_editable"):
            raise RuntimeError(
                "Nie wykryłem aktywnego pola tekstowego pod kursorem. "
                "Kliknij pole wpisywania i spróbuj ponownie."
            )
        if info.get("looks_like_address_bar") and not allow_address_bar:
            raise RuntimeError(
                "Wykryłem pasek adresu lub wyszukiwania. "
                "Dla bezpieczeństwa przerwałem wpisywanie tekstu."
            )

        ActionRegistry._write_clipboard_text_sync(text)
        ActionRegistry._send_paste_shortcut_sync()
        field_name = str(info.get("field_name") or "pole tekstowe")
        return (
            f"Wkleiłem tekst do pola „{field_name}” w oknie „{window_title or 'bez tytułu'}”.",
            {
                "window_title": window_title,
                "process_name": process_name,
                "field_name": field_name,
                "characters": len(text),
                "safe_for_typing": bool(info.get("safe_for_typing")),
            },
        )

    @staticmethod
    def _text_target_info_sync() -> dict[str, Any]:
        try:
            import win32api
            import win32con
            import win32gui
            import win32process
        except ImportError as exc:
            raise RuntimeError(
                f"Brak zależności Windows do weryfikacji pola tekstowego: {exc}"
            ) from exc

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            raise RuntimeError("Nie znaleziono aktywnego okna.")
        window_title = (win32gui.GetWindowText(hwnd) or "").strip()

        process_name = ""
        try:
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                False,
                process_id,
            )
            try:
                process_name = os.path.basename(win32process.GetModuleFileNameEx(handle, 0))
            finally:
                win32api.CloseHandle(handle)
        except Exception:
            process_name = ""

        cursor_x, cursor_y = win32api.GetCursorPos()
        field_name = ""
        control_type = ""
        automation_id = ""
        class_name = ""
        try:
            from pywinauto import Desktop

            wrapper = Desktop(backend="uia").from_point(cursor_x, cursor_y)
            field_name = (getattr(wrapper, "window_text", lambda: "")() or "").strip()
            element_info = getattr(wrapper, "element_info", None)
            if element_info is not None:
                element_name = str(getattr(element_info, "name", "") or "").strip()
                if element_name:
                    field_name = element_name
                control_type = str(getattr(element_info, "control_type", "") or "").strip()
                automation_id = str(getattr(element_info, "automation_id", "") or "").strip()
                class_name = str(getattr(element_info, "class_name", "") or "").strip()
        except Exception:
            pass

        control_lower = control_type.casefold()
        class_lower = class_name.casefold()
        is_editable = control_type in TEXT_CONTROL_TYPES or class_lower.startswith("edit")
        combined = " ".join(
            item
            for item in (
                field_name.casefold(),
                automation_id.casefold(),
                class_lower,
                control_lower,
            )
            if item
        )
        looks_like_address_bar = any(marker in combined for marker in ADDRESS_BAR_MARKERS)
        process_lower = process_name.casefold()
        window_lower = window_title.casefold()
        if (
            not looks_like_address_bar
            and process_lower in BROWSER_PROCESS_NAMES
            and "search" in combined
            and "chatgpt" not in window_lower
            and "gemini" not in window_lower
        ):
            looks_like_address_bar = True

        safe_for_typing = bool(is_editable and not looks_like_address_bar)
        return {
            "window_title": window_title,
            "process_name": process_name,
            "field_name": field_name,
            "control_type": control_type,
            "automation_id": automation_id,
            "class_name": class_name,
            "cursor_x": cursor_x,
            "cursor_y": cursor_y,
            "is_editable": bool(is_editable),
            "looks_like_address_bar": bool(looks_like_address_bar),
            "safe_for_typing": safe_for_typing,
        }

    @staticmethod
    def _copy_selected_text_sync() -> tuple[str, dict[str, Any]]:
        previous = ActionRegistry._read_clipboard_text_sync()
        ActionRegistry._send_copy_shortcut_sync()
        copied = ""
        for _ in range(10):
            time.sleep(0.05)
            copied = ActionRegistry._read_clipboard_text_sync()
            if copied.strip():
                break
        copied = copied.strip()
        if not copied:
            raise RuntimeError(
                "Nie wykryłem zaznaczonego tekstu do skopiowania. Najpierw zaznacz tekst."
            )

        if previous.strip() and copied == previous.strip():
            message = "Zaznaczony tekst jest już w schowku."
        else:
            message = "Skopiowałem zaznaczony tekst."
        return message, {"text": copied, "source": "selection"}

    @staticmethod
    def _copy_text_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        candidates = ActionRegistry._text_candidates_under_cursor_sync()
        if not candidates:
            raise RuntimeError(
                "Nie wykryłem tekstu pod kursorem. Wskaż dokładniej element z tekstem."
            )
        text = re.sub(r"\s+", " ", candidates[0]).strip()
        if len(text) > MAX_CURSOR_TEXT_CHARS:
            raise RuntimeError("Tekst pod kursorem jest zbyt długi. Zaznacz mniejszy fragment.")
        ActionRegistry._write_clipboard_text_sync(text)
        return (
            "Skopiowałem tekst spod kursora.",
            {"text": text, "source": "cursor_text"},
        )

    @staticmethod
    def _copy_email_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        candidates = ActionRegistry._text_candidates_under_cursor_sync()
        found_multiple = False
        for candidate in candidates:
            emails = ActionRegistry._extract_emails_from_text(candidate)
            if len(emails) == 1:
                email = emails[0]
                ActionRegistry._write_clipboard_text_sync(email)
                return (
                    "Skopiowałem adres e-mail.",
                    {"email": email, "source": "cursor_text"},
                )
            if len(emails) > 1:
                found_multiple = True

        if found_multiple:
            raise RuntimeError(
                "Pod kursorem wykryłem kilka adresów e-mail. Wskaż dokładniej jeden adres."
            )
        raise RuntimeError(
            "Nie wykryłem adresu e-mail pod kursorem. Wskaż bezpośrednio tekst lub link z adresem."
        )

    @staticmethod
    def _copy_number_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        candidates = ActionRegistry._text_candidates_under_cursor_sync()
        if not candidates:
            selected = ActionRegistry._safe_copy_selected_text_sync()
            if selected:
                candidates = [selected]
        for candidate in candidates:
            number = ActionRegistry._extract_number_from_text(candidate)
            if number:
                ActionRegistry._write_clipboard_text_sync(number)
                return (
                    f"Skopiowałem numer: {number}.",
                    {"value": number, "source": "cursor_text"},
                )
        raise RuntimeError(
            "Nie wykryłem numeru pod kursorem. Spróbuj wskazać tekst z numerem lub zaznaczyć go."
        )

    @staticmethod
    def _copy_sentence_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        candidates = ActionRegistry._text_candidates_under_cursor_sync()
        if not candidates:
            selected = ActionRegistry._safe_copy_selected_text_sync()
            if selected:
                candidates = [selected]
        for candidate in candidates:
            sentence = ActionRegistry._extract_sentence_from_text(candidate)
            if sentence:
                ActionRegistry._write_clipboard_text_sync(sentence)
                return (
                    "Skopiowałem zdanie pod kursorem.",
                    {"text": sentence, "source": "cursor_text"},
                )
        raise RuntimeError(
            "Nie wykryłem pełnego zdania pod kursorem. Spróbuj wskazać obszar z tekstem."
        )

    @staticmethod
    def _select_sentence_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        text = ActionRegistry._select_uia_text_under_cursor_sync("sentence")
        return (
            "Zaznaczyłem zdanie pod kursorem.",
            {"text": text, "unit": "sentence", "source": "uia_text_pattern"},
        )

    @staticmethod
    def _select_paragraph_under_cursor_sync() -> tuple[str, dict[str, Any]]:
        text = ActionRegistry._select_uia_text_under_cursor_sync("paragraph")
        return (
            "Zaznaczyłem akapit pod kursorem.",
            {"text": text, "unit": "paragraph", "source": "uia_text_pattern"},
        )

    @staticmethod
    def _select_uia_text_under_cursor_sync(unit: str) -> str:
        if unit not in {"sentence", "paragraph"}:
            raise ValueError(f"Nieobsługiwana jednostka tekstu: {unit}")
        try:
            import win32api
            from pywinauto import Desktop
            from pywinauto.uia_defines import IUIA
        except ImportError as exc:
            raise RuntimeError(f"Brak Windows UI Automation do zaznaczania tekstu: {exc}") from exc

        cursor_x, cursor_y = win32api.GetCursorPos()
        try:
            wrapper = Desktop(backend="uia").from_point(cursor_x, cursor_y)
            uia = IUIA()
            client = uia.ui_automation_client
            pattern = None
            current = wrapper
            for _ in range(8):
                element_info = getattr(current, "element_info", None)
                element = getattr(element_info, "element", None)
                if element is not None:
                    try:
                        unknown = element.GetCurrentPattern(client.UIA_TextPatternId)
                        if unknown:
                            pattern = unknown.QueryInterface(client.IUIAutomationTextPattern)
                    except Exception:
                        pattern = None
                if pattern is not None:
                    break
                try:
                    current = current.parent()
                except Exception:
                    break
            if pattern is None:
                raise RuntimeError(
                    "Aplikacja nie udostępnia tekstu pod kursorem przez UI Automation."
                )

            point = client.tagPOINT(cursor_x, cursor_y)
            cursor_range = pattern.RangeFromPoint(point)
            if not cursor_range:
                raise RuntimeError("Nie znalazłem zakresu tekstu pod kursorem.")
            paragraph_range = cursor_range.Clone()
            paragraph_range.ExpandToEnclosingUnit(client.TextUnit_Paragraph)
            paragraph_text = str(paragraph_range.GetText(MAX_UIA_SELECTION_TEXT_CHARS + 1) or "")
            if not paragraph_text.strip():
                raise RuntimeError("Zakres pod kursorem nie zawiera tekstu.")
            if len(paragraph_text) > MAX_UIA_SELECTION_TEXT_CHARS:
                raise RuntimeError(
                    "Akapit pod kursorem jest zbyt długi do bezpiecznego zaznaczenia."
                )

            target_range = paragraph_range
            if unit == "sentence":
                prefix_range = paragraph_range.Clone()
                prefix_range.MoveEndpointByRange(
                    client.TextPatternRangeEndpoint_End,
                    cursor_range,
                    client.TextPatternRangeEndpoint_Start,
                )
                prefix = str(prefix_range.GetText(MAX_UIA_SELECTION_TEXT_CHARS + 1) or "")
                span = ActionRegistry._sentence_span_at_offset(paragraph_text, len(prefix))
                if span is None:
                    raise RuntimeError("Nie rozpoznałem granic zdania pod kursorem.")
                start, end, _ = span
                target_range = paragraph_range.Clone()
                target_range.MoveEndpointByRange(
                    client.TextPatternRangeEndpoint_End,
                    paragraph_range,
                    client.TextPatternRangeEndpoint_Start,
                )
                target_range.MoveEndpointByUnit(
                    client.TextPatternRangeEndpoint_End,
                    client.TextUnit_Character,
                    end,
                )
                target_range.MoveEndpointByUnit(
                    client.TextPatternRangeEndpoint_Start,
                    client.TextUnit_Character,
                    start,
                )

            selected_text = str(
                target_range.GetText(MAX_UIA_SELECTION_TEXT_CHARS + 1) or ""
            ).strip()
            if not selected_text:
                raise RuntimeError("Wybrany zakres tekstu jest pusty.")
            target_range.Select()
            return selected_text
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Aplikacja nie pozwoliła zaznaczyć tekstu pod kursorem przez UI Automation."
            ) from exc

    @staticmethod
    def _sentence_span_at_offset(
        text: str,
        offset: int,
    ) -> tuple[int, int, str] | None:
        if not text.strip():
            return None
        spans: list[tuple[int, int, str]] = []
        for match in re.finditer(
            r"[^.!?\r\n]+(?:[.!?]+(?=\s|$)|(?=\r?\n|$))",
            text,
        ):
            start, end = match.span()
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end, text[start:end]))
        if not spans:
            return None
        cursor = max(0, min(offset, len(text)))

        def distance(span: tuple[int, int, str]) -> int:
            start, end, _ = span
            if start <= cursor <= end:
                return 0
            return min(abs(cursor - start), abs(cursor - end))

        return min(spans, key=distance)

    @staticmethod
    def _safe_copy_selected_text_sync() -> str:
        try:
            _, data = ActionRegistry._copy_selected_text_sync()
            return str(data.get("text") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _read_clipboard_text_sync() -> str:
        try:
            import win32clipboard
            import win32con
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do odczytu schowka: {exc}") from exc

        for _ in range(6):
            try:
                win32clipboard.OpenClipboard()
                try:
                    if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                        value = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                        return str(value or "")
                    return ""
                finally:
                    win32clipboard.CloseClipboard()
            except Exception:
                time.sleep(0.02)
        return ""

    @staticmethod
    def _write_clipboard_text_sync(text: str) -> None:
        try:
            import win32clipboard
            import win32con
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do zapisu schowka: {exc}") from exc
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Nie można zapisać pustego tekstu do schowka.")
        for _ in range(6):
            try:
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, cleaned)
                    return
                finally:
                    win32clipboard.CloseClipboard()
            except Exception:
                time.sleep(0.02)
        raise RuntimeError("Nie udało się zapisać tekstu do schowka.")

    @staticmethod
    def _send_copy_shortcut_sync() -> None:
        try:
            import win32api
            import win32con
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do skrótu kopiowania: {exc}") from exc
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord("C"), 0, 0, 0)
        win32api.keybd_event(ord("C"), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

    @staticmethod
    def _send_paste_shortcut_sync() -> None:
        try:
            import win32api
            import win32con
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do skrótu wklejania: {exc}") from exc
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord("V"), 0, 0, 0)
        win32api.keybd_event(ord("V"), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

    @staticmethod
    def _rename_under_cursor_sync(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
            import time

            import win32api
            import win32con
        except ImportError as exc:
            raise RuntimeError(f"Brak zależności Windows do zmiany nazwy: {exc}") from exc

        new_name = str(args.get("new_name") or "").strip()
        # Click once to focus the item under the cursor, then F2.
        cursor = win32api.GetCursorPos()
        win32api.SetCursorPos(cursor)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.12)
        win32api.keybd_event(win32con.VK_F2, 0, 0, 0)
        win32api.keybd_event(win32con.VK_F2, 0, win32con.KEYEVENTF_KEYUP, 0)
        data: dict[str, Any] = {"cursor": {"x": cursor[0], "y": cursor[1]}, "renamed": False}
        if not new_name:
            return "Rozpoczęto zmianę nazwy (F2). Podaj nową nazwę lub wpisz ją ręcznie.", data

        time.sleep(0.18)
        ActionRegistry._write_clipboard_text_sync(new_name)
        ActionRegistry._send_paste_shortcut_sync()
        time.sleep(0.08)
        win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
        win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
        data["renamed"] = True
        data["new_name"] = new_name
        return f"Zmieniono nazwę na: {new_name}", data

    @staticmethod
    def _text_candidates_under_cursor_sync() -> list[str]:
        try:
            import ctypes

            from pywinauto import Desktop
        except ImportError:
            return []

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        point = POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)) == 0:
            return []

        try:
            wrapper = Desktop(backend="uia").from_point(point.x, point.y)
        except Exception:
            return []

        candidates: list[str] = []

        def collect_text(value: str | None) -> None:
            text = (value or "").strip()
            if text:
                text = re.sub(r"\s+", " ", text)
                if text and text not in candidates:
                    candidates.append(text)

        collect_text(getattr(wrapper, "window_text", lambda: "")())
        element_info = getattr(wrapper, "element_info", None)
        if element_info is not None:
            collect_text(getattr(element_info, "name", ""))

        if hasattr(wrapper, "texts"):
            try:
                for text in wrapper.texts():
                    collect_text(text)
            except Exception:
                pass

        parent = wrapper
        for _ in range(3):
            try:
                parent = parent.parent()
            except Exception:
                break
            collect_text(getattr(parent, "window_text", lambda: "")())
            parent_info = getattr(parent, "element_info", None)
            if parent_info is not None:
                collect_text(getattr(parent_info, "name", ""))

        return candidates

    @staticmethod
    def _extract_number_from_text(text: str) -> str | None:
        for match in NUMBER_PATTERN.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if any(char.isdigit() for char in value):
                return value
        return None

    @staticmethod
    def _extract_emails_from_text(text: str) -> list[str]:
        emails: list[str] = []
        seen: set[str] = set()
        for match in EMAIL_PATTERN.finditer(text):
            email = match.group(0).strip()
            normalized = email.casefold()
            if normalized not in seen:
                seen.add(normalized)
                emails.append(email)
        return emails

    @staticmethod
    def _extract_sentence_from_text(text: str) -> str | None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return None
        sentences = [
            sentence.strip(" -")
            for sentence in re.findall(r"[^.!?\n]+[.!?]?", cleaned)
            if sentence.strip()
        ]
        for sentence in sentences:
            if len(sentence.split()) >= 2:
                return sentence
        if len(cleaned.split()) >= 2:
            return cleaned
        return None

    async def _open_url(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        url = str(args.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or len(url) > 2048:
            raise ValueError("Dozwolone są wyłącznie poprawne adresy HTTP/HTTPS.")
        opened = await asyncio.to_thread(webbrowser.open, url, 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć adresu.")
        return "Otwarto adres.", {"url": url}

    async def _open_folder(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        folder_id = str(args.get("folder_id") or "").strip()
        target = ALLOWED_FOLDERS.get(folder_id)
        if target is None:
            raise ValueError("Dozwolony jest wyłącznie folder z allowlisty.")
        await asyncio.to_thread(os.startfile, target)
        return "Otwarto Ten komputer.", {"folder_id": folder_id}

    async def _open_app(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        app_id = str(args.get("app_id") or "").strip().casefold()
        target = ALLOWED_APPS.get(app_id)
        if target is None:
            raise ValueError("Dozwolona jest wyłącznie aplikacja z allowlisty.")
        await asyncio.to_thread(os.startfile, target)
        return "Otwarto WhatsApp.", {"app_id": app_id}

    async def _create_note(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = str(args.get("text") or "").strip()
        if not text:
            raise ValueError("Treść notatki jest pusta.")
        _, data = await self._invoke_uivision(
            macro="voiceloop_notatka.json",
            var1=text[:8000],
            var2="",
            var3="",
        )
        return "Notatka została zapisana.", data

    async def _run_uivision_macro(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        macro = str(args.get("macro") or "").strip()
        if (
            len(macro) > 160
            or ".." in macro
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", macro)
        ):
            raise ValueError("Nieprawidłowa nazwa makra.")
        project_macro = self.settings.project_root / "uivision" / "macros" / macro
        runtime_macro = self.settings.ui_vision_home_path / "macros" / macro
        if not project_macro.is_file():
            raise FileNotFoundError(f"Makro nie znajduje się na allowliście projektu: {macro}")
        if not runtime_macro.is_file():
            raise FileNotFoundError(
                f"Makro nie jest zsynchronizowane z runtime: {macro}. "
                "Uruchom scripts\\sync-uivision.ps1."
            )
        return await self._invoke_uivision(
            macro=macro,
            var1=str(args.get("var1") or "")[:8000],
            var2=str(args.get("var2") or "")[:8000],
            var3=str(args.get("var3") or "")[:8000],
        )

    async def _invoke_uivision(
        self, *, macro: str, var1: str, var2: str, var3: str
    ) -> tuple[str, dict[str, Any]]:
        runner = self.settings.project_root / "scripts" / "run-uivision.ps1"
        if not runner.exists():
            raise FileNotFoundError(f"Brak runnera UI.Vision: {runner}")
        self._current_process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            "-Macro",
            macro,
            "-Var1",
            var1,
            "-Var2",
            var2,
            "-Var3",
            var3,
            "-TimeoutSeconds",
            str(self.settings.uivision_timeout_seconds),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                self._current_process.communicate(),
                timeout=self.settings.uivision_timeout_seconds + 10,
            )
        except TimeoutError as exc:
            await self.stop()
            raise TimeoutError("UI.Vision przekroczył limit czasu.") from exc
        finally:
            process = self._current_process
            self._current_process = None
        if process is None:
            raise RuntimeError("Proces UI.Vision został przerwany.")
        output = stdout.decode("utf-8", errors="replace").strip()
        error = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(error or output or f"UI.Vision exit code {process.returncode}")
        return output or f"Makro {macro} zakończone.", {"macro": macro}

    async def _remember(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ValueError("Brak treści do zapamiętania.")
        item = await self._create_manual_memory(
            MemoryCreate(
                kind=str(args.get("kind") or "fact")[:50],
                content=content,
                sensitivity="private",
                source="assistant",
            )
        )
        return "Zapamiętano.", {"memory_id": item.id}

    async def _remember_last_source(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
            requested_index = int(args.get("index", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Parametr index musi być liczbą całkowitą.") from exc
        index = max(1, min(requested_index, 20))

        cached = await self._load_last_web_sources()
        sources = cached.get("sources", []) if isinstance(cached, dict) else []
        if not isinstance(sources, list) or not sources:
            raise RuntimeError(
                "Brak ostatnich źródeł do zapamiętania. Najpierw użyj wyszukiwania internetowego."
            )
        if index > len(sources):
            raise ValueError(f"Nie ma źródła nr {index}. Dostępnych źródeł: {len(sources)}.")

        source = sources[index - 1]
        title = str(source.get("title") or source.get("url") or "bez tytułu").strip()
        url = str(source.get("url") or "").strip()
        if not url:
            raise RuntimeError("Wybrane źródło nie ma adresu URL.")
        snippet = str(source.get("snippet") or "").strip()
        provider = str(source.get("provider") or "").strip()
        query = str(cached.get("query") or "").strip()
        endpoint_check = cached.get("endpoint_check")
        note = str(args.get("note") or "").strip()

        content_lines = [
            f"Źródło WWW: {title}",
            f"URL: {url}",
        ]
        if query:
            content_lines.append(f"Zapytanie: {query}")
        if provider:
            content_lines.append(f"Provider: {provider}")
        if snippet:
            content_lines.append(f"Opis: {snippet[:600]}")
        if isinstance(endpoint_check, dict):
            api_name = str(endpoint_check.get("api_name") or "").strip()
            endpoint = str(endpoint_check.get("endpoint") or "").strip()
            found = bool(endpoint_check.get("endpoint_found"))
            access = endpoint_check.get("documentation_access")
            access_label = ""
            if isinstance(access, dict):
                access_label = str(access.get("availability_label") or "").strip()
            status = "potwierdzony" if found else "niepotwierdzony"
            if api_name or endpoint:
                detail = " ".join(part for part in (api_name, endpoint) if part).strip()
                content_lines.append(f"Weryfikacja endpointu ({status}): {detail}")
            if access_label:
                content_lines.append(f"Dostępność dokumentacji: {access_label}")
        if note:
            content_lines.append(f"Notatka: {note[:500]}")

        item = await self._create_manual_memory(
            MemoryCreate(
                kind=str(args.get("kind") or "web_source")[:50],
                content="\n".join(content_lines)[:10000],
                sensitivity="private",
                source="assistant",
            )
        )
        return (
            f"Zapamiętałem źródło {index}: {title}.",
            {
                "memory_id": item.id,
                "index": index,
                "available_sources": len(sources),
                "title": title,
                "url": url,
            },
        )

    async def _recall(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("Brak zapytania do pamięci.")
        vector_items: list[dict[str, Any]] = []
        if (
            self.embeddings is not None
            and self.embeddings.enabled
            and self.embeddings.accepts_private_text()
        ):
            try:
                documents = memory_query_documents(query[:2000])
                vector_names = tuple(
                    name for name in MEMORY_VECTOR_NAMES if name in documents
                )
                vectors = await self.embeddings.embed_queries(
                    [documents[name] for name in vector_names]
                )
                if len(vectors) != len(vector_names):
                    raise EmbeddingUnavailableError(
                        "embedding count mismatch for recall query documents"
                    )
                query_vectors = dict(zip(vector_names, vectors, strict=True))
                hits = []
                if (
                    self.qdrant is not None
                    and self.qdrant.enabled
                    and self.qdrant.accepts_private_data()
                ):
                    try:
                        hits = await self.qdrant.search(
                            query_vectors=query_vectors,
                            query_weights=memory_query_weights(
                                query,
                                adaptive=self.settings.vector_memory_adaptive_query_weights,
                                base_weights=self.settings.vector_memory_weights,
                            ),
                            limit=10,
                            min_score=self.settings.vector_memory_min_score,
                            rrf_k=self.settings.vector_memory_rrf_k,
                        )
                    except QdrantMemoryError as exc:
                        LOGGER.warning("Qdrant recall unavailable; using SQLite fallback: %s", exc)
                if not hits and query_vectors.get("semantic"):
                    hits = await self.memory.search_vector_memories(
                        query_vectors["semantic"],
                        limit=10,
                        min_score=self.settings.vector_memory_min_score,
                    )
                vector_items = [
                    {
                        "source": hit.source,
                        "source_id": hit.source_id,
                        "title": hit.title,
                        "content": hit.content,
                        "metadata": hit.metadata,
                        "score": hit.score,
                        "created_at": hit.created_at.isoformat(),
                    }
                    for hit in hits
                ]
            except EmbeddingUnavailableError:
                vector_items = []
        if vector_items:
            return (
                f"Znaleziono {len(vector_items)} wpisów w pamięci semantycznej.",
                {"items": vector_items, "retrieval": "vector_v2"},
            )
        items = await self.memory.list_memories(limit=200)
        normalized_query = query.casefold()
        matched = [
            item for item in items if normalized_query in item.content.casefold()
        ][:10]
        return (
            f"Znaleziono {len(matched)} wpisów.",
            {
                "items": [item.model_dump(mode="json") for item in matched],
                "retrieval": "lexical_fallback",
            },
        )

    async def _create_manual_memory(self, item: MemoryCreate) -> MemoryItem:
        if self.manual_memory is not None:
            return await self.manual_memory.create(item)
        return await self.memory.create_memory(item)

    async def _speak_text(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = str(args.get("text") or "").strip()
        if not text:
            raise ValueError("Brak tekstu do wypowiedzenia.")
        await self.tts.speak(text)
        return "Wypowiedziano tekst.", {}

    async def _list_capabilities(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        catalog = self.capability_catalog()
        actions = list(catalog["voiceloop_actions"])
        query = " ".join(str(args.get("query") or "").casefold().split())
        query_tokens = {
            token.strip(".,!?;:„”\"'")
            for token in query.split()
            if len(token.strip(".,!?;:„”\"'")) >= 4
            and token
            not in {
                "potrafisz",
                "umiesz",
                "powiedz",
                "powiedziec",
                "powiedzieć",
                "zeby",
                "żeby",
                "jakie",
                "mozesz",
                "możesz",
            }
        }

        def relevance(item: dict[str, Any]) -> tuple[int, int]:
            haystack = " ".join(
                [
                    str(item.get("label") or ""),
                    str(item.get("description") or ""),
                    " ".join(str(value) for value in item.get("positive_examples") or []),
                ]
            ).casefold()
            matches = sum(token in haystack for token in query_tokens)
            return matches, -len(haystack)

        candidates = sorted(actions, key=relevance, reverse=True)
        relevant = [item for item in candidates if relevance(item)[0] > 0][:3]
        asks_how = bool(
            re.search(r"\bjak\b.*\b(?:powiedziec|powiedzieć|poprosic|poprosić)\b", query)
        )
        asks_if = bool(re.search(r"\b(?:czy\s+)?(?:umiesz|potrafisz)\b", query))
        asks_close_by_name = bool(
            re.search(r"\b(?:zamkn|wylacz|wyłącz)\w*\b", query)
            and re.search(r"\b(?:nazw\w*|samej\s+nazwie)\b", query)
        )
        if asks_if and asks_close_by_name:
            message = (
                "Nie zamykam jeszcze dowolnej aplikacji po samej nazwie. "
                "Mogę bezpiecznie zamknąć okno pod kursorem i poproszę wtedy o potwierdzenie."
            )
        elif relevant and asks_how:
            item = relevant[0]
            examples = item.get("positive_examples") or []
            example = str(examples[0]) if examples else str(item["label"])
            confirmation = (
                " Będę prosić o potwierdzenie."
                if item.get("confirmation_required")
                else " Nie wymaga to potwierdzenia."
            )
            message = f"Powiedz: „{example}”.{confirmation}"
        elif relevant and asks_if:
            descriptions = "; ".join(
                (
                    f"{item['spoken_name']}, na przykład "
                    f"„{(item.get('positive_examples') or [item['label']])[0]}”"
                )
                for item in relevant
            )
            message = f"Tak, najbliższe realne możliwości to: {descriptions}."
        elif asks_if:
            message = (
                "Tego jeszcze nie wykonuję. Mogę opisać aktywne okno, otworzyć "
                "dozwoloną aplikację lub stronę i podać dokładną komendę dla tych działań."
            )
        else:
            grouped: dict[str, list[str]] = {}
            for item in actions:
                grouped.setdefault(str(item["category"]), []).append(
                    str(item["spoken_name"])
                )
            groups = [
                f"{category}: {', '.join(labels[:2])}"
                for category, labels in grouped.items()
                if category != "inne"
            ]
            message = (
                "Potrafię działać w kilku grupach. "
                + "; ".join(groups[:6])
                + ". Zapytaj „czy umiesz X”, a podam konkretną komendę i warunki."
            )
        return message[:1800], catalog

    async def _store_last_web_sources(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        endpoint_check: dict[str, Any] | None = None,
    ) -> None:
        normalized_sources: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            if not url:
                continue
            title = str(result.get("title") or url).strip()
            snippet = str(result.get("snippet") or "").strip()
            provider = str(result.get("provider") or "").strip()
            normalized_sources.append(
                {
                    "title": title[:300],
                    "url": url[:2000],
                    "snippet": snippet[:1000],
                    "provider": provider[:80],
                }
            )
            if len(normalized_sources) >= 20:
                break

        payload: dict[str, Any] = {
            "query": query[:400],
            "sources": normalized_sources,
        }
        if endpoint_check:
            payload["endpoint_check"] = endpoint_check

        self._last_web_sources = payload
        try:
            await self.memory.set_state(
                LAST_WEB_SOURCES_STATE_KEY,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
        except Exception:
            # Nie przerywamy wyszukiwania, gdy pamięć stanu jest chwilowo niedostępna.
            return

    async def _load_last_web_sources(self) -> dict[str, Any]:
        if self._last_web_sources is not None:
            return self._last_web_sources

        raw_state = await self.memory.get_state(LAST_WEB_SOURCES_STATE_KEY)
        if not raw_state:
            self._last_web_sources = {"query": "", "sources": []}
            return self._last_web_sources
        try:
            payload = json.loads(raw_state)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if not isinstance(payload.get("sources"), list):
            payload["sources"] = []
        if not isinstance(payload.get("query"), str):
            payload["query"] = ""
        self._last_web_sources = payload
        return payload
