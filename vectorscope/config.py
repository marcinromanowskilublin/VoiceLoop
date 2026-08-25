"""Konfiguracja Vectorscope wyprowadzona z ustawień VoiceLoopa.

Każdy próg pokazywany w panelu jest czytany z VoiceLoopa, a nie przepisywany
tutaj. Dzięki temu panel nie może zacząć opisywać wartości, których asystent
już nie używa.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from voiceloop.embeddings import OpenAICompatibleEmbeddingClient
from voiceloop.screenpipe_memory import ACTIVITY_DUPLICATE_MIN_SCORE
from voiceloop.settings import Settings, get_settings

VECTORSCOPE_HOST = "127.0.0.1"
VECTORSCOPE_PORT = 8770

# Prefiksy zadania modelu nomic. Porównywanie wektorów policzonych różnymi
# prefiksami tworzy klastry, które są artefaktem prefiksu, nie znaczenia.
PREFIX_QUERY = "search_query"
PREFIX_DOCUMENT = "search_document"
PREFIX_NONE = "none"
PREFIXES = (PREFIX_DOCUMENT, PREFIX_QUERY, PREFIX_NONE)

# Kontekst nomic-embed-text-v2-moe. Dłuższe jednostki są po cichu ucinane,
# więc panel musi ostrzegać, zamiast udawać, że policzył cały tekst.
EMBEDDING_CONTEXT_TOKENS = 512
CHARS_PER_TOKEN_ESTIMATE = 3.2

# Literał z voiceloop/screenpipe_memory.py:311 (wyszukiwanie historii pokrewnej).
SCREENPIPE_RELATED_HISTORY_MIN_SCORE = 0.30


@dataclass(frozen=True)
class Threshold:
    key: str
    value: float
    label: str
    origin: str


def vectorscope_data_dir(settings: Settings) -> Path:
    return settings.data_dir / "vectorscope"


def build_embedding_client(settings: Settings) -> OpenAICompatibleEmbeddingClient:
    """Ten sam klient i te same prefiksy, co w listener/voiceloop/app.py:165."""

    return OpenAICompatibleEmbeddingClient(
        base_url=(settings.local_embeddings_base_url or settings.lm_studio_base_url),
        api_key=settings.local_embeddings_api_key or settings.lm_studio_api_key,
        model=settings.local_embeddings_model,
        timeout_seconds=settings.local_embeddings_timeout_seconds,
        enabled=settings.local_embeddings_enabled,
    )


def collect_thresholds(settings: Settings) -> list[Threshold]:
    """Progi, które realnie rządzą retrievalem — czytane, nie przepisywane."""

    return [
        Threshold(
            key="vector_memory_min_score",
            value=settings.vector_memory_min_score,
            label="Pamięć wektorowa — odcięcie",
            origin="settings.py",
        ),
        Threshold(
            key="capability_match_min_score",
            value=settings.capability_match_min_score,
            label="Dopasowanie capability",
            origin="settings.py",
        ),
        Threshold(
            key="routing_v2_execute_min_score",
            value=settings.routing_v2_execute_min_score,
            label="Routing v2 — wykonanie",
            origin="settings.py",
        ),
        Threshold(
            key="routing_v2_execute_min_margin",
            value=settings.routing_v2_execute_min_margin,
            label="Routing v2 — margines top2",
            origin="settings.py",
        ),
        Threshold(
            key="corpus_routing_margin_threshold",
            value=settings.corpus_routing_margin_threshold,
            label="Korpus — margines routingu",
            origin="settings.py",
        ),
        Threshold(
            key="stt_min_action_confidence",
            value=settings.stt_min_action_confidence,
            label="STT — minimalna pewność akcji",
            origin="settings.py",
        ),
        Threshold(
            key="behavior_digest_min_confidence",
            value=settings.behavior_digest_min_confidence,
            label="Digest zachowania — pewność",
            origin="settings.py",
        ),
        Threshold(
            key="screenpipe_duplicate_min_score",
            value=float(ACTIVITY_DUPLICATE_MIN_SCORE),
            label="Screenpipe — deduplikacja kubełka",
            origin="screenpipe_memory.py",
        ),
        Threshold(
            key="screenpipe_related_history_min_score",
            value=SCREENPIPE_RELATED_HISTORY_MIN_SCORE,
            label="Screenpipe — historia pokrewna",
            origin="screenpipe_memory.py:311",
        ),
    ]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1)


def exceeds_context(text: str) -> bool:
    return estimate_tokens(text) > EMBEDDING_CONTEXT_TOKENS


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()
