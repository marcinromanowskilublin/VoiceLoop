from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from .corpus.privacy import redact_text

MEMORY_VECTOR_NAMES = (
    "semantic",
    "topic",
    "intent",
    "decision",
    "person_context",
)
MEMORY_VECTOR_WEIGHTS = {
    "semantic": 0.40,
    "topic": 0.20,
    "intent": 0.15,
    "decision": 0.15,
    "person_context": 0.10,
}
MEMORY_DOCUMENT_SCHEMA_VERSION = "memory-documents-v2"
MEMORY_QUERY_DOCUMENTS_VERSION = "memory-query-documents-v2"
MEMORY_DOCUMENT_FORMAT_VERSION = MEMORY_DOCUMENT_SCHEMA_VERSION
MEMORY_QUERY_FORMAT_VERSION = MEMORY_QUERY_DOCUMENTS_VERSION
MEMORY_QUERY_SCHEMA_VERSION = MEMORY_QUERY_DOCUMENTS_VERSION
MEMORY_QUERY_VERSION = MEMORY_QUERY_DOCUMENTS_VERSION

_QUERY_MARKERS: dict[str, tuple[re.Pattern[str], ...]] = {
    "topic": (
        re.compile(r"\b(?:temat|temacie|dotyczy|dotyczyło|sprawie|o czym)\b", re.IGNORECASE),
    ),
    "intent": (
        re.compile(
            r"\b(?:cel|po co|dlaczego|zamiar|zamierza\w*|chciał\w*|planował\w*)\b",
            re.IGNORECASE,
        ),
    ),
    "decision": (
        re.compile(
            r"\b(?:decyzj\w*|ustal\w*|postanow\w*|zdecydow\w*|"
            r"wybral\w*|wybrał\w*|następn\w*\s+krok\w*)\b",
            re.IGNORECASE,
        ),
    ),
    "person_context": (
        re.compile(
            r"\b(?:kto|komu|kogo|z kim|dla kogo|osob\w*|klient\w*|"
            r"pacjent\w*|rozmówc\w*)\b",
            re.IGNORECASE,
        ),
    ),
}


def memory_vector_documents(
    *,
    summary: str,
    topic: str = "",
    intent: str = "",
    decision: str = "",
    person_context: str = "",
    observations: Iterable[str] = (),
    redact: bool = True,
) -> dict[str, str]:
    """Build independent memory documents, omitting every empty aspect."""

    clean = _safe_text if redact else _plain_text
    safe_summary = clean(summary)
    safe_observations = _unique_nonempty(clean(item) for item in observations)
    safe_observations = [
        item for item in safe_observations if item.casefold() != safe_summary.casefold()
    ]

    documents: dict[str, str] = {}
    semantic_parts = [safe_summary] if safe_summary else []
    if safe_observations:
        semantic_parts.append(
            "Zredagowane obserwacje:\n"
            + "\n".join(f"- {item}" for item in safe_observations[:10])
        )
    if semantic_parts:
        documents["semantic"] = (
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            + "\n\n".join(semantic_parts)
        )

    aspect_documents = {
        "topic": ("Temat zapamiętanej informacji", topic),
        "intent": ("Cel lub intencja zaobserwowanej aktywności", intent),
        "decision": ("Jawna decyzja, ustalenie lub następny krok", decision),
        "person_context": ("Jawny kontekst osoby lub relacji", person_context),
    }
    for name, (label, value) in aspect_documents.items():
        safe_value = clean(value)
        if safe_value:
            documents[name] = f"{label}:\n{safe_value}"
    return documents


def memory_query_documents(
    query: str,
    *,
    version: str = MEMORY_QUERY_DOCUMENTS_VERSION,
) -> dict[str, str]:
    """Build five versioned, space-specific query documents."""

    if version != MEMORY_QUERY_DOCUMENTS_VERSION:
        raise ValueError(f"Nieobsługiwana wersja dokumentów zapytania: {version}")
    clean_query = " ".join(query.split()).strip()
    if not clean_query:
        return {}
    return {
        "semantic": (
            "Wyszukaj zdarzenie lub informację odpowiadającą ogólnemu znaczeniu pytania:\n"
            f"{clean_query}"
        ),
        "topic": (
            "Wyszukaj temat, projekt lub obszar, którego dotyczy pytanie:\n"
            f"{clean_query}"
        ),
        "intent": (
            "Wyszukaj cel, zamiar lub potrzebę stojącą za treścią pytania:\n"
            f"{clean_query}"
        ),
        "decision": (
            "Wyszukaj jawną decyzję, ustalenie albo następny krok istotny dla pytania:\n"
            f"{clean_query}"
        ),
        "person_context": (
            "Wyszukaj osobę, rozmówcę lub relację istotną dla pytania:\n"
            f"{clean_query}"
        ),
    }


def memory_query_weights(
    query: str,
    *,
    adaptive: bool = True,
    base_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return deterministic weights, optionally boosted by explicit query markers."""

    raw_weights = base_weights or MEMORY_VECTOR_WEIGHTS
    weights = {
        name: max(0.0, float(raw_weights.get(name, 0.0)))
        for name in MEMORY_VECTOR_NAMES
    }
    if adaptive:
        for name, patterns in _QUERY_MARKERS.items():
            if any(pattern.search(query) for pattern in patterns):
                weights[name] *= 2.0
    total = sum(weights.values())
    if total <= 0:
        return {name: 1.0 / len(MEMORY_VECTOR_NAMES) for name in MEMORY_VECTOR_NAMES}
    return {name: value / total for name, value in weights.items()}


def _safe_text(value: object) -> str:
    redacted, _ = redact_text(str(value or ""))
    return redacted.strip()


def _plain_text(value: object) -> str:
    return str(value or "").strip()


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if not value or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique
