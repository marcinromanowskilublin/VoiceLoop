"""Pomiar skutku niespójnych prefiksów zadania w pamięci Screenpipe.

Nowa ścieżka indeksowania woła `embed_documents` (prefiks `search_document: `),
stara `embed_texts` (bez prefiksu, screenpipe_memory.py:230), a zapytania idą
zawsze przez `embed_queries` (`search_query: `). Ten moduł mierzy, ile realnie
kosztuje to rozjechanie — w cosinusach i w liczbie rekordów, które przez to
wypadają poniżej progu odcięcia.

Nic tutaj nie modyfikuje pamięci VoiceLoopa. To wyłącznie pomiar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from voiceloop.embeddings import EmbeddingUnavailableError

from . import geometry
from .config import (
    PREFIX_DOCUMENT,
    PREFIX_NONE,
    PREFIX_QUERY,
    build_embedding_client,
    settings,
)
from .embed import embed_texts_with_prefix


@dataclass(frozen=True)
class Probe:
    query: str
    document: str


DEFAULT_PROBES: tuple[Probe, ...] = (
    Probe(
        query="co robiłem w Cursorze",
        document=(
            "Temat zapamiętanej informacji:\n"
            "Praca w edytorze Cursor nad projektem VoiceLoop, "
            "przeglądanie kodu i uruchamianie testów."
        ),
    ),
    Probe(
        query="o czym była rozmowa z pacjentem",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Rozmowa z pacjentem o nasileniu lęku i skuteczności leczenia."
        ),
    ),
    Probe(
        query="jaka była decyzja co do dawki",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Ustalono zwiększenie dawki i kontrolę za dwa tygodnie."
        ),
    ),
    Probe(
        query="co ustaliliśmy na spotkaniu zespołu",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Zespół ustalił termin wdrożenia i podział zadań na sprint."
        ),
    ),
    Probe(
        query="kto dzwonił w sprawie recepty",
        document=(
            "Jawny kontekst osoby lub relacji:\n"
            "Telefon od pacjentki w sprawie przedłużenia recepty."
        ),
    ),
    Probe(
        query="jakie strony przeglądałem w przeglądarce",
        document=(
            "Cel lub intencja zaobserwowanej aktywności:\n"
            "Przeglądanie dokumentacji technicznej w Chrome i szukanie rozwiązania błędu."
        ),
    ),
)


async def run_prefix_check(
    *,
    probes: tuple[Probe, ...] = DEFAULT_PROBES,
    min_score: float | None = None,
) -> dict[str, Any]:
    active = settings()
    threshold = (
        min_score if min_score is not None else active.vector_memory_min_score
    )
    client = build_embedding_client(active)

    queries = [probe.query for probe in probes]
    documents = [probe.document for probe in probes]

    try:
        query_result = await embed_texts_with_prefix(client, queries, prefix=PREFIX_QUERY)
        document_result = await embed_texts_with_prefix(
            client, documents, prefix=PREFIX_DOCUMENT
        )
        raw_result = await embed_texts_with_prefix(client, documents, prefix=PREFIX_NONE)
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    query_vectors = geometry.l2_normalize(query_result.vectors)
    document_vectors = geometry.l2_normalize(document_result.vectors)
    raw_vectors = geometry.l2_normalize(raw_result.vectors)

    rows: list[dict[str, Any]] = []
    for index, probe in enumerate(probes):
        prefixed = float(query_vectors[index] @ document_vectors[index])
        legacy = float(query_vectors[index] @ raw_vectors[index])
        same_text = float(document_vectors[index] @ raw_vectors[index])
        rows.append(
            {
                "query": probe.query,
                "document": probe.document.splitlines()[-1][:120],
                "cosine_with_prefix": round(prefixed, 6),
                "cosine_without_prefix": round(legacy, 6),
                "delta": round(prefixed - legacy, 6),
                "same_text_across_prefixes": round(same_text, 6),
                "passes_with_prefix": prefixed >= threshold,
                "passes_without_prefix": legacy >= threshold,
            }
        )

    prefixed_scores = np.array([row["cosine_with_prefix"] for row in rows])
    legacy_scores = np.array([row["cosine_without_prefix"] for row in rows])
    identity_scores = np.array([row["same_text_across_prefixes"] for row in rows])

    lost = [row for row in rows if row["passes_with_prefix"] and not row["passes_without_prefix"]]
    gained = [
        row for row in rows if row["passes_without_prefix"] and not row["passes_with_prefix"]
    ]

    # Czy sam prefiks tworzy sztuczny podział w przestrzeni? Porównujemy średni
    # cosinus wewnątrz jednej konwencji ze średnim cosinusem między konwencjami.
    within_document = _mean_offdiagonal(document_vectors @ document_vectors.T)
    within_raw = _mean_offdiagonal(raw_vectors @ raw_vectors.T)
    cross = float(np.mean(document_vectors @ raw_vectors.T))

    return {
        "ok": True,
        "threshold": threshold,
        "model": document_result.model,
        "dimension": document_result.dimension,
        "probe_count": len(rows),
        "rows": rows,
        "summary": {
            "mean_with_prefix": round(float(prefixed_scores.mean()), 6),
            "mean_without_prefix": round(float(legacy_scores.mean()), 6),
            "mean_delta": round(float((prefixed_scores - legacy_scores).mean()), 6),
            "max_delta": round(float((prefixed_scores - legacy_scores).max()), 6),
            "min_delta": round(float((prefixed_scores - legacy_scores).min()), 6),
            "mean_same_text_similarity": round(float(identity_scores.mean()), 6),
            "min_same_text_similarity": round(float(identity_scores.min()), 6),
            "lost_below_threshold": len(lost),
            "gained_above_threshold": len(gained),
            "mean_within_document_prefix": round(within_document, 6),
            "mean_within_no_prefix": round(within_raw, 6),
            "mean_across_conventions": round(cross, 6),
        },
        "interpretation": _interpret(
            mean_delta=float((prefixed_scores - legacy_scores).mean()),
            identity=float(identity_scores.mean()),
            lost=len(lost),
        ),
    }


def _mean_offdiagonal(matrix: np.ndarray) -> float:
    count = matrix.shape[0]
    if count < 2:
        return 0.0
    upper = np.triu_indices(count, 1)
    return float(np.mean(matrix[upper]))


def _interpret(*, mean_delta: float, identity: float, lost: int) -> list[str]:
    notes: list[str] = []
    if identity < 0.99:
        notes.append(
            f"Ten sam tekst pod dwoma prefiksami ma cosinus {identity:.3f}, nie 1.000 — "
            "czyli prefiks realnie przesuwa wektor i obie konwencje nie są wymienne."
        )
    else:
        notes.append(
            "Prefiks prawie nie zmienia wektora w tym modelu — niespójność jest "
            "wtedy problemem porządkowym, nie pomiarowym."
        )
    if mean_delta > 0.01:
        notes.append(
            f"Zapytania trafiają w dokumenty z prefiksem lepiej średnio o {mean_delta:.3f} "
            "cosinusa. Rekordy ze starej ścieżki są więc systematycznie karane."
        )
    elif mean_delta < -0.01:
        notes.append(
            f"Nieoczekiwanie lepiej wypada brak prefiksu (o {abs(mean_delta):.3f}). "
            "Warto to sprawdzić na własnym korpusie przed jakąkolwiek migracją."
        )
    else:
        notes.append(
            "Różnica średnich jest w granicach szumu — niespójność prefiksów nie "
            "wygląda tu na główne źródło słabego retrievalu."
        )
    if lost:
        notes.append(
            f"{lost} z sond przechodzi próg tylko z prefiksem — dokładnie tyle rekordów "
            "ze starej ścieżki zniknęłoby z kontekstu przy tym progu."
        )
    return notes
