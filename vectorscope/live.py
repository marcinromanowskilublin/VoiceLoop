"""Pomiar na prawdziwej kolekcji Qdranta, wyłącznie do odczytu.

Wszystko, co panel mierzył dotąd, opierało się na korpusie napisanym na
potrzeby testu. To wystarcza, żeby orzec o właściwościach modelu, ale nie
o właściwościach *tej* pamięci. Ten moduł pyta realną kolekcję.

Metoda nie wymaga zbioru z etykietami. Bierzemy treść zapisanych dokumentów,
budujemy z niej zapytania produkcyjną funkcją `memory_query_documents`
i odpytujemy Qdranta **bez progu**. Dla każdego zapytania znamy jedną
poprawną odpowiedź: punkt, z którego treść pochodzi. Reszta wyników to szum
o znanym statusie.

Ograniczenie, o którym trzeba pamiętać przy czytaniu wyników: zapytanie
wywiedzione z dokumentu jest łatwiejsze niż prawdziwe pytanie użytkownika,
więc „sygnał" jest tu górnym oszacowaniem. Rozkład szumu jest natomiast
prawdziwy — to realne wektory z realnej kolekcji.

Moduł nie zapisuje niczego. Używa wyłącznie `scroll` i `query_points`.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
from voiceloop.embeddings import EmbeddingUnavailableError
from voiceloop.memory_vectorization import (
    MEMORY_VECTOR_NAMES,
    MEMORY_VECTOR_WEIGHTS,
    memory_query_documents,
    memory_vector_documents,
)
from voiceloop.qdrant_memory import QdrantMemoryError, QdrantVectorStore
from voiceloop.screenpipe_memory import ACTIVITY_DUPLICATE_MIN_SCORE

# Rdzeń pomiaru mieszka w kodzie produkcyjnym, żeby strażnik progów i panel
# mierzyły dokładnie to samo. Dwie kopie tej samej funkcji rozjechałyby się
# przy pierwszej poprawce, a wtedy przyrząd przestałby świadczyć o systemie.
from voiceloop.threshold_measure import percentiles as _percentiles
from voiceloop.threshold_measure import scroll_points as _scroll_points

from .config import PREFIX_DOCUMENT, PREFIX_QUERY, build_embedding_client, settings
from .embed import embed_texts_with_prefix

SAMPLE_SEED = 20260825
DEFAULT_PROBES = 30
DEFAULT_DEPTH = 200
# Jedno źródło prawdy z produkcją — wcześniej tu stało 0.92, a worker miał 0.97.
DEDUP_THRESHOLD = ACTIVITY_DUPLICATE_MIN_SCORE
LOGGER = logging.getLogger(__name__)


async def measure_live_collection(
    *,
    probe_count: int = DEFAULT_PROBES,
    depth: int = DEFAULT_DEPTH,
) -> dict[str, Any]:
    active = settings()
    store = QdrantVectorStore(active)
    if not getattr(store, "enabled", False):
        return {"ok": False, "message": "Qdrant jest wyłączony w ustawieniach."}

    client = store.client
    collection = store.collection_name
    threshold = float(active.vector_memory_min_score)

    try:
        points = await _scroll_points(client, collection, limit=2000)
    except (QdrantMemoryError, Exception) as exc:  # noqa: BLE001
        return {"ok": False, "message": f"Qdrant nie odpowiada: {exc}"}

    usable = [item for item in points if item["content"].strip()]
    if len(usable) < 5:
        return {
            "ok": False,
            "message": f"Za mało dokumentów z treścią: {len(usable)} z {len(points)}.",
        }

    generator = random.Random(SAMPLE_SEED)
    probes = generator.sample(usable, min(probe_count, len(usable)))

    embeddings = build_embedding_client(active)
    axis_scores: dict[str, list[float]] = {name: [] for name in MEMORY_VECTOR_NAMES}
    axis_self: dict[str, list[float]] = {name: [] for name in MEMORY_VECTOR_NAMES}
    axis_rank: dict[str, list[int]] = {name: [] for name in MEMORY_VECTOR_NAMES}
    axis_missing: dict[str, int] = {name: 0 for name in MEMORY_VECTOR_NAMES}

    for probe in probes:
        documents = memory_query_documents(probe["content"][:2000])
        if not documents:
            continue
        names = [name for name in MEMORY_VECTOR_NAMES if name in documents]
        try:
            vectors = await embeddings.embed_queries([documents[name] for name in names])
        except EmbeddingUnavailableError as exc:
            return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

        for name, vector in zip(names, vectors, strict=True):
            try:
                response = await client.query_points(
                    collection_name=collection,
                    query=list(vector),
                    using=name,
                    limit=depth,
                    score_threshold=None,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Pominięto oś %s podczas sondy live: %s", name, exc)
                continue

            found_self = False
            for rank, point in enumerate(response.points, start=1):
                score = float(point.score)
                if str(point.id) == probe["id"]:
                    axis_self[name].append(score)
                    axis_rank[name].append(rank)
                    found_self = True
                else:
                    axis_scores[name].append(score)
            if not found_self:
                axis_missing[name] += 1

    axes = [
        _axis_report(
            name=name,
            noise=np.array(axis_scores[name]),
            self_scores=np.array(axis_self[name]),
            ranks=axis_rank[name],
            missing=axis_missing[name],
            threshold=threshold,
        )
        for name in MEMORY_VECTOR_NAMES
    ]
    measured = [item for item in axes if item["noise_count"] > 0]

    return {
        "ok": True,
        "collection": collection,
        "points_in_collection": len(points),
        "points_with_content": len(usable),
        "probe_count": len(probes),
        "depth": depth,
        "threshold": threshold,
        "threshold_key": "vector_memory_min_score",
        "axes": axes,
        "sources": _source_histogram(points),
        "interpretation": _interpret(measured, threshold),
        "caveat": (
            "Zapytania wywiedziono z treści zapisanych dokumentów, więc sygnał jest "
            "górnym oszacowaniem. Rozkład szumu pochodzi z realnych wektorów."
        ),
    }


async def measure_dedup_probe(*, sample: int = 60) -> dict[str, Any]:
    """Czy próg deduplikacji Screenpipe może w ogóle zadziałać.

    Kod produkcyjny wektoryzuje zapisywany dokument jako `search_document:` wraz
    z nagłówkiem szablonu, a przy sprawdzaniu duplikatu pyta `search_query:`
    o samą surową treść. To dwa różne miejsca w przestrzeni. Test jest szczelny,
    bo z payloadu można odtworzyć dokładnie ten tekst, który trafił do wektora:
    `content` to podsumowanie z digestu, a `observations` leżą w metadanych.

    Mierzymy podobieństwo *tego samego tekstu* do jego własnego zapisanego
    wektora w obu wariantach. Jeśli w wariancie produkcyjnym nie sięga progu,
    deduplikacja nie może zadziałać nigdy — niezależnie od danych.
    """

    active = settings()
    store = QdrantVectorStore(active)
    if not getattr(store, "enabled", False):
        return {"ok": False, "message": "Qdrant jest wyłączony w ustawieniach."}

    try:
        points = await _scroll_points(
            store.client,
            store.collection_name,
            limit=1200,
            with_vectors=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"Qdrant nie odpowiada: {exc}"}

    usable = [
        item
        for item in points
        if item["content"].strip() and item["vector"] is not None
    ]
    if len(usable) < 5:
        return {"ok": False, "message": f"Za mało punktów z wektorem semantic: {len(usable)}."}

    generator = random.Random(SAMPLE_SEED)
    chosen = generator.sample(usable, min(sample, len(usable)))

    rebuilt: list[str] = []
    plain: list[str] = []
    stored: list[np.ndarray] = []
    for item in chosen:
        documents = memory_vector_documents(
            summary=item["content"],
            observations=item["observations"],
            redact=False,
        )
        semantic = documents.get("semantic")
        if not semantic:
            continue
        rebuilt.append(semantic)
        plain.append(item["content"])
        stored.append(_unit(np.asarray(item["vector"], dtype=float)))

    if not rebuilt:
        return {"ok": False, "message": "Nie udało się odtworzyć żadnego dokumentu semantycznego."}

    client = build_embedding_client(active)
    try:
        as_document = await embed_texts_with_prefix(client, rebuilt, prefix=PREFIX_DOCUMENT)
        as_query = await embed_texts_with_prefix(client, plain, prefix=PREFIX_QUERY)
    except (EmbeddingUnavailableError, ValueError) as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    stored_matrix = np.vstack(stored)
    document_matrix = _unit_rows(as_document.vectors)
    query_matrix = _unit_rows(as_query.vectors)

    self_document = np.sum(document_matrix * stored_matrix, axis=1)
    self_query = np.sum(query_matrix * stored_matrix, axis=1)

    cross = document_matrix @ stored_matrix.T
    np.fill_diagonal(cross, np.nan)
    noise = cross[~np.isnan(cross)]

    return {
        "ok": True,
        "threshold": DEDUP_THRESHOLD,
        "threshold_name": "ACTIVITY_DUPLICATE_MIN_SCORE",
        "sample": len(rebuilt),
        "identical_text_as_document": {
            **_percentiles(self_document),
            "reaches_threshold": round(float(np.mean(self_document >= DEDUP_THRESHOLD)), 3),
        },
        "identical_text_as_query": {
            **_percentiles(self_query),
            "reaches_threshold": round(float(np.mean(self_query >= DEDUP_THRESHOLD)), 3),
        },
        "distinct_documents": {
            **_percentiles(noise),
            "above_threshold": round(float(np.mean(noise >= DEDUP_THRESHOLD)), 3),
        },
        "verdict": _dedup_verdict(self_document, self_query, noise),
    }


def _dedup_verdict(
    self_document: np.ndarray,
    self_query: np.ndarray,
    noise: np.ndarray,
) -> list[str]:
    notes: list[str] = []
    query_max = float(self_query.max())
    document_min = float(self_document.min())

    if query_max < DEDUP_THRESHOLD:
        notes.append(
            f"Wariant produkcyjny nie sięga progu ani raz: najwyższe podobieństwo "
            f"identycznego tekstu do własnego wektora to {query_max:.3f}, próg "
            f"{DEDUP_THRESHOLD}. Deduplikacja Screenpipe nie odrzuciła nigdy niczego "
            "i nie mogła."
        )
    else:
        reached = float(np.mean(self_query >= DEDUP_THRESHOLD))
        notes.append(
            f"Wariant produkcyjny sięga progu w {reached:.0%} przypadków "
            "dla tekstu identycznego z zapisanym."
        )

    notes.append(
        f"Po zrównaniu przestrzeni ten sam tekst wraca do własnego wektora z "
        f"podobieństwem co najmniej {document_min:.3f}, czyli próg {DEDUP_THRESHOLD} "
        "jest osiągalny."
    )

    share = float(np.mean(noise >= DEDUP_THRESHOLD))
    if share > 0.02:
        notes.append(
            f"Ostrożnie z tą samą wartością progu: {share:.1%} par *różnych* dokumentów "
            f"też przekracza {DEDUP_THRESHOLD} (p99 szumu {float(np.percentile(noise, 99)):.3f}). "
            "Dokumenty Screenpipe są do siebie bardzo podobne, więc próg odrzuci też "
            "treści nowe."
        )
    else:
        notes.append(
            f"Różne dokumenty przekraczają {DEDUP_THRESHOLD} w {share:.1%} par "
            f"(p99 szumu {float(np.percentile(noise, 99)):.3f}), więc próg rozdziela "
            "duplikat od nowej treści."
        )
    return notes


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _axis_report(
    *,
    name: str,
    noise: np.ndarray,
    self_scores: np.ndarray,
    ranks: list[int],
    missing: int,
    threshold: float,
) -> dict[str, Any]:
    return {
        "axis": name,
        "weight": MEMORY_VECTOR_WEIGHTS[name],
        "noise_count": int(noise.size),
        "self_count": int(self_scores.size),
        "not_found_in_depth": missing,
        "noise": _percentiles(noise),
        "self": _percentiles(self_scores),
        "median_self_rank": float(np.median(ranks)) if ranks else 0.0,
        "self_rank_one": (
            round(float(np.mean([rank == 1 for rank in ranks])), 3) if ranks else 0.0
        ),
        "noise_above_threshold": (
            round(float(np.mean(noise >= threshold)), 3) if noise.size else 0.0
        ),
        "self_above_threshold": (
            round(float(np.mean(self_scores >= threshold)), 3) if self_scores.size else 0.0
        ),
    }


def _source_histogram(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for point in points:
        counts[point["source"] or "(brak)"] = counts.get(point["source"] or "(brak)", 0) + 1
    return [
        {"source": source, "count": count}
        for source, count in sorted(counts.items(), key=lambda item: -item[1])
    ]


def _interpret(axes: list[dict[str, Any]], threshold: float) -> list[str]:
    if not axes:
        return ["Żadna oś nie zwróciła wyników — kolekcja może nie mieć nazwanych wektorów."]

    notes: list[str] = []
    medians = {item["axis"]: item["noise"]["p50"] for item in axes}
    lowest = min(medians, key=medians.get)
    highest = max(medians, key=medians.get)
    notes.append(
        f"Na realnej kolekcji mediana szumu waha się od {medians[lowest]:.3f} ({lowest}) "
        f"do {medians[highest]:.3f} ({highest}), czyli o {medians[highest] - medians[lowest]:.3f}. "
        "Osie nie leżą w tym samym miejscu skali."
    )

    passing = {item["axis"]: item["noise_above_threshold"] for item in axes}
    inert = [axis for axis, share in passing.items() if share > 0.95]
    if len(inert) == len(axes):
        lowest_noise = min(item["noise"]["min"] for item in axes)
        notes.append(
            f"Próg {threshold} przepuszcza ponad 95% szumu w każdej osi. Najniższy "
            f"zmierzony cosinus na realnych danych to {lowest_noise:.3f}. "
            "Ten próg nie odrzuca niczego."
        )
    elif inert:
        notes.append(
            f"Próg {threshold} jest martwy w osiach: {', '.join(inert)}. "
            "W pozostałych coś odrzuca, więc znaczy w każdej co innego."
        )
    else:
        worst = max(passing, key=passing.get)
        notes.append(
            f"Próg {threshold} przepuszcza od {min(passing.values()):.0%} do "
            f"{passing[worst]:.0%} szumu zależnie od osi."
        )

    separations = []
    for item in axes:
        if item["self_count"]:
            separations.append((item["axis"], item["self"]["p50"] - item["noise"]["p50"]))
    if separations:
        best = max(separations, key=lambda pair: pair[1])
        worst = min(separations, key=lambda pair: pair[1])
        notes.append(
            f"Odstęp trafienia od szumu: najlepiej {best[0]} ({best[1]:.3f}), "
            f"najgorzej {worst[0]} ({worst[1]:.3f}). To górne oszacowanie, bo zapytania "
            "pochodzą z treści dokumentów."
        )

    weak = [item["axis"] for item in axes if item["self_count"] and item["self_rank_one"] < 0.5]
    if weak:
        notes.append(
            f"W osiach {', '.join(weak)} dokument źródłowy rzadziej niż w połowie "
            "przypadków wychodzi na pierwsze miejsce, mimo że zapytanie pochodzi wprost "
            "z jego treści. Te osie wnoszą do fuzji głównie szum."
        )
    return notes
