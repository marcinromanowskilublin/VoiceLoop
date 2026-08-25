"""Orkiestracja analizy: fragmenty, wektory, geometria, miary zniekształcenia.

Przepływ jest jednokierunkowy i celowo rozdzielony: klient embeddingów wysyła do
LM Studio **tekst**, a wraca z niego **wektor 768D**. Dopiero ten wektor jest
wejściem do geometrii. Vectorscope nigdy nie zapisuje wektorów do Qdranta —
pamięć zapisuje VoiceLoop, panel ją wyłącznie odpytuje.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import geometry
from .anchors import ANCHOR_PAIRS, anchor_texts
from .config import PREFIX_DOCUMENT, build_embedding_client, collect_thresholds, settings
from .embed import embed_texts_with_prefix
from .fragments import (
    LEVELS,
    Fragment,
    build_fragments,
    merge_identical,
    select_levels,
)
from .store import RecordingStore

MAX_FRAGMENTS = 400
MAX_HEATMAP_FRAGMENTS = 220
NEIGHBOUR_COUNT = 8

ANCHOR_GROUP = "__kotwice__"
REFERENCE_GROUP = "__referencja__"

PROJECTION_MDS = "mds"
PROJECTION_PCA = "pca"
PROJECTIONS = (PROJECTION_MDS, PROJECTION_PCA)


@dataclass
class AnalysisRequest:
    recording_ids: list[str]
    levels: list[str]
    prefix: str = PREFIX_DOCUMENT
    neighbours: int = 4
    threshold: float | None = 0.15
    projection: str = PROJECTION_MDS
    include_anchors: bool = True
    reference_texts: list[str] | None = None
    merge_identical: bool = False


def nearest_shown_ancestor(
    fragment_id: str,
    parent_of: dict[str, str | None],
    index_by_id: dict[str, int],
) -> int | None:
    """Najbliższy przodek obecny na wykresie, także gdy poziomy nie sąsiadują.

    Przy wyborze samych słów i zdań rodzicem słowa jest fraza, której na
    wykresie nie ma. Bez wspinaczki po łańcuchu linie rodzic–dziecko zniknęłyby
    zupełnie, choć hierarchia istnieje. Zbiór odwiedzonych chroni przed
    zapętleniem, gdyby dane na dysku okazały się uszkodzone.
    """

    seen: set[str] = {fragment_id}
    current = parent_of.get(fragment_id)
    while current and current not in seen:
        if current in index_by_id:
            return index_by_id[current]
        seen.add(current)
        current = parent_of.get(current)
    return None


def hierarchy_edges(
    fragment_ids: list[str],
    parent_of: dict[str, str | None],
    index_by_id: dict[str, int],
) -> list[dict[str, int]]:
    edges: list[dict[str, int]] = []
    for fragment_id in fragment_ids:
        ancestor = nearest_shown_ancestor(fragment_id, parent_of, index_by_id)
        if ancestor is not None and ancestor != index_by_id[fragment_id]:
            edges.append({"source": index_by_id[fragment_id], "target": ancestor})
    return edges


async def run_analysis(
    request: AnalysisRequest,
    store: RecordingStore,
) -> dict[str, Any]:
    levels = [level for level in request.levels if level in LEVELS]
    if not levels:
        raise ValueError(f"Wybierz przynajmniej jeden poziom z {list(LEVELS)}.")
    if request.projection not in PROJECTIONS:
        raise ValueError(f"Nieznany rzut: {request.projection}")

    active_settings = settings()
    client = build_embedding_client(active_settings)

    warnings: list[str] = []
    fragments: list[Fragment] = []
    vectors_parts: list[np.ndarray] = []
    parent_text_by_id: dict[str, str] = {}
    parent_of: dict[str, str | None] = {}
    merged_members: dict[str, list[str]] = {}
    processed_ids: list[str] = []
    model_name = "nieznany"
    embed_started = time.perf_counter()

    for recording_id in request.recording_ids:
        meta = store.read_meta(recording_id)
        transcript = store.read_transcript(recording_id)
        if not transcript:
            warnings.append(f"{meta.label}: brak transkrypcji — nagranie pominięte.")
            continue

        all_fragments = build_fragments(
            transcript=transcript,
            recording_id=recording_id,
            recording_label=meta.label,
        )
        if not all_fragments:
            warnings.append(f"{meta.label}: transkrypt nie dał żadnych fragmentów.")
            continue
        store.write_fragments(
            recording_id,
            [fragment.to_payload(index) for index, fragment in enumerate(all_fragments)],
        )
        for fragment in all_fragments:
            parent_text_by_id[fragment.id] = fragment.text
            parent_of[fragment.id] = fragment.parent_id

        meta_changed = False
        for level in levels:
            level_fragments = select_levels(all_fragments, [level])
            if not level_fragments:
                continue
            if request.merge_identical:
                level_fragments, members = merge_identical(level_fragments)
                for fragment, group in zip(level_fragments, members, strict=True):
                    if len(group) > 1:
                        merged_members[fragment.id] = group

            unique_texts: list[str] = []
            seen: set[str] = set()
            for fragment in level_fragments:
                if fragment.text not in seen:
                    seen.add(fragment.text)
                    unique_texts.append(fragment.text)

            cached = store.read_vectors(recording_id, level, request.prefix, unique_texts)
            if cached is not None and cached.shape[0] == len(unique_texts):
                matrix = cached
            else:
                result = await embed_texts_with_prefix(
                    client, unique_texts, prefix=request.prefix
                )
                matrix = result.vectors
                model_name = result.model
                store.write_vectors(
                    recording_id,
                    level,
                    request.prefix,
                    unique_texts,
                    matrix,
                    model=result.model,
                )
                meta.record_embedding_run(
                    level=level,
                    prefix=request.prefix,
                    model=result.model,
                    dimension=result.dimension,
                    fragment_count=len(level_fragments),
                    over_context=len(result.over_context),
                )
                meta_changed = True
                if result.over_context:
                    warnings.append(
                        f"{meta.label} / {level}: {len(result.over_context)} fragmentów "
                        "przekracza 512 tokenów kontekstu — model uciął je po cichu."
                    )

            lookup = {text: matrix[index] for index, text in enumerate(unique_texts)}
            fragments.extend(level_fragments)
            vectors_parts.append(
                np.vstack([lookup[fragment.text] for fragment in level_fragments])
            )

        processed_ids.append(recording_id)
        if meta_changed:
            store.write_meta(meta)

    reference_texts = [
        text.strip() for text in (request.reference_texts or []) if text.strip()
    ]
    if reference_texts:
        result = await embed_texts_with_prefix(client, reference_texts, prefix=request.prefix)
        model_name = result.model
        for index, text in enumerate(reference_texts, start=1):
            fragments.append(
                Fragment(
                    id=f"{REFERENCE_GROUP}:r{index}",
                    level="reference",
                    text=text,
                    start_ms=None,
                    end_ms=None,
                    parent_id=None,
                    recording_id=REFERENCE_GROUP,
                    recording_label="referencja",
                    speaker=None,
                    order=index,
                    word_count=len(text.split()),
                )
            )
        vectors_parts.append(result.vectors)

    anchor_offset: int | None = None
    if request.include_anchors:
        texts = anchor_texts()
        result = await embed_texts_with_prefix(client, texts, prefix=request.prefix)
        model_name = result.model
        anchor_offset = len(fragments)
        for index, text in enumerate(texts, start=1):
            fragments.append(
                Fragment(
                    id=f"{ANCHOR_GROUP}:a{index}",
                    level="anchor",
                    text=text,
                    start_ms=None,
                    end_ms=None,
                    parent_id=None,
                    recording_id=ANCHOR_GROUP,
                    recording_label="kotwica",
                    speaker=None,
                    order=index,
                    word_count=len(text.split()),
                )
            )
        vectors_parts.append(result.vectors)

    embed_ms = (time.perf_counter() - embed_started) * 1000.0

    if not fragments or not vectors_parts:
        return {
            "ok": False,
            "message": "Brak fragmentów do analizy. Nagraj coś i zrób transkrypcję.",
            "warnings": warnings,
        }

    dimensions = {part.shape[1] for part in vectors_parts if part.size}
    if len(dimensions) > 1:
        raise ValueError(f"Niespójny wymiar wektorów: {sorted(dimensions)}")

    vectors = np.vstack([part for part in vectors_parts if part.size])

    if len(fragments) > MAX_FRAGMENTS:
        warnings.append(
            f"Ograniczono z {len(fragments)} do {MAX_FRAGMENTS} fragmentów, żeby panel "
            "pozostał czytelny. Wybierz wyższy poziom albo włącz scalanie identycznych."
        )
        fragments = fragments[:MAX_FRAGMENTS]
        vectors = vectors[:MAX_FRAGMENTS]
        if anchor_offset is not None and anchor_offset >= MAX_FRAGMENTS:
            anchor_offset = None

    geometry_started = time.perf_counter()
    count = len(fragments)
    cosine = geometry.cosine_matrix(vectors)
    distance = geometry.cosine_distance(cosine)

    edges = geometry.knn_edges(cosine, k=request.neighbours, threshold=request.threshold)
    if edges.truncated:
        warnings.append(
            f"Graf ma {edges.total_candidates} krawędzi kandydujących; pokazano "
            f"{geometry.MAX_GRAPH_EDGES} najsilniejszych. Podnieś próg, żeby zobaczyć mniej."
        )

    linkage = geometry.average_linkage(distance)
    order = geometry.leaf_order(linkage, count)
    segments = geometry.dendrogram_segments(linkage, count, order)

    projection = (
        geometry.smacof(distance)
        if request.projection == PROJECTION_MDS
        else geometry.pca_2d(vectors)
    )
    projected_distance = geometry.pairwise_euclidean(projection.coords)

    metric_limit = geometry.max_valid_neighbours(count)
    metric_k = max(1, min(request.neighbours if request.neighbours > 0 else 4, metric_limit))
    trust = geometry.trustworthiness(distance, projected_distance, metric_k)
    cont = geometry.continuity(distance, projected_distance, metric_k)
    stress = geometry.kruskal_stress(distance, projected_distance)
    preservation = geometry.neighbourhood_preservation(distance, projected_distance, metric_k)
    shepard_high, shepard_low, shepard_sampled = geometry.shepard_sample(
        distance, projected_distance
    )
    centres, counts = geometry.pair_histogram(cosine)
    geometry_ms = (time.perf_counter() - geometry_started) * 1000.0

    for recording_id in processed_ids:
        try:
            meta = store.read_meta(recording_id)
        except (FileNotFoundError, ValueError):
            continue
        meta.record_timing("embedding", embed_ms)
        meta.record_timing("geometry", geometry_ms)
        store.write_meta(meta)

    include_heatmap = count <= MAX_HEATMAP_FRAGMENTS
    if not include_heatmap:
        warnings.append(
            f"Heatmapa wyłączona przy {count} fragmentach (limit {MAX_HEATMAP_FRAGMENTS}) — "
            "macierz byłaby nieczytelna. Dendrogram i graf działają dalej."
        )

    index_by_id = {fragment.id: index for index, fragment in enumerate(fragments)}
    hierarchy = hierarchy_edges(
        [fragment.id for fragment in fragments], parent_of, index_by_id
    )

    return {
        "ok": True,
        "warnings": warnings,
        "levels": levels,
        "prefix": request.prefix,
        "model": model_name,
        "dimension": int(vectors.shape[1]),
        "fragment_count": count,
        "merge_identical": request.merge_identical,
        "timings_ms": {
            "embedding": round(embed_ms, 2),
            "geometry": round(geometry_ms, 2),
        },
        "fragments": [
            fragment.to_payload(
                index,
                parent_text=parent_text_by_id.get(fragment.parent_id or ""),
            )
            | {"merged_ids": merged_members.get(fragment.id, [])}
            for index, fragment in enumerate(fragments)
        ],
        "groups": _groups(fragments),
        "level_counts": _level_counts(fragments),
        "edges": [
            {"source": source, "target": target, "cosine": round(value, 6)}
            for source, target, value in edges.pairs
        ],
        "hierarchy_edges": hierarchy,
        "edge_stats": {
            "shown": len(edges.pairs),
            "candidates": edges.total_candidates,
            "truncated": edges.truncated,
            "neighbours": request.neighbours,
            "threshold": request.threshold,
        },
        "heatmap": (
            {
                "order": order,
                "labels": [fragments[index].text for index in order],
                "matrix": [
                    [round(float(cosine[row, column]), 4) for column in order]
                    for row in order
                ],
            }
            if include_heatmap
            else None
        ),
        "dendrogram": {
            "segments": segments,
            "order": order,
            "labels": [fragments[index].text for index in order],
            "metric": (
                "odległość cosinusowa, linkage średni (UPGMA), "
                "liczone w pełnych 768 wymiarach — bez rzutowania"
            ),
        },
        "projection": {
            "method": projection.method,
            "coords": [
                [float(projection.coords[index, 0]), float(projection.coords[index, 1])]
                for index in range(count)
            ],
            "explained_variance": projection.explained_variance,
            "explained_variance_total": projection.explained_variance_total,
            "iterations": projection.iterations,
        },
        "distortion": {
            "trustworthiness": trust,
            "continuity": cont,
            "stress": stress,
            "metric_neighbours": metric_k,
            "metric_neighbours_limit": metric_limit,
            "per_unit_preservation": [float(value) for value in preservation],
            "shepard": {
                "high": [round(value, 5) for value in shepard_high],
                "low": [round(value, 5) for value in shepard_low],
                "sampled": shepard_sampled,
            },
        },
        "histogram": {"centres": centres, "counts": counts},
        "thresholds": [
            {
                "key": item.key,
                "value": item.value,
                "label": item.label,
                "origin": item.origin,
            }
            for item in collect_thresholds(active_settings)
        ],
        "neighbours": _nearest_neighbours(cosine, fragments),
        "anchors": (
            _anchor_scores(cosine, fragments, anchor_offset)
            if anchor_offset is not None
            else []
        ),
    }


def _groups(fragments: list[Fragment]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        entry = seen.setdefault(
            fragment.recording_id,
            {"id": fragment.recording_id, "label": fragment.recording_label, "count": 0},
        )
        entry["count"] += 1
    return list(seen.values())


def _level_counts(fragments: list[Fragment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fragment in fragments:
        counts[fragment.level] = counts.get(fragment.level, 0) + 1
    return counts


def _nearest_neighbours(
    cosine: np.ndarray,
    fragments: list[Fragment],
) -> list[list[dict[str, Any]]]:
    count = len(fragments)
    if count < 2:
        return [[] for _ in range(count)]
    work = cosine.copy()
    np.fill_diagonal(work, -np.inf)
    limit = min(NEIGHBOUR_COUNT, count - 1)
    payload: list[list[dict[str, Any]]] = []
    for row in range(count):
        candidates = np.argpartition(-work[row], limit - 1)[:limit]
        ordered = candidates[np.argsort(-work[row][candidates])]
        payload.append(
            [
                {
                    "index": int(index),
                    "text": fragments[int(index)].text,
                    "level": fragments[int(index)].level,
                    "group": fragments[int(index)].recording_id,
                    "cosine": round(float(cosine[row, int(index)]), 6),
                }
                for index in ordered
            ]
        )
    return payload


def _anchor_scores(
    cosine: np.ndarray,
    fragments: list[Fragment],
    anchor_offset: int,
) -> list[dict[str, Any]]:
    lookup: dict[str, int] = {}
    for index in range(anchor_offset, len(fragments)):
        lookup.setdefault(fragments[index].text, index)

    payload: list[dict[str, Any]] = []
    for pair in ANCHOR_PAIRS:
        left = lookup.get(pair.left)
        right = lookup.get(pair.right)
        if left is None or right is None:
            continue
        value = 1.0 if left == right else float(cosine[left, right])
        payload.append(
            {
                "key": pair.key,
                "relation": pair.relation,
                "expectation": pair.expectation,
                "left": pair.left,
                "right": pair.right,
                "cosine": round(value, 6),
            }
        )
    return payload
