"""Tryb diagnostyczny: pięć osi pamięci VoiceLoopa osobno i po fuzji.

Odwzorowuje ścieżkę z `Assistant._vector_memories_for_request`
(listener/voiceloop/assistant.py:916-1007), żeby panel pokazywał ten sam
retrieval, który realnie karmi model — a nie własną, podobnie wyglądającą wersję.

Dodatkowo odpytuje każdą oś pojedynczo. Fuzja RRF ukrywa, która przestrzeń
wciągnęła dany rekord; pojedyncze rankingi to odsłaniają.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voiceloop.embeddings import EmbeddingUnavailableError
from voiceloop.memory import MemoryStore
from voiceloop.memory_vectorization import (
    MEMORY_VECTOR_NAMES,
    memory_query_documents,
    memory_query_weights,
)
from voiceloop.qdrant_memory import QdrantMemoryError, QdrantVectorStore

from .config import build_embedding_client, settings

CONTENT_PREVIEW = 400


@dataclass
class DiagnosticsRequest:
    query: str
    limit: int | None = None
    min_score: float | None = None
    adaptive: bool | None = None


async def run_diagnostics(request: DiagnosticsRequest) -> dict[str, Any]:
    active = settings()
    query = request.query.strip()
    if not query:
        return {"ok": False, "message": "Podaj pytanie, które ma trafić do pamięci."}

    limit = request.limit or active.vector_memory_context_limit
    min_score = (
        request.min_score
        if request.min_score is not None
        else active.vector_memory_min_score
    )
    adaptive = (
        request.adaptive
        if request.adaptive is not None
        else active.vector_memory_adaptive_query_weights
    )

    documents = memory_query_documents(query[:2000])
    axis_names = tuple(name for name in MEMORY_VECTOR_NAMES if name in documents)
    if not axis_names:
        return {"ok": False, "message": "Nie udało się zbudować dokumentów zapytania."}

    client = build_embedding_client(active)
    try:
        vectors = await client.embed_queries([documents[name] for name in axis_names])
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}
    if len(vectors) != len(axis_names):
        return {
            "ok": False,
            "message": (
                f"LM Studio zwrócił {len(vectors)} wektorów dla {len(axis_names)} osi — "
                "przerywam, bo mapowanie osi byłoby zmyślone."
            ),
        }

    query_vectors = dict(zip(axis_names, vectors, strict=True))
    weights = memory_query_weights(
        query,
        adaptive=adaptive,
        base_weights=active.vector_memory_weights or None,
    )

    payload: dict[str, Any] = {
        "ok": True,
        "query": query,
        "query_documents": {name: documents[name] for name in axis_names},
        "axes": list(axis_names),
        "weights": {name: round(float(weights.get(name, 0.0)), 6) for name in axis_names},
        "base_weights": {
            name: round(float(active.vector_memory_weights.get(name, 0.0)), 6)
            for name in axis_names
        },
        "adaptive_weights": adaptive,
        "rrf_k": active.vector_memory_rrf_k,
        "min_score": min_score,
        "limit": limit,
        "dimension": len(vectors[0]) if vectors else 0,
        "per_axis": {},
        "fused": [],
        "warnings": [],
    }

    store = QdrantVectorStore(active)
    try:
        healthy, detail = await store.health()
        payload["qdrant"] = {
            "enabled": store.enabled,
            "url": store.url,
            "collection": store.collection_name,
            "healthy": healthy,
            "detail": detail,
        }
        if store.enabled and healthy and store.accepts_private_data():
            try:
                fused = await store.search(
                    query_vectors=query_vectors,
                    limit=limit,
                    min_score=min_score,
                    query_weights=weights,
                    rrf_k=active.vector_memory_rrf_k,
                )
                payload["fused"] = [_fused_payload(hit) for hit in fused]
            except QdrantMemoryError as exc:
                payload["warnings"].append(f"Fuzja RRF nieudana: {exc}")

            for name in axis_names:
                try:
                    hits = await store.search(
                        query_vectors={name: query_vectors[name]},
                        limit=limit,
                        min_score=min_score,
                        vector_names=(name,),
                        query_weights={name: 1.0},
                        rrf_k=active.vector_memory_rrf_k,
                    )
                except QdrantMemoryError as exc:
                    payload["warnings"].append(f"Oś {name} nieudana: {exc}")
                    payload["per_axis"][name] = []
                    continue
                payload["per_axis"][name] = [
                    _axis_payload(hit, name, rank)
                    for rank, hit in enumerate(hits, start=1)
                ]
        else:
            payload["warnings"].append(
                "Qdrant niedostępny — uruchom scripts/start-qdrant.ps1. "
                "Pokazuję awaryjny wynik z SQLite na pojedynczej osi semantycznej."
            )
            payload["fallback"] = await _sqlite_fallback(
                active,
                query_vectors.get("semantic", []),
                limit=limit,
                min_score=min_score,
            )
    finally:
        await store.close()

    payload["source_summary"] = _source_summary(payload)
    return payload


def _fused_payload(hit: Any) -> dict[str, Any]:
    evidence = getattr(hit, "evidence", {}) or {}
    return {
        "source": hit.source,
        "source_id": hit.source_id,
        "title": hit.title,
        "content": (hit.content or "")[:CONTENT_PREVIEW],
        "created_at": hit.created_at.isoformat() if hit.created_at else None,
        "fusion_score": round(float(getattr(hit, "fusion_score", hit.score)), 6),
        "vector_scores": {
            name: round(float(value), 6)
            for name, value in (getattr(hit, "vector_scores", {}) or {}).items()
        },
        "vector_ranks": dict(getattr(hit, "vector_ranks", {}) or {}),
        "evidence": {
            name: {
                "score": round(float(item.get("score", 0.0)), 6),
                "rank": int(item.get("rank", 0)),
                "weight": round(float(item.get("weight", 0.0)), 6),
                "rrf_contribution": round(float(item.get("rrf_contribution", 0.0)), 8),
            }
            for name, item in evidence.items()
        },
        "axes_hit": sorted(evidence.keys()),
    }


def _axis_payload(hit: Any, axis: str, rank: int) -> dict[str, Any]:
    evidence = (getattr(hit, "evidence", {}) or {}).get(axis, {})
    return {
        "rank": rank,
        "source": hit.source,
        "source_id": hit.source_id,
        "title": hit.title,
        "content": (hit.content or "")[:CONTENT_PREVIEW],
        "cosine": round(float(evidence.get("score", hit.score)), 6),
    }


async def _sqlite_fallback(
    active: Any,
    semantic_vector: list[float],
    *,
    limit: int,
    min_score: float,
) -> dict[str, Any]:
    database = active.data_dir / "voiceloop.db"
    if not semantic_vector:
        return {"available": False, "reason": "brak wektora semantycznego"}
    if not database.exists():
        return {"available": False, "reason": f"brak bazy {database}"}
    memory = MemoryStore(database)
    try:
        hits = await memory.search_vector_memories(
            semantic_vector,
            limit=limit,
            min_score=min_score,
        )
    except Exception as exc:  # noqa: BLE001 - awaryjna ścieżka nie może wywrócić panelu
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "database": str(database),
        "items": [
            {
                "source": hit.source,
                "source_id": hit.source_id,
                "title": hit.title,
                "content": (hit.content or "")[:CONTENT_PREVIEW],
                "cosine": round(float(hit.score), 6),
            }
            for hit in hits
        ],
    }


def _source_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in payload.get("fused", []):
        source = item.get("source") or "(brak)"
        counts[source] = counts.get(source, 0) + 1
    for axis_hits in (payload.get("per_axis") or {}).values():
        for item in axis_hits:
            source = item.get("source") or "(brak)"
            counts.setdefault(source, 0)
    return [
        {"source": source, "fused_count": count}
        for source, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
