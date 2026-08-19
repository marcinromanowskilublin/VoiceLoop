from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .schema import (
    MemoryRetrievalEvalRecordV1,
    MemoryRetrievalMetricsV1,
    MemoryRetrievalScoreCardV1,
)

MemorySearch = Callable[
    [MemoryRetrievalEvalRecordV1, int],
    Awaitable[Sequence[Any]],
]


async def evaluate_memory_retrieval(
    *,
    records: Sequence[MemoryRetrievalEvalRecordV1],
    search: MemorySearch,
    k: int = 5,
) -> tuple[list[MemoryRetrievalScoreCardV1], MemoryRetrievalMetricsV1]:
    safe_k = max(1, min(int(k), 100))
    cards: list[MemoryRetrievalScoreCardV1] = []
    for record in records:
        hits = list(await search(record, safe_k))[:safe_k]
        predicted = tuple(str(getattr(hit, "source_id", "") or "") for hit in hits)
        relevant = set(record.expected_source_ids)
        first_rank = next(
            (index for index, source_id in enumerate(predicted, start=1) if source_id in relevant),
            None,
        )
        reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0
        seen_relevant: set[str] = set()
        dcg = 0.0
        for index, source_id in enumerate(predicted, start=1):
            if source_id not in relevant or source_id in seen_relevant:
                continue
            seen_relevant.add(source_id)
            dcg += 1.0 / math.log2(index + 1)
        ideal_hits = min(len(relevant), safe_k)
        ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        provenance_complete = bool(hits) and all(
            bool(str(getattr(hit, "source", "") or "").strip())
            and bool(str(getattr(hit, "source_id", "") or "").strip())
            for hit in hits
        )
        observed_vector_names: tuple[str, ...] = ()
        if first_rank is not None:
            relevant_hit = hits[first_rank - 1]
            vector_scores = getattr(relevant_hit, "vector_scores", {})
            if isinstance(vector_scores, dict):
                observed_vector_names = tuple(sorted(str(name) for name in vector_scores))
        vector_evidence_complete = (
            set(record.expected_vector_names).issubset(observed_vector_names)
            if record.expected_vector_names
            else True
        )
        cards.append(
            MemoryRetrievalScoreCardV1(
                example_id=record.example_id,
                query=record.query,
                expected_source_ids=record.expected_source_ids,
                predicted_source_ids=predicted,
                expected_vector_names=record.expected_vector_names,
                observed_vector_names=observed_vector_names,
                first_relevant_rank=first_rank,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=(dcg / ideal_dcg if ideal_dcg else 0.0),
                hit_at_k=first_rank is not None,
                provenance_complete=provenance_complete,
                vector_evidence_complete=vector_evidence_complete,
            )
        )

    sample_count = len(cards)
    metrics = MemoryRetrievalMetricsV1(
        sample_count=sample_count,
        k=safe_k,
        recall_at_k=_mean([1.0 if card.hit_at_k else 0.0 for card in cards]),
        mean_reciprocal_rank=_mean([card.reciprocal_rank for card in cards]),
        mean_ndcg_at_k=_mean([card.ndcg_at_k for card in cards]),
        provenance_coverage=_mean(
            [1.0 if card.provenance_complete else 0.0 for card in cards]
        ),
        vector_evidence_coverage=_mean(
            [1.0 if card.vector_evidence_complete else 0.0 for card in cards]
        ),
    )
    return cards, metrics


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
