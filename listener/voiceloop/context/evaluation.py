from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import ContextPackV1


class ContextRetrievalEvalRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    example_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=2000)
    expected_source_ids: tuple[str, ...] = ()
    expected_axes: tuple[str, ...] = ()


class ContextRetrievalScoreV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    predicted_source_ids: tuple[str, ...]
    first_relevant_rank: int | None = Field(default=None, ge=1)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    hit_at_k: bool
    abstention_correct: bool
    provenance_complete: bool
    observed_axes: tuple[str, ...]
    expected_axes_present: bool


class ContextRetrievalMetricsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_count: int = Field(ge=0)
    k: int = Field(ge=1, le=100)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    abstention_accuracy: float = Field(ge=0.0, le=1.0)
    provenance_coverage: float = Field(ge=0.0, le=1.0)
    reserve_axis_coverage: float = Field(ge=0.0, le=1.0)


ContextRetrieve = Callable[
    [ContextRetrievalEvalRecordV1, int],
    Awaitable[ContextPackV1],
]


async def evaluate_context_retrieval(
    *,
    records: Sequence[ContextRetrievalEvalRecordV1],
    retrieve: ContextRetrieve,
    k: int = 8,
) -> tuple[list[ContextRetrievalScoreV1], ContextRetrievalMetricsV1]:
    safe_k = max(1, min(int(k), 100))
    scores: list[ContextRetrievalScoreV1] = []
    for record in records:
        pack = await retrieve(record, safe_k)
        items = list(pack.items[:safe_k])
        predicted = tuple(item.source_id for item in items)
        relevant = set(record.expected_source_ids)
        first_rank = next(
            (
                index
                for index, source_id in enumerate(predicted, start=1)
                if source_id in relevant
            ),
            None,
        )
        observed_axes = tuple(
            dict.fromkeys(
                item.selection_reason.split(":", 1)[1]
                for item in items
                if item.selection_reason.startswith("semantic_scout:")
            )
        )
        negative = not relevant
        scores.append(
            ContextRetrievalScoreV1(
                example_id=record.example_id,
                predicted_source_ids=predicted,
                first_relevant_rank=first_rank,
                reciprocal_rank=(1.0 / first_rank if first_rank else 0.0),
                hit_at_k=first_rank is not None,
                abstention_correct=(not items if negative else first_rank is not None),
                provenance_complete=bool(items)
                and all(item.source and item.source_id for item in items),
                observed_axes=observed_axes,
                expected_axes_present=set(record.expected_axes).issubset(observed_axes),
            )
        )
    metrics = ContextRetrievalMetricsV1(
        sample_count=len(scores),
        k=safe_k,
        recall_at_k=_mean(
            [
                1.0 if score.hit_at_k else 0.0
                for record, score in zip(records, scores, strict=True)
                if record.expected_source_ids
            ]
        ),
        mean_reciprocal_rank=_mean([score.reciprocal_rank for score in scores]),
        abstention_accuracy=_mean(
            [1.0 if score.abstention_correct else 0.0 for score in scores]
        ),
        provenance_coverage=_mean(
            [1.0 if score.provenance_complete else 0.0 for score in scores]
        ),
        reserve_axis_coverage=_mean(
            [1.0 if score.expected_axes_present else 0.0 for score in scores]
        ),
    )
    return scores, metrics


class ContextShadowComparisonV1(BaseModel):
    """Side-by-side metrics. A positive recall delta does not enable recall."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_count: int = Field(ge=0)
    k: int = Field(ge=1, le=100)
    baseline: ContextRetrievalMetricsV1
    candidate: ContextRetrievalMetricsV1
    recall_delta: float
    reciprocal_rank_delta: float
    abstention_delta: float
    provenance_delta: float
    candidate_recall_not_worse: bool
    candidate_abstention_not_worse: bool


async def compare_context_retrieval_shadow(
    *,
    records: Sequence[ContextRetrievalEvalRecordV1],
    baseline: ContextRetrieve,
    candidate: ContextRetrieve,
    k: int = 8,
) -> ContextShadowComparisonV1:
    """Compare the current retriever with the timeline pack. Does not switch traffic."""

    _, baseline_metrics = await evaluate_context_retrieval(
        records=records,
        retrieve=baseline,
        k=k,
    )
    _, candidate_metrics = await evaluate_context_retrieval(
        records=records,
        retrieve=candidate,
        k=k,
    )
    recall_delta = candidate_metrics.recall_at_k - baseline_metrics.recall_at_k
    abstention_delta = (
        candidate_metrics.abstention_accuracy - baseline_metrics.abstention_accuracy
    )
    return ContextShadowComparisonV1(
        sample_count=candidate_metrics.sample_count,
        k=candidate_metrics.k,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        recall_delta=recall_delta,
        reciprocal_rank_delta=(
            candidate_metrics.mean_reciprocal_rank - baseline_metrics.mean_reciprocal_rank
        ),
        abstention_delta=abstention_delta,
        provenance_delta=(
            candidate_metrics.provenance_coverage - baseline_metrics.provenance_coverage
        ),
        candidate_recall_not_worse=recall_delta >= -1e-9,
        candidate_abstention_not_worse=abstention_delta >= -1e-9,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
