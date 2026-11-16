from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .dual_track import DualTrackRun, DualTrackVerdict
from .schema import ContextPackV1


class ContextRetrievalEvalRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    example_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=2000)
    expected_source_ids: tuple[str, ...] = ()
    expected_axes: tuple[str, ...] = ()
    observed_at: datetime | None = None
    category: str = Field(default="general", min_length=1, max_length=80)
    risk: Literal["low", "medium", "high"] = "low"
    expected_verdict: DualTrackVerdict | None = None
    expected_time_bounded: bool = True

    @field_validator("observed_at")
    @classmethod
    def _aware_observed_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("observed_at must be timezone-aware")
        return value


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
    expected_source_count: int = Field(ge=0)
    relevant_retrieved_count: int = Field(ge=0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    source_recall_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)


class ContextRetrievalMetricsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_count: int = Field(ge=0)
    k: int = Field(ge=1, le=100)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    abstention_accuracy: float = Field(ge=0.0, le=1.0)
    answerability_accuracy: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0)
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
        relevant_retrieved = sum(
            1 for source_id in dict.fromkeys(predicted) if source_id in relevant
        )
        precision = relevant_retrieved / safe_k
        source_recall = (
            relevant_retrieved / len(relevant) if relevant else 0.0
        )
        gains = [1.0 if source_id in relevant else 0.0 for source_id in predicted]
        ideal = [1.0] * min(len(relevant), safe_k)
        ndcg = _dcg(gains) / _dcg(ideal) if ideal else 0.0
        scores.append(
            ContextRetrievalScoreV1(
                example_id=record.example_id,
                predicted_source_ids=predicted,
                first_relevant_rank=first_rank,
                reciprocal_rank=(1.0 / first_rank if first_rank else 0.0),
                hit_at_k=first_rank is not None,
                abstention_correct=(not items if negative else first_rank is not None),
                provenance_complete=bool(items)
                and all(
                    item.source
                    and item.source_id
                    and item.content_hash
                    and item.selection_reason
                    for item in items
                ),
                observed_axes=observed_axes,
                expected_axes_present=set(record.expected_axes).issubset(observed_axes),
                expected_source_count=len(relevant),
                relevant_retrieved_count=relevant_retrieved,
                precision_at_k=precision,
                source_recall_at_k=source_recall,
                ndcg_at_k=ndcg,
            )
        )
    positive_scores = [
        score
        for record, score in zip(records, scores, strict=True)
        if record.expected_source_ids
    ]
    negative_scores = [
        score
        for record, score in zip(records, scores, strict=True)
        if not record.expected_source_ids
    ]
    result_scores = [score for score in scores if score.predicted_source_ids]
    answerability = [
        score.hit_at_k if record.expected_source_ids else not score.predicted_source_ids
        for record, score in zip(records, scores, strict=True)
    ]
    metrics = ContextRetrievalMetricsV1(
        sample_count=len(scores),
        k=safe_k,
        recall_at_k=_mean(
            [score.source_recall_at_k for score in positive_scores]
        ),
        hit_rate_at_k=_mean(
            [1.0 if score.hit_at_k else 0.0 for score in positive_scores]
        ),
        precision_at_k=_mean(
            [score.precision_at_k for score in positive_scores]
        ),
        ndcg_at_k=_mean([score.ndcg_at_k for score in positive_scores]),
        mean_reciprocal_rank=_mean(
            [score.reciprocal_rank for score in positive_scores]
        ),
        abstention_accuracy=_mean(
            [
                1.0 if not score.predicted_source_ids else 0.0
                for score in negative_scores
            ]
        ),
        answerability_accuracy=_mean(
            [1.0 if correct else 0.0 for correct in answerability]
        ),
        false_positive_rate=_mean(
            [
                1.0 if score.predicted_source_ids else 0.0
                for score in negative_scores
            ]
        ),
        provenance_coverage=_mean(
            [
                1.0 if score.provenance_complete else 0.0
                for score in result_scores
            ]
        ),
        reserve_axis_coverage=_mean(
            [
                1.0 if score.expected_axes_present else 0.0
                for record, score in zip(records, scores, strict=True)
                if record.expected_axes
            ]
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


class ContextDualTrackExampleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    category: str
    risk: Literal["low", "medium", "high"]
    verdict: DualTrackVerdict
    expected_verdict: DualTrackVerdict | None = None
    verdict_matches: bool | None = None
    time_bounded: bool
    expected_time_bounded: bool
    time_boundary_matches: bool
    baseline_count: int = Field(ge=0)
    cascade_count: int = Field(ge=0)
    overlap_count: int = Field(ge=0)
    baseline_stop_reason: str
    cascade_stop_reason: str


class ContextDualTrackReportV1(BaseModel):
    """Decision-grade report that still cannot promote retrieval."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sample_count: int = Field(ge=0)
    k: int = Field(ge=1, le=100)
    baseline: ContextRetrievalMetricsV1
    cascade: ContextRetrievalMetricsV1
    recall_delta: float
    reciprocal_rank_delta: float
    abstention_delta: float
    provenance_delta: float
    verdict_counts: dict[str, int]
    labeled_verdict_count: int = Field(ge=0)
    verdict_accuracy: float = Field(ge=0.0, le=1.0)
    time_boundary_accuracy: float = Field(ge=0.0, le=1.0)
    time_boundary_all_match: bool
    cascade_recall_not_worse: bool
    cascade_abstention_not_worse: bool
    promotion_allowed: Literal[False] = False
    examples: tuple[ContextDualTrackExampleV1, ...]


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


DualTrackRetrieve = Callable[
    [ContextRetrievalEvalRecordV1, int],
    Awaitable[DualTrackRun],
]


async def evaluate_context_dual_track(
    *,
    records: Sequence[ContextRetrievalEvalRecordV1],
    retrieve: DualTrackRetrieve,
    k: int = 8,
) -> ContextDualTrackReportV1:
    safe_k = max(1, min(int(k), 100))
    runs: dict[str, DualTrackRun] = {}
    for record in records:
        runs[record.example_id] = await retrieve(record, safe_k)

    async def _baseline(record: ContextRetrievalEvalRecordV1, _limit: int):
        return runs[record.example_id].baseline

    async def _cascade(record: ContextRetrievalEvalRecordV1, _limit: int):
        return runs[record.example_id].cascade

    _, baseline_metrics = await evaluate_context_retrieval(
        records=records,
        retrieve=_baseline,
        k=safe_k,
    )
    _, cascade_metrics = await evaluate_context_retrieval(
        records=records,
        retrieve=_cascade,
        k=safe_k,
    )
    verdict_counts: dict[str, int] = {}
    examples: list[ContextDualTrackExampleV1] = []
    labeled_matches: list[bool] = []
    time_boundary_matches: list[bool] = []
    for record in records:
        trace = runs[record.example_id].trace
        verdict_counts[trace.verdict] = verdict_counts.get(trace.verdict, 0) + 1
        verdict_matches = (
            trace.verdict == record.expected_verdict
            if record.expected_verdict is not None
            else None
        )
        if verdict_matches is not None:
            labeled_matches.append(verdict_matches)
        time_boundary_match = trace.bounded == record.expected_time_bounded
        time_boundary_matches.append(time_boundary_match)
        examples.append(
            ContextDualTrackExampleV1(
                example_id=record.example_id,
                category=record.category,
                risk=record.risk,
                verdict=trace.verdict,
                expected_verdict=record.expected_verdict,
                verdict_matches=verdict_matches,
                time_bounded=trace.bounded,
                expected_time_bounded=record.expected_time_bounded,
                time_boundary_matches=time_boundary_match,
                baseline_count=trace.baseline_count,
                cascade_count=trace.cascade_count,
                overlap_count=trace.overlap_count,
                baseline_stop_reason=trace.baseline_stop_reason,
                cascade_stop_reason=trace.cascade_stop_reason,
            )
        )
    recall_delta = cascade_metrics.recall_at_k - baseline_metrics.recall_at_k
    abstention_delta = (
        cascade_metrics.abstention_accuracy - baseline_metrics.abstention_accuracy
    )
    return ContextDualTrackReportV1(
        sample_count=len(records),
        k=safe_k,
        baseline=baseline_metrics,
        cascade=cascade_metrics,
        recall_delta=recall_delta,
        reciprocal_rank_delta=(
            cascade_metrics.mean_reciprocal_rank
            - baseline_metrics.mean_reciprocal_rank
        ),
        abstention_delta=abstention_delta,
        provenance_delta=(
            cascade_metrics.provenance_coverage
            - baseline_metrics.provenance_coverage
        ),
        verdict_counts=verdict_counts,
        labeled_verdict_count=len(labeled_matches),
        verdict_accuracy=_mean([1.0 if item else 0.0 for item in labeled_matches]),
        time_boundary_accuracy=_mean(
            [1.0 if item else 0.0 for item in time_boundary_matches]
        ),
        time_boundary_all_match=all(time_boundary_matches),
        cascade_recall_not_worse=recall_delta >= -1e-9,
        cascade_abstention_not_worse=abstention_delta >= -1e-9,
        examples=tuple(examples),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(
        gain / math.log2(index + 1)
        for index, gain in enumerate(gains, start=1)
    )
