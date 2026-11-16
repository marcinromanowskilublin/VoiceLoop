from __future__ import annotations

import pytest

from voiceloop.context.evaluation import (
    ContextRetrievalEvalRecordV1,
    evaluate_context_retrieval,
)
from voiceloop.context.schema import (
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
)


def item(source_id: str) -> ContextItemV1:
    return ContextItemV1(
        source="synthetic",
        source_id=source_id,
        scope=ContextScope.EPISODIC,
        content=f"Treść {source_id}",
        trust=ContextTrust.DERIVED,
        selection_reason="timeline_episode_fts",
    )


@pytest.mark.asyncio
async def test_recall_counts_all_expected_sources_not_only_one_hit() -> None:
    records = [
        ContextRetrievalEvalRecordV1(
            example_id="multi-source",
            query="co ustalono",
            expected_source_ids=("a", "b"),
        ),
        ContextRetrievalEvalRecordV1(
            example_id="negative",
            query="brak",
        ),
    ]

    async def retrieve(record, _k):
        if record.example_id == "multi-source":
            return ContextPackV1(
                question=record.query,
                items=(item("a"), item("noise")),
            )
        return ContextPackV1(question=record.query, items=())

    scores, metrics = await evaluate_context_retrieval(
        records=records,
        retrieve=retrieve,
        k=2,
    )

    positive = scores[0]
    assert positive.hit_at_k is True
    assert positive.relevant_retrieved_count == 1
    assert positive.source_recall_at_k == 0.5
    assert positive.precision_at_k == 0.5
    assert 0.0 < positive.ndcg_at_k < 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 0.5
    assert metrics.abstention_accuracy == 1.0
    assert metrics.answerability_accuracy == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.provenance_coverage == 1.0


@pytest.mark.asyncio
async def test_negative_false_positive_is_reported_separately() -> None:
    record = ContextRetrievalEvalRecordV1(
        example_id="negative",
        query="nieistniejące",
    )

    async def retrieve(record, _k):
        return ContextPackV1(question=record.query, items=(item("noise"),))

    _, metrics = await evaluate_context_retrieval(
        records=[record],
        retrieve=retrieve,
        k=3,
    )

    assert metrics.abstention_accuracy == 0.0
    assert metrics.answerability_accuracy == 0.0
    assert metrics.false_positive_rate == 1.0


@pytest.mark.asyncio
async def test_duplicate_source_ids_do_not_inflate_ndcg() -> None:
    record = ContextRetrievalEvalRecordV1(
        example_id="duplicate",
        query="duplikat",
        expected_source_ids=("a",),
    )

    async def retrieve(record, _k):
        return ContextPackV1(
            question=record.query,
            items=(item("a"), item("a")),
        )

    scores, metrics = await evaluate_context_retrieval(
        records=[record],
        retrieve=retrieve,
        k=2,
    )

    assert scores[0].ndcg_at_k == 1.0
    assert metrics.ndcg_at_k == 1.0


@pytest.mark.asyncio
async def test_adaptive_metadata_preserves_axis_coverage() -> None:
    record = ContextRetrievalEvalRecordV1(
        example_id="axis",
        query="temat",
        expected_source_ids=("a",),
        expected_axes=("topic",),
    )
    adaptive_item = item("a").model_copy(
        update={
            "selection_reason": "adaptive_hybrid:vector",
            "metadata": {"retrieval_axes": ["semantic", "topic"]},
        }
    )

    async def retrieve(record, _k):
        return ContextPackV1(question=record.query, items=(adaptive_item,))

    scores, metrics = await evaluate_context_retrieval(
        records=[record],
        retrieve=retrieve,
        k=2,
    )

    assert scores[0].observed_axes == ("semantic", "topic")
    assert scores[0].expected_axes_present is True
    assert metrics.reserve_axis_coverage == 1.0
