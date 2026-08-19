from types import SimpleNamespace

import pytest

from voiceloop.corpus.memory_eval import evaluate_memory_retrieval
from voiceloop.corpus.schema import (
    MemoryRetrievalEvalRecordV1,
    MemoryRetrievalRuntimeConfigV1,
)


@pytest.mark.asyncio
async def test_memory_retrieval_metrics_use_source_ids_and_provenance() -> None:
    records = [
        MemoryRetrievalEvalRecordV1(
            example_id="decision-query",
            query="Co ustaliliśmy w sprawie Deepgramu?",
            expected_source_ids=("meeting:7", "meeting:8"),
            expected_vector_names=("decision", "topic"),
        ),
        MemoryRetrievalEvalRecordV1(
            example_id="missing-query",
            query="Nieistniejąca decyzja",
            expected_source_ids=("meeting:99",),
        ),
    ]

    async def search(record, _k):
        if record.example_id == "decision-query":
            return [
                SimpleNamespace(source="screenpipe_meeting", source_id="meeting:1"),
                SimpleNamespace(
                    source="screenpipe_meeting",
                    source_id="meeting:7",
                    vector_scores={"decision": 0.9, "topic": 0.8},
                ),
            ]
        return []

    cards, metrics = await evaluate_memory_retrieval(records=records, search=search, k=5)

    assert cards[0].first_relevant_rank == 2
    assert cards[0].reciprocal_rank == 0.5
    assert cards[0].hit_at_k is True
    assert cards[0].provenance_complete is True
    assert cards[1].hit_at_k is False
    assert metrics.sample_count == 2
    assert metrics.recall_at_k == 0.5
    assert metrics.mean_reciprocal_rank == 0.25
    assert metrics.provenance_coverage == 0.5
    assert metrics.vector_evidence_coverage == 1.0


def test_memory_retrieval_runtime_fingerprint_covers_weights() -> None:
    runtime = MemoryRetrievalRuntimeConfigV1(
        collection="voiceloop_memory_v2",
        embedding_model="nomic",
        query_format_version="memory-query-documents-v2",
        vector_weights={
            "semantic": 0.4,
            "topic": 0.2,
            "intent": 0.15,
            "decision": 0.15,
            "person_context": 0.1,
        },
        adaptive_query_weights=True,
        min_score=0.15,
        rrf_k=60,
    )

    changed = runtime.model_copy(
        update={"vector_weights": {**runtime.vector_weights, "decision": 0.2}}
    )

    assert runtime.fingerprint() != changed.fingerprint()
