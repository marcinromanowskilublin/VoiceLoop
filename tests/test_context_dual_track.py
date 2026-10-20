import asyncio
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from voiceloop.context.dual_track import (
    DualTrackRun,
    DualTrackRunner,
    DualTrackTraceV1,
    arbitrate_context_packs,
)
from voiceloop.context.evaluation import (
    ContextRetrievalEvalRecordV1,
    evaluate_context_dual_track,
)
from voiceloop.context.schema import (
    ContextEpisodeV1,
    ContextEventV1,
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
)
from voiceloop.corpus.storage import read_jsonl
from voiceloop.memory import MemoryStore

DAY = datetime(2026, 10, 18, 12, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "fixtures"


def item(source_id: str) -> ContextItemV1:
    return ContextItemV1(
        source="synthetic",
        source_id=source_id,
        scope=ContextScope.EPISODIC,
        content=f"synthetic {source_id}",
        trust=ContextTrust.DERIVED,
    )


@pytest.mark.parametrize(
    ("baseline_ids", "cascade_ids", "verdict", "reason"),
    [
        (("a", "b"), ("a", "c"), "agreement", "top_match"),
        (("a",), ("b",), "conflict", "disjoint_results"),
        (("a", "b"), ("b", "a"), "uncertain", "partial_overlap"),
        ((), ("a",), "uncertain", "missing_track"),
        ((), (), "uncertain", "no_candidates"),
    ],
)
def test_dual_track_arbiter_is_deterministic(
    baseline_ids,
    cascade_ids,
    verdict,
    reason,
) -> None:
    baseline = ContextPackV1(
        question="synthetic",
        items=tuple(item(source_id) for source_id in baseline_ids),
    )
    cascade = ContextPackV1(
        question="synthetic",
        items=tuple(item(source_id) for source_id in cascade_ids),
    )

    decision = arbitrate_context_packs(baseline, cascade)

    assert decision.verdict == verdict
    assert decision.reason == reason


def test_dual_track_arbiter_conflicts_on_different_canonical_versions() -> None:
    baseline_item = item("same-source")
    cascade_item = ContextItemV1(
        source=baseline_item.source,
        source_id=baseline_item.source_id,
        scope=ContextScope.EPISODIC,
        content="inna wersja kanoniczna",
        trust=ContextTrust.DERIVED,
    )

    decision = arbitrate_context_packs(
        ContextPackV1(question="synthetic", items=(baseline_item,)),
        ContextPackV1(question="synthetic", items=(cascade_item,)),
    )

    assert decision.verdict == "conflict"
    assert decision.reason == "disjoint_results"


async def seeded_store(tmp_path):
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    hits = []
    for index in range(2):
        event = ContextEventV1.from_observation(
            source="synthetic",
            source_id=f"event-{index}",
            started_at=DAY + timedelta(minutes=index),
            text=f"Syntetyczne źródło {index}.",
        )
        await store.upsert_context_event(event)
        episode = ContextEpisodeV1(
            episode_id=f"episode-{index}",
            source="synthetic",
            source_id=f"source-{index}",
            started_at=DAY + timedelta(minutes=index),
            ended_at=DAY + timedelta(minutes=index + 1),
            summary=f"Ustalenie syntetyczne {index}.",
            source_event_ids=(event.event_id,),
        )
        await store.upsert_context_episode(episode)
        hits.append(
            SimpleNamespace(
                source=episode.source,
                source_id=episode.source_id,
                score=1.0 - index / 10,
                metadata={
                    "episode_id": episode.episode_id,
                    "content_hash": episode.content_hash,
                    "source_event_hashes": {
                        event.event_id: event.content_hash,
                    },
                },
            )
        )
    return store, hits


class EmbeddingsStub:
    enabled = True

    def accepts_private_text(self) -> bool:
        return True

    async def embed_queries(self, documents):
        return [[1.0, float(index)] for index, _ in enumerate(documents)]


class QdrantStub:
    enabled = True
    collection_name = "voiceloop_context_episodes_v1"

    def __init__(self, hits) -> None:
        self.hits = hits

    def accepts_private_data(self) -> bool:
        return True

    async def search(self, *, vector_names, **_kwargs):
        if len(vector_names) > 1:
            return self.hits
        if vector_names[0] == "semantic":
            return self.hits[:1]
        return self.hits[1:]


@pytest.mark.asyncio
async def test_dual_track_runs_parallel_tracks_and_persists_content_free_trace(
    tmp_path,
) -> None:
    store, hits = await seeded_store(tmp_path)
    runner = DualTrackRunner(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub(hits),  # type: ignore[arg-type]
    )

    result = await runner.run(
        "Co ustaliliśmy wczoraj?",
        request_id="private-request-id",
        now=DAY + timedelta(days=1),
    )
    stored = await store.list_context_dual_track_traces()

    assert result.trace.verdict == "agreement"
    assert result.trace.baseline_axes_attempted == (
        "semantic",
        "topic",
        "intent",
        "decision",
        "person_context",
    )
    assert result.trace.cascade_waves[0] == ("semantic", "decision")
    assert len(stored) == 1
    serialized = stored[0].model_dump_json()
    assert "private-request-id" not in serialized
    assert "source-0" not in serialized
    assert "Co ustaliliśmy" not in serialized
    summary = await store.context_dual_track_summary()
    assert summary["sample_count"] == 1
    assert summary["verdict_counts"] == {"agreement": 1}
    assert summary["time_precision_counts"] == {"exact": 1}
    assert summary["relevance_gate_counts"] == {"context_only": 1}
    assert summary["epistemic_role_counts"] == {}


@pytest.mark.asyncio
async def test_dual_track_keeps_local_timezone_for_relative_day(tmp_path) -> None:
    store, hits = await seeded_store(tmp_path)
    runner = DualTrackRunner(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub(hits),  # type: ignore[arg-type]
    )
    local_midnight = datetime(
        2026,
        10,
        19,
        0,
        30,
        tzinfo=timezone(timedelta(hours=2)),
    )

    result = await runner.run(
        "Co ustaliliśmy wczoraj?",
        request_id="local-timezone",
        now=local_midnight,
    )

    assert result.trace.verdict == "agreement"
    assert result.baseline.items
    assert result.cascade.items


@pytest.mark.asyncio
async def test_dual_track_rejects_unbounded_query_without_providers(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    runner = DualTrackRunner(
        memory=store,
        embeddings=object(),  # type: ignore[arg-type]
        qdrant=SimpleNamespace(  # type: ignore[arg-type]
            collection_name="voiceloop_context_episodes_v1"
        ),
    )

    result = await runner.run("Co ustaliliśmy?", request_id="request-unbounded")

    assert result.trace.reason == "unbounded_rejected"
    assert result.trace.time_precision == "unknown"
    assert result.trace.time_relation == "unbounded"
    assert result.trace.baseline_axes_attempted == ()
    assert result.baseline.items == ()
    assert result.cascade.items == ()


@pytest.mark.asyncio
async def test_dual_track_handles_approximate_epistemic_statement_and_persists_roles(
    tmp_path,
) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    approx_day = DAY - timedelta(days=30)
    event = ContextEventV1.from_observation(
        source="synthetic",
        source_id="event-month-ago",
        started_at=approx_day,
        text="Miesiąc temu omawialiśmy architekturę kontekstu.",
    )
    await store.upsert_context_event(event)
    episode = ContextEpisodeV1(
        episode_id="episode-month-ago",
        source="synthetic",
        source_id="source-month-ago",
        started_at=approx_day,
        ended_at=approx_day + timedelta(minutes=5),
        summary="Architektura kontekstu z zeszłego miesiąca.",
        source_event_ids=(event.event_id,),
    )
    await store.upsert_context_episode(episode)
    hit = SimpleNamespace(
        source=episode.source,
        source_id=episode.source_id,
        score=0.95,
        metadata={
            "episode_id": episode.episode_id,
            "content_hash": episode.content_hash,
            "source_event_hashes": {
                event.event_id: event.content_hash,
            },
        },
    )
    runner = DualTrackRunner(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub([hit]),  # type: ignore[arg-type]
    )
    raw_query = (
        "A jeszcze miesiąc temu mówiliśmy, że będzie wspaniale — "
        "co się zmieniło w projekcie?"
    )

    result = await runner.run(
        raw_query,
        request_id="private-approx-request",
        now=DAY,
    )
    stored_traces = await store.list_context_dual_track_traces()

    assert result.statement is not None
    assert result.trace.bounded is True
    assert result.trace.time_precision == "approximate"
    assert (
        result.trace.time_relation
        == "approximately_one_month_before_observation"
    )
    assert result.trace.epistemic_role == "prediction"
    assert result.trace.relevance_gate == "investigate"
    assert result.trace.verdict == "agreement"
    assert result.trace.cascade_waves[0] == ("semantic", "topic")
    assert len(stored_traces) == 1
    assert stored_traces[0].time_precision == "approximate"
    assert stored_traces[0].epistemic_role == "prediction"
    assert stored_traces[0].relevance_gate == "investigate"
    serialized = stored_traces[0].model_dump_json()
    assert raw_query not in serialized
    assert "private-approx-request" not in serialized
    with store._connect() as connection:
        telemetry_count = connection.execute(
            "SELECT COUNT(*) FROM context_cascade_shadow_telemetry"
        ).fetchone()[0]
    assert telemetry_count == 1
    summary = await store.context_dual_track_summary()
    assert summary["time_precision_counts"] == {"approximate": 1}
    assert summary["relevance_gate_counts"] == {"investigate": 1}
    assert summary["epistemic_role_counts"] == {"prediction": 1}


@pytest.mark.asyncio
async def test_dual_track_can_delegate_temporal_role_persistence(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    runner = DualTrackRunner(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub([]),  # type: ignore[arg-type]
    )

    result = await runner.run(
        "Miesiąc temu mówiliśmy, że będzie wspaniale.",
        request_id="delegated-statement",
        now=DAY,
        persist_statement=False,
    )

    assert result.statement is not None
    assert len(await store.list_context_dual_track_traces()) == 1
    with store._connect() as connection:
        telemetry_count = connection.execute(
            "SELECT COUNT(*) FROM context_cascade_shadow_telemetry"
        ).fetchone()[0]
    assert telemetry_count == 0


@pytest.mark.asyncio
async def test_dual_track_enforces_concurrency_limit(tmp_path) -> None:
    store, hits = await seeded_store(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEmbeddings(EmbeddingsStub):
        async def embed_queries(self, documents):
            started.set()
            await release.wait()
            return await super().embed_queries(documents)

    runner = DualTrackRunner(
        memory=store,
        embeddings=BlockingEmbeddings(),  # type: ignore[arg-type]
        qdrant=QdrantStub(hits),  # type: ignore[arg-type]
        max_inflight=1,
    )
    first = asyncio.create_task(
        runner.run(
            "Co ustaliliśmy wczoraj?",
            request_id="first",
            now=DAY + timedelta(days=1),
        )
    )
    await started.wait()

    second = await runner.run(
        "Co ustaliliśmy wczoraj?",
        request_id="second",
        now=DAY + timedelta(days=1),
    )
    release.set()
    await first

    assert second.trace.reason == "concurrency_limit"
    assert runner.inflight == 0


@pytest.mark.asyncio
async def test_dual_track_timeout_reports_attempted_axes(tmp_path) -> None:
    store, hits = await seeded_store(tmp_path)

    class HangingEmbeddings(EmbeddingsStub):
        async def embed_queries(self, documents):
            await asyncio.sleep(1)
            return await super().embed_queries(documents)

    runner = DualTrackRunner(
        memory=store,
        embeddings=HangingEmbeddings(),  # type: ignore[arg-type]
        qdrant=QdrantStub(hits),  # type: ignore[arg-type]
        timeout_seconds=0.1,
    )

    result = await runner.run(
        "Co ustaliliśmy wczoraj?",
        request_id="timeout",
        now=DAY + timedelta(days=1),
    )

    assert result.trace.reason == "timeout"
    assert len(result.trace.baseline_axes_attempted) == 5
    assert len(result.trace.cascade_axes_attempted) == 2
    assert result.trace.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_dual_track_timeout_preserves_completed_track_and_cancels_other(
    tmp_path,
) -> None:
    store, hits = await seeded_store(tmp_path)
    cascade_cancelled = asyncio.Event()

    class PartialEmbeddings(EmbeddingsStub):
        async def embed_queries(self, documents):
            if len(documents) == 5:
                return await super().embed_queries(documents)
            try:
                await asyncio.sleep(1)
            finally:
                cascade_cancelled.set()
            return await super().embed_queries(documents)

    runner = DualTrackRunner(
        memory=store,
        embeddings=PartialEmbeddings(),  # type: ignore[arg-type]
        qdrant=QdrantStub(hits),  # type: ignore[arg-type]
        timeout_seconds=0.1,
    )

    result = await runner.run(
        "Co ustaliliśmy wczoraj?",
        request_id="partial-timeout",
        now=DAY + timedelta(days=1),
    )

    assert result.trace.reason == "timeout"
    assert result.trace.baseline_count == 2
    assert result.trace.cascade_count == 0
    assert cascade_cancelled.is_set()
    assert runner.inflight == 0


def test_synthetic_dual_track_gold_has_balanced_40_examples() -> None:
    records = read_jsonl(
        FIXTURES / "context-dual-track-gold-v1.jsonl",
        ContextRetrievalEvalRecordV1,
    )

    assert len(records) == 40
    assert {record.category for record in records} == {
        "decision",
        "topic",
        "intent",
        "person_context",
        "general",
        "negative",
    }
    assert all(record.observed_at is not None for record in records)
    assert all(record.expected_time_bounded for record in records)
    assert sum(record.risk == "high" for record in records) >= 8
    assert {
        verdict: sum(record.expected_verdict == verdict for record in records)
        for verdict in ("agreement", "conflict", "uncertain")
    } == {
        "agreement": 24,
        "conflict": 8,
        "uncertain": 8,
    }


@pytest.mark.asyncio
async def test_dual_track_report_compares_rrf_and_cascade_without_promotion() -> None:
    records = [
        ContextRetrievalEvalRecordV1(
            example_id="positive",
            query="Co ustaliliśmy wczoraj?",
            expected_source_ids=("source-1",),
            expected_verdict="agreement",
        ),
        ContextRetrievalEvalRecordV1(
            example_id="negative",
            query="Nieistniejący zapis z wczoraj",
            expected_verdict="uncertain",
        ),
    ]

    async def retrieve(record, _limit):
        positive = record.example_id == "positive"
        baseline = ContextPackV1(
            question=record.query,
            items=(item("source-1"),) if positive else (),
        )
        cascade = ContextPackV1(
            question=record.query,
            items=(item("source-1"),) if positive else (),
        )
        trace = DualTrackTraceV1(
            trace_id=("1" if positive else "2") * 24,
            created_at=DAY,
            expires_at=DAY + timedelta(days=30),
            bounded=True,
            verdict="agreement" if positive else "uncertain",
            reason="top_match" if positive else "no_candidates",
            baseline_count=len(baseline.items),
            cascade_count=len(cascade.items),
            overlap_count=1 if positive else 0,
            top_match=positive,
        )
        return DualTrackRun(baseline=baseline, cascade=cascade, trace=trace)

    report = await evaluate_context_dual_track(
        records=records,
        retrieve=retrieve,
    )

    assert report.baseline.recall_at_k == 1.0
    assert report.cascade.recall_at_k == 1.0
    assert report.verdict_accuracy == 1.0
    assert report.time_boundary_accuracy == 1.0
    assert report.time_boundary_all_match is True
    assert report.verdict_counts == {"agreement": 1, "uncertain": 1}
    assert report.promotion_allowed is False
