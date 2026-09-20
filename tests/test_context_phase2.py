from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from voiceloop.context.entities import (
    ContextEntityCandidateV1,
    ContextEntityKind,
)
from voiceloop.context.entity_registry import ContextEntityRegistry
from voiceloop.context.evaluation import (
    ContextRetrievalEvalRecordV1,
    evaluate_context_retrieval,
)
from voiceloop.context.foreground import ForegroundIdentity, ForegroundSampler
from voiceloop.context.lifecycle import ContextLifecycleService
from voiceloop.context.meeting_ingest import MeetingTimelineIngestor
from voiceloop.context.retrieval import TimeFirstRetriever
from voiceloop.context.schema import (
    ContextEpisodeV1,
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
    stable_episode_id,
)
from voiceloop.context.vectorization import ContextEpisodeVectorizer
from voiceloop.memory import MemoryStore


@pytest.mark.asyncio
async def test_lifecycle_prunes_qdrant_before_local_tombstone(tmp_path) -> None:
    class QdrantStub:
        enabled = True

        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []

        async def delete_memory(self, *, source: str, source_id: str) -> None:
            self.deleted.append((source, source_id))

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = ContextEpisodeV1(
        episode_id=stable_episode_id("timeline", "expired"),
        source="timeline",
        source_id="expired",
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
        ended_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
        title="Stary epizod",
        summary="Wygasły kontekst.",
        source_event_ids=("event-1",),
        expires_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    await store.upsert_context_episode(episode)
    qdrant = QdrantStub()
    lifecycle = ContextLifecycleService(
        memory=store,
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    dry = await lifecycle.prune_expired(
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )
    assert dry.episodes == 1
    assert qdrant.deleted == []

    applied = await lifecycle.prune_expired(
        now=datetime(2026, 9, 20, tzinfo=UTC),
        dry_run=False,
    )
    assert applied.qdrant_deleted == 1
    assert qdrant.deleted == [("timeline", "expired")]
    stored = await store.get_context_episode(episode.episode_id)
    assert stored is not None and stored.deleted_at is not None


@pytest.mark.asyncio
async def test_lifecycle_fails_closed_when_qdrant_delete_fails(tmp_path) -> None:
    class BrokenQdrant:
        enabled = True

        async def delete_memory(self, **_kwargs) -> None:
            raise RuntimeError("qdrant down")

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = ContextEpisodeV1(
        episode_id=stable_episode_id("timeline", "keep"),
        source="timeline",
        source_id="keep",
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
        ended_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
        title="Epizod",
        summary="Nie kasuj lokalnie po błędzie Qdrant.",
        source_event_ids=("event-1",),
        expires_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    await store.upsert_context_episode(episode)
    lifecycle = ContextLifecycleService(
        memory=store,
        qdrant=BrokenQdrant(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="qdrant down"):
        await lifecycle.prune_expired(
            now=datetime(2026, 9, 20, tzinfo=UTC),
            dry_run=False,
        )
    stored = await store.get_context_episode(episode.episode_id)
    assert stored is not None and stored.deleted_at is None


@pytest.mark.asyncio
async def test_entity_candidate_requires_two_sources_and_strong_signal(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    registry = ContextEntityRegistry(store)
    candidate = await registry.submit_candidate(
        ContextEntityCandidateV1(
            canonical="Jan Kowalski",
            kind=ContextEntityKind.PERSON,
            aliases=("Jan", "Janowi"),
            evidence_source_ids=("ocr:1",),
            strong_evidence_count=0,
        )
    )
    assert candidate.ready_for_review is False
    with pytest.raises(ValueError, match="two independent"):
        await registry.approve_candidate(candidate.candidate_id)

    merged = await registry.submit_candidate(
        ContextEntityCandidateV1(
            canonical="Jan Kowalski",
            kind=ContextEntityKind.PERSON,
            aliases=("Jankowi",),
            evidence_source_ids=("chat_header:2",),
            strong_evidence_count=1,
        )
    )
    assert merged.candidate_id == candidate.candidate_id
    assert merged.ready_for_review is True
    entity = await registry.approve_candidate(merged.candidate_id)
    assert entity.approved is True
    resolved = await registry.resolve("Janowi", kind="person")
    assert resolved[0].entity_id == entity.entity_id


@pytest.mark.asyncio
async def test_meeting_recorder_and_screenpipe_share_timeline(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    start = datetime(2026, 9, 19, 10, tzinfo=UTC)
    await store.create_meeting_session(
        session_id="session-1",
        started_at=start,
        audio_dir=tmp_path / "audio",
        title="Spotkanie VoiceLoop",
    )
    await store.save_meeting_transcript_segment(
        session_id="session-1",
        segment_key="segment-1",
        channel="input",
        speaker_label="Użytkownik",
        speaker_id=0,
        device_name="mikrofon",
        start_time=start,
        end_time=start.replace(minute=1),
        text="Ustaliliśmy timeline.",
        transcript=None,
        source="test",
    )
    await store.save_screenpipe_transcript(
        chunk_id="screenpipe-1",
        meeting_id=7,
        device_name="speaker",
        device_type="output",
        start_time="2026-09-19T11:00:00+00:00",
        end_time="2026-09-19T11:01:00+00:00",
        text="Rozmowa o Qdrant.",
        source="test",
    )
    ingestor = MeetingTimelineIngestor(store)

    recorded = await ingestor.ingest_recorded_meetings()
    screenpipe = await ingestor.ingest_screenpipe_meetings()

    assert (recorded.events, recorded.episodes) == (1, 1)
    assert (screenpipe.events, screenpipe.episodes) == (1, 1)
    assert len(await store.search_context_episodes(query="timeline")) == 1
    assert len(await store.search_context_episodes(query="Qdrant")) == 1


def test_foreground_sampler_marks_observed_identity(monkeypatch) -> None:
    import voiceloop.context.foreground as foreground

    monkeypatch.setattr(
        foreground,
        "_foreground_identity",
        lambda: ForegroundIdentity(
            hwnd=22,
            process_id=11,
            process_name="Cursor.exe",
            window_title="VoiceLoop - Cursor",
            window_class="Chrome_WidgetWin_1",
        ),
    )

    event = ForegroundSampler().capture(
        observed_at=datetime(2026, 9, 20, 5, tzinfo=UTC)
    )

    assert event.is_foreground is True
    assert event.foreground_confidence.value == "observed"
    assert event.metadata["hwnd"] == 22


@pytest.mark.asyncio
async def test_semantic_scout_uses_reserve_axis_only_after_sparse_exact_results(
    tmp_path,
) -> None:
    class EmbeddingsStub:
        enabled = True

        def __init__(self) -> None:
            self.documents: list[str] = []

        def accepts_private_text(self) -> bool:
            return True

        async def embed_queries(self, documents):
            self.documents.extend(documents)
            return [[1.0, 0.0] for _ in documents]

    class QdrantStub:
        enabled = True

        def __init__(self) -> None:
            self.axes: list[str] = []

        def accepts_private_data(self) -> bool:
            return True

        async def search(self, *, vector_names, **_kwargs):
            axis = vector_names[0]
            self.axes.append(axis)
            return [
                SimpleNamespace(
                    source="screenpipe_meeting",
                    source_id="meeting:7",
                    title="Ustalenia",
                    content="Ustalono timeline jako oś główną.",
                    score=0.9,
                    created_at=datetime(2026, 9, 19, 12, tzinfo=UTC),
                    metadata={
                        "provenance": {
                            "time": "2026-09-19T12:00:00+00:00",
                        }
                    },
                )
            ]

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    embeddings = EmbeddingsStub()
    qdrant = QdrantStub()
    retriever = TimeFirstRetriever(
        memory=store,
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        min_exact_evidence=2,
    )
    pack = await retriever.retrieve(
        "Co ustaliliśmy wczoraj?",
        now=datetime(2026, 9, 20, 12, tzinfo=UTC),
    )

    assert len(pack.items) == 1
    assert qdrant.axes == ["semantic", "decision"]
    assert pack.items[0].selection_reason == "semantic_scout:semantic"


@pytest.mark.asyncio
async def test_context_eval_measures_recall_abstention_and_axes() -> None:
    records = [
        ContextRetrievalEvalRecordV1(
            example_id="decision",
            query="Co ustaliliśmy?",
            expected_source_ids=("meeting:7",),
            expected_axes=("decision",),
        ),
        ContextRetrievalEvalRecordV1(
            example_id="negative",
            query="Nieistniejąca rozmowa",
        ),
    ]

    async def retrieve(record, _limit):
        if record.example_id == "negative":
            return ContextPackV1(question=record.query)
        item = ContextItemV1(
            source="screenpipe_meeting",
            source_id="meeting:7",
            scope=ContextScope.EPISODIC,
            content="Ustalono timeline.",
            trust=ContextTrust.DERIVED,
            selection_reason="semantic_scout:decision",
        )
        return ContextPackV1(question=record.query, items=(item,))

    scores, metrics = await evaluate_context_retrieval(
        records=records,
        retrieve=retrieve,
        k=5,
    )

    assert scores[0].first_relevant_rank == 1
    assert scores[1].abstention_correct is True
    assert metrics.recall_at_k == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.reserve_axis_coverage == 1.0


@pytest.mark.asyncio
async def test_episode_vectorizer_indexes_only_nonempty_axes(tmp_path) -> None:
    class EmbeddingsStub:
        enabled = True

        def accepts_private_text(self) -> bool:
            return True

        async def embed_documents(self, documents):
            return [[float(index), 1.0] for index, _ in enumerate(documents, start=1)]

    class QdrantStub:
        enabled = True

        def __init__(self) -> None:
            self.upsert = None

        def accepts_private_data(self) -> bool:
            return True

        async def upsert_memory(self, **kwargs):
            self.upsert = kwargs

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = ContextEpisodeV1(
        episode_id=stable_episode_id("timeline", "episode-1"),
        source="timeline",
        source_id="episode-1",
        started_at=datetime(2026, 9, 19, tzinfo=UTC),
        ended_at=datetime(2026, 9, 19, 1, tzinfo=UTC),
        title="VoiceLoop",
        summary="Praca nad kontekstem.",
        source_event_ids=("event-1",),
    )
    await store.upsert_context_episode(episode)
    qdrant = QdrantStub()
    vectorizer = ContextEpisodeVectorizer(
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        memory=store,
    )

    result = await vectorizer.index_episode(
        episode,
        decision="Czas jest główną osią kontekstu.",
    )

    assert result.vector_spaces == ("semantic", "decision")
    assert set(qdrant.upsert["vectors"]) == {"semantic", "decision"}
    stored = await store.get_context_episode(episode.episode_id)
    assert stored is not None
    assert stored.vector_spaces == ("semantic", "decision")
