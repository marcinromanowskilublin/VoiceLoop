from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voiceloop.context.schema import (
    ContextEpisodeV1,
    ContextEventV1,
    ContextIndexOperation,
    ContextIndexStatus,
    stable_episode_id,
)
from voiceloop.context.vectorization import ContextEpisodeIndexer
from voiceloop.memory import MemoryStore

NOW = datetime(2026, 11, 16, 20, tzinfo=UTC)


class EmbeddingsStub:
    enabled = True

    def __init__(self, model: str = "embedding-a") -> None:
        self.configured_model = model

    def accepts_private_text(self) -> bool:
        return True

    async def resolve_model(self) -> str:
        return self.configured_model

    async def embed_documents(self, documents):
        return [[1.0, float(index)] for index, _ in enumerate(documents)]


class QdrantStub:
    enabled = True
    collection_name = "voiceloop_context_episodes_v1"

    def __init__(self) -> None:
        self.points: dict[tuple[str, str], tuple[str, str]] = {}
        self.upsert_count = 0
        self.deleted: list[tuple[str, str]] = []
        self.fail_upsert = False

    def accepts_private_data(self) -> bool:
        return True

    async def collection_exists(self) -> bool:
        return bool(self.points)

    async def has_memory(
        self,
        *,
        source,
        source_id,
        content_hash,
        metadata_match,
    ):
        return self.points.get((source, source_id)) == (
            content_hash,
            metadata_match["vector_index_hash"],
        )

    async def upsert_memory(
        self,
        *,
        source,
        source_id,
        content_hash,
        metadata,
        **_kwargs,
    ):
        if self.fail_upsert:
            raise RuntimeError("synthetic qdrant outage")
        self.points[(source, source_id)] = (
            content_hash,
            metadata["vector_index_hash"],
        )
        self.upsert_count += 1

    async def delete_memory(self, *, source: str, source_id: str) -> None:
        self.points.pop((source, source_id), None)
        self.deleted.append((source, source_id))


async def seeded_episode(
    store: MemoryStore,
    *,
    summary: str = "Kanoniczne ustalenie.",
    vectorize: bool | None = None,
) -> ContextEpisodeV1:
    event = ContextEventV1.from_observation(
        source="test",
        source_id="event-1",
        started_at=NOW,
        text="Kanoniczne źródło.",
    )
    await store.upsert_context_event(event)
    metadata = {}
    if vectorize is not None:
        metadata["vectorize"] = vectorize
    episode = ContextEpisodeV1(
        episode_id=stable_episode_id("test", "episode-1"),
        source="test",
        source_id="episode-1",
        started_at=NOW,
        ended_at=NOW,
        summary=summary,
        source_event_ids=(event.event_id,),
        metadata=metadata,
    )
    return await store.upsert_context_episode(episode)


@pytest.mark.asyncio
async def test_episode_upsert_transactionally_requests_index(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()

    episode = await seeded_episode(store)
    state = await store.get_context_index_state(episode.episode_id)
    jobs = await store.list_context_index_jobs()

    assert state is not None
    assert state.desired_operation is ContextIndexOperation.UPSERT
    assert state.desired_revision == episode.content_hash
    assert state.applied_revision is None
    assert state.status is ContextIndexStatus.PENDING
    assert len(jobs) == 1
    assert jobs[0].episode_id == episode.episode_id
    assert jobs[0].desired_revision == episode.content_hash


@pytest.mark.asyncio
async def test_vectorize_false_is_explicitly_not_requested(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()

    episode = await seeded_episode(store, vectorize=False)
    state = await store.get_context_index_state(episode.episode_id)

    assert state is not None
    assert state.status is ContextIndexStatus.NOT_REQUESTED
    assert await store.list_context_index_jobs() == []


@pytest.mark.asyncio
async def test_successful_index_applies_matching_revision_and_profile(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)
    qdrant = QdrantStub()
    indexer = ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub("embedding-a"),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    )

    report = await indexer.index(dry_run=False)
    state = await store.get_context_index_state(episode.episode_id)
    stored = await store.get_context_episode(episode.episode_id)
    applied_jobs = await store.list_context_index_jobs(
        statuses=(ContextIndexStatus.APPLIED,),
    )

    assert report.indexed == 1
    assert report.outbox_pending == 0
    assert state is not None
    assert state.status is ContextIndexStatus.APPLIED
    assert state.applied_revision == state.desired_revision
    assert state.index_fingerprint
    assert state.index_profile and state.index_profile.endswith(":embedding-a")
    assert state.vector_spaces == ("semantic",)
    assert stored is not None and stored.vector_spaces == ("semantic",)
    assert len(applied_jobs) == 1


@pytest.mark.asyncio
async def test_index_failure_is_visible_and_retryable(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)
    qdrant = QdrantStub()
    qdrant.fail_upsert = True
    indexer = ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    )

    with pytest.raises(RuntimeError, match="synthetic qdrant outage"):
        await indexer.index(dry_run=False)
    failed = await store.get_context_index_state(episode.episode_id)

    assert failed is not None
    assert failed.status is ContextIndexStatus.FAILED
    assert failed.attempts == 1
    assert failed.last_error_type == "RuntimeError"

    qdrant.fail_upsert = False
    report = await indexer.index(dry_run=False)
    recovered = await store.get_context_index_state(episode.episode_id)

    assert report.indexed == 1
    assert recovered is not None
    assert recovered.status is ContextIndexStatus.APPLIED
    assert recovered.applied_revision == recovered.desired_revision


@pytest.mark.asyncio
async def test_stale_completion_cannot_overwrite_new_desired_revision(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    first = await seeded_episode(store, summary="Pierwsza wersja.")
    second = ContextEpisodeV1(
        episode_id=first.episode_id,
        source=first.source,
        source_id=first.source_id,
        started_at=first.started_at,
        ended_at=first.ended_at,
        summary="Druga wersja.",
        source_event_ids=first.source_event_ids,
        updated_at=NOW,
    )
    second = await store.upsert_context_episode(second)

    applied = await store.mark_context_index_applied(
        episode_id=first.episode_id,
        operation=ContextIndexOperation.UPSERT,
        desired_revision=first.content_hash,
        index_fingerprint="a" * 64,
        index_profile="test-profile",
        vector_spaces=("semantic",),
    )
    state = await store.get_context_index_state(first.episode_id)
    stale_jobs = await store.list_context_index_jobs(
        statuses=(ContextIndexStatus.SUPERSEDED,),
    )

    assert applied is False
    assert state is not None
    assert state.desired_revision == second.content_hash
    assert state.status is ContextIndexStatus.PENDING
    assert any(job.desired_revision == first.content_hash for job in stale_jobs)


@pytest.mark.asyncio
async def test_embedding_profile_change_forces_reindex(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    await seeded_episode(store)
    qdrant = QdrantStub()

    first = ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub("embedding-a"),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    )
    second = ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub("embedding-b"),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    )

    await first.index(dry_run=False)
    report = await second.index(dry_run=False)

    assert qdrant.upsert_count == 2
    assert report.indexed == 1


@pytest.mark.asyncio
async def test_tombstone_requests_delete_and_indexer_applies_it(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)
    qdrant = QdrantStub()
    indexer = ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    )
    await indexer.index(dry_run=False)

    assert await store.tombstone_context_episode(episode.episode_id) is True
    pending = await store.get_context_index_state(episode.episode_id)
    report = await indexer.index(dry_run=False)
    applied = await store.get_context_index_state(episode.episode_id)

    assert pending is not None
    assert pending.desired_operation is ContextIndexOperation.DELETE
    assert pending.status is ContextIndexStatus.PENDING
    assert report.deleted == 1
    assert applied is not None
    assert applied.desired_operation is ContextIndexOperation.DELETE
    assert applied.status is ContextIndexStatus.APPLIED
    assert qdrant.deleted == [(episode.source, episode.source_id)]
