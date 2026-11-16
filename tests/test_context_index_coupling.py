from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

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
        self.points: dict[tuple[str, str], tuple[str, str, int]] = {}
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
        point = self.points.get((source, source_id))
        return point is not None and point[:2] == (
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
            metadata["index_generation"],
        )
        self.upsert_count += 1

    async def delete_memory(self, *, source: str, source_id: str) -> None:
        self.points.pop((source, source_id), None)
        self.deleted.append((source, source_id))

    async def delete_memory_if_matches(
        self,
        *,
        source: str,
        source_id: str,
        content_hash: str,
        index_fingerprint: str,
        index_generation: int,
    ) -> bool:
        point = self.points.get((source, source_id))
        if point is None or point[0] != content_hash:
            return False
        if index_fingerprint and point[1] != index_fingerprint:
            return False
        if index_generation is not None and point[2] != index_generation:
            return False
        self.points.pop((source, source_id), None)
        self.deleted.append((source, source_id))
        return True


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
    pending = await store.get_context_index_state(episode.episode_id)
    jobs = await store.list_context_index_jobs()
    qdrant = QdrantStub()
    report = await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)
    state = await store.get_context_index_state(episode.episode_id)

    assert pending is not None
    assert pending.desired_operation is ContextIndexOperation.DELETE
    assert pending.status is ContextIndexStatus.PENDING
    assert len(jobs) == 1
    assert jobs[0].operation is ContextIndexOperation.DELETE
    assert report.deleted == 1
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

    failed_report = await indexer.index(dry_run=False)
    failed = await store.get_context_index_state(episode.episode_id)

    assert failed_report.failed == 1
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
    first_state = await store.get_context_index_state(first.episode_id)
    assert first_state is not None
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
        generation=first_state.desired_generation,
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


@pytest.mark.asyncio
async def test_initialize_backfills_existing_episode_without_index_state(
    tmp_path,
) -> None:
    database = tmp_path / "voice.db"
    store = MemoryStore(database)
    await store.initialize()
    episode = await seeded_episode(store)
    with store._connect() as connection:
        connection.execute(
            "DELETE FROM context_index_outbox WHERE episode_id = ?",
            (episode.episode_id,),
        )
        connection.execute(
            "DELETE FROM context_index_state WHERE episode_id = ?",
            (episode.episode_id,),
        )

    reloaded = MemoryStore(database)
    await reloaded.initialize()
    state = await reloaded.get_context_index_state(episode.episode_id)
    jobs = await reloaded.list_context_index_jobs()

    assert state is not None
    assert state.status is ContextIndexStatus.PENDING
    assert len(jobs) == 1
    assert jobs[0].source == episode.source
    assert jobs[0].source_id == episode.source_id


@pytest.mark.asyncio
async def test_same_revision_jobs_reactivate_across_restore_delete_cycles(
    tmp_path,
) -> None:
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

    for _ in range(2):
        assert await store.tombstone_context_episode(episode.episode_id) is True
        delete_report = await indexer.index(dry_run=False)
        assert delete_report.deleted == 1
        restored = episode.model_copy(
            update={
                "deleted_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        await store.upsert_context_episode(restored)
        upsert_report = await indexer.index(dry_run=False)
        assert upsert_report.indexed == 1

    assert (await store.context_index_summary())["outbox_ready"] == 0
    assert qdrant.deleted == [
        (episode.source, episode.source_id),
        (episode.source, episode.source_id),
    ]


@pytest.mark.asyncio
async def test_vectorize_opt_out_supersedes_pending_upsert(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)

    opted_out = episode.model_copy(
        update={
            "metadata": {"vectorize": False},
            "updated_at": datetime.now(UTC),
        }
    )
    await store.upsert_context_episode(opted_out)
    pending = await store.get_context_index_state(episode.episode_id)
    ready = await store.list_context_index_jobs()
    superseded = await store.list_context_index_jobs(
        statuses=(ContextIndexStatus.SUPERSEDED,),
    )
    qdrant = QdrantStub()
    await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)
    state = await store.get_context_index_state(episode.episode_id)

    assert pending is not None
    assert pending.desired_operation is ContextIndexOperation.DELETE
    assert pending.status is ContextIndexStatus.PENDING
    assert state is not None
    assert state.status is ContextIndexStatus.NOT_REQUESTED
    assert len(ready) == 1
    assert ready[0].operation is ContextIndexOperation.DELETE
    assert len(superseded) == 1


@pytest.mark.asyncio
async def test_hard_delete_preserves_durable_qdrant_delete_job(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)
    expired = NOW - timedelta(seconds=1)
    with store._connect() as connection:
        connection.execute(
            "UPDATE context_episodes SET expires_at = ? WHERE episode_id = ?",
            (expired.isoformat(), episode.episode_id),
        )

    await store.prune_expired_private_data(
        managed_audio_root=tmp_path / "meetings",
        now=NOW,
    )
    jobs = await store.list_context_index_jobs(
        operation=ContextIndexOperation.DELETE,
    )

    assert await store.get_context_episode(episode.episode_id) is None
    assert await store.get_context_index_state(episode.episode_id) is None
    assert len(jobs) == 1
    assert jobs[0].source == episode.source
    assert jobs[0].source_id == episode.source_id

    qdrant = QdrantStub()
    qdrant.points[(episode.source, episode.source_id)] = (
        episode.content_hash,
        "old-fingerprint",
        1,
    )
    report = await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)

    assert report.deleted == 1
    assert (episode.source, episode.source_id) not in qdrant.points
    assert (await store.context_index_summary())["outbox_ready"] == 0


@pytest.mark.asyncio
async def test_invalid_episode_does_not_starve_later_index_jobs(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    invalid = ContextEpisodeV1(
        episode_id="episode-0-invalid",
        source="test",
        source_id="invalid",
        started_at=NOW,
        ended_at=NOW,
        summary="Brakujące źródło.",
        source_event_ids=("missing-event",),
    )
    await store.upsert_context_episode(invalid)
    valid = await seeded_episode(store)
    qdrant = QdrantStub()

    report = await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)
    invalid_state = await store.get_context_index_state(invalid.episode_id)
    valid_state = await store.get_context_index_state(valid.episode_id)

    assert report.failed == 1
    assert report.indexed == 1
    assert invalid_state is not None
    assert invalid_state.status is ContextIndexStatus.FAILED
    assert valid_state is not None
    assert valid_state.status is ContextIndexStatus.APPLIED


@pytest.mark.asyncio
async def test_tombstone_during_remote_upsert_compensates_stale_point(
    tmp_path,
) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)

    class RacingQdrant(QdrantStub):
        async def upsert_memory(self, **kwargs):
            await super().upsert_memory(**kwargs)
            await store.tombstone_context_episode(episode.episode_id)

    qdrant = RacingQdrant()
    report = await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)
    state = await store.get_context_index_state(episode.episode_id)

    assert report.indexed == 0
    assert qdrant.points == {}
    assert state is not None
    assert state.desired_operation is ContextIndexOperation.DELETE
    assert state.status is ContextIndexStatus.PENDING


@pytest.mark.asyncio
async def test_migration_rebuilds_legacy_outbox_without_cascade_fk(
    tmp_path,
) -> None:
    database = tmp_path / "voice.db"
    store = MemoryStore(database)
    await store.initialize()
    episode = await seeded_episode(store)

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP INDEX idx_context_index_outbox_ready")
        connection.execute(
            "ALTER TABLE context_index_outbox RENAME TO context_index_outbox_v2"
        )
        connection.execute(
            """
            CREATE TABLE context_index_outbox (
                job_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL DEFAULT 1,
                episode_id TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                desired_revision TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at TEXT NOT NULL,
                claimed_at TEXT,
                last_error_type TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(episode_id, operation, desired_revision),
                FOREIGN KEY(episode_id) REFERENCES context_episodes(episode_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO context_index_outbox
            SELECT job_id, schema_version, episode_id, source, source_id,
                   operation, desired_revision, status, attempts,
                   available_at, claimed_at, last_error_type,
                   created_at, updated_at
            FROM context_index_outbox_v2
            """
        )
        connection.execute("DROP TABLE context_index_outbox_v2")
        connection.commit()
    finally:
        connection.close()

    migrated = MemoryStore(database)
    await migrated.initialize()
    with migrated._connect() as connection:
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(context_index_outbox)"
        ).fetchall()

    assert foreign_keys == []
    assert await migrated.tombstone_context_episode(episode.episode_id)
    with migrated._connect() as connection:
        connection.execute(
            "DELETE FROM context_episodes WHERE episode_id = ?",
            (episode.episode_id,),
        )
    jobs = await migrated.list_context_index_jobs(
        operation=ContextIndexOperation.DELETE,
    )
    assert len(jobs) == 1
    assert jobs[0].source == episode.source


@pytest.mark.asyncio
async def test_opt_out_deletes_crash_window_remote_point(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)
    state = await store.get_context_index_state(episode.episode_id)
    assert state is not None
    qdrant = QdrantStub()
    qdrant.points[(episode.source, episode.source_id)] = (
        episode.content_hash,
        "remote-before-sql-apply",
        state.desired_generation,
    )

    await store.upsert_context_episode(
        episode.model_copy(
            update={
                "metadata": {"vectorize": False},
                "updated_at": datetime.now(UTC),
            }
        )
    )
    report = await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)

    assert report.deleted == 1
    assert qdrant.points == {}
    final = await store.get_context_index_state(episode.episode_id)
    assert final is not None
    assert final.status is ContextIndexStatus.NOT_REQUESTED


@pytest.mark.asyncio
async def test_late_delete_cannot_remove_restored_generation(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)

    class RestoreDuringDelete(QdrantStub):
        restored = False

        async def delete_memory_if_matches(self, **kwargs):
            if not self.restored:
                self.restored = True
                await store.upsert_context_episode(
                    episode.model_copy(
                        update={
                            "deleted_at": None,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
                await ContextEpisodeIndexer(
                    memory=store,
                    embeddings=EmbeddingsStub("restored-model"),  # type: ignore[arg-type]
                    qdrant=self,  # type: ignore[arg-type]
                    primary_collection="voiceloop_memory",
                ).index(dry_run=False)
            return await super().delete_memory_if_matches(**kwargs)

    qdrant = RestoreDuringDelete()
    initial = ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub("initial-model"),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    )
    await initial.index(dry_run=False)
    assert await store.tombstone_context_episode(episode.episode_id)

    await initial.index(dry_run=False)
    state = await store.get_context_index_state(episode.episode_id)
    point = qdrant.points[(episode.source, episode.source_id)]

    assert state is not None
    assert state.desired_operation is ContextIndexOperation.UPSERT
    assert state.status is ContextIndexStatus.APPLIED
    assert point[2] == state.applied_generation
    assert point[1] == state.index_fingerprint


@pytest.mark.asyncio
async def test_concurrent_embedding_profiles_use_single_claim(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode = await seeded_episode(store)

    class BlockingQdrant(QdrantStub):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_memory(self, **kwargs):
            if kwargs["metadata"]["index_profile"].endswith(":embedding-a"):
                self.started.set()
                await self.release.wait()
            await super().upsert_memory(**kwargs)

    qdrant = BlockingQdrant()
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

    first_task = asyncio.create_task(first.index(dry_run=False))
    await qdrant.started.wait()
    second_report = await second.index(dry_run=False)
    qdrant.release.set()
    first_report = await first_task
    state = await store.get_context_index_state(episode.episode_id)
    point = qdrant.points[(episode.source, episode.source_id)]

    assert first_report.indexed == 1
    assert second_report.indexed == 0
    assert state is not None
    assert state.status is ContextIndexStatus.APPLIED
    assert state.index_profile and state.index_profile.endswith(":embedding-a")
    assert point[1] == state.index_fingerprint
    assert point[2] == state.applied_generation


@pytest.mark.asyncio
async def test_terminal_outbox_jobs_expire_and_are_pruned(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    await seeded_episode(store)
    qdrant = QdrantStub()
    await ContextEpisodeIndexer(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        primary_collection="voiceloop_memory",
    ).index(dry_run=False)
    jobs = await store.list_context_index_jobs(
        statuses=(ContextIndexStatus.APPLIED,),
    )

    assert len(jobs) == 1
    assert jobs[0].expires_at is not None
    cutoff = jobs[0].expires_at + timedelta(seconds=1)
    assert await store.prune_context_index_outbox(now=cutoff) == 1
    assert (
        await store.prune_context_index_outbox(
            now=cutoff,
            dry_run=False,
        )
        == 1
    )
    assert (
        await store.list_context_index_jobs(
            statuses=(ContextIndexStatus.APPLIED,),
        )
        == []
    )
