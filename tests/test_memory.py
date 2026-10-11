import os
from datetime import UTC, datetime, timedelta

import pytest

from voiceloop.context.schema import ContextEpisodeV1, ContextEventV1, stable_episode_id
from voiceloop.manual_memory import MANUAL_MEMORY_SOURCE, ManualMemoryService
from voiceloop.memory import MemoryStore
from voiceloop.models import (
    CommandPlan,
    CommandRequest,
    CommandStatus,
    MemoryCreate,
    PlanStep,
    TranscriptEnvelopeV1,
)

pytestmark = pytest.mark.asyncio


async def test_command_lifecycle(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="otwórz kalendarz")
    await store.create_command(request)
    plan = CommandPlan(
        request_id=request.request_id,
        intent="open_calendar",
        response_text="Otwieram.",
        confidence=1,
        steps=[PlanStep(action_id="open_calendar")],
    )

    updated = await store.update_command(
        request.request_id,
        status=CommandStatus.QUEUED,
        plan=plan,
    )

    assert updated is not None
    assert updated.status is CommandStatus.QUEUED
    assert updated.plan == plan
    assert (await store.recent_commands())[0].request_id == request.request_id
    assert [event["status"] for event in await store.command_events(request.request_id)] == [
        "received",
        "queued",
    ]


async def test_command_transcript_is_persisted_with_word_metadata(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    transcript = TranscriptEnvelopeV1.from_text(
        "Otwórz kalendarz",
        confidence=0.94,
        speaker_ids=(0,),
    )
    request = CommandRequest.from_transcript(transcript)

    await store.create_command(request)

    with store._connect() as connection:
        row = connection.execute(
            "SELECT segment_id, transcript_json FROM command_transcripts WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()
    assert row is not None
    assert row["segment_id"] == transcript.segment_id
    assert TranscriptEnvelopeV1.model_validate_json(row["transcript_json"]) == transcript


async def test_new_private_rows_receive_ttl_but_explicit_memories_do_not(tmp_path) -> None:
    store = MemoryStore(
        tmp_path / "voice.db",
        command_transcript_retention_days=30,
        meeting_audio_retention_days=7,
    )
    await store.initialize()
    transcript = TranscriptEnvelopeV1.from_text("Zapisz prywatne polecenie")
    request = CommandRequest.from_transcript(transcript)
    await store.create_command(request)
    await store.add_message("user", "Prywatna rozmowa")
    memory = await store.create_memory(
        MemoryCreate(kind="note", content="Jawna pamięć bez TTL.")
    )

    with store._connect() as connection:
        command = connection.execute(
            "SELECT created_at, expires_at FROM commands WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()
        transcript_row = connection.execute(
            "SELECT expires_at FROM command_transcripts WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()
        conversation = connection.execute(
            "SELECT created_at, expires_at FROM conversation"
        ).fetchone()
        memory_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(memories)").fetchall()
        }

    assert command is not None and transcript_row is not None and conversation is not None
    assert datetime.fromisoformat(command["expires_at"]) == (
        datetime.fromisoformat(command["created_at"]) + timedelta(days=30)
    )
    assert transcript_row["expires_at"] == command["expires_at"]
    assert datetime.fromisoformat(conversation["expires_at"]) == (
        datetime.fromisoformat(conversation["created_at"]) + timedelta(days=30)
    )
    assert "expires_at" not in memory_columns
    assert (await store.list_memories())[0].id == memory.id


async def test_command_ttl_uses_server_acceptance_time(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db", command_transcript_retention_days=30)
    await store.initialize()
    client_time = datetime.now(UTC) + timedelta(days=6)
    request = CommandRequest(text="Polecenie", created_at=client_time)
    before = datetime.now(UTC)

    await store.create_command(request)

    after = datetime.now(UTC)
    with store._connect() as connection:
        row = connection.execute(
            """
            SELECT client_created_at, created_at, expires_at
            FROM commands WHERE request_id = ?
            """,
            (request.request_id,),
        ).fetchone()
    accepted_at = datetime.fromisoformat(row["created_at"])
    assert before <= accepted_at <= after
    assert datetime.fromisoformat(row["client_created_at"]) == client_time
    assert datetime.fromisoformat(row["expires_at"]) == accepted_at + timedelta(days=30)
    assert datetime.fromisoformat(row["expires_at"]) != client_time + timedelta(days=30)


@pytest.mark.parametrize(
    "created_at",
    [
        datetime(2026, 10, 11),
        datetime.now(UTC) - timedelta(days=8),
        datetime.now(UTC) + timedelta(days=8),
    ],
)
async def test_command_rejects_untrusted_client_time(created_at) -> None:
    with pytest.raises(ValueError):
        CommandRequest(text="Polecenie", created_at=created_at)


@pytest.mark.parametrize("request_id", ["tekst prywatny", "x" * 121, "/tmp/secret"])
async def test_command_rejects_content_bearing_request_ids(request_id) -> None:
    with pytest.raises(ValueError):
        CommandRequest(request_id=request_id, text="Polecenie")


async def test_retention_policy_effective_time_survives_restart(tmp_path) -> None:
    database = tmp_path / "voice.db"
    first = MemoryStore(database)
    await first.initialize()
    effective_at = first.retention_policy_effective_at

    second = MemoryStore(database)
    await second.initialize()

    assert second.retention_policy_effective_at == effective_at


async def test_context_upsert_preserves_legacy_null_until_content_changes(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    legacy = ContextEventV1.from_observation(
        source="local_document",
        source_id="legacy",
        started_at=now,
        text="Treść historyczna.",
    )
    await store.upsert_context_event(legacy)
    requested_expiry = now + timedelta(days=14)

    unchanged = await store.upsert_context_event(
        legacy.model_copy(update={"expires_at": requested_expiry})
    )
    changed = await store.upsert_context_event(
        ContextEventV1.from_observation(
            source=legacy.source,
            source_id=legacy.source_id,
            started_at=now,
            text="Nowa treść po wdrożeniu.",
            expires_at=requested_expiry,
        )
    )

    assert unchanged.expires_at is None
    assert changed.expires_at == requested_expiry


async def test_retention_prunes_only_explicitly_expired_new_rows(tmp_path) -> None:
    audio_root = tmp_path / "meetings"
    audio_root.mkdir()
    managed_audio = audio_root / "chunk.wav"
    managed_audio.write_bytes(b"audio")
    external_audio = tmp_path / "external.wav"
    external_audio.write_bytes(b"external")
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    session = await store.create_meeting_session(
        session_id="retention-session",
        started_at=now,
        audio_dir=audio_root,
    )
    await store.save_meeting_audio_file(
        session_id=session.session_id,
        chunk_id="managed",
        channel="input",
        device_name="mic",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        source_path=managed_audio,
        archived_path=managed_audio,
    )
    await store.save_meeting_audio_file(
        session_id=session.session_id,
        chunk_id="external",
        channel="output",
        device_name="speaker",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        source_path=external_audio,
        archived_path=external_audio,
    )
    await store.add_message("user", "Nowy tekst z TTL")
    await store.create_memory(MemoryCreate(kind="note", content="Bez TTL"))
    legacy_created = (now - timedelta(days=365)).isoformat()
    expired = (now - timedelta(seconds=1)).isoformat()
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO conversation(role, content, created_at, expires_at)
            VALUES ('user', 'legacy', ?, NULL)
            """,
            (legacy_created,),
        )
        connection.execute(
            "UPDATE conversation SET expires_at = ? WHERE content = 'Nowy tekst z TTL'",
            (expired,),
        )
        connection.execute(
            "UPDATE meeting_audio_files SET expires_at = ?",
            (expired,),
        )

    preview = await store.preview_legacy_retention_migration(
        grace_days=0,
        preview_started_at=now,
        now=now,
    )
    report = await store.prune_expired_private_data(
        managed_audio_root=audio_root,
        now=now,
    )

    with store._connect() as connection:
        conversations = connection.execute(
            "SELECT content, expires_at FROM conversation"
        ).fetchall()
        audio_count = connection.execute(
            "SELECT COUNT(*) FROM meeting_audio_files"
        ).fetchone()[0]
    assert preview["conversation"] == 1
    assert [(row["content"], row["expires_at"]) for row in conversations] == [
        ("legacy", None)
    ]
    assert audio_count == 0
    assert report["meeting_audio"] == 2
    assert report["audio_files_removed"] == 1
    assert managed_audio.exists() is False
    assert external_audio.exists() is True
    assert [item.content for item in await store.list_memories()] == ["Bez TTL"]


async def test_retention_dry_run_does_not_delete_rows_or_files(tmp_path) -> None:
    audio_root = tmp_path / "meetings"
    audio_path = audio_root / "session" / "audio" / "chunk.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    session = await store.create_meeting_session(
        session_id="dry-run",
        started_at=now,
        audio_dir=audio_path.parent,
    )
    await store.save_meeting_audio_file(
        session_id=session.session_id,
        chunk_id="chunk",
        channel="input",
        device_name="mic",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        source_path=audio_path,
        archived_path=audio_path,
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE meeting_audio_files SET expires_at = ?",
            ((now - timedelta(seconds=1)).isoformat(),),
        )

    report = await store.prune_expired_private_data(
        managed_audio_root=audio_root,
        now=now,
        dry_run=True,
        grace_until=now + timedelta(days=7),
    )

    assert report["dry_run"] == 1
    assert report["meeting_audio"] == 1
    assert audio_path.exists()
    with store._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM meeting_audio_files WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()[0]
            == 1
        )


async def test_retention_rollout_grace_prevents_all_deletion(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    await store.add_message("user", "Chronione przez rollout grace")
    with store._connect() as connection:
        connection.execute(
            "UPDATE conversation SET expires_at = ?",
            ((now - timedelta(seconds=1)).isoformat(),),
        )

    report = await store.prune_expired_private_data(
        managed_audio_root=tmp_path / "meetings",
        now=now,
        grace_until=now + timedelta(days=1),
    )

    assert report == {"grace_active": 1, "dry_run": 0}
    with store._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM conversation WHERE content = ?",
                ("Chronione przez rollout grace",),
            ).fetchone()[0]
            == 1
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
async def test_audio_prune_never_follows_managed_symlink(tmp_path) -> None:
    audio_root = tmp_path / "meetings"
    audio_dir = audio_root / "session" / "audio"
    audio_dir.mkdir(parents=True)
    target = tmp_path / "unrelated-private.txt"
    target.write_text("do not delete", encoding="utf-8")
    link = audio_dir / "chunk.wav"
    link.symlink_to(target)
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    session = await store.create_meeting_session(
        session_id="symlink",
        started_at=now,
        audio_dir=audio_dir,
    )
    await store.save_meeting_audio_file(
        session_id=session.session_id,
        chunk_id="chunk",
        channel="input",
        device_name="mic",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        source_path=link,
        archived_path=link,
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE meeting_audio_files SET expires_at = ?",
            ((now - timedelta(seconds=1)).isoformat(),),
        )

    report = await store.prune_expired_private_data(
        managed_audio_root=audio_root,
        now=now,
    )

    assert target.read_text(encoding="utf-8") == "do not delete"
    assert link.is_symlink()
    assert report["audio_files_removed"] == 0
    assert report["audio_files_failed"] == 1
    with store._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM meeting_audio_files WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()[0]
            == 1
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
async def test_audio_prune_rejects_symlinked_parent_directory(tmp_path) -> None:
    audio_root = tmp_path / "meetings"
    audio_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target = outside_dir / "chunk.wav"
    target.write_bytes(b"do not delete")
    linked_session = audio_root / "session"
    linked_session.symlink_to(outside_dir, target_is_directory=True)
    linked_path = linked_session / "chunk.wav"
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    session = await store.create_meeting_session(
        session_id="parent-symlink",
        started_at=now,
        audio_dir=linked_session,
    )
    await store.save_meeting_audio_file(
        session_id=session.session_id,
        chunk_id="chunk",
        channel="input",
        device_name="mic",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        source_path=linked_path,
        archived_path=linked_path,
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE meeting_audio_files SET expires_at = ?",
            ((now - timedelta(seconds=1)).isoformat(),),
        )

    report = await store.prune_expired_private_data(
        managed_audio_root=audio_root,
        now=now,
    )

    assert target.read_bytes() == b"do not delete"
    assert report["audio_files_failed"] == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
async def test_orphan_reconciliation_is_bounded_and_policy_anchored(tmp_path) -> None:
    audio_root = tmp_path / "meetings"
    audio_dir = audio_root / "session" / "audio"
    audio_dir.mkdir(parents=True)
    orphan = audio_dir / "orphan.wav"
    referenced = audio_dir / "referenced.wav"
    fresh = audio_dir / "fresh.wav"
    target = tmp_path / "outside.wav"
    for path in (orphan, referenced, fresh, target):
        path.write_bytes(path.name.encode())
    moment = datetime.now(UTC)
    old_timestamp = (moment - timedelta(minutes=5)).timestamp()
    os.utime(orphan, (old_timestamp, old_timestamp))
    os.utime(referenced, (old_timestamp, old_timestamp))
    (audio_dir / "linked.wav").symlink_to(target)
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    session = await store.create_meeting_session(
        session_id="session",
        started_at=moment,
        audio_dir=audio_dir,
    )
    await store.save_meeting_audio_file(
        session_id=session.session_id,
        chunk_id="referenced",
        channel="input",
        device_name="mic",
        start_time=moment,
        end_time=moment + timedelta(seconds=1),
        source_path=referenced,
        archived_path=referenced,
    )

    preview = await store.reconcile_orphaned_meeting_audio(
        managed_audio_root=audio_root,
        grace_seconds=60,
        now=moment,
        dry_run=True,
        eligible_after=moment - timedelta(seconds=10),
    )
    report = await store.reconcile_orphaned_meeting_audio(
        managed_audio_root=audio_root,
        grace_seconds=60,
        now=moment,
        eligible_after=moment - timedelta(seconds=10),
    )

    assert preview["orphan_candidates"] == 1
    assert preview["orphan_removed"] == 0
    assert report["orphan_removed"] == 1
    assert orphan.exists() is False
    assert referenced.exists()
    assert fresh.exists()
    assert target.exists()
    assert (audio_dir / "linked.wav").is_symlink()


async def test_retention_deletes_expired_context_copies_and_empty_session(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    expired = now - timedelta(seconds=1)
    event = ContextEventV1.from_observation(
        source="screenpipe_meeting",
        source_id="segment",
        started_at=now - timedelta(days=2),
        text="Prywatny transkrypt.",
        expires_at=expired,
    )
    await store.upsert_context_event(event)
    episode = ContextEpisodeV1(
        episode_id=stable_episode_id("screenpipe_meeting", "meeting"),
        source="screenpipe_meeting",
        source_id="meeting",
        started_at=event.started_at,
        ended_at=event.started_at,
        title="Prywatne spotkanie",
        summary="Prywatny transkrypt.",
        source_event_ids=(event.event_id,),
        expires_at=expired,
    )
    await store.upsert_context_episode(episode)
    await store.create_meeting_session(
        session_id="expired-session",
        started_at=now,
        title="Prywatny tytuł",
        audio_dir=tmp_path / "meetings" / "expired-session" / "audio",
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE meeting_sessions SET expires_at = ? WHERE session_id = ?",
            (expired.isoformat(), "expired-session"),
        )

    report = await store.prune_expired_private_data(
        managed_audio_root=tmp_path / "meetings",
        now=now,
    )

    assert report["context_events"] == 1
    assert report["context_episodes"] == 1
    assert report["meeting_sessions"] == 1
    assert await store.get_context_event(event.event_id) is None
    assert await store.get_context_episode(episode.episode_id) is None
    with store._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM meeting_sessions WHERE session_id = ?",
                ("expired-session",),
            ).fetchone()[0]
            == 0
        )


async def test_meeting_segment_emotions_are_persisted(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    session = await store.create_meeting_session(
        session_id="meeting-emotions",
        started_at=now,
        audio_dir=tmp_path / "audio",
    )
    await store.save_meeting_transcript_segment(
        session_id=session.session_id,
        segment_key="seg-1",
        channel="output",
        speaker_label="Rozmówca",
        speaker_id=0,
        device_name="speaker",
        start_time=now,
        end_time=now + timedelta(seconds=2),
        text="To brzmi spokojnie.",
        transcript=None,
        source="test",
    )

    updated = await store.save_meeting_segment_emotions(
        segment_key="seg-1",
        emotions=[
            {"name": "Calmness", "score": 0.72},
            {"name": "Interest", "score": 0.31},
        ],
    )

    segments = await store.list_meeting_transcript_segments(session.session_id)
    assert updated is True
    assert segments[0].emotions == (
        {"name": "Calmness", "score": 0.72},
        {"name": "Interest", "score": 0.31},
    )


async def test_memory_create_list_delete(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    created = await store.create_memory(
        MemoryCreate(kind="preference", content="Lubię ciemny motyw.")
    )

    items = await store.list_memories(kind="preference")
    deleted = await store.delete_memory(created.id)

    assert [item.content for item in items] == ["Lubię ciemny motyw."]
    assert deleted is True
    assert await store.list_memories() == []


async def test_manual_memory_is_vectorized_locally_and_deleted_consistently(
    tmp_path,
) -> None:
    class EmbeddingsStub:
        enabled = True
        configured_model = "local-test"
        _resolved_model = None

        def __init__(self) -> None:
            self.documents: list[str] = []

        def accepts_private_text(self) -> bool:
            return True

        async def embed_documents(self, documents):
            self.documents = list(documents)
            return [[float(index), 1.0] for index, _ in enumerate(documents, start=1)]

    class QdrantStub:
        enabled = True

        def __init__(self) -> None:
            self.upsert = None
            self.deleted = None

        def accepts_private_data(self) -> bool:
            return True

        async def has_memory(self, **_kwargs):
            return False

        async def upsert_memory(self, **kwargs):
            self.upsert = kwargs

        async def delete_memory(self, **kwargs):
            self.deleted = kwargs

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    embeddings = EmbeddingsStub()
    qdrant = QdrantStub()
    service = ManualMemoryService(
        memory=store,
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    created = await service.create(
        MemoryCreate(
            kind="preference",
            content="Lubię kontakt przez marcinek@example.com.",
        )
    )

    assert qdrant.upsert["source"] == MANUAL_MEMORY_SOURCE
    assert qdrant.upsert["source_id"] == str(created.id)
    assert set(qdrant.upsert["vectors"]) == {
        "semantic",
        "topic",
        "intent",
        "person_context",
    }
    assert any("marcinek@example.com" in document for document in embeddings.documents)
    assert await service.delete(created.id) is True
    assert qdrant.deleted == {
        "source": MANUAL_MEMORY_SOURCE,
        "source_id": str(created.id),
    }
    assert await store.list_memories() == []


async def test_manual_memory_never_embeds_when_endpoint_is_not_private(tmp_path) -> None:
    class RemoteEmbeddingsStub:
        enabled = True

        def accepts_private_text(self) -> bool:
            return False

        async def embed_documents(self, _documents):
            raise AssertionError("private memory must not reach a remote embedding endpoint")

    class QdrantStub:
        enabled = True

        def accepts_private_data(self) -> bool:
            return True

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    service = ManualMemoryService(
        memory=store,
        embeddings=RemoteEmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub(),  # type: ignore[arg-type]
    )

    created = await service.create(
        MemoryCreate(kind="fact", content="To pozostaje tylko lokalnie.")
    )

    assert created.content == "To pozostaje tylko lokalnie."
    assert [item.id for item in await store.list_memories()] == [created.id]


async def test_vector_memory_search(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()

    await store.upsert_vector_memory(
        source="screenpipe_activity",
        source_id="cursor",
        title="Cursor",
        content="Praca nad projektem VoiceLoop w Cursorze.",
        embedding=[1.0, 0.0, 0.0],
        metadata={"app_name": "Cursor"},
    )
    await store.upsert_vector_memory(
        source="screenpipe_activity",
        source_id="browser",
        title="Browser",
        content="Czytanie wiadomości w przeglądarce.",
        embedding=[0.0, 1.0, 0.0],
    )

    hits = await store.search_vector_memories([0.9, 0.1, 0.0], limit=1)

    assert len(hits) == 1
    assert hits[0].source_id == "cursor"
    assert hits[0].metadata["app_name"] == "Cursor"


async def test_vector_memory_metadata_and_expiry_pruning(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    now = datetime.now(UTC)
    await store.upsert_vector_memory(
        source="screenpipe_behavior",
        source_id="expired",
        title="Expired",
        content="Stary wpis.",
        embedding=[1.0, 0.0],
        metadata={
            "content_hash": "old-hash",
            "expires_at": (now - timedelta(seconds=1)).isoformat(),
        },
    )
    await store.upsert_vector_memory(
        source="screenpipe_behavior",
        source_id="fresh",
        title="Fresh",
        content="Świeży wpis.",
        embedding=[0.0, 1.0],
        metadata={
            "content_hash": "fresh-hash",
            "expires_at": (now + timedelta(days=1)).isoformat(),
        },
    )

    metadata = await store.vector_memory_metadata(
        source="screenpipe_behavior",
        source_id="fresh",
    )
    removed = await store.prune_expired_vector_memories(now=now)

    assert metadata is not None
    assert metadata["content_hash"] == "fresh-hash"
    assert removed == 1
    assert await store.has_vector_memory(
        source="screenpipe_behavior",
        source_id="expired",
    ) is False
    assert await store.has_vector_memory(
        source="screenpipe_behavior",
        source_id="fresh",
    ) is True
