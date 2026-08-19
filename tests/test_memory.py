from datetime import UTC, datetime, timedelta

import pytest

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
