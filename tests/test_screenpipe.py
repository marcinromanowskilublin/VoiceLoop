from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.behavior_digest import DigestedMemory
from voiceloop.memory import MemoryStore
from voiceloop.qdrant_memory import QdrantUnavailableError
from voiceloop.screenpipe import ScreenpipeClient, ScreenpipeContext, ScreenpipeTextItem
from voiceloop.screenpipe_memory import (
    ACTIVITY_DUPLICATE_MIN_SCORE,
    ScreenpipeVectorMemoryWorker,
)
from voiceloop.settings import Settings
from voiceloop.tts import WindowsTTS


def _dedup_worker(tmp_path, *, qdrant, embeddings):
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    return ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=ScreenpipeClient(settings),
        memory=MemoryStore(tmp_path / "voice.db"),
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_duplicate_bucket_is_caught_by_content_hash_without_embedding(tmp_path) -> None:
    """Kubełek powtórzony bajt w bajt nie powinien kosztować ani jednego wektora."""

    embed_calls: list[list[str]] = []

    class FakeEmbeddings:
        enabled = True

        async def embed_documents(self, texts):
            embed_calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "has_content_hash": AsyncMock(return_value=True),
            "search": AsyncMock(return_value=[]),
        },
    )()
    worker = _dedup_worker(tmp_path, qdrant=qdrant, embeddings=FakeEmbeddings())

    assert await worker._looks_like_duplicate("Praca nad pamięcią VoiceLoop.") is True
    assert embed_calls == []
    qdrant.search.assert_not_awaited()
    hashed = qdrant.has_content_hash.await_args.kwargs
    assert len(hashed["content_hash"]) == 64
    assert hashed["source"] == "screenpipe_behavior"


@pytest.mark.asyncio
async def test_cheap_stage_does_not_guess_at_paraphrases(tmp_path) -> None:
    """Przed digestem mamy surowy OCR, w bazie podsumowanie modelu.

    To dwa różne rodzaje tekstu, więc żaden próg dla tej pary nie jest
    skalibrowany. Tani etap sprawdza wyłącznie odcisk treści i nie udaje, że
    umie więcej — porównanie semantyczne należy do etapu po digeście.
    """

    class FakeEmbeddings:
        enabled = True

        async def embed_documents(self, texts):
            raise AssertionError("Tani etap nie ma prawa wektoryzować.")

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "has_content_hash": AsyncMock(return_value=False),
            "search": AsyncMock(return_value=[object()]),
        },
    )()
    worker = _dedup_worker(tmp_path, qdrant=qdrant, embeddings=FakeEmbeddings())

    assert await worker._looks_like_duplicate("Zupełnie nowa treść.") is False
    assert await worker._looks_like_duplicate("   ") is False
    qdrant.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_check_does_not_pretend_absence_when_qdrant_is_down(
    tmp_path,
) -> None:
    """Awaria magazynu ≠ „duplikatu nie ma" — indeksowanie musi się zatrzymać."""

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "has_content_hash": AsyncMock(
                side_effect=QdrantUnavailableError("down")
            ),
            "search": AsyncMock(return_value=[]),
        },
    )()
    worker = _dedup_worker(
        tmp_path,
        qdrant=qdrant,
        embeddings=type("E", (), {"enabled": True})(),
    )

    with pytest.raises(QdrantUnavailableError):
        await worker._looks_like_duplicate("Treść, której nie umiemy sprawdzić.")
    qdrant.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_duplicate_propagates_unavailable_instead_of_writing(
    tmp_path,
) -> None:
    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "search": AsyncMock(side_effect=QdrantUnavailableError("down")),
        },
    )()
    worker = _dedup_worker(
        tmp_path,
        qdrant=qdrant,
        embeddings=type("E", (), {"enabled": True})(),
    )

    with pytest.raises(QdrantUnavailableError):
        await worker._semantic_duplicate_exists(
            source="screenpipe_behavior",
            semantic_vector=[1.0, 0.0, 0.0],
        )


@pytest.mark.asyncio
async def test_semantic_duplicate_uses_the_vector_it_is_about_to_store(tmp_path) -> None:
    """Porównanie idzie dokument do dokumentu i nie kosztuje embeddingu.

    Wektor jest ten sam, którym zaraz zapisalibyśmy punkt, więc obie strony leżą
    w identycznej przestrzeni. Dopiero tutaj próg 0.97 ma sens: tekst identyczny
    wraca na 1.000, a dokumenty faktycznie różne mają p99 = 0.874.
    """

    class FakeEmbeddings:
        enabled = True

        async def embed_documents(self, texts):
            raise AssertionError("Ten etap ma używać gotowego wektora.")

    qdrant = type(
        "FakeQdrant",
        (),
        {"enabled": True, "search": AsyncMock(return_value=[object()])},
    )()
    worker = _dedup_worker(tmp_path, qdrant=qdrant, embeddings=FakeEmbeddings())

    vector = [1.0, 0.0, 0.0]
    assert (
        await worker._semantic_duplicate_exists(
            source="screenpipe_behavior",
            semantic_vector=vector,
        )
        is True
    )
    call = qdrant.search.await_args
    assert call.args[0] is vector
    assert call.kwargs["vector_names"] == ("semantic",)
    assert call.kwargs["min_score"] == ACTIVITY_DUPLICATE_MIN_SCORE
    assert call.kwargs["source"] == "screenpipe_behavior"

    qdrant.search.return_value = []
    assert (
        await worker._semantic_duplicate_exists(
            source="screenpipe_behavior",
            semantic_vector=vector,
        )
        is False
    )
    assert (
        await worker._semantic_duplicate_exists(
            source="screenpipe_behavior",
            semantic_vector=[],
        )
        is False
    )


@pytest.mark.asyncio
async def test_related_history_reports_cosine_not_rank_artifact(tmp_path) -> None:
    """Przy jednej osi `fusion_score` jest funkcją rangi, nie podobieństwa.

    Pierwszy wynik dostaje 1.000 niezależnie od tego, jak daleko leży, a ta
    liczba trafiała wprost do kontekstu modelu jako „score".
    """

    class FakeEmbeddings:
        enabled = True

        async def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    hit = type(
        "Hit",
        (),
        {
            "title": "Wcześniejsza aktywność",
            "content": "Praca nad pamięcią.",
            "score": 1.0,
            "vector_scores": {"semantic": 0.61},
        },
    )()
    qdrant = type(
        "FakeQdrant",
        (),
        {"enabled": True, "search": AsyncMock(return_value=[hit])},
    )()
    worker = _dedup_worker(tmp_path, qdrant=qdrant, embeddings=FakeEmbeddings())

    history = await worker._related_history("Praca nad pamięcią.")

    assert history == ["Wcześniejsza aktywność: Praca nad pamięcią. (cosinus=0.610)"]


@pytest.mark.asyncio
async def test_recent_context_uses_latest_window_metadata(monkeypatch, tmp_path) -> None:
    client = ScreenpipeClient(Settings(voiceloop_data_dir=str(tmp_path)))
    search = AsyncMock(
        return_value=[
            {
                "content": {
                    "app_name": "Cursor.exe",
                    "window_name": "VoiceLoop - Cursor",
                    "timestamp": "2026-08-09T21:00:00Z",
                }
            }
        ]
    )
    monkeypatch.setattr(client, "_search", search)

    context = await client.recent_context()

    assert context == ScreenpipeContext(
        app_name="Cursor.exe",
        window_name="VoiceLoop - Cursor",
        timestamp="2026-08-09T21:00:00Z",
    )
    assert search.await_count == 1


@pytest.mark.asyncio
async def test_audio_chunk_combines_absolute_timestamp_with_offsets(
    monkeypatch,
    tmp_path,
) -> None:
    client = ScreenpipeClient(Settings(voiceloop_data_dir=str(tmp_path)))
    search = AsyncMock(
        return_value=[
            {
                "content": {
                    "chunk_id": 10,
                    "file_path": str(tmp_path / "audio.mp4"),
                    "device_name": "Microphone",
                    "device_type": "Input",
                    "timestamp": "2026-08-07T21:38:50+02:00",
                    "start_time": 29.0,
                    "end_time": 30.5,
                    "text": "test",
                }
            }
        ]
    )
    monkeypatch.setattr(client, "_search", search)
    start = datetime(2026, 8, 7, tzinfo=UTC)

    chunks = await client.audio_chunks(start=start, end=start)

    assert chunks[0].chunk_id == "10:29.000000:30.500000"
    assert chunks[0].start_time == "2026-08-07T21:39:19+02:00"
    assert chunks[0].end_time == "2026-08-07T21:39:20.500000+02:00"
    assert chunks[0].start_offset_seconds == 29.0
    assert chunks[0].end_offset_seconds == 30.5


@pytest.mark.asyncio
async def test_active_window_falls_back_to_screenpipe(monkeypatch, tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(settings, MemoryStore(tmp_path / "voice.db"), WindowsTTS())
    monkeypatch.setattr(
        registry,
        "_describe_active_window_sync",
        lambda: (
            "Aktywne okno nie ma dostępnego tytułu ani nazwy procesu.",
            {"window_title": "", "process_name": ""},
        ),
    )
    registry.screenpipe.recent_context = AsyncMock(
        return_value=ScreenpipeContext(
            app_name="Notepad.exe",
            window_name="Bez tytułu — Notatnik",
            timestamp="2026-08-09T21:00:00Z",
        )
    )

    message, data = await registry._describe_active_window({})

    assert "Screenpipe" in message
    assert data["source"] == "screenpipe"
    assert data["process_name"] == "Notepad.exe"


@pytest.mark.asyncio
async def test_youtube_context_search_is_conservative(monkeypatch, tmp_path) -> None:
    client = ScreenpipeClient(Settings(voiceloop_data_dir=str(tmp_path)))
    search = AsyncMock(
        return_value=[
            {
                "content": {
                    "app_name": "chrome.exe",
                    "window_name": "Wykład - YouTube",
                    "browser_url": "https://www.youtube.com/watch?v=abc",
                }
            }
        ]
    )
    monkeypatch.setattr(client, "_search", search)

    now = datetime.now(UTC)
    assert await client.has_youtube_context(start=now, end=now) is True
    assert search.await_args.kwargs["query"] == "youtube"


@pytest.mark.asyncio
async def test_screenpipe_vector_memory_indexes_recent_activity(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path), screenpipe_vector_recent_minutes=30)
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.recent_activity = AsyncMock(
        return_value=[
            ScreenpipeContext(
                app_name="Cursor.exe",
                window_name="VoiceLoop - Cursor",
                timestamp="2026-08-09T21:00:00Z",
            )
        ]
    )

    class FakeEmbeddings:
        enabled = True

        async def embed_texts(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=screenpipe,
        memory=memory,
        embeddings=FakeEmbeddings(),
    )

    indexed = await worker.index_recent_activity()
    hits = await memory.search_vector_memories([1.0, 0.0, 0.0], limit=1)

    assert indexed == 1
    assert hits[0].source == "screenpipe_activity"
    assert "VoiceLoop" in hits[0].content


@pytest.mark.asyncio
async def test_vector_worker_writes_any_nonempty_vector_subset_for_activity(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        behavior_digest_recent_minutes=30,
    )
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.recent_text_activity = AsyncMock(
        return_value=[
            ScreenpipeTextItem(
                app_name="Cursor.exe",
                window_name="VoiceLoop - Cursor",
                timestamp="2026-08-10T00:00:00Z",
                browser_url="",
                text="Implementacja pamięci Qdrant.",
                content_type="OCR",
            )
        ]
    )

    class FakeEmbeddings:
        enabled = True

        async def embed_query(self, text):
            return [1.0, 0.0, 0.0]

        async def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "search": AsyncMock(return_value=[]),
            "upsert_memory": AsyncMock(),
            "has_memory": AsyncMock(return_value=False),
        },
    )()
    digester = type(
        "FakeDigester",
        (),
        {
            "digest": AsyncMock(
                return_value=DigestedMemory(
                    summary="Praca nad pamięcią Qdrant.",
                    topic="",
                    intent="Wdrożenie pamięci",
                    decision="",
                    person_context="Indywidualny kontekst użytkownika.",
                    confidence=0.9,
                )
            )
        },
    )()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=screenpipe,
        memory=memory,
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        digester=digester,  # type: ignore[arg-type]
    )

    indexed = await worker.index_recent_activity()
    call = qdrant.upsert_memory.await_args.kwargs

    assert indexed == 1
    assert set(call["vectors"]) == {"semantic", "intent", "person_context"}
    assert call["source"] == "screenpipe_behavior"
    assert call["metadata"]["vector_profile"] == "dynamic_named_subset_v2"
    assert call["metadata"]["provenance"]["source_id"] == call["source_id"]
    assert call["expires_at"] is not None
    assert call["metadata"]["expires_at"] == call["expires_at"].isoformat()


@pytest.mark.asyncio
async def test_vector_worker_skips_low_confidence_activity_digest(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        behavior_digest_recent_minutes=10,
    )
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.recent_text_activity = AsyncMock(
        return_value=[
            ScreenpipeTextItem(
                app_name="Cursor.exe",
                window_name="VoiceLoop - Cursor",
                timestamp="2026-08-10T00:00:00Z",
                browser_url="",
                text="Minimize\nMaximize\nClose\nPraca nad ważną pamięcią.",
                content_type="OCR",
            )
        ]
    )

    class FakeEmbeddings:
        enabled = True

        async def embed_query(self, text):
            return [1.0, 0.0, 0.0]

        async def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "search": AsyncMock(return_value=[]),
            "upsert_memory": AsyncMock(),
            "has_memory": AsyncMock(return_value=False),
        },
    )()
    digester = type(
        "FakeDigester",
        (),
        {
            "digest": AsyncMock(
                return_value=DigestedMemory(
                    summary="Surowy OCR bez pewnej analizy.",
                    topic="Niski confidence",
                    intent="Nie zapisuj do Qdrant",
                    confidence=0.25,
                )
            )
        },
    )()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=screenpipe,
        memory=memory,
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        digester=digester,  # type: ignore[arg-type]
    )

    indexed = await worker.index_recent_activity()

    assert indexed == 0
    qdrant.upsert_memory.assert_not_awaited()


def test_screenpipe_memory_cleans_ocr_chrome() -> None:
    cleaned = ScreenpipeVectorMemoryWorker._clean_ocr_text(
        "Minimize\nMaximize\nClose\nFile\nEdit\nRealna treść projektu\n"
        "vscode-file://ignored\nRealna treść projektu\n"
    )

    assert "Minimize" not in cleaned
    assert "vscode-file" not in cleaned
    assert cleaned == "Realna treść projektu"


@pytest.mark.asyncio
async def test_vector_worker_writes_any_nonempty_vector_subset_for_meetings(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        behavior_digest_recent_minutes=30,
    )
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.recent_text_activity = AsyncMock(return_value=[])

    await memory.save_screenpipe_transcript(
        chunk_id="meeting-chunk-1",
        meeting_id=42,
        device_name="Mic",
        device_type="Input",
        start_time="2026-08-10T05:00:00Z",
        end_time="2026-08-10T05:01:00Z",
        text="Ustalamy plan rozmowy z klientem i następne kroki.",
        source="deepgram",
    )

    class FakeEmbeddings:
        enabled = True

        async def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "upsert_memory": AsyncMock(),
            "has_memory": AsyncMock(return_value=False),
        },
    )()
    digester = type(
        "FakeDigester",
        (),
        {
            "digest": AsyncMock(
                return_value=DigestedMemory(
                    summary="Rozmowa o planie klienta.",
                    topic="Spotkanie klient",
                    intent="",
                    decision="Potwierdzić następne kroki",
                    person_context="",
                    confidence=0.9,
                )
            )
        },
    )()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=screenpipe,
        memory=memory,
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        digester=digester,  # type: ignore[arg-type]
    )

    indexed = await worker.index_recent_activity()
    call = qdrant.upsert_memory.await_args.kwargs

    assert indexed == 1
    assert set(call["vectors"]) == {
        "semantic",
        "topic",
        "decision",
    }
    assert call["source"] == "screenpipe_meeting"
    content_hash = call["content_hash"]
    assert len(content_hash) == 64
    assert call["metadata"]["content_hash"] == content_hash
    assert qdrant.has_memory.await_args.kwargs["content_hash"] == content_hash
    assert call["expires_at"] is not None


@pytest.mark.asyncio
async def test_vector_worker_redacts_before_digest_and_embedding(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        behavior_digest_recent_minutes=10,
    )
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    screenpipe = ScreenpipeClient(settings)
    screenpipe.recent_text_activity = AsyncMock(
        return_value=[
            ScreenpipeTextItem(
                app_name="Cursor.exe",
                window_name="Sekrety - Cursor",
                timestamp="2026-08-10T00:00:00Z",
                browser_url="",
                text="api_key=verysecretvalue kontakt anna@example.com",
                content_type="OCR",
            )
        ]
    )
    embedded_documents: list[str] = []

    class FakeEmbeddings:
        enabled = True
        configured_model = "nomic-test"

        async def embed_query(self, text):
            return [1.0, 0.0, 0.0]

        async def embed_documents(self, texts):
            embedded_documents.extend(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "search": AsyncMock(return_value=[]),
            "upsert_memory": AsyncMock(),
            "has_memory": AsyncMock(return_value=False),
        },
    )()
    digester = type(
        "FakeDigester",
        (),
        {
            "configured_model": "qwen-test",
            "digest": AsyncMock(
                return_value=DigestedMemory(
                    summary="api_key=verysecretvalue kontakt anna@example.com",
                    observations=["anna@example.com"],
                    confidence=0.9,
                )
            ),
        },
    )()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=screenpipe,
        memory=memory,
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        digester=digester,  # type: ignore[arg-type]
    )

    assert await worker.index_recent_activity() == 1

    digest_call = digester.digest.await_args.kwargs
    stored = qdrant.upsert_memory.await_args.kwargs
    serialized_documents = "\n".join(embedded_documents)
    assert "verysecretvalue" not in digest_call["content"]
    assert "anna@example.com" not in digest_call["content"]
    assert "[SECRET]" in digest_call["content"]
    assert "verysecretvalue" not in serialized_documents
    assert "anna@example.com" not in serialized_documents
    assert "verysecretvalue" not in stored["content"]
    assert {"secret", "email"} <= set(stored["metadata"]["privacy_redactions"])
    assert stored["metadata"]["provenance"]["model"] == "qwen-test"
    assert stored["metadata"]["provenance"]["schema"] == "memory-documents-v2"


@pytest.mark.asyncio
async def test_meeting_is_reindexed_only_after_content_hash_changes(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    await memory.save_screenpipe_transcript(
        chunk_id="chunk-1",
        meeting_id=7,
        device_name="Mic",
        device_type="Input",
        start_time="2026-08-10T05:00:00Z",
        end_time="2026-08-10T05:01:00Z",
        text="Pierwsza część rozmowy.",
        source="deepgram",
    )

    class FakeEmbeddings:
        enabled = True

        async def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    class FakeQdrant:
        enabled = True

        def __init__(self) -> None:
            self.hashes: dict[tuple[str, str], str] = {}
            self.upserts: list[dict[str, object]] = []

        async def has_memory(self, *, source, source_id, content_hash=None):
            return self.hashes.get((source, source_id)) == content_hash

        async def upsert_memory(self, **kwargs):
            key = (str(kwargs["source"]), str(kwargs["source_id"]))
            self.hashes[key] = str(kwargs["content_hash"])
            self.upserts.append(kwargs)

    qdrant = FakeQdrant()
    digester = type(
        "FakeDigester",
        (),
        {
            "digest": AsyncMock(
                return_value=DigestedMemory(
                    summary="Digest spotkania.",
                    decision="Kontynuować.",
                    confidence=0.9,
                )
            )
        },
    )()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=ScreenpipeClient(settings),
        memory=memory,
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        digester=digester,  # type: ignore[arg-type]
    )

    first = await worker._index_meeting_transcripts()
    unchanged = await worker._index_meeting_transcripts()
    await memory.save_screenpipe_transcript(
        chunk_id="chunk-2",
        meeting_id=7,
        device_name="Mic",
        device_type="Input",
        start_time="2026-08-10T05:01:00Z",
        end_time="2026-08-10T05:02:00Z",
        text="Nowa część rozmowy zmienia hash.",
        source="deepgram",
    )
    changed = await worker._index_meeting_transcripts()

    assert (first, unchanged, changed) == (1, 0, 1)
    assert len(qdrant.upserts) == 2
    assert qdrant.upserts[0]["content_hash"] != qdrant.upserts[1]["content_hash"]


@pytest.mark.asyncio
async def test_legacy_migration_writes_semantic_vector_only(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    await memory.upsert_vector_memory(
        source="screenpipe_activity",
        source_id="legacy-1",
        title="Starsza pamięć",
        content="Starszy wpis.",
        embedding=[1.0, 0.0, 0.0],
        metadata={},
    )
    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "has_memory": AsyncMock(return_value=False),
            "upsert_memory": AsyncMock(),
        },
    )()
    embeddings = type("FakeEmbeddings", (), {"enabled": True})()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=ScreenpipeClient(settings),
        memory=memory,
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    assert await worker.migrate_legacy_memories() == 1

    call = qdrant.upsert_memory.await_args.kwargs
    assert set(call["vectors"]) == {"semantic"}
    assert call["metadata"]["vector_profile"] == "legacy_semantic_only_v2"
    assert call["ttl_seconds"] == 14 * 86400
    assert (
        await memory.get_state("qdrant_legacy_migration_v2_semantic_only")
        == "done"
    )


@pytest.mark.asyncio
async def test_vector_worker_prunes_only_when_explicitly_enabled(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        vector_memory_prune_enabled=True,
    )
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    await memory.upsert_vector_memory(
        source="screenpipe_behavior",
        source_id="expired",
        title="Expired",
        content="Stary wpis.",
        embedding=[1.0, 0.0],
        metadata={"expires_at": "2020-01-01T00:00:00+00:00"},
    )
    qdrant = type(
        "FakeQdrant",
        (),
        {
            "enabled": True,
            "prune_expired": AsyncMock(return_value=2),
        },
    )()
    worker = ScreenpipeVectorMemoryWorker(
        settings=settings,
        screenpipe=ScreenpipeClient(settings),
        memory=memory,
        embeddings=type("FakeEmbeddings", (), {"enabled": True})(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    removed = await worker.prune_expired_memories(force=True)

    assert removed == 3
    qdrant.prune_expired.assert_awaited_once_with(dry_run=False)
    assert await memory.has_vector_memory(
        source="screenpipe_behavior",
        source_id="expired",
    ) is False
