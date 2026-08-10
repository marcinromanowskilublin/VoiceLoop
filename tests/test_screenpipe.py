from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.behavior_digest import DigestedMemory
from voiceloop.memory import MemoryStore
from voiceloop.screenpipe import ScreenpipeClient, ScreenpipeContext, ScreenpipeTextItem
from voiceloop.screenpipe_memory import ScreenpipeVectorMemoryWorker
from voiceloop.settings import Settings
from voiceloop.tts import WindowsTTS


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
async def test_vector_worker_writes_three_core_qdrant_vectors_for_activity(tmp_path) -> None:
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
                    topic="VoiceLoop",
                    intent="Wdrożenie pamięci",
                    decision="Użyć pięciu named vectors.",
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


@pytest.mark.asyncio
async def test_vector_worker_writes_five_qdrant_vectors_for_meetings(tmp_path) -> None:
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
                    intent="Ustalenie planu",
                    decision="Potwierdzić następne kroki",
                    person_context="Rozmowa z klientem",
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
        "intent",
        "decision",
        "person_context",
    }
    assert call["source"] == "screenpipe_meeting"
