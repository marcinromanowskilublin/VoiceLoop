from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from voiceloop.context import (
    ContextAssembler,
    ContextEntityResolver,
    ContextEventV1,
    ContextScope,
    ContextTrust,
    ForegroundConfidence,
    TimeFirstRetriever,
    build_time_first_query,
    group_events_into_episodes,
    stable_context_id,
)
from voiceloop.context.ingest import ScreenpipeTimelineIngestor
from voiceloop.corpus.schema import ProperNameEntryV1, ProperNameLexiconV1
from voiceloop.memory import MemoryStore
from voiceloop.screenpipe import ScreenpipeClient
from voiceloop.settings import Settings


def test_context_event_has_stable_id_hash_and_foreground_provenance() -> None:
    event = ContextEventV1.from_observation(
        source="screenpipe",
        source_id="frame-7",
        started_at=datetime(2026, 9, 19, 8, tzinfo=UTC),
        app_name="Cursor",
        window_title="VoiceLoop",
        text="Praca nad timeline.",
    )

    assert event.event_id == stable_context_id("screenpipe", "frame-7")
    assert len(event.content_hash) == 64
    assert event.foreground_confidence is ForegroundConfidence.UNKNOWN

    with pytest.raises(ValidationError, match="foreground state"):
        event.model_copy(
            update={
                "is_foreground": True,
                "foreground_confidence": ForegroundConfidence.UNKNOWN,
            }
        ).model_validate(
            {
                **event.model_dump(),
                "is_foreground": True,
                "foreground_confidence": "unknown",
            }
        )


def test_context_assembler_keeps_top_vector_and_deduplicates() -> None:
    assembler = ContextAssembler()
    pack = assembler.assemble(
        question="Co ustaliliśmy?",
        session_id="session-1",
        vector_contexts=["najlepsze trafienie", "drugie trafienie"],
        manual_contexts=["jawna pamięć", "najlepsze trafienie"],
        action_summaries=["ostatnia akcja"],
        notices=["brak świeżych źródeł"],
    )

    rendered = pack.memory_strings(limit=20)
    assert rendered[0].startswith("Local context data (source=system_notice")
    assert "najlepsze trafienie" in rendered[1]
    assert sum("najlepsze trafienie" in item for item in rendered) == 1
    assert pack.sources == {
        "system_notice": 1,
        "vector_memory": 2,
        "manual_memory": 1,
        "recent_action": 1,
    }


def test_entity_resolver_uses_only_approved_aliases() -> None:
    lexicon = ProperNameLexiconV1(
        entries=(
            ProperNameEntryV1(
                canonical="Jan Kowalski",
                aliases=("Jan", "Janowi"),
                category="person",
                approved=True,
            ),
            ProperNameEntryV1(
                canonical="Niepewna osoba",
                aliases=("Niepewnej osobie",),
                category="person",
                approved=False,
            ),
        )
    )
    resolver = ContextEntityResolver(lexicon)

    resolved = resolver.resolve("Janowi")
    assert resolved is not None
    assert resolved.canonical == "Jan Kowalski"
    assert resolved.kind == "person"
    assert resolver.resolve("Niepewnej osobie") is None


@pytest.mark.asyncio
async def test_context_event_store_fts_upsert_and_tombstone(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    assert store.context_fts_enabled is True
    first = ContextEventV1.from_observation(
        source="screenpipe",
        source_id="frame-1",
        started_at=datetime(2026, 9, 19, 10, tzinfo=UTC),
        app_name="Cursor",
        window_title="VoiceLoop - Cursor",
        text="Projekt VoiceLoop i warstwa kontekstu.",
    )
    stored = await store.upsert_context_event(first)

    hits = await store.search_context_events(
        query="VoiceLoop",
        start=datetime(2026, 9, 19, tzinfo=UTC),
        end=datetime(2026, 9, 20, tzinfo=UTC),
    )
    assert [item.event_id for item in hits] == [stored.event_id]

    changed = ContextEventV1.from_observation(
        source="screenpipe",
        source_id="frame-1",
        started_at=first.started_at,
        app_name="Cursor",
        window_title="VoiceLoop - Cursor",
        text="Projekt VoiceLoop i nowy retriever czasowy.",
    )
    await store.upsert_context_event(changed)
    assert await store.search_context_events(query="kontekstu") == []
    assert len(await store.search_context_events(query="retriever")) == 1

    assert await store.tombstone_context_event(stored.event_id) is True
    assert await store.search_context_events(query="VoiceLoop") == []


@pytest.mark.asyncio
async def test_conversation_history_can_be_scoped_to_session(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    await store.add_message("user", "sesja alfa", session_id="alpha")
    await store.add_message("user", "sesja beta", session_id="beta")

    assert await store.recent_messages(session_id="alpha") == [
        {"role": "user", "content": "sesja alfa"}
    ]
    assert len(await store.recent_messages()) == 2


@pytest.mark.asyncio
async def test_events_group_into_searchable_episode(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    events = [
        ContextEventV1.from_observation(
            source="screenpipe",
            source_id=f"event-{index}",
            started_at=datetime(2026, 9, 19, 10, index, tzinfo=UTC),
            app_name="Cursor",
            window_title="VoiceLoop",
            text=text,
        )
        for index, text in enumerate(
            ("Projekt VoiceLoop.", "Ustalono timeline jako oś główną."),
            start=1,
        )
    ]
    episodes = group_events_into_episodes(events)
    assert len(episodes) == 1
    assert episodes[0].source_event_ids == tuple(event.event_id for event in events)
    await store.upsert_context_episode(episodes[0])

    hits = await store.search_context_episodes(query="timeline")
    assert len(hits) == 1
    assert hits[0].metadata["digest_method"] == "deterministic_timeline_v1"
    assert await store.tombstone_context_episode(hits[0].episode_id) is True
    assert await store.search_context_episodes(query="timeline") == []


@pytest.mark.asyncio
async def test_context_prune_is_dry_run_by_default(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    expired = ContextEventV1.from_observation(
        source="screenpipe",
        source_id="expired",
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
        text="Stary kontekst.",
        expires_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    await store.upsert_context_event(expired)

    cutoff = datetime(2026, 9, 20, tzinfo=UTC)
    assert await store.prune_expired_context_records(now=cutoff) == {
        "events": 1,
        "episodes": 0,
    }
    assert await store.get_context_event(expired.event_id) is not None
    assert await store.prune_expired_context_records(
        now=cutoff,
        dry_run=False,
    ) == {"events": 1, "episodes": 0}
    assert await store.search_context_events(include_expired=True) == []


@pytest.mark.asyncio
async def test_screenpipe_text_search_paginates_all_media(tmp_path, monkeypatch) -> None:
    client = ScreenpipeClient(Settings(voiceloop_data_dir=str(tmp_path)))
    offsets: list[int] = []
    queries: list[str | None] = []

    async def fake_search(**kwargs):
        offsets.append(kwargs["offset"])
        queries.append(kwargs["query"])
        count = 50 if kwargs["offset"] == 0 else 3
        return [
            {
                "type": "OCR",
                "content": {
                    "timestamp": f"2026-09-19T08:00:{index:02d}+00:00",
                    "app_name": "Cursor",
                    "window_name": "VoiceLoop",
                    "text": f"fragment {kwargs['offset'] + index}",
                },
            }
            for index in range(count)
        ]

    monkeypatch.setattr(client, "_search", fake_search)
    items = await client.text_activity_between(
        start=datetime(2026, 9, 19, tzinfo=UTC),
        end=datetime(2026, 9, 20, tzinfo=UTC),
        query="VoiceLoop",
        max_results=100,
    )

    assert offsets == [0, 50]
    assert queries == ["VoiceLoop", "VoiceLoop"]
    assert len(items) == 53


@pytest.mark.asyncio
async def test_screenpipe_ingestor_builds_events_and_episodes(tmp_path) -> None:
    from voiceloop.screenpipe import ScreenpipeTextItem

    class ScreenpipeStub:
        async def text_activity_between(self, **_kwargs):
            return [
                ScreenpipeTextItem(
                    app_name="Cursor",
                    window_name="VoiceLoop",
                    timestamp="2026-09-19T10:01:00+00:00",
                    browser_url="",
                    text="Projekt VoiceLoop.",
                    content_type="OCR",
                ),
                ScreenpipeTextItem(
                    app_name="Cursor",
                    window_name="VoiceLoop",
                    timestamp="2026-09-19T10:06:00+00:00",
                    browser_url="",
                    text="Timeline jest osią główną.",
                    content_type="OCR",
                ),
            ]

    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    ingestor = ScreenpipeTimelineIngestor(
        screenpipe=ScreenpipeStub(),  # type: ignore[arg-type]
        memory=store,
    )
    report = await ingestor.ingest_range(
        start=datetime(2026, 9, 19, 10, tzinfo=UTC),
        end=datetime(2026, 9, 19, 11, tzinfo=UTC),
    )

    assert report.observed_items == 2
    assert report.stored_events == 2
    assert report.stored_episodes == 1
    assert len(await store.search_context_events(query="VoiceLoop")) == 2
    assert len(await store.search_context_episodes(query="timeline")) == 1


def test_time_first_query_resolves_polish_yesterday() -> None:
    local_tz = timezone(timedelta(hours=2))
    now = datetime(2026, 9, 20, 5, 30, tzinfo=local_tz)
    plan = build_time_first_query(
        "Przypomnij, co mówiłem Janowi o VoiceLoop wczoraj",
        now=now,
    )

    assert plan.used_time_filter is True
    assert plan.start == datetime(2026, 9, 18, 22, tzinfo=UTC)
    assert plan.end is not None and plan.end.date().isoformat() == "2026-09-19"
    assert "voiceloop" in plan.search_text
    assert "janowi" in plan.search_text


@pytest.mark.asyncio
async def test_time_first_retriever_returns_timeline_evidence(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    await store.upsert_context_event(
        ContextEventV1.from_observation(
            source="screenpipe",
            source_id="event-1",
            started_at=datetime(2026, 9, 19, 12, tzinfo=UTC),
            app_name="Cursor",
            window_title="VoiceLoop",
            text="Rozmowa o projekcie VoiceLoop.",
        )
    )
    retriever = TimeFirstRetriever(memory=store)
    pack = await retriever.retrieve(
        "Co robiłem przy VoiceLoop wczoraj?",
        now=datetime(2026, 9, 20, 12, tzinfo=UTC),
    )

    assert len(pack.items) == 1
    assert pack.items[0].source == "screenpipe"
    assert pack.items[0].selection_reason == "timeline_sql_fts"
    assert pack.items[0].scope is ContextScope.EPISODIC
    assert pack.items[0].trust is ContextTrust.OBSERVED
