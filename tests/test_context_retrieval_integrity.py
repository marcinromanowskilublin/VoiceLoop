from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.context.retrieval import TimeFirstRetriever, build_time_first_query
from voiceloop.context.schema import ContextEpisodeV1, ContextEventV1
from voiceloop.context.vectorization import ContextEpisodeVectorizer
from voiceloop.memory import MemoryStore
from voiceloop.screenpipe import ScreenpipeError
from voiceloop.settings import Settings

DAY = datetime(2026, 9, 19, 12, tzinfo=UTC)


async def seeded(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    event = ContextEventV1.from_observation(
        source="test", source_id="source", started_at=DAY,
        text="Oryginalny dokument.",
    )
    await store.upsert_context_event(event)
    episode = ContextEpisodeV1(
        episode_id="episode-1", source="timeline", source_id="episode-1",
        started_at=DAY, ended_at=DAY + timedelta(hours=1),
        summary="Kanoniczna treść SQL.", source_event_ids=(event.event_id,),
    )
    await store.upsert_context_episode(episode)
    hit = SimpleNamespace(
        source=episode.source, source_id=episode.source_id,
        content="NIE UŻYWAJ TEJ KOPII Z QDRANT", title="Stary tytuł",
        score=1.0,
        metadata={
            "episode_id": episode.episode_id,
            "content_hash": episode.content_hash,
            "source_event_hashes": {event.event_id: event.content_hash},
            "time": "2099-01-01T00:00:00+00:00",
        },
    )
    return store, event, episode, hit


def plan():
    return build_time_first_query("wczoraj", now=DAY + timedelta(days=1))


@pytest.mark.asyncio
async def test_semantic_candidate_uses_sql_content_time_and_confidence(tmp_path):
    store, _, episode, hit = await seeded(tmp_path)
    item = await TimeFirstRetriever(memory=store)._verified_episode_item(
        hit, plan=plan(), axis="semantic",
    )
    assert item is not None
    assert item.content == episode.summary
    assert item.started_at == DAY
    assert item.confidence == episode.as_context_item().confidence
    assert item.confidence != hit.score
    assert item.retrieval_score == 1.0
    assert item.metadata["source_event_ids"] == list(episode.source_event_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "no_id", "missing_episode", "stale_episode", "wrong_source", "wrong_source_id",
    "missing_hash", "missing_source_hashes", "stale_source", "deleted_episode",
    "expired_episode", "deleted_source", "expired_source", "missing_source",
    "changed_source", "outside_time", "nan_score",
])
async def test_semantic_candidate_rejects_invalid_evidence(tmp_path, fault):
    store, event, episode, hit = await seeded(tmp_path)
    if fault == "no_id":
        hit.metadata.pop("episode_id")
    elif fault == "missing_episode":
        hit.metadata["episode_id"] = "absent"
    elif fault == "stale_episode":
        hit.metadata["content_hash"] = "stale"
    elif fault == "wrong_source":
        hit.source = "wrong"
    elif fault == "wrong_source_id":
        hit.source_id = "wrong"
    elif fault == "missing_hash":
        hit.metadata.pop("content_hash")
    elif fault == "missing_source_hashes":
        hit.metadata.pop("source_event_hashes")
    elif fault == "stale_source":
        hit.metadata["source_event_hashes"][event.event_id] = "old-version"
    elif fault == "deleted_episode":
        await store.tombstone_context_episode(episode.episode_id)
    elif fault == "expired_episode":
        await store.upsert_context_episode(episode.model_copy(update={
            "expires_at": datetime.now(UTC) - timedelta(seconds=1),
        }))
    elif fault == "deleted_source":
        await store.tombstone_context_event(event.event_id)
    elif fault == "expired_source":
        await store.upsert_context_event(event.model_copy(update={
            "expires_at": datetime.now(UTC) - timedelta(seconds=1),
        }))
    elif fault == "missing_source":
        store.get_context_event = AsyncMock(return_value=None)
    elif fault == "changed_source":
        await store.upsert_context_event(ContextEventV1.from_observation(
            source=event.source, source_id=event.source_id,
            started_at=DAY, text="Zmieniony dokument.",
        ))
    elif fault == "nan_score":
        hit.score = float("nan")
    query_plan = (
        build_time_first_query("dzisiaj", now=DAY + timedelta(days=3))
        if fault == "outside_time" else plan()
    )
    assert await TimeFirstRetriever(memory=store)._verified_episode_item(
        hit, plan=query_plan, axis="semantic",
    ) is None


@pytest.mark.asyncio
async def test_semantic_interval_overlap_matches_sql(tmp_path):
    store, _, _, hit = await seeded(tmp_path)
    query_plan = plan().model_copy(update={"start": DAY + timedelta(minutes=30)})
    assert await TimeFirstRetriever(memory=store)._verified_episode_item(
        hit, plan=query_plan, axis="semantic",
    ) is not None


@pytest.mark.parametrize("timestamp", [
    "invalid", "", "2026-09-19T12:00:00", "2026-09-18T12:00:00Z", None,
])
def test_screenpipe_invalid_or_outside_timestamp_is_not_invented(timestamp):
    raw = SimpleNamespace(timestamp=timestamp)
    assert TimeFirstRetriever._screenpipe_context_items(
        [raw], existing=[], plan=plan(),
    ) == []


def test_screenpipe_valid_timestamp_keeps_source_time():
    raw = SimpleNamespace(
        timestamp="2026-09-19T14:00:00+02:00", app_name="test", window_name="test",
        text="tekst", content_type="ocr", browser_url="",
    )
    items = TimeFirstRetriever._screenpipe_context_items([raw], existing=[], plan=plan())
    assert len(items) == 1
    assert items[0].started_at == DAY


@pytest.mark.asyncio
@pytest.mark.parametrize("screenpipe_down", [False, True])
async def test_empty_time_recall_never_falls_back_to_unbounded_search(tmp_path, screenpipe_down):
    store = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    store.search_vector_memories = AsyncMock(side_effect=AssertionError("unbounded"))
    store.list_memories = AsyncMock(side_effect=AssertionError("unbounded"))
    embeddings = Mock(enabled=False)
    screenpipe = None
    if screenpipe_down:
        screenpipe = Mock()
        screenpipe.text_activity_between = AsyncMock(side_effect=ScreenpipeError("down"))
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path), context_timeline_recall_enabled=True),
        store, Mock(), embeddings=embeddings, screenpipe=screenpipe,
    )
    message, payload = await registry._recall({"query": "Co ustaliliśmy wczoraj?"})
    assert payload["items"] == []
    assert payload["retrieval"] == "timeline_v1"
    assert payload["time_filter"]["start"]
    assert "0 wpisów" in message
    store.search_vector_memories.assert_not_called()
    store.list_memories.assert_not_called()


@pytest.mark.asyncio
async def test_indexed_episode_carries_source_versions_for_retrieval(tmp_path):
    store, event, episode, _ = await seeded(tmp_path)
    embeddings = Mock(enabled=True)
    embeddings.embed_documents = AsyncMock(return_value=[[1.0, 0.0]])
    qdrant = Mock(enabled=True)
    qdrant.upsert_memory = AsyncMock()
    await ContextEpisodeVectorizer(
        embeddings=embeddings, qdrant=qdrant, memory=store,
    ).index_episode(episode)
    payload = qdrant.upsert_memory.call_args.kwargs
    assert payload["metadata"]["source_event_hashes"] == {event.event_id: event.content_hash}
    assert payload["content_hash"] == episode.content_hash
