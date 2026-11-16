from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from voiceloop.context.adaptive_retrieval import AdaptiveHybridRetriever
from voiceloop.context.policy import resolve_retrieval_policy
from voiceloop.context.retrieval import (
    TimeFirstRetriever,
    build_time_first_query,
)
from voiceloop.context.schema import (
    ContextEpisodeV1,
    ContextEventV1,
    ContextIndexOperation,
)
from voiceloop.memory import MemoryStore

DAY = datetime(2026, 11, 15, 12, tzinfo=UTC)


class EmbeddingsStub:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def accepts_private_text(self) -> bool:
        return True

    async def embed_queries(self, documents):
        self.calls.append(list(documents))
        return [[1.0, float(index)] for index, _ in enumerate(documents)]


class QdrantStub:
    enabled = True

    def __init__(self, hits_by_axis) -> None:
        self.hits_by_axis = hits_by_axis
        self.calls: list[tuple[str, ...]] = []

    def accepts_private_data(self) -> bool:
        return True

    async def search(self, *, vector_names, **_kwargs):
        axes = tuple(vector_names)
        self.calls.append(axes)
        if len(axes) == 1:
            return list(self.hits_by_axis.get(axes[0], ()))
        return []


async def add_episode(
    store: MemoryStore,
    *,
    source_id: str,
    summary: str,
    minute: int,
):
    event = ContextEventV1.from_observation(
        source="test",
        source_id=f"event-{source_id}",
        started_at=DAY + timedelta(minutes=minute),
        text=summary,
    )
    await store.upsert_context_event(event)
    episode = ContextEpisodeV1(
        episode_id=f"episode-{source_id}",
        source="test",
        source_id=source_id,
        started_at=event.started_at,
        ended_at=event.started_at + timedelta(minutes=1),
        summary=summary,
        source_event_ids=(event.event_id,),
    )
    await store.upsert_context_episode(episode)
    hit = SimpleNamespace(
        source=episode.source,
        source_id=episode.source_id,
        title="stale vector title",
        content="NIE UŻYWAJ PAYLOADU QDRANT",
        score=0.9,
        metadata={
            "episode_id": episode.episode_id,
            "content_hash": episode.content_hash,
            "source_event_hashes": {event.event_id: event.content_hash},
        },
    )
    return episode, hit


def test_policy_changes_by_domain_without_relaxing_sql_verification() -> None:
    query = "Wymyśl nowy motyw dla postaci i zaproponuj wariant narracji."

    general = resolve_retrieval_policy(query, domain_hint="general")
    windows = resolve_retrieval_policy(query, domain_hint="windows")
    rpg = resolve_retrieval_policy(query, domain_hint="rpg")

    assert rpg.vector_weight > general.vector_weight > windows.vector_weight
    assert rpg.qmax > windows.qmax
    assert rpg.min_axis_coverage == 2
    assert rpg.always_vector_scout is True
    assert all(
        policy.require_canonical_sql and not policy.serving_eligible
        for policy in (general, windows, rpg)
    )


def test_windows_exact_operation_stays_exact_first() -> None:
    policy = resolve_retrieval_policy(
        r"Otwórz C:\projekty\VoiceLoop\README.md",
        domain_hint="windows",
    )

    assert policy.query_class == "operational"
    assert policy.strategy == "exact_only"
    assert policy.vector_expansion_needed(exact_evidence=0) is False
    assert policy.fts_weight >= 0.8


def test_high_risk_policy_requires_broader_evidence() -> None:
    low = resolve_retrieval_policy(
        "Jaką decyzję podjęliśmy wczoraj?",
        risk="low",
        time_bounded=True,
    )
    high = resolve_retrieval_policy(
        "Jaką decyzję podjęliśmy wczoraj?",
        risk="high",
        time_bounded=True,
    )

    assert high.fts_weight > low.fts_weight
    assert high.min_axis_coverage >= 2
    assert "guard:high_risk_abstention" in high.reason_codes


@pytest.mark.asyncio
async def test_exact_evidence_avoids_unnecessary_vector_cost(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    await add_episode(
        store,
        source_id="decision",
        summary="Ustaliliśmy wdrożenie projektu Atlas.",
        minute=0,
    )
    embeddings = EmbeddingsStub()
    retriever = AdaptiveHybridRetriever(
        memory=store,
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=QdrantStub({}),  # type: ignore[arg-type]
    )

    run = await retriever.retrieve(
        "Co ustaliliśmy wczoraj o wdrożeniu projektu Atlas?",
        domain_hint="windows",
        now=DAY + timedelta(days=1),
    )

    assert run.pack.items
    assert run.trace.exact_count == 1
    assert run.trace.vector_count == 0
    assert run.trace.stop_reason == "exact_evidence_sufficient"
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_vector_expansion_uses_only_sql_hydrated_content(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode, hit = await add_episode(
        store,
        source_id="motif",
        summary="Postać boi się utraty pamięci.",
        minute=0,
    )
    embeddings = EmbeddingsStub()
    retriever = AdaptiveHybridRetriever(
        memory=store,
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=QdrantStub(
            {
                "semantic": [hit],
                "topic": [hit],
                "person_context": [hit],
            }
        ),  # type: ignore[arg-type]
    )

    run = await retriever.retrieve(
        "Wymyśl wczoraj nowy motyw narracyjny dla tej postaci.",
        domain_hint="rpg",
        now=DAY + timedelta(days=1),
    )

    assert run.trace.vector_count == 1
    assert run.trace.axes_queried
    assert run.pack.items[0].content == episode.summary
    assert "NIE UŻYWAJ" not in run.pack.items[0].content
    assert run.pack.items[0].selection_reason.startswith("adaptive_hybrid:")


@pytest.mark.asyncio
async def test_wave_fusion_prevents_first_axis_from_monopolizing_results(
    tmp_path,
) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    target, target_hit = await add_episode(
        store,
        source_id="target",
        summary="Ustaliliśmy, że Borut chroni most.",
        minute=0,
    )
    _, distractor_hit = await add_episode(
        store,
        source_id="distractor",
        summary="Rozmawialiśmy ogólnie o lesie.",
        minute=1,
    )
    target_hit.score = 0.8
    distractor_hit.score = 0.95
    qdrant = QdrantStub(
        {
            "semantic": [distractor_hit, target_hit],
            "decision": [target_hit],
        }
    )
    retriever = AdaptiveHybridRetriever(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    run = await retriever.retrieve(
        "Jaką decyzję podjęliśmy wczoraj w sprawie Boruta?",
        domain_hint="rpg",
        risk="high",
        now=DAY + timedelta(days=1),
        limit=2,
    )

    assert run.trace.axes_queried[:2] == ("semantic", "decision")
    assert run.pack.items[0].source_id == target.source_id


@pytest.mark.asyncio
async def test_unbounded_policy_is_shadow_guarded_without_vector_search(
    tmp_path,
) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    embeddings = EmbeddingsStub()
    qdrant = QdrantStub({})
    retriever = AdaptiveHybridRetriever(
        memory=store,
        embeddings=embeddings,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    run = await retriever.retrieve(
        "Wymyśl motyw dla postaci.",
        domain_hint="rpg",
    )

    assert run.trace.stop_reason == "unbounded_shadow_guard"
    assert run.trace.serving is False
    assert embeddings.calls == []
    assert qdrant.calls == []


@pytest.mark.asyncio
async def test_stale_generation_cannot_hide_live_generation_in_wave_fusion(
    tmp_path,
) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    episode, base_hit = await add_episode(
        store,
        source_id="versioned",
        summary="Ustaliliśmy żywą wersję decyzji.",
        minute=0,
    )
    state = await store.get_context_index_state(episode.episode_id)
    assert state is not None
    fingerprint = "f" * 64
    profile = "test-profile"
    claim_token = await store.claim_context_index_job(
        episode_id=episode.episode_id,
        operation=ContextIndexOperation.UPSERT,
        desired_revision=episode.content_hash,
        generation=state.desired_generation,
        index_fingerprint=fingerprint,
        index_profile=profile,
    )
    assert claim_token
    assert await store.mark_context_index_applied(
        episode_id=episode.episode_id,
        operation=ContextIndexOperation.UPSERT,
        desired_revision=episode.content_hash,
        generation=state.desired_generation,
        claim_token=claim_token,
        index_fingerprint=fingerprint,
        index_profile=profile,
        vector_spaces=("semantic",),
    )
    live_metadata = {
        **base_hit.metadata,
        "index_generation": state.desired_generation,
        "index_claim_token": claim_token,
        "vector_index_hash": fingerprint,
    }
    stale_hit = SimpleNamespace(
        **{
            **base_hit.__dict__,
            "score": 0.99,
            "metadata": {
                **live_metadata,
                "index_generation": 0,
                "index_claim_token": "stale",
                "vector_index_hash": "stale",
            },
        }
    )
    live_hit = SimpleNamespace(
        **{
            **base_hit.__dict__,
            "score": 0.80,
            "metadata": live_metadata,
        }
    )
    query = "Jaką decyzję podjęliśmy wczoraj?"
    plan = build_time_first_query(query, now=DAY + timedelta(days=1))
    policy = resolve_retrieval_policy(
        query,
        domain_hint="rpg",
        risk="high",
        time_bounded=True,
    )
    retriever = TimeFirstRetriever(
        memory=store,
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub(
            {
                "semantic": [stale_hit, live_hit],
                "decision": [stale_hit, live_hit],
            }
        ),  # type: ignore[arg-type]
        scout_mode="cascade_2_2_1",
        policy=policy,
    )

    pack = await retriever.retrieve_vector_candidates(
        query,
        query_plan=plan,
        limit=2,
    )

    assert [item.source_id for item in pack.items] == [episode.source_id]
