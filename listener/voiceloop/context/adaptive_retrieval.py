from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from .policy import (
    RetrievalDomain,
    RetrievalPolicyV1,
    RetrievalRisk,
    resolve_retrieval_policy,
)
from .retrieval import CascadeTrace, TimeFirstRetriever, build_time_first_query
from .schema import ContextItemV1, ContextPackV1

if TYPE_CHECKING:
    from ..embeddings import OpenAICompatibleEmbeddingClient
    from ..memory import MemoryStore
    from ..qdrant_memory import QdrantVectorStore
    from ..screenpipe import ScreenpipeClient


class AdaptiveRetrievalTraceV1(BaseModel):
    """Content-free explanation of a policy-controlled retrieval run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=16, max_length=16)
    policy_version: str
    domain: RetrievalDomain
    query_class: str
    risk: RetrievalRisk
    strategy: str
    time_bounded: bool
    fts_weight: float = Field(ge=0.0, le=1.0)
    vector_weight: float = Field(ge=0.0, le=1.0)
    exact_count: int = Field(ge=0)
    vector_attempted: bool
    vector_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    axes_attempted: tuple[str, ...] = ()
    axes_queried: tuple[str, ...] = ()
    waves: tuple[tuple[str, ...], ...] = ()
    verified_rejected: int = Field(default=0, ge=0)
    stop_reason: str = Field(min_length=1, max_length=80)
    duration_ms: float = Field(ge=0.0)
    serving: Literal[False] = False


@dataclass(frozen=True)
class AdaptiveRetrievalRun:
    pack: ContextPackV1
    policy: RetrievalPolicyV1
    trace: AdaptiveRetrievalTraceV1


class AdaptiveHybridRetriever:
    """FTS-first retrieval with policy-controlled, SQL-verified vector expansion."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        screenpipe: ScreenpipeClient | None = None,
        embeddings: OpenAICompatibleEmbeddingClient | None = None,
        qdrant: QdrantVectorStore | None = None,
    ) -> None:
        self.memory = memory
        self.screenpipe = screenpipe
        self.embeddings = embeddings
        self.qdrant = qdrant

    async def retrieve(
        self,
        query: str,
        *,
        domain_hint: RetrievalDomain = "general",
        risk: RetrievalRisk = "low",
        session_id: str | None = None,
        now: datetime | None = None,
        limit: int = 8,
    ) -> AdaptiveRetrievalRun:
        started = time.perf_counter()
        safe_limit = max(1, min(int(limit), 30))
        plan = build_time_first_query(query, now=now)
        policy = resolve_retrieval_policy(
            query,
            domain_hint=domain_hint,
            risk=risk,
            time_bounded=plan.used_time_filter,
        )
        exact_retriever = TimeFirstRetriever(
            memory=self.memory,
            screenpipe=self.screenpipe,
            min_exact_evidence=10,
        )
        exact_pack = await exact_retriever.retrieve(
            query,
            session_id=session_id,
            now=now,
            limit=safe_limit,
            query_plan=plan,
        )

        vector_pack = ContextPackV1(question=query, session_id=session_id)
        vector_trace = CascadeTrace(stop_reason="not_needed")
        vector_attempted = False
        if policy.strategy == "exact_only":
            stop_reason = "policy_exact_only"
        elif policy.require_time_bound and not plan.used_time_filter:
            stop_reason = "unbounded_shadow_guard"
        elif not policy.vector_expansion_needed(
            exact_evidence=len(exact_pack.items)
        ):
            stop_reason = "exact_evidence_sufficient"
        else:
            vector_attempted = True
            vector_retriever = TimeFirstRetriever(
                memory=self.memory,
                embeddings=self.embeddings,
                qdrant=self.qdrant,
                scout_mode="cascade_2_2_1",
                policy=policy,
            )
            vector_pack = await vector_retriever.retrieve_vector_candidates(
                query,
                query_plan=plan,
                limit=min(
                    30,
                    safe_limit * policy.candidate_multiplier,
                ),
            )
            vector_trace = vector_retriever.last_trace
            stop_reason = vector_trace.stop_reason

        selected = _weighted_rrf(
            exact_pack.items,
            vector_pack.items,
            policy=policy,
            limit=safe_limit,
        )
        source_counts: dict[str, int] = {}
        for item in selected:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        pack = ContextPackV1(
            question=query,
            session_id=session_id,
            items=selected,
            sources=source_counts,
        )
        trace = AdaptiveRetrievalTraceV1(
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            domain=policy.domain,
            query_class=policy.query_class,
            risk=policy.risk,
            strategy=policy.strategy,
            time_bounded=plan.used_time_filter,
            fts_weight=policy.fts_weight,
            vector_weight=policy.vector_weight,
            exact_count=len(exact_pack.items),
            vector_attempted=vector_attempted,
            vector_count=len(vector_pack.items),
            result_count=len(selected),
            axes_attempted=vector_trace.axes_attempted,
            axes_queried=vector_trace.axes_queried,
            waves=vector_trace.waves,
            verified_rejected=vector_trace.verified_rejected,
            stop_reason=stop_reason,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return AdaptiveRetrievalRun(pack=pack, policy=policy, trace=trace)


def _weighted_rrf(
    exact_items: tuple[ContextItemV1, ...],
    vector_items: tuple[ContextItemV1, ...],
    *,
    policy: RetrievalPolicyV1,
    limit: int,
    rrf_k: int = 60,
) -> tuple[ContextItemV1, ...]:
    tracks = (
        ("fts", exact_items, policy.fts_weight),
        ("vector", vector_items, policy.vector_weight),
    )
    aggregate: dict[
        tuple[str, str, str],
        dict[str, object],
    ] = {}
    active_weight = sum(weight for _, items, weight in tracks if items)
    if active_weight <= 0:
        return ()
    maximum = active_weight / (rrf_k + 1)
    for track, items, weight in tracks:
        if not items or weight <= 0:
            continue
        for rank, item in enumerate(items, start=1):
            key = (item.source, item.source_id, item.content_hash)
            entry = aggregate.setdefault(
                key,
                {
                    "item": item,
                    "score": 0.0,
                    "tracks": [],
                    "axes": [],
                },
            )
            entry["score"] = float(entry["score"]) + weight / (rrf_k + rank)
            cast_tracks = entry["tracks"]
            assert isinstance(cast_tracks, list)
            cast_tracks.append(track)
            if track == "vector" and item.selection_reason.startswith(
                "semantic_scout:"
            ):
                cast_axes = entry["axes"]
                assert isinstance(cast_axes, list)
                cast_axes.extend(
                    item.selection_reason.split(":", 1)[1].split("+")
                )

    ordered = sorted(
        aggregate.values(),
        key=lambda entry: (
            float(entry["score"]),
            len(entry["tracks"]) if isinstance(entry["tracks"], list) else 0,
        ),
        reverse=True,
    )
    selected: list[ContextItemV1] = []
    for entry in ordered[:limit]:
        item = entry["item"]
        tracks_for_item = entry["tracks"]
        axes_for_item = entry["axes"]
        assert isinstance(item, ContextItemV1)
        assert isinstance(tracks_for_item, list)
        assert isinstance(axes_for_item, list)
        score = min(float(entry["score"]) / maximum, 1.0)
        selected.append(
            item.model_copy(
                update={
                    "retrieval_score": score,
                    "selection_reason": (
                        "adaptive_hybrid:" + "+".join(tracks_for_item)
                    ),
                    "metadata": {
                        **item.metadata,
                        "retrieval_axes": list(dict.fromkeys(axes_for_item)),
                    },
                }
            )
        )
    return tuple(selected)
