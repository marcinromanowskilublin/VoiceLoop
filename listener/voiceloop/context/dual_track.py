from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..embeddings import OpenAICompatibleEmbeddingClient
from ..memory import MemoryStore
from ..qdrant_memory import QdrantVectorStore
from .retrieval import CascadeTrace, TimeFirstRetriever, build_time_first_query
from .schema import ContextItemV1, ContextPackV1

DualTrackVerdict = Literal["agreement", "conflict", "uncertain"]


class DualTrackTraceV1(BaseModel):
    """Content-free, expiring evidence that two retrieval tracks were compared."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    trace_id: str = Field(min_length=24, max_length=24)
    created_at: datetime
    expires_at: datetime
    bounded: bool
    verdict: DualTrackVerdict
    reason: str = Field(min_length=1, max_length=80)
    baseline_count: int = Field(ge=0, le=30)
    cascade_count: int = Field(ge=0, le=30)
    overlap_count: int = Field(ge=0, le=30)
    top_match: bool
    baseline_result_ids: tuple[str, ...] = ()
    cascade_result_ids: tuple[str, ...] = ()
    baseline_axes_attempted: tuple[str, ...] = ()
    baseline_axes_queried: tuple[str, ...] = ()
    cascade_axes_attempted: tuple[str, ...] = ()
    cascade_axes_queried: tuple[str, ...] = ()
    cascade_waves: tuple[tuple[str, ...], ...] = ()
    baseline_stop_reason: str = Field(default="not_started", max_length=80)
    cascade_stop_reason: str = Field(default="not_started", max_length=80)
    baseline_verified_rejected: int = Field(default=0, ge=0)
    cascade_verified_rejected: int = Field(default=0, ge=0)
    baseline_duration_ms: float = Field(default=0.0, ge=0.0)
    cascade_duration_ms: float = Field(default=0.0, ge=0.0)
    total_duration_ms: float = Field(default=0.0, ge=0.0)
    error_type: str | None = Field(default=None, max_length=120)


@dataclass(frozen=True)
class DualTrackDecision:
    verdict: DualTrackVerdict
    reason: str
    overlap_count: int
    top_match: bool


@dataclass(frozen=True)
class DualTrackRun:
    baseline: ContextPackV1
    cascade: ContextPackV1
    trace: DualTrackTraceV1


def arbitrate_context_packs(
    baseline: ContextPackV1,
    cascade: ContextPackV1,
) -> DualTrackDecision:
    """Conservative deterministic arbiter; it never selects a serving result."""

    baseline_ids = _canonical_item_ids(baseline.items)
    cascade_ids = _canonical_item_ids(cascade.items)
    if not baseline_ids and not cascade_ids:
        return DualTrackDecision(
            verdict="uncertain",
            reason="no_candidates",
            overlap_count=0,
            top_match=False,
        )
    if not baseline_ids or not cascade_ids:
        return DualTrackDecision(
            verdict="uncertain",
            reason="missing_track",
            overlap_count=0,
            top_match=False,
        )
    overlap = len(set(baseline_ids) & set(cascade_ids))
    top_match = baseline_ids[0] == cascade_ids[0]
    if top_match:
        return DualTrackDecision(
            verdict="agreement",
            reason="top_match",
            overlap_count=overlap,
            top_match=True,
        )
    if overlap == 0:
        return DualTrackDecision(
            verdict="conflict",
            reason="disjoint_results",
            overlap_count=0,
            top_match=False,
        )
    return DualTrackDecision(
        verdict="uncertain",
        reason="partial_overlap",
        overlap_count=overlap,
        top_match=False,
    )


class DualTrackRunner:
    """Runs RRF5 and 2+2+1 in parallel without changing production behavior."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        embeddings: OpenAICompatibleEmbeddingClient,
        qdrant: QdrantVectorStore,
        max_inflight: int = 1,
        timeout_seconds: float = 5.0,
        trace_retention_days: int = 30,
        qmax: int = 5,
        wave_size: int = 2,
    ) -> None:
        self.memory = memory
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.max_inflight = max(1, min(int(max_inflight), 8))
        self.timeout_seconds = max(0.1, min(float(timeout_seconds), 30.0))
        self.trace_retention_days = max(
            1,
            min(int(trace_retention_days), 365),
        )
        self.qmax = max(1, min(int(qmax), 5))
        self.wave_size = max(1, min(int(wave_size), 2))
        self._inflight = 0
        self._slot_lock = asyncio.Lock()

    @property
    def inflight(self) -> int:
        return self._inflight

    def status(self) -> dict[str, object]:
        return {
            "mode": "shadow",
            "serving": False,
            "collection": self.qdrant.collection_name,
            "max_inflight": self.max_inflight,
            "inflight": self.inflight,
        }

    async def run(
        self,
        query: str,
        *,
        request_id: str,
        now: datetime | None = None,
        limit: int = 8,
        persist: bool = True,
    ) -> DualTrackRun:
        started = time.perf_counter()
        observed_at = now or datetime.now().astimezone()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        created_at = observed_at.astimezone(UTC)
        trace_id = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:24]
        plan = build_time_first_query(query, now=observed_at)
        empty = ContextPackV1(question=query)
        if not plan.used_time_filter:
            trace = self._trace(
                trace_id=trace_id,
                created_at=created_at,
                verdict="uncertain",
                reason="unbounded_rejected",
                baseline=empty,
                cascade=empty,
                baseline_trace=CascadeTrace(stop_reason="unbounded_rejected"),
                cascade_trace=CascadeTrace(stop_reason="unbounded_rejected"),
                timings={},
                total_started=started,
                bounded=False,
            )
            await self._persist(trace, enabled=persist)
            return DualTrackRun(empty, empty, trace)
        if not await self._try_enter():
            trace = self._trace(
                trace_id=trace_id,
                created_at=created_at,
                verdict="uncertain",
                reason="concurrency_limit",
                baseline=empty,
                cascade=empty,
                baseline_trace=CascadeTrace(
                    time_bounded=True,
                    stop_reason="concurrency_limit",
                ),
                cascade_trace=CascadeTrace(
                    time_bounded=True,
                    stop_reason="concurrency_limit",
                ),
                timings={},
                total_started=started,
                bounded=True,
            )
            await self._persist(trace, enabled=persist)
            return DualTrackRun(empty, empty, trace)

        baseline_retriever = TimeFirstRetriever(
            memory=self.memory,
            embeddings=self.embeddings,
            qdrant=self.qdrant,
            scout_mode="rrf5",
        )
        cascade_retriever = TimeFirstRetriever(
            memory=self.memory,
            embeddings=self.embeddings,
            qdrant=self.qdrant,
            scout_mode="cascade_2_2_1",
            qmax=self.qmax,
            wave_size=self.wave_size,
        )
        timings: dict[str, float] = {}

        async def measured(name: str, retriever: TimeFirstRetriever) -> ContextPackV1:
            track_started = time.perf_counter()
            try:
                return await retriever.retrieve_vector_candidates(
                    query,
                    query_plan=plan,
                    limit=limit,
                )
            finally:
                timings[name] = round(
                    (time.perf_counter() - track_started) * 1000,
                    3,
                )

        error_type: str | None = None
        reason_override: str | None = None
        baseline = empty
        cascade = empty
        tasks = {
            "baseline": asyncio.create_task(
                measured("baseline", baseline_retriever)
            ),
            "cascade": asyncio.create_task(
                measured("cascade", cascade_retriever)
            ),
        }
        try:
            _, pending = await asyncio.wait(
                set(tasks.values()),
                timeout=self.timeout_seconds,
            )
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                error_type = "TimeoutError"
                reason_override = "timeout"
            errors: list[str] = []
            for name, task in tasks.items():
                if task.cancelled():
                    continue
                exception = task.exception()
                if exception is not None:
                    errors.append(type(exception).__name__)
                    continue
                if name == "baseline":
                    baseline = task.result()
                else:
                    cascade = task.result()
            if errors and reason_override is None:
                error_type = errors[0]
                reason_override = "track_error"
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            await self._leave()

        decision = arbitrate_context_packs(baseline, cascade)
        trace = self._trace(
            trace_id=trace_id,
            created_at=created_at,
            verdict=("uncertain" if reason_override else decision.verdict),
            reason=reason_override or decision.reason,
            baseline=baseline,
            cascade=cascade,
            baseline_trace=baseline_retriever.last_trace,
            cascade_trace=cascade_retriever.last_trace,
            timings=timings,
            total_started=started,
            bounded=True,
            error_type=error_type,
        )
        await self._persist(trace, enabled=persist)
        return DualTrackRun(baseline, cascade, trace)

    async def _try_enter(self) -> bool:
        async with self._slot_lock:
            if self._inflight >= self.max_inflight:
                return False
            self._inflight += 1
            return True

    async def _leave(self) -> None:
        async with self._slot_lock:
            self._inflight = max(0, self._inflight - 1)

    async def _persist(self, trace: DualTrackTraceV1, *, enabled: bool) -> None:
        if enabled:
            await self.memory.save_context_dual_track_trace(trace)

    def _trace(
        self,
        *,
        trace_id: str,
        created_at: datetime,
        verdict: DualTrackVerdict,
        reason: str,
        baseline: ContextPackV1,
        cascade: ContextPackV1,
        baseline_trace: CascadeTrace,
        cascade_trace: CascadeTrace,
        timings: dict[str, float],
        total_started: float,
        bounded: bool,
        error_type: str | None = None,
    ) -> DualTrackTraceV1:
        decision = arbitrate_context_packs(baseline, cascade)
        return DualTrackTraceV1(
            trace_id=trace_id,
            created_at=created_at,
            expires_at=created_at + timedelta(days=self.trace_retention_days),
            bounded=bounded,
            verdict=verdict,
            reason=reason,
            baseline_count=len(baseline.items),
            cascade_count=len(cascade.items),
            overlap_count=decision.overlap_count,
            top_match=decision.top_match,
            baseline_result_ids=_opaque_item_ids(trace_id, baseline.items),
            cascade_result_ids=_opaque_item_ids(trace_id, cascade.items),
            baseline_axes_attempted=baseline_trace.axes_attempted,
            baseline_axes_queried=baseline_trace.axes_queried,
            cascade_axes_attempted=cascade_trace.axes_attempted,
            cascade_axes_queried=cascade_trace.axes_queried,
            cascade_waves=cascade_trace.waves,
            baseline_stop_reason=baseline_trace.stop_reason,
            cascade_stop_reason=cascade_trace.stop_reason,
            baseline_verified_rejected=baseline_trace.verified_rejected,
            cascade_verified_rejected=cascade_trace.verified_rejected,
            baseline_duration_ms=timings.get("baseline", 0.0),
            cascade_duration_ms=timings.get("cascade", 0.0),
            total_duration_ms=round(
                (time.perf_counter() - total_started) * 1000,
                3,
            ),
            error_type=error_type,
        )


def _canonical_item_ids(items: tuple[ContextItemV1, ...]) -> tuple[str, ...]:
    return tuple(
        f"{item.source}:{item.source_id}:{item.content_hash}" for item in items
    )


def _opaque_item_ids(
    trace_id: str,
    items: tuple[ContextItemV1, ...],
) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            (
                f"{trace_id}:{item.source}:{item.source_id}:"
                f"{item.content_hash}"
            ).encode()
        ).hexdigest()[:24]
        for item in items
    )
