from __future__ import annotations

import asyncio
import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..memory_vectorization import (
    MEMORY_VECTOR_NAMES,
    memory_query_documents,
    memory_query_weights,
)
from .schema import (
    ContextEventV1,
    ContextItemV1,
    ContextPackV1,
)

if TYPE_CHECKING:
    from ..embeddings import OpenAICompatibleEmbeddingClient
    from ..memory import MemoryStore
    from ..qdrant_memory import QdrantVectorStore
    from ..screenpipe import ScreenpipeClient

_STOPWORDS = {
    "a",
    "ale",
    "co",
    "czy",
    "dni",
    "dnia",
    "dniach",
    "do",
    "dzis",
    "dzisiaj",
    "godzin",
    "godziny",
    "godzinach",
    "i",
    "jak",
    "jeszcze",
    "kilka",
    "kilku",
    "mi",
    "miesiac",
    "miesiaca",
    "miesiace",
    "miesiecy",
    "mowilismy",
    "na",
    "o",
    "od",
    "okolo",
    "ostatni",
    "ostatnia",
    "ostatnie",
    "ostatniego",
    "ostatnich",
    "ostatnim",
    "paru",
    "pare",
    "przedwczoraj",
    "przypomnij",
    "rok",
    "roku",
    "temu",
    "tydzien",
    "tygodni",
    "tygodnia",
    "tygodnie",
    "w",
    "wczoraj",
    "z",
    "ze",
    "zeszlym",
    "zeszlego",
}

TimePrecision = Literal["exact", "approximate", "unknown"]
TimeSource = Literal["none", "relative_text", "semantic_relation"]


class TimeFirstQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str = Field(min_length=1, max_length=8000)
    search_text: str = Field(default="", max_length=1000)
    start: datetime | None = None
    end: datetime | None = None
    used_time_filter: bool = False
    time_precision: TimePrecision = "unknown"
    time_relation: str = Field(default="unbounded", min_length=1, max_length=80)
    time_source: TimeSource = "none"
    time_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    preferred_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CascadeTrace:
    time_stage: str = "Czas"
    time_bounded: bool = False
    time_precision: TimePrecision = "unknown"
    waves: tuple[tuple[str, ...], ...] = ()
    axes_attempted: tuple[str, ...] = ()
    axes_queried: tuple[str, ...] = ()
    stop_reason: str = "not_started"
    candidate_count: int = 0
    verified_rejected: int = 0


def build_time_first_query(
    query: str,
    *,
    now: datetime | None = None,
) -> TimeFirstQueryPlan:
    local_now = now or datetime.now().astimezone()
    if local_now.tzinfo is None or local_now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    normalized = _normalized_text(query)
    start: datetime | None = None
    end: datetime | None = None
    precision: TimePrecision = "unknown"
    relation = "unbounded"
    source: TimeSource = "none"
    confidence = 0.0
    preferred_axes: tuple[str, ...] = ()
    if "przedwczoraj" in normalized:
        day = local_now.date() - timedelta(days=2)
        start = datetime.combine(day, time.min, tzinfo=local_now.tzinfo)
        end = datetime.combine(day, time.max, tzinfo=local_now.tzinfo)
        precision = "exact"
        relation = "two_days_before_observation"
        source = "relative_text"
        confidence = 0.95
    elif "wczoraj" in normalized:
        day = local_now.date() - timedelta(days=1)
        start = datetime.combine(day, time.min, tzinfo=local_now.tzinfo)
        end = datetime.combine(day, time.max, tzinfo=local_now.tzinfo)
        precision = "exact"
        relation = "one_day_before_observation"
        source = "relative_text"
        confidence = 0.95
    elif re.search(r"\b(?:dzis|dzisiaj)\b", normalized):
        start = datetime.combine(local_now.date(), time.min, tzinfo=local_now.tzinfo)
        end = local_now
        precision = "exact"
        relation = "same_day_as_observation"
        source = "relative_text"
        confidence = 0.95
    else:
        recent = re.search(
            r"\b(?:ostatni(?:e|ch|ego|m|a)?)\s+(\d{1,3})\s+"
            r"(godzin(?:y|ach)?|dni|dnia|dniach|tygodni(?:e|ach)?|miesiec(?:y|ach)?|miesiac(?:e|ach)?)\b",
            normalized,
        )
        if recent:
            amount = max(1, min(int(recent.group(1)), 365))
            unit = recent.group(2)
            if unit.startswith("godzin"):
                delta = timedelta(hours=amount)
                relation = f"last_{amount}_hours"
            elif unit.startswith("tygodni"):
                delta = timedelta(days=amount * 7)
                relation = f"last_{amount}_weeks"
            elif unit.startswith("miesi"):
                delta = timedelta(days=amount * 30)
                relation = f"last_{amount}_months"
            else:
                delta = timedelta(days=amount)
                relation = f"last_{amount}_days"
            start = local_now - delta
            end = local_now
            precision = "exact" if not unit.startswith("miesi") else "approximate"
            source = "relative_text"
            confidence = 0.9 if precision == "exact" else 0.7
        else:
            approx = _approximate_relative_window(normalized, local_now=local_now)
            if approx is not None:
                start, end, relation, confidence, preferred_axes = approx
                precision = "approximate"
                source = "relative_text"
    return TimeFirstQueryPlan(
        original_query=query,
        search_text=_search_text(query),
        start=start.astimezone(UTC) if start else None,
        end=end.astimezone(UTC) if end else None,
        used_time_filter=start is not None,
        time_precision=precision,
        time_relation=relation,
        time_source=source,
        time_confidence=confidence,
        preferred_axes=preferred_axes,
    )


class TimeFirstRetriever:
    """Retrieve exact timeline evidence before any future semantic expansion."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        screenpipe: ScreenpipeClient | None = None,
        embeddings: OpenAICompatibleEmbeddingClient | None = None,
        qdrant: QdrantVectorStore | None = None,
        min_exact_evidence: int = 2,
        scout_mode: Literal["sequential", "rrf5", "cascade_2_2_1"] = "sequential",
        qmax: int = 5,
        wave_size: int = 2,
        preferred_axes: tuple[str, ...] = (),
    ) -> None:
        self.memory = memory
        self.screenpipe = screenpipe
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.min_exact_evidence = max(1, min(int(min_exact_evidence), 10))
        self.scout_mode = scout_mode
        self.qmax = max(1, min(int(qmax), len(MEMORY_VECTOR_NAMES)))
        self.wave_size = max(1, min(int(wave_size), 2))
        self.preferred_axes = tuple(
            axis for axis in preferred_axes if axis in MEMORY_VECTOR_NAMES
        )
        self.last_trace = CascadeTrace()

    async def retrieve(
        self,
        query: str,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
        limit: int = 8,
        query_plan: TimeFirstQueryPlan | None = None,
    ) -> ContextPackV1:
        plan = query_plan or build_time_first_query(query, now=now)
        self.last_trace = CascadeTrace(
            time_bounded=plan.used_time_filter,
            time_precision=plan.time_precision,
        )
        if self.scout_mode in {"rrf5", "cascade_2_2_1"} and not _is_bounded_plan(plan):
            self.last_trace = CascadeTrace(
                time_bounded=False,
                time_precision=plan.time_precision,
                stop_reason="unbounded_rejected",
            )
            return ContextPackV1(question=query, session_id=session_id)
        safe_limit = max(1, min(int(limit), 30))
        episodes = await self.memory.search_context_episodes(
            query=plan.search_text,
            start=plan.start,
            end=plan.end,
            limit=safe_limit,
        )
        items = [
            episode.as_context_item(selection_reason="timeline_episode_fts")
            for episode in episodes
        ]
        events = await self.memory.search_context_events(
            query=plan.search_text,
            start=plan.start,
            end=plan.end,
            limit=max(1, safe_limit - len(items)),
        )
        items.extend(
            event.as_context_item(selection_reason="timeline_sql_fts")
            for event in events
            if event.event_id not in {
                source_event_id
                for episode in episodes
                for source_event_id in episode.source_event_ids
            }
        )
        if (
            len(items) < safe_limit
            and self.screenpipe is not None
            and plan.start is not None
            and plan.end is not None
        ):
            raw_items = await self.screenpipe.text_activity_between(
                start=plan.start,
                end=plan.end,
                query=plan.search_text or None,
                max_results=max(50, safe_limit * 10),
            )
            items.extend(
                self._screenpipe_context_items(raw_items, existing=items, plan=plan)
            )
        if len(items) < self.min_exact_evidence:
            items.extend(
                await self._semantic_scout(
                    query,
                    plan=plan,
                    existing=items,
                    limit=safe_limit - len(items),
                )
            )
        elif self.scout_mode == "cascade_2_2_1":
            self.last_trace = CascadeTrace(
                time_bounded=plan.used_time_filter,
                time_precision=plan.time_precision,
                stop_reason="exact_evidence",
                candidate_count=len(items),
            )
        selected = tuple(items[:safe_limit])
        source_counts: dict[str, int] = {}
        for item in selected:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        return ContextPackV1(
            question=query,
            session_id=session_id,
            items=selected,
            sources=source_counts,
        )

    async def retrieve_vector_candidates(
        self,
        query: str,
        *,
        query_plan: TimeFirstQueryPlan,
        limit: int = 8,
    ) -> ContextPackV1:
        """Run only the vector track after a shared, already-bounded time gate."""

        self.last_trace = CascadeTrace(
            time_bounded=query_plan.used_time_filter,
            time_precision=query_plan.time_precision,
        )
        if self.scout_mode in {"rrf5", "cascade_2_2_1"} and not _is_bounded_plan(
            query_plan
        ):
            self.last_trace = CascadeTrace(
                time_bounded=False,
                time_precision=query_plan.time_precision,
                stop_reason="unbounded_rejected",
            )
            return ContextPackV1(question=query)
        safe_limit = max(1, min(int(limit), 30))
        items = await self._semantic_scout(
            query,
            plan=query_plan,
            existing=[],
            limit=safe_limit,
        )
        selected = tuple(items[:safe_limit])
        source_counts: dict[str, int] = {}
        for item in selected:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        return ContextPackV1(
            question=query,
            items=selected,
            sources=source_counts,
        )

    async def _semantic_scout(
        self,
        query: str,
        *,
        plan: TimeFirstQueryPlan,
        existing: list[ContextItemV1],
        limit: int,
    ) -> list[ContextItemV1]:
        if (
            limit <= 0
            or self.embeddings is None
            or self.qdrant is None
            or not self.embeddings.enabled
            or not self.qdrant.enabled
            or not self.embeddings.accepts_private_text()
            or not self.qdrant.accepts_private_data()
        ):
            if self.scout_mode in {"rrf5", "cascade_2_2_1"}:
                self.last_trace = CascadeTrace(
                    time_bounded=plan.used_time_filter,
                    time_precision=plan.time_precision,
                    stop_reason="provider_unavailable",
                )
            return []
        from ..embeddings import EmbeddingUnavailableError
        from ..qdrant_memory import QdrantMemoryError

        query_documents = memory_query_documents(query[:2000])
        if self.scout_mode == "rrf5":
            return await self._rrf5_scout(
                query_documents,
                query=query,
                plan=plan,
                existing=existing,
                limit=limit,
            )
        if self.scout_mode == "cascade_2_2_1":
            return await self._cascade_scout(
                query_documents,
                query=query,
                plan=plan,
                existing=existing,
                limit=limit,
            )
        axes = (
            "semantic",
            *self.preferred_axes,
            *(axis for axis in plan.preferred_axes if axis in MEMORY_VECTOR_NAMES),
            *_reserve_axes_for(query),
        )
        ordered_axes: list[str] = []
        for axis in axes:
            if axis in MEMORY_VECTOR_NAMES and axis not in ordered_axes:
                ordered_axes.append(axis)
        results: list[ContextItemV1] = []
        seen = {item.content_hash for item in existing}
        try:
            for axis in ordered_axes:
                document = query_documents.get(axis)
                if not document:
                    continue
                vectors = await self.embeddings.embed_queries([document])
                if len(vectors) != 1:
                    continue
                hits = await self.qdrant.search(
                    query_vectors={axis: vectors[0]},
                    vector_names=(axis,),
                    # Leave room for stale/orphaned candidates rejected by SQL checks.
                    limit=min(30, max(10, (limit - len(results)) * 3)),
                    min_score=0.0,
                )
                for hit in hits:
                    item = await self._verified_episode_item(hit, plan=plan, axis=axis)
                    if item is None:
                        continue
                    if item.content_hash in seen:
                        continue
                    seen.add(item.content_hash)
                    results.append(item)
                    if len(results) >= limit:
                        return results
                if len(results) >= min(self.min_exact_evidence, limit):
                    break
        except (EmbeddingUnavailableError, QdrantMemoryError):
            return []
        return results

    async def _rrf5_scout(
        self,
        query_documents: dict[str, str],
        *,
        query: str,
        plan: TimeFirstQueryPlan,
        existing: list[ContextItemV1],
        limit: int,
    ) -> list[ContextItemV1]:
        from ..embeddings import EmbeddingUnavailableError
        from ..qdrant_memory import QdrantMemoryError

        axes = tuple(
            axis for axis in MEMORY_VECTOR_NAMES if query_documents.get(axis)
        )
        self.last_trace = CascadeTrace(
            time_bounded=plan.used_time_filter,
            time_precision=plan.time_precision,
            waves=(axes,) if axes else (),
            axes_attempted=axes,
            stop_reason="in_progress",
        )
        results: list[ContextItemV1] = []
        rejected = 0
        seen = {item.content_hash for item in existing}
        stop_reason = "rrf_complete"
        queried: tuple[str, ...] = ()
        try:
            vectors = await self.embeddings.embed_queries(
                [query_documents[axis] for axis in axes]
            )
            if len(vectors) != len(axes):
                stop_reason = "embedding_count_mismatch"
            else:
                hits = await self.qdrant.search(
                    query_vectors=dict(zip(axes, vectors, strict=True)),
                    vector_names=axes,
                    query_weights=memory_query_weights(query),
                    limit=min(30, max(10, limit * 3)),
                    min_score=0.0,
                )
                queried = axes
                for hit in hits:
                    item = await self._verified_episode_item(
                        hit,
                        plan=plan,
                        axis="rrf5",
                    )
                    if item is None:
                        rejected += 1
                        continue
                    if item.content_hash in seen:
                        continue
                    seen.add(item.content_hash)
                    results.append(item)
                    if len(results) >= limit:
                        stop_reason = "limit_reached"
                        break
        except (EmbeddingUnavailableError, QdrantMemoryError):
            stop_reason = "provider_unavailable"
            results = []
        self.last_trace = CascadeTrace(
            time_bounded=plan.used_time_filter,
            time_precision=plan.time_precision,
            waves=(axes,) if axes else (),
            axes_attempted=axes,
            axes_queried=queried,
            stop_reason=stop_reason,
            candidate_count=len(results),
            verified_rejected=rejected,
        )
        return results

    def _cascade_waves_for(
        self,
        query: str,
        *,
        plan: TimeFirstQueryPlan | None = None,
    ) -> tuple[tuple[str, ...], ...]:
        ordered: list[str] = []
        plan_axes = plan.preferred_axes if plan is not None else ()
        for axis in (
            "semantic",
            *self.preferred_axes,
            *plan_axes,
            *_reserve_axes_for(query),
            *MEMORY_VECTOR_NAMES,
        ):
            if axis in MEMORY_VECTOR_NAMES and axis not in ordered:
                ordered.append(axis)
        selected = ordered[: self.qmax]
        return tuple(
            tuple(selected[index : index + self.wave_size])
            for index in range(0, len(selected), self.wave_size)
        )

    async def _cascade_scout(
        self,
        query_documents: dict[str, str],
        *,
        query: str,
        plan: TimeFirstQueryPlan,
        existing: list[ContextItemV1],
        limit: int,
    ) -> list[ContextItemV1]:
        from ..embeddings import EmbeddingUnavailableError
        from ..qdrant_memory import QdrantMemoryError

        waves = self._cascade_waves_for(query, plan=plan)
        results: list[ContextItemV1] = []
        seen = {item.content_hash for item in existing}
        attempted: list[str] = []
        queried: list[str] = []
        completed_waves: list[tuple[str, ...]] = []
        rejected = 0
        stop_reason = "axes_exhausted"
        try:
            for wave in waves:
                active_axes = tuple(
                    axis for axis in wave if query_documents.get(axis)
                )
                if not active_axes:
                    continue
                attempted.extend(active_axes)
                self.last_trace = CascadeTrace(
                    time_bounded=plan.used_time_filter,
                    time_precision=plan.time_precision,
                    waves=(*tuple(completed_waves), active_axes),
                    axes_attempted=tuple(attempted),
                    axes_queried=tuple(queried),
                    stop_reason="in_progress",
                    candidate_count=len(results),
                    verified_rejected=rejected,
                )
                documents = [query_documents[axis] for axis in active_axes]
                vectors = await self.embeddings.embed_queries(documents)
                if len(vectors) != len(active_axes):
                    stop_reason = "embedding_count_mismatch"
                    break
                searches = [
                    self.qdrant.search(
                        query_vectors={axis: vector},
                        vector_names=(axis,),
                        limit=min(30, max(10, (limit - len(results)) * 3)),
                        min_score=0.0,
                    )
                    for axis, vector in zip(active_axes, vectors, strict=True)
                ]
                hit_groups = await asyncio.gather(*searches)
                queried.extend(active_axes)
                completed_waves.append(active_axes)
                for axis, hits in zip(active_axes, hit_groups, strict=True):
                    for hit in hits:
                        item = await self._verified_episode_item(
                            hit,
                            plan=plan,
                            axis=axis,
                        )
                        if item is None:
                            rejected += 1
                            continue
                        if item.content_hash in seen:
                            continue
                        seen.add(item.content_hash)
                        results.append(item)
                        if len(results) >= limit:
                            stop_reason = "limit_reached"
                            break
                    if len(results) >= limit:
                        break
                if len(results) >= min(self.min_exact_evidence, limit):
                    if stop_reason != "limit_reached":
                        stop_reason = "enough_evidence"
                    break
        except (EmbeddingUnavailableError, QdrantMemoryError):
            stop_reason = "provider_unavailable"
            results = []
        self.last_trace = CascadeTrace(
            time_bounded=plan.used_time_filter,
            time_precision=plan.time_precision,
            waves=tuple(completed_waves),
            axes_attempted=tuple(attempted),
            axes_queried=tuple(queried),
            stop_reason=stop_reason,
            candidate_count=len(results),
            verified_rejected=rejected,
        )
        return results

    async def _verified_episode_item(
        self, hit, *, plan: TimeFirstQueryPlan, axis: str,
    ) -> ContextItemV1 | None:
        """Qdrant proposes an ID; SQL supplies current content, time and provenance.

        Legacy points without a canonical episode/version are deliberately not
        admitted to time-first recall. They remain on the legacy recall path.
        """
        metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
        episode_id = metadata.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            return None
        episode = await self.memory.get_context_episode(episode_id)
        now = datetime.now(UTC)
        if (
            episode is None
            or episode.deleted_at is not None
            or (episode.expires_at is not None and episode.expires_at <= now)
            or (episode.source, episode.source_id) != (hit.source, hit.source_id)
            or metadata.get("content_hash") != episode.content_hash
            or not _range_matches(episode.started_at, episode.ended_at, plan)
        ):
            return None
        source_hashes = metadata.get("source_event_hashes")
        if not isinstance(source_hashes, dict):
            return None
        for event_id in episode.source_event_ids:
            event = await self.memory.get_context_event(event_id)
            if (
                event is None
                or event.deleted_at is not None
                or (event.expires_at is not None and event.expires_at <= now)
                or source_hashes.get(event_id) != event.content_hash
            ):
                return None
        score = float(hit.score)
        if not math.isfinite(score):
            return None
        item = episode.as_context_item(selection_reason=f"semantic_scout:{axis}")
        # Preserve the canonical item's confidence; ranking is not probability.
        return item.model_copy(update={"retrieval_score": score})

    @staticmethod
    def _screenpipe_context_items(
        raw_items,
        *,
        existing: list[ContextItemV1],
        plan: TimeFirstQueryPlan,
    ) -> list[ContextItemV1]:
        seen = {item.content_hash for item in existing}
        results: list[ContextItemV1] = []
        for raw in raw_items:
            timestamp = _parse_timestamp(raw.timestamp)
            if timestamp is None or not _range_matches(timestamp, timestamp, plan):
                continue
            source_raw = "\n".join(
                (raw.timestamp, raw.app_name, raw.window_name, raw.text)
            )
            source_id = hashlib.sha256(source_raw.encode("utf-8")).hexdigest()
            event = ContextEventV1.from_observation(
                source="screenpipe",
                source_id=source_id,
                started_at=timestamp,
                event_type=raw.content_type or "screenpipe_text",
                app_name=raw.app_name,
                window_title=raw.window_name,
                text=raw.text,
                metadata={"browser_url": raw.browser_url},
            )
            item = event.as_context_item(selection_reason="screenpipe_time_search")
            if item.content_hash in seen:
                continue
            seen.add(item.content_hash)
            results.append(item)
        return results


_POLISH_ASCII_MAP = str.maketrans(
    {
        "ł": "l",
        "Ł": "l",
    }
)


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD", value.translate(_POLISH_ASCII_MAP).casefold()
    )
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _search_text(value: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[\w-]+", _normalized_text(value), flags=re.UNICODE)
        if len(token) > 1 and token not in _STOPWORDS and not token.isdigit()
    ]
    return " ".join(tokens[:12])


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _approximate_relative_window(
    normalized: str,
    *,
    local_now: datetime,
) -> tuple[datetime, datetime, str, float, tuple[str, ...]] | None:
    """Resolve approximate relative temporal expressions into bounded intervals."""
    epistemic = _has_epistemic_marker(normalized)
    preferred_axes = ("semantic", "topic") if epistemic else ()

    if re.search(r"\b(?:kilka|kilku|pare|paru)\s+dni\s+temu\b", normalized):
        start = local_now - timedelta(days=7)
        end = local_now - timedelta(days=2)
        return (
            start,
            end,
            "approximately_several_days_before_observation",
            0.6,
            preferred_axes,
        )

    if re.search(
        r"\b(?:kilka|kilku|pare|paru)\s+tygodni\s+temu\b",
        normalized,
    ):
        start = local_now - timedelta(days=35)
        end = local_now - timedelta(days=14)
        return (
            start,
            end,
            "approximately_several_weeks_before_observation",
            0.55,
            preferred_axes,
        )

    if re.search(
        r"\b(?:kilka|kilku|pare|paru)\s+(?:miesiec(?:y)?|miesiac(?:e|y)?)\s+temu\b",
        normalized,
    ):
        start = local_now - timedelta(days=120)
        end = local_now - timedelta(days=45)
        return (
            start,
            end,
            "approximately_several_months_before_observation",
            0.55,
            preferred_axes,
        )

    month_ago = re.search(
        r"\b(?:(\d{1,2})\s+)?(?:miesiac(?:e|y)?|miesiec(?:y)?)\s+temu\b"
        r"|\bw\s+(?:zeszlym|ostatnim|poprzednim)\s+miesiacu\b",
        normalized,
    )
    if month_ago:
        count = max(1, min(int(month_ago.group(1) or 1), 12))
        center_days = 30 * count
        half_window = 7 * count
        start = local_now - timedelta(days=center_days + half_window)
        end = local_now - timedelta(days=max(1, center_days - half_window))
        relation = (
            "approximately_one_month_before_observation"
            if count == 1
            else f"approximately_{count}_months_before_observation"
        )
        return start, end, relation, 0.65, preferred_axes

    week_ago = re.search(
        r"\b(?:(\d{1,2})\s+)?(?:tydzien|tygodni(?:e)?)\s+temu\b"
        r"|\bw\s+(?:zeszlym|ostatnim|poprzednim)\s+tygodniu\b",
        normalized,
    )
    if week_ago:
        count = max(1, min(int(week_ago.group(1) or 1), 52))
        center_days = 7 * count
        half_window = max(2, 2 * count)
        start = local_now - timedelta(days=center_days + half_window)
        end = local_now - timedelta(days=max(1, center_days - half_window))
        relation = (
            "approximately_one_week_before_observation"
            if count == 1
            else f"approximately_{count}_weeks_before_observation"
        )
        return start, end, relation, 0.7, preferred_axes

    return None


def _has_epistemic_marker(normalized: str) -> bool:
    if re.search(r"\bbedzi(?:e|emy|esz)\b", normalized):
        return True
    return any(marker in normalized for marker in ("mysl", "uwaz", "wydaje"))


def _reserve_axes_for(query: str) -> tuple[str, ...]:
    normalized = _normalized_text(query)
    axes: list[str] = []
    markers = (
        ("decision", ("ustal", "decyz", "zdecyd", "postanow")),
        ("intent", ("dlaczego", "po co", "cel", "zamiar")),
        ("person_context", ("kto", "komu", "kogo", "z kim", "osob")),
        ("topic", ("temat", "o czym", "projekt")),
    )
    for axis, values in markers:
        if any(value in normalized for value in values):
            axes.append(axis)
    return tuple(axes)


def _range_matches(start: datetime, end: datetime, plan: TimeFirstQueryPlan) -> bool:
    """Use the same interval-overlap semantics as SQL timeline retrieval."""
    return (plan.start is None or end >= plan.start) and (
        plan.end is None or start <= plan.end
    )


def _is_bounded_plan(plan: TimeFirstQueryPlan) -> bool:
    return bool(
        plan.used_time_filter
        and plan.start is not None
        and plan.end is not None
        and plan.start.tzinfo is not None
        and plan.start.utcoffset() is not None
        and plan.end.tzinfo is not None
        and plan.end.utcoffset() is not None
        and plan.start <= plan.end
    )
