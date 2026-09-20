from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ..memory_vectorization import memory_query_documents
from .schema import (
    ContextEventV1,
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
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
    "do",
    "dzis",
    "dzisiaj",
    "i",
    "jak",
    "mi",
    "na",
    "o",
    "od",
    "ostatnie",
    "ostatnich",
    "przypomnij",
    "temu",
    "w",
    "wczoraj",
    "z",
    "ze",
}


class TimeFirstQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str = Field(min_length=1, max_length=8000)
    search_text: str = Field(default="", max_length=1000)
    start: datetime | None = None
    end: datetime | None = None
    used_time_filter: bool = False


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
    if "przedwczoraj" in normalized:
        day = local_now.date() - timedelta(days=2)
        start = datetime.combine(day, time.min, tzinfo=local_now.tzinfo)
        end = datetime.combine(day, time.max, tzinfo=local_now.tzinfo)
    elif "wczoraj" in normalized:
        day = local_now.date() - timedelta(days=1)
        start = datetime.combine(day, time.min, tzinfo=local_now.tzinfo)
        end = datetime.combine(day, time.max, tzinfo=local_now.tzinfo)
    elif re.search(r"\b(?:dzis|dzisiaj)\b", normalized):
        start = datetime.combine(local_now.date(), time.min, tzinfo=local_now.tzinfo)
        end = local_now
    else:
        recent = re.search(
            r"\b(?:ostatnie|ostatnich)\s+(\d{1,3})\s+"
            r"(godzin(?:y|ach)?|dni|dnia|dniach)\b",
            normalized,
        )
        if recent:
            amount = max(1, min(int(recent.group(1)), 365))
            delta = (
                timedelta(hours=amount)
                if recent.group(2).startswith("godzin")
                else timedelta(days=amount)
            )
            start = local_now - delta
            end = local_now
    return TimeFirstQueryPlan(
        original_query=query,
        search_text=_search_text(query),
        start=start.astimezone(UTC) if start else None,
        end=end.astimezone(UTC) if end else None,
        used_time_filter=start is not None,
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
    ) -> None:
        self.memory = memory
        self.screenpipe = screenpipe
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.min_exact_evidence = max(1, min(int(min_exact_evidence), 10))

    async def retrieve(
        self,
        query: str,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
        limit: int = 8,
    ) -> ContextPackV1:
        plan = build_time_first_query(query, now=now)
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
                self._screenpipe_context_items(raw_items, existing=items)
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
            return []
        from ..embeddings import EmbeddingUnavailableError
        from ..qdrant_memory import QdrantMemoryError

        query_documents = memory_query_documents(query[:2000])
        axes = ("semantic", *_reserve_axes_for(query))
        results: list[ContextItemV1] = []
        seen = {item.content_hash for item in existing}
        try:
            for axis in axes:
                document = query_documents.get(axis)
                if not document:
                    continue
                vectors = await self.embeddings.embed_queries([document])
                if len(vectors) != 1:
                    continue
                hits = await self.qdrant.search(
                    query_vectors={axis: vectors[0]},
                    vector_names=(axis,),
                    limit=max(1, limit - len(results)),
                    min_score=0.0,
                )
                for hit in hits:
                    if not _hit_matches_time(hit, plan):
                        continue
                    item = ContextItemV1(
                        source=hit.source,
                        source_id=hit.source_id,
                        scope=ContextScope.EPISODIC,
                        kind="semantic_retrieval",
                        title=hit.title,
                        content=hit.content,
                        started_at=_hit_time(hit),
                        trust=ContextTrust.DERIVED,
                        confidence=min(max(float(hit.score), 0.0), 1.0),
                        retrieval_score=float(hit.score),
                        selection_reason=f"semantic_scout:{axis}",
                        metadata=(
                            hit.metadata if isinstance(hit.metadata, dict) else {}
                        ),
                    )
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

    @staticmethod
    def _screenpipe_context_items(
        raw_items,
        *,
        existing: list[ContextItemV1],
    ) -> list[ContextItemV1]:
        seen = {item.content_hash for item in existing}
        results: list[ContextItemV1] = []
        for raw in raw_items:
            timestamp = _parse_timestamp(raw.timestamp)
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


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _search_text(value: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[\w-]+", _normalized_text(value), flags=re.UNICODE)
        if len(token) > 1 and token not in _STOPWORDS and not token.isdigit()
    ]
    return " ".join(tokens[:12])


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(UTC)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


def _hit_source_time(hit) -> datetime | None:
    metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
    provenance = metadata.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    raw = (
        provenance.get("time")
        or metadata.get("time")
        or metadata.get("timestamp")
    )
    if raw is None or not str(raw).strip():
        return None
    return _parse_timestamp(str(raw))


def _hit_time(hit) -> datetime:
    source_time = _hit_source_time(hit)
    if source_time is not None:
        return source_time
    created_at = getattr(hit, "created_at", None)
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            return created_at.replace(tzinfo=UTC)
        return created_at.astimezone(UTC)
    return datetime.now(UTC)


def _hit_matches_time(hit, plan: TimeFirstQueryPlan) -> bool:
    if plan.start is None and plan.end is None:
        return True
    source_time = _hit_source_time(hit)
    if source_time is None:
        return False
    if plan.start is not None and source_time < plan.start:
        return False
    return plan.end is None or source_time <= plan.end
