"""Explicit projections into the timeline. They do not run on startup."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ..corpus.privacy import redact_text
from ..memory import MemoryStore
from .schema import ContextEpisodeV1, ContextEventV1, stable_episode_id

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from ..models import MemoryItem
    from ..windows_context import LocalPathRecord


@dataclass(frozen=True)
class ProjectionReport:
    stored_events: int
    stored_episodes: int
    skipped: int


async def project_windows_projects(
    memory: MemoryStore,
    records: Sequence[LocalPathRecord],
    *,
    observed_at: datetime | None = None,
    ttl_days: int = 14,
) -> ProjectionReport:
    """Store project names as timeline metadata. Paths stay out of event text."""

    moment = _aware(observed_at or datetime.now(UTC))
    safe_ttl = max(0, min(int(ttl_days), 3650))
    expires_at = moment + timedelta(days=safe_ttl) if safe_ttl else None
    stored = 0
    skipped = 0
    for record in records:
        name = (record.name or "").strip()
        if not record.is_project or not record.exists or not name or not record.path:
            skipped += 1
            continue
        source_id = hashlib.sha256(record.path.encode("utf-8")).hexdigest()
        # `first_seen` keeps the timeline honest across repeated reconciliations;
        # using "now" would push every known project to the present on each pass.
        first_seen = min(_aware(record.first_seen), moment)
        event = ContextEventV1.from_observation(
            source="windows_context",
            source_id=source_id,
            started_at=first_seen,
            event_type="local_project",
            app_name="windows",
            window_title=name,
            text=f"Projekt lokalny: {name} ({record.project_type or 'nieznany typ'}).",
            metadata={
                "path_sha256": source_id,
                "project_type": record.project_type,
                "last_seen": moment.isoformat(),
                "vectorize": False,
            },
            expires_at=expires_at,
        )
        await memory.upsert_context_event(event)
        stored += 1
    return ProjectionReport(stored_events=stored, stored_episodes=0, skipped=skipped)


class MemoryTimelineMigrator:
    """Copies existing SQL memories into timeline rows. Idempotent and explicit."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    async def migrate(self, *, limit: int = 200) -> ProjectionReport:
        items = await self.memory.list_memories(limit=limit)
        stored_events = 0
        stored_episodes = 0
        skipped = 0
        for item in items:
            event = _memory_event(item)
            if event is None:
                skipped += 1
                continue
            stored = await self.memory.upsert_context_event(event)
            episode = ContextEpisodeV1(
                episode_id=stable_episode_id("manual_memory", f"memory:{item.id}"),
                source="manual_memory",
                source_id=f"memory:{item.id}",
                started_at=stored.started_at,
                ended_at=stored.started_at,
                title=f"Pamięć {item.kind}",
                summary=stored.text,
                source_event_ids=(stored.event_id,),
                metadata={
                    "memory_id": item.id,
                    "memory_kind": item.kind,
                    "migration": "manual_memory_v1",
                    "vectorize": False,
                },
                sensitivity=item.sensitivity or "private",
            )
            await self.memory.upsert_context_episode(episode)
            stored_events += 1
            stored_episodes += 1
        return ProjectionReport(
            stored_events=stored_events,
            stored_episodes=stored_episodes,
            skipped=skipped,
        )


def _memory_event(item: MemoryItem) -> ContextEventV1 | None:
    """Explicit memories keep their creation time and never expire."""

    redacted, flags = redact_text(item.content)
    if not redacted.strip() or "secret" in flags or "pesel" in flags:
        return None
    return ContextEventV1.from_observation(
        source="manual_memory",
        source_id=str(item.id),
        started_at=_aware(item.created_at),
        event_type="user_memory",
        text=redacted[:20000],
        metadata={
            "memory_id": item.id,
            "memory_kind": item.kind,
            "memory_source": item.source,
            "redaction_flags": flags,
            "vectorize": False,
        },
        sensitivity=item.sensitivity or "private",
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
