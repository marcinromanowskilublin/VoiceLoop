from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from ..memory import MemoryStore
from ..screenpipe import ScreenpipeClient, ScreenpipeTextItem
from .episodes import group_events_into_episodes
from .schema import ContextEventV1


@dataclass(frozen=True)
class TimelineIngestReport:
    observed_items: int
    stored_events: int
    stored_episodes: int


class ScreenpipeTimelineIngestor:
    """Explicit, bounded Screenpipe-to-timeline adapter; it has no background loop."""

    def __init__(
        self,
        *,
        screenpipe: ScreenpipeClient,
        memory: MemoryStore,
        bucket_minutes: int = 10,
    ) -> None:
        self.screenpipe = screenpipe
        self.memory = memory
        self.bucket_minutes = max(1, min(int(bucket_minutes), 60))

    async def ingest_range(
        self,
        *,
        start: datetime,
        end: datetime,
        query: str | None = None,
        max_results: int = 500,
        build_episodes: bool = True,
    ) -> TimelineIngestReport:
        raw_items = await self.screenpipe.text_activity_between(
            start=start,
            end=end,
            query=query,
            max_results=max_results,
        )
        events = [self._event_from_item(item) for item in raw_items]
        stored_events = [
            await self.memory.upsert_context_event(event) for event in events
        ]
        stored_episode_count = 0
        if build_episodes and stored_events:
            episodes = group_events_into_episodes(
                stored_events,
                bucket_minutes=self.bucket_minutes,
                source="screenpipe_timeline",
            )
            for episode in episodes:
                await self.memory.upsert_context_episode(episode)
            stored_episode_count = len(episodes)
        return TimelineIngestReport(
            observed_items=len(raw_items),
            stored_events=len(stored_events),
            stored_episodes=stored_episode_count,
        )

    @staticmethod
    def _event_from_item(item: ScreenpipeTextItem) -> ContextEventV1:
        timestamp = _parse_timestamp(item.timestamp)
        identity = "\n".join(
            (item.timestamp, item.app_name, item.window_name, item.text)
        )
        source_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return ContextEventV1.from_observation(
            source="screenpipe",
            source_id=source_id,
            started_at=timestamp,
            event_type=item.content_type or "screenpipe_text",
            app_name=item.app_name,
            window_title=item.window_name,
            text=item.text,
            metadata={
                "browser_url": item.browser_url,
                "foreground_confidence": "unknown",
            },
        )


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(UTC)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
