from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..memory import MemoryStore
from ..qdrant_memory import QdrantVectorStore


@dataclass(frozen=True)
class ContextPruneReport:
    events: int
    episodes: int
    qdrant_deleted: int
    dry_run: bool


class ContextLifecycleService:
    """Coordinates fail-closed deletion across canonical SQL, FTS and Qdrant."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        qdrant: QdrantVectorStore | None = None,
    ) -> None:
        self.memory = memory
        self.qdrant = qdrant

    async def tombstone_episode(self, episode_id: str) -> bool:
        episode = await self.memory.get_context_episode(episode_id)
        if episode is None or episode.deleted_at is not None:
            return False
        if self.qdrant is not None and self.qdrant.enabled:
            await self.qdrant.delete_memory(
                source=episode.source,
                source_id=episode.source_id,
            )
        return await self.memory.tombstone_context_episode(episode_id)

    async def prune_expired(
        self,
        *,
        now: datetime | None = None,
        dry_run: bool = True,
    ) -> ContextPruneReport:
        cutoff = now or datetime.now(UTC)
        counts = await self.memory.prune_expired_context_records(
            now=cutoff,
            dry_run=True,
        )
        if dry_run:
            return ContextPruneReport(
                events=counts["events"],
                episodes=counts["episodes"],
                qdrant_deleted=0,
                dry_run=True,
            )
        expired_episodes = await self.memory.list_expired_context_episodes(
            now=cutoff,
            limit=max(1, counts["episodes"]),
        )
        qdrant_deleted = 0
        if self.qdrant is not None and self.qdrant.enabled:
            for episode in expired_episodes:
                await self.qdrant.delete_memory(
                    source=episode.source,
                    source_id=episode.source_id,
                )
                qdrant_deleted += 1
        applied = await self.memory.prune_expired_context_records(
            now=cutoff,
            dry_run=False,
        )
        return ContextPruneReport(
            events=applied["events"],
            episodes=applied["episodes"],
            qdrant_deleted=qdrant_deleted,
            dry_run=False,
        )
