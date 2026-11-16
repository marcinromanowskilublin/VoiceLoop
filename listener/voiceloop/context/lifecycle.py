from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from ..memory import MemoryStore
from ..qdrant_memory import QdrantVectorStore
from .schema import ContextIndexOperation


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
        tombstoned = await self.memory.tombstone_context_episode(episode_id)
        if tombstoned and self.qdrant is not None and self.qdrant.enabled:
            await self._delete_and_mark(episode)
        return tombstoned

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
        applied = await self.memory.prune_expired_context_records(
            now=cutoff,
            dry_run=False,
        )
        qdrant_deleted = 0
        if self.qdrant is not None and self.qdrant.enabled:
            for episode in expired_episodes:
                if await self._delete_and_mark(episode):
                    qdrant_deleted += 1
        return ContextPruneReport(
            events=applied["events"],
            episodes=applied["episodes"],
            qdrant_deleted=qdrant_deleted,
            dry_run=False,
        )

    async def _delete_and_mark(self, episode) -> bool:
        profile = "context-lifecycle-direct-delete-v1"
        state = await self.memory.get_context_index_state(episode.episode_id)
        if (
            state is None
            or state.desired_operation is not ContextIndexOperation.DELETE
        ):
            return False
        fingerprint = hashlib.sha256(
            (
                f"{profile}:{episode.episode_id}:"
                f"{episode.content_hash}:{state.desired_generation}"
            ).encode()
        ).hexdigest()
        claimed = await self.memory.claim_context_index_job(
            episode_id=episode.episode_id,
            operation=ContextIndexOperation.DELETE,
            desired_revision=episode.content_hash,
            generation=state.desired_generation,
            index_fingerprint=fingerprint,
            index_profile=profile,
        )
        if not claimed:
            return False
        try:
            removed = await self.qdrant.delete_memory_through_generation(
                source=episode.source,
                source_id=episode.source_id,
                max_generation=state.desired_generation,
            )
        except Exception as exc:
            await self.memory.mark_context_index_failed(
                episode_id=episode.episode_id,
                operation=ContextIndexOperation.DELETE,
                desired_revision=episode.content_hash,
                generation=state.desired_generation,
                claim_token=claimed,
                index_fingerprint=fingerprint,
                index_profile=profile,
                error_type=type(exc).__name__,
            )
            raise
        applied = await self.memory.mark_context_index_applied(
            episode_id=episode.episode_id,
            operation=ContextIndexOperation.DELETE,
            desired_revision=episode.content_hash,
            generation=state.desired_generation,
            claim_token=claimed,
            index_fingerprint=fingerprint,
            index_profile=profile,
        )
        return applied and removed
