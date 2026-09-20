from __future__ import annotations

from ..corpus.schema import ProperNameLexiconV1
from ..memory import MemoryStore
from .entities import (
    ContextEntityCandidateV1,
    ContextEntityV1,
    ResolvedEntity,
    stable_entity_id,
)


class ContextEntityRegistry:
    """Durable reviewed entities; candidates never become canonical silently."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    async def import_approved_lexicon(
        self,
        lexicon: ProperNameLexiconV1,
    ) -> int:
        imported = 0
        for entry in lexicon.entries:
            if not entry.approved:
                continue
            await self.memory.upsert_context_entity(
                ContextEntityV1.from_approved_entry(entry)
            )
            imported += 1
        return imported

    async def submit_candidate(
        self,
        candidate: ContextEntityCandidateV1,
    ) -> ContextEntityCandidateV1:
        return await self.memory.upsert_context_entity_candidate(candidate)

    async def approve_candidate(
        self,
        candidate_id: str,
    ) -> ContextEntityV1:
        candidates = await self.memory.list_context_entity_candidates(
            status="pending",
            limit=1000,
        )
        candidate = next(
            (item for item in candidates if item.candidate_id == candidate_id),
            None,
        )
        if candidate is None:
            raise ValueError("unknown pending entity candidate")
        if not candidate.ready_for_review:
            raise ValueError(
                "entity candidate requires two independent sources and one strong signal"
            )
        entity = ContextEntityV1(
            entity_id=stable_entity_id(
                candidate.kind.value,
                candidate.canonical,
            ),
            canonical=candidate.canonical,
            kind=candidate.kind,
            aliases=candidate.aliases,
            sensitivity=candidate.sensitivity,
            approved=True,
            evidence_source_ids=candidate.evidence_source_ids,
        )
        stored = await self.memory.upsert_context_entity(entity)
        await self.memory.update_context_entity_candidate_status(
            candidate_id,
            status="approved",
        )
        return stored

    async def reject_candidate(self, candidate_id: str) -> bool:
        updated = await self.memory.update_context_entity_candidate_status(
            candidate_id,
            status="rejected",
        )
        return updated is not None

    async def resolve(
        self,
        value: str,
        *,
        kind: str | None = None,
    ) -> tuple[ResolvedEntity, ...]:
        entities = await self.memory.resolve_context_entities(value, kind=kind)
        return tuple(
            ResolvedEntity(
                entity_id=entity.entity_id,
                canonical=entity.canonical,
                kind=entity.kind.value,
                matched_text=value,
                confidence=1.0,
            )
            for entity in entities
        )
