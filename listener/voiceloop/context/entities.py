from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..corpus.schema import ProperNameEntryV1, ProperNameLexiconV1


class ContextEntityKind(StrEnum):
    PERSON = "person"
    PROJECT = "project"
    TOOL = "tool"
    ORGANIZATION = "org"


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    canonical: str
    kind: str
    matched_text: str
    confidence: float


class ContextEntityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    entity_id: str = Field(min_length=1, max_length=120)
    canonical: str = Field(min_length=1, max_length=300)
    normalized_canonical: str = Field(default="", max_length=300)
    kind: ContextEntityKind
    aliases: tuple[str, ...] = ()
    sensitivity: str = Field(default="private_personal", max_length=40)
    approved: bool = True
    evidence_source_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at", "updated_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("entity timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _normalize(self) -> ContextEntityV1:
        normalized = normalize_entity_text(self.canonical)
        if not self.normalized_canonical:
            self.normalized_canonical = normalized
        elif self.normalized_canonical != normalized:
            raise ValueError("normalized_canonical does not match canonical")
        expected_id = stable_entity_id(self.kind.value, self.canonical)
        if self.entity_id != expected_id:
            raise ValueError("entity_id does not match kind and canonical")
        return self

    @classmethod
    def from_approved_entry(cls, entry: ProperNameEntryV1) -> ContextEntityV1:
        known_kinds = {item.value for item in ContextEntityKind}
        kind = (
            ContextEntityKind(entry.category)
            if entry.category in known_kinds
            else ContextEntityKind.TOOL
        )
        return cls(
            entity_id=stable_entity_id(kind.value, entry.canonical),
            canonical=entry.canonical,
            kind=kind,
            aliases=tuple(
                dict.fromkeys((*entry.aliases, *entry.common_stt_errors))
            ),
            approved=entry.approved,
            evidence_source_ids=entry.evidence_sample_ids,
            metadata={"pronunciation_hint": entry.pronunciation_hint},
        )


class ContextEntityCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    candidate_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=120,
    )
    canonical: str = Field(min_length=1, max_length=300)
    normalized_canonical: str = Field(default="", max_length=300)
    kind: ContextEntityKind
    aliases: tuple[str, ...] = ()
    evidence_source_ids: tuple[str, ...] = Field(min_length=1)
    strong_evidence_count: int = Field(default=0, ge=0)
    status: Literal["pending", "approved", "rejected"] = "pending"
    sensitivity: str = Field(default="private_personal", max_length=40)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at", "updated_at")
    @classmethod
    def _aware_candidate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("candidate timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _normalize_candidate(self) -> ContextEntityCandidateV1:
        normalized = normalize_entity_text(self.canonical)
        if not self.normalized_canonical:
            self.normalized_canonical = normalized
        elif self.normalized_canonical != normalized:
            raise ValueError("normalized_canonical does not match canonical")
        return self

    @property
    def ready_for_review(self) -> bool:
        return len(set(self.evidence_source_ids)) >= 2 and self.strong_evidence_count >= 1


def stable_entity_id(kind: str, canonical: str) -> str:
    normalized = normalize_entity_text(canonical)
    return str(uuid5(NAMESPACE_URL, f"voiceloop-entity:{kind}:{normalized}"))


def normalize_entity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w -]+", " ", without_marks).split())


class ContextEntityResolver:
    """Approved exact-name resolver; it never silently merges candidate entities."""

    def __init__(self, lexicon: ProperNameLexiconV1 | None = None) -> None:
        self.lexicon = lexicon or ProperNameLexiconV1()
        self._variants: dict[str, ProperNameEntryV1] = {}
        for entry in self.lexicon.entries:
            if not entry.approved:
                continue
            for variant in (entry.canonical, *entry.aliases, *entry.common_stt_errors):
                normalized = normalize_entity_text(variant)
                if normalized:
                    self._variants[normalized] = entry

    def resolve(self, value: str) -> ResolvedEntity | None:
        normalized = normalize_entity_text(value)
        entry = self._variants.get(normalized)
        if entry is None:
            return None
        known_kinds = {item.value for item in ContextEntityKind}
        kind = entry.category if entry.category in known_kinds else "tool"
        return ResolvedEntity(
            entity_id=stable_entity_id(kind, entry.canonical),
            canonical=entry.canonical,
            kind=kind,
            matched_text=value,
            confidence=1.0,
        )

    def find_in_text(self, text: str) -> tuple[ResolvedEntity, ...]:
        normalized_text = f" {normalize_entity_text(text)} "
        matches: list[ResolvedEntity] = []
        seen: set[str] = set()
        for variant, entry in sorted(
            self._variants.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if f" {variant} " not in normalized_text:
                continue
            kind = (
                entry.category
                if entry.category in {item.value for item in ContextEntityKind}
                else "tool"
            )
            entity_id = stable_entity_id(kind, entry.canonical)
            if entity_id in seen:
                continue
            seen.add(entity_id)
            matches.append(
                ResolvedEntity(
                    entity_id=entity_id,
                    canonical=entry.canonical,
                    kind=kind,
                    matched_text=variant,
                    confidence=1.0,
                )
            )
        return tuple(matches)
