from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5

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
