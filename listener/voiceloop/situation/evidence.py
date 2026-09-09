"""EvidenceItemV1 — kontrakt dowodu sytuacji.

To nie jest `commitments.schema.EvidenceItem`.
To nie jest źródło `action_id` ani plan dla LLM.
Pamięć A/B/C pozostaje retrieval; ten typ tylko opisuje dowód.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from voiceloop.models import utc_now


class EvidenceSourceType(StrEnum):
    USER_EXPLICIT = "user_explicit"
    TRANSCRIPT = "transcript"
    TOOL_RESULT = "tool_result"
    SCREEN_OBSERVATION = "screen_observation"
    MEMORY_RETRIEVAL = "memory_retrieval"
    COMMITMENT_DETECTOR = "commitment_detector"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_FACT = "system_fact"


class EvidenceTrustClass(StrEnum):
    AUTHORITATIVE = "authoritative"
    USER_ASSERTED = "user_asserted"
    OBSERVED = "observed"
    DERIVED = "derived"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    MODEL_INFERENCE = "model_inference"


_INJECTION_MARKERS = (
    "system:",
    "ignore previous",
    "usuń wszystkie pliki",
    "usun wszystkie pliki",
    "delete all files",
    "rm -rf",
)


def content_hash_for(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def looks_like_untrusted_injection(content: str) -> bool:
    normalized = " ".join((content or "").casefold().split())
    return any(marker in normalized for marker in _INJECTION_MARKERS)


class EvidenceItemV1(BaseModel):
    """Dowód sytuacji. Nie jest planem i nie wiąże akcji."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120)
    source_type: EvidenceSourceType
    source_id: str = Field(min_length=1, max_length=240)
    captured_at: datetime
    content: str = Field(min_length=1, max_length=8000)
    speaker: str | None = Field(default=None, max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    trust_class: EvidenceTrustClass
    scope: str = Field(min_length=1, max_length=120)
    expires_at: datetime | None = None
    content_hash: str = Field(default="", max_length=64)

    @field_validator("captured_at", "expires_at")
    @classmethod
    def _require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_provenance_and_hash(self) -> EvidenceItemV1:
        if not self.source_id.strip():
            raise ValueError("missing provenance: source_id")
        if not self.source_type:
            raise ValueError("missing provenance: source_type")
        if self.expires_at is not None and self.expires_at < self.captured_at:
            raise ValueError("expires_at precedes captured_at")
        expected = content_hash_for(self.content)
        if not self.content_hash:
            self.content_hash = expected
        elif self.content_hash != expected:
            raise ValueError("content_hash does not match content")
        if (
            self.source_type is EvidenceSourceType.MODEL_INFERENCE
            and self.trust_class is EvidenceTrustClass.AUTHORITATIVE
        ):
            raise ValueError("model_inference cannot be authoritative")
        return self


def evidence_from_memory_retrieval(
    *,
    content: str,
    source_id: str,
    scope: str = "memory_retrieval",
    speaker: str | None = None,
    confidence: float = 0.4,
    captured_at: datetime | None = None,
) -> EvidenceItemV1:
    """Opisz trafienie pamięci jako dowód — nigdy jako plan akcji."""

    trust = (
        EvidenceTrustClass.UNTRUSTED_EXTERNAL
        if looks_like_untrusted_injection(content)
        else EvidenceTrustClass.DERIVED
    )
    return EvidenceItemV1(
        source_type=EvidenceSourceType.MEMORY_RETRIEVAL,
        source_id=source_id,
        captured_at=captured_at or utc_now(),
        content=content,
        speaker=speaker,
        confidence=confidence,
        trust_class=trust,
        scope=scope,
    )


def evidence_has_action_plan(item: EvidenceItemV1) -> bool:
    """Evidence nie jest ActionPlan / CommandPlan. Zawsze False."""

    del item
    return False


def evidence_action_ids(item: EvidenceItemV1) -> list[str]:
    """Brak wiązania akcji. Puste nawet gdy treść wygląda na polecenie."""

    del item
    return []
