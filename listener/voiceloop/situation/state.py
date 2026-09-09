"""SituationState V1 — ledger zdarzeń i redukcja read-only.

Planer nie czyta tego stanu jako źródła akcji.
LLM nie zapisuje stanu. StateProposal (ETAP 4) idzie przez lokalną politykę
i reducer; aktor zapisu pozostaje local_code.
Brak tabeli SQL `situation` — ledger jest procesowy.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from voiceloop.models import utc_now
from voiceloop.situation.evidence import EvidenceItemV1, EvidenceTrustClass

_FACT_TRUST = {
    EvidenceTrustClass.AUTHORITATIVE,
    EvidenceTrustClass.USER_ASSERTED,
    EvidenceTrustClass.OBSERVED,
}


class SituationKind(StrEnum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    REQUEST = "request"
    COMMITMENT = "commitment"
    INTENTION = "intention"
    DECISION = "decision"


class SituationActor(StrEnum):
    LOCAL_CODE = "local_code"


class SituationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120)
    occurred_at: datetime
    kind: SituationKind
    summary: str = Field(min_length=1, max_length=1000)
    evidence: EvidenceItemV1
    actor: SituationActor = SituationActor.LOCAL_CODE

    @field_validator("occurred_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _kind_requires_provenance(self) -> SituationEvent:
        if self.actor is not SituationActor.LOCAL_CODE:
            raise ValueError("only local_code may append situation events")
        if self.kind is SituationKind.FACT and self.evidence.trust_class not in _FACT_TRUST:
            raise ValueError("FACT requires authoritative, user_asserted, or observed evidence")
        if (
            self.kind is SituationKind.COMMITMENT
            and self.evidence.trust_class is EvidenceTrustClass.UNTRUSTED_EXTERNAL
        ):
            raise ValueError("untrusted evidence cannot become COMMITMENT")
        return self


class SituationItemV1(BaseModel):
    item_id: str
    kind: SituationKind
    summary: str
    evidence_id: str
    created_at: datetime


class SituationStateV1(BaseModel):
    schema_version: Literal[1] = 1
    generated_at: datetime
    items: tuple[SituationItemV1, ...] = ()
    event_count: int = 0


def reduce_events(events: tuple[SituationEvent, ...] | list[SituationEvent]) -> SituationStateV1:
    """Czysta redukcja: jedno zdarzenie → jeden item. Bez UPDATE fact."""

    items = tuple(
        SituationItemV1(
            item_id=event.event_id,
            kind=event.kind,
            summary=event.summary,
            evidence_id=event.evidence.evidence_id,
            created_at=event.occurred_at,
        )
        for event in events
    )
    return SituationStateV1(
        generated_at=utc_now(),
        items=items,
        event_count=len(items),
    )


def situation_action_ids(state: SituationStateV1) -> list[str]:
    """Stan nie jest planem. Zero action_id."""

    del state
    return []


class SituationStore:
    """Append-only ledger. Brak CRUD UPDATE. Brak zapisu z aktorem llm."""

    def __init__(self) -> None:
        self._events: list[SituationEvent] = []

    def append_event(self, event: SituationEvent) -> SituationEvent:
        if event.actor is not SituationActor.LOCAL_CODE:
            raise ValueError("LLM cannot write SituationState")
        self._events.append(event)
        return event

    def snapshot(self) -> SituationStateV1:
        return reduce_events(self._events)

    @property
    def events(self) -> tuple[SituationEvent, ...]:
        return tuple(self._events)
