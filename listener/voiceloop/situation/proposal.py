"""StateProposal V1 — LLM może proponować, lokalny kod zapisuje.

Ścieżka (symetria z CommandPlan → ActionPolicy → Executor):
    StateProposal → schema → StatePolicy → StateReducer → append_event

Wołane tylko z testów / przyszłego shadow. Planer tego nie importuje.
LLM nie jest aktorem zapisu. Nie ma nowej akcji w ActionRegistry.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from voiceloop.models import utc_now
from voiceloop.situation.evidence import EvidenceItemV1, EvidenceSourceType, EvidenceTrustClass
from voiceloop.situation.state import (
    SituationActor,
    SituationEvent,
    SituationKind,
    SituationStore,
)

_FACT_TRUST = {
    EvidenceTrustClass.AUTHORITATIVE,
    EvidenceTrustClass.USER_ASSERTED,
    EvidenceTrustClass.OBSERVED,
}
_ACCEPTANCE_TRUST = {
    EvidenceTrustClass.AUTHORITATIVE,
    EvidenceTrustClass.USER_ASSERTED,
}
_MODEL_ONLY = {
    EvidenceTrustClass.MODEL_INFERENCE,
    EvidenceTrustClass.DERIVED,
}
_GATED_PROMOTE = {
    (SituationKind.HYPOTHESIS, SituationKind.FACT),
    (SituationKind.REQUEST, SituationKind.COMMITMENT),
    (SituationKind.INTENTION, SituationKind.DECISION),
}


class StateProposalOp(StrEnum):
    PROPOSE_FACT = "propose_fact"
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    PROPOSE_REQUEST = "propose_request"
    PROPOSE_COMMITMENT = "propose_commitment"
    PROPOSE_INTENTION = "propose_intention"
    PROPOSE_DECISION = "propose_decision"
    PROMOTE = "promote"
    CLAIM_ACTION_SUCCEEDED = "claim_action_succeeded"


_OP_TO_KIND = {
    StateProposalOp.PROPOSE_FACT: SituationKind.FACT,
    StateProposalOp.PROPOSE_HYPOTHESIS: SituationKind.HYPOTHESIS,
    StateProposalOp.PROPOSE_REQUEST: SituationKind.REQUEST,
    StateProposalOp.PROPOSE_COMMITMENT: SituationKind.COMMITMENT,
    StateProposalOp.PROPOSE_INTENTION: SituationKind.INTENTION,
    StateProposalOp.PROPOSE_DECISION: SituationKind.DECISION,
}


class StateProposal(BaseModel):
    """Propozycja modelu. Nie jest stanem i nie jest CommandPlan."""

    model_config = ConfigDict(extra="forbid")

    op: StateProposalOp
    content: str = Field(min_length=1, max_length=1000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    from_kind: SituationKind | None = None
    to_kind: SituationKind | None = None
    target_item_id: str | None = Field(default=None, max_length=120)

    @field_validator("evidence_refs")
    @classmethod
    def _refs_are_nonempty(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for ref in value:
            item = ref.strip()
            if not item:
                raise ValueError("evidence_ref must be non-empty")
            cleaned.append(item)
        return cleaned

    @model_validator(mode="after")
    def _promote_requires_kinds(self) -> StateProposal:
        if self.op is StateProposalOp.PROMOTE:
            if self.from_kind is None or self.to_kind is None:
                raise ValueError("promote requires from_kind and to_kind")
            if self.from_kind is self.to_kind:
                raise ValueError("promote requires a kind change")
        return self


class StatePolicyVerdict(BaseModel):
    accepted: bool
    reason: str
    kind: SituationKind | None = None
    evidence: EvidenceItemV1 | None = None


class StateProposalResult(BaseModel):
    accepted: bool
    reason: str
    event: SituationEvent | None = None
    op: StateProposalOp


def _index_store_evidence(store: SituationStore | None) -> dict[str, EvidenceItemV1]:
    if store is None:
        return {}
    return {event.evidence.evidence_id: event.evidence for event in store.events}


def _resolve_evidence(
    proposal: StateProposal,
    evidence_index: Mapping[str, EvidenceItemV1],
    store: SituationStore | None,
) -> tuple[list[EvidenceItemV1], tuple[str, ...]]:
    catalog = {**_index_store_evidence(store), **dict(evidence_index)}
    resolved: list[EvidenceItemV1] = []
    missing: list[str] = []
    for ref in proposal.evidence_refs:
        item = catalog.get(ref)
        if item is None:
            missing.append(ref)
        else:
            resolved.append(item)
    return resolved, tuple(missing)


def _target_event(
    store: SituationStore | None,
    target_item_id: str | None,
) -> SituationEvent | None:
    if store is None or not target_item_id:
        return None
    for event in store.events:
        if event.event_id == target_item_id:
            return event
    return None


def _best_evidence(
    resolved: list[EvidenceItemV1],
    allowed: set[EvidenceTrustClass],
) -> EvidenceItemV1 | None:
    for item in resolved:
        if item.trust_class in allowed:
            return item
    return None


def _all_model_authored(resolved: list[EvidenceItemV1]) -> bool:
    if not resolved:
        return True
    return all(
        item.trust_class in _MODEL_ONLY
        or item.source_type is EvidenceSourceType.MODEL_INFERENCE
        for item in resolved
    )


class StatePolicy:
    """Lokalna polityka stanu. Werdykt modelu nie obniża wymagań dowodu."""

    def enforce(
        self,
        proposal: StateProposal,
        evidence_index: Mapping[str, EvidenceItemV1],
        store: SituationStore | None = None,
    ) -> StatePolicyVerdict:
        if proposal.op is StateProposalOp.CLAIM_ACTION_SUCCEEDED:
            return StatePolicyVerdict(
                accepted=False,
                reason="action success must originate from executor, not LLM text",
            )

        resolved, missing = _resolve_evidence(proposal, evidence_index, store)
        if missing:
            return StatePolicyVerdict(
                accepted=False,
                reason=f"unknown evidence_refs: {', '.join(missing)}",
            )
        if not resolved:
            return StatePolicyVerdict(
                accepted=False,
                reason="state mutation requires provenance",
            )

        if proposal.op is StateProposalOp.PROMOTE:
            return self._enforce_promote(proposal, resolved, store)

        kind = _OP_TO_KIND[proposal.op]
        return self._enforce_kind(kind, resolved)

    def _enforce_promote(
        self,
        proposal: StateProposal,
        resolved: list[EvidenceItemV1],
        store: SituationStore | None,
    ) -> StatePolicyVerdict:
        from_kind = proposal.from_kind
        to_kind = proposal.to_kind
        assert from_kind is not None and to_kind is not None
        pair = (from_kind, to_kind)
        if pair not in _GATED_PROMOTE:
            return StatePolicyVerdict(
                accepted=False,
                reason=f"forbidden promote {from_kind.value} → {to_kind.value}",
            )

        if proposal.target_item_id:
            target = _target_event(store, proposal.target_item_id)
            if target is None:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="unknown target_item_id",
                )
            if target.kind is not from_kind:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="target kind mismatch",
                )

        if pair == (SituationKind.HYPOTHESIS, SituationKind.FACT):
            evidence = _best_evidence(resolved, _FACT_TRUST)
            if evidence is None:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="hypothesis cannot become fact without suitable evidence",
                )
            return StatePolicyVerdict(
                accepted=True,
                reason="local policy accepted fact from new evidence",
                kind=SituationKind.FACT,
                evidence=evidence,
            )

        if pair == (SituationKind.REQUEST, SituationKind.COMMITMENT):
            if _all_model_authored(resolved):
                return StatePolicyVerdict(
                    accepted=False,
                    reason="request cannot become accepted commitment because the model said so",
                )
            evidence = _best_evidence(resolved, _ACCEPTANCE_TRUST)
            if evidence is None or evidence.source_type is EvidenceSourceType.MODEL_INFERENCE:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="request cannot become accepted commitment because the model said so",
                )
            return StatePolicyVerdict(
                accepted=True,
                reason="local policy accepted commitment from user/authoritative evidence",
                kind=SituationKind.COMMITMENT,
                evidence=evidence,
            )

        if pair == (SituationKind.INTENTION, SituationKind.DECISION):
            evidence = _best_evidence(resolved, _FACT_TRUST)
            if evidence is None or _all_model_authored(resolved):
                return StatePolicyVerdict(
                    accepted=False,
                    reason="intention cannot become decision from model text alone",
                )
            return StatePolicyVerdict(
                accepted=True,
                reason="local policy accepted decision from observed evidence",
                kind=SituationKind.DECISION,
                evidence=evidence,
            )

        return StatePolicyVerdict(
            accepted=False,
            reason=f"forbidden promote {from_kind.value} → {to_kind.value}",
        )

    def _enforce_kind(
        self,
        kind: SituationKind,
        resolved: list[EvidenceItemV1],
    ) -> StatePolicyVerdict:
        if kind is SituationKind.FACT:
            evidence = _best_evidence(resolved, _FACT_TRUST)
            if evidence is None:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="FACT requires authoritative, user_asserted, or observed evidence",
                )
            return StatePolicyVerdict(
                accepted=True,
                reason="local policy accepted fact",
                kind=kind,
                evidence=evidence,
            )

        if kind is SituationKind.COMMITMENT:
            if _all_model_authored(resolved):
                return StatePolicyVerdict(
                    accepted=False,
                    reason="request cannot become accepted commitment because the model said so",
                )
            evidence = _best_evidence(resolved, _ACCEPTANCE_TRUST)
            if evidence is None or evidence.source_type is EvidenceSourceType.MODEL_INFERENCE:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="COMMITMENT requires user_asserted or authoritative evidence",
                )
            return StatePolicyVerdict(
                accepted=True,
                reason="local policy accepted commitment",
                kind=kind,
                evidence=evidence,
            )

        if kind is SituationKind.DECISION:
            evidence = _best_evidence(resolved, _FACT_TRUST)
            if evidence is None:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="DECISION requires authoritative, user_asserted, or observed evidence",
                )
            return StatePolicyVerdict(
                accepted=True,
                reason="local policy accepted decision",
                kind=kind,
                evidence=evidence,
            )

        # hypothesis / request / intention: nie awansują rodzaju; model_inference jest OK.
        if any(item.trust_class is EvidenceTrustClass.UNTRUSTED_EXTERNAL for item in resolved):
            if kind is SituationKind.HYPOTHESIS:
                return StatePolicyVerdict(
                    accepted=False,
                    reason="untrusted evidence cannot become durable hypothesis via LLM",
                )
        return StatePolicyVerdict(
            accepted=True,
            reason=f"local policy accepted {kind.value}",
            kind=kind,
            evidence=resolved[0],
        )


class StateReducer:
    """Deterministyczna redukcja zaakceptowanej propozycji do zdarzenia local_code."""

    def reduce(self, proposal: StateProposal, verdict: StatePolicyVerdict) -> SituationEvent:
        if not verdict.accepted or verdict.kind is None or verdict.evidence is None:
            raise ValueError("refusing to reduce a rejected StateProposal")
        return SituationEvent(
            occurred_at=utc_now(),
            kind=verdict.kind,
            summary=proposal.content,
            evidence=verdict.evidence,
            actor=SituationActor.LOCAL_CODE,
        )


def apply_state_proposal(
    store: SituationStore,
    proposal: StateProposal | dict[str, Any],
    evidence_index: Mapping[str, EvidenceItemV1] | None = None,
) -> StateProposalResult:
    """Schema → StatePolicy → StateReducer → SituationStore.append_event.

    Nie wołać z planera. Trzeci argument to lokalny katalog dowodów, nie treść LLM.
    """

    validated = (
        proposal
        if isinstance(proposal, StateProposal)
        else StateProposal.model_validate(proposal)
    )
    policy = StatePolicy()
    reducer = StateReducer()
    verdict = policy.enforce(validated, evidence_index or {}, store)
    if not verdict.accepted:
        return StateProposalResult(
            accepted=False,
            reason=verdict.reason,
            event=None,
            op=validated.op,
        )
    event = reducer.reduce(validated, verdict)
    stored = store.append_event(event)
    return StateProposalResult(
        accepted=True,
        reason=verdict.reason,
        event=stored,
        op=validated.op,
    )
