from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .retrieval import (
    TimeFirstQueryPlan,
    build_time_first_query,
    normalize_context_text,
)


class AtomicClaimProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=1000)
    evidence_span: str = Field(min_length=1, max_length=1000)
    role: Literal["prediction", "opinion"]


class TemporalRoleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["utterance", "content"]
    start: datetime | None = None
    end: datetime | None = None
    relation: str
    precision: Literal["approximate", "unknown"]
    source: Literal["relative_text", "semantic_relation"]
    confidence: float = Field(ge=0.0, le=1.0)


class TemporalStatementShadowV1(BaseModel):
    """Transient controller result; raw text stays in the command provenance row."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1, max_length=8000)
    provenance_request_id: str = Field(min_length=1, max_length=120)
    observed_at: datetime
    utterance_time: TemporalRoleV1
    content_time: TemporalRoleV1
    epistemic_role: Literal["prediction", "opinion"]
    relevance_gate: Literal["context_only", "candidate", "investigate"]
    preferred_axes: tuple[Literal["semantic", "topic"], ...] = (
        "semantic",
        "topic",
    )
    approved_claims: tuple[AtomicClaimProposal, ...] = ()
    gap: str = "compound_claim_requires_review"
    axis_insufficiency: tuple[str, ...] = ("claim_axis_not_available",)

    def query_plan(self) -> TimeFirstQueryPlan:
        """Reuse the canonical parser, overriding only reviewed temporal roles."""

        plan = build_time_first_query(self.raw_text, now=self.observed_at)
        return plan.model_copy(
            update={
                "start": self.utterance_time.start,
                "end": self.utterance_time.end,
                "used_time_filter": (
                    self.utterance_time.start is not None
                    and self.utterance_time.end is not None
                ),
                "time_precision": self.utterance_time.precision,
                "time_relation": self.utterance_time.relation,
                "time_source": self.utterance_time.source,
                "time_confidence": self.utterance_time.confidence,
                "preferred_axes": tuple(self.preferred_axes),
            }
        )

    def telemetry(self) -> dict[str, object]:
        """Return content-free fields safe for local shadow telemetry."""

        return {
            "provenance_id": hashlib.sha256(
                self.provenance_request_id.encode("utf-8")
            ).hexdigest()[:24],
            "epistemic_role": self.epistemic_role,
            "relevance_gate": self.relevance_gate,
            "utterance_precision": self.utterance_time.precision,
            "utterance_source": self.utterance_time.source,
            "utterance_relation": self.utterance_time.relation,
            "content_precision": self.content_time.precision,
            "content_relation": self.content_time.relation,
            "approved_claim_count": len(self.approved_claims),
            "gap_present": bool(self.gap),
            "axis_insufficiency": list(self.axis_insufficiency),
        }


def analyze_temporal_statement_shadow(
    text: str,
    *,
    request_id: str,
    now: datetime | None = None,
    comparative_question: bool = False,
    conflict: bool = False,
    decision_context: bool = False,
    action_impact: bool = False,
    llm_proposals: tuple[AtomicClaimProposal, ...] = (),
) -> TemporalStatementShadowV1 | None:
    """Recognize approximate compound temporal statements without expanding live facts."""

    observed_at = now or datetime.now().astimezone()
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    observed_utc = observed_at.astimezone(UTC)

    normalized = normalize_context_text(text)
    plan = build_time_first_query(text, now=observed_utc)
    if (
        not plan.used_time_filter
        or plan.time_precision != "approximate"
        or plan.start is None
        or plan.end is None
    ):
        return None
    epistemic_role: Literal["prediction", "opinion"]
    if re.search(r"\bbedzi(?:e|emy|esz)\b", normalized):
        epistemic_role = "prediction"
    elif any(marker in normalized for marker in ("mysl", "uwaz", "wydaje")):
        epistemic_role = "opinion"
    else:
        return None

    relevance_gate = shadow_relevance_gate(
        has_epistemic_content=True,
        comparative_question=comparative_question,
        conflict=conflict,
        decision_context=decision_context,
        action_impact=action_impact,
    )
    approved = tuple(
        proposal
        for proposal in llm_proposals[:3]
        if proposal.role == epistemic_role
        and normalize_context_text(proposal.evidence_span) in normalized
    )
    content_relation = (
        "future_relative_to_utterance"
        if epistemic_role == "prediction"
        else "contemporaneous_with_utterance"
    )
    return TemporalStatementShadowV1(
        raw_text=text,
        provenance_request_id=request_id,
        observed_at=observed_utc,
        utterance_time=TemporalRoleV1(
            role="utterance",
            start=plan.start,
            end=plan.end,
            relation=plan.time_relation,
            precision="approximate",
            source="relative_text",
            confidence=plan.time_confidence or 0.65,
        ),
        content_time=TemporalRoleV1(
            role="content",
            relation=content_relation,
            precision="unknown",
            source="semantic_relation",
            confidence=0.55,
        ),
        epistemic_role=epistemic_role,
        relevance_gate=relevance_gate,
        approved_claims=approved,
        gap="" if approved else "compound_claim_requires_review",
    )


def build_temporal_shadow_context(
    text: str,
    *,
    request_id: str,
    now: datetime | None = None,
) -> tuple[TimeFirstQueryPlan, TemporalStatementShadowV1 | None]:
    """Build a unified time-first query plan and optional epistemic shadow analysis."""

    observed_at = now or datetime.now().astimezone()
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    normalized = normalize_context_text(text)
    statement = analyze_temporal_statement_shadow(
        text,
        request_id=request_id,
        now=observed_at,
        comparative_question=comparative_shadow_question(text),
        conflict=any(
            marker in normalized
            for marker in ("sprzecz", "konflikt", "nie zgadza", "ale teraz")
        ),
        decision_context=any(
            marker in normalized
            for marker in ("decyz", "wybier", "ustalmy", "postanow")
        ),
        action_impact=any(
            marker in normalized
            for marker in ("mam zrobic", "wykonaj", "uruchom", "usun")
        ),
    )
    if statement is not None:
        return statement.query_plan(), statement
    return build_time_first_query(text, now=observed_at), None


def shadow_relevance_gate(
    *,
    has_epistemic_content: bool,
    comparative_question: bool = False,
    conflict: bool = False,
    decision_context: bool = False,
    action_impact: bool = False,
) -> Literal["context_only", "candidate", "investigate"]:
    if not has_epistemic_content:
        return "context_only"
    if any((comparative_question, conflict, decision_context, action_impact)):
        return "investigate"
    return "candidate"


def comparative_shadow_question(text: str) -> bool:
    normalized = normalize_context_text(text)
    return any(
        marker in normalized
        for marker in (
            "porown",
            "co sie zmienilo",
            "czy nadal",
            "wtedy a teraz",
        )
    )
