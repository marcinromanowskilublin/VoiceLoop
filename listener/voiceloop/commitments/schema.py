from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class CommitmentType(StrEnum):
    INTENTION = "intention"
    PROMISE = "promise"
    REQUEST = "request"
    COMMAND = "command"
    REFUSAL = "refusal"
    CHEAP_SIGNAL = "cheap_signal"
    BOUNDARY = "boundary"


class CommitmentDirection(StrEnum):
    USER_TO_OTHER = "user_to_other"
    OTHER_TO_USER = "other_to_user"
    USER_TO_SELF = "user_to_self"
    OTHER_TO_USER_COMMITMENT = "other_to_user_commitment"
    MUTUAL = "mutual"
    UNCLEAR = "unclear"


class CommitmentStatus(StrEnum):
    CAPTURED = "captured"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_USER_REVIEW = "needs_user_review"
    ACCEPTED = "accepted"
    DECLINED = "declined"


class TranscriptChunk(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=120)
    speaker: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=8000)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_time_range(self) -> TranscriptChunk:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("chunk end_seconds must not precede start_seconds")
        return self


class EvidenceItem(BaseModel):
    kind: Literal["rule", "vector", "temporal", "resolver"]
    label: str = Field(min_length=1, max_length=120)
    score: float = Field(ge=0.0, le=1.0)
    detail: str | None = Field(default=None, max_length=500)


class CommitmentScores(BaseModel):
    action_clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    deadline_clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    owner_clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    pressure_level: float = Field(default=0.0, ge=0.0, le=1.0)
    autonomy_level: float = Field(default=0.5, ge=0.0, le=1.0)
    time_cost: float = Field(default=0.3, ge=0.0, le=1.0)
    cognitive_cost: float = Field(default=0.3, ge=0.0, le=1.0)
    emotional_cost: float = Field(default=0.2, ge=0.0, le=1.0)
    benefit_level: float = Field(default=0.4, ge=0.0, le=1.0)
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CommitmentItem(BaseModel):
    item_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120)
    source_chunk_id: str = Field(min_length=1, max_length=120)
    speaker: str = Field(min_length=1, max_length=120)
    raw_text: str = Field(min_length=1, max_length=8000)
    type: CommitmentType
    direction: CommitmentDirection = CommitmentDirection.UNCLEAR
    status: CommitmentStatus = CommitmentStatus.CAPTURED
    normalized_task: str | None = Field(default=None, max_length=1000)
    known_elements: dict[str, Any] = Field(default_factory=dict)
    missing_elements: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    scores: CommitmentScores = Field(default_factory=CommitmentScores)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CommitmentAnalysisResult(BaseModel):
    schema_version: Literal[1] = 1
    items: tuple[CommitmentItem, ...] = ()
