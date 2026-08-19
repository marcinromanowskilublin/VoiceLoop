from __future__ import annotations

import hashlib
import math
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


class CommandSource(StrEnum):
    PANEL = "panel"
    DEEPGRAM = "deepgram"
    VOICEATTACK = "voiceattack"
    API = "api"
    N8N = "n8n"


def normalize_transcript_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().split())


class TranscriptWordV1(BaseModel):
    word: str = Field(min_length=1, max_length=200)
    punctuated_word: str | None = Field(default=None, max_length=250)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> TranscriptWordV1:
        if self.end_seconds < self.start_seconds:
            raise ValueError("word end_seconds must not precede start_seconds")
        return self


class TranscriptEnvelopeV1(BaseModel):
    schema_version: Literal[1] = 1
    segment_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120)
    raw_text: str = Field(min_length=1, max_length=8000)
    normalized_text: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="pl", min_length=2, max_length=20)
    confidence_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_min: float | None = Field(default=None, ge=0.0, le=1.0)
    words: tuple[TranscriptWordV1, ...] = ()
    started_at_seconds: float | None = Field(default=None, ge=0.0)
    ended_at_seconds: float | None = Field(default=None, ge=0.0)
    speaker_ids: tuple[int, ...] = ()
    is_final: bool = True
    speech_final: bool = True
    model: str | None = Field(default=None, max_length=160)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_envelope(self) -> TranscriptEnvelopeV1:
        expected_text = normalize_transcript_text(self.raw_text)
        if self.normalized_text != expected_text:
            raise ValueError("normalized_text does not match normalized raw_text")
        if (
            self.started_at_seconds is not None
            and self.ended_at_seconds is not None
            and self.ended_at_seconds < self.started_at_seconds
        ):
            raise ValueError("ended_at_seconds must not precede started_at_seconds")
        if (
            self.confidence_mean is not None
            and self.confidence_min is not None
            and self.confidence_min > self.confidence_mean
        ):
            raise ValueError("confidence_min must not exceed confidence_mean")
        word_speakers = {word.speaker_id for word in self.words if word.speaker_id is not None}
        if any(
            not isinstance(speaker_id, int) or isinstance(speaker_id, bool) or speaker_id < 0
            for speaker_id in self.speaker_ids
        ):
            raise ValueError("speaker_ids must contain non-negative integers")
        if self.speaker_ids != tuple(sorted(set(self.speaker_ids))):
            raise ValueError("speaker_ids must be unique and sorted")
        if word_speakers and not word_speakers.issubset(set(self.speaker_ids)):
            raise ValueError("speaker_ids must include all word speaker IDs")
        return self

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        language: str = "pl",
        confidence: float | None = None,
        speaker_ids: tuple[int, ...] = (),
        is_final: bool = True,
        speech_final: bool = True,
        model: str | None = None,
    ) -> TranscriptEnvelopeV1:
        cleaned = normalize_transcript_text(text)
        return cls(
            raw_text=text.strip(),
            normalized_text=cleaned,
            language=language,
            confidence_mean=confidence,
            confidence_min=confidence,
            speaker_ids=tuple(sorted(set(speaker_ids))),
            is_final=is_final,
            speech_final=speech_final,
            model=model,
        )


class SegmentationDecisionV1(StrEnum):
    SIMPLE = "simple"
    COMPOUND = "compound"
    AMBIGUOUS = "ambiguous"
    NON_COMMAND = "non_command"


class ResolutionStatusV1(StrEnum):
    RESOLVED = "resolved"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


class TextSpanV1(BaseModel):
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_span(self) -> TextSpanV1:
        if self.end_char <= self.start_char:
            raise ValueError("span end_char must be greater than start_char")
        return self


class SubtaskV1(BaseModel):
    schema_version: Literal[1] = 1
    subtask_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=2000)
    source_text: str = Field(min_length=1, max_length=2000)
    normalized_text: str = Field(min_length=1, max_length=2000)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    order: int = Field(ge=0)
    operation: str | None = Field(default=None, max_length=80)
    target: str | None = Field(default=None, max_length=200)
    raw_arguments: dict[str, str] = Field(default_factory=dict)
    segmentation_confidence: float = Field(ge=0.0, le=1.0)
    has_unrecognized_text: bool = False

    @model_validator(mode="after")
    def validate_subtask(self) -> SubtaskV1:
        if self.end_char <= self.start_char:
            raise ValueError("subtask end_char must be greater than start_char")
        if self.normalized_text != normalize_transcript_text(self.text):
            raise ValueError("subtask normalized_text does not match text")
        return self


class SegmentationResultV1(BaseModel):
    schema_version: Literal[1] = 1
    decision: SegmentationDecisionV1
    subtasks: tuple[SubtaskV1, ...] = ()
    unrecognized_spans: tuple[TextSpanV1, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_result(self) -> SegmentationResultV1:
        orders = [subtask.order for subtask in self.subtasks]
        if orders != list(range(len(self.subtasks))):
            raise ValueError("subtask order must be contiguous and start at zero")
        if any(
            current.start_char < previous.end_char
            for previous, current in zip(
                self.subtasks,
                self.subtasks[1:],
                strict=False,
            )
        ):
            raise ValueError("subtask source spans must not overlap")
        if self.decision is SegmentationDecisionV1.SIMPLE and len(self.subtasks) != 1:
            raise ValueError("simple segmentation must contain exactly one subtask")
        if self.decision is SegmentationDecisionV1.COMPOUND and len(self.subtasks) < 2:
            raise ValueError("compound segmentation must contain at least two subtasks")
        if self.decision is SegmentationDecisionV1.NON_COMMAND and self.subtasks:
            raise ValueError("non-command segmentation must not contain subtasks")
        return self


class SubtaskEmbeddingV1(BaseModel):
    schema_version: Literal[1] = 1
    subtask_id: str
    semantic: tuple[float, ...]
    intent: tuple[float, ...]
    target_context: tuple[float, ...]
    embedding_model: str = Field(min_length=1, max_length=200)
    dimension: int = Field(gt=0)
    format_version: Literal[
        "capability-query-v1",
        "capability-query-v2:routing-taxonomy-v2",
    ] = "capability-query-v2:routing-taxonomy-v2"
    normalized_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_vectors(self) -> SubtaskEmbeddingV1:
        vectors = (self.semantic, self.intent, self.target_context)
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("all subtask vectors must match dimension")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("subtask vectors must contain only finite values")
        return self

    @classmethod
    def text_hash(cls, text: str) -> str:
        normalized = normalize_transcript_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ResolutionCandidateV1(BaseModel):
    action_id: str = Field(min_length=1, max_length=120)
    vector_score: float = Field(ge=-1.0, le=1.0)
    vector_scores: dict[str, float] = Field(default_factory=dict)
    vector_ranks: dict[str, int] = Field(default_factory=dict)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    missing_vector_names: tuple[str, ...] = ()
    lexical_score: float = Field(ge=0.0, le=1.0)
    argument_compatibility: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    extracted_args: dict[str, Any] = Field(default_factory=dict)
    eligible: bool = True
    rejection_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_vector_scores(self) -> ResolutionCandidateV1:
        supported = {"semantic", "intent", "target_context"}
        observed = set(self.vector_scores)
        if not observed or not observed <= supported:
            raise ValueError("candidate vector scores must use supported capability spaces")
        if any(
            not math.isfinite(value) or value < -1.0 or value > 1.0
            for value in self.vector_scores.values()
        ):
            raise ValueError("candidate vector scores must be finite cosine scores")
        if not set(self.vector_ranks) <= observed or any(
            not isinstance(rank, int) or rank < 1 for rank in self.vector_ranks.values()
        ):
            raise ValueError("candidate vector ranks must refer to observed vector scores")
        derived_coverage = len(observed) / len(supported)
        if self.coverage is None:
            self.coverage = derived_coverage
        elif abs(self.coverage - derived_coverage) > 1e-9:
            raise ValueError("candidate coverage must match observed vector evidence")
        derived_missing = tuple(sorted(supported - observed))
        if not self.missing_vector_names:
            self.missing_vector_names = derived_missing
        elif tuple(sorted(self.missing_vector_names)) != derived_missing:
            raise ValueError("candidate missing vectors must match observed evidence")
        return self


class ResolutionDecisionV1(BaseModel):
    schema_version: Literal[1] = 1
    subtask_id: str
    candidates: tuple[ResolutionCandidateV1, ...] = ()
    top1_action_id: str | None = None
    margin_top2: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: ResolutionStatusV1
    reason: str | None = Field(default=None, max_length=500)
    stt_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    catalog_hash: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_decision(self) -> ResolutionDecisionV1:
        action_ids = {candidate.action_id for candidate in self.candidates}
        if self.top1_action_id is not None and self.top1_action_id not in action_ids:
            raise ValueError("top1_action_id must identify a candidate")
        scores = [candidate.combined_score for candidate in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError("resolution candidates must be sorted by combined score")
        if self.decision is ResolutionStatusV1.RESOLVED:
            if self.top1_action_id is None or not self.candidates:
                raise ValueError("resolved decision must identify a candidate")
            if not self.candidates[0].eligible:
                raise ValueError("resolved decision top candidate must be eligible")
            if self.margin_top2 is None:
                raise ValueError("resolved decision requires a top-2 margin")
        return self


class CommandStatus(StrEnum):
    RECEIVED = "received"
    PLANNING = "planning"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    QUEUED = "queued"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommandRequest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    source: CommandSource = CommandSource.API
    text: str | None = Field(default=None, max_length=8000)
    command_id: str | None = Field(default=None, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)
    include_screen: bool = False
    allow_cloud: bool = False
    transcript_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    transcript: TranscriptEnvelopeV1 | None = None
    managed_voice_turn: bool = False
    interaction_session_id: str | None = Field(default=None, max_length=120)
    conversation_turn_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_input(self) -> CommandRequest:
        if self.interaction_session_id is not None:
            normalized_session_id = self.interaction_session_id.strip()
            self.interaction_session_id = normalized_session_id or None
        if self.transcript is not None:
            if self.text is None:
                self.text = self.transcript.normalized_text
            elif normalize_transcript_text(self.text) != self.transcript.normalized_text:
                raise ValueError("request text does not match transcript envelope")
            if self.transcript_confidence is None:
                self.transcript_confidence = self.transcript.confidence_mean
            elif (
                self.transcript.confidence_mean is not None
                and abs(self.transcript_confidence - self.transcript.confidence_mean) > 1e-6
            ):
                raise ValueError("request confidence does not match transcript envelope")
        if not (self.text and self.text.strip()) and not (
            self.command_id and self.command_id.strip()
        ):
            raise ValueError("text or command_id is required")
        return self

    @classmethod
    def from_transcript(
        cls,
        transcript: TranscriptEnvelopeV1,
        *,
        allow_cloud: bool = False,
        managed_voice_turn: bool = False,
        interaction_session_id: str | None = None,
        conversation_turn_id: int | None = None,
    ) -> CommandRequest:
        return cls(
            source=CommandSource.DEEPGRAM,
            text=transcript.normalized_text,
            allow_cloud=allow_cloud,
            transcript_confidence=transcript.confidence_mean,
            transcript=transcript,
            managed_voice_turn=managed_voice_turn,
            interaction_session_id=interaction_session_id,
            conversation_turn_id=conversation_turn_id,
        )

    @property
    def effective_transcript_confidence(self) -> float | None:
        values = [
            value
            for value in (
                self.transcript_confidence,
                self.transcript.confidence_min if self.transcript else None,
            )
            if value is not None
        ]
        return min(values) if values else None


class CapabilityMatchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    action_id: str = Field(max_length=120)
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW
    confirmation_required: bool = False
    success_condition: str | None = Field(default=None, max_length=500)


class CommandPlan(BaseModel):
    schema_version: int = SCHEMA_VERSION
    request_id: str
    intent: str = Field(max_length=120)
    response_text: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=1000)
    steps: list[PlanStep] = Field(default_factory=list)
    provider: str = "deterministic"
    model: str | None = None
    speak_result: bool = False
    managed_voice_turn: bool = False

    @property
    def confirmation_required(self) -> bool:
        return any(step.confirmation_required for step in self.steps)


class ActionResult(BaseModel):
    action_id: str
    success: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0


class CommandView(BaseModel):
    request_id: str
    source: str
    input_text: str
    status: CommandStatus
    intent: str | None = None
    response_text: str | None = None
    provider: str | None = None
    model: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    plan: CommandPlan | None = None
    results: list[ActionResult] = Field(default_factory=list)


class CommandAccepted(BaseModel):
    request_id: str
    status: CommandStatus
    plan: CommandPlan | None = None
    duplicate: bool = False


class ConversationTurnTrace(BaseModel):
    schema_version: Literal[1] = 1
    session_id: str = Field(min_length=1, max_length=120)
    turn_id: int = Field(ge=0)
    request_id: str | None = Field(default=None, max_length=120)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = Field(default="active", max_length=40)
    phases_ms: dict[str, int] = Field(default_factory=dict)
    metrics_ms: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTraceSnapshot(BaseModel):
    traces: list[ConversationTurnTrace] = Field(default_factory=list)
    aggregates: dict[str, dict[str, int]] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    kind: str = Field(default="web_search", max_length=80)
    query: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2048)
    snippet: str = Field(default="", max_length=1200)
    provider: str = Field(default="", max_length=80)
    published_at: str | None = Field(default=None, max_length=120)
    retrieved_at: datetime = Field(default_factory=utc_now)


class HealthComponent(BaseModel):
    status: str
    detail: str | None = None
    version: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, HealthComponent]


class MemoryCreate(BaseModel):
    kind: str = Field(default="fact", max_length=50)
    content: str = Field(min_length=1, max_length=10000)
    sensitivity: str = Field(default="private", max_length=30)
    source: str = Field(default="user", max_length=50)


class MemoryItem(MemoryCreate):
    id: int
    created_at: datetime
    updated_at: datetime


class ScreenSnapshot(BaseModel):
    captured_at: datetime = Field(default_factory=utc_now)
    window_title: str = ""
    process_name: str = ""
    image_path: str | None = None
    controls: list[dict[str, Any]] = Field(default_factory=list)


class TurnContext(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=120)
    recent_turns: list[dict[str, str]] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    tool_observations: list[ToolObservation] = Field(default_factory=list)
    local_time: str = Field(max_length=120)
    screen: ScreenSnapshot | None = None
    sources: dict[str, int] = Field(default_factory=dict)
