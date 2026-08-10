from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
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

    @model_validator(mode="after")
    def require_input(self) -> CommandRequest:
        if not (self.text and self.text.strip()) and not (
            self.command_id and self.command_id.strip()
        ):
            raise ValueError("text or command_id is required")
        return self


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
