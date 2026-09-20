from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def stable_context_id(source: str, source_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"voiceloop-context:{source}:{source_id}"))


def stable_episode_id(source: str, source_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"voiceloop-episode:{source}:{source_id}"))


def content_hash_for(*parts: object) -> str:
    payload = json.dumps(
        [str(part or "") for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ContextScope(StrEnum):
    LIVE = "live"
    SESSION = "session"
    EPISODIC = "episodic"
    DURABLE = "durable"


class ContextTrust(StrEnum):
    AUTHORITATIVE = "authoritative"
    USER_ASSERTED = "user_asserted"
    OBSERVED = "observed"
    DERIVED = "derived"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    MODEL_INFERENCE = "model_inference"


class ForegroundConfidence(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class ContextItemV1(BaseModel):
    """One bounded, non-executable context item selected for a model turn."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    context_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=240)
    scope: ContextScope
    kind: str = Field(default="observation", min_length=1, max_length=80)
    title: str = Field(default="", max_length=500)
    content: str = Field(min_length=1, max_length=20000)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    trust: ContextTrust
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: str = Field(default="private", max_length=40)
    expires_at: datetime | None = None
    content_hash: str = Field(default="", max_length=64)
    retrieval_score: float | None = None
    selection_reason: str = Field(default="", max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("started_at", "ended_at", "expires_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("context timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_range_and_hash(self) -> ContextItemV1:
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("context ended_at precedes started_at")
        if (
            self.started_at is not None
            and self.expires_at is not None
            and self.expires_at < self.started_at
        ):
            raise ValueError("context expires_at precedes started_at")
        expected = content_hash_for(
            self.source,
            self.source_id,
            self.title,
            self.content,
        )
        if not self.content_hash:
            self.content_hash = expected
        elif self.content_hash != expected:
            raise ValueError("context content_hash does not match content")
        return self

    def prompt_text(self) -> str:
        """Serialize as explicitly untrusted data while preserving provenance."""

        time_value = self.started_at.isoformat() if self.started_at is not None else "unknown"
        header = (
            f"source={self.source}; source_id={self.source_id}; scope={self.scope.value}; "
            f"kind={self.kind}; time={time_value}; trust={self.trust.value}"
        )
        title = f"; title={self.title}" if self.title else ""
        return f"Local context data ({header}{title}): {self.content}"[:2400]


class ContextPackV1(BaseModel):
    """A measured, ordered context package for one turn."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    question: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=120)
    generated_at: datetime = Field(default_factory=utc_now)
    items: tuple[ContextItemV1, ...] = ()
    sources: dict[str, int] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def _aware_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(UTC)

    def memory_strings(self, *, limit: int = 20) -> list[str]:
        safe_limit = max(0, min(int(limit), 100))
        return [item.prompt_text() for item in self.items[:safe_limit]]


class ContextEventV1(BaseModel):
    """Canonical timeline event stored in SQLite, never an executable intent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=240)
    event_type: str = Field(default="observation", min_length=1, max_length=80)
    started_at: datetime
    ended_at: datetime | None = None
    app_name: str = Field(default="", max_length=500)
    process_name: str = Field(default="", max_length=300)
    window_title: str = Field(default="", max_length=1000)
    is_foreground: bool | None = None
    foreground_confidence: ForegroundConfidence = ForegroundConfidence.UNKNOWN
    focus_duration_ms: int | None = Field(default=None, ge=0)
    text: str = Field(default="", max_length=20000)
    content_hash: str = Field(default="", max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)
    sensitivity: str = Field(default="private", max_length=40)
    expires_at: datetime | None = None
    deleted_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_observation(
        cls,
        *,
        source: str,
        source_id: str,
        started_at: datetime,
        event_type: str = "observation",
        ended_at: datetime | None = None,
        app_name: str = "",
        process_name: str = "",
        window_title: str = "",
        is_foreground: bool | None = None,
        foreground_confidence: ForegroundConfidence = ForegroundConfidence.UNKNOWN,
        focus_duration_ms: int | None = None,
        text: str = "",
        metadata: dict[str, Any] | None = None,
        sensitivity: str = "private",
        expires_at: datetime | None = None,
    ) -> ContextEventV1:
        return cls(
            event_id=stable_context_id(source, source_id),
            source=source,
            source_id=source_id,
            event_type=event_type,
            started_at=started_at,
            ended_at=ended_at,
            app_name=app_name,
            process_name=process_name,
            window_title=window_title,
            is_foreground=is_foreground,
            foreground_confidence=foreground_confidence,
            focus_duration_ms=focus_duration_ms,
            text=text,
            metadata=dict(metadata or {}),
            sensitivity=sensitivity,
            expires_at=expires_at,
        )

    @field_validator(
        "started_at",
        "ended_at",
        "expires_at",
        "deleted_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def _aware_event_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_event(self) -> ContextEventV1:
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("event ended_at precedes started_at")
        expected = content_hash_for(
            self.source,
            self.source_id,
            self.event_type,
            self.started_at.isoformat(),
            self.ended_at.isoformat() if self.ended_at else "",
            self.app_name,
            self.process_name,
            self.window_title,
            self.text,
        )
        if not self.content_hash:
            self.content_hash = expected
        elif self.content_hash != expected:
            raise ValueError("event content_hash does not match event")
        if (
            self.is_foreground is not None
            and self.foreground_confidence is ForegroundConfidence.UNKNOWN
        ):
            raise ValueError("known foreground state requires observed or inferred confidence")
        return self

    def as_context_item(self, *, selection_reason: str = "") -> ContextItemV1:
        content = self.text or self.window_title or self.app_name or self.event_type
        return ContextItemV1(
            context_id=self.event_id,
            source=self.source,
            source_id=self.source_id,
            scope=ContextScope.EPISODIC,
            kind=self.event_type,
            title=self.window_title or self.app_name,
            content=content,
            started_at=self.started_at,
            ended_at=self.ended_at,
            trust=ContextTrust.OBSERVED,
            confidence=1.0 if self.foreground_confidence is ForegroundConfidence.OBSERVED else 0.7,
            sensitivity=self.sensitivity,
            expires_at=self.expires_at,
            selection_reason=selection_reason,
            metadata={
                **self.metadata,
                "app_name": self.app_name,
                "process_name": self.process_name,
                "is_foreground": self.is_foreground,
                "foreground_confidence": self.foreground_confidence.value,
            },
        )


class ContextEpisodeV1(BaseModel):
    """A deterministic or reviewed digest over a bounded timeline range."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    episode_id: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=240)
    started_at: datetime
    ended_at: datetime
    title: str = Field(default="", max_length=500)
    summary: str = Field(min_length=1, max_length=20000)
    source_event_ids: tuple[str, ...] = Field(min_length=1, max_length=1000)
    app_names: tuple[str, ...] = ()
    person_ids: tuple[str, ...] = ()
    project_ids: tuple[str, ...] = ()
    vector_spaces: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = Field(default="", max_length=64)
    sensitivity: str = Field(default="private", max_length=40)
    expires_at: datetime | None = None
    deleted_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "started_at",
        "ended_at",
        "expires_at",
        "deleted_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def _aware_episode_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("episode timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_episode(self) -> ContextEpisodeV1:
        if self.ended_at < self.started_at:
            raise ValueError("episode ended_at precedes started_at")
        expected = content_hash_for(
            self.source,
            self.source_id,
            self.started_at.isoformat(),
            self.ended_at.isoformat(),
            self.title,
            self.summary,
            *self.source_event_ids,
        )
        if not self.content_hash:
            self.content_hash = expected
        elif self.content_hash != expected:
            raise ValueError("episode content_hash does not match episode")
        return self

    def as_context_item(
        self,
        *,
        selection_reason: str = "timeline_episode",
    ) -> ContextItemV1:
        return ContextItemV1(
            context_id=self.episode_id,
            source=self.source,
            source_id=self.source_id,
            scope=ContextScope.EPISODIC,
            kind="episode",
            title=self.title,
            content=self.summary,
            started_at=self.started_at,
            ended_at=self.ended_at,
            trust=ContextTrust.DERIVED,
            confidence=0.8,
            sensitivity=self.sensitivity,
            expires_at=self.expires_at,
            selection_reason=selection_reason,
            metadata={
                **self.metadata,
                "source_event_ids": list(self.source_event_ids),
                "app_names": list(self.app_names),
                "person_ids": list(self.person_ids),
                "project_ids": list(self.project_ids),
                "vector_spaces": list(self.vector_spaces),
            },
        )
