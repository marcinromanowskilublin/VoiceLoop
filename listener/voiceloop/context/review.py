"""Review-only context items. None of these functions execute or approve actions."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ..corpus.privacy import redact_text
from .schema import (
    ContextEventV1,
    ContextItemV1,
    ContextScope,
    ContextTrust,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps commitments out of startup
    from ..commitments.schema import CommitmentItem
    from ..models import ScreenSnapshot

_DEICTIC = re.compile(
    r"\b(?:tutaj|to okno|tego okna|na tym ekranie|na ekranie)\b",
    re.IGNORECASE,
)
LIVE_SCREEN_TTL_SECONDS = 300


def question_is_deictic(text: str) -> bool:
    """Match phrases that point at the current screen, not the common word „to”."""

    return bool(_DEICTIC.search(text or ""))


def live_screen_item(snapshot: ScreenSnapshot) -> ContextItemV1 | None:
    """Build an ephemeral context item. The caller must not persist the image."""

    title = (snapshot.window_title or "").strip()
    process = (snapshot.process_name or "").strip()
    if not title and not process:
        return None
    label = title or process
    captured_at = _aware(snapshot.captured_at)
    return ContextItemV1(
        source="live_screen",
        source_id=f"screen:{captured_at.isoformat()}",
        scope=ContextScope.LIVE,
        kind="live_screen",
        title=label[:500],
        content=f"Bieżące okno: {label}. Proces: {process or 'nieznany'}.",
        started_at=captured_at,
        trust=ContextTrust.OBSERVED,
        confidence=0.6,
        sensitivity="private",
        expires_at=captured_at + timedelta(seconds=LIVE_SCREEN_TTL_SECONDS),
        selection_reason="deictic_live_screen",
        metadata={"process_name": process, "persisted": False},
    )


def commitment_review_event(
    item: CommitmentItem,
    *,
    request_id: str,
    observed_at: datetime | None = None,
    ttl_days: int = 14,
) -> ContextEventV1 | None:
    """A reviewable shadow row. It is not a commitment acceptance and not an action."""

    raw = (item.normalized_task or item.raw_text or "").strip()
    redacted, flags = redact_text(raw)
    if not redacted or "secret" in flags or "pesel" in flags:
        return None
    identity = f"{request_id}:{item.type.value}:{redacted}"
    source_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    moment = _aware(observed_at or datetime.now(UTC))
    safe_ttl = max(0, min(int(ttl_days), 3650))
    return ContextEventV1.from_observation(
        source="commitment_shadow",
        source_id=source_id,
        started_at=moment,
        event_type="commitment_review",
        text=redacted[:8000],
        metadata={
            "request_id": request_id,
            "commitment_type": item.type.value,
            "commitment_status": item.status.value,
            "review_required": True,
            "executable": False,
            "vectorize": False,
        },
        sensitivity="private",
        expires_at=moment + timedelta(days=safe_ttl) if safe_ttl else None,
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
