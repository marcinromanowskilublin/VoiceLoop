from __future__ import annotations

import re
import unicodedata

from .models import CommandPlan, CommandRequest, PlanStep, RiskLevel


def normalize_text(value: str) -> str:
    normalized = value.casefold().replace("ł", "l")
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9ąćęłńóśźż:/._ -]+", " ", normalized).strip()


def deterministic_plan(request: CommandRequest) -> CommandPlan | None:
    raw = (request.command_id or request.text or "").strip()
    text = normalize_text(raw)
    if not text:
        return None

    if text in {"ping", "voice_test", "voice test", "test petli", "test glosu"}:
        return CommandPlan(
            request_id=request.request_id,
            intent="voice_test",
            response_text="Pętla VoiceLoop działa.",
            confidence=1.0,
            steps=[],
            provider="deterministic",
        )

    if text in {"open_calendar", "open calendar"} or "otworz kalendarz" in text:
        return _single_step(
            request,
            intent="open_calendar",
            response_text="Otwieram kalendarz.",
            action_id="open_calendar",
        )

    if text in {"open_browser", "open browser"} or "otworz przegladark" in text:
        return _single_step(
            request,
            intent="open_browser",
            response_text="Otwieram przeglądarkę.",
            action_id="open_browser",
        )

    if text in {"open_chat", "open chat"} or "otworz czat" in text:
        return _single_step(
            request,
            intent="open_chat",
            response_text="Otwieram czat.",
            action_id="open_chat",
        )

    if text in {"stop", "stop now", "abort", "przerwij", "zatrzymaj"}:
        return CommandPlan(
            request_id=request.request_id,
            intent="stop",
            response_text="Zatrzymuję.",
            confidence=1.0,
            steps=[],
            provider="deterministic",
        )

    if text in {"note", "notatka", "take a note", "new note"}:
        return CommandPlan(
            request_id=request.request_id,
            intent="create_note",
            response_text="Co mam zapisać w notatce?",
            confidence=1.0,
            requires_clarification=True,
            clarification_question="Co mam zapisać w notatce?",
            provider="deterministic",
        )

    note_match = re.search(
        r"(?:zapisz|utworz|stworz|dodaj)(?: mi)? (?:notatke|notatka)(?: o tresci)? (.+)",
        text,
    )
    if note_match:
        content = raw[-len(note_match.group(1)) :].strip()
        return _single_step(
            request,
            intent="create_note",
            response_text="Tworzę notatkę.",
            action_id="create_note",
            args={"text": content},
            risk=RiskLevel.MEDIUM,
        )

    return None


def _single_step(
    request: CommandRequest,
    *,
    intent: str,
    response_text: str,
    action_id: str,
    args: dict[str, object] | None = None,
    risk: RiskLevel = RiskLevel.LOW,
) -> CommandPlan:
    return CommandPlan(
        request_id=request.request_id,
        intent=intent,
        response_text=response_text,
        confidence=1.0,
        steps=[PlanStep(action_id=action_id, args=args or {}, risk=risk)],
        provider="deterministic",
    )
