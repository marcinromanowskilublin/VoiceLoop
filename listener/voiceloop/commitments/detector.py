from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from .schema import (
    CommitmentDirection,
    CommitmentItem,
    CommitmentStatus,
    CommitmentType,
    EvidenceItem,
    TranscriptChunk,
)

USER_SPEAKERS = {"user", "uzytkownik", "użytkownik", "speaker_user"}

CHEAP_SIGNAL_PATTERNS = {
    "vague_intention": re.compile(
        r"\b(postaram sie|postaram się|sprobuje|spróbuję|zobacze|zobaczę|pomysle|pomyślę)\b"
    ),
    "vague_action": re.compile(r"\b(ogarne|ogarnę|zajme sie|zajmę się|zalatwie|załatwię)\b"),
    "weak_reply": re.compile(r"\b(dam znac|dam znać|jakos to bedzie|jakoś to będzie)\b"),
    "identity_signal": re.compile(r"\b(bede lepszy|będę lepszy|poprawie sie|poprawię się)\b"),
}
PROMISE_PATTERN = re.compile(r"\b(obiecuje|obiecuję|zobowiazuje sie|zobowiązuję się)\b")
REQUEST_PATTERN = re.compile(r"\b(czy mozesz|czy możesz|prosze|proszę|wyslij|wyślij)\b")
COMMAND_PATTERN = re.compile(r"\b(musisz|masz|zrob|zrób)\b")
REFUSAL_PATTERN = re.compile(
    r"\b(nie zrobie|nie zrobię|nie wysle|nie wyślę|nie zgadzam sie|nie zgadzam się)\b"
)
ACTION_PATTERN = re.compile(
    r"\b("
    r"wysle|wyślę|przesle|prześlę|zadzwonie|zadzwonię|oddam|przygotuje|przygotuję|"
    r"przyniose|przyniosę|zaplace|zapłacę|wypelnie|wypełnię|zrobie|zrobię"
    r")\b"
)
DEADLINE_PATTERN = re.compile(
    r"\b("
    r"dzisiaj|jutro|pojutrze|w piatek|w piątek|do piatku|do piątku|"
    r"przed wizyta|przed wizytą|do \d{1,2}(?::\d{2})?"
    r")\b"
)
PRESSURE_PATTERN = re.compile(r"\b(musisz|natychmiast|teraz|inaczej|masz)\b")
VERIFICATION_PATTERN = re.compile(
    r"\b(potwierdzenie|potwierdze|potwierdzę|sms|mail|mailem|pdf|odhacze|odhaczę)\b"
)


def normalize_for_rules(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def detect_commitment_items(
    chunks: Iterable[TranscriptChunk],
    *,
    user_speakers: set[str] | None = None,
) -> list[CommitmentItem]:
    users = {normalize_for_rules(value) for value in (user_speakers or USER_SPEAKERS)}
    items: list[CommitmentItem] = []
    for chunk in chunks:
        item = detect_chunk(chunk, user_speakers=users)
        if item is not None:
            items.append(item)
    return items


def detect_chunk(
    chunk: TranscriptChunk,
    *,
    user_speakers: set[str] | None = None,
) -> CommitmentItem | None:
    text = normalize_for_rules(chunk.text)
    speaker = normalize_for_rules(chunk.speaker)
    users = user_speakers or USER_SPEAKERS

    evidence: list[EvidenceItem] = []
    cheap_label = _first_cheap_signal(text, evidence)
    deadline = _first_match(DEADLINE_PATTERN, text)
    action = _first_match(ACTION_PATTERN, text)
    has_verification = bool(VERIFICATION_PATTERN.search(text))
    has_pressure = bool(PRESSURE_PATTERN.search(text))

    commitment_type = _commitment_type(text, cheap_label=cheap_label)
    if commitment_type is None:
        return None

    if deadline:
        evidence.append(EvidenceItem(kind="rule", label="deadline", score=0.8, detail=deadline))
    if action:
        evidence.append(EvidenceItem(kind="rule", label="action", score=0.8, detail=action))
    if has_pressure:
        evidence.append(EvidenceItem(kind="rule", label="pressure", score=0.7))

    missing = _missing_elements(
        commitment_type=commitment_type,
        action=action,
        deadline=deadline,
        has_verification=has_verification,
    )
    direction = _direction(commitment_type=commitment_type, speaker=speaker, user_speakers=users)
    status = _status(commitment_type=commitment_type, direction=direction, missing=missing)
    known = {}
    if deadline:
        known["deadline"] = deadline
    if action:
        known["action"] = action
    if cheap_label:
        known["cheap_signal_type"] = cheap_label

    return CommitmentItem(
        source_chunk_id=chunk.chunk_id,
        speaker=chunk.speaker,
        raw_text=chunk.text,
        type=commitment_type,
        direction=direction,
        status=status,
        normalized_task=_normalized_task(chunk.text, commitment_type),
        known_elements=known,
        missing_elements=tuple(missing),
        clarification_questions=tuple(_clarification_questions(missing)),
        evidence=tuple(evidence),
        confidence=_confidence(evidence, missing),
    )


def _first_cheap_signal(text: str, evidence: list[EvidenceItem]) -> str | None:
    for label, pattern in CHEAP_SIGNAL_PATTERNS.items():
        match = pattern.search(text)
        if match:
            evidence.append(
                EvidenceItem(
                    kind="rule",
                    label=label,
                    score=0.85,
                    detail=match.group(0),
                )
            )
            return label
    return None


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def _commitment_type(text: str, *, cheap_label: str | None) -> CommitmentType | None:
    if REFUSAL_PATTERN.search(text):
        return CommitmentType.REFUSAL
    if COMMAND_PATTERN.search(text):
        return CommitmentType.COMMAND
    if REQUEST_PATTERN.search(text):
        return CommitmentType.REQUEST
    if PROMISE_PATTERN.search(text):
        return CommitmentType.PROMISE
    if cheap_label:
        return CommitmentType.CHEAP_SIGNAL
    if ACTION_PATTERN.search(text):
        return CommitmentType.INTENTION
    return None


def _direction(
    *,
    commitment_type: CommitmentType,
    speaker: str,
    user_speakers: set[str],
) -> CommitmentDirection:
    speaker_is_user = speaker in user_speakers
    if commitment_type in {CommitmentType.REQUEST, CommitmentType.COMMAND}:
        if speaker_is_user:
            return CommitmentDirection.USER_TO_OTHER
        return CommitmentDirection.OTHER_TO_USER
    if commitment_type in {
        CommitmentType.PROMISE,
        CommitmentType.CHEAP_SIGNAL,
        CommitmentType.INTENTION,
    }:
        if speaker_is_user:
            return CommitmentDirection.USER_TO_OTHER
        return CommitmentDirection.OTHER_TO_USER_COMMITMENT
    if commitment_type is CommitmentType.REFUSAL:
        if speaker_is_user:
            return CommitmentDirection.USER_TO_OTHER
        return CommitmentDirection.OTHER_TO_USER
    return CommitmentDirection.UNCLEAR


def _missing_elements(
    *,
    commitment_type: CommitmentType,
    action: str | None,
    deadline: str | None,
    has_verification: bool,
) -> list[str]:
    missing: list[str] = []
    if commitment_type is CommitmentType.CHEAP_SIGNAL or action is None:
        missing.append("action")
    if deadline is None and commitment_type not in {
        CommitmentType.REFUSAL,
        CommitmentType.BOUNDARY,
    }:
        missing.append("deadline")
    if not has_verification and commitment_type not in {
        CommitmentType.REQUEST,
        CommitmentType.COMMAND,
    }:
        missing.append("verification_method")
    return missing


def _status(
    *,
    commitment_type: CommitmentType,
    direction: CommitmentDirection,
    missing: list[str],
) -> CommitmentStatus:
    if direction is CommitmentDirection.OTHER_TO_USER:
        return CommitmentStatus.NEEDS_USER_REVIEW
    if commitment_type is CommitmentType.CHEAP_SIGNAL or missing:
        return CommitmentStatus.NEEDS_CLARIFICATION
    return CommitmentStatus.CAPTURED


def _clarification_questions(missing: list[str]) -> list[str]:
    questions = {
        "action": "Co dokładnie ma zostać zrobione?",
        "deadline": "Do kiedy konkretnie?",
        "verification_method": "Po czym poznasz, że to jest zrobione?",
    }
    return [questions[item] for item in missing if item in questions][:3]


def _normalized_task(text: str, commitment_type: CommitmentType) -> str | None:
    cleaned = text.strip()
    if commitment_type is CommitmentType.REQUEST:
        return f"Rozważyć prośbę: {cleaned}"
    if commitment_type is CommitmentType.COMMAND:
        return f"Przejrzeć polecenie przed przyjęciem: {cleaned}"
    if commitment_type is CommitmentType.CHEAP_SIGNAL:
        return None
    return cleaned


def _confidence(evidence: list[EvidenceItem], missing: list[str]) -> float:
    score = 0.45 + min(len(evidence), 4) * 0.1 - min(len(missing), 3) * 0.05
    return max(0.1, min(score, 0.95))
