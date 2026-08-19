from __future__ import annotations

from datetime import UTC, datetime

from ..router import normalize_text
from .schema import (
    JournalCategory,
    ProjectJournalCandidateV1,
    ReviewStatus,
    SpeakerStatus,
    UtteranceOrigin,
    UtteranceRecord,
)
from .storage import sha256_text

_CATEGORY_MARKERS: tuple[tuple[JournalCategory, tuple[str, ...]], ...] = (
    (
        JournalCategory.DECISION,
        (
            "decyduję",
            "decyduje",
            "ustalamy",
            "ustaliłem",
            "ustalilem",
            "wybieram",
            "robimy tak",
            "zróbmy tak",
            "zrobmy tak",
        ),
    ),
    (
        JournalCategory.GOAL,
        (
            "celem jest",
            "chcę zbudować",
            "chce zbudowac",
            "chcę żeby",
            "chce zeby",
            "docelowo",
            "musimy osiągnąć",
            "musimy osiagnac",
        ),
    ),
    (
        JournalCategory.OPEN_PROBLEM,
        (
            "problemem jest",
            "otwarty problem",
            "nie działa",
            "nie dziala",
            "błąd",
            "blad",
            "luka",
        ),
    ),
    (
        JournalCategory.ARCHITECTURE_CHANGE,
        (
            "zmiana architektury",
            "pipeline",
            "routing v2",
            "przebudować",
            "przebudowac",
            "refaktor",
        ),
    ),
    (
        JournalCategory.TO_VERIFY,
        (
            "trzeba sprawdzić",
            "trzeba sprawdzic",
            "do sprawdzenia",
            "zweryfikować",
            "zweryfikowac",
            "porównać",
            "porownac",
        ),
    ),
)
_EXPLICIT_DECISION_MARKERS = set(_CATEGORY_MARKERS[0][1])


def extract_project_journal_candidates(
    records: list[UtteranceRecord],
) -> list[ProjectJournalCandidateV1]:
    candidates: list[ProjectJournalCandidateV1] = []
    seen_content: set[str] = set()
    for record in records:
        if record.origin is not UtteranceOrigin.CURSOR_USER:
            continue
        if record.speaker_status is not SpeakerStatus.SELF:
            continue
        normalized = normalize_text(record.text)
        if not normalized:
            continue
        category, confidence = _classify_journal_entry(normalized)
        if category is None:
            continue
        looks_like_question = record.text.rstrip().endswith("?") or normalized.startswith(
            ("czy ", "co ", "jak ", "gdzie ", "kiedy ", "dlaczego ")
        )
        if looks_like_question and not any(
            marker in normalized for marker in _EXPLICIT_DECISION_MARKERS
        ):
            continue
        content_hash = sha256_text(f"{category.value}:{normalized}")
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        candidates.append(
            ProjectJournalCandidateV1(
                candidate_id=content_hash[:24],
                category=category,
                summary=" ".join(record.text.split())[:2000],
                evidence_utterance_ids=(record.utterance_id,),
                source_session_id=record.session_id,
                source_timestamp=record.captured_at,
                confidence=confidence,
            )
        )
    return candidates


def decide_journal_candidate(
    candidates: list[ProjectJournalCandidateV1],
    *,
    candidate_id: str,
    approve: bool,
    confirmation: str,
) -> list[ProjectJournalCandidateV1]:
    if confirmation != candidate_id:
        raise ValueError("--confirm musi być identyczne z candidate_id.")
    found = False
    result: list[ProjectJournalCandidateV1] = []
    now = datetime.now(UTC)
    for candidate in candidates:
        if candidate.candidate_id != candidate_id:
            result.append(candidate)
            continue
        found = True
        if candidate.status is not ReviewStatus.PENDING:
            raise ValueError("Kandydat dziennika został już rozpatrzony.")
        result.append(
            candidate.model_copy(
                update={
                    "status": ReviewStatus.APPROVED if approve else ReviewStatus.REJECTED,
                    "reviewed_at": now,
                }
            )
        )
    if not found:
        raise ValueError("Nie znaleziono kandydata dziennika.")
    return result


def approved_journal_entries(
    candidates: list[ProjectJournalCandidateV1],
) -> list[ProjectJournalCandidateV1]:
    return [candidate for candidate in candidates if candidate.status is ReviewStatus.APPROVED]


def _classify_journal_entry(
    normalized: str,
) -> tuple[JournalCategory | None, float]:
    matches: list[tuple[JournalCategory, int]] = []
    for category, markers in _CATEGORY_MARKERS:
        count = sum(marker in normalized for marker in markers)
        if count:
            matches.append((category, count))
    if not matches:
        return None, 0.0
    category, count = max(
        matches,
        key=lambda item: (item[1], -list(JournalCategory).index(item[0])),
    )
    return category, min(0.95, 0.70 + 0.08 * (count - 1))
