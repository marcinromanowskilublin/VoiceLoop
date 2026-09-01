from __future__ import annotations

from .schema import (
    CommitmentDirection,
    CommitmentItem,
    CommitmentScores,
    CommitmentType,
)

LOW = 0.15
MEDIUM = 0.5
HIGH = 0.85


def score_commitment(item: CommitmentItem) -> CommitmentItem:
    evidence_labels = {evidence.label for evidence in item.evidence}
    known = item.known_elements
    missing = set(item.missing_elements)

    action_clarity = _action_clarity(item, known, missing)
    deadline_clarity = _deadline_clarity(known, missing)
    owner_clarity = _owner_clarity(item)
    pressure_level = _pressure_level(item, evidence_labels)
    autonomy_level = _autonomy_level(item, pressure_level)

    time_cost = _time_cost(item)
    cognitive_cost = _cognitive_cost(item, missing)
    emotional_cost = _emotional_cost(item, pressure_level)
    benefit_level = _benefit_level(item)
    priority_score = _priority(
        action_clarity=action_clarity,
        deadline_clarity=deadline_clarity,
        owner_clarity=owner_clarity,
        benefit_level=benefit_level,
        pressure_level=pressure_level,
        autonomy_level=autonomy_level,
        time_cost=time_cost,
        cognitive_cost=cognitive_cost,
        emotional_cost=emotional_cost,
    )

    return item.model_copy(
        update={
            "scores": CommitmentScores(
                action_clarity=action_clarity,
                deadline_clarity=deadline_clarity,
                owner_clarity=owner_clarity,
                pressure_level=pressure_level,
                autonomy_level=autonomy_level,
                time_cost=time_cost,
                cognitive_cost=cognitive_cost,
                emotional_cost=emotional_cost,
                benefit_level=benefit_level,
                priority_score=priority_score,
            )
        }
    )


def _action_clarity(
    item: CommitmentItem,
    known: dict[str, object],
    missing: set[str],
) -> float:
    if "action" in known:
        return 0.8
    if item.type is CommitmentType.CHEAP_SIGNAL:
        return 0.2
    if "action" in missing:
        return LOW
    return MEDIUM


def _deadline_clarity(known: dict[str, object], missing: set[str]) -> float:
    if "deadline" in known:
        return HIGH
    if "deadline" in missing:
        return 0.1
    return MEDIUM


def _owner_clarity(item: CommitmentItem) -> float:
    if item.direction is CommitmentDirection.UNCLEAR:
        return 0.2
    if item.direction is CommitmentDirection.MUTUAL:
        return 0.6
    return 0.75


def _pressure_level(item: CommitmentItem, evidence_labels: set[str]) -> float:
    pressure_level = 0.75 if "pressure" in evidence_labels else LOW
    if item.type is CommitmentType.COMMAND:
        return max(pressure_level, 0.8)
    if item.type is CommitmentType.REQUEST:
        return max(pressure_level, 0.4)
    if item.type is CommitmentType.CHEAP_SIGNAL:
        return max(pressure_level, 0.25)
    return pressure_level


def _autonomy_level(item: CommitmentItem, pressure_level: float) -> float:
    autonomy_level = 1.0 - pressure_level
    if item.type is CommitmentType.COMMAND:
        return min(autonomy_level, 0.2)
    if item.type is CommitmentType.REQUEST:
        return min(autonomy_level, 0.65)
    if item.type in {CommitmentType.PROMISE, CommitmentType.INTENTION}:
        return max(autonomy_level, 0.65)
    return max(0.0, min(autonomy_level, 1.0))


def _time_cost(item: CommitmentItem) -> float:
    text = item.raw_text.casefold()
    if any(
        word in text
        for word in (
            "mail",
            "sms",
            "zadzwonie",
            "zadzwonię",
            "wysle",
            "wyślę",
            "potwierdze",
            "potwierdzę",
        )
    ):
        return 0.25
    if any(
        word in text
        for word in (
            "samochod",
            "samochód",
            "pojade",
            "pojadę",
            "oddam",
            "przygotuje",
            "przygotuję",
            "wypelnie",
            "wypełnię",
        )
    ):
        return 0.6
    return 0.35


def _cognitive_cost(item: CommitmentItem, missing: set[str]) -> float:
    if item.type is CommitmentType.CHEAP_SIGNAL:
        return 0.75
    base = 0.35 + 0.15 * min(len(missing), 3)
    return min(base, 0.9)


def _emotional_cost(item: CommitmentItem, pressure_level: float) -> float:
    if item.type is CommitmentType.COMMAND:
        return max(0.55, pressure_level * 0.85)
    if item.type is CommitmentType.REQUEST:
        return max(0.35, pressure_level * 0.75)
    return max(0.2, pressure_level * 0.7)


def _benefit_level(item: CommitmentItem) -> float:
    if item.type in {CommitmentType.PROMISE, CommitmentType.INTENTION}:
        return 0.65
    if item.type in {CommitmentType.REQUEST, CommitmentType.COMMAND}:
        return 0.45
    if item.type is CommitmentType.CHEAP_SIGNAL:
        return 0.35
    return 0.4


def _priority(
    *,
    action_clarity: float,
    deadline_clarity: float,
    owner_clarity: float,
    benefit_level: float,
    pressure_level: float,
    autonomy_level: float,
    time_cost: float,
    cognitive_cost: float,
    emotional_cost: float,
) -> float:
    clarity = (action_clarity + deadline_clarity + owner_clarity) / 3
    total_cost = (time_cost + cognitive_cost + emotional_cost) / 3
    raw = (
        0.30 * deadline_clarity
        + 0.25 * benefit_level
        + 0.20 * clarity
        + 0.15 * pressure_level
        + 0.10 * total_cost
        - 0.05 * (1.0 - autonomy_level)
    )
    return max(0.0, min(raw, 1.0))
