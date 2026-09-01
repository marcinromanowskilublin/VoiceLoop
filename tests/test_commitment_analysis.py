from voiceloop.commitments import (
    CommitmentDirection,
    CommitmentStatus,
    CommitmentType,
    TranscriptChunk,
    analyze_commitments,
)


def analyze_one(text: str, *, speaker: str = "user"):
    result = analyze_commitments(
        [TranscriptChunk(chunk_id="ch_1", speaker=speaker, text=text)],
        user_speakers={"user"},
    )
    assert len(result.items) == 1
    return result.items[0]


def test_vague_intention_keeps_deadline_and_requests_clarification() -> None:
    item = analyze_one("Dobra, postaram się to ogarnąć w piątek.")

    assert item.type is CommitmentType.CHEAP_SIGNAL
    assert item.status is CommitmentStatus.NEEDS_CLARIFICATION
    assert item.known_elements["deadline"] == "w piątek"
    assert "action" in item.missing_elements
    assert "verification_method" in item.missing_elements
    assert item.scores.cognitive_cost > 0.7


def test_user_promise_with_deadline_is_captured_but_not_executed() -> None:
    item = analyze_one("Wyślę ci dokumenty do piątku.")

    assert item.type is CommitmentType.INTENTION
    assert item.direction is CommitmentDirection.USER_TO_OTHER
    assert item.status is CommitmentStatus.NEEDS_CLARIFICATION
    assert item.known_elements["action"] == "wyślę"
    assert item.known_elements["deadline"] == "do piątku"
    assert item.scores.deadline_clarity > 0.8


def test_request_to_user_requires_review_instead_of_acceptance() -> None:
    item = analyze_one("Wyślij mi dokumenty.", speaker="speaker_2")

    assert item.type is CommitmentType.REQUEST
    assert item.direction is CommitmentDirection.OTHER_TO_USER
    assert item.status is CommitmentStatus.NEEDS_USER_REVIEW
    assert item.scores.autonomy_level < 0.7


def test_command_to_user_has_pressure_and_requires_review() -> None:
    item = analyze_one("Musisz mi to wysłać dzisiaj.", speaker="speaker_2")

    assert item.type is CommitmentType.COMMAND
    assert item.direction is CommitmentDirection.OTHER_TO_USER
    assert item.status is CommitmentStatus.NEEDS_USER_REVIEW
    assert item.scores.pressure_level >= 0.75
    assert item.scores.priority_score > 0.5
