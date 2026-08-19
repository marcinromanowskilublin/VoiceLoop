from types import SimpleNamespace

import pytest

from voiceloop.app import (
    approve_memory_candidate,
    create_memory_candidate,
    reject_memory_candidate,
)
from voiceloop.corpus.candidates import (
    CandidateDecisionError,
    CandidateStoreError,
    MemoryCandidateStore,
    extract_memory_candidates,
)
from voiceloop.corpus.schema import (
    CandidateStatus,
    CorpusSplit,
    MemoryCandidateApprovalRequest,
    MemoryCandidateCreate,
    MemoryCandidateKind,
    SpeakerStatus,
    UtteranceOrigin,
    UtteranceRecord,
)
from voiceloop.corpus.storage import sha256_text
from voiceloop.memory import MemoryStore
from voiceloop.settings import Settings


def utterance(text: str, *, utterance_id: str = "u1") -> UtteranceRecord:
    return UtteranceRecord(
        utterance_id=utterance_id,
        source_id="source",
        origin=UtteranceOrigin.CURSOR_USER,
        session_id="session",
        word_count=len(text.split()),
        char_count=len(text),
        text_sha256=sha256_text(text.casefold()),
        text=text,
        speaker_status=SpeakerStatus.SELF,
        split=CorpusSplit.TRAIN,
    )


async def test_candidate_must_be_explicitly_approved(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    memory = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.set_active_scope("run-current")
    await memory.initialize()
    candidate = MemoryCandidateCreate(
        candidate_id="candidate-1",
        kind=MemoryCandidateKind.PREFERENCE,
        proposed_content="Wolę krótkie odpowiedzi.",
        evidence_utterance_ids=("u1",),
    )
    stored = await store.upsert(candidate)

    assert await memory.list_memories() == []
    approved, item = await store.approve(
        "candidate-1",
        memory,
        expected_content_sha256=stored.content_sha256,
    )

    assert approved.status is CandidateStatus.APPROVED
    assert approved.memory_id == item.id
    assert item.source == "corpus_approved"
    assert item.sensitivity == "private"

    repeated, repeated_item = await store.approve(
        "candidate-1",
        memory,
        expected_content_sha256=stored.content_sha256,
    )
    assert repeated.memory_id == repeated_item.id == item.id
    assert len(await memory.list_memories()) == 1


async def test_rejected_candidate_never_reaches_memory(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    memory = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.set_active_scope("run-current")
    await memory.initialize()
    stored = await store.upsert(
        MemoryCandidateCreate(
            candidate_id="candidate-2",
            kind=MemoryCandidateKind.FACT,
            proposed_content="Moim celem jest test.",
        )
    )

    rejected = await store.reject("candidate-2")

    assert rejected.status is CandidateStatus.REJECTED
    assert await memory.list_memories() == []
    with pytest.raises(CandidateDecisionError):
        await store.approve(
            "candidate-2",
            memory,
            expected_content_sha256=stored.content_sha256,
        )


async def test_blocked_candidate_cannot_be_approved(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    memory = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.set_active_scope("run-current")
    await memory.initialize()
    stored = await store.upsert(
        MemoryCandidateCreate(
            candidate_id="blocked",
            kind=MemoryCandidateKind.FACT,
            proposed_content="Zablokowana treść.",
            status=CandidateStatus.BLOCKED,
            block_reason="medical_or_third_party",
        )
    )

    with pytest.raises(CandidateDecisionError):
        await store.approve(
            "blocked",
            memory,
            expected_content_sha256=stored.content_sha256,
        )


def test_candidate_extraction_is_conservative() -> None:
    records = [
        utterance("Wolę krótkie odpowiedzi.", utterance_id="preference"),
        utterance("To jest zwykłe zdanie.", utterance_id="ordinary"),
        utterance("Moim celem jest ukończenie VoiceLoop.", utterance_id="fact"),
    ]

    candidates = extract_memory_candidates(records)

    assert [item.kind for item in candidates] == [
        MemoryCandidateKind.PREFERENCE,
        MemoryCandidateKind.FACT,
    ]
    assert all(item.status is CandidateStatus.PENDING for item in candidates)


async def test_protected_api_flow_blocks_sensitive_candidate(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    memory = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.set_active_scope("run-current")
    await memory.initialize()
    services = SimpleNamespace(
        settings=Settings(voiceloop_data_dir=str(tmp_path), corpus_enabled=True),
        corpus_candidates=store,
        memory=memory,
    )
    sensitive = await create_memory_candidate(
        MemoryCandidateCreate(
            candidate_id="medical",
            kind=MemoryCandidateKind.FACT,
            proposed_content="Pacjent ma diagnozę testową.",
        ),
        services,
    )
    pending = await create_memory_candidate(
        MemoryCandidateCreate(
            candidate_id="preference",
            kind=MemoryCandidateKind.PREFERENCE,
            proposed_content="Wolę krótkie odpowiedzi.",
        ),
        services,
    )

    assert sensitive.status is CandidateStatus.BLOCKED
    assert pending.status is CandidateStatus.PENDING
    approval = await approve_memory_candidate(
        "preference",
        MemoryCandidateApprovalRequest(
            content_sha256=pending.content_sha256,
        ),
        services,
    )
    assert approval["candidate"]["status"] == "approved"
    rejected = await reject_memory_candidate(
        (
            await store.upsert(
                MemoryCandidateCreate(
                    candidate_id="reject-me",
                    kind=MemoryCandidateKind.FACT,
                    proposed_content="Moim celem jest test.",
                )
            )
        ).candidate_id,
        services,
    )
    assert rejected.status is CandidateStatus.REJECTED


async def test_candidate_content_is_redacted_immutable_and_hash_bound(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    memory = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    await store.set_active_scope("run-current")
    await memory.initialize()
    stored = await store.upsert(
        MemoryCandidateCreate(
            candidate_id="immutable",
            kind=MemoryCandidateKind.PREFERENCE,
            proposed_content="Wolę kontakt przez jan@example.com.",
            manifest_id="run-a",
        )
    )

    assert stored.proposed_content == "Wolę kontakt przez [EMAIL]."
    with pytest.raises(CandidateStoreError):
        await store.upsert(
            MemoryCandidateCreate(
                candidate_id="immutable",
                kind=MemoryCandidateKind.PREFERENCE,
                proposed_content="Wolę zupełnie inną treść.",
                manifest_id="run-a",
            )
        )
    with pytest.raises(CandidateDecisionError, match="Hash treści"):
        await store.approve(
            "immutable",
            memory,
            expected_content_sha256="0" * 64,
        )
    with pytest.raises(CandidateDecisionError, match="bieżącego zakresu"):
        await store.approve(
            "immutable",
            memory,
            expected_content_sha256=stored.content_sha256,
        )
    assert await memory.list_memories() == []


async def test_stale_manifest_candidates_are_blocked(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    await store.initialize()
    await store.set_active_scope("run-old")
    await store.upsert(
        MemoryCandidateCreate(
            candidate_id="old-run",
            kind=MemoryCandidateKind.FACT,
            proposed_content="Moim celem jest test.",
            manifest_id="run-old",
        )
    )

    blocked = await store.set_active_scope("run-new")
    candidate = await store.get("old-run")

    assert blocked == 1
    assert candidate is not None
    assert candidate.status is CandidateStatus.BLOCKED
    assert candidate.block_reason == "stale_manifest"
    assert candidate.proposed_content == "[BLOCKED_SENSITIVE_CONTENT]"


async def test_candidate_without_active_scope_is_blocked(tmp_path) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    await store.initialize()

    candidate = await store.upsert(
        MemoryCandidateCreate(
            candidate_id="no-scope",
            kind=MemoryCandidateKind.PREFERENCE,
            proposed_content="Wolę krótkie odpowiedzi.",
        )
    )

    assert candidate.status is CandidateStatus.BLOCKED
    assert candidate.block_reason == "missing_active_scope"


@pytest.mark.parametrize(
    "text",
    [
        "Mam cukrzycę.",
        "Przyjmuję insulinę.",
        "Biorę metforminę.",
        "Jestem w ciąży.",
    ],
)
async def test_self_diagnosis_is_blocked_before_review(tmp_path, text) -> None:
    store = MemoryCandidateStore(tmp_path / "candidates.db")
    await store.initialize()
    await store.set_active_scope("run-current")

    candidate = await store.upsert(
        MemoryCandidateCreate(
            candidate_id="health",
            kind=MemoryCandidateKind.FACT,
            proposed_content=text,
        )
    )

    assert candidate.status is CandidateStatus.BLOCKED
    assert candidate.block_reason == "medical_sensitive"
    assert candidate.proposed_content == "[BLOCKED_SENSITIVE_CONTENT]"
