import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.corpus.analysis import (
    aggregate_routing_metrics,
    build_routing_eval_records,
    build_style_profile,
    character_error_rate,
    evaluate_style_profile,
    routing_holdout_action_ids,
    score_routing_result,
    word_error_rate,
)
from voiceloop.corpus.dedupe import assign_splits, deduplicate
from voiceloop.corpus.extract import (
    SourceChangedError,
    extract_audio_file,
    extract_cursor_file,
    extract_from_manifest,
)
from voiceloop.corpus.inventory import build_manifest, inspect_text_source
from voiceloop.corpus.local_only import (
    LocalOnlyViolation,
    load_style_profile,
    require_loopback_url,
)
from voiceloop.corpus.pipeline import evaluate_routing, evaluate_routing_v2
from voiceloop.corpus.privacy import apply_privacy_gate, redact_text
from voiceloop.corpus.schema import (
    CorpusSplit,
    ExpectedIntent,
    RoutingEvalRecord,
    RoutingV2RuntimeConfig,
    SourceKind,
    SpeakerStatus,
    SttErrorType,
    UtteranceOrigin,
    UtteranceRecord,
)
from voiceloop.corpus.storage import sha256_text
from voiceloop.memory import MemoryStore
from voiceloop.settings import Settings


def utterance(
    text: str,
    *,
    utterance_id: str = "u1",
    speaker: SpeakerStatus = SpeakerStatus.SELF,
) -> UtteranceRecord:
    return UtteranceRecord(
        utterance_id=utterance_id,
        source_id="source",
        origin=UtteranceOrigin.CURSOR_USER,
        session_id="session",
        captured_at=datetime.now(UTC),
        word_count=len(text.split()),
        char_count=len(text),
        text_sha256=sha256_text(text.casefold()),
        text=text,
        speaker_status=speaker,
    )


def test_inventory_contains_hashes_and_counts_but_no_text(tmp_path) -> None:
    audio = tmp_path / "transcript.txt"
    audio.write_text("# 2026-08-17\nala ma kota", encoding="utf-8")
    cursor_root = tmp_path / "cursor"
    transcript = cursor_root / "project" / "agent-transcripts" / "chat" / "chat.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps({"role": "user", "message": {"content": "krótki test"}}) + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest(
        audio_transcript=audio,
        cursor_projects_root=cursor_root,
    )
    payload = manifest.model_dump_json()

    assert manifest.unique_source_count == 2
    assert manifest.included_word_count >= 5
    assert '"sha256"' in payload
    assert "ala ma kota" not in payload
    assert "krótki test" not in payload


def test_overlap_chunk_is_excluded(tmp_path) -> None:
    chunk = tmp_path / "output" / "chunks" / "chunk_001.txt"
    chunk.parent.mkdir(parents=True)
    chunk.write_text("raz dwa trzy", encoding="utf-8")

    record = inspect_text_source(chunk, kind=SourceKind.AUDIO_TRANSCRIPT)

    assert record.included is False
    assert record.exclude_reason == "overlap_chunk"


def test_extraction_rejects_source_changed_after_manifest(tmp_path) -> None:
    audio = tmp_path / "transcript.txt"
    audio.write_text("pierwsza wersja", encoding="utf-8")
    manifest = build_manifest(audio_transcript=audio, cursor_projects_root=None)
    audio.write_text("zmieniona wersja", encoding="utf-8")

    with pytest.raises(SourceChangedError):
        extract_from_manifest(manifest)


def test_cursor_parser_keeps_query_and_excludes_code(tmp_path) -> None:
    path = tmp_path / "chat.jsonl"
    rows = [
        {
            "role": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<timestamp>2026-08-17T10:00:00+00:00</timestamp>"
                            "<user_query>Otwórz kalendarz.</user_query>"
                        ),
                    }
                ]
            },
        },
        {
            "role": "user",
            "message": {"content": "```python\ndef unsafe():\n    pass\n```"},
        },
        {"role": "assistant", "message": {"content": "Nie licz mnie."}},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = extract_cursor_file(path, "source")
    changed_source_hash = extract_cursor_file(path, "different-source")

    assert [record.text for record in result.records] == ["Otwórz kalendarz."]
    assert result.records[0].speaker_status is SpeakerStatus.SELF
    assert result.records[0].session_id == changed_source_hash.records[0].session_id
    assert result.technical_excluded_count == 1


def test_audio_parser_is_fail_closed_for_speaker(tmp_path) -> None:
    path = tmp_path / "transcript.txt"
    path.write_text(
        "# 2026-08-17\n\nPierwszy blok testowy.\n\nDrugi blok.",
        encoding="utf-8",
    )

    result = extract_audio_file(path, "audio")
    gate = apply_privacy_gate(result.records[0])

    assert len(result.records) == 2
    assert result.records[0].speaker_status is SpeakerStatus.UNKNOWN
    assert gate.clean is None
    assert gate.quarantine is not None
    assert gate.quarantine.reason == "speaker_unknown"
    assert not hasattr(gate.quarantine, "text")


def test_audio_source_requires_explicit_speaker_decision(tmp_path) -> None:
    path = tmp_path / "transcript.txt"
    path.write_text("Krótka własna wypowiedź.", encoding="utf-8")

    approved = extract_audio_file(
        path,
        "audio",
        speaker_status=SpeakerStatus.SELF,
    )
    gate = apply_privacy_gate(approved.records[0])

    assert gate.clean is not None
    assert gate.quarantine is None


def test_dedupe_and_split_keep_sessions_together() -> None:
    first = utterance("Otwórz kalendarz teraz", utterance_id="first")
    duplicate = utterance("Otwórz kalendarz teraz", utterance_id="duplicate")
    other = utterance("Zamknij aktywne okno", utterance_id="other")

    records = assign_splits(deduplicate([first, duplicate, other]))

    assert records[1].duplicate_of == "first"
    assert records[1].is_near_duplicate is True
    assert records[1].split.value == "unused"
    assert records[0].split.value in {"train", "holdout"}
    assert records[0].split == records[2].split


def test_short_near_duplicates_are_clustered_before_split() -> None:
    first = utterance("Otwórz teraz kalendarz", utterance_id="short-first")
    near = utterance("Otwórz kalendarz", utterance_id="short-near")

    records = deduplicate([first, near])

    assert records[1].duplicate_of == "short-first"
    assert records[1].is_near_duplicate is True


def test_short_dedupe_compares_five_and_four_word_phrases() -> None:
    five_words = utterance(
        "Otwórz teraz kartę z YouTube",
        utterance_id="five",
    )
    four_words = utterance(
        "Otwórz kartę z YouTube",
        utterance_id="four",
    )

    records = deduplicate([five_words, four_words])

    assert records[1].duplicate_of == "five"


def test_redaction_and_medical_gate() -> None:
    redacted, flags = redact_text("Napisz do mnie na jan@example.com.")
    medical = apply_privacy_gate(utterance("Pacjent ma diagnozę testową.", utterance_id="medical"))

    assert redacted == "Napisz do mnie na [EMAIL]."
    assert flags == ["email"]
    assert medical.clean is None
    assert medical.quarantine is not None
    assert medical.quarantine.reason == "medical_or_third_party"


def test_valid_pesel_is_quarantined() -> None:
    result = apply_privacy_gate(utterance("Mój numer to 44051401458.", utterance_id="pesel"))

    assert result.clean is None
    assert result.quarantine is not None
    assert "pesel" in result.quarantine.pii_flags


def test_style_profile_contains_aggregates_not_quotes() -> None:
    records = [utterance("Otwórz kalendarz.", utterance_id=f"u{index}") for index in range(12)]

    profile = build_style_profile(records, manifest_id="manifest")
    payload = profile.model_dump_json()

    assert profile.utterance_count == 12
    assert profile.directness > 0.9
    assert "Otwórz kalendarz" not in payload
    assert profile.enabled is False


def test_style_profile_uses_train_and_requires_passing_holdout(tmp_path) -> None:
    train = [
        utterance("Otwórz kalendarz.", utterance_id=f"train-{index}").model_copy(
            update={"split": CorpusSplit.TRAIN}
        )
        for index in range(24)
    ]
    holdout = [
        utterance("Otwórz notatnik.", utterance_id=f"holdout-{index}").model_copy(
            update={"split": CorpusSplit.HOLDOUT}
        )
        for index in range(24)
    ]
    profile = build_style_profile(
        train + holdout,
        manifest_id="manifest",
        enabled=True,
    )
    report = evaluate_style_profile(profile, train + holdout)
    profile_path = tmp_path / "profile-v1.json"
    report_path = tmp_path / "holdout-report-v1.json"
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    report_path.write_text(report.model_dump_json(), encoding="utf-8")

    loaded = load_style_profile(profile_path, enabled=True)

    assert profile.utterance_count == len(train)
    assert report.passes_quality_gate is True
    assert loaded is not None
    assert loaded.profile_id == profile.profile_id


def test_eval_records_and_metrics() -> None:
    definitions = [
        {
            "id": "open_browser",
            "routing_examples": ["otwórz przeglądarkę"],
        }
    ]
    records = build_routing_eval_records(definitions)
    card = score_routing_result(
        records[0],
        [("open_browser", 0.9), ("open_url", 0.7)],
        min_score=0.2,
        margin_threshold=0.05,
        stt_threshold=0.75,
    )
    metrics = aggregate_routing_metrics([card])

    assert len(records) == 6
    assert all("manual_holdout" in record.tags for record in records)
    assert all(record.gold_text.casefold() != "otwórz przeglądarkę" for record in records)
    assert card.margin_top2 == pytest.approx(0.2)
    assert card.hit_at_1 is True
    assert metrics.top1_accuracy == 1.0
    assert metrics.catalog_coverage == 1.0
    assert metrics.quality_gate_passed is False
    assert "base_example_count_below_6" in metrics.quality_gate_failures
    assert word_error_rate("ala ma kota", "ala kota") == pytest.approx(1 / 3)
    assert character_error_rate("kot", "kat") == pytest.approx(1 / 3)


def test_manual_routing_holdout_covers_entire_capability_catalog(tmp_path) -> None:
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        MemoryStore(tmp_path / "offline.db"),
        SimpleNamespace(),  # type: ignore[arg-type]
    )
    definitions = registry.capability_catalog()["voiceloop_actions"]
    catalog_ids = {str(definition["id"]) for definition in definitions}

    records = build_routing_eval_records(definitions)
    bases_by_action = {
        action_id: {
            record.base_example_id for record in records if record.expected_action_id == action_id
        }
        for action_id in catalog_ids
    }

    assert routing_holdout_action_ids() == catalog_ids
    assert all(len(base_ids) >= 2 for base_ids in bases_by_action.values())


@pytest.mark.asyncio
async def test_routing_evaluator_uses_topk_and_margin() -> None:
    example = RoutingEvalRecord(
        example_id="open",
        gold_text="otwórz przeglądarkę",
        stt_text="otworz przegladarke",
        stt_confidence=0.85,
        error_type=SttErrorType.FLEXION,
        expected_action_id="open_browser",
        expected_intent=ExpectedIntent.TASK,
    )

    async def search(_text: str):
        return SimpleNamespace(
            matches=[
                SimpleNamespace(action_id="open_browser", score=0.82),
                SimpleNamespace(action_id="open_url", score=0.51),
            ]
        )

    cards, metrics = await evaluate_routing(
        records=[example],
        search=search,
        min_score=0.2,
        margin_threshold=0.05,
        stt_threshold=0.75,
        expected_action_ids={"open_browser", "open_url"},
    )

    assert cards[0].topk_action_ids == ("open_browser", "open_url")
    assert cards[0].margin_top2 == pytest.approx(0.31)
    assert metrics.top1_accuracy == 1.0
    assert metrics.topk_recall == 1.0
    assert metrics.catalog_coverage == 0.5
    assert "catalog_action_coverage_incomplete" in metrics.quality_gate_failures


@pytest.mark.asyncio
async def test_routing_v2_quality_gate_requires_resolution_and_safe_abstention() -> None:
    records = [
        RoutingEvalRecord(
            example_id="open-1",
            base_example_id="open-base-1",
            gold_text="uruchom przeglądarkę",
            stt_text="uruchom przeglądarkę",
            stt_confidence=0.96,
            expected_action_id="open_browser",
            expected_intent=ExpectedIntent.TASK,
        ),
        RoutingEvalRecord(
            example_id="open-2",
            base_example_id="open-base-2",
            gold_text="włącz program do internetu",
            stt_text="włącz program do internetu",
            stt_confidence=0.94,
            expected_action_id="open_browser",
            expected_intent=ExpectedIntent.TASK,
        ),
        RoutingEvalRecord(
            example_id="open-low",
            base_example_id="open-base-1",
            gold_text="uruchom przeglądarkę",
            stt_text="uruchom",
            stt_confidence=0.40,
            error_type=SttErrorType.LOW_CONFIDENCE,
            expected_action_id="open_browser",
            expected_intent=ExpectedIntent.TASK,
        ),
        RoutingEvalRecord(
            example_id="conversation-1",
            base_example_id="conversation-1",
            gold_text="halo",
            stt_text="halo",
            stt_confidence=0.96,
            expected_intent=ExpectedIntent.CONVERSATION,
        ),
    ]

    async def route(request):
        is_conversation = request.text == "halo"
        should_abstain = request.transcript_confidence < 0.75 or is_conversation
        candidate = SimpleNamespace(action_id="open_browser")
        decision = SimpleNamespace(
            decision=SimpleNamespace(
                value="unsupported"
                if is_conversation
                else "clarify"
                if request.transcript_confidence < 0.75
                else "resolved"
            ),
            reason=(
                "no_command_operation"
                if is_conversation
                else "low_stt_confidence"
                if request.transcript_confidence < 0.75
                else None
            ),
            candidates=[] if is_conversation else [candidate],
        )
        plan = (
            None
            if should_abstain
            else SimpleNamespace(steps=[SimpleNamespace(action_id="open_browser", args={})])
        )
        return SimpleNamespace(
            plan=plan,
            decisions=(decision,),
            segmentation=SimpleNamespace(decision=SimpleNamespace(value="simple")),
            blocked_reason=decision.reason,
        )

    cards, metrics = await evaluate_routing_v2(
        records=records,
        route=route,
        stt_threshold=0.75,
        expected_action_ids={"open_browser"},
        catalog_hash="catalog",
        runtime_config={
            "candidate_limit": 10,
            "execute_min_score": 0.50,
            "execute_min_margin": 0.10,
            "stt_threshold": 0.75,
            "max_subtasks": 12,
            "embedding_model": "fake-embedding",
            "embedding_dimension": 3,
            "capability_collection": "fake-capabilities",
            "catalog_hash": "catalog",
        },
    )

    assert len(cards) == 4
    assert metrics.catalog_coverage == 1.0
    assert metrics.resolved_accuracy == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.expected_calibration_error == 0.0
    assert metrics.safe_abstention_recall == 1.0
    assert metrics.unsafe_resolution_count == 0
    assert metrics.quality_gate_passed is True


@pytest.mark.asyncio
async def test_routing_v2_counts_wrong_executed_plan_as_unsafe() -> None:
    records = [
        RoutingEvalRecord(
            example_id="wrong-plan",
            base_example_id="open-base-1",
            gold_text="uruchom przeglądarkę",
            stt_text="uruchom przeglądarkę",
            stt_confidence=0.96,
            expected_action_id="open_browser",
            expected_intent=ExpectedIntent.TASK,
        )
    ]

    async def route(_request):
        candidate = SimpleNamespace(
            action_id="open_browser",
            combined_score=0.99,
        )
        decision = SimpleNamespace(
            decision=SimpleNamespace(value="resolved"),
            reason=None,
            candidates=[candidate],
            top1_action_id="open_browser",
            margin_top2=0.50,
        )
        return SimpleNamespace(
            plan=SimpleNamespace(
                steps=[
                    SimpleNamespace(action_id="open_browser", args={}),
                    SimpleNamespace(
                        action_id="close_window_under_cursor",
                        args={},
                    ),
                ]
            ),
            decisions=(decision,),
            segmentation=SimpleNamespace(decision=SimpleNamespace(value="simple")),
            blocked_reason=None,
        )

    cards, metrics = await evaluate_routing_v2(
        records=records,
        route=route,
        stt_threshold=0.75,
        expected_action_ids={"open_browser"},
        catalog_hash="catalog",
        runtime_config={
            "candidate_limit": 10,
            "execute_min_score": 0.50,
            "execute_min_margin": 0.10,
            "stt_threshold": 0.75,
            "max_subtasks": 12,
            "embedding_model": "fake-embedding",
            "embedding_dimension": 3,
            "capability_collection": "fake-capabilities",
            "catalog_hash": "catalog",
        },
    )

    assert cards[0].exact_plan_match is False
    assert cards[0].unsafe_resolution is True
    assert metrics.unsafe_resolution_count == 1
    assert "unsafe_resolution_detected" in metrics.quality_gate_failures
    assert metrics.quality_gate_passed is False


def test_eval_schema_rejects_ambiguous_task() -> None:
    with pytest.raises(ValueError, match="intent=ambiguous"):
        RoutingEvalRecord(
            example_id="bad",
            gold_text="zrób coś",
            stt_text="zrób coś",
            stt_confidence=0.5,
            error_type=SttErrorType.LOW_CONFIDENCE,
            expected_intent=ExpectedIntent.TASK,
            expected_action_id="open_browser",
            ambiguous=True,
        )


def test_local_url_guard_rejects_lan_and_cloud() -> None:
    assert require_loopback_url("http://127.0.0.1:1234/v1").startswith("http://")
    assert require_loopback_url("http://localhost:1234/v1").startswith("http://")
    with pytest.raises(LocalOnlyViolation):
        require_loopback_url("http://192.168.0.183:1234/v1")
    with pytest.raises(LocalOnlyViolation):
        require_loopback_url("https://example.com/v1")


def test_routing_runtime_fingerprint_covers_algorithm_contract() -> None:
    base = RoutingV2RuntimeConfig(
        candidate_limit=10,
        execute_min_score=0.50,
        execute_min_margin=0.10,
        stt_threshold=0.75,
        max_subtasks=12,
        embedding_model="fake-embedding",
        embedding_dimension=768,
        capability_collection="voiceloop_capabilities_v2",
        catalog_hash="catalog",
    )

    assert base.fingerprint() != base.model_copy(
        update={"taxonomy_version": "routing-taxonomy-v3"}
    ).fingerprint()
    assert base.fingerprint() != base.model_copy(
        update={"vector_weights": {**base.vector_weights, "semantic": 0.5}}
    ).fingerprint()
