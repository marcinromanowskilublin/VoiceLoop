from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from voiceloop.capability_index import (
    CapabilityDocuments,
    CapabilityMatch,
    CapabilitySearchResult,
    SubtaskCapabilitySearch,
)
from voiceloop.corpus.cli import _run_async, build_parser, main
from voiceloop.corpus.routing_calibration import (
    derive_plan_labels,
    evaluate_routing_calibration,
    fit_routing_calibration,
    temporal_group_split,
    validate_calibration_artifact_for_evaluation,
    validate_calibration_observations,
)
from voiceloop.corpus.schema import (
    RoutingCalibrationArtifactStatus,
    RoutingCalibrationArtifactV1,
    RoutingCalibrationClassCountsV1,
    RoutingCalibrationCoefficientsV1,
    RoutingCalibrationCredibilityGatesV1,
    RoutingCalibrationObservationV1,
    RoutingCalibrationSafetyOutcome,
    RoutingCalibrationSetRole,
)
from voiceloop.models import (
    CommandRequest,
    CommandSource,
    ResolutionCandidateV1,
    ResolutionDecisionV1,
    ResolutionStatusV1,
    SubtaskEmbeddingV1,
)
from voiceloop.routing.calibration import (
    RoutingCalibrationObservationStore,
    RoutingCalibrationRecorder,
    RoutingCalibrationRuntime,
    RoutingCalibrationRuntimeStatus,
    build_calibration_observations,
)
from voiceloop.routing.segmenter import segment_command
from voiceloop.routing.service import RoutingQualityGate, RoutingV2Outcome, RoutingV2Service
from voiceloop.settings import Settings


def _hex(symbol: str) -> str:
    return symbol * 64


def _observation(
    index: int,
    *,
    score: float,
    action_correct: bool | None,
    observed_at: datetime,
    group_id: str | None = None,
    source: str = "shadow_runtime",
    role: RoutingCalibrationSetRole = RoutingCalibrationSetRole.REPRESENTATIVE,
    runtime_fingerprint: str = _hex("a"),
    catalog_hash: str = "catalog",
    safety_outcome: RoutingCalibrationSafetyOutcome | None = None,
    predicted_action_id: str | None = "open_browser",
    expected_action_id: str | None = None,
    session_id: str | None = None,
    dataset_id: str = "routing_v2_shadow_runtime",
    source_record_id: str | None = None,
    split_group_override: str | None = None,
) -> RoutingCalibrationObservationV1:
    observation = RoutingCalibrationObservationV1(
        request_id=f"request-{index}",
        dataset_id=dataset_id,
        source_record_id=source_record_id or f"record-{index}",
        observed_at=observed_at,
        source=source,
        normalized_text_sha256=_hex("b"),
        runtime_fingerprint=runtime_fingerprint,
        catalog_hash=catalog_hash,
        session_id=session_id,
        split_group_override=split_group_override,
        group_id=group_id or f"group-{index}",
        set_role=role,
        subtask_index=0,
        subtask_count=1,
        decision="resolved",
        predicted_action_id=predicted_action_id,
        expected_action_id=expected_action_id,
        ranking_score=score,
        margin_top2=max(0.0, min(score * 0.5, 1.0)),
        vector_score=score,
        lexical_score=min(1.0, score + 0.1),
        argument_score=1.0,
        vector_coverage=1.0,
        stt_confidence=0.95,
        candidate_count=2,
        eligible=True,
        rejection_reasons=(),
        action_correct=action_correct,
        action_sequence_correct=action_correct if action_correct is not None else None,
        exact_plan_correct=action_correct if action_correct is not None else None,
        safety_outcome=safety_outcome,
        snapshot_sha256=None,
    )
    return observation.model_copy(
        update={"snapshot_sha256": observation.recompute_snapshot_sha256()}
    )


def _artifact(
    *,
    runtime_fingerprint: str,
    catalog_hash: str,
    status: RoutingCalibrationArtifactStatus = RoutingCalibrationArtifactStatus.READY,
    a: float = 4.0,
    b: float = -2.0,
) -> RoutingCalibrationArtifactV1:
    gates = RoutingCalibrationCredibilityGatesV1(
        point_ece_le_0_10=True,
        ci_upper_ece_le_0_15=True,
        brier_improves_over_raw=True,
        logloss_not_worse_than_intercept=True,
        passed=True,
    )
    train_component_ids = tuple(f"comp-train-{index:04d}" for index in range(240))
    eval_component_ids = tuple(f"comp-eval-{index:04d}" for index in range(60))
    artifact = RoutingCalibrationArtifactV1(
        status=status,
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        coefficients=RoutingCalibrationCoefficientsV1(a=a, b=b),
        class_counts=RoutingCalibrationClassCountsV1(successes=120, errors=120),
        group_count=240,
        dataset_sha256=_hex("c"),
        split_sha256=_hex("d"),
        training_window_start=datetime(2026, 1, 1, tzinfo=UTC),
        training_window_end=datetime(2026, 2, 1, tzinfo=UTC),
        evaluation_window_start=datetime(2026, 2, 2, tzinfo=UTC),
        evaluation_window_end=datetime(2026, 3, 1, tzinfo=UTC),
        train_cutoff=datetime(2026, 2, 1, tzinfo=UTC),
        runtime_fingerprint=runtime_fingerprint,
        catalog_hash=catalog_hash,
        train_component_ids=train_component_ids,
        evaluation_component_ids=eval_component_ids,
        excluded_component_ids=(),
        train_group_tokens=("g:train-hash", "s:2026-01-01T00:00:00+00:00"),
        credibility_gates=gates,
        metrics={
            "eval_brier_score": 0.1,
            "eval_equal_mass_ece": 0.05,
            "point_ece": 0.05,
            "ece_ci95_upper": 0.10,
            "brier_delta_vs_raw": -0.02,
            "brier_delta_vs_raw_ci95_upper": -0.005,
            "log_loss_delta_vs_intercept": -0.01,
            "log_loss_delta_vs_intercept_ci95_upper": 0.0,
        },
        artifact_fingerprint=_hex("0"),
    )
    return artifact.model_copy(update={"artifact_fingerprint": artifact.recompute_fingerprint()})


def _decision(score: float = 0.8) -> ResolutionDecisionV1:
    candidate = ResolutionCandidateV1(
        action_id="open_browser",
        vector_score=score,
        vector_scores={
            "semantic": score,
            "intent": score,
            "target_context": score,
        },
        vector_ranks={"semantic": 1, "intent": 1, "target_context": 1},
        coverage=1.0,
        lexical_score=score,
        argument_compatibility=1.0,
        combined_score=score,
        extracted_args={},
        eligible=True,
        rejection_reasons=(),
    )
    return ResolutionDecisionV1(
        subtask_id="subtask-1",
        candidates=(candidate,),
        top1_action_id="open_browser",
        margin_top2=0.5,
        decision=ResolutionStatusV1.RESOLVED,
        stt_confidence=0.95,
        catalog_hash="catalog",
    )


def _updated_row(
    row: RoutingCalibrationObservationV1,
    **updates: Any,
) -> RoutingCalibrationObservationV1:
    changed = row.model_copy(update={**updates, "snapshot_sha256": None})
    return changed.model_copy(update={"snapshot_sha256": changed.recompute_snapshot_sha256()})


def _training_rows(count: int = 320) -> list[RoutingCalibrationObservationV1]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[RoutingCalibrationObservationV1] = []
    for index in range(count):
        score = index / max(1, count - 1)
        label = score >= 0.5
        rows.append(
            _observation(
                index,
                score=score,
                action_correct=label,
                observed_at=start + timedelta(minutes=index),
                group_id=f"group-{index}",
                safety_outcome=(
                    RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT
                    if label
                    else RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
                ),
            )
        )
    return rows


def test_fit_platt_is_monotonic() -> None:
    artifact, _report = fit_routing_calibration(
        _training_rows(),
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert artifact.status in {
        RoutingCalibrationArtifactStatus.READY,
        RoutingCalibrationArtifactStatus.NOT_REPORTABLE,
    }
    scores = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
    probabilities = [
        1.0 / (1.0 + math.exp(-(artifact.coefficients.a * score + artifact.coefficients.b)))
        for score in scores
    ]
    assert all(
        probabilities[index] <= probabilities[index + 1]
        for index in range(len(probabilities) - 1)
    )


def test_fit_is_deterministic_for_same_dataset() -> None:
    rows = _training_rows()
    first_artifact, first_report = fit_routing_calibration(
        rows,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        bootstrap_samples=250,
        bootstrap_seed=11,
    )
    second_artifact, second_report = fit_routing_calibration(
        rows,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        bootstrap_samples=250,
        bootstrap_seed=11,
    )
    assert first_artifact.model_dump(mode="json") == second_artifact.model_dump(mode="json")
    assert first_report.model_dump(mode="json") == second_report.model_dump(mode="json")


def test_fit_can_reach_ready_with_independent_holdout() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(12345)
    rows: list[RoutingCalibrationObservationV1] = []
    for index in range(900):
        score = float(rng.uniform(0.0, 1.0))
        probability = score**3
        label = bool(rng.random() < probability)
        rows.append(
            _observation(
                index,
                score=score,
                action_correct=label,
                observed_at=start + timedelta(minutes=index),
                group_id=f"ready-group-{index}",
                session_id=f"session-{index}",
                safety_outcome=(
                    RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT
                    if label
                    else RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
                ),
            )
        )
    artifact, report = fit_routing_calibration(
        rows,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        bootstrap_samples=200,
        bootstrap_seed=13,
    )
    assert artifact.status is RoutingCalibrationArtifactStatus.READY
    assert report.status == RoutingCalibrationArtifactStatus.READY.value
    assert report.credibility_gates is not None
    assert report.credibility_gates.passed is True


def test_fit_refuses_single_class_and_low_group_count() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    single_class = [
        _observation(
            index,
            score=0.9,
            action_correct=True,
            observed_at=start + timedelta(minutes=index),
            group_id=f"g-{index}",
        )
        for index in range(220)
    ]
    too_few_groups = [
        _observation(
            index,
            score=0.8 if index % 2 == 0 else 0.2,
            action_correct=index % 2 == 0,
            observed_at=start + timedelta(minutes=index),
            group_id=f"small-{index // 2}",
        )
        for index in range(120)
    ]
    artifact_single, _ = fit_routing_calibration(
        single_class,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
    )
    artifact_small, _ = fit_routing_calibration(
        too_few_groups,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
    )
    assert artifact_single.status is RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA
    assert artifact_small.status is RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA


def test_fit_empty_dataset_returns_fail_closed_artifact() -> None:
    artifact, report = fit_routing_calibration(
        [],
        runtime_fingerprint=_hex("0"),
        catalog_hash="unknown",
        bootstrap_samples=100,
    )
    assert artifact.status is RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA
    assert artifact.group_count == 0
    assert artifact.class_counts.successes == 0
    assert artifact.class_counts.errors == 0
    assert report.status == RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA.value
    assert report.credibility_gates is not None
    assert report.credibility_gates.passed is False


def test_validate_detects_temporal_group_leakage() -> None:
    cutoff = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _observation(
            1,
            score=0.8,
            action_correct=True,
            observed_at=cutoff - timedelta(days=1),
            group_id="g1",
        ),
        _observation(
            2,
            score=0.3,
            action_correct=False,
            observed_at=cutoff + timedelta(days=1),
            group_id="g1",
        ),
        _observation(
            3,
            score=0.7,
            action_correct=True,
            observed_at=cutoff + timedelta(days=2),
            group_id="g2",
        ),
    ]
    report = validate_calibration_observations(rows, train_cutoff=cutoff, for_training=False)
    assert report["temporal_component_leakage_count"] == 0
    assert report["excluded_component_count"] == 1


def test_fit_rejects_personal_holdout_training_rows() -> None:
    rows = _training_rows()
    rows[0] = rows[0].model_copy(
        update={
            "source": "routing_personal_holdout",
            "label_source": "personal_holdout",
        }
    )
    with pytest.raises(ValueError, match="Zakazane pochodzenie kalibracji"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )


def test_fit_rejects_forbidden_dataset_even_if_only_in_challenge_after_cutoff() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    rows = _training_rows(240)
    forbidden = _observation(
        9999,
        score=0.7,
        action_correct=True,
        observed_at=cutoff + timedelta(days=10),
        role=RoutingCalibrationSetRole.CHALLENGE,
        dataset_id="routing_v1_162",
    )
    with pytest.raises(ValueError, match="Zakazane dataset_id"):
        fit_routing_calibration(
            rows + [forbidden],
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
            train_cutoff=cutoff,
        )


def test_fit_rejects_personal_markers_in_source_fields_even_with_shadow_dataset() -> None:
    rows = _training_rows(240)
    rows[0] = rows[0].model_copy(
        update={
            "dataset_id": "routing_v2_shadow_runtime",
            "source": "panel_personal-holdout",
            "source_record_id": "routing-v1.jsonl#1",
        }
    )
    with pytest.raises(ValueError, match="Zakazane pochodzenie kalibracji"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"schema_version": 999}, RoutingCalibrationRuntimeStatus.UNSUPPORTED_SCHEMA),
        (
            {
                "schema_version": 1,
                "coefficients": {"a": float("nan"), "b": 0.0},
            },
            RoutingCalibrationRuntimeStatus.NONFINITE_ARTIFACT,
        ),
    ],
)
def test_runtime_rejects_unknown_schema_and_nonfinite_artifact(
    tmp_path: Path,
    payload: dict[str, Any],
    expected_status: RoutingCalibrationRuntimeStatus,
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is expected_status
    assert inference.p_action_correct == (None,)


def test_runtime_rejects_malformed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{", encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
    assert inference.p_action_correct == (None,)


def test_runtime_handles_non_utf8_artifact_in_report_only(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"\x80\x81\x82")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
    assert inference.p_action_correct == (None,)


def test_runtime_off_ignores_non_utf8_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"\x80\x81\x82")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="off",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    asyncio.run(runtime.preload())
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.OFF
    assert inference.p_action_correct == (None,)


def test_runtime_detects_fingerprint_mismatch_and_runtime_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    valid = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    tampered = valid.model_copy(update={"artifact_fingerprint": _hex("f")})
    artifact_path.write_text(tampered.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    mismatch = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert mismatch.status is RoutingCalibrationRuntimeStatus.ARTIFACT_FINGERPRINT_MISMATCH
    assert mismatch.p_action_correct == (None,)

    artifact_path.write_text(valid.model_dump_json(), encoding="utf-8")
    runtime_reloaded = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    runtime_mismatch = runtime_reloaded.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("b"),
        expected_catalog_hash="catalog",
    )
    assert runtime_mismatch.status is RoutingCalibrationRuntimeStatus.RUNTIME_FINGERPRINT_MISMATCH
    assert runtime_mismatch.p_action_correct == (None,)


def test_runtime_cache_revalidates_expected_runtime_and_catalog(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    valid = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    artifact_path.write_text(valid.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    ready = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert ready.status is RoutingCalibrationRuntimeStatus.READY
    runtime_mismatch = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("b"),
        expected_catalog_hash="catalog",
    )
    assert runtime_mismatch.status is RoutingCalibrationRuntimeStatus.RUNTIME_FINGERPRINT_MISMATCH
    catalog_mismatch = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="other-catalog",
    )
    assert catalog_mismatch.status is RoutingCalibrationRuntimeStatus.CATALOG_HASH_MISMATCH


def test_runtime_not_reportable_artifact_emits_null_probability(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact = _artifact(
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        status=RoutingCalibrationArtifactStatus.NOT_REPORTABLE,
    )
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    asyncio.run(runtime.preload())
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.NOT_REPORTABLE
    assert inference.p_action_correct == (None,)


def test_runtime_emits_probability_only_for_ready(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact-ready.json"
    artifact = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference_ready = runtime.infer(
        (_decision(0.9),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference_ready.status is RoutingCalibrationRuntimeStatus.READY
    assert inference_ready.p_action_correct[0] is not None


def test_runtime_fails_closed_for_invalid_ready_contract(tmp_path: Path) -> None:
    artifact_path = tmp_path / "invalid-ready.json"
    payload = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_dump(
        mode="json"
    )
    payload["credibility_gates"]["passed"] = False
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference = runtime.infer(
        (_decision(0.8),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
    assert inference.p_action_correct == (None,)


def test_runtime_rejects_overlapping_ready_artifact_sets(tmp_path: Path) -> None:
    artifact_path = tmp_path / "invalid-overlap-ready.json"
    artifact = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    overlap = artifact.model_copy(
        update={
            "excluded_component_ids": ("comp-eval-0000",),
        }
    )
    overlap = overlap.model_copy(update={"artifact_fingerprint": overlap.recompute_fingerprint()})
    artifact_path.write_text(overlap.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
    assert inference.p_action_correct == (None,)


def test_runtime_rejects_ready_artifact_with_inconsistent_gate_metrics(tmp_path: Path) -> None:
    artifact_path = tmp_path / "invalid-gates-ready.json"
    artifact = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    invalid = artifact.model_copy(
        update={
            "metrics": {
                **artifact.metrics,
                "brier_delta_vs_raw_ci95_upper": 0.01,
            }
        }
    )
    invalid = invalid.model_copy(
        update={"artifact_fingerprint": invalid.recompute_fingerprint()}
    )
    artifact_path.write_text(invalid.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    inference = runtime.infer(
        (_decision(),),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
    assert inference.p_action_correct == (None,)


def test_runtime_ready_returns_null_probability_for_challenge_decision(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact-ready.json"
    artifact = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    runtime = RoutingCalibrationRuntime(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(artifact_path),
        )
    )
    challenge_decision = _decision(0.9).model_copy(update={"margin_top2": 0.05})
    inference = runtime.infer(
        (_decision(0.9), challenge_decision),
        expected_runtime_fingerprint=_hex("a"),
        expected_catalog_hash="catalog",
    )
    assert inference.status is RoutingCalibrationRuntimeStatus.READY
    assert inference.p_action_correct[0] is not None
    assert inference.p_action_correct[1] is None


def test_fit_rejects_mixed_runtime_and_catalog_context() -> None:
    rows = _training_rows(260)
    rows[1] = _updated_row(rows[1], runtime_fingerprint=_hex("b"))
    with pytest.raises(ValueError, match="Mieszane runtime_fingerprint"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
            train_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
        )
    rows = _training_rows(260)
    rows[1] = _updated_row(rows[1], catalog_hash="catalog-v2")
    with pytest.raises(ValueError, match="Mieszane catalog_hash"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
            train_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
        )


def test_temporal_split_with_explicit_cutoff_keeps_empty_train() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        _observation(
            index,
            score=0.8 if index % 2 == 0 else 0.2,
            action_correct=index % 2 == 0,
            observed_at=cutoff + timedelta(days=5 + index),
            group_id=f"late-{index}",
        )
        for index in range(20)
    ]
    split = temporal_group_split(rows, train_cutoff=cutoff)
    assert split.train == ()
    assert len(split.evaluate) == len(rows)
    assert set(split.train_group_ids).isdisjoint(set(split.evaluate_group_ids))


def test_temporal_split_excludes_components_crossing_cutoff() -> None:
    cutoff = datetime(2026, 1, 10, tzinfo=UTC)
    cross_group = "dup-hash"
    rows = [
        _observation(
            1,
            score=0.9,
            action_correct=True,
            observed_at=cutoff - timedelta(days=1),
            group_id=cross_group,
            session_id="2026-01-10T10:00:00+00:00",
        ),
        _observation(
            2,
            score=0.2,
            action_correct=False,
            observed_at=cutoff + timedelta(days=1),
            group_id=cross_group,
            session_id="2026-01-10T10:00:00+00:00",
        ),
    ]
    split = temporal_group_split(rows, train_cutoff=cutoff)
    assert split.train == ()
    assert split.evaluate == ()
    assert len(split.excluded_group_ids) == 1


def test_component_grouping_bridges_session_and_duplicate_tokens() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        _observation(
            1,
            score=0.9,
            action_correct=True,
            observed_at=base,
            group_id="dup-a",
            session_id="sess-1",
        ),
        _observation(
            2,
            score=0.2,
            action_correct=False,
            observed_at=base + timedelta(minutes=1),
            group_id="dup-b",
            session_id="sess-1",
        ),
        _observation(
            3,
            score=0.8,
            action_correct=True,
            observed_at=base + timedelta(minutes=2),
            group_id="dup-b",
            session_id="sess-2",
        ),
    ]
    split = temporal_group_split(rows, train_cutoff=base + timedelta(days=1))
    assert len(split.train_group_ids) == 1


def test_temporal_split_uses_unlabeled_bridge_from_full_observations() -> None:
    cutoff = datetime(2026, 1, 2, tzinfo=UTC)
    labeled_train = _observation(
        1,
        score=0.9,
        action_correct=True,
        observed_at=cutoff - timedelta(hours=1),
        group_id="dup-a",
        session_id="sess-bridge",
        role=RoutingCalibrationSetRole.REPRESENTATIVE,
    )
    labeled_eval = _observation(
        2,
        score=0.2,
        action_correct=False,
        observed_at=cutoff + timedelta(hours=1),
        group_id="dup-b",
        session_id="sess-far",
        role=RoutingCalibrationSetRole.REPRESENTATIVE,
    )
    bridge_unlabeled = _observation(
        3,
        score=0.5,
        action_correct=None,
        observed_at=cutoff - timedelta(minutes=10),
        group_id="dup-b",
        session_id="sess-bridge",
        role=RoutingCalibrationSetRole.CHALLENGE,
    )
    split = temporal_group_split(
        [labeled_train, labeled_eval],
        train_cutoff=cutoff,
        dependency_rows=[labeled_train, labeled_eval, bridge_unlabeled],
    )
    assert split.train == ()
    assert split.evaluate == ()
    assert len(split.excluded_group_ids) == 1


def test_safety_component_denominator_uses_challenge_bridge() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    first = _updated_row(
        _observation(
            1,
            score=0.9,
            action_correct=True,
            observed_at=base,
            group_id="dup-a",
            session_id="sess-a",
            safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        ),
        request_id="req-1",
    )
    second = _updated_row(
        _observation(
            2,
            score=0.8,
            action_correct=True,
            observed_at=base + timedelta(minutes=1),
            group_id="dup-b",
            session_id="sess-b",
            safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        ),
        request_id="req-2",
    )
    bridge = _observation(
        3,
        score=0.4,
        action_correct=None,
        observed_at=base + timedelta(minutes=2),
        group_id="dup-b",
        session_id="sess-a",
        role=RoutingCalibrationSetRole.CHALLENGE,
        safety_outcome=None,
    )
    report = evaluate_routing_calibration(
        [first, second, bridge],
        coefficients=(2.0, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=120,
        bootstrap_seed=9,
    )
    assert report.safety_labeled_plan_count == 2
    assert report.safety_labeled_component_count == 1


def test_fit_uses_representative_rows_only() -> None:
    representative = _training_rows(260)
    challenge = [
        _observation(
            10_000 + index,
            score=0.99,
            action_correct=False,
            observed_at=datetime(2026, 3, 1, tzinfo=UTC) + timedelta(minutes=index),
            group_id=f"challenge-{index}",
            role=RoutingCalibrationSetRole.CHALLENGE,
        )
        for index in range(30)
    ]
    artifact_only, _ = fit_routing_calibration(
        representative,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        train_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
    )
    artifact_mixed, _ = fit_routing_calibration(
        representative + challenge,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        train_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert artifact_only.status is artifact_mixed.status
    assert artifact_only.coefficients == artifact_mixed.coefficients
    assert artifact_only.class_counts == artifact_mixed.class_counts
    assert artifact_only.group_count == artifact_mixed.group_count
    assert artifact_only.metrics == artifact_mixed.metrics


def test_fit_evaluation_keeps_bridge_components_from_full_dependency_rows() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    rep_a = _observation(
        1,
        score=0.9,
        action_correct=True,
        observed_at=cutoff + timedelta(hours=1),
        group_id="g-a",
        session_id="sess-a",
        role=RoutingCalibrationSetRole.REPRESENTATIVE,
        safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
    )
    rep_b = _observation(
        2,
        score=0.2,
        action_correct=False,
        observed_at=cutoff + timedelta(hours=2),
        group_id="g-b",
        session_id="sess-b",
        role=RoutingCalibrationSetRole.REPRESENTATIVE,
        safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
    )
    bridge = _observation(
        3,
        score=0.5,
        action_correct=None,
        observed_at=cutoff + timedelta(hours=1, minutes=30),
        group_id="g-b",
        session_id="sess-a",
        role=RoutingCalibrationSetRole.CHALLENGE,
    )
    artifact, report = fit_routing_calibration(
        [rep_a, rep_b, bridge],
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        train_cutoff=cutoff,
        bootstrap_samples=120,
        bootstrap_seed=11,
    )
    assert artifact.status is not RoutingCalibrationArtifactStatus.READY
    assert len(artifact.evaluation_component_ids) == 1
    assert report.group_count == 1
    assert report.safety_labeled_component_count == 1


def test_fit_reports_no_holdout_without_train_fallback() -> None:
    rows = _training_rows(260)
    artifact, report = fit_routing_calibration(
        rows,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        train_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert artifact.status is RoutingCalibrationArtifactStatus.NOT_REPORTABLE
    assert report.status == RoutingCalibrationArtifactStatus.NOT_REPORTABLE.value
    assert report.representative_sample_count == 0
    assert report.has_independent_holdout is False
    assert report.credibility_gates is not None
    assert report.credibility_gates.passed is False


def test_derive_plan_labels_returns_exact_null_for_unknown_args() -> None:
    action_sequence_correct, exact_plan_correct, safety = derive_plan_labels(
        predicted_action_ids=("open_browser",),
        predicted_step_args=({},),
        expected_action_ids=("open_browser",),
        expected_step_args=({},),
        expected_abstention=False,
        arguments_complete=False,
    )
    assert action_sequence_correct is True
    assert exact_plan_correct is None
    assert safety is None


def test_evaluate_report_bootstrap_is_deterministic() -> None:
    rows = _training_rows(260)
    first = evaluate_routing_calibration(
        rows,
        coefficients=(3.2, -1.6),
        status="ready",
        bootstrap_samples=400,
        bootstrap_seed=5,
    )
    second = evaluate_routing_calibration(
        rows,
        coefficients=(3.2, -1.6),
        status="ready",
        bootstrap_samples=400,
        bootstrap_seed=5,
    )
    assert first.ece_ci95_lower == pytest.approx(second.ece_ci95_lower)
    assert first.ece_ci95_upper == pytest.approx(second.ece_ci95_upper)
    assert first.brier_ci95_lower == pytest.approx(second.brier_ci95_lower)
    assert first.brier_ci95_upper == pytest.approx(second.brier_ci95_upper)
    assert first.log_loss_ci95_lower == pytest.approx(second.log_loss_ci95_lower)
    assert first.log_loss_ci95_upper == pytest.approx(second.log_loss_ci95_upper)
    assert first.brier_delta_vs_raw_ci95_upper == pytest.approx(
        second.brier_delta_vs_raw_ci95_upper
    )
    assert first.log_loss_delta_vs_intercept_ci95_upper == pytest.approx(
        second.log_loss_delta_vs_intercept_ci95_upper
    )
    assert first.selective_risk_by_coverage == second.selective_risk_by_coverage


@pytest.mark.asyncio
async def test_recorder_overflow_is_nonblocking_and_stops_cleanly() -> None:
    class SlowStore:
        async def initialize(self) -> None:
            return None

        async def append_many(self, _observations) -> int:
            await asyncio.sleep(0.05)
            return 1

    recorder = RoutingCalibrationRecorder(
        store=SlowStore(),  # type: ignore[arg-type]
        queue_limit=1,
        enabled=True,
    )
    await recorder.start()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    started = time.perf_counter()
    for index in range(200):
        recorder.record(
            [
                _observation(
                    index,
                    score=0.7,
                    action_correct=True,
                    observed_at=base + timedelta(seconds=index),
                    group_id=f"overflow-{index}",
                )
            ]
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2
    assert recorder.dropped_count > 0
    await recorder.stop()


@pytest.mark.asyncio
async def test_recorder_failed_count_increments_on_store_error() -> None:
    class FailingStore:
        async def initialize(self) -> None:
            return None

        async def append_many(self, _observations) -> int:
            raise RuntimeError("boom")

    recorder = RoutingCalibrationRecorder(
        store=FailingStore(),  # type: ignore[arg-type]
        queue_limit=10,
        enabled=True,
    )
    await recorder.start()
    recorder.record(
        [
            _observation(
                1,
                score=0.6,
                action_correct=True,
                observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]
    )
    await asyncio.sleep(0.05)
    await recorder.stop()
    assert recorder.failed_count >= 1


@pytest.mark.asyncio
async def test_recorder_init_failure_does_not_raise_and_disables() -> None:
    class BrokenStore:
        async def initialize(self) -> None:
            raise OSError("cannot open sqlite")

        async def append_many(self, _observations) -> int:
            return 0

    recorder = RoutingCalibrationRecorder(
        store=BrokenStore(),  # type: ignore[arg-type]
        queue_limit=10,
        enabled=True,
    )
    await recorder.start()
    assert recorder.enabled is False
    assert recorder.failed_count >= 1
    await recorder.stop()


def test_observation_contract_validates_expected_vs_predicted_action() -> None:
    with pytest.raises(ValueError, match="action_correct must match"):
        _observation(
            1,
            score=0.8,
            action_correct=False,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            predicted_action_id="open_browser",
            expected_action_id="open_browser",
        )
    valid = _observation(
        2,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        predicted_action_id="open_browser",
        expected_action_id="open_browser",
    )
    assert valid.action_correct is True


@pytest.mark.asyncio
async def test_sqlite_store_roundtrip_and_label_preserving_upsert(tmp_path: Path) -> None:
    store = RoutingCalibrationObservationStore(tmp_path / "calibration.sqlite")
    await store.initialize()
    labeled = _observation(
        1,
        score=0.9,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        expected_action_id="open_browser",
    ).model_copy(
        update={
            "action_sequence_correct": True,
            "exact_plan_correct": True,
            "label_source": "manual",
            "safety_outcome": RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        }
    )
    await store.append_many([labeled])
    runtime_refresh = labeled.model_copy(
        update={
            "action_correct": None,
            "action_sequence_correct": None,
            "exact_plan_correct": None,
            "safety_outcome": None,
            "label_source": None,
            "expected_action_id": None,
            "predicted_action_id": "open_browser",
        }
    )
    await store.append_many([runtime_refresh])
    rows = await store.fetch_recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.action_correct is True
    assert row.action_sequence_correct is True
    assert row.exact_plan_correct is True
    assert row.label_source == "manual"
    assert row.expected_action_id == "open_browser"
    assert row.observed_at == labeled.observed_at
    assert row.p_action_correct == labeled.p_action_correct


@pytest.mark.asyncio
async def test_sqlite_store_rejects_conflicting_immutable_snapshot(tmp_path: Path) -> None:
    store = RoutingCalibrationObservationStore(tmp_path / "calibration.sqlite")
    await store.initialize()
    row = _observation(
        1,
        score=0.9,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await store.append_many([row])
    conflict = row.model_copy(update={"snapshot_sha256": _hex("f")})
    await store.append_many([conflict])
    loaded = await store.fetch_recent(limit=10)
    assert loaded[0].snapshot_sha256 == row.snapshot_sha256


@pytest.mark.asyncio
async def test_sqlite_store_rejects_feature_or_time_conflict(tmp_path: Path) -> None:
    store = RoutingCalibrationObservationStore(tmp_path / "calibration.sqlite")
    await store.initialize()
    row = _observation(
        2,
        score=0.7,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await store.append_many([row])
    changed = row.model_copy(
        update={
            "observed_at": datetime(2026, 1, 2, tzinfo=UTC),
            "snapshot_sha256": None,
        }
    )
    changed = changed.model_copy(update={"snapshot_sha256": changed.recompute_snapshot_sha256()})
    with pytest.raises(ValueError, match="Conflicting immutable snapshot"):
        await store.append_many([changed])


@pytest.mark.asyncio
async def test_sqlite_store_label_correction_overwrites_previous_label(tmp_path: Path) -> None:
    store = RoutingCalibrationObservationStore(tmp_path / "calibration.sqlite")
    await store.initialize()
    first_label = _observation(
        1,
        score=0.9,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        predicted_action_id="open_browser",
        expected_action_id="open_browser",
    ).model_copy(
        update={
            "action_sequence_correct": True,
            "exact_plan_correct": True,
            "label_source": "manual",
            "safety_outcome": RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        }
    )
    await store.append_many([first_label])
    correction = first_label.model_copy(
        update={
            "predicted_action_id": "open_browser",
            "expected_action_id": "open_url",
            "action_correct": False,
            "action_sequence_correct": False,
            "exact_plan_correct": None,
            "safety_outcome": RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION,
            "label_source": "manual_correction",
        }
    )
    await store.append_many([correction])
    loaded = await store.fetch_recent(limit=10)
    assert len(loaded) == 1
    row = loaded[0]
    assert row.expected_action_id == "open_url"
    assert row.action_correct is False
    assert row.action_sequence_correct is False
    assert row.exact_plan_correct is None
    assert row.safety_outcome is RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
    assert row.label_source == "manual_correction"


@pytest.mark.asyncio
async def test_sqlite_store_additive_migration_for_new_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE routing_calibration_observations (
                request_id TEXT NOT NULL,
                subtask_index INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                normalized_text_sha256 TEXT NOT NULL,
                runtime_fingerprint TEXT NOT NULL,
                catalog_hash TEXT NOT NULL,
                session_id TEXT,
                group_id TEXT NOT NULL,
                set_role TEXT NOT NULL,
                subtask_count INTEGER NOT NULL,
                decision TEXT NOT NULL,
                ranking_score REAL,
                margin_top2 REAL,
                vector_score REAL,
                lexical_score REAL,
                argument_score REAL,
                vector_coverage REAL,
                stt_confidence REAL,
                candidate_count INTEGER NOT NULL,
                eligible INTEGER,
                rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
                p_action_correct REAL,
                action_correct INTEGER,
                action_sequence_correct INTEGER,
                exact_plan_correct INTEGER,
                safety_outcome TEXT,
                label_source TEXT,
                PRIMARY KEY (request_id, subtask_index)
            );
            """
        )
    store = RoutingCalibrationObservationStore(db_path)
    await store.initialize()
    observation = _observation(
        1,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        expected_action_id="open_browser",
    )
    await store.append_many([observation])
    loaded = await store.fetch_recent(limit=10)
    assert loaded[0].predicted_action_id == "open_browser"
    assert loaded[0].expected_action_id == "open_browser"


@pytest.mark.asyncio
async def test_sqlite_store_migrates_populated_legacy_rows_as_unverified(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-populated.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE routing_calibration_observations (
                request_id TEXT NOT NULL,
                subtask_index INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                normalized_text_sha256 TEXT NOT NULL,
                runtime_fingerprint TEXT NOT NULL,
                catalog_hash TEXT NOT NULL,
                session_id TEXT,
                group_id TEXT NOT NULL,
                set_role TEXT NOT NULL,
                subtask_count INTEGER NOT NULL,
                decision TEXT NOT NULL,
                ranking_score REAL,
                margin_top2 REAL,
                vector_score REAL,
                lexical_score REAL,
                argument_score REAL,
                vector_coverage REAL,
                stt_confidence REAL,
                candidate_count INTEGER NOT NULL,
                eligible INTEGER,
                rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
                p_action_correct REAL,
                action_correct INTEGER,
                action_sequence_correct INTEGER,
                exact_plan_correct INTEGER,
                safety_outcome TEXT,
                label_source TEXT,
                expected_action_id TEXT,
                PRIMARY KEY (request_id, subtask_index)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO routing_calibration_observations (
                request_id, subtask_index, observed_at, source, normalized_text_sha256,
                runtime_fingerprint, catalog_hash, session_id, group_id, set_role,
                subtask_count, decision, ranking_score, margin_top2, vector_score,
                lexical_score, argument_score, vector_coverage, stt_confidence,
                candidate_count, eligible, rejection_reasons_json, p_action_correct,
                action_correct, action_sequence_correct, exact_plan_correct,
                safety_outcome, label_source, expected_action_id
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            (
                "legacy-req",
                0,
                datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "panel",
                _hex("b"),
                _hex("a"),
                "catalog",
                None,
                "g-legacy",
                "representative",
                1,
                "resolved",
                0.8,
                0.2,
                0.8,
                0.9,
                1.0,
                1.0,
                0.95,
                2,
                1,
                "[]",
                None,
                1,
                1,
                1,
                "safe_execute_correct",
                "manual",
                "open_browser",
            ),
        )
    store = RoutingCalibrationObservationStore(db_path)
    await store.initialize()
    loaded = await store.fetch_recent(limit=10)
    assert loaded[0].dataset_id == "legacy_unverified"
    assert loaded[0].source_record_id == "legacy-req:0"
    assert loaded[0].predicted_action_id is None
    assert loaded[0].expected_action_id is None
    assert loaded[0].action_correct is True
    with pytest.raises(ValueError, match="shadow provenance"):
        fit_routing_calibration(
            loaded,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )


@pytest.mark.asyncio
async def test_sqlite_store_migrates_legacy_split_override_without_label_source(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-grouping.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE routing_calibration_observations (
                request_id TEXT NOT NULL,
                subtask_index INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                normalized_text_sha256 TEXT NOT NULL,
                runtime_fingerprint TEXT NOT NULL,
                catalog_hash TEXT NOT NULL,
                session_id TEXT,
                split_group_override TEXT,
                group_id TEXT NOT NULL,
                set_role TEXT NOT NULL,
                subtask_count INTEGER NOT NULL,
                decision TEXT NOT NULL,
                ranking_score REAL,
                margin_top2 REAL,
                vector_score REAL,
                lexical_score REAL,
                argument_score REAL,
                vector_coverage REAL,
                stt_confidence REAL,
                candidate_count INTEGER NOT NULL,
                eligible INTEGER,
                rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
                p_action_correct REAL,
                action_correct INTEGER,
                action_sequence_correct INTEGER,
                exact_plan_correct INTEGER,
                safety_outcome TEXT,
                label_source TEXT,
                PRIMARY KEY (request_id, subtask_index)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO routing_calibration_observations (
                request_id, subtask_index, observed_at, source, normalized_text_sha256,
                runtime_fingerprint, catalog_hash, session_id, split_group_override, group_id,
                set_role, subtask_count, decision, ranking_score, margin_top2, vector_score,
                lexical_score, argument_score, vector_coverage, stt_confidence,
                candidate_count, eligible, rejection_reasons_json, p_action_correct,
                    action_correct, action_sequence_correct, exact_plan_correct,
                    safety_outcome, label_source
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
            """,
            (
                "legacy-group",
                0,
                datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "panel",
                _hex("b"),
                _hex("a"),
                "catalog",
                None,
                "manual-override",
                "g-legacy",
                "representative",
                1,
                "resolved",
                0.6,
                0.2,
                0.6,
                0.7,
                1.0,
                1.0,
                0.9,
                2,
                1,
                "[]",
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
    store = RoutingCalibrationObservationStore(db_path)
    await store.initialize()
    loaded = await store.fetch_recent(limit=10)
    assert loaded[0].split_group_override == "manual-override"
    assert loaded[0].label_source == "legacy_grouping"
    assert loaded[0].expected_action_id is None
    assert loaded[0].action_correct is None


def test_fit_rejects_forbidden_manifest_hash_even_with_shadow_dataset() -> None:
    rows = _training_rows(240)
    rows[0] = _updated_row(
        rows[0],
        manifest_sha256=(
            "7345e06d44716b297fd6afc90b12a9337ef7e5780edd8f8050288a1359107880"
        ),
    )
    with pytest.raises(ValueError, match="Zakazany manifest"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )


@pytest.mark.asyncio
async def test_sqlite_store_ignores_stale_label_snapshot_hash(tmp_path: Path) -> None:
    store = RoutingCalibrationObservationStore(tmp_path / "calibration.sqlite")
    await store.initialize()
    row = _observation(
        1,
        score=0.8,
        action_correct=None,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await store.append_many([row])
    with sqlite3.connect(tmp_path / "calibration.sqlite") as connection:
        connection.execute(
            """
            INSERT INTO routing_calibration_labels (
                request_id, subtask_index, snapshot_sha256, expected_action_id,
                action_correct, action_sequence_correct, exact_plan_correct,
                safety_outcome, label_source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id, subtask_index) DO UPDATE SET
                snapshot_sha256=excluded.snapshot_sha256,
                expected_action_id=excluded.expected_action_id,
                action_correct=excluded.action_correct,
                label_source=excluded.label_source
            """,
            (
                row.request_id,
                row.subtask_index,
                _hex("f"),
                "open_browser",
                1,
                1,
                1,
                "safe_execute_correct",
                "manual",
                datetime.now(UTC).isoformat(),
            ),
        )
    loaded = await store.fetch_recent(limit=10)
    assert loaded[0].expected_action_id is None


@pytest.mark.asyncio
async def test_sqlite_store_initialize_fails_closed_on_tampered_snapshot(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "tampered.sqlite"
    store = RoutingCalibrationObservationStore(db_path)
    await store.initialize()
    row = _observation(
        1,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await store.append_many([row])
    original_snapshot = row.snapshot_sha256
    assert original_snapshot is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE routing_calibration_observations
            SET ranking_score=?
            WHERE request_id=? AND subtask_index=?
            """,
            (0.123, row.request_id, row.subtask_index),
        )
    with pytest.raises(ValueError, match="snapshot_sha256 mismatch"):
        await store.initialize()
    with sqlite3.connect(db_path) as connection:
        current_snapshot = connection.execute(
            """
            SELECT snapshot_sha256 FROM routing_calibration_observations
            WHERE request_id=? AND subtask_index=?
            """,
            (row.request_id, row.subtask_index),
        ).fetchone()[0]
    assert current_snapshot == original_snapshot


@pytest.mark.asyncio
async def test_sqlite_store_initialize_fails_closed_on_nonfinite_scalar(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "tampered-nonfinite.sqlite"
    store = RoutingCalibrationObservationStore(db_path)
    await store.initialize()
    row = _observation(
        1,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await store.append_many([row])
    original_snapshot = row.snapshot_sha256
    assert original_snapshot is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE routing_calibration_observations
            SET ranking_score=?
            WHERE request_id=? AND subtask_index=?
            """,
            (float("nan"), row.request_id, row.subtask_index),
        )
    with pytest.raises(ValueError, match="snapshot_sha256 mismatch|Non-finite numeric value"):
        await store.initialize()
    with sqlite3.connect(db_path) as connection:
        current_snapshot = connection.execute(
            """
            SELECT snapshot_sha256 FROM routing_calibration_observations
            WHERE request_id=? AND subtask_index=?
            """,
            (row.request_id, row.subtask_index),
        ).fetchone()[0]
    assert current_snapshot == original_snapshot


def test_build_observations_use_predicted_action_and_cross_source_grouping() -> None:
    decision = _decision()
    inference = SimpleNamespace(p_action_correct=(0.42,))  # type: ignore[assignment]
    request_panel = CommandRequest(source=CommandSource.PANEL, text="otwórz Chrome")
    request_deepgram = CommandRequest(source=CommandSource.DEEPGRAM, text="otwórz  Chrome")
    panel_rows = build_calibration_observations(
        request=request_panel,
        decisions=(decision,),
        subtask_count=1,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        inference=inference,
    )
    deepgram_rows = build_calibration_observations(
        request=request_deepgram,
        decisions=(decision,),
        subtask_count=3,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        inference=inference,
    )
    assert panel_rows[0].group_id == deepgram_rows[0].group_id
    assert panel_rows[0].session_id is None
    assert panel_rows[0].predicted_action_id == "open_browser"


def test_build_observations_use_only_interaction_session_id() -> None:
    decision = _decision()
    inference = SimpleNamespace(p_action_correct=(0.42,))  # type: ignore[assignment]
    shared_session = "conversation-session-1"
    request_a = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        interaction_session_id=shared_session,
    )
    request_b = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        interaction_session_id=shared_session,
    )
    request_c = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
    )
    rows_a = build_calibration_observations(
        request=request_a,
        decisions=(decision,),
        subtask_count=1,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        inference=inference,
    )
    rows_b = build_calibration_observations(
        request=request_b,
        decisions=(decision,),
        subtask_count=1,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        inference=inference,
    )
    rows_c = build_calibration_observations(
        request=request_c,
        decisions=(decision,),
        subtask_count=1,
        runtime_fingerprint=_hex("a"),
        catalog_hash="catalog",
        inference=inference,
    )
    assert rows_a[0].session_id == shared_session
    assert rows_b[0].session_id == shared_session
    assert rows_c[0].session_id is None


def test_evaluate_rejects_insufficient_status_from_passing_gates() -> None:
    rows = _training_rows(260)
    report = evaluate_routing_calibration(
        rows,
        coefficients=(3.0, -1.0),
        status=RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA.value,
        require_independent_holdout=True,
        bootstrap_samples=200,
        bootstrap_seed=3,
    )
    assert report.credibility_gates is not None
    assert report.credibility_gates.passed is False
    assert report.brier_delta_vs_raw_ci95_upper is not None
    assert report.log_loss_delta_vs_intercept_ci95_upper is not None


def test_evaluate_requires_independent_holdout_for_gate_pass() -> None:
    rows = _training_rows(260)
    report = evaluate_routing_calibration(
        rows,
        coefficients=(3.0, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        require_independent_holdout=True,
        independent_holdout_available=False,
        bootstrap_samples=200,
        bootstrap_seed=3,
    )
    assert report.credibility_gates is not None
    assert report.has_independent_holdout is False
    assert report.credibility_gates.passed is False


def test_report_counts_include_unlabeled_challenge_rows() -> None:
    rows = _training_rows(240)
    rows.append(
        _observation(
            9990,
            score=0.2,
            action_correct=None,
            observed_at=datetime(2026, 1, 2, tzinfo=UTC),
            role=RoutingCalibrationSetRole.CHALLENGE,
        )
    )
    report = evaluate_routing_calibration(
        rows,
        coefficients=(2.0, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=120,
        bootstrap_seed=7,
    )
    assert report.sample_count == len(rows)
    assert report.challenge_sample_count >= 1


def test_evaluate_rejects_conflicting_plan_safety_labels() -> None:
    base_seed = _observation(
        1,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        source_record_id="same-request-1",
    )
    base = _updated_row(base_seed, request_id="same-request")
    conflicting = _updated_row(
        base,
        subtask_index=1,
        subtask_count=2,
        safety_outcome=RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION,
    )
    with pytest.raises(ValueError, match="Konflikt safety_outcome"):
        evaluate_routing_calibration(
            [base, conflicting],
            coefficients=(2.0, -1.0),
            status=RoutingCalibrationArtifactStatus.READY.value,
        )


def test_safety_zero_event_bound_uses_component_denominator() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    first = _updated_row(
        _observation(
            1,
            score=0.9,
            action_correct=True,
            observed_at=base,
            session_id="sess-shared",
            safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        ),
        request_id="req-1",
    )
    second = _updated_row(
        _observation(
            2,
            score=0.8,
            action_correct=True,
            observed_at=base + timedelta(minutes=1),
            session_id="sess-shared",
            safety_outcome=RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT,
        ),
        request_id="req-2",
    )
    rows = [first, second]
    report = evaluate_routing_calibration(
        rows,
        coefficients=(2.0, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=120,
        bootstrap_seed=9,
    )
    assert report.safety_labeled_plan_count == 2
    assert report.unsafe_event_count == 0
    assert report.safety_labeled_component_count == 1
    assert report.unsafe_component_count == 0
    assert report.unsafe_zero_event_upper_bound_95 == pytest.approx(0.95)


def test_ece_uses_minimum_bin_occupancy() -> None:
    rows = _training_rows(260)
    report_260 = evaluate_routing_calibration(
        rows,
        coefficients=(2.5, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    rows_39 = _training_rows(39)
    report_39 = evaluate_routing_calibration(
        rows_39,
        coefficients=(2.5, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert report_260.equal_mass_ece_bins == 10
    assert report_39.equal_mass_ece_bins == 1


def test_ece_is_permutation_invariant_for_tied_probabilities() -> None:
    rows = [
        _observation(
            index,
            score=0.5,
            action_correct=index % 2 == 0,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
            group_id=f"g-{index}",
        )
        for index in range(60)
    ]
    report_a = evaluate_routing_calibration(
        rows,
        coefficients=(0.0, 0.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=120,
        bootstrap_seed=5,
    )
    report_b = evaluate_routing_calibration(
        list(reversed(rows)),
        coefficients=(0.0, 0.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=120,
        bootstrap_seed=5,
    )
    assert report_a.equal_mass_ece == pytest.approx(report_b.equal_mass_ece)


def test_bootstrap_requires_minimum_independent_groups() -> None:
    rows = [
        _observation(
            index,
            score=0.9 if index % 2 == 0 else 0.2,
            action_correct=index % 2 == 0,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
            group_id=f"few-{index // 4}",
        )
        for index in range(40)
    ]
    report = evaluate_routing_calibration(
        rows,
        coefficients=(2.0, -1.0),
        status=RoutingCalibrationArtifactStatus.READY.value,
        bootstrap_samples=120,
        bootstrap_seed=5,
    )
    assert report.ece_ci95_upper is None
    assert report.credibility_gates is not None
    assert report.credibility_gates.passed is False


def test_validate_artifact_for_evaluation_rejects_tamper_and_mixed_context() -> None:
    rows = _training_rows(220)
    artifact = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog")
    tampered = artifact.model_copy(update={"artifact_fingerprint": _hex("f")})
    with pytest.raises(ValueError, match="artifact_fingerprint"):
        validate_calibration_artifact_for_evaluation(tampered, rows)
    mixed_rows = list(rows)
    mixed_rows[0] = _updated_row(mixed_rows[0], runtime_fingerprint=_hex("b"))
    with pytest.raises(ValueError, match="mieszane runtime_fingerprint"):
        validate_calibration_artifact_for_evaluation(artifact, mixed_rows)


def test_validate_artifact_rejects_train_token_overlap() -> None:
    rows = [
        _observation(
            index,
            score=0.8 if index % 2 == 0 else 0.3,
            action_correct=index % 2 == 0,
            observed_at=datetime(2026, 2, 2, tzinfo=UTC) + timedelta(minutes=index),
            group_id=f"same-{index}",
            session_id="session-overlap",
        )
        for index in range(30)
    ]
    artifact = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_copy(
        update={
            "training_window_end": datetime(2026, 2, 1, tzinfo=UTC),
            "train_cutoff": datetime(2026, 2, 1, tzinfo=UTC),
            "evaluation_component_ids": (),
            "train_group_tokens": ("s:session-overlap",),
        }
    )
    artifact = artifact.model_copy(
        update={"artifact_fingerprint": artifact.recompute_fingerprint()}
    )
    with pytest.raises(ValueError, match="overlap token"):
        validate_calibration_artifact_for_evaluation(artifact, rows)


def test_fit_rejects_missing_or_mismatched_snapshot_hash() -> None:
    rows = _training_rows(240)
    rows[0] = rows[0].model_copy(update={"snapshot_sha256": None})
    with pytest.raises(ValueError, match="Brak snapshot_sha256"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )
    rows = _training_rows(240)
    rows[0] = rows[0].model_copy(update={"snapshot_sha256": _hex("f")})
    with pytest.raises(ValueError, match="snapshot_sha256 nie zgadza"):
        fit_routing_calibration(
            rows,
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )


def test_ready_artifact_contract_requires_passed_gates() -> None:
    invalid_payload = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_dump(
        mode="json"
    )
    invalid_payload["credibility_gates"]["passed"] = False
    with pytest.raises(ValueError, match="credibility_gates.passed must equal"):
        RoutingCalibrationArtifactV1.model_validate(invalid_payload)


def test_ready_artifact_contract_requires_tokens_and_disjoint_sets() -> None:
    invalid_payload = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_dump(
        mode="json"
    )
    invalid_payload["train_group_tokens"] = []
    with pytest.raises(ValueError, match="READY artifact requires split manifest and cutoff"):
        RoutingCalibrationArtifactV1.model_validate(invalid_payload)
    invalid_payload = _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_dump(
        mode="json"
    )
    invalid_payload["excluded_component_ids"] = ["comp-eval-0000"]
    with pytest.raises(ValueError, match="requires disjoint train/evaluation/excluded"):
        RoutingCalibrationArtifactV1.model_validate(invalid_payload)


@pytest.mark.asyncio
async def test_cli_validate_and_fit_integration_with_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "corpus" / "routing_calibration" / "observations-v1.db"
    monkeypatch.setenv("VOICELOOP_DATA_DIR", str(tmp_path))
    store = RoutingCalibrationObservationStore(store_path)
    await store.initialize()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        _observation(
            index,
            score=0.9 if index % 2 == 0 else 0.1,
            action_correct=index % 2 == 0,
            observed_at=start + timedelta(minutes=index),
            group_id=f"few-groups-{index // 4}",
            runtime_fingerprint=_hex("a"),
            catalog_hash="catalog",
        )
        for index in range(80)
    ]
    await store.append_many(rows)
    parser = build_parser()
    validate_args = parser.parse_args(
        ["validate-routing-calibration", "--data-root", str(tmp_path / "corpus")]
    )
    validate_exit = await _run_async(validate_args)
    assert validate_exit == 0
    validate_out = json.loads(capsys.readouterr().out)
    assert Path(validate_out["dataset_path"]) == store_path.resolve()

    artifact_path = tmp_path / "artifact.json"
    report_path = tmp_path / "report.json"
    fit_args = parser.parse_args(
        [
            "fit-routing-calibration",
            "--data-root",
            str(tmp_path / "corpus"),
            "--artifact",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )
    fit_exit = await _run_async(fit_args)
    assert fit_exit == 3
    fit_out = json.loads(capsys.readouterr().out)
    assert fit_out["status"] == RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA.value
    assert artifact_path.is_file()
    assert report_path.is_file()
    eval_args = parser.parse_args(
        [
            "evaluate-routing-calibration",
            "--data-root",
            str(tmp_path / "corpus"),
            "--artifact",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )
    eval_exit = await _run_async(eval_args)
    assert eval_exit == 3
    eval_out = json.loads(capsys.readouterr().out)
    assert Path(eval_out["dataset_path"]) == store_path.resolve()


@pytest.mark.asyncio
async def test_cli_validate_handles_legacy_sqlite_without_labels_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOICELOOP_DATA_DIR", str(tmp_path))
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE routing_calibration_observations (
                request_id TEXT NOT NULL,
                subtask_index INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                normalized_text_sha256 TEXT NOT NULL,
                runtime_fingerprint TEXT NOT NULL,
                catalog_hash TEXT NOT NULL,
                session_id TEXT,
                group_id TEXT NOT NULL,
                set_role TEXT NOT NULL,
                subtask_count INTEGER NOT NULL,
                decision TEXT NOT NULL,
                ranking_score REAL,
                margin_top2 REAL,
                vector_score REAL,
                lexical_score REAL,
                argument_score REAL,
                vector_coverage REAL,
                stt_confidence REAL,
                candidate_count INTEGER NOT NULL,
                eligible INTEGER,
                rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
                p_action_correct REAL,
                action_correct INTEGER,
                action_sequence_correct INTEGER,
                exact_plan_correct INTEGER,
                safety_outcome TEXT,
                label_source TEXT,
                PRIMARY KEY (request_id, subtask_index)
            );
            INSERT INTO routing_calibration_observations (
                request_id, subtask_index, observed_at, source, normalized_text_sha256,
                runtime_fingerprint, catalog_hash, session_id, group_id, set_role,
                subtask_count, decision, ranking_score, margin_top2, vector_score,
                lexical_score, argument_score, vector_coverage, stt_confidence,
                candidate_count, eligible, rejection_reasons_json, p_action_correct,
                action_correct, action_sequence_correct, exact_plan_correct,
                safety_outcome, label_source
            ) VALUES (
                'legacy-1', 0, '2026-01-01T00:00:00+00:00', 'panel',
                'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'catalog', NULL, 'g-1', 'representative', 1, 'resolved',
                0.8, 0.2, 0.8, 0.8, 1.0, 1.0, 0.9, 1, 1, '[]', NULL,
                1, 1, 1, 'safe_execute_correct', 'manual'
            );
            """
        )
    parser = build_parser()
    args = parser.parse_args(
        ["validate-routing-calibration", "--data-root", str(data_root)]
    )
    exit_code = await _run_async(args)
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert Path(output["dataset_path"]) == db_path.resolve()
    assert output["observation_count"] >= 1
    assert output["labeled_count"] >= 1


@pytest.mark.asyncio
async def test_cli_validate_handles_empty_sqlite_without_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOICELOOP_DATA_DIR", str(tmp_path))
    with sqlite3.connect(db_path):
        pass
    parser = build_parser()
    args = parser.parse_args(
        ["validate-routing-calibration", "--data-root", str(data_root)]
    )
    exit_code = await _run_async(args)
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert Path(output["dataset_path"]) == db_path.resolve()
    assert output["observation_count"] == 0
    assert output["labeled_count"] == 0


def test_cli_validate_fails_on_missing_core_column_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOICELOOP_DATA_DIR", str(tmp_path))
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE routing_calibration_observations (
                request_id TEXT NOT NULL,
                subtask_index INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                runtime_fingerprint TEXT NOT NULL,
                catalog_hash TEXT NOT NULL,
                group_id TEXT NOT NULL,
                set_role TEXT NOT NULL,
                subtask_count INTEGER NOT NULL,
                decision TEXT NOT NULL,
                candidate_count INTEGER NOT NULL,
                eligible INTEGER,
                rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (request_id, subtask_index)
            );
            """
        )
    exit_code = main(["validate-routing-calibration", "--data-root", str(data_root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ok"] is False
    assert "missing core columns" in output["error"]


def test_cli_validate_fails_on_invalid_rejection_reasons_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = RoutingCalibrationObservationStore(db_path)
    asyncio.run(store.initialize())
    row = _observation(
        1,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    asyncio.run(store.append_many([row]))
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE routing_calibration_observations
            SET rejection_reasons_json=?
            WHERE request_id=? AND subtask_index=?
            """,
            ("{bad-json", row.request_id, row.subtask_index),
        )
    exit_code = main(["validate-routing-calibration", "--data-root", str(data_root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ok"] is False
    assert "Invalid rejection_reasons_json" in output["error"]


def test_cli_validate_fails_on_nonfinite_numeric_and_invalid_bool(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = RoutingCalibrationObservationStore(db_path)
    asyncio.run(store.initialize())
    row = _observation(
        1,
        score=0.8,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    asyncio.run(store.append_many([row]))
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE routing_calibration_observations
            SET ranking_score=?, eligible=?
            WHERE request_id=? AND subtask_index=?
            """,
            (float("inf"), 2, row.request_id, row.subtask_index),
        )
    exit_code = main(["validate-routing-calibration", "--data-root", str(data_root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ok"] is False
    assert "Non-finite numeric value" in output["error"]


@pytest.mark.asyncio
async def test_cli_explicit_missing_dataset_path_fails_closed(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    missing = tmp_path / "missing.sqlite"
    validate_args = parser.parse_args(
        ["validate-routing-calibration", "--dataset", str(missing)]
    )
    fit_args = parser.parse_args(["fit-routing-calibration", "--dataset", str(missing)])
    eval_args = parser.parse_args(
        [
            "evaluate-routing-calibration",
            "--dataset",
            str(missing),
            "--artifact",
            str(tmp_path / "artifact.json"),
        ]
    )
    with pytest.raises(ValueError, match="Nie znaleziono datasetu kalibracji"):
        await _run_async(validate_args)
    with pytest.raises(ValueError, match="Nie znaleziono datasetu kalibracji"):
        await _run_async(fit_args)
    with pytest.raises(ValueError, match="Nie znaleziono datasetu kalibracji"):
        await _run_async(eval_args)


def test_cli_missing_default_dataset_returns_structured_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    data_root.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_dump_json(),
        encoding="utf-8",
    )
    commands = [
        ["validate-routing-calibration", "--data-root", str(data_root)],
        ["fit-routing-calibration", "--data-root", str(data_root)],
        [
            "evaluate-routing-calibration",
            "--data-root",
            str(data_root),
            "--artifact",
            str(artifact_path),
        ],
    ]
    for argv in commands:
        exit_code = main(argv)
        output = json.loads(capsys.readouterr().out)
        assert exit_code == 2
        assert output["ok"] is False
        assert "Nie znaleziono datasetu kalibracji" in output["error"]


def test_cli_explicit_directory_dataset_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "dataset.sqlite"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        _artifact(runtime_fingerprint=_hex("a"), catalog_hash="catalog").model_dump_json(),
        encoding="utf-8",
    )
    commands = [
        ["validate-routing-calibration", "--dataset", str(dataset_dir)],
        ["fit-routing-calibration", "--dataset", str(dataset_dir)],
        [
            "evaluate-routing-calibration",
            "--dataset",
            str(dataset_dir),
            "--artifact",
            str(artifact_path),
        ],
    ]
    for argv in commands:
        exit_code = main(argv)
        output = json.loads(capsys.readouterr().out)
        assert exit_code == 2
        assert output["ok"] is False
        assert "Dataset kalibracji musi być plikiem" in output["error"]


def test_cli_validate_fails_closed_on_tampered_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = RoutingCalibrationObservationStore(db_path)
    asyncio.run(store.initialize())
    row = _observation(
        1,
        score=0.85,
        action_correct=True,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    asyncio.run(store.append_many([row]))
    original_snapshot = row.snapshot_sha256
    assert original_snapshot is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE routing_calibration_observations
            SET vector_score=?
            WHERE request_id=? AND subtask_index=?
            """,
            (0.001, row.request_id, row.subtask_index),
        )
    exit_code = main(
        ["validate-routing-calibration", "--data-root", str(data_root)]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ok"] is False
    assert "snapshot_sha256 mismatch" in output["error"]
    with sqlite3.connect(db_path) as connection:
        current_snapshot = connection.execute(
            """
            SELECT snapshot_sha256 FROM routing_calibration_observations
            WHERE request_id=? AND subtask_index=?
            """,
            (row.request_id, row.subtask_index),
        ).fetchone()[0]
    assert current_snapshot == original_snapshot


def test_cli_validate_fails_closed_for_corrupt_sqlite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "corpus"
    db_path = data_root / "routing_calibration" / "observations-v1.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"not-a-sqlite-database")
    exit_code = main(["validate-routing-calibration", "--data-root", str(data_root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ok"] is False
    assert "SQLite" in output["error"]


@pytest.mark.asyncio
async def test_cli_rejects_path_collisions(tmp_path: Path) -> None:
    dataset = tmp_path / "same.json"
    dataset.write_text("", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "fit-routing-calibration",
            "--dataset",
            str(dataset),
            "--artifact",
            str(dataset),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    with pytest.raises(ValueError, match="różne ścieżki"):
        await _run_async(args)

def _definition(action_id: str, *, args_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": action_id,
        "label": action_id.replace("_", " "),
        "description": action_id.replace("_", " "),
        "args_schema": args_schema
        or {"type": "object", "properties": {}, "additionalProperties": False},
        "risk": "low",
        "confirmation_required": False,
        "routing_examples": [],
        "available_in_voiceattack": False,
    }


def _search_for(
    subtask,
    matches: list[tuple[str, float, dict[str, float]]],
    *,
    catalog_hash: str = "catalog",
) -> SubtaskCapabilitySearch:
    embedding = SubtaskEmbeddingV1(
        subtask_id=subtask.subtask_id,
        semantic=(1.0, 0.0, 0.0),
        intent=(0.0, 1.0, 0.0),
        target_context=(0.0, 0.0, 1.0),
        embedding_model="fake-embedding",
        dimension=3,
        normalized_text_sha256=SubtaskEmbeddingV1.text_hash(subtask.normalized_text),
    )
    result = CapabilitySearchResult(
        query_documents=CapabilityDocuments(
            semantic=subtask.text,
            intent=subtask.operation or "",
            target_context=subtask.target or "",
        ),
        matches=[
            CapabilityMatch(
                action_id=action_id,
                label=action_id,
                description=action_id,
                score=score,
                vector_scores=vector_scores,
                risk="low",
                confirmation_required=False,
                available_in_voiceattack=False,
            )
            for action_id, score, vector_scores in matches
        ],
        catalog_hash=catalog_hash,
    )
    return SubtaskCapabilitySearch(subtask=subtask, embedding=embedding, result=result)


def _decision_signature(decisions: tuple[ResolutionDecisionV1, ...]) -> list[dict[str, Any]]:
    return [
        {
            "decision": decision.decision.value,
            "reason": decision.reason,
            "top1_action_id": decision.top1_action_id,
            "margin_top2": decision.margin_top2,
            "candidate_count": len(decision.candidates),
            "candidates": [
                {
                    "action_id": candidate.action_id,
                    "combined_score": round(candidate.combined_score, 8),
                    "eligible": candidate.eligible,
                    "rejection_reasons": tuple(candidate.rejection_reasons),
                    "args": dict(candidate.extracted_args),
                }
                for candidate in decision.candidates
            ],
        }
        for decision in decisions
    ]


@pytest.mark.asyncio
async def test_report_only_calibration_keeps_bit_identical_plan_and_decisions(
    tmp_path: Path,
) -> None:
    definitions = [
        _definition("open_browser"),
        _definition(
            "open_url",
            args_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
    ]

    class FakeIndex:
        catalog_hash = "catalog"
        embeddings = SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        )
        _dimension = 3
        collection_name = "fake-capabilities"
        taxonomy_version = "routing-taxonomy-v2"
        document_format_version = "capability-document-v2:routing-taxonomy-v2"
        query_format_version = "capability-query-v2:routing-taxonomy-v2"
        vector_fusion = "weighted-rrf-v1"
        rank_fusion_k = 60
        vector_weights: ClassVar[dict[str, float]] = {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }

        async def search_subtasks(self, subtasks, **_kwargs):
            return [
                _search_for(
                    subtask,
                    [
                        (
                            "open_browser",
                            0.95,
                            {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                        ),
                        (
                            "open_url",
                            0.40,
                            {"semantic": 0.40, "intent": 0.60, "target_context": 0.20},
                        ),
                    ],
                )
                for subtask in subtasks
            ]

    base_settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=False,
        routing_v2_shadow_mode=True,
        routing_v2_calibration_mode="off",
    )
    service_off = RoutingV2Service(
        base_settings,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    ready_artifact = _artifact(
        runtime_fingerprint=service_off.runtime_fingerprint,
        catalog_hash="catalog",
        a=3.0,
        b=-1.0,
    )
    ready_path = tmp_path / "routing-calibration-ready.json"
    ready_path.write_text(ready_artifact.model_dump_json(), encoding="utf-8")
    not_reportable_artifact = ready_artifact.model_copy(
        update={"status": RoutingCalibrationArtifactStatus.NOT_REPORTABLE}
    )
    not_reportable_artifact = not_reportable_artifact.model_copy(
        update={"artifact_fingerprint": not_reportable_artifact.recompute_fingerprint()}
    )
    not_reportable_path = tmp_path / "routing-calibration-not-reportable.json"
    not_reportable_path.write_text(
        not_reportable_artifact.model_dump_json(),
        encoding="utf-8",
    )
    malformed_path = tmp_path / "routing-calibration-malformed.json"
    malformed_path.write_text("{", encoding="utf-8")

    service_ready = RoutingV2Service(
        base_settings.model_copy(
            update={
                "routing_v2_calibration_mode": "report_only",
                "routing_v2_calibration_artifact_file": str(ready_path),
            }
        ),
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    service_not_reportable = RoutingV2Service(
        base_settings.model_copy(
            update={
                "routing_v2_calibration_mode": "report_only",
                "routing_v2_calibration_artifact_file": str(not_reportable_path),
            }
        ),
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    service_malformed = RoutingV2Service(
        base_settings.model_copy(
            update={
                "routing_v2_calibration_mode": "report_only",
                "routing_v2_calibration_artifact_file": str(malformed_path),
            }
        ),
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )

    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        transcript_confidence=0.95,
    )
    outcome_off = await service_off.evaluate(request)
    outcome_ready = await service_ready.evaluate(request)
    outcome_not_reportable = await service_not_reportable.evaluate(request)
    outcome_malformed = await service_malformed.evaluate(request)

    def _normalized_segmentation(outcome: RoutingV2Outcome) -> dict[str, Any]:
        payload = outcome.segmentation.model_dump(mode="json")
        for subtask in payload.get("subtasks", []):
            subtask.pop("subtask_id", None)
        return payload

    def _normalized_decisions(outcome: RoutingV2Outcome) -> list[dict[str, Any]]:
        payload = [item.model_dump(mode="json") for item in outcome.decisions]
        for decision in payload:
            decision.pop("subtask_id", None)
        return payload

    def _plan_dump(outcome: RoutingV2Outcome) -> dict[str, Any] | None:
        if outcome.plan is None:
            return None
        payload = outcome.plan.model_dump(mode="json")
        for step in payload.get("steps", []):
            step.pop("id", None)
            step.pop("depends_on", None)
        return payload

    def _assembly_dump(outcome: RoutingV2Outcome) -> dict[str, Any]:
        return {
            "blocked_reason": outcome.assembly.blocked_reason,
            "has_plan": outcome.assembly.plan is not None,
            "plan": _plan_dump(outcome),
        }

    reference_segmentation = _normalized_segmentation(outcome_off)
    reference_decisions = _normalized_decisions(outcome_off)
    reference_assembly = _assembly_dump(outcome_off)
    reference_execution = service_off.plan_execution_allowed(outcome_off.plan)
    for service, outcome in (
        (service_ready, outcome_ready),
        (service_not_reportable, outcome_not_reportable),
        (service_malformed, outcome_malformed),
    ):
        assert _normalized_segmentation(outcome) == reference_segmentation
        assert _normalized_decisions(outcome) == reference_decisions
        assert _assembly_dump(outcome) == reference_assembly
        assert _plan_dump(outcome) == _plan_dump(outcome_off)
        assert service.plan_execution_allowed(outcome.plan) == reference_execution

    shadow_ready = service_ready.shadow_payload(request, outcome_ready, legacy_plan=None)
    shadow_not_reportable = service_not_reportable.shadow_payload(
        request, outcome_not_reportable, legacy_plan=None
    )
    shadow_malformed = service_malformed.shadow_payload(
        request, outcome_malformed, legacy_plan=None
    )
    assert shadow_ready["calibration"]["status"] == RoutingCalibrationRuntimeStatus.READY.value
    assert (
        shadow_not_reportable["calibration"]["status"]
        == RoutingCalibrationRuntimeStatus.NOT_REPORTABLE.value
    )
    assert (
        shadow_malformed["calibration"]["status"]
        == RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT.value
    )


@pytest.mark.asyncio
async def test_report_only_invariance_across_compound_and_challenge_cases(
    tmp_path: Path,
) -> None:
    definitions = [
        _definition("open_browser"),
        _definition(
            "open_url",
            args_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
    ]

    class FakeIndex:
        catalog_hash = "catalog"
        embeddings = SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        )
        _dimension = 3
        collection_name = "fake-capabilities"
        taxonomy_version = "routing-taxonomy-v2"
        document_format_version = "capability-document-v2:routing-taxonomy-v2"
        query_format_version = "capability-query-v2:routing-taxonomy-v2"
        vector_fusion = "weighted-rrf-v1"
        rank_fusion_k = 60
        vector_weights: ClassVar[dict[str, float]] = {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }

        async def search_subtasks(self, subtasks, **_kwargs):
            rows = []
            for subtask in subtasks:
                if (subtask.target or "").casefold() in {"youtube", "url"}:
                    matches = [
                        (
                            "open_url",
                            0.96,
                            {"semantic": 0.96, "intent": 0.96, "target_context": 0.96},
                        ),
                        (
                            "open_browser",
                            0.40,
                            {"semantic": 0.40, "intent": 0.50, "target_context": 0.30},
                        ),
                    ]
                else:
                    matches = [
                        (
                            "open_browser",
                            0.95,
                            {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                        ),
                        (
                            "open_url",
                            0.40,
                            {"semantic": 0.40, "intent": 0.60, "target_context": 0.20},
                        ),
                    ]
                rows.append(_search_for(subtask, matches))
            return rows

    base_settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=False,
        routing_v2_shadow_mode=True,
        routing_v2_calibration_mode="off",
    )
    seed_service = RoutingV2Service(
        base_settings,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    ready_artifact = _artifact(
        runtime_fingerprint=seed_service.runtime_fingerprint,
        catalog_hash="catalog",
    )
    ready_path = tmp_path / "ready-invariance.json"
    ready_path.write_text(ready_artifact.model_dump_json(), encoding="utf-8")
    nr_artifact = ready_artifact.model_copy(
        update={"status": RoutingCalibrationArtifactStatus.NOT_REPORTABLE}
    )
    nr_artifact = nr_artifact.model_copy(
        update={"artifact_fingerprint": nr_artifact.recompute_fingerprint()}
    )
    nr_path = tmp_path / "nr-invariance.json"
    nr_path.write_text(nr_artifact.model_dump_json(), encoding="utf-8")
    malformed_path = tmp_path / "malformed-invariance.json"
    malformed_path.write_text("{", encoding="utf-8")
    services = {
        "off": RoutingV2Service(
            base_settings,
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions,
        ),
        "ready": RoutingV2Service(
            base_settings.model_copy(
                update={
                    "routing_v2_calibration_mode": "report_only",
                    "routing_v2_calibration_artifact_file": str(ready_path),
                }
            ),
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions,
        ),
        "not_reportable": RoutingV2Service(
            base_settings.model_copy(
                update={
                    "routing_v2_calibration_mode": "report_only",
                    "routing_v2_calibration_artifact_file": str(nr_path),
                }
            ),
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions,
        ),
        "malformed": RoutingV2Service(
            base_settings.model_copy(
                update={
                    "routing_v2_calibration_mode": "report_only",
                    "routing_v2_calibration_artifact_file": str(malformed_path),
                }
            ),
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions,
        ),
    }

    def _norm(outcome: RoutingV2Outcome) -> dict[str, Any]:
        seg = outcome.segmentation.model_dump(mode="json")
        for subtask in seg.get("subtasks", []):
            subtask.pop("subtask_id", None)
        dec = [d.model_dump(mode="json") for d in outcome.decisions]
        for item in dec:
            item.pop("subtask_id", None)
        plan = outcome.plan.model_dump(mode="json") if outcome.plan is not None else None
        if plan is not None:
            for step in plan.get("steps", []):
                step.pop("id", None)
                step.pop("depends_on", None)
        return {
            "segmentation": seg,
            "decisions": dec,
            "assembly_blocked": outcome.assembly.blocked_reason,
            "plan": plan,
        }

    requests = [
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="otwórz Chrome",
            transcript_confidence=0.95,
        ),
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="otwórz youtube",
            transcript_confidence=0.95,
        ),
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="otwórz Chrome i otwórz youtube",
            transcript_confidence=0.95,
        ),
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="otwórz Chrome",
            transcript_confidence=0.10,
        ),
    ]
    for request in requests:
        baseline = await services["off"].evaluate(request.model_copy())
        baseline_norm = _norm(baseline)
        baseline_exec = services["off"].plan_execution_allowed(baseline.plan)
        for key in ("ready", "not_reportable", "malformed"):
            outcome = await services[key].evaluate(request.model_copy())
            assert _norm(outcome) == baseline_norm
            assert services[key].plan_execution_allowed(outcome.plan) == baseline_exec


@pytest.mark.asyncio
async def test_report_only_does_not_change_canary_execution_gate(
    tmp_path: Path,
) -> None:
    definitions_low = [_definition("open_browser"), _definition("open_url")]
    definitions_high = [_definition("open_browser")]
    definitions_high[0]["risk"] = "medium"

    class FakeIndex:
        catalog_hash = "catalog"
        embeddings = SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        )
        _dimension = 3
        collection_name = "fake-capabilities"
        taxonomy_version = "routing-taxonomy-v2"
        document_format_version = "capability-document-v2:routing-taxonomy-v2"
        query_format_version = "capability-query-v2:routing-taxonomy-v2"
        vector_fusion = "weighted-rrf-v1"
        rank_fusion_k = 60
        vector_weights: ClassVar[dict[str, float]] = {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }

        async def search_subtasks(self, subtasks, **_kwargs):
            return [
                _search_for(
                    subtask,
                    [
                        (
                            "open_browser",
                            0.95,
                            {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                        ),
                        (
                            "open_url",
                            0.40,
                            {"semantic": 0.40, "intent": 0.60, "target_context": 0.20},
                        ),
                    ],
                )
                for subtask in subtasks
            ]

    base = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=True,
        routing_v2_shadow_mode=False,
        routing_v2_canary_enabled=True,
        routing_v2_canary_action_ids="open_browser",
        routing_v2_calibration_mode="off",
    )
    seed = RoutingV2Service(base, capability_index=FakeIndex(), definitions=definitions_low)  # type: ignore[arg-type]
    artifact = _artifact(runtime_fingerprint=seed.runtime_fingerprint, catalog_hash="catalog")
    artifact_path = tmp_path / "canary-ready.json"
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    services = {
        "off_low": RoutingV2Service(
            base,
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions_low,
        ),
        "ready_low": RoutingV2Service(
            base.model_copy(
                update={
                    "routing_v2_calibration_mode": "report_only",
                    "routing_v2_calibration_artifact_file": str(artifact_path),
                }
            ),
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions_low,
        ),
        "off_high": RoutingV2Service(
            base,
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions_high,
        ),
        "ready_high": RoutingV2Service(
            base.model_copy(
                update={
                    "routing_v2_calibration_mode": "report_only",
                    "routing_v2_calibration_artifact_file": str(artifact_path),
                }
            ),
            capability_index=FakeIndex(),  # type: ignore[arg-type]
            definitions=definitions_high,
        ),
    }
    for service in services.values():
        service.quality_gate = RoutingQualityGate(
            passed=True,
            reason="passed",
            path="in-memory",
            catalog_coverage=1.0,
            expected_action_count=len(service.definitions),
            runtime_fingerprint=service.runtime_fingerprint,
        )
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        transcript_confidence=0.95,
    )
    off_low = await services["off_low"].evaluate(request.model_copy())
    ready_low = await services["ready_low"].evaluate(request.model_copy())
    off_high = await services["off_high"].evaluate(request.model_copy())
    ready_high = await services["ready_high"].evaluate(request.model_copy())
    assert services["off_low"].plan_execution_allowed(off_low.plan) is True
    assert services["ready_low"].plan_execution_allowed(ready_low.plan) is True
    assert services["off_high"].plan_execution_allowed(off_high.plan) is False
    assert services["ready_high"].plan_execution_allowed(ready_high.plan) is False


@pytest.mark.asyncio
async def test_service_off_start_ignores_non_utf8_artifact_and_keeps_outputs(
    tmp_path: Path,
) -> None:
    definitions = [_definition("open_browser"), _definition("open_url")]

    class FakeIndex:
        catalog_hash = "catalog"
        embeddings = SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        )
        _dimension = 3
        collection_name = "fake-capabilities"
        taxonomy_version = "routing-taxonomy-v2"
        document_format_version = "capability-document-v2:routing-taxonomy-v2"
        query_format_version = "capability-query-v2:routing-taxonomy-v2"
        vector_fusion = "weighted-rrf-v1"
        rank_fusion_k = 60
        vector_weights: ClassVar[dict[str, float]] = {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }

        async def search_subtasks(self, subtasks, **_kwargs):
            return [
                _search_for(
                    subtask,
                    [
                        (
                            "open_browser",
                            0.95,
                            {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                        ),
                        (
                            "open_url",
                            0.40,
                            {"semantic": 0.40, "intent": 0.60, "target_context": 0.20},
                        ),
                    ],
                )
                for subtask in subtasks
            ]

    bad_artifact = tmp_path / "bad-artifact.bin"
    bad_artifact.write_bytes(b"\x80\x81\x82")
    settings_plain = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=False,
        routing_v2_shadow_mode=True,
        routing_v2_calibration_mode="off",
    )
    settings_bad = settings_plain.model_copy(
        update={"routing_v2_calibration_artifact_file": str(bad_artifact)}
    )
    service_plain = RoutingV2Service(
        settings_plain,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    service_bad = RoutingV2Service(
        settings_bad,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    await service_plain.start()
    await service_bad.start()
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        transcript_confidence=0.95,
    )
    outcome_plain = await service_plain.evaluate(
        request.model_copy(update={"request_id": "plain"})
    )
    outcome_bad = await service_bad.evaluate(request.model_copy(update={"request_id": "bad"}))
    assert _decision_signature(outcome_plain.decisions) == _decision_signature(
        outcome_bad.decisions
    )
    assert outcome_plain.plan is not None and outcome_bad.plan is not None
    assert [step.action_id for step in outcome_plain.plan.steps] == [
        step.action_id for step in outcome_bad.plan.steps
    ]
    await service_plain.close()
    await service_bad.close()


@pytest.mark.asyncio
async def test_service_report_only_handles_non_utf8_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    definitions = [_definition("open_browser"), _definition("open_url")]

    class FakeIndex:
        catalog_hash = "catalog"
        embeddings = SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        )
        _dimension = 3
        collection_name = "fake-capabilities"
        taxonomy_version = "routing-taxonomy-v2"
        document_format_version = "capability-document-v2:routing-taxonomy-v2"
        query_format_version = "capability-query-v2:routing-taxonomy-v2"
        vector_fusion = "weighted-rrf-v1"
        rank_fusion_k = 60
        vector_weights: ClassVar[dict[str, float]] = {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }

        async def search_subtasks(self, subtasks, **_kwargs):
            return [
                _search_for(
                    subtask,
                    [
                        (
                            "open_browser",
                            0.95,
                            {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                        ),
                        (
                            "open_url",
                            0.40,
                            {"semantic": 0.40, "intent": 0.60, "target_context": 0.20},
                        ),
                    ],
                )
                for subtask in subtasks
            ]

    bad_artifact = tmp_path / "bad-artifact-report.bin"
    bad_artifact.write_bytes(b"\x80\x81\x82")
    service = RoutingV2Service(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            routing_v2_execute=False,
            routing_v2_shadow_mode=True,
            routing_v2_calibration_mode="report_only",
            routing_v2_calibration_artifact_file=str(bad_artifact),
        ),
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    await service.start()
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        transcript_confidence=0.95,
    )
    outcome = await service.evaluate(request)
    shadow = service.shadow_payload(request, outcome, legacy_plan=None)
    assert (
        shadow["calibration"]["status"]
        == RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT.value
    )
    await service.close()


@pytest.mark.asyncio
async def test_service_lifecycle_preloads_ready_artifact_and_persists_snapshot(
    tmp_path: Path,
) -> None:
    definitions = [_definition("open_browser"), _definition("open_url")]

    class FakeIndex:
        catalog_hash = "catalog"
        embeddings = SimpleNamespace(
            _resolved_model="fake-embedding",
            configured_model="fake-embedding",
        )
        _dimension = 3
        collection_name = "fake-capabilities"
        taxonomy_version = "routing-taxonomy-v2"
        document_format_version = "capability-document-v2:routing-taxonomy-v2"
        query_format_version = "capability-query-v2:routing-taxonomy-v2"
        vector_fusion = "weighted-rrf-v1"
        rank_fusion_k = 60
        vector_weights: ClassVar[dict[str, float]] = {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }

        async def search_subtasks(self, subtasks, **_kwargs):
            return [
                _search_for(
                    subtask,
                    [
                        (
                            "open_browser",
                            0.95,
                            {"semantic": 0.95, "intent": 0.95, "target_context": 0.95},
                        ),
                        (
                            "open_url",
                            0.40,
                            {"semantic": 0.40, "intent": 0.60, "target_context": 0.20},
                        ),
                    ],
                )
                for subtask in subtasks
            ]

    ready_artifact_path = tmp_path / "ready-artifact.json"
    report_only_settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        routing_v2_execute=False,
        routing_v2_shadow_mode=True,
        routing_v2_calibration_mode="report_only",
        routing_v2_calibration_artifact_file=str(ready_artifact_path),
        routing_v2_calibration_store_file=str(tmp_path / "obs.sqlite"),
    )
    service_report = RoutingV2Service(
        report_only_settings,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    ready_artifact = _artifact(
        runtime_fingerprint=service_report.runtime_fingerprint,
        catalog_hash="catalog",
    )
    ready_artifact_path.write_text(ready_artifact.model_dump_json(), encoding="utf-8")
    service_report = RoutingV2Service(
        report_only_settings,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    await service_report.start()
    request = CommandRequest(
        source=CommandSource.DEEPGRAM,
        text="otwórz Chrome",
        transcript_confidence=0.95,
    )
    outcome_report = await service_report.evaluate(request)
    shadow_report = service_report.shadow_payload(request, outcome_report, legacy_plan=None)
    assert shadow_report["calibration"]["status"] == RoutingCalibrationRuntimeStatus.READY.value
    assert shadow_report["calibration"]["p_action_correct"][0] is not None
    await asyncio.sleep(0.05)
    persisted = await service_report.calibration_recorder.store.fetch_recent(limit=20)
    assert persisted
    assert persisted[0].snapshot_sha256 is not None
    await service_report.close()

    off_settings = report_only_settings.model_copy(
        update={"routing_v2_calibration_mode": "off"}
    )
    service_off = RoutingV2Service(
        off_settings,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    outcome_off = await service_off.evaluate(request.model_copy(update={"request_id": "off-req"}))
    assert _decision_signature(outcome_off.decisions) == _decision_signature(
        outcome_report.decisions
    )

    not_reportable_artifact = ready_artifact.model_copy(
        update={"status": RoutingCalibrationArtifactStatus.NOT_REPORTABLE}
    )
    not_reportable_artifact = not_reportable_artifact.model_copy(
        update={"artifact_fingerprint": not_reportable_artifact.recompute_fingerprint()}
    )
    not_reportable_path = tmp_path / "nr-artifact.json"
    not_reportable_path.write_text(not_reportable_artifact.model_dump_json(), encoding="utf-8")
    not_reportable_settings = report_only_settings.model_copy(
        update={"routing_v2_calibration_artifact_file": str(not_reportable_path)}
    )
    service_nr = RoutingV2Service(
        not_reportable_settings,
        capability_index=FakeIndex(),  # type: ignore[arg-type]
        definitions=definitions,
    )
    await service_nr.start()
    outcome_nr = await service_nr.evaluate(request.model_copy(update={"request_id": "nr-req"}))
    shadow_nr = service_nr.shadow_payload(request, outcome_nr, legacy_plan=None)
    assert (
        shadow_nr["calibration"]["status"]
        == RoutingCalibrationRuntimeStatus.NOT_REPORTABLE.value
    )
    assert shadow_nr["calibration"]["p_action_correct"][0] is None
    await service_nr.close()


@pytest.mark.asyncio
async def test_split_group_override_updates_label_only_and_grouping(tmp_path: Path) -> None:
    store = RoutingCalibrationObservationStore(tmp_path / "override.sqlite")
    await store.initialize()
    base = _observation(
        10,
        score=0.75,
        action_correct=None,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        group_id="dup-a",
        source_record_id="manual-override-1",
    )
    before_snapshot = base.snapshot_sha256
    await store.append_many([base])
    labeled_override = base.model_copy(
        update={"split_group_override": "paraphrase-override", "label_source": "manual"}
    )
    await store.append_many([labeled_override])
    loaded = await store.fetch_recent(limit=10)
    assert loaded[0].snapshot_sha256 == before_snapshot
    assert loaded[0].split_group_override == "paraphrase-override"

    row_a = loaded[0].model_copy(
        update={
            "action_correct": True,
            "set_role": RoutingCalibrationSetRole.REPRESENTATIVE,
        }
    )
    row_b = _updated_row(
        _observation(
            11,
            score=0.30,
            action_correct=False,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=1),
            group_id="dup-b",
            source_record_id="manual-override-2",
        ),
        split_group_override="paraphrase-override",
        label_source="manual",
    ).model_copy(update={"set_role": RoutingCalibrationSetRole.REPRESENTATIVE})
    split = temporal_group_split([row_a, row_b], train_cutoff=datetime(2026, 1, 2, tzinfo=UTC))
    assert len(split.train_group_ids) == 1


def test_shadow_payload_fails_closed_when_calibration_inference_raises(tmp_path: Path) -> None:
    service = RoutingV2Service(
        Settings(voiceloop_data_dir=str(tmp_path)),
        capability_index=SimpleNamespace(catalog_hash="catalog"),  # type: ignore[arg-type]
        definitions=[_definition("open_browser")],
    )
    outcome = RoutingV2Outcome(segmentation=segment_command("otwórz Chrome"))
    service._infer_calibration = lambda _decisions: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("boom")
    )
    payload = service.shadow_payload(
        CommandRequest(text="otwórz Chrome"),
        outcome,
        legacy_plan=None,
    )
    assert (
        payload["calibration"]["status"]
        == RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT.value
    )
