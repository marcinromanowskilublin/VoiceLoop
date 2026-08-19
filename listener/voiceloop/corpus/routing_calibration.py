from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .schema import (
    RoutingCalibrationArtifactStatus,
    RoutingCalibrationArtifactV1,
    RoutingCalibrationClassCountsV1,
    RoutingCalibrationCoefficientsV1,
    RoutingCalibrationCredibilityGatesV1,
    RoutingCalibrationObservationV1,
    RoutingCalibrationReportV1,
    RoutingCalibrationSafetyOutcome,
    RoutingCalibrationSetRole,
)
from .storage import read_jsonl, sha256_text

MIN_INDEPENDENT_GROUPS = 200
MIN_SUCCESSES = 20
MIN_ERRORS = 20
MIN_BOOTSTRAP_GROUPS = 20
FORBIDDEN_DATASET_IDS = frozenset(
    {
        "routing_v1_162",
        "routing-v1-162",
        "routing-v1",
        "routing-v1.jsonl",
        "routing-personal-holdout-47",
        "routing_personal_holdout_47",
        "routing_personal_holdout",
        "routing-personal-holdout",
        "routing-personal-holdout-v1",
    }
)
SHADOW_DATASET_PREFIX = "routing_v2_shadow_runtime"
FORBIDDEN_MANIFEST_SHA256 = frozenset(
    {
        "7345e06d44716b297fd6afc90b12a9337ef7e5780edd8f8050288a1359107880",
        "a1f1dac765f41a78bf2c06b1bee39e23a98ae254f57e147ba5637f19bb189d7e",
    }
)


@dataclass(frozen=True, slots=True)
class CalibrationSplit:
    train: tuple[RoutingCalibrationObservationV1, ...]
    evaluate: tuple[RoutingCalibrationObservationV1, ...]
    train_group_ids: tuple[str, ...]
    evaluate_group_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...]
    train_group_tokens: tuple[str, ...]
    train_cutoff: datetime

    @property
    def split_hash(self) -> str:
        payload = json.dumps(
            {
                "train_groups": self.train_group_ids,
                "evaluate_groups": self.evaluate_group_ids,
                "excluded_groups": self.excluded_group_ids,
                "train_group_tokens": self.train_group_tokens,
                "train_cutoff": self.train_cutoff.isoformat(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256_text(payload)


def load_calibration_observations(path: Path) -> list[RoutingCalibrationObservationV1]:
    if not path.exists():
        raise ValueError(f"Nie znaleziono datasetu kalibracji: {path}")
    if not path.is_file():
        raise ValueError(f"Dataset kalibracji musi być plikiem: {path}")
    suffix = path.suffix.casefold()
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return _load_from_sqlite(path)
    return read_jsonl(path, RoutingCalibrationObservationV1)


def validate_calibration_observations(
    observations: Sequence[RoutingCalibrationObservationV1],
    *,
    train_cutoff: datetime | None = None,
    for_training: bool = False,
) -> dict[str, Any]:
    rows = tuple(observations)
    _reject_forbidden_provenance(rows, require_shadow=for_training)
    _validate_snapshots(rows)
    labeled = [row for row in rows if row.action_correct is not None]
    representative = [
        row for row in labeled if row.set_role is RoutingCalibrationSetRole.REPRESENTATIVE
    ]
    challenge_count = sum(
        1 for row in rows if row.set_role is RoutingCalibrationSetRole.CHALLENGE
    )
    split = temporal_group_split(
        representative,
        train_cutoff=train_cutoff,
        dependency_rows=rows,
    )
    groups = _component_groups_for_rows(representative, rows)
    class_counts = RoutingCalibrationClassCountsV1(
        successes=sum(1 for row in representative if row.action_correct is True),
        errors=sum(1 for row in representative if row.action_correct is False),
    )
    if for_training:
        _reject_personal_holdout(representative)
    leakage_components = sorted(
        set(split.train_group_ids).intersection(set(split.evaluate_group_ids))
    )
    return {
        "schema_version": 1,
        "observation_count": len(rows),
        "labeled_count": len(labeled),
        "representative_labeled_count": len(representative),
        "challenge_sample_count": challenge_count,
        "group_count": len(groups),
        "success_count": class_counts.successes,
        "error_count": class_counts.errors,
        "personal_holdout_found": any(_is_personal_holdout(row) for row in rows),
        "train_cutoff": split.train_cutoff.isoformat() if representative else None,
        "temporal_component_leakage_count": len(leakage_components),
        "temporal_component_leakage_ids": leakage_components,
        "excluded_component_count": len(split.excluded_group_ids),
        "excluded_component_ids": list(split.excluded_group_ids),
    }


def fit_routing_calibration(
    observations: Sequence[RoutingCalibrationObservationV1],
    *,
    runtime_fingerprint: str,
    catalog_hash: str,
    train_cutoff: datetime | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 17,
) -> tuple[RoutingCalibrationArtifactV1, RoutingCalibrationReportV1]:
    all_rows = tuple(observations)
    _reject_forbidden_provenance(all_rows, require_shadow=True)
    _validate_snapshots(all_rows)
    labeled_rows = _labeled_rows(all_rows)
    representative_labeled = [
        row for row in labeled_rows if row.set_role is RoutingCalibrationSetRole.REPRESENTATIVE
    ]
    split = temporal_group_split(
        representative_labeled,
        train_cutoff=train_cutoff,
        dependency_rows=all_rows,
    )
    train_rows = list(split.train)
    _reject_personal_holdout(train_rows)
    _ensure_consistent_training_context(
        train_rows,
        runtime_fingerprint=runtime_fingerprint,
        catalog_hash=catalog_hash,
    )
    class_counts = RoutingCalibrationClassCountsV1(
        successes=sum(1 for row in train_rows if row.action_correct is True),
        errors=sum(1 for row in train_rows if row.action_correct is False),
    )
    train_group_count = len(split.train_group_ids)
    insufficient_data = _is_insufficient_data(
        train_rows,
        class_counts=class_counts,
        group_count=train_group_count,
    )
    ranking_scores = np.asarray(
        [float(row.ranking_score) for row in train_rows],
        dtype=np.float64,
    )
    labels = np.asarray(
        [1.0 if row.action_correct else 0.0 for row in train_rows],
        dtype=np.float64,
    )
    coefficients = (
        _fit_monotonic_platt(ranking_scores, labels)
        if (not insufficient_data and len(train_rows) > 0)
        else _insufficient_coefficients(labels)
    )
    status = RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA
    eval_rows = tuple(split.evaluate)
    eval_component_ids = set(split.evaluate_group_ids)
    full_component_map = _row_component_map(all_rows)
    eval_dependency_rows = tuple(
        row
        for row in all_rows
        if full_component_map.get(_row_identity(row)) in eval_component_ids
    )
    report_status = (
        RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA.value
        if insufficient_data
        else RoutingCalibrationArtifactStatus.READY.value
    )
    report = evaluate_routing_calibration(
        observations=eval_dependency_rows,
        artifact=None,
        coefficients=coefficients,
        status=report_status,
        artifact_fingerprint=None,
        require_independent_holdout=True,
        independent_holdout_available=bool(eval_rows),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    if not insufficient_data:
        report_passed = bool(report.credibility_gates and report.credibility_gates.passed)
        if not eval_rows:
            status = RoutingCalibrationArtifactStatus.NOT_REPORTABLE
        elif report_passed:
            status = RoutingCalibrationArtifactStatus.READY
        else:
            status = RoutingCalibrationArtifactStatus.NOT_REPORTABLE
    report = report.model_copy(update={"status": status.value})
    artifact = RoutingCalibrationArtifactV1(
        status=status,
        created_at=_window_end(train_rows) or datetime(1970, 1, 1, tzinfo=UTC),
        coefficients=RoutingCalibrationCoefficientsV1(a=coefficients[0], b=coefficients[1]),
        class_counts=class_counts,
        group_count=train_group_count,
        dataset_sha256=_dataset_hash(labeled_rows),
        split_sha256=split.split_hash,
        training_window_start=_window_start(train_rows),
        training_window_end=_window_end(train_rows),
        evaluation_window_start=_window_start(eval_rows),
        evaluation_window_end=_window_end(eval_rows),
        train_cutoff=split.train_cutoff,
        runtime_fingerprint=runtime_fingerprint,
        catalog_hash=catalog_hash,
        train_component_ids=split.train_group_ids,
        evaluation_component_ids=split.evaluate_group_ids,
        excluded_component_ids=split.excluded_group_ids,
        train_group_tokens=split.train_group_tokens,
        credibility_gates=report.credibility_gates,
        metrics={
            "train_sample_count": float(len(train_rows)),
            "train_group_count": float(train_group_count),
            "eval_sample_count": float(report.sample_count),
            "eval_brier_score": float(report.brier_score or 0.0),
            "eval_log_loss": float(report.log_loss or 0.0),
            "eval_equal_mass_ece": float(report.equal_mass_ece or 0.0),
            "point_ece": float(
                report.equal_mass_ece if report.equal_mass_ece is not None else 0.0
            ),
            "ece_ci95_upper": float(
                report.ece_ci95_upper if report.ece_ci95_upper is not None else 1.0
            ),
            "brier_delta_vs_raw": float(
                report.brier_delta_vs_raw if report.brier_delta_vs_raw is not None else 0.0
            ),
            "brier_delta_vs_raw_ci95_upper": float(
                report.brier_delta_vs_raw_ci95_upper
                if report.brier_delta_vs_raw_ci95_upper is not None
                else 1.0
            ),
            "log_loss_delta_vs_intercept": float(
                report.log_loss_delta_vs_intercept
                if report.log_loss_delta_vs_intercept is not None
                else 0.0
            ),
            "log_loss_delta_vs_intercept_ci95_upper": float(
                report.log_loss_delta_vs_intercept_ci95_upper
                if report.log_loss_delta_vs_intercept_ci95_upper is not None
                else 1.0
            ),
        },
        artifact_fingerprint="0" * 64,
    )
    fingerprint = artifact.recompute_fingerprint()
    artifact = artifact.model_copy(update={"artifact_fingerprint": fingerprint})
    report = report.model_copy(update={"artifact_fingerprint": fingerprint})
    return artifact, report


def evaluate_routing_calibration(
    observations: Sequence[RoutingCalibrationObservationV1],
    *,
    artifact: RoutingCalibrationArtifactV1 | None = None,
    coefficients: tuple[float, float] | None = None,
    status: str | None = None,
    artifact_fingerprint: str | None = None,
    require_independent_holdout: bool = False,
    independent_holdout_available: bool | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 17,
) -> RoutingCalibrationReportV1:
    all_rows = tuple(observations)
    _reject_forbidden_provenance(all_rows, require_shadow=True)
    _validate_snapshots(all_rows)
    (
        safety_labeled_plan_count,
        unsafe_count_plan,
        safety_labeled_component_count,
        unsafe_component_count,
    ) = _safety_summary_counts(all_rows)
    zero_unsafe_upper = (
        _zero_event_binomial_upper_bound(n=safety_labeled_component_count, alpha=0.05)
        if unsafe_component_count == 0 and safety_labeled_component_count > 0
        else None
    )
    rows = _labeled_rows(all_rows)
    representative_rows = [
        row for row in rows if row.set_role is RoutingCalibrationSetRole.REPRESENTATIVE
    ]
    challenge_count = sum(
        1 for row in all_rows if row.set_role is RoutingCalibrationSetRole.CHALLENGE
    )
    if artifact is not None:
        coefficients = (artifact.coefficients.a, artifact.coefficients.b)
        status = artifact.status.value
        artifact_fingerprint = artifact.artifact_fingerprint
    if coefficients is None:
        coefficients = (0.0, 0.0)
    if status is None:
        status = "ready"
    has_independent_holdout = (
        independent_holdout_available
        if independent_holdout_available is not None
        else (not require_independent_holdout or len(representative_rows) > 0)
    )
    if not representative_rows:
        gates = RoutingCalibrationCredibilityGatesV1(
            point_ece_le_0_10=False,
            ci_upper_ece_le_0_15=False,
            brier_improves_over_raw=False,
            logloss_not_worse_than_intercept=False,
            passed=False,
        )
        return RoutingCalibrationReportV1(
            status=status,
            artifact_fingerprint=artifact_fingerprint,
            sample_count=len(all_rows),
            representative_sample_count=0,
            challenge_sample_count=challenge_count,
            group_count=0,
            unsafe_event_count=unsafe_count_plan,
            safety_labeled_plan_count=safety_labeled_plan_count,
            unsafe_component_count=unsafe_component_count,
            safety_labeled_component_count=safety_labeled_component_count,
            unsafe_zero_event_upper_bound_95=zero_unsafe_upper,
            credibility_gates=gates,
            has_independent_holdout=has_independent_holdout,
        )
    y = np.asarray(
        [1.0 if row.action_correct else 0.0 for row in representative_rows],
        dtype=np.float64,
    )
    raw_scores = np.asarray(
        [max(0.0, min(float(row.ranking_score), 1.0)) for row in representative_rows],
        dtype=np.float64,
    )
    calibrated = _sigmoid(raw_scores * coefficients[0] + coefficients[1])
    brier = _brier_score(calibrated, y)
    log_loss = _log_loss(calibrated, y)
    raw_brier = _brier_score(raw_scores, y)
    raw_log_loss = _log_loss(raw_scores, y)
    intercept_baseline = np.full_like(y, np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    intercept_log_loss = _log_loss(intercept_baseline, y)
    ece_bins = _equal_mass_ece_bins(len(representative_rows))
    ece_edges = _equal_mass_edges(calibrated, bins=ece_bins)
    ece = _equal_mass_ece(calibrated, y, bins=ece_bins, fixed_edges=ece_edges)
    brier_delta = brier - raw_brier
    log_loss_delta = log_loss - intercept_log_loss
    ece_lower, ece_upper = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _equal_mass_ece(
            _sigmoid(
                np.asarray(
                    [
                        max(0.0, min(float(row.ranking_score), 1.0))
                        for row in sample_rows
                    ],
                    dtype=np.float64,
                )
                * coefficients[0]
                + coefficients[1]
            ),
            np.asarray(
                [1.0 if row.action_correct else 0.0 for row in sample_rows],
                dtype=np.float64,
            ),
            bins=_equal_mass_ece_bins(len(sample_rows)),
            fixed_edges=ece_edges,
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    brier_lower, brier_upper = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _brier_score(
            _sigmoid(
                np.asarray(
                    [max(0.0, min(float(row.ranking_score), 1.0)) for row in sample_rows],
                    dtype=np.float64,
                )
                * coefficients[0]
                + coefficients[1]
            ),
            np.asarray(
                [1.0 if row.action_correct else 0.0 for row in sample_rows],
                dtype=np.float64,
            ),
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    log_loss_lower, log_loss_upper = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _log_loss(
            _sigmoid(
                np.asarray(
                    [max(0.0, min(float(row.ranking_score), 1.0)) for row in sample_rows],
                    dtype=np.float64,
                )
                * coefficients[0]
                + coefficients[1]
            ),
            np.asarray(
                [1.0 if row.action_correct else 0.0 for row in sample_rows],
                dtype=np.float64,
            ),
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    brier_delta_lower, brier_delta_upper = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _calibrated_brier_delta(
            sample_rows,
            coefficients=coefficients,
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    log_loss_delta_lower, log_loss_delta_upper = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _calibrated_log_loss_delta_vs_intercept(
            sample_rows,
            coefficients=coefficients,
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    slope, intercept = _calibration_slope_intercept(calibrated, y)
    selective = _selective_risk(calibrated, y)
    non_ready_status = status != RoutingCalibrationArtifactStatus.READY.value
    slope_ci_low, slope_ci_high = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _calibration_slope_intercept_from_rows(
            sample_rows,
            coefficients=coefficients,
            return_part="slope",
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    intercept_ci_low, intercept_ci_high = _grouped_bootstrap_ci(
        representative_rows,
        metric=lambda sample_rows: _calibration_slope_intercept_from_rows(
            sample_rows,
            coefficients=coefficients,
            return_part="intercept",
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        dependency_rows=all_rows,
    )
    gate_point_ece = (ece <= 0.10) and not non_ready_status and has_independent_holdout
    gate_ci_ece = (
        ece_upper is not None
        and ece_upper <= 0.15
        and not non_ready_status
        and has_independent_holdout
    )
    gate_brier = (
        brier_delta_upper is not None
        and brier_delta_upper < 0.0
        and not non_ready_status
        and has_independent_holdout
    )
    gate_logloss = (
        log_loss_delta_upper is not None
        and log_loss_delta_upper <= 0.0
        and not non_ready_status
        and has_independent_holdout
    )
    gates = RoutingCalibrationCredibilityGatesV1(
        point_ece_le_0_10=gate_point_ece,
        ci_upper_ece_le_0_15=gate_ci_ece,
        brier_improves_over_raw=gate_brier,
        logloss_not_worse_than_intercept=gate_logloss,
        passed=bool(gate_point_ece and gate_ci_ece and gate_brier and gate_logloss),
    )
    return RoutingCalibrationReportV1(
        status=status,
        artifact_fingerprint=artifact_fingerprint,
        sample_count=len(all_rows),
        representative_sample_count=len(representative_rows),
        challenge_sample_count=challenge_count,
        group_count=len(_component_groups_for_rows(representative_rows, all_rows)),
        brier_score=brier,
        log_loss=log_loss,
        equal_mass_ece=ece,
        equal_mass_ece_bins=ece_bins,
        ece_ci95_lower=ece_lower,
        ece_ci95_upper=ece_upper,
        brier_ci95_lower=brier_lower,
        brier_ci95_upper=brier_upper,
        log_loss_ci95_lower=log_loss_lower,
        log_loss_ci95_upper=log_loss_upper,
        calibration_intercept=intercept,
        calibration_slope=slope,
        selective_risk_by_coverage=selective,
        raw_brier_score=raw_brier,
        raw_log_loss=raw_log_loss,
        intercept_only_log_loss=intercept_log_loss,
        brier_delta_vs_raw=brier_delta,
        brier_delta_vs_raw_ci95_lower=brier_delta_lower,
        brier_delta_vs_raw_ci95_upper=brier_delta_upper,
        log_loss_delta_vs_intercept=log_loss_delta,
        log_loss_delta_vs_intercept_ci95_lower=log_loss_delta_lower,
        log_loss_delta_vs_intercept_ci95_upper=log_loss_delta_upper,
        unsafe_event_count=unsafe_count_plan,
        safety_labeled_plan_count=safety_labeled_plan_count,
        unsafe_component_count=unsafe_component_count,
        safety_labeled_component_count=safety_labeled_component_count,
        unsafe_zero_event_upper_bound_95=zero_unsafe_upper,
        calibration_intercept_ci95_lower=intercept_ci_low,
        calibration_intercept_ci95_upper=intercept_ci_high,
        calibration_slope_ci95_lower=slope_ci_low,
        calibration_slope_ci95_upper=slope_ci_high,
        selective_diagnostics_ci95=_selective_ci95(
            representative_rows,
            coefficients=coefficients,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            dependency_rows=all_rows,
        ),
        credibility_gates=gates,
        has_independent_holdout=has_independent_holdout,
    )


def temporal_group_split(
    observations: Sequence[RoutingCalibrationObservationV1],
    *,
    train_cutoff: datetime | None = None,
    dependency_rows: Sequence[RoutingCalibrationObservationV1] | None = None,
) -> CalibrationSplit:
    rows = tuple(observations)
    dependency = tuple(dependency_rows) if dependency_rows is not None else rows
    ordered_dependency_rows = sorted(
        dependency,
        key=lambda row: (row.observed_at, row.group_id, row.request_id, row.subtask_index),
    )
    if not rows:
        now = datetime.now(UTC)
        return CalibrationSplit(
            train=(),
            evaluate=(),
            train_group_ids=(),
            evaluate_group_ids=(),
            excluded_group_ids=(),
            train_group_tokens=(),
            train_cutoff=now,
        )
    groups = _component_groups(ordered_dependency_rows)
    row_to_component = _row_component_map(ordered_dependency_rows)
    component_times: dict[str, tuple[datetime, datetime]] = {}
    for component_id, component_rows in groups.items():
        component_times[component_id] = (
            min(row.observed_at for row in component_rows),
            max(row.observed_at for row in component_rows),
        )
    ordered_components = sorted(
        component_times,
        key=lambda item: (component_times[item][1], item),
    )
    if train_cutoff is None:
        train_group_target = max(1, int(math.floor(len(ordered_components) * 0.8)))
        candidate_cutoff = component_times[ordered_components[train_group_target - 1]][1]
        cutoff = candidate_cutoff
    else:
        cutoff = train_cutoff
    train_components = {
        component_id
        for component_id in ordered_components
        if component_times[component_id][1] <= cutoff
    }
    evaluate_components = [
        component_id
        for component_id in ordered_components
        if component_times[component_id][0] > cutoff
    ]
    if set(train_components).intersection(evaluate_components):
        raise ValueError("Temporal split leakage: train/evaluate groups overlap")
    train_rows = []
    evaluate_rows = []
    for row in rows:
        component_id = row_to_component.get(_row_identity(row))
        if component_id is None:
            continue
        if component_id in train_components:
            train_rows.append(row)
        elif component_id in evaluate_components:
            evaluate_rows.append(row)
    train_component_ids = sorted(
        {row_to_component[_row_identity(row)] for row in train_rows}
    )
    evaluate_component_ids = sorted(
        {row_to_component[_row_identity(row)] for row in evaluate_rows}
    )
    excluded_components = [
        component_id
        for component_id in ordered_components
        if component_id not in train_component_ids
        and component_id not in evaluate_component_ids
    ]
    train_tokens = sorted(
        {
            token
            for component_id in train_component_ids
            for row in groups[component_id]
            for token in _component_tokens(row)
        }
    )
    return CalibrationSplit(
        train=tuple(train_rows),
        evaluate=tuple(evaluate_rows),
        train_group_ids=tuple(train_component_ids),
        evaluate_group_ids=tuple(evaluate_component_ids),
        excluded_group_ids=tuple(sorted(excluded_components)),
        train_group_tokens=tuple(train_tokens),
        train_cutoff=cutoff,
    )


def derive_plan_labels(
    *,
    predicted_action_ids: Sequence[str],
    predicted_step_args: Sequence[dict[str, Any]],
    expected_action_ids: Sequence[str],
    expected_step_args: Sequence[dict[str, Any]],
    expected_abstention: bool,
    arguments_complete: bool,
) -> tuple[bool, bool | None, RoutingCalibrationSafetyOutcome | None]:
    abstained = len(predicted_action_ids) == 0
    action_sequence_correct = (
        tuple(predicted_action_ids) == tuple(expected_action_ids)
        if not expected_abstention
        else abstained
    )
    if not arguments_complete:
        exact_plan_correct: bool | None = None
    else:
        exact_plan_correct = bool(
            action_sequence_correct
            and tuple(predicted_step_args) == tuple(expected_step_args)
        )
    if expected_abstention:
        safety_outcome = (
            RoutingCalibrationSafetyOutcome.SAFE_ABSTAIN
            if abstained
            else RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
        )
    elif abstained:
        safety_outcome = RoutingCalibrationSafetyOutcome.MISSED_ACTION
    elif action_sequence_correct and exact_plan_correct is True:
        safety_outcome = RoutingCalibrationSafetyOutcome.SAFE_EXECUTE_CORRECT
    elif action_sequence_correct and exact_plan_correct is None:
        safety_outcome = None
    else:
        safety_outcome = RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
    return action_sequence_correct, exact_plan_correct, safety_outcome


def _labeled_rows(
    observations: Sequence[RoutingCalibrationObservationV1],
) -> tuple[RoutingCalibrationObservationV1, ...]:
    return tuple(
        row
        for row in observations
        if row.ranking_score is not None and row.action_correct is not None
    )


def _is_insufficient_data(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    class_counts: RoutingCalibrationClassCountsV1,
    group_count: int,
) -> bool:
    if not rows:
        return True
    if class_counts.successes == 0 or class_counts.errors == 0:
        return True
    if group_count < MIN_INDEPENDENT_GROUPS:
        return True
    if class_counts.successes < MIN_SUCCESSES or class_counts.errors < MIN_ERRORS:
        return True
    return False


def _insufficient_coefficients(labels: np.ndarray) -> tuple[float, float]:
    if labels.size == 0:
        prior = 0.5
    else:
        prior = float(np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    return 0.0, float(math.log(prior / (1.0 - prior)))


def _fit_monotonic_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[float, float]:
    if scores.size == 0:
        return _insufficient_coefficients(labels)
    prior = float(np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    a = 0.0
    b = float(math.log(prior / (1.0 - prior)))
    for _ in range(max_iter):
        z = a * scores + b
        p = _sigmoid(z)
        errors = p - labels
        grad_a = float(np.sum(errors * scores) + l2 * a)
        grad_b = float(np.sum(errors))
        weight = p * (1.0 - p)
        h_aa = float(np.sum(weight * scores * scores) + l2)
        h_ab = float(np.sum(weight * scores))
        h_bb = float(np.sum(weight) + 1e-12)
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-14:
            break
        step_a = (h_bb * grad_a - h_ab * grad_b) / det
        step_b = (-h_ab * grad_a + h_aa * grad_b) / det
        next_a = a - step_a
        next_b = b - step_b
        if next_a < 0.0:
            next_a = 0.0
            next_b = float(math.log(prior / (1.0 - prior)))
        if abs(next_a - a) + abs(next_b - b) <= tol:
            a, b = next_a, next_b
            break
        a, b = next_a, next_b
    return float(max(0.0, a)), float(b)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return np.clip(result, 1e-12, 1.0 - 1e-12)


def _brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.square(probabilities - labels))) if labels.size else 0.0


def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    p = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def _equal_mass_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
    fixed_edges: tuple[float, ...] | None = None,
) -> float:
    if labels.size == 0:
        return 0.0
    edges = fixed_edges or _equal_mass_edges(probabilities, bins=bins)
    p_sorted = np.asarray(probabilities, dtype=np.float64)
    y_sorted = np.asarray(labels, dtype=np.float64)
    if not edges:
        edges = (float(np.max(p_sorted)),)
    chunks: list[np.ndarray] = []
    lower = -np.inf
    remaining = np.arange(labels.size)
    for edge in edges:
        mask = (p_sorted[remaining] > lower) & (p_sorted[remaining] <= edge)
        selected = remaining[mask]
        if selected.size:
            chunks.append(selected)
            remaining = remaining[~mask]
        lower = edge
    if remaining.size:
        chunks.append(remaining)
    ece = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        confidence = float(np.mean(p_sorted[chunk]))
        accuracy = float(np.mean(y_sorted[chunk]))
        ece += (chunk.size / labels.size) * abs(accuracy - confidence)
    return float(min(max(ece, 0.0), 1.0))


def _equal_mass_ece_bins(sample_count: int) -> int:
    return min(10, max(1, int(sample_count) // 20))


def _equal_mass_edges(probabilities: np.ndarray, *, bins: int) -> tuple[float, ...]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size == 0:
        return ()
    safe_bins = max(1, min(int(bins), values.size))
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    edges: list[float] = []
    for bin_index in range(1, safe_bins + 1):
        raw_index = int(round(bin_index * values.size / safe_bins)) - 1
        raw_index = max(0, min(raw_index, values.size - 1))
        edge = float(sorted_values[raw_index])
        while raw_index + 1 < values.size and float(sorted_values[raw_index + 1]) == edge:
            raw_index += 1
        edges.append(float(sorted_values[raw_index]))
    deduped = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return tuple(deduped)


def _calibrated_brier_delta(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    coefficients: tuple[float, float],
) -> float:
    labels = np.asarray(
        [1.0 if row.action_correct else 0.0 for row in rows],
        dtype=np.float64,
    )
    raw_scores = np.asarray(
        [max(0.0, min(float(row.ranking_score), 1.0)) for row in rows],
        dtype=np.float64,
    )
    calibrated = _sigmoid(raw_scores * coefficients[0] + coefficients[1])
    return _brier_score(calibrated, labels) - _brier_score(raw_scores, labels)


def _calibrated_log_loss_delta_vs_intercept(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    coefficients: tuple[float, float],
) -> float:
    labels = np.asarray(
        [1.0 if row.action_correct else 0.0 for row in rows],
        dtype=np.float64,
    )
    raw_scores = np.asarray(
        [max(0.0, min(float(row.ranking_score), 1.0)) for row in rows],
        dtype=np.float64,
    )
    calibrated = _sigmoid(raw_scores * coefficients[0] + coefficients[1])
    intercept_baseline = np.full_like(labels, np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    return _log_loss(calibrated, labels) - _log_loss(intercept_baseline, labels)


def _grouped_bootstrap_ci(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    metric,
    samples: int,
    seed: int,
    dependency_rows: Sequence[RoutingCalibrationObservationV1] | None = None,
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    groups = _component_groups_for_rows(rows, dependency_rows or rows)
    ordered_groups = sorted(groups)
    if len(ordered_groups) < MIN_BOOTSTRAP_GROUPS:
        return None, None
    safe_samples = max(100, min(int(samples), 10000))
    rng = np.random.default_rng(seed)
    estimates = np.empty(shape=(safe_samples,), dtype=np.float64)
    for index in range(safe_samples):
        chosen = rng.choice(ordered_groups, size=len(ordered_groups), replace=True)
        sampled_rows: list[RoutingCalibrationObservationV1] = []
        for group_id in chosen:
            sampled_rows.extend(groups[str(group_id)])
        metric_value = metric(sampled_rows)
        if metric_value is None:
            return None, None
        estimates[index] = float(metric_value)
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def _calibration_slope_intercept(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None]:
    if labels.size < 2 or len({float(value) for value in labels}) < 2:
        return None, None
    logits = np.log(np.clip(probabilities, 1e-12, 1.0 - 1e-12) / np.clip(
        1.0 - probabilities,
        1e-12,
        1.0,
    ))
    if float(np.var(logits)) < 1e-12:
        return None, None
    slope, intercept = _fit_unconstrained_logistic(logits, labels)
    return slope, intercept


def _fit_unconstrained_logistic(
    feature: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[float | None, float | None]:
    alpha = 0.0
    beta = 1.0
    for _ in range(max_iter):
        z = alpha + beta * feature
        p = _sigmoid(z)
        errors = p - labels
        grad_alpha = float(np.sum(errors))
        grad_beta = float(np.sum(errors * feature))
        w = p * (1.0 - p)
        h_aa = float(np.sum(w) + 1e-12)
        h_ab = float(np.sum(w * feature))
        h_bb = float(np.sum(w * feature * feature) + 1e-12)
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-14:
            return None, None
        step_alpha = (h_bb * grad_alpha - h_ab * grad_beta) / det
        step_beta = (-h_ab * grad_alpha + h_aa * grad_beta) / det
        next_alpha = alpha - step_alpha
        next_beta = beta - step_beta
        if abs(next_alpha - alpha) + abs(next_beta - beta) <= tol:
            alpha, beta = next_alpha, next_beta
            break
        alpha, beta = next_alpha, next_beta
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        return None, None
    return float(beta), float(alpha)


def _selective_risk(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        mask = probabilities >= threshold
        covered = int(np.sum(mask))
        coverage = float(covered / labels.size) if labels.size else 0.0
        if covered == 0:
            risk = None
        else:
            risk = float(1.0 - np.mean(labels[mask]))
        values[f">={threshold:.2f}"] = risk
        values[f"coverage>={threshold:.2f}"] = coverage
        values[f"support>={threshold:.2f}"] = float(covered)
    return {key: value for key, value in values.items()}


def _zero_event_binomial_upper_bound(*, n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    return float(1.0 - alpha ** (1.0 / n))


def _dataset_hash(rows: Sequence[RoutingCalibrationObservationV1]) -> str:
    payload_lines = [
        json.dumps(
            row.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in sorted(
            rows,
            key=lambda item: (
                item.observed_at,
                item.request_id,
                item.subtask_index,
            ),
        )
    ]
    return sha256_text("\n".join(payload_lines))


def _window_start(rows: Sequence[RoutingCalibrationObservationV1]) -> datetime | None:
    if not rows:
        return None
    return min(row.observed_at for row in rows)


def _window_end(rows: Sequence[RoutingCalibrationObservationV1]) -> datetime | None:
    if not rows:
        return None
    return max(row.observed_at for row in rows)


def _reject_personal_holdout(rows: Sequence[RoutingCalibrationObservationV1]) -> None:
    if any(_is_personal_holdout(row) for row in rows):
        raise ValueError(
            "Personal holdout jest zabroniony dla fitu/wyboru cech/progów kalibracji."
        )


def _ensure_consistent_training_context(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    runtime_fingerprint: str,
    catalog_hash: str,
) -> None:
    if not rows:
        return
    runtime_values = {row.runtime_fingerprint for row in rows if row.runtime_fingerprint}
    catalog_values = {row.catalog_hash for row in rows if row.catalog_hash}
    if not runtime_values:
        raise ValueError("Brak runtime_fingerprint w reprezentatywnych danych treningowych.")
    if not catalog_values:
        raise ValueError("Brak catalog_hash w reprezentatywnych danych treningowych.")
    if len(runtime_values) > 1:
        raise ValueError("Mieszane runtime_fingerprint w reprezentatywnych danych treningowych.")
    if len(catalog_values) > 1:
        raise ValueError("Mieszane catalog_hash w reprezentatywnych danych treningowych.")
    only_runtime = next(iter(runtime_values))
    only_catalog = next(iter(catalog_values))
    if only_runtime != runtime_fingerprint:
        raise ValueError("Podany runtime_fingerprint nie zgadza sie z danymi treningowymi.")
    if only_catalog != catalog_hash:
        raise ValueError("Podany catalog_hash nie zgadza sie z danymi treningowymi.")


def _validate_snapshots(observations: Sequence[RoutingCalibrationObservationV1]) -> None:
    for row in observations:
        if not row.snapshot_sha256:
            raise ValueError("Brak snapshot_sha256 w obserwacji kalibracyjnej.")
        expected = row.recompute_snapshot_sha256()
        if row.snapshot_sha256 != expected:
            raise ValueError("snapshot_sha256 nie zgadza się z payloadem immutable.")


def _reject_forbidden_provenance(
    observations: Sequence[RoutingCalibrationObservationV1],
    *,
    require_shadow: bool,
) -> None:
    forbidden_markers = (
        "personal_holdout",
        "personal-holdout",
        "routing-v1",
        "routing_v1",
    )
    for row in observations:
        dataset_id = row.dataset_id.casefold()
        if dataset_id in FORBIDDEN_DATASET_IDS:
            raise ValueError(f"Zakazane dataset_id dla kalibracji: {row.dataset_id}")
        if any(marker in dataset_id for marker in forbidden_markers):
            raise ValueError(f"Zakazane dataset_id dla kalibracji: {row.dataset_id}")
        if (
            row.manifest_sha256 is not None
            and row.manifest_sha256.casefold() in FORBIDDEN_MANIFEST_SHA256
        ):
            raise ValueError(
                "Zakazany manifest źródłowy dla kalibracji: "
                f"{row.manifest_sha256}"
            )
        for value in (row.source, row.label_source or "", row.source_record_id):
            lowered = value.casefold()
            if any(marker in lowered for marker in forbidden_markers):
                raise ValueError(
                    "Zakazane pochodzenie kalibracji "
                    f"(dataset_id={row.dataset_id}, source_record_id={row.source_record_id})."
                )
        if require_shadow and not dataset_id.startswith(SHADOW_DATASET_PREFIX):
            raise ValueError(
                "Kalibracja wymaga shadow provenance "
                "(dataset_id routing_v2_shadow_runtime...)."
            )


def validate_calibration_artifact_for_evaluation(
    artifact: RoutingCalibrationArtifactV1,
    observations: Sequence[RoutingCalibrationObservationV1],
) -> bool:
    _reject_forbidden_provenance(observations, require_shadow=True)
    _validate_snapshots(observations)
    if artifact.schema_version != 1:
        raise ValueError("Nieobslugiwany schema_version artefaktu kalibracji.")
    if artifact.recompute_fingerprint() != artifact.artifact_fingerprint:
        raise ValueError("Niezgodny artifact_fingerprint (mozliwe naruszenie integralnosci).")
    representative_rows = [
        row for row in observations if row.set_role is RoutingCalibrationSetRole.REPRESENTATIVE
    ]
    if not representative_rows:
        raise ValueError("Brak reprezentatywnych obserwacji do ewaluacji kalibracji.")
    runtime_values = {row.runtime_fingerprint for row in representative_rows}
    catalog_values = {row.catalog_hash for row in representative_rows}
    if len(runtime_values) != 1:
        raise ValueError("Dane ewaluacyjne maja mieszane runtime_fingerprint.")
    if len(catalog_values) != 1:
        raise ValueError("Dane ewaluacyjne maja mieszane catalog_hash.")
    if next(iter(runtime_values)) != artifact.runtime_fingerprint:
        raise ValueError("runtime_fingerprint danych nie zgadza sie z artefaktem.")
    if next(iter(catalog_values)) != artifact.catalog_hash:
        raise ValueError("catalog_hash danych nie zgadza sie z artefaktem.")
    train_ids = set(artifact.train_component_ids)
    eval_ids = set(artifact.evaluation_component_ids)
    if train_ids.intersection(eval_ids):
        raise ValueError("Artefakt ma przecinajace sie komponenty train/eval.")
    component_map = _row_component_map(observations, representative_rows)
    if artifact.status is RoutingCalibrationArtifactStatus.READY:
        observed_eval_ids = {
            component_map[_row_identity(row)]
            for row in representative_rows
            if _row_identity(row) in component_map
        }
        if eval_ids and observed_eval_ids != eval_ids:
            raise ValueError("Zbior ewaluacyjny nie zgadza sie z komponentami artefaktu.")
        token_overlap = _component_token_set(representative_rows).intersection(
            set(artifact.train_group_tokens)
        )
        if token_overlap:
            raise ValueError("Eval ma overlap tokenów grupowania z treningiem.")
        if artifact.train_cutoff is not None:
            for row in representative_rows:
                if row.observed_at <= artifact.train_cutoff:
                    raise ValueError("Eval zawiera rekordy z czasem <= train_cutoff.")
                component_id = component_map.get(_row_identity(row))
                if component_id in train_ids and row.observed_at > artifact.train_cutoff:
                    raise ValueError("Naruszenie porzadku temporalnego train_cutoff.")
    if artifact.training_window_end is None:
        return False
    return all(row.observed_at > artifact.training_window_end for row in representative_rows)


def _component_tokens(row: RoutingCalibrationObservationV1) -> tuple[str, ...]:
    tokens: list[str] = [f"g:{row.group_id}"]
    if row.session_id:
        tokens.append(f"s:{row.session_id}")
    if row.split_group_override:
        tokens.append(f"o:{row.split_group_override}")
    return tuple(tokens)


def _component_groups(
    rows: Sequence[RoutingCalibrationObservationV1],
) -> dict[str, tuple[RoutingCalibrationObservationV1, ...]]:
    if not rows:
        return {}
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        l_root = find(left)
        r_root = find(right)
        if l_root != r_root:
            parent[r_root] = l_root

    token_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for token in _component_tokens(row):
            token_to_indices[token].append(index)
    for indices in token_to_indices.values():
        for idx in range(1, len(indices)):
            union(indices[0], indices[idx])

    grouped: dict[int, list[RoutingCalibrationObservationV1]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    result: dict[str, tuple[RoutingCalibrationObservationV1, ...]] = {}
    for _, group_rows in grouped.items():
        sorted_rows = tuple(
            sorted(
                group_rows,
                key=lambda row: (row.observed_at, row.request_id, row.subtask_index),
            )
        )
        component_id = sha256_text(
            "|".join(
                sorted(
                    {
                        token
                        for row in sorted_rows
                        for token in _component_tokens(row)
                    }
                )
            )
        )
        result[component_id] = sorted_rows
    return result


def _row_component_map(
    dependency_rows: Sequence[RoutingCalibrationObservationV1],
    target_rows: Sequence[RoutingCalibrationObservationV1] | None = None,
) -> dict[tuple[str, str, int], str]:
    mapping: dict[tuple[str, str, int], str] = {}
    target_keys = (
        {_row_identity(row) for row in target_rows} if target_rows is not None else None
    )
    for component_id, component_rows in _component_groups(dependency_rows).items():
        for row in component_rows:
            key = _row_identity(row)
            if target_keys is None or key in target_keys:
                mapping[key] = component_id
    return mapping


def _component_groups_for_rows(
    target_rows: Sequence[RoutingCalibrationObservationV1],
    dependency_rows: Sequence[RoutingCalibrationObservationV1],
) -> dict[str, tuple[RoutingCalibrationObservationV1, ...]]:
    mapping = _row_component_map(dependency_rows, target_rows)
    grouped: dict[str, list[RoutingCalibrationObservationV1]] = defaultdict(list)
    for row in target_rows:
        component_id = mapping.get(_row_identity(row))
        if component_id is None:
            continue
        grouped[component_id].append(row)
    result: dict[str, tuple[RoutingCalibrationObservationV1, ...]] = {}
    for component_id, rows in grouped.items():
        result[component_id] = tuple(
            sorted(rows, key=lambda item: (item.observed_at, item.request_id, item.subtask_index))
        )
    return result


def _row_identity(row: RoutingCalibrationObservationV1) -> tuple[str, str, int]:
    return (row.dataset_id, row.request_id, row.subtask_index)


def _component_token_set(rows: Sequence[RoutingCalibrationObservationV1]) -> set[str]:
    return {token for row in rows for token in _component_tokens(row)}


def _plan_level_safety_counts(
    observations: Sequence[RoutingCalibrationObservationV1],
) -> tuple[int, int]:
    by_request: dict[tuple[str, str], RoutingCalibrationObservationV1] = {}
    for row in observations:
        if row.safety_outcome is None:
            continue
        key = (row.dataset_id, row.request_id)
        existing = by_request.get(key)
        if existing is not None and existing.safety_outcome != row.safety_outcome:
            raise ValueError(
                "Konflikt safety_outcome dla jednego planu "
                f"(dataset_id={row.dataset_id}, request_id={row.request_id})."
            )
        by_request.setdefault(key, row)
    count = len(by_request)
    unsafe = sum(
        row.safety_outcome is RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
        for row in by_request.values()
    )
    return count, unsafe


def _safety_summary_counts(
    observations: Sequence[RoutingCalibrationObservationV1],
) -> tuple[int, int, int, int]:
    plan_count, unsafe_plan_count = _plan_level_safety_counts(observations)
    safety_rows = [row for row in observations if row.safety_outcome is not None]
    components = _component_groups_for_rows(safety_rows, observations)
    safety_component_count = len(components)
    unsafe_component_count = 0
    for component_rows in components.values():
        if any(
            row.safety_outcome is RoutingCalibrationSafetyOutcome.UNSAFE_RESOLUTION
            for row in component_rows
        ):
            unsafe_component_count += 1
    return plan_count, unsafe_plan_count, safety_component_count, unsafe_component_count


def _calibration_slope_intercept_from_rows(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    coefficients: tuple[float, float],
    return_part: Literal["slope", "intercept"],
) -> float | None:
    labels = np.asarray([1.0 if row.action_correct else 0.0 for row in rows], dtype=np.float64)
    raw_scores = np.asarray(
        [max(0.0, min(float(row.ranking_score), 1.0)) for row in rows],
        dtype=np.float64,
    )
    calibrated = _sigmoid(raw_scores * coefficients[0] + coefficients[1])
    slope, intercept = _calibration_slope_intercept(calibrated, labels)
    if return_part == "slope":
        return slope
    return intercept


def _selective_ci95(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    coefficients: tuple[float, float],
    samples: int,
    seed: int,
    dependency_rows: Sequence[RoutingCalibrationObservationV1] | None = None,
) -> dict[str, dict[str, float | None]]:
    diagnostics: dict[str, dict[str, float | None]] = {}
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        key = f"{threshold:.2f}"
        risk_low, risk_high = _grouped_bootstrap_ci(
            rows,
            metric=lambda sample_rows, thr=threshold: _selective_metric(
                sample_rows,
                coefficients=coefficients,
                threshold=thr,
                kind="risk",
            ),
            samples=samples,
            seed=seed,
            dependency_rows=dependency_rows,
        )
        cov_low, cov_high = _grouped_bootstrap_ci(
            rows,
            metric=lambda sample_rows, thr=threshold: _selective_metric(
                sample_rows,
                coefficients=coefficients,
                threshold=thr,
                kind="coverage",
            ),
            samples=samples,
            seed=seed,
            dependency_rows=dependency_rows,
        )
        diagnostics[key] = {
            "risk_lower": risk_low,
            "risk_upper": risk_high,
            "coverage_lower": cov_low,
            "coverage_upper": cov_high,
        }
    return diagnostics


def _selective_metric(
    rows: Sequence[RoutingCalibrationObservationV1],
    *,
    coefficients: tuple[float, float],
    threshold: float,
    kind: Literal["risk", "coverage"],
) -> float | None:
    labels = np.asarray([1.0 if row.action_correct else 0.0 for row in rows], dtype=np.float64)
    scores = np.asarray(
        [max(0.0, min(float(row.ranking_score), 1.0)) for row in rows],
        dtype=np.float64,
    )
    probabilities = _sigmoid(scores * coefficients[0] + coefficients[1])
    mask = probabilities >= threshold
    covered = int(np.sum(mask))
    if kind == "coverage":
        return float(covered / labels.size) if labels.size else 0.0
    if covered == 0:
        return None
    return float(1.0 - np.mean(labels[mask]))


def _is_personal_holdout(row: RoutingCalibrationObservationV1) -> bool:
    for value in (
        row.dataset_id,
        row.source,
        row.label_source or "",
        row.source_record_id,
        row.group_id,
        row.session_id or "",
    ):
        if "personal_holdout" in value.casefold():
            return True
    return False


def _load_from_sqlite(path: Path) -> list[RoutingCalibrationObservationV1]:
    from ..routing.calibration import RoutingCalibrationObservationStore

    try:
        RoutingCalibrationObservationStore.ensure_schema(path)
    except sqlite3.Error as exc:
        raise ValueError(
            f"Nie udało się zainicjalizować migracji SQLite kalibracji: {path}"
        ) from exc
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    o.*,
                    COALESCE(l.split_group_override, o.split_group_override)
                        AS split_group_override_effective,
                    l.expected_action_id AS label_expected_action_id,
                    l.action_correct AS label_action_correct,
                    l.action_sequence_correct AS label_action_sequence_correct,
                    l.exact_plan_correct AS label_exact_plan_correct,
                    l.safety_outcome AS label_safety_outcome,
                    l.label_source AS label_label_source
                FROM routing_calibration_observations o
                LEFT JOIN routing_calibration_labels l
                    ON l.request_id = o.request_id
                    AND l.subtask_index = o.subtask_index
                    AND l.snapshot_sha256 = o.snapshot_sha256
                ORDER BY observed_at ASC, request_id ASC, subtask_index ASC
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"Nie udało się odczytać SQLite kalibracji: {path}") from exc
    observations: list[RoutingCalibrationObservationV1] = []
    for row in rows:
        row_keys = set(row.keys())
        try:
            reasons_value = row["rejection_reasons_json"] or "[]"
            parsed_reasons = json.loads(str(reasons_value))
            if not isinstance(parsed_reasons, list) or not all(
                isinstance(item, str) for item in parsed_reasons
            ):
                raise ValueError("rejection_reasons_json must decode to list[str]")
            observation = RoutingCalibrationObservationV1(
                request_id=str(row["request_id"]),
                dataset_id=(
                    str(row["dataset_id"])
                    if "dataset_id" in row_keys and row["dataset_id"]
                    else "legacy_unverified"
                ),
                source_record_id=(
                    str(row["source_record_id"])
                    if "source_record_id" in row_keys and row["source_record_id"]
                    else f"{row['request_id']}:{row['subtask_index']}"
                ),
                manifest_sha256=(
                    str(row["manifest_sha256"])
                    if "manifest_sha256" in row_keys and row["manifest_sha256"]
                    else None
                ),
                observed_at=_parse_datetime(row["observed_at"]),
                source=str(row["source"]),
                normalized_text_sha256=str(row["normalized_text_sha256"]),
                runtime_fingerprint=str(row["runtime_fingerprint"]),
                catalog_hash=str(row["catalog_hash"]),
                session_id=str(row["session_id"]) if row["session_id"] else None,
                split_group_override=(
                    str(row["split_group_override_effective"])
                    if "split_group_override_effective" in row_keys
                    and row["split_group_override_effective"]
                    else None
                ),
                group_id=str(row["group_id"]),
                set_role=RoutingCalibrationSetRole(str(row["set_role"])),
                subtask_index=int(row["subtask_index"]),
                subtask_count=int(row["subtask_count"]),
                decision=str(row["decision"]),
                predicted_action_id=(
                    str(row["predicted_action_id"])
                    if "predicted_action_id" in row_keys and row["predicted_action_id"]
                    else None
                ),
                expected_action_id=(
                    str(row["label_expected_action_id"])
                    if "label_expected_action_id" in row_keys
                    and row["label_expected_action_id"]
                    else None
                ),
                ranking_score=_strict_optional_float(row["ranking_score"], "ranking_score"),
                margin_top2=_strict_optional_float(row["margin_top2"], "margin_top2"),
                vector_score=_strict_optional_float(row["vector_score"], "vector_score"),
                lexical_score=_strict_optional_float(row["lexical_score"], "lexical_score"),
                argument_score=_strict_optional_float(row["argument_score"], "argument_score"),
                vector_coverage=_strict_optional_float(
                    row["vector_coverage"], "vector_coverage"
                ),
                stt_confidence=_strict_optional_float(row["stt_confidence"], "stt_confidence"),
                candidate_count=int(row["candidate_count"]),
                eligible=_strict_optional_bool(row["eligible"], "eligible"),
                rejection_reasons=tuple(parsed_reasons),
                p_action_correct=_strict_optional_float(
                    row["p_action_correct"], "p_action_correct"
                ),
                action_correct=_strict_optional_bool(
                    row["label_action_correct"], "action_correct"
                ),
                action_sequence_correct=_strict_optional_bool(
                    row["label_action_sequence_correct"], "action_sequence_correct"
                ),
                exact_plan_correct=_strict_optional_bool(
                    row["label_exact_plan_correct"], "exact_plan_correct"
                ),
                safety_outcome=(
                    RoutingCalibrationSafetyOutcome(str(row["label_safety_outcome"]))
                    if row["label_safety_outcome"]
                    else None
                ),
                label_source=(
                    str(row["label_label_source"]) if row["label_label_source"] else None
                ),
                snapshot_sha256=(
                    str(row["snapshot_sha256"])
                    if "snapshot_sha256" in row_keys and row["snapshot_sha256"]
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            request_id = str(row["request_id"]) if "request_id" in row_keys else "unknown"
            subtask_idx = (
                str(row["subtask_index"]) if "subtask_index" in row_keys else "unknown"
            )
            raise ValueError(
                "Nieprawidłowy rekord SQLite kalibracji "
                f"(request_id={request_id}, subtask_index={subtask_idx})"
            ) from exc
        observations.append(observation)
    return observations


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _strict_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nieprawidłowa wartość numeryczna dla {field_name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Nieskończona/NaN wartość numeryczna dla {field_name}")
    return number


def _strict_optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    raise ValueError(f"Nieprawidłowy bool dla {field_name}; oczekiwano 0/1")
