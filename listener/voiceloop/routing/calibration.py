from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..corpus.schema import (
    RoutingCalibrationArtifactStatus,
    RoutingCalibrationArtifactV1,
    RoutingCalibrationMode,
    RoutingCalibrationObservationV1,
    RoutingCalibrationSetRole,
)
from ..models import (
    CommandRequest,
    ResolutionDecisionV1,
    ResolutionStatusV1,
    normalize_transcript_text,
)
from ..settings import Settings

LOGGER = logging.getLogger("voiceloop.routing.calibration")


def _now_utc() -> datetime:
    return datetime.now(UTC)


class RoutingCalibrationRuntimeStatus(StrEnum):
    OFF = "off"
    MISSING_ARTIFACT = "missing_artifact"
    MALFORMED_ARTIFACT = "malformed_artifact"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    NONFINITE_ARTIFACT = "nonfinite_artifact"
    ARTIFACT_FINGERPRINT_MISMATCH = "artifact_fingerprint_mismatch"
    RUNTIME_FINGERPRINT_MISMATCH = "runtime_fingerprint_mismatch"
    CATALOG_HASH_MISMATCH = "catalog_hash_mismatch"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_REPORTABLE = "not_reportable"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class RoutingCalibrationInference:
    mode: RoutingCalibrationMode
    status: RoutingCalibrationRuntimeStatus
    artifact_fingerprint: str | None
    p_action_correct: tuple[float | None, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "artifact_fingerprint": self.artifact_fingerprint,
            "p_action_correct": list(self.p_action_correct),
        }


@dataclass(slots=True)
class _ArtifactCache:
    loaded_at_mtime_ns: int | None = None
    loaded_at_size: int | None = None
    artifact: RoutingCalibrationArtifactV1 | None = None
    parse_status: RoutingCalibrationRuntimeStatus = RoutingCalibrationRuntimeStatus.MISSING_ARTIFACT
    fingerprint: str | None = None


class RoutingCalibrationRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = _parse_mode(settings.routing_v2_calibration_mode)
        self.artifact_path = settings.routing_v2_calibration_artifact_path
        self._cache = _ArtifactCache()
        self._preloaded = False

    async def preload(self) -> None:
        if self.mode is RoutingCalibrationMode.OFF:
            self._preloaded = True
            self._cache = _ArtifactCache(
                artifact=None,
                parse_status=RoutingCalibrationRuntimeStatus.OFF,
                fingerprint=None,
            )
            return
        await asyncio.to_thread(self._load_artifact_for_startup)

    def _load_artifact_for_startup(self) -> None:
        if self.mode is RoutingCalibrationMode.OFF:
            self._preloaded = True
            self._cache = _ArtifactCache(
                artifact=None,
                parse_status=RoutingCalibrationRuntimeStatus.OFF,
                fingerprint=None,
            )
            return
        try:
            self._load_artifact()
        except Exception:
            self._cache = _ArtifactCache(
                artifact=None,
                parse_status=RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT,
                fingerprint=None,
            )
        finally:
            self._preloaded = True

    def infer(
        self,
        decisions: Sequence[ResolutionDecisionV1],
        *,
        expected_runtime_fingerprint: str,
        expected_catalog_hash: str,
    ) -> RoutingCalibrationInference:
        if self.mode is RoutingCalibrationMode.OFF:
            return RoutingCalibrationInference(
                mode=self.mode,
                status=RoutingCalibrationRuntimeStatus.OFF,
                artifact_fingerprint=None,
                p_action_correct=tuple(None for _ in decisions),
            )
        if not self._preloaded:
            try:
                artifact, parse_status, fingerprint = self._load_artifact()
            except Exception:
                artifact, parse_status, fingerprint = (
                    None,
                    RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT,
                    None,
                )
            self._preloaded = True
        else:
            artifact, parse_status, fingerprint = (
                self._cache.artifact,
                self._cache.parse_status,
                self._cache.fingerprint,
            )
        status = self._status_for_expectations(
            artifact,
            parse_status=parse_status,
            expected_runtime_fingerprint=expected_runtime_fingerprint,
            expected_catalog_hash=expected_catalog_hash,
        )
        probabilities = tuple(
            _calibrated_probability(
                artifact,
                ranking_score=_ranking_score(decision),
            )
            if (
                status is RoutingCalibrationRuntimeStatus.READY
                and _decision_probability_eligible(decision)
            )
            else None
            for decision in decisions
        )
        return RoutingCalibrationInference(
            mode=self.mode,
            status=status,
            artifact_fingerprint=fingerprint,
            p_action_correct=probabilities,
        )

    def _load_artifact(self) -> tuple[
        RoutingCalibrationArtifactV1 | None,
        RoutingCalibrationRuntimeStatus,
        str | None,
    ]:
        path = self.artifact_path
        if not path.is_file():
            return (
                None,
                RoutingCalibrationRuntimeStatus.MISSING_ARTIFACT,
                None,
            )
        try:
            stat = path.stat()
        except OSError:
            return None, RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT, None
        cache = self._cache
        if (
            cache.loaded_at_mtime_ns == stat.st_mtime_ns
            and cache.loaded_at_size == stat.st_size
        ):
            return cache.artifact, cache.parse_status, cache.fingerprint
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            status = RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
            artifact = None
            fingerprint = None
        else:
            status, artifact, fingerprint = self._validate_artifact(payload)
        self._cache = _ArtifactCache(
            loaded_at_mtime_ns=stat.st_mtime_ns,
            loaded_at_size=stat.st_size,
            artifact=artifact,
            parse_status=status,
            fingerprint=fingerprint,
        )
        return artifact, status, fingerprint

    def _validate_artifact(
        self,
        payload: Any,
    ) -> tuple[
        RoutingCalibrationRuntimeStatus,
        RoutingCalibrationArtifactV1 | None,
        str | None,
    ]:
        if not isinstance(payload, dict):
            return RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT, None, None
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            return RoutingCalibrationRuntimeStatus.UNSUPPORTED_SCHEMA, None, None
        coefficients = payload.get("coefficients")
        if isinstance(coefficients, dict):
            for field in ("a", "b"):
                if field not in coefficients:
                    continue
                try:
                    value = float(coefficients[field])
                except (TypeError, ValueError):
                    return RoutingCalibrationRuntimeStatus.NONFINITE_ARTIFACT, None, None
                if not math.isfinite(value):
                    return RoutingCalibrationRuntimeStatus.NONFINITE_ARTIFACT, None, None
        try:
            artifact = RoutingCalibrationArtifactV1.model_validate(payload)
        except ValueError:
            return RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT, None, None
        if not math.isfinite(artifact.coefficients.a) or not math.isfinite(
            artifact.coefficients.b
        ):
            return RoutingCalibrationRuntimeStatus.NONFINITE_ARTIFACT, None, None
        recomputed = artifact.recompute_fingerprint()
        if recomputed != artifact.artifact_fingerprint:
            return (
                RoutingCalibrationRuntimeStatus.ARTIFACT_FINGERPRINT_MISMATCH,
                None,
                artifact.artifact_fingerprint,
            )
        if artifact.status is RoutingCalibrationArtifactStatus.INSUFFICIENT_DATA:
            return (
                RoutingCalibrationRuntimeStatus.INSUFFICIENT_DATA,
                artifact,
                artifact.artifact_fingerprint,
            )
        if artifact.status is RoutingCalibrationArtifactStatus.NOT_REPORTABLE:
            return (
                RoutingCalibrationRuntimeStatus.NOT_REPORTABLE,
                artifact,
                artifact.artifact_fingerprint,
            )
        return (
            RoutingCalibrationRuntimeStatus.READY,
            artifact,
            artifact.artifact_fingerprint,
        )

    def _status_for_expectations(
        self,
        artifact: RoutingCalibrationArtifactV1 | None,
        *,
        parse_status: RoutingCalibrationRuntimeStatus,
        expected_runtime_fingerprint: str,
        expected_catalog_hash: str,
    ) -> RoutingCalibrationRuntimeStatus:
        if parse_status is not RoutingCalibrationRuntimeStatus.READY:
            return parse_status
        if artifact is None:
            return RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT
        if artifact.runtime_fingerprint != expected_runtime_fingerprint:
            return RoutingCalibrationRuntimeStatus.RUNTIME_FINGERPRINT_MISMATCH
        if artifact.catalog_hash != expected_catalog_hash:
            return RoutingCalibrationRuntimeStatus.CATALOG_HASH_MISMATCH
        return RoutingCalibrationRuntimeStatus.READY


class RoutingCalibrationObservationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def initialize(self) -> None:
        await asyncio.to_thread(self.ensure_schema, self.database_path)

    @classmethod
    def ensure_schema(cls, database_path: Path) -> None:
        core_required_columns = {
            "request_id",
            "subtask_index",
            "observed_at",
            "source",
            "normalized_text_sha256",
            "runtime_fingerprint",
            "catalog_hash",
            "group_id",
            "set_role",
            "subtask_count",
            "decision",
            "candidate_count",
            "eligible",
            "rejection_reasons_json",
        }
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            existing_table = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name='routing_calibration_observations'
                """
            ).fetchone()
            preexisting_columns: set[str] = set()
            if existing_table is not None:
                preexisting_columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(routing_calibration_observations)"
                    ).fetchall()
                }
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS routing_calibration_observations (
                    request_id TEXT NOT NULL,
                    subtask_index INTEGER NOT NULL,
                    dataset_id TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    manifest_sha256 TEXT,
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
                    predicted_action_id TEXT,
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
                    snapshot_sha256 TEXT NOT NULL,
                    PRIMARY KEY (request_id, subtask_index)
                );
                CREATE TABLE IF NOT EXISTS routing_calibration_labels (
                    request_id TEXT NOT NULL,
                    subtask_index INTEGER NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    split_group_override TEXT,
                    expected_action_id TEXT,
                    action_correct INTEGER,
                    action_sequence_correct INTEGER,
                    exact_plan_correct INTEGER,
                    safety_outcome TEXT,
                    label_source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (request_id, subtask_index),
                    FOREIGN KEY (request_id, subtask_index)
                        REFERENCES routing_calibration_observations(request_id, subtask_index)
                );
                CREATE INDEX IF NOT EXISTS idx_routing_calibration_observed_at
                    ON routing_calibration_observations(observed_at);
                CREATE INDEX IF NOT EXISTS idx_routing_calibration_group
                    ON routing_calibration_observations(group_id, observed_at);
                CREATE INDEX IF NOT EXISTS idx_routing_calibration_role
                    ON routing_calibration_observations(set_role, observed_at);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(routing_calibration_observations)"
                ).fetchall()
            }
            for column, definition in (
                ("dataset_id", "TEXT"),
                ("source_record_id", "TEXT"),
                ("manifest_sha256", "TEXT"),
                ("split_group_override", "TEXT"),
                ("predicted_action_id", "TEXT"),
                ("p_action_correct", "REAL"),
                ("snapshot_sha256", "TEXT"),
            ):
                if column not in columns:
                    connection.execute(
                        "ALTER TABLE routing_calibration_observations "
                        f"ADD COLUMN {column} {definition}"
                    )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(routing_calibration_observations)"
                ).fetchall()
            }
            missing_core_columns = sorted(core_required_columns - columns)
            if missing_core_columns:
                raise ValueError(
                    "SQLite observation schema missing core columns: "
                    + ", ".join(missing_core_columns)
                )
            connection.execute(
                """
                UPDATE routing_calibration_observations
                SET dataset_id = COALESCE(dataset_id, 'legacy_unverified')
                WHERE dataset_id IS NULL OR dataset_id = ''
                """
            )
            connection.execute(
                """
                UPDATE routing_calibration_observations
                SET source_record_id = COALESCE(
                    source_record_id,
                    request_id || ':' || subtask_index
                )
                WHERE source_record_id IS NULL OR source_record_id = ''
                """
            )
            label_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(routing_calibration_labels)"
                ).fetchall()
            }
            if "split_group_override" not in label_columns:
                connection.execute(
                    """
                    ALTER TABLE routing_calibration_labels
                    ADD COLUMN split_group_override TEXT
                    """
                )
                label_columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(routing_calibration_labels)"
                    ).fetchall()
                }
            missing_label_columns = sorted(
                {
                    "request_id",
                    "subtask_index",
                    "snapshot_sha256",
                    "split_group_override",
                    "expected_action_id",
                    "action_correct",
                    "action_sequence_correct",
                    "exact_plan_correct",
                    "safety_outcome",
                    "label_source",
                    "updated_at",
                }
                - label_columns
            )
            if missing_label_columns:
                raise ValueError(
                    "SQLite label schema missing core columns: "
                    + ", ".join(missing_label_columns)
                )
            rows = connection.execute(
                """
                SELECT * FROM routing_calibration_observations
                ORDER BY observed_at ASC, request_id ASC, subtask_index ASC
                """
            ).fetchall()
            had_snapshot_column_pre_migration = "snapshot_sha256" in preexisting_columns
            for row in rows:
                computed_snapshot = _snapshot_from_existing_row(row)
                stored_snapshot_raw = (
                    str(row["snapshot_sha256"]).strip()
                    if "snapshot_sha256" in row.keys() and row["snapshot_sha256"] is not None
                    else ""
                )
                snapshot = stored_snapshot_raw
                if snapshot:
                    if snapshot != computed_snapshot:
                        raise ValueError(
                            "snapshot_sha256 mismatch for persisted immutable observation "
                            f"{row['request_id']}:{row['subtask_index']}"
                        )
                else:
                    # Legacy rows that predate snapshot storage establish
                    # one canonical immutable snapshot during additive migration.
                    snapshot = computed_snapshot
                    connection.execute(
                        """
                        UPDATE routing_calibration_observations
                        SET snapshot_sha256 = ?
                        WHERE request_id=? AND subtask_index=?
                        """,
                        (snapshot, str(row["request_id"]), int(row["subtask_index"])),
                    )
                if (
                    had_snapshot_column_pre_migration
                    and stored_snapshot_raw
                    and snapshot != stored_snapshot_raw
                ):
                    raise ValueError(
                        "snapshot provenance violation during migration "
                        f"for {row['request_id']}:{row['subtask_index']}"
                    )
                label_source = (
                    str(row["label_source"])
                    if "label_source" in row.keys() and row["label_source"] is not None
                    else None
                )
                split_group_override = (
                    str(row["split_group_override"])
                    if "split_group_override" in row.keys() and row["split_group_override"]
                    else None
                )
                has_label_columns = all(
                    column_name in row.keys()
                    for column_name in (
                        "action_correct",
                        "action_sequence_correct",
                        "exact_plan_correct",
                        "safety_outcome",
                    )
                )
                if label_source is not None and has_label_columns:
                    connection.execute(
                        """
                        INSERT INTO routing_calibration_labels (
                            request_id, subtask_index, snapshot_sha256,
                            split_group_override,
                            expected_action_id, action_correct, action_sequence_correct,
                            exact_plan_correct, safety_outcome, label_source, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(request_id, subtask_index) DO NOTHING
                        """,
                        (
                            str(row["request_id"]),
                            int(row["subtask_index"]),
                            snapshot,
                            split_group_override,
                            (
                                str(row["expected_action_id"])
                                if (
                                    "expected_action_id" in row.keys()
                                    and "predicted_action_id" in row.keys()
                                    and row["expected_action_id"]
                                    and row["predicted_action_id"]
                                )
                                else None
                            ),
                            _opt_bool_to_int(row["action_correct"]),
                            _opt_bool_to_int(row["action_sequence_correct"]),
                            _opt_bool_to_int(row["exact_plan_correct"]),
                            str(row["safety_outcome"]) if row["safety_outcome"] else None,
                            label_source,
                            str(row["observed_at"] or datetime.now(UTC).isoformat()),
                        ),
                    )
                elif split_group_override is not None:
                    connection.execute(
                        """
                        INSERT INTO routing_calibration_labels (
                            request_id, subtask_index, snapshot_sha256,
                            split_group_override,
                            expected_action_id, action_correct, action_sequence_correct,
                            exact_plan_correct, safety_outcome, label_source, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(request_id, subtask_index) DO NOTHING
                        """,
                        (
                            str(row["request_id"]),
                            int(row["subtask_index"]),
                            snapshot,
                            split_group_override,
                            None,
                            None,
                            None,
                            None,
                            None,
                            "legacy_grouping",
                            str(row["observed_at"] or datetime.now(UTC).isoformat()),
                        ),
                    )

    async def append_many(
        self,
        observations: Sequence[RoutingCalibrationObservationV1],
    ) -> int:
        rows = [observation.model_dump(mode="json") for observation in observations]
        if not rows:
            return 0

        def append_sync() -> int:
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO routing_calibration_observations (
                        request_id, subtask_index, dataset_id, source_record_id, manifest_sha256,
                        observed_at, source,
                        normalized_text_sha256, runtime_fingerprint, catalog_hash,
                        session_id, split_group_override, group_id, set_role,
                        subtask_count, decision,
                        predicted_action_id,
                        ranking_score, margin_top2, vector_score, lexical_score,
                        argument_score, vector_coverage, stt_confidence,
                        candidate_count, eligible, rejection_reasons_json,
                        p_action_correct, snapshot_sha256
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(request_id, subtask_index) DO UPDATE SET
                        snapshot_sha256 = routing_calibration_observations.snapshot_sha256
                    """,
                    [
                        (
                            str(row["request_id"]),
                            int(row["subtask_index"]),
                            str(row["dataset_id"]),
                            str(row["source_record_id"]),
                            str(row["manifest_sha256"]) if row.get("manifest_sha256") else None,
                            str(row["observed_at"]),
                            str(row["source"]),
                            str(row["normalized_text_sha256"]),
                            str(row["runtime_fingerprint"]),
                            str(row["catalog_hash"]),
                            str(row["session_id"]) if row.get("session_id") else None,
                            None,
                            str(row["group_id"]),
                            str(row["set_role"]),
                            int(row["subtask_count"]),
                            str(row["decision"]),
                            str(row["predicted_action_id"])
                            if row.get("predicted_action_id")
                            else None,
                            _opt_float(row.get("ranking_score")),
                            _opt_float(row.get("margin_top2")),
                            _opt_float(row.get("vector_score")),
                            _opt_float(row.get("lexical_score")),
                            _opt_float(row.get("argument_score")),
                            _opt_float(row.get("vector_coverage")),
                            _opt_float(row.get("stt_confidence")),
                            int(row["candidate_count"]),
                            _opt_bool_to_int(row.get("eligible")),
                            json.dumps(
                                list(row.get("rejection_reasons") or ()),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            _opt_float(row.get("p_action_correct")),
                            _snapshot_from_row_payload(row),
                        )
                        for row in rows
                    ],
                )
                for row in rows:
                    snapshot = _snapshot_from_row_payload(row)
                    existing = connection.execute(
                        """
                        SELECT snapshot_sha256 FROM routing_calibration_observations
                        WHERE request_id=? AND subtask_index=?
                        """,
                        (str(row["request_id"]), int(row["subtask_index"])),
                    ).fetchone()
                    if existing is not None and str(existing["snapshot_sha256"]) != str(
                        snapshot
                    ):
                        raise ValueError(
                            "Conflicting immutable snapshot for request_id/subtask_index."
                        )
                labeled_rows = [
                    row
                    for row in rows
                    if (
                        row.get("label_source") is not None
                        or row.get("expected_action_id") is not None
                        or row.get("action_correct") is not None
                        or row.get("action_sequence_correct") is not None
                        or row.get("exact_plan_correct") is not None
                        or row.get("safety_outcome") is not None
                        or row.get("split_group_override") is not None
                    )
                ]
                if labeled_rows:
                    connection.executemany(
                        """
                        INSERT INTO routing_calibration_labels (
                            request_id, subtask_index, snapshot_sha256,
                            split_group_override,
                            expected_action_id, action_correct, action_sequence_correct,
                            exact_plan_correct, safety_outcome, label_source, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(request_id, subtask_index) DO UPDATE SET
                            snapshot_sha256 = excluded.snapshot_sha256,
                            split_group_override = excluded.split_group_override,
                            expected_action_id = excluded.expected_action_id,
                            action_correct = excluded.action_correct,
                            action_sequence_correct = excluded.action_sequence_correct,
                            exact_plan_correct = excluded.exact_plan_correct,
                            safety_outcome = excluded.safety_outcome,
                            label_source = excluded.label_source,
                            updated_at = excluded.updated_at
                        """,
                        [
                            (
                                str(row["request_id"]),
                                int(row["subtask_index"]),
                                _snapshot_from_row_payload(row),
                                str(row["split_group_override"])
                                if row.get("split_group_override")
                                else None,
                                str(row["expected_action_id"])
                                if row.get("expected_action_id")
                                else None,
                                _opt_bool_to_int(row.get("action_correct")),
                                _opt_bool_to_int(row.get("action_sequence_correct")),
                                _opt_bool_to_int(row.get("exact_plan_correct")),
                                str(row["safety_outcome"])
                                if row.get("safety_outcome")
                                else None,
                                str(row["label_source"] or "manual"),
                                datetime.now(UTC).isoformat(),
                            )
                            for row in labeled_rows
                        ],
                    )
            return len(rows)

        return await asyncio.to_thread(append_sync)

    async def fetch_recent(
        self,
        *,
        limit: int = 5000,
    ) -> list[RoutingCalibrationObservationV1]:
        safe_limit = max(1, min(limit, 200000))

        def fetch_sync() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT
                        o.*,
                        COALESCE(
                            l.split_group_override,
                            o.split_group_override
                        ) AS split_group_override_effective,
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
                    ORDER BY observed_at ASC, subtask_index ASC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        rows = await asyncio.to_thread(fetch_sync)
        observations: list[RoutingCalibrationObservationV1] = []
        for row in rows:
            observations.append(
                RoutingCalibrationObservationV1(
                    request_id=str(row["request_id"]),
                    dataset_id=str(row["dataset_id"]),
                    source_record_id=str(row["source_record_id"]),
                    manifest_sha256=(
                        str(row["manifest_sha256"]) if row["manifest_sha256"] else None
                    ),
                    observed_at=datetime.fromisoformat(str(row["observed_at"])),
                    source=str(row["source"]),
                    normalized_text_sha256=str(row["normalized_text_sha256"]),
                    runtime_fingerprint=str(row["runtime_fingerprint"]),
                    catalog_hash=str(row["catalog_hash"]),
                    session_id=str(row["session_id"]) if row["session_id"] else None,
                    split_group_override=(
                        str(row["split_group_override_effective"])
                        if row["split_group_override_effective"]
                        else None
                    ),
                    group_id=str(row["group_id"]),
                    set_role=RoutingCalibrationSetRole(str(row["set_role"])),
                    subtask_index=int(row["subtask_index"]),
                    subtask_count=int(row["subtask_count"]),
                    decision=str(row["decision"]),
                    predicted_action_id=(
                        str(row["predicted_action_id"])
                        if row["predicted_action_id"]
                        else None
                    ),
                    expected_action_id=(
                        str(row["label_expected_action_id"])
                        if row["label_expected_action_id"]
                        else None
                    ),
                    ranking_score=_row_float(row["ranking_score"], field_name="ranking_score"),
                    margin_top2=_row_float(row["margin_top2"], field_name="margin_top2"),
                    vector_score=_row_float(row["vector_score"], field_name="vector_score"),
                    lexical_score=_row_float(row["lexical_score"], field_name="lexical_score"),
                    argument_score=_row_float(
                        row["argument_score"], field_name="argument_score"
                    ),
                    vector_coverage=_row_float(
                        row["vector_coverage"], field_name="vector_coverage"
                    ),
                    stt_confidence=_row_float(row["stt_confidence"], field_name="stt_confidence"),
                    candidate_count=int(row["candidate_count"]),
                    eligible=_row_bool(row["eligible"], field_name="eligible"),
                    rejection_reasons=tuple(
                        json.loads(row["rejection_reasons_json"] or "[]")
                    ),
                    p_action_correct=_row_float(
                        row["p_action_correct"], field_name="p_action_correct"
                    ),
                    action_correct=_row_bool(
                        row["label_action_correct"], field_name="action_correct"
                    ),
                    action_sequence_correct=_row_bool(
                        row["label_action_sequence_correct"],
                        field_name="action_sequence_correct",
                    ),
                    exact_plan_correct=_row_bool(
                        row["label_exact_plan_correct"], field_name="exact_plan_correct"
                    ),
                    safety_outcome=str(row["label_safety_outcome"])
                    if row["label_safety_outcome"]
                    else None,
                    label_source=(
                        str(row["label_label_source"])
                        if row["label_label_source"]
                        else None
                    ),
                    snapshot_sha256=str(row["snapshot_sha256"]),
                )
            )
        return observations


class RoutingCalibrationRecorder:
    def __init__(
        self,
        *,
        store: RoutingCalibrationObservationStore,
        queue_limit: int,
        enabled: bool,
    ) -> None:
        self.store = store
        self.queue_limit = max(10, min(int(queue_limit), 200000))
        self.enabled = enabled
        self._queue: asyncio.Queue[tuple[RoutingCalibrationObservationV1, ...] | None] = (
            asyncio.Queue(maxsize=self.queue_limit)
        )
        self._worker: asyncio.Task[None] | None = None
        self._started = False
        self._dropped_count = 0
        self._accepted_count = 0
        self._failed_count = 0

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    async def start(self) -> None:
        if not self.enabled or self._started:
            return
        try:
            await self.store.initialize()
        except Exception:
            LOGGER.exception("Routing calibration recorder initialization failed")
            self._failed_count += 1
            self.enabled = False
            return
        self._worker = asyncio.create_task(
            self._worker_loop(),
            name="routing-calibration-recorder",
        )
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            await self._queue.put(None)
        worker = self._worker
        self._worker = None
        if worker is not None:
            await worker

    def record(self, observations: Sequence[RoutingCalibrationObservationV1]) -> None:
        if not self.enabled or not self._started:
            return
        if not observations:
            return
        payload = tuple(observations)
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            self._dropped_count += len(payload)
            return
        self._accepted_count += len(payload)

    async def _worker_loop(self) -> None:
        while True:
            batch = await self._queue.get()
            try:
                if batch is None:
                    return
                await self.store.append_many(batch)
            except Exception:
                self._failed_count += len(batch) if batch is not None else 0
                LOGGER.exception("Routing calibration recorder failed to append observations")
            finally:
                self._queue.task_done()


def build_calibration_observations(
    *,
    request: CommandRequest,
    decisions: Sequence[ResolutionDecisionV1],
    subtask_count: int,
    runtime_fingerprint: str,
    catalog_hash: str,
    inference: RoutingCalibrationInference,
) -> tuple[RoutingCalibrationObservationV1, ...]:
    if not decisions:
        return ()
    normalized = normalize_transcript_text(request.text or "")
    canonical_text = _canonical_duplicate_text(normalized)
    normalized_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    group_id = normalized_hash
    session_id = request.interaction_session_id
    dataset_id = "routing_v2_shadow_runtime"
    observed_at = request.created_at.astimezone(UTC)
    observations: list[RoutingCalibrationObservationV1] = []
    for index, decision in enumerate(decisions):
        candidate = decision.candidates[0] if decision.candidates else None
        set_role = _infer_set_role(decision)
        base_observation = RoutingCalibrationObservationV1(
            request_id=request.request_id,
            dataset_id=dataset_id,
            source_record_id=f"{request.request_id}:{index}",
            observed_at=observed_at,
            source=request.source.value,
            normalized_text_sha256=normalized_hash,
            runtime_fingerprint=runtime_fingerprint,
            catalog_hash=decision.catalog_hash or catalog_hash,
            session_id=session_id,
            group_id=group_id,
            set_role=set_role,
            subtask_index=index,
            subtask_count=max(1, subtask_count),
            decision=decision.decision.value,
            predicted_action_id=decision.top1_action_id,
            ranking_score=_ranking_score(decision),
            margin_top2=decision.margin_top2,
            vector_score=candidate.vector_score if candidate is not None else None,
            lexical_score=candidate.lexical_score if candidate is not None else None,
            argument_score=candidate.argument_compatibility if candidate is not None else None,
            vector_coverage=candidate.coverage if candidate is not None else None,
            stt_confidence=decision.stt_confidence,
            candidate_count=len(decision.candidates),
            eligible=candidate.eligible if candidate is not None else None,
            rejection_reasons=(
                tuple(candidate.rejection_reasons) if candidate is not None else ()
            ),
            p_action_correct=(
                inference.p_action_correct[index]
                if index < len(inference.p_action_correct)
                else None
            ),
            snapshot_sha256=None,
        )
        observations.append(
            base_observation.model_copy(
                update={"snapshot_sha256": base_observation.recompute_snapshot_sha256()}
            )
        )
    return tuple(observations)


def build_challenge_observation(
    *,
    request: CommandRequest,
    reason: str,
    decision: str,
    runtime_fingerprint: str,
    catalog_hash: str,
) -> RoutingCalibrationObservationV1:
    normalized = normalize_transcript_text(request.text or "")
    canonical_text = _canonical_duplicate_text(normalized)
    normalized_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    session_id = request.interaction_session_id
    observed_at = request.created_at.astimezone(UTC)
    observation = RoutingCalibrationObservationV1(
        request_id=request.request_id,
        dataset_id="routing_v2_shadow_runtime",
        source_record_id=f"{request.request_id}:challenge:0",
        observed_at=observed_at,
        source=request.source.value,
        normalized_text_sha256=normalized_hash,
        runtime_fingerprint=runtime_fingerprint,
        catalog_hash=catalog_hash,
        session_id=session_id,
        group_id=normalized_hash,
        set_role=RoutingCalibrationSetRole.CHALLENGE,
        subtask_index=0,
        subtask_count=1,
        decision=decision,
        predicted_action_id=None,
        candidate_count=0,
        rejection_reasons=(reason,),
        snapshot_sha256=None,
    )
    return observation.model_copy(
        update={"snapshot_sha256": observation.recompute_snapshot_sha256()}
    )


def build_calibration_recorder(settings: Settings) -> RoutingCalibrationRecorder:
    mode = _parse_mode(settings.routing_v2_calibration_mode)
    enabled = mode is RoutingCalibrationMode.REPORT_ONLY
    store = RoutingCalibrationObservationStore(settings.routing_v2_calibration_store_path)
    return RoutingCalibrationRecorder(
        store=store,
        queue_limit=settings.routing_v2_calibration_queue_limit,
        enabled=enabled,
    )


def _parse_mode(value: str) -> RoutingCalibrationMode:
    normalized = str(value or "").strip().lower()
    try:
        return RoutingCalibrationMode(normalized)
    except ValueError:
        LOGGER.warning("Unknown routing calibration mode=%s; fallback to off", normalized)
        return RoutingCalibrationMode.OFF


def _calibrated_probability(
    artifact: RoutingCalibrationArtifactV1 | None,
    *,
    ranking_score: float | None,
) -> float | None:
    if artifact is None:
        return None
    if ranking_score is None or not math.isfinite(ranking_score):
        return None
    score = max(0.0, min(float(ranking_score), 1.0))
    z = artifact.coefficients.a * score + artifact.coefficients.b
    if not math.isfinite(z):
        return None
    if z >= 0.0:
        exp_component = math.exp(-z)
        probability = 1.0 / (1.0 + exp_component)
    else:
        exp_component = math.exp(z)
        probability = exp_component / (1.0 + exp_component)
    if not math.isfinite(probability):
        return None
    return max(0.0, min(probability, 1.0))


def _ranking_score(decision: ResolutionDecisionV1) -> float | None:
    if not decision.candidates:
        return None
    value = decision.candidates[0].combined_score
    if not math.isfinite(value):
        return None
    return max(0.0, min(float(value), 1.0))


def _infer_set_role(decision: ResolutionDecisionV1) -> RoutingCalibrationSetRole:
    if decision.decision is not ResolutionStatusV1.RESOLVED:
        return RoutingCalibrationSetRole.CHALLENGE
    if decision.margin_top2 is None or decision.margin_top2 < 0.15:
        return RoutingCalibrationSetRole.CHALLENGE
    if decision.candidates and (
        not decision.candidates[0].eligible or decision.candidates[0].rejection_reasons
    ):
        return RoutingCalibrationSetRole.CHALLENGE
    return RoutingCalibrationSetRole.REPRESENTATIVE


def _decision_probability_eligible(decision: ResolutionDecisionV1) -> bool:
    return _infer_set_role(decision) is RoutingCalibrationSetRole.REPRESENTATIVE


def _opt_bool_to_int(value: Any) -> int | None:
    parsed = _strict_optional_bool(value, field_name="bool_scalar")
    if parsed is None:
        return None
    return 1 if parsed else 0


def _row_bool(value: Any, *, field_name: str) -> bool | None:
    return _strict_optional_bool(value, field_name=field_name)


def _opt_float(value: Any) -> float | None:
    return _strict_optional_float(value, field_name="float_scalar")


def _row_float(value: Any, *, field_name: str) -> float | None:
    return _strict_optional_float(value, field_name=field_name)


def _parse_datetime_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _canonical_duplicate_text(text: str) -> str:
    lowered = text.casefold()
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in lowered)
    return " ".join(cleaned.split())


def _observation_snapshot_sha256(**payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _snapshot_from_row_payload(row: dict[str, Any]) -> str:
    raw_reasons = row.get("rejection_reasons")
    if raw_reasons is None:
        rejection_reasons: tuple[str, ...] = ()
    elif isinstance(raw_reasons, (list, tuple)):
        if not all(isinstance(item, str) for item in raw_reasons):
            raise ValueError("rejection_reasons must contain only strings")
        rejection_reasons = tuple(str(item) for item in raw_reasons)
    else:
        raise ValueError("rejection_reasons must be list/tuple[str]")
    observation = RoutingCalibrationObservationV1(
        request_id=str(row["request_id"]),
        dataset_id=str(row["dataset_id"]),
        source_record_id=str(row["source_record_id"]),
        manifest_sha256=str(row["manifest_sha256"]) if row.get("manifest_sha256") else None,
        observed_at=_parse_datetime_utc(str(row["observed_at"])),
        source=str(row["source"]),
        normalized_text_sha256=str(row["normalized_text_sha256"]),
        runtime_fingerprint=str(row["runtime_fingerprint"]),
        catalog_hash=str(row["catalog_hash"]),
        session_id=str(row["session_id"]) if row.get("session_id") else None,
        split_group_override=None,
        group_id=str(row["group_id"]),
        set_role=RoutingCalibrationSetRole(str(row["set_role"])),
        subtask_index=int(row["subtask_index"]),
        subtask_count=int(row["subtask_count"]),
        decision=str(row["decision"]),
        predicted_action_id=(
            str(row["predicted_action_id"]) if row.get("predicted_action_id") else None
        ),
        ranking_score=_strict_optional_float(row.get("ranking_score"), field_name="ranking_score"),
        margin_top2=_strict_optional_float(row.get("margin_top2"), field_name="margin_top2"),
        vector_score=_strict_optional_float(row.get("vector_score"), field_name="vector_score"),
        lexical_score=_strict_optional_float(row.get("lexical_score"), field_name="lexical_score"),
        argument_score=_strict_optional_float(
            row.get("argument_score"), field_name="argument_score"
        ),
        vector_coverage=_strict_optional_float(
            row.get("vector_coverage"), field_name="vector_coverage"
        ),
        stt_confidence=_strict_optional_float(
            row.get("stt_confidence"), field_name="stt_confidence"
        ),
        candidate_count=int(row["candidate_count"]),
        eligible=_strict_optional_bool(row.get("eligible"), field_name="eligible"),
        rejection_reasons=rejection_reasons,
        p_action_correct=_strict_optional_float(
            row.get("p_action_correct"), field_name="p_action_correct"
        ),
        snapshot_sha256=None,
    )
    return observation.recompute_snapshot_sha256()


def _snapshot_from_existing_row(row: sqlite3.Row) -> str:
    payload = dict(row)
    if "rejection_reasons_json" in payload:
        try:
            parsed = json.loads(str(payload.get("rejection_reasons_json") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid rejection_reasons_json in SQLite observation") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("rejection_reasons_json must decode to list[str]")
        payload["rejection_reasons"] = tuple(parsed)
    return _snapshot_from_row_payload(payload)


def _strict_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field_name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {field_name}")
    return number


def _strict_optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    raise ValueError(f"Invalid boolean scalar for {field_name}; expected 0/1 or bool")
