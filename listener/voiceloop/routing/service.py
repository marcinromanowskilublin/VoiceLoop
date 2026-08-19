from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..capability_index import CapabilityIndex, SubtaskCapabilitySearch
from ..corpus.schema import RoutingCalibrationMode, RoutingV2RuntimeConfig
from ..models import (
    CommandPlan,
    CommandRequest,
    CommandSource,
    ResolutionDecisionV1,
    ResolutionStatusV1,
    SegmentationDecisionV1,
    SegmentationResultV1,
)
from ..settings import Settings
from .assembler import AssemblyResult, assemble_plan, clarification_plan
from .calibration import (
    RoutingCalibrationInference,
    RoutingCalibrationRuntime,
    RoutingCalibrationRuntimeStatus,
    build_calibration_observations,
    build_calibration_recorder,
    build_challenge_observation,
)
from .resolver import (
    MINIMUM_VECTOR_COVERAGE,
    MISSING_SPACE_POLICY,
    RESOLVER_WEIGHTS,
    SINGLE_CANDIDATE_POLICY,
    resolve_subtasks,
)
from .segmenter import segment_command


@dataclass(frozen=True, slots=True)
class RoutingQualityGate:
    passed: bool
    reason: str
    path: str
    catalog_coverage: float = 0.0
    expected_action_count: int = 0
    runtime_fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "path": self.path,
            "catalog_coverage": self.catalog_coverage,
            "expected_action_count": self.expected_action_count,
            "runtime_fingerprint": self.runtime_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RoutingV2Outcome:
    segmentation: SegmentationResultV1
    searches: tuple[SubtaskCapabilitySearch, ...] = ()
    decisions: tuple[ResolutionDecisionV1, ...] = ()
    assembly: AssemblyResult = AssemblyResult(None, "not_assembled")
    blocked_reason: str | None = None

    @property
    def plan(self) -> CommandPlan | None:
        return self.assembly.plan


class RoutingV2Service:
    def __init__(
        self,
        settings: Settings,
        *,
        capability_index: CapabilityIndex,
        definitions: list[dict[str, Any]],
    ) -> None:
        self.settings = settings
        self.capability_index = capability_index
        self.definitions = [
            dict(definition) for definition in definitions if str(definition.get("id") or "")
        ]
        self._definitions_by_id = {
            str(definition["id"]): definition for definition in self.definitions
        }
        self.quality_gate = load_quality_gate(
            settings.routing_v2_quality_gate_path,
            expected_action_count=len(self.definitions),
            expected_catalog_hash=capability_index.catalog_hash,
        )
        self.calibration_runtime = RoutingCalibrationRuntime(settings)
        self.calibration_recorder = build_calibration_recorder(settings)

    async def start(self) -> None:
        if self.calibration_runtime.mode is not RoutingCalibrationMode.OFF:
            await self.calibration_runtime.preload()
        await self.calibration_recorder.start()

    async def close(self) -> None:
        await self.calibration_recorder.stop()

    @property
    def live_execution_requested(self) -> bool:
        return bool(
            self.settings.routing_v2_enabled
            and not self.settings.routing_v2_shadow_mode
            and self.settings.routing_v2_execute
        )

    @property
    def execution_enabled(self) -> bool:
        return bool(
            self.live_execution_requested
            and self.quality_gate.passed
            and self.runtime_fingerprint == self.quality_gate.runtime_fingerprint
        )

    @property
    def canary_enabled(self) -> bool:
        return bool(self.execution_enabled and self.settings.routing_v2_canary_enabled)

    @property
    def canary_action_ids(self) -> frozenset[str]:
        return frozenset(
            action_id.strip()
            for action_id in self.settings.routing_v2_canary_action_ids.split(",")
            if action_id.strip()
        )

    def plan_execution_allowed(self, plan: CommandPlan | None) -> bool:
        if not self.execution_enabled or plan is None or not plan.steps:
            return False
        if not self.canary_enabled:
            return True
        allowed_ids = self.canary_action_ids
        if not allowed_ids:
            return False
        for step in plan.steps:
            definition = self._definitions_by_id.get(step.action_id)
            if (
                step.action_id not in allowed_ids
                or definition is None
                or str(definition.get("risk") or "") != "low"
                or bool(definition.get("confirmation_required"))
                or step.confirmation_required
            ):
                return False
        return True

    @property
    def activation_block_reason(self) -> str | None:
        if not self.live_execution_requested:
            return None
        if not self.quality_gate.passed:
            return f"quality_gate:{self.quality_gate.reason}"
        if self.runtime_fingerprint != self.quality_gate.runtime_fingerprint:
            return "quality_gate:runtime_config_mismatch"
        return None

    def health(self) -> tuple[bool, str]:
        if not self.settings.routing_v2_enabled:
            return False, "wyłączony w konfiguracji"
        if self.canary_enabled:
            mode = "canary"
        else:
            mode = "execute" if self.execution_enabled else "shadow"
        if not self.quality_gate.passed:
            gate = self.quality_gate.reason
        elif self.runtime_fingerprint != self.quality_gate.runtime_fingerprint:
            gate = "runtime_config_mismatch"
        else:
            gate = "zaliczona"
        calibration_mode = self.calibration_runtime.mode.value
        return True, f"{mode}; quality gate: {gate}; calibration: {calibration_mode}"

    def activation_guard_plan(
        self,
        request: CommandRequest,
        outcome: RoutingV2Outcome,
    ) -> CommandPlan | None:
        reason = self.activation_block_reason
        if (
            reason is None
            or request.command_id
            or outcome.segmentation.decision is SegmentationDecisionV1.NON_COMMAND
        ):
            return None
        return clarification_plan(request, reason=reason)

    def runtime_config(self) -> dict[str, Any]:
        embeddings = getattr(self.capability_index, "embeddings", None)
        embedding_model = (
            getattr(embeddings, "_resolved_model", None)
            or getattr(embeddings, "configured_model", None)
            or ""
        )
        return {
            "schema_version": 1,
            "candidate_limit": max(
                2,
                min(int(self.settings.routing_v2_candidate_limit), 20),
            ),
            "execute_min_score": max(
                0.0,
                min(float(self.settings.routing_v2_execute_min_score), 1.0),
            ),
            "execute_min_margin": max(
                0.0,
                min(float(self.settings.routing_v2_execute_min_margin), 1.0),
            ),
            "stt_threshold": max(
                0.0,
                min(float(self.settings.stt_min_action_confidence), 1.0),
            ),
            "max_subtasks": max(
                1,
                min(int(self.settings.routing_v2_max_subtasks), 100),
            ),
            "embedding_model": embedding_model,
            "embedding_dimension": int(getattr(self.capability_index, "_dimension", 0) or 0),
            "capability_collection": str(getattr(self.capability_index, "collection_name", "")),
            "catalog_hash": self.capability_index.catalog_hash,
            "vector_distance": "cosine",
            "taxonomy_version": str(
                getattr(self.capability_index, "taxonomy_version", "routing-taxonomy-v2")
            ),
            "document_format_version": str(
                getattr(
                    self.capability_index,
                    "document_format_version",
                    "capability-document-v2:routing-taxonomy-v2",
                )
            ),
            "query_format_version": str(
                getattr(
                    self.capability_index,
                    "query_format_version",
                    "capability-query-v2:routing-taxonomy-v2",
                )
            ),
            "vector_fusion": str(
                getattr(self.capability_index, "vector_fusion", "weighted-rrf-v1")
            ),
            "rank_fusion_k": int(
                getattr(self.capability_index, "rank_fusion_k", 60)
            ),
            "vector_weights": dict(
                getattr(
                    self.capability_index,
                    "vector_weights",
                    {
                        "semantic": 1.0,
                        "intent": 1.0,
                        "target_context": 1.0,
                    },
                )
            ),
            "resolver_weights": dict(RESOLVER_WEIGHTS),
            "missing_space_policy": MISSING_SPACE_POLICY,
            "minimum_vector_coverage": MINIMUM_VECTOR_COVERAGE,
            "single_candidate_policy": SINGLE_CANDIDATE_POLICY,
            "implementation_version": "routing-v2-runtime-7",
        }

    @property
    def runtime_fingerprint(self) -> str:
        try:
            config = RoutingV2RuntimeConfig.model_validate(self.runtime_config())
        except ValueError:
            return ""
        return config.fingerprint()

    def unavailable_outcome(
        self,
        request: CommandRequest,
        *,
        reason: str,
    ) -> RoutingV2Outcome | None:
        if request.command_id:
            return None
        segmentation = segment_command(
            (request.text or "").strip(),
            max_subtasks=self.settings.routing_v2_max_subtasks,
        )
        if segmentation.decision is SegmentationDecisionV1.NON_COMMAND:
            return None
        blocked_reason = f"routing_unavailable:{reason}"
        return RoutingV2Outcome(
            segmentation=segmentation,
            assembly=AssemblyResult(
                clarification_plan(request, reason=blocked_reason),
                blocked_reason,
            ),
            blocked_reason=blocked_reason,
        )

    async def evaluate(self, request: CommandRequest) -> RoutingV2Outcome:
        if request.command_id:
            segmentation = SegmentationResultV1(
                decision=SegmentationDecisionV1.NON_COMMAND,
                confidence=1.0,
                reason="explicit_command_id_fast_path",
            )
            return RoutingV2Outcome(
                segmentation=segmentation,
                blocked_reason="explicit_command_id_fast_path",
            )

        voice_gate_reason = self._voice_gate_reason(request)
        if voice_gate_reason is not None:
            segmentation = SegmentationResultV1(
                decision=SegmentationDecisionV1.AMBIGUOUS,
                confidence=0.0,
                reason=voice_gate_reason,
            )
            outcome = RoutingV2Outcome(
                segmentation=segmentation,
                assembly=AssemblyResult(
                    clarification_plan(request, reason=voice_gate_reason),
                    voice_gate_reason,
                ),
                blocked_reason=voice_gate_reason,
            )
            self._record_challenge_observation(request, reason=voice_gate_reason)
            return outcome

        text = (request.text or "").strip()
        segmentation = segment_command(
            text,
            max_subtasks=self.settings.routing_v2_max_subtasks,
        )
        if segmentation.decision is SegmentationDecisionV1.NON_COMMAND:
            outcome = RoutingV2Outcome(
                segmentation=segmentation,
                blocked_reason=segmentation.reason or "non_command",
            )
            self._record_challenge_observation(
                request,
                reason=segmentation.reason or "non_command",
            )
            return outcome
        if segmentation.decision is SegmentationDecisionV1.AMBIGUOUS:
            reason = segmentation.reason or "ambiguous_segmentation"
            outcome = RoutingV2Outcome(
                segmentation=segmentation,
                assembly=AssemblyResult(
                    clarification_plan(request, reason=reason),
                    reason,
                ),
                blocked_reason=reason,
            )
            self._record_challenge_observation(request, reason=reason)
            return outcome

        searches = tuple(
            await self.capability_index.search_subtasks(
                segmentation.subtasks,
                limit=max(
                    2,
                    min(int(self.settings.routing_v2_candidate_limit), 20),
                ),
                min_score=-1.0,
            )
        )
        confidence = request.effective_transcript_confidence
        if request.source is CommandSource.DEEPGRAM and confidence is None:
            confidence = 0.0
        decisions = resolve_subtasks(
            searches,
            definitions=self.definitions,
            transcript_confidence=confidence,
            min_score=max(
                0.0,
                min(float(self.settings.routing_v2_execute_min_score), 1.0),
            ),
            min_margin=max(
                0.0,
                min(float(self.settings.routing_v2_execute_min_margin), 1.0),
            ),
            stt_threshold=max(
                0.0,
                min(float(self.settings.stt_min_action_confidence), 1.0),
            ),
        )
        unresolved = next(
            (
                decision
                for decision in decisions
                if decision.decision is not ResolutionStatusV1.RESOLVED
            ),
            None,
        )
        if unresolved is not None:
            reason = f"{unresolved.decision.value}:{unresolved.reason or 'unresolved_subtask'}"
            outcome = RoutingV2Outcome(
                segmentation=segmentation,
                searches=searches,
                decisions=decisions,
                assembly=AssemblyResult(
                    clarification_plan(request, reason=reason),
                    reason,
                ),
                blocked_reason=reason,
            )
            self._record_calibration_observations(request, outcome)
            return outcome

        assembly = assemble_plan(
            request,
            segmentation,
            decisions,
            definitions=self.definitions,
            max_steps=self.settings.routing_v2_max_subtasks,
        )
        if assembly.plan is None and assembly.blocked_reason:
            assembly = AssemblyResult(
                clarification_plan(request, reason=assembly.blocked_reason),
                assembly.blocked_reason,
            )
        outcome = RoutingV2Outcome(
            segmentation=segmentation,
            searches=searches,
            decisions=decisions,
            assembly=assembly,
            blocked_reason=assembly.blocked_reason,
        )
        self._record_calibration_observations(request, outcome)
        return outcome

    def _voice_gate_reason(self, request: CommandRequest) -> str | None:
        if request.source is not CommandSource.DEEPGRAM:
            return None
        if (
            self.settings.conversation_ignore_multi_speaker
            and request.transcript is not None
            and len(set(request.transcript.speaker_ids)) > 1
        ):
            return "multiple_speakers"
        confidence = request.effective_transcript_confidence
        if confidence is None:
            return "missing_stt_confidence"
        threshold = max(
            0.0,
            min(float(self.settings.stt_min_action_confidence), 1.0),
        )
        if confidence < threshold:
            return "low_stt_confidence"
        return None

    def shadow_payload(
        self,
        request: CommandRequest,
        outcome: RoutingV2Outcome,
        *,
        legacy_plan: CommandPlan | None,
    ) -> dict[str, Any]:
        try:
            calibration = self._infer_calibration(outcome.decisions)
        except Exception:
            calibration = RoutingCalibrationInference(
                mode=self.calibration_runtime.mode,
                status=RoutingCalibrationRuntimeStatus.MALFORMED_ARTIFACT,
                artifact_fingerprint=None,
                p_action_correct=tuple(None for _ in outcome.decisions),
            )
        calibration_payload = calibration.as_dict()
        calibration_payload["recorder"] = {
            "accepted_count": self.calibration_recorder.accepted_count,
            "dropped_count": self.calibration_recorder.dropped_count,
            "failed_count": self.calibration_recorder.failed_count,
        }
        v1_signature = _plan_signature(legacy_plan)
        v2_signature = _plan_signature(outcome.plan)
        return {
            "schema_version": 1,
            "request_id": request.request_id,
            "mode": (
                "canary"
                if self.canary_enabled
                else ("execute" if self.execution_enabled else "shadow")
            ),
            "execution_requested": bool(self.settings.routing_v2_execute),
            "execution_enabled": self.execution_enabled,
            "canary_enabled": self.canary_enabled,
            "canary_action_ids": sorted(self.canary_action_ids),
            "canary_plan_allowed": self.plan_execution_allowed(outcome.plan),
            "quality_gate": self.quality_gate.as_dict(),
            "runtime_config": self.runtime_config(),
            "runtime_fingerprint": self.runtime_fingerprint,
            "segmentation": outcome.segmentation.model_dump(mode="json"),
            "embeddings": [
                {
                    "schema_version": search.embedding.schema_version,
                    "subtask_id": search.embedding.subtask_id,
                    "embedding_model": search.embedding.embedding_model,
                    "dimension": search.embedding.dimension,
                    "format_version": search.embedding.format_version,
                    "normalized_text_sha256": (search.embedding.normalized_text_sha256),
                    "vector_names": [
                        "semantic",
                        "intent",
                        "target_context",
                    ],
                }
                for search in outcome.searches
            ],
            "decisions": [decision.model_dump(mode="json") for decision in outcome.decisions],
            "blocked_reason": outcome.blocked_reason,
            "v1_action_ids": _action_ids(legacy_plan),
            "v2_action_ids": _action_ids(outcome.plan),
            "v1_signature": v1_signature,
            "v2_signature": v2_signature,
            "plans_match": bool(
                v1_signature is not None
                and v2_signature is not None
                and v1_signature == v2_signature
            ),
            "both_abstained": v1_signature is None and v2_signature is None,
            "calibration": calibration_payload,
        }

    def _record_calibration_observations(
        self,
        request: CommandRequest,
        outcome: RoutingV2Outcome,
    ) -> None:
        if not outcome.decisions:
            return
        try:
            calibration = self._infer_calibration(outcome.decisions)
            observations = build_calibration_observations(
                request=request,
                decisions=outcome.decisions,
                subtask_count=len(outcome.segmentation.subtasks),
                runtime_fingerprint=self.runtime_fingerprint or ("0" * 64),
                catalog_hash=self.capability_index.catalog_hash,
                inference=calibration,
            )
            self.calibration_recorder.record(observations)
        except Exception:
            # Recorder działa best-effort i nie może wpływać na routing.
            return

    def _record_challenge_observation(
        self,
        request: CommandRequest,
        *,
        reason: str,
    ) -> None:
        try:
            observation = build_challenge_observation(
                request=request,
                reason=reason,
                decision="clarify",
                runtime_fingerprint=self.runtime_fingerprint or ("0" * 64),
                catalog_hash=self.capability_index.catalog_hash,
            )
            self.calibration_recorder.record((observation,))
        except Exception:
            return

    def _infer_calibration(
        self,
        decisions: tuple[ResolutionDecisionV1, ...],
    ) -> RoutingCalibrationInference:
        return self.calibration_runtime.infer(
            decisions,
            expected_runtime_fingerprint=self.runtime_fingerprint,
            expected_catalog_hash=self.capability_index.catalog_hash,
        )


def load_quality_gate(
    path: Path,
    *,
    expected_action_count: int,
    expected_catalog_hash: str,
) -> RoutingQualityGate:
    safe_path = str(path)
    if not path.is_file():
        return RoutingQualityGate(False, "missing_report", safe_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RoutingQualityGate(False, "invalid_report", safe_path)
    if not isinstance(payload, dict):
        return RoutingQualityGate(False, "invalid_report", safe_path)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        payload = metrics
    passed = payload.get("quality_gate_passed") is True
    schema_version = _safe_int(payload.get("schema_version"))
    coverage = _safe_float(payload.get("catalog_coverage"))
    sample_count = _safe_int(payload.get("sample_count"))
    base_example_count = _safe_int(payload.get("base_example_count"))
    reported_action_count = _safe_int(payload.get("action_count"))
    reported_expected_action_count = _safe_int(payload.get("expected_action_count"))
    resolved_accuracy = _safe_float(payload.get("resolved_accuracy"))
    topk_recall = _safe_float(payload.get("topk_recall"))
    safe_abstention_recall = _safe_float(payload.get("safe_abstention_recall"))
    unsafe_resolution_count = _safe_int(payload.get("unsafe_resolution_count"))
    runtime_fingerprint = str(payload.get("runtime_fingerprint") or "")
    runtime_config_payload = payload.get("runtime_config")
    try:
        runtime_config = RoutingV2RuntimeConfig.model_validate(runtime_config_payload)
    except ValueError:
        runtime_config = None
    failures = payload.get("quality_gate_failures")
    catalog_hash = str(payload.get("catalog_hash") or "")
    if schema_version != 2:
        reason = "unsupported_report_schema"
    elif not passed:
        reason = "quality_gate_failed"
    elif coverage < 1.0:
        reason = "catalog_coverage_incomplete"
    elif reported_action_count != expected_action_count:
        reason = "catalog_action_count_mismatch"
    elif reported_expected_action_count != expected_action_count:
        reason = "expected_action_count_mismatch"
    elif not catalog_hash:
        reason = "catalog_hash_missing"
    elif catalog_hash != expected_catalog_hash:
        reason = "catalog_hash_mismatch"
    elif runtime_config is None:
        reason = "runtime_config_invalid"
    elif runtime_config.catalog_hash != catalog_hash:
        reason = "runtime_config_catalog_mismatch"
    elif runtime_config.fingerprint() != runtime_fingerprint:
        reason = "runtime_fingerprint_invalid"
    elif sample_count < base_example_count or base_example_count < expected_action_count * 2:
        reason = "evaluation_sample_incomplete"
    elif resolved_accuracy < 0.80:
        reason = "resolved_accuracy_below_0_80"
    elif topk_recall < 0.95:
        reason = "topk_below_0_95"
    elif safe_abstention_recall < 0.95:
        reason = "safe_abstention_below_0_95"
    elif unsafe_resolution_count != 0:
        reason = "unsafe_resolution_detected"
    elif not isinstance(failures, list | tuple) or failures:
        reason = "quality_gate_failures_present"
    else:
        reason = "passed"
    return RoutingQualityGate(
        passed=reason == "passed",
        reason=reason,
        path=safe_path,
        catalog_coverage=coverage,
        expected_action_count=reported_expected_action_count,
        runtime_fingerprint=runtime_fingerprint,
    )


def _plan_signature(plan: CommandPlan | None) -> str | None:
    if plan is None or not plan.steps:
        return None
    payload = [
        {
            "action_id": step.action_id,
            "args": step.args,
        }
        for step in plan.steps
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action_ids(plan: CommandPlan | None) -> list[str]:
    return [step.action_id for step in plan.steps] if plan is not None else []


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
