from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import (
    CommandRequest,
    ResolutionStatusV1,
    TranscriptEnvelopeV1,
)
from .analysis import (
    aggregate_routing_metrics,
    build_routing_eval_records,
    build_style_profile,
    character_error_rate,
    evaluate_style_profile,
    score_routing_result,
    word_error_rate,
)
from .candidates import MemoryCandidateStore, extract_memory_candidates
from .dedupe import assign_splits, deduplicate
from .extract import extract_from_manifest
from .inventory import build_manifest
from .privacy import apply_privacy_gate
from .schema import (
    CorpusRunReport,
    ExpectedIntent,
    QuarantineRecord,
    RoutingEvalRecord,
    RoutingMetrics,
    RoutingScoreCard,
    RoutingV2Metrics,
    RoutingV2RuntimeConfig,
    RoutingV2ScoreCard,
    SourceManifest,
    SttErrorType,
    StyleProfile,
    UtteranceRecord,
)
from .storage import sha256_text, write_json, write_jsonl


@dataclass(frozen=True)
class CorpusPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifests" / "sources-v1.json"

    @property
    def speaker_decisions(self) -> Path:
        return self.root / "manifests" / "speaker-decisions-v1.json"

    @property
    def clean(self) -> Path:
        return self.root / "derived" / "utterances-clean-v1.jsonl"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine" / "metadata-v1.jsonl"

    @property
    def eval_set(self) -> Path:
        return self.root / "eval" / "routing-v1.jsonl"

    @property
    def stt_metrics(self) -> Path:
        return self.root / "eval" / "stt-metrics-v1.json"

    @property
    def routing_scores(self) -> Path:
        return self.root / "eval" / "routing-scores-v1.jsonl"

    @property
    def routing_metrics(self) -> Path:
        return self.root / "eval" / "routing-metrics-v1.json"

    @property
    def routing_v2_scores(self) -> Path:
        return self.root / "eval" / "routing-v2-scores.jsonl"

    @property
    def routing_v2_metrics(self) -> Path:
        return self.root / "eval" / "routing-v2-metrics.json"

    @property
    def routing_personal_holdout(self) -> Path:
        return self.root / "eval" / "routing-personal-holdout-v1.jsonl"

    @property
    def routing_personal_scores(self) -> Path:
        return self.root / "eval" / "routing-personal-scores-v1.jsonl"

    @property
    def routing_personal_metrics(self) -> Path:
        return self.root / "eval" / "routing-personal-metrics-v1.json"

    @property
    def routing_calibration_observations(self) -> Path:
        return self.root / "routing_calibration" / "observations-v1.db"

    @property
    def routing_calibration_dataset(self) -> Path:
        return self.root / "eval" / "routing-calibration-v1.jsonl"

    @property
    def routing_calibration_artifact(self) -> Path:
        return self.root / "eval" / "routing-calibration-artifact-v1.json"

    @property
    def routing_calibration_report(self) -> Path:
        return self.root / "eval" / "routing-calibration-report-v1.json"

    @property
    def memory_retrieval_eval(self) -> Path:
        return self.root / "eval" / "memory-retrieval-v1.jsonl"

    @property
    def memory_retrieval_scores(self) -> Path:
        return self.root / "eval" / "memory-retrieval-scores-v1.jsonl"

    @property
    def memory_retrieval_metrics(self) -> Path:
        return self.root / "eval" / "memory-retrieval-metrics-v1.json"

    @property
    def voice_root(self) -> Path:
        return self.root / "eval" / "voice-v1"

    @property
    def voice_manifest(self) -> Path:
        return self.root / "manifests" / "voice-sources-v1.json"

    @property
    def voice_candidates(self) -> Path:
        return self.voice_root / "candidates-v1.jsonl"

    @property
    def voice_samples(self) -> Path:
        return self.voice_root / "samples-v1.jsonl"

    @property
    def voice_annotations(self) -> Path:
        return self.voice_root / "annotations-v1.jsonl"

    @property
    def voice_annotations_development(self) -> Path:
        return self.voice_root / "annotations-development-v1.jsonl"

    @property
    def voice_annotations_holdout(self) -> Path:
        return self.voice_root / "annotations-holdout-v1.jsonl"

    @property
    def voice_annotation_template(self) -> Path:
        return self.voice_root / "annotation-template-v1.jsonl"

    @property
    def voice_annotation_template_development(self) -> Path:
        return self.voice_root / "annotation-template-development-v1.jsonl"

    @property
    def voice_annotation_template_holdout(self) -> Path:
        return self.voice_root / "annotation-template-holdout-v1.jsonl"

    @property
    def voice_annotation_review(self) -> Path:
        return self.voice_root / "review-v1.html"

    @property
    def voice_annotation_review_development(self) -> Path:
        return self.voice_root / "review-development-v1.html"

    @property
    def voice_annotation_review_holdout(self) -> Path:
        return self.voice_root / "review-holdout-v1.html"

    @property
    def voice_splits(self) -> Path:
        return self.voice_root / "splits-v1.json"

    @property
    def voice_validation(self) -> Path:
        return self.voice_root / "validation-v1.json"

    @property
    def voice_cache(self) -> Path:
        return self.voice_root / "cache" / "deepgram"

    @property
    def voice_transcripts(self) -> Path:
        return self.voice_root / "transcripts-v1.jsonl"

    @property
    def voice_runs(self) -> Path:
        return self.voice_root / "runs"

    @property
    def proper_names(self) -> Path:
        return self.root / "lexicon" / "proper-names-v1.json"

    @property
    def action_reliability_json(self) -> Path:
        return self.root / "reports" / "action-reliability-v1.json"

    @property
    def action_reliability_markdown(self) -> Path:
        return self.root / "reports" / "action-reliability-v1.md"

    @property
    def journal_candidates(self) -> Path:
        return self.root / "journal" / "candidates-v1.jsonl"

    @property
    def journal_entries(self) -> Path:
        return self.root / "journal" / "entries-v1.jsonl"

    @property
    def style_profile(self) -> Path:
        return self.root / "style" / "profile-v1.json"

    @property
    def style_holdout_report(self) -> Path:
        return self.root / "style" / "holdout-report-v1.json"

    @property
    def candidates_database(self) -> Path:
        return self.root / "memory_candidates" / "candidates.db"

    @property
    def report(self) -> Path:
        return self.root / "manifests" / "last-run-v1.json"


async def run_pipeline(
    *,
    paths: CorpusPaths,
    audio_transcript: Path | None,
    cursor_projects_root: Path | None,
    capability_definitions: list[dict],
    style_enabled: bool = False,
    holdout_percent: int = 20,
    trusted_audio_source_ids: set[str] | None = None,
) -> CorpusRunReport:
    manifest = build_manifest(
        audio_transcript=audio_transcript,
        cursor_projects_root=cursor_projects_root,
    )
    write_json(paths.manifest, manifest)
    extraction = extract_from_manifest(
        manifest,
        trusted_audio_source_ids=trusted_audio_source_ids,
    )
    deduplicated = deduplicate(extraction.records)
    duplicate_count = sum(record.is_near_duplicate for record in deduplicated)

    clean: list[UtteranceRecord] = []
    quarantine: list[QuarantineRecord] = []
    for record in deduplicated:
        if record.is_near_duplicate:
            continue
        decision = apply_privacy_gate(record)
        if decision.clean is not None:
            clean.append(decision.clean)
        if decision.quarantine is not None:
            quarantine.append(decision.quarantine)
    clean = assign_splits(clean, holdout_percent=holdout_percent)
    write_jsonl(paths.clean, clean)
    write_jsonl(paths.quarantine, quarantine)

    eval_records = build_routing_eval_records(capability_definitions)
    write_jsonl(paths.eval_set, eval_records)
    write_json(paths.stt_metrics, _stt_metrics(eval_records))

    style_profile = build_style_profile(
        clean,
        manifest_id=manifest.manifest_id,
        enabled=style_enabled,
    )
    write_json(paths.style_profile, style_profile)
    write_json(
        paths.style_holdout_report,
        evaluate_style_profile(style_profile, clean),
    )

    candidate_store = MemoryCandidateStore(paths.candidates_database)
    await candidate_store.initialize()
    candidate_scope_id = corpus_scope_id(
        manifest.manifest_id,
        trusted_audio_source_ids or set(),
    )
    await candidate_store.set_active_scope(candidate_scope_id)
    for candidate in extract_memory_candidates(
        clean,
        manifest_id=candidate_scope_id,
    ):
        await candidate_store.upsert(candidate)

    report = CorpusRunReport(
        manifest_id=manifest.manifest_id,
        extracted_count=len(extraction.records),
        clean_count=len(clean),
        quarantined_count=len(quarantine),
        duplicate_count=duplicate_count,
        clean_word_count=sum(record.word_count for record in clean),
        technical_excluded_count=extraction.technical_excluded_count,
    )
    write_json(paths.report, report)
    return report


async def evaluate_routing(
    *,
    records: list[RoutingEvalRecord],
    search,
    min_score: float,
    margin_threshold: float,
    stt_threshold: float,
    expected_action_ids: set[str] | None = None,
) -> tuple[list[RoutingScoreCard], RoutingMetrics]:
    cards: list[RoutingScoreCard] = []
    for record in records:
        result = await search(record.stt_text)
        matches = [(str(match.action_id), float(match.score)) for match in result.matches]
        cards.append(
            score_routing_result(
                record,
                matches,
                min_score=min_score,
                margin_threshold=margin_threshold,
                stt_threshold=stt_threshold,
            )
        )
    return cards, aggregate_routing_metrics(
        cards,
        expected_action_ids=expected_action_ids,
    )


async def evaluate_routing_v2(
    *,
    records: list[RoutingEvalRecord],
    route,
    stt_threshold: float,
    expected_action_ids: set[str],
    catalog_hash: str,
    runtime_config: dict[str, Any] | Callable[[], dict[str, Any]],
) -> tuple[list[RoutingV2ScoreCard], RoutingV2Metrics]:
    cards: list[RoutingV2ScoreCard] = []
    for record in records:
        request = CommandRequest.from_transcript(
            TranscriptEnvelopeV1.from_text(
                record.stt_text,
                confidence=record.stt_confidence,
                speaker_ids=record.speaker_ids,
            )
        )
        outcome = await route(request)
        predicted_action_ids = (
            tuple(step.action_id for step in outcome.plan.steps) if outcome.plan is not None else ()
        )
        predicted_step_args = (
            tuple(dict(step.args) for step in outcome.plan.steps)
            if outcome.plan is not None
            else ()
        )
        expected_plan_action_ids = record.plan_action_ids
        arguments_match = bool(
            not record.expected_step_args or predicted_step_args == record.expected_step_args
        )
        exact_plan_match = bool(
            expected_plan_action_ids
            and predicted_action_ids == expected_plan_action_ids
            and arguments_match
        )
        first_decision = outcome.decisions[0] if outcome.decisions else None
        topk_action_ids = (
            tuple(candidate.action_id for candidate in first_decision.candidates)
            if first_decision is not None
            else ()
        )
        top_candidate = (
            first_decision.candidates[0]
            if first_decision is not None and first_decision.candidates
            else None
        )
        expected_abstention = bool(
            record.ambiguous
            or record.expected_intent
            in {ExpectedIntent.AMBIGUOUS, ExpectedIntent.CONVERSATION}
            or len(record.speaker_ids) > 1
            or record.error_type is SttErrorType.LOW_CONFIDENCE
            or record.stt_confidence < stt_threshold
        )
        abstained = not predicted_action_ids
        expected_action_id = record.expected_action_id
        hit_at_1 = exact_plan_match
        hit_at_k = bool(
            expected_plan_action_ids
            and len(outcome.decisions) == len(expected_plan_action_ids)
            and all(
                expected_id
                in {candidate.action_id for candidate in outcome.decisions[index].candidates}
                for index, expected_id in enumerate(expected_plan_action_ids)
            )
        )
        if first_decision is not None:
            decision = first_decision.decision.value
            reason = first_decision.reason
        elif outcome.segmentation.decision.value == "ambiguous":
            decision = ResolutionStatusV1.CLARIFY.value
            reason = outcome.blocked_reason
        else:
            decision = ResolutionStatusV1.UNSUPPORTED.value
            reason = outcome.blocked_reason
        cards.append(
            RoutingV2ScoreCard(
                example_id=record.example_id,
                base_example_id=record.base_example_id,
                expected_action_id=expected_action_id,
                expected_action_ids=expected_plan_action_ids,
                expected_step_args=record.expected_step_args,
                predicted_action_ids=predicted_action_ids,
                predicted_step_args=predicted_step_args,
                topk_action_ids=topk_action_ids,
                top1_action_id=(
                    getattr(first_decision, "top1_action_id", None)
                    if first_decision is not None
                    else None
                ),
                top1_combined_score=(
                    getattr(top_candidate, "combined_score", None)
                    if top_candidate is not None
                    else None
                ),
                margin_top2=(
                    getattr(first_decision, "margin_top2", None)
                    if first_decision is not None
                    else None
                ),
                decision=decision,
                reason=reason,
                expected_abstention=expected_abstention,
                abstained=abstained,
                arguments_match=arguments_match,
                exact_plan_match=exact_plan_match,
                hit_at_1=hit_at_1,
                hit_at_k=hit_at_k,
                unsafe_resolution=bool(
                    (expected_abstention and not abstained)
                    or (not expected_abstention and not abstained and not exact_plan_match)
                ),
            )
        )

    expected_actions = set(expected_action_ids)
    bases_by_action: dict[str, set[str]] = {}
    for card in cards:
        if card.expected_action_id is None:
            continue
        bases_by_action.setdefault(card.expected_action_id, set()).add(
            card.base_example_id or card.example_id
        )
    covered_actions = set(bases_by_action) & expected_actions
    eligible = [card for card in cards if not card.expected_abstention]
    expected_abstentions = [card for card in cards if card.expected_abstention]
    resolved_accuracy = _metric_ratio(
        sum(card.hit_at_1 for card in eligible),
        len(eligible),
    )
    topk_recall = _metric_ratio(
        sum(card.hit_at_k for card in eligible),
        len(eligible),
    )
    reciprocal_ranks = []
    for card in eligible:
        if card.expected_action_id is None:
            continue
        try:
            rank = card.topk_action_ids.index(card.expected_action_id) + 1
        except ValueError:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / rank)
    mean_reciprocal_rank = _metric_ratio(
        sum(reciprocal_ranks),
        len(reciprocal_ranks),
    )
    margins = [card.margin_top2 for card in cards if card.margin_top2 is not None]
    mean_margin_top2 = _metric_ratio(sum(margins), len(margins))
    calibration_pairs = [
        (card.top1_combined_score, card.hit_at_1)
        for card in eligible
        if card.top1_combined_score is not None
    ]
    expected_calibration_error = _expected_calibration_error(calibration_pairs)
    safe_abstention_recall = _metric_ratio(
        sum(card.abstained for card in expected_abstentions),
        len(expected_abstentions),
        empty_value=1.0,
    )
    unsafe_resolution_count = sum(card.unsafe_resolution for card in cards)
    failures: list[str] = []
    if covered_actions != expected_actions:
        failures.append("catalog_action_coverage_incomplete")
    if any(len(bases_by_action.get(action_id, set())) < 2 for action_id in expected_actions):
        failures.append("per_action_coverage_below_2")
    if resolved_accuracy < 0.80:
        failures.append("resolved_accuracy_below_0_80")
    if topk_recall < 0.95:
        failures.append("topk_below_0_95")
    if safe_abstention_recall < 0.95:
        failures.append("safe_abstention_below_0_95")
    if unsafe_resolution_count:
        failures.append("unsafe_resolution_detected")
    runtime_payload = runtime_config() if callable(runtime_config) else runtime_config
    checked_runtime_config = RoutingV2RuntimeConfig.model_validate(runtime_payload)
    metrics = RoutingV2Metrics(
        catalog_hash=catalog_hash,
        runtime_config=checked_runtime_config,
        runtime_fingerprint=checked_runtime_config.fingerprint(),
        sample_count=len(cards),
        base_example_count=len(
            {base_id for base_ids in bases_by_action.values() for base_id in base_ids}
        ),
        action_count=len(bases_by_action),
        expected_action_count=len(expected_actions),
        catalog_coverage=_metric_ratio(
            len(covered_actions),
            len(expected_actions),
        ),
        resolved_accuracy=resolved_accuracy,
        topk_recall=topk_recall,
        mean_reciprocal_rank=mean_reciprocal_rank,
        mean_margin_top2=mean_margin_top2,
        expected_calibration_error=expected_calibration_error,
        safe_abstention_recall=safe_abstention_recall,
        unsafe_resolution_count=unsafe_resolution_count,
        quality_gate_passed=not failures,
        quality_gate_failures=tuple(failures),
    )
    return cards, metrics


def _metric_ratio(
    numerator: int,
    denominator: int,
    *,
    empty_value: float = 0.0,
) -> float:
    return numerator / denominator if denominator else empty_value


def _expected_calibration_error(
    pairs: list[tuple[float, bool]],
    *,
    bins: int = 10,
) -> float:
    if not pairs:
        return 0.0
    safe_bins = max(1, min(int(bins), 100))
    total = len(pairs)
    error = 0.0
    for index in range(safe_bins):
        lower = index / safe_bins
        upper = (index + 1) / safe_bins
        bucket = [
            (float(confidence), bool(correct))
            for confidence, correct in pairs
            if lower <= float(confidence) < upper
            or (index == safe_bins - 1 and float(confidence) == 1.0)
        ]
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(1.0 if item[1] else 0.0 for item in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - mean_confidence)
    return min(max(error, 0.0), 1.0)


def _stt_metrics(records: list[RoutingEvalRecord]) -> dict[str, float | int]:
    count = len(records)
    if not count:
        return {
            "schema_version": 1,
            "sample_count": 0,
            "mean_wer": 0.0,
            "mean_cer": 0.0,
        }
    return {
        "schema_version": 1,
        "sample_count": count,
        "mean_wer": sum(word_error_rate(record.gold_text, record.stt_text) for record in records)
        / count,
        "mean_cer": sum(
            character_error_rate(record.gold_text, record.stt_text) for record in records
        )
        / count,
    }


def load_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_style_profile(path: Path) -> StyleProfile:
    return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))


def corpus_scope_id(
    manifest_id: str,
    trusted_audio_source_ids: set[str],
) -> str:
    return sha256_text("|".join((manifest_id, *sorted(trusted_audio_source_ids))))[:20]
