from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..models import CommandRequest, TranscriptEnvelopeV1
from ..router import normalize_text
from ..screenpipe_deepgram import DeepgramFileTranscriber
from .analysis import character_error_rate, word_error_rate
from .prosody import analyze_prosody, question_intonation_score
from .schema import (
    ExpectedVoiceOutcome,
    VoiceEvalMetricsV1,
    VoiceEvalPredictionV1,
    VoiceEvalSampleV1,
    VoiceEvalSplit,
    VoiceGoldAnnotationV1,
    VoiceIntentLabel,
)
from .storage import sha256_text, write_json
from .voice_eval import VoiceEvalError

RouteCallback = Callable[[CommandRequest], Awaitable[Any]]

_QUESTION_WORDS = {
    "czy",
    "co",
    "kto",
    "gdzie",
    "jak",
    "kiedy",
    "dlaczego",
    "ile",
    "który",
    "ktory",
}
_TASK_WORDS = {
    "otwórz",
    "otworz",
    "zamknij",
    "włącz",
    "wlacz",
    "wyłącz",
    "wylacz",
    "skopiuj",
    "zapisz",
    "utwórz",
    "utworz",
    "wyślij",
    "wyslij",
    "przypomnij",
    "pokaż",
    "pokaz",
}
_CANCEL_WORDS = {"anuluj", "przerwij", "stop", "nieważne", "niewazne"}


class VoiceNoSpeechError(VoiceEvalError):
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        super().__init__(f"Deepgram nie wykrył mowy w próbce {sample_id}.")


class DeepgramReplayCache:
    def __init__(
        self,
        cache_root: Path,
        *,
        transcriber: DeepgramFileTranscriber,
    ) -> None:
        self.cache_root = cache_root
        self.transcriber = transcriber

    async def envelope_for(
        self,
        sample: VoiceEvalSampleV1,
        *,
        eval_root: Path,
        allow_remote: bool = False,
        confirmation: str = "",
    ) -> TranscriptEnvelopeV1:
        if sample.audio is None:
            raise VoiceEvalError(f"Próbka {sample.sample_id} nie ma audio.")
        key = sha256_text(
            "|".join(
                (
                    sample.audio.clip_sha256,
                    self.transcriber.model,
                    self.transcriber.language,
                    "smart_format=true",
                    "punctuate=true",
                    "diarize=true",
                    "utterances=true",
                )
            )
        )
        cache_path = self.cache_root / f"{key}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("status") == "no_speech":
                raise VoiceNoSpeechError(sample.sample_id)
            return TranscriptEnvelopeV1.model_validate(cached)
        if not allow_remote or confirmation != "DEEPGRAM_AUDIO_UPLOAD":
            raise VoiceEvalError(
                f"Brak cache Deepgram dla {sample.sample_id}; wysyłka wymaga "
                "--allow-remote --confirm DEEPGRAM_AUDIO_UPLOAD."
            )
        audio_path = _safe_eval_path(eval_root, sample.audio.relative_path)
        envelope = await self.transcriber.transcribe_envelope(audio_path)
        if envelope is None:
            write_json(
                cache_path,
                {
                    "schema_version": 1,
                    "status": "no_speech",
                    "sample_id": sample.sample_id,
                },
            )
            raise VoiceNoSpeechError(sample.sample_id)
        write_json(cache_path, envelope)
        return envelope


async def evaluate_voice_dataset(
    *,
    samples: list[VoiceEvalSampleV1],
    annotations: list[VoiceGoldAnnotationV1],
    envelopes: dict[str, TranscriptEnvelopeV1],
    eval_root: Path,
    run_id: str,
    route: RouteCallback | None = None,
    required_sample_count: int = 120,
) -> tuple[list[VoiceEvalPredictionV1], VoiceEvalMetricsV1]:
    annotation_by_id = {item.sample_id: item for item in annotations}
    predictions: list[VoiceEvalPredictionV1] = []
    unsafe_count = 0
    exact_plan_hits = 0
    topk_plan_hits = 0
    routing_reciprocal_ranks: list[float] = []
    expected_execution_count = 0
    expected_abstention_count = 0
    safe_abstention_hits = 0
    wers: list[float] = []
    cers: list[float] = []
    punctuation_scores: list[float] = []
    question_mark_matches: list[bool] = []

    for sample in samples:
        annotation = annotation_by_id.get(sample.sample_id)
        envelope = envelopes.get(sample.sample_id)
        errors: list[str] = []
        if annotation is None:
            errors.append("missing_annotation")
        if envelope is None:
            errors.append("missing_transcript")
        transcript_text = envelope.raw_text if envelope is not None else ""
        word_total = len((annotation.literal_text if annotation else transcript_text).split())
        prosody = analyze_prosody(sample, eval_root=eval_root, word_count=word_total)
        textual_label, textual_score = classify_textual_intent(transcript_text)
        prosody_score = question_intonation_score(prosody)
        prosodic_label = (
            VoiceIntentLabel.QUESTION
            if prosody_score >= 0.55
            else VoiceIntentLabel.CONVERSATION
        )
        routing: dict[str, Any] = {}
        semantic_label = textual_label
        semantic_score = textual_score
        if route is not None and envelope is not None:
            try:
                outcome = await route(CommandRequest.from_transcript(envelope))
                routing = _routing_payload(outcome)
                if routing["predicted_action_ids"]:
                    semantic_label = VoiceIntentLabel.TASK
                    semantic_score = float(routing.get("top1_score") or 0.5)
                elif routing.get("decision") == "clarify":
                    semantic_label = VoiceIntentLabel.AMBIGUOUS
                    semantic_score = 0.6
            except Exception as exc:
                errors.append(f"routing:{type(exc).__name__}")
        fused_label, fused_score = fuse_intent_scores(
            textual_label=textual_label,
            textual_score=textual_score,
            prosodic_label=prosodic_label,
            prosodic_score=prosody_score,
            semantic_label=semantic_label,
            semantic_score=semantic_score,
        )

        if annotation is not None and envelope is not None:
            wers.append(word_error_rate(annotation.literal_text, transcript_text))
            cers.append(character_error_rate(annotation.literal_text, transcript_text))
            punctuation_scores.append(
                punctuation_f1(annotation.punctuated_text, transcript_text)
            )
            question_mark_matches.append(
                annotation.punctuated_text.rstrip().endswith("?")
                == transcript_text.rstrip().endswith("?")
            )
            predicted_actions = tuple(routing.get("predicted_action_ids") or ())
            predicted_args = tuple(routing.get("predicted_step_args") or ())
            if annotation.expected_outcome is ExpectedVoiceOutcome.EXECUTE:
                expected_execution_count += 1
                exact = (
                    predicted_actions == annotation.expected_action_ids
                    and (
                        not annotation.expected_step_args
                        or predicted_args == annotation.expected_step_args
                    )
                )
                exact_plan_hits += exact
                topk_by_subtask = tuple(
                    tuple(str(action_id) for action_id in candidates)
                    for candidates in routing.get("topk_action_ids_by_subtask", ())
                )
                topk_plan_hit = bool(
                    len(topk_by_subtask) == len(annotation.expected_action_ids)
                    and all(
                        expected_action_id in topk_by_subtask[index]
                        for index, expected_action_id in enumerate(
                            annotation.expected_action_ids
                        )
                    )
                )
                topk_plan_hits += topk_plan_hit
                for index, expected_action_id in enumerate(annotation.expected_action_ids):
                    candidates = (
                        topk_by_subtask[index]
                        if index < len(topk_by_subtask)
                        else ()
                    )
                    try:
                        rank = candidates.index(expected_action_id) + 1
                    except ValueError:
                        routing_reciprocal_ranks.append(0.0)
                    else:
                        routing_reciprocal_ranks.append(1.0 / rank)
                if predicted_actions and not exact:
                    unsafe_count += 1
            if annotation.expected_abstention:
                expected_abstention_count += 1
                if not predicted_actions:
                    safe_abstention_hits += 1
                else:
                    unsafe_count += 1

        predictions.append(
            VoiceEvalPredictionV1(
                run_id=run_id,
                sample_id=sample.sample_id,
                transcript_text=transcript_text,
                transcript_confidence=(
                    envelope.confidence_mean if envelope is not None else None
                ),
                transcript_words=(
                    tuple(word.model_dump(mode="json") for word in envelope.words)
                    if envelope is not None
                    else ()
                ),
                prosody=prosody,
                textual_label=textual_label,
                textual_score=textual_score,
                prosodic_label=prosodic_label,
                prosodic_score=prosody_score,
                semantic_label=semantic_label,
                semantic_score=semantic_score,
                fused_label=fused_label,
                fused_score=fused_score,
                routing=routing,
                errors=tuple(errors),
            )
        )

    annotated_pairs = [
        (prediction, annotation_by_id[prediction.sample_id])
        for prediction in predictions
        if prediction.sample_id in annotation_by_id
    ]
    textual_accuracy = _label_accuracy(
        annotated_pairs,
        lambda prediction: prediction.textual_label,
    )
    prosodic_accuracy = _label_accuracy(
        annotated_pairs,
        lambda prediction: prediction.prosodic_label,
    )
    semantic_accuracy = _label_accuracy(
        annotated_pairs,
        lambda prediction: prediction.semantic_label,
    )
    fused_accuracy = _label_accuracy(
        annotated_pairs,
        lambda prediction: prediction.fused_label,
    )
    intonation_questions = [
        prediction
        for prediction, annotation in annotated_pairs
        if annotation.intent is VoiceIntentLabel.QUESTION
        and "question_intonation" in annotation.prosody_tags
    ]
    intonation_hits = sum(
        prediction.prosodic_label is VoiceIntentLabel.QUESTION
        for prediction in intonation_questions
    )
    pairs_by_tag: dict[
        str,
        list[tuple[VoiceEvalPredictionV1, VoiceGoldAnnotationV1]],
    ] = {}
    for prediction, annotation in annotated_pairs:
        for tag in _annotation_tags(annotation):
            pairs_by_tag.setdefault(tag, []).append((prediction, annotation))
    fused_accuracy_by_tag = {
        tag: _label_accuracy(pairs, lambda prediction: prediction.fused_label)
        for tag, pairs in sorted(pairs_by_tag.items())
    }
    wer_by_tag = {
        tag: sum(
            word_error_rate(annotation.literal_text, prediction.transcript_text)
            for prediction, annotation in pairs
        )
        / len(pairs)
        for tag, pairs in sorted(pairs_by_tag.items())
    }
    failures: list[str] = []
    if len(samples) != required_sample_count:
        failures.append(
            f"sample_count_not_{required_sample_count}"
        )
    if len(annotations) != len(samples):
        failures.append("annotations_incomplete")
    routing_exact_plan_accuracy = (
        exact_plan_hits / expected_execution_count
        if expected_execution_count
        else 0.0
    )
    routing_topk_recall = (
        topk_plan_hits / expected_execution_count
        if expected_execution_count
        else 0.0
    )
    if expected_execution_count and routing_exact_plan_accuracy < 0.80:
        failures.append("routing_exact_plan_below_0_80")
    if expected_execution_count and routing_topk_recall < 0.95:
        failures.append("routing_topk_below_0_95")
    if unsafe_count:
        failures.append("unsafe_resolution_detected")
    metrics = VoiceEvalMetricsV1(
        run_id=run_id,
        sample_count=len(samples),
        development_count=sum(
            sample.split is VoiceEvalSplit.DEVELOPMENT for sample in samples
        ),
        holdout_count=sum(sample.split is VoiceEvalSplit.HOLDOUT for sample in samples),
        annotated_count=len(annotations),
        mean_wer=sum(wers) / len(wers) if wers else 0.0,
        mean_cer=sum(cers) / len(cers) if cers else 0.0,
        mean_punctuation_f1=(
            sum(punctuation_scores) / len(punctuation_scores)
            if punctuation_scores
            else 0.0
        ),
        question_mark_accuracy=(
            sum(question_mark_matches) / len(question_mark_matches)
            if question_mark_matches
            else 0.0
        ),
        textual_macro_accuracy=textual_accuracy,
        prosodic_macro_accuracy=prosodic_accuracy,
        semantic_macro_accuracy=semantic_accuracy,
        fused_macro_accuracy=fused_accuracy,
        question_intonation_recall=(
            intonation_hits / len(intonation_questions) if intonation_questions else 0.0
        ),
        routing_exact_plan_accuracy=routing_exact_plan_accuracy,
        routing_topk_recall=routing_topk_recall,
        routing_mean_reciprocal_rank=(
            sum(routing_reciprocal_ranks) / len(routing_reciprocal_ranks)
            if routing_reciprocal_ranks
            else 0.0
        ),
        safe_abstention_recall=(
            safe_abstention_hits / expected_abstention_count
            if expected_abstention_count
            else 1.0
        ),
        unsafe_resolution_count=unsafe_count,
        unavailable_prosody_count=sum(
            not prediction.prosody.available
            for prediction in predictions
            if prediction.prosody is not None
        ),
        fused_accuracy_by_tag=fused_accuracy_by_tag,
        wer_by_tag=wer_by_tag,
        quality_gate_passed=not failures,
        quality_gate_failures=tuple(failures),
    )
    return predictions, metrics


def classify_textual_intent(text: str) -> tuple[VoiceIntentLabel, float]:
    normalized = normalize_text(text)
    words = set(normalized.split())
    if words & _CANCEL_WORDS:
        return VoiceIntentLabel.CANCELLATION, 0.95
    if words & _QUESTION_WORDS:
        return VoiceIntentLabel.QUESTION, 0.90
    if words & _TASK_WORDS:
        return VoiceIntentLabel.TASK, 0.85
    if not normalized:
        return VoiceIntentLabel.AMBIGUOUS, 0.0
    return VoiceIntentLabel.CONVERSATION, 0.65


def fuse_intent_scores(
    *,
    textual_label: VoiceIntentLabel,
    textual_score: float,
    prosodic_label: VoiceIntentLabel,
    prosodic_score: float,
    semantic_label: VoiceIntentLabel,
    semantic_score: float,
) -> tuple[VoiceIntentLabel, float]:
    scores: dict[VoiceIntentLabel, float] = {}
    scores[textual_label] = scores.get(textual_label, 0.0) + textual_score * 0.35
    scores[prosodic_label] = scores.get(prosodic_label, 0.0) + prosodic_score * 0.30
    scores[semantic_label] = scores.get(semantic_label, 0.0) + semantic_score * 0.35
    label, score = max(scores.items(), key=lambda item: (item[1], item[0].value))
    return label, max(0.0, min(score, 1.0))


def render_voice_report(metrics: VoiceEvalMetricsV1) -> str:
    return (
        "# VoiceLoop Voice Evaluation V1\n\n"
        f"- Próbki: {metrics.sample_count}\n"
        f"- Development/holdout: {metrics.development_count}/{metrics.holdout_count}\n"
        f"- Adnotacje: {metrics.annotated_count}\n"
        f"- WER: {metrics.mean_wer:.3f}\n"
        f"- CER: {metrics.mean_cer:.3f}\n"
        f"- Interpunkcja F1: {metrics.mean_punctuation_f1:.3f}\n"
        f"- Trafność znaku zapytania: {metrics.question_mark_accuracy:.3f}\n"
        f"- Trafność tekstowa: {metrics.textual_macro_accuracy:.3f}\n"
        f"- Trafność prozodyczna: {metrics.prosodic_macro_accuracy:.3f}\n"
        f"- Trafność semantyczna: {metrics.semantic_macro_accuracy:.3f}\n"
        f"- Trafność łączna: {metrics.fused_macro_accuracy:.3f}\n"
        f"- Recall pytań intonacyjnych: {metrics.question_intonation_recall:.3f}\n"
        f"- Exact plan: {metrics.routing_exact_plan_accuracy:.3f}\n"
        f"- Routing top-k: {metrics.routing_topk_recall:.3f}\n"
        f"- Routing MRR: {metrics.routing_mean_reciprocal_rank:.3f}\n"
        f"- Safe abstention: {metrics.safe_abstention_recall:.3f}\n"
        f"- Niebezpieczne rozstrzygnięcia: {metrics.unsafe_resolution_count}\n"
        f"- Bramka jakości: {'PASS' if metrics.quality_gate_passed else 'FAIL'}\n"
    )


def _routing_payload(outcome: Any) -> dict[str, Any]:
    plan = getattr(outcome, "plan", None)
    steps = tuple(getattr(plan, "steps", ()) or ())
    decisions = tuple(getattr(outcome, "decisions", ()) or ())
    first_decision = decisions[0] if decisions else None
    candidates = tuple(getattr(first_decision, "candidates", ()) or ())
    candidates_by_subtask = [
        [
            str(getattr(candidate, "action_id", ""))
            for candidate in tuple(getattr(decision, "candidates", ()) or ())
        ]
        for decision in decisions
    ]
    return {
        "predicted_action_ids": [
            str(step.action_id) for step in steps
        ],
        "predicted_step_args": [
            dict(step.args) for step in steps
        ],
        "decision": (
            getattr(getattr(first_decision, "decision", None), "value", None)
            if first_decision is not None
            else None
        ),
        "reason": getattr(first_decision, "reason", None),
        "top1_action_id": (
            str(getattr(candidates[0], "action_id", "")) if candidates else None
        ),
        "top1_score": (
            float(getattr(candidates[0], "combined_score", 0.0)) if candidates else None
        ),
        "margin_top2": getattr(first_decision, "margin_top2", None),
        "topk_action_ids": [str(getattr(item, "action_id", "")) for item in candidates],
        "topk_action_ids_by_subtask": candidates_by_subtask,
    }


def punctuation_f1(reference: str, hypothesis: str) -> float:
    punctuation = ".,?!:;"
    reference_counts = Counter(character for character in reference if character in punctuation)
    hypothesis_counts = Counter(
        character for character in hypothesis if character in punctuation
    )
    reference_total = sum(reference_counts.values())
    hypothesis_total = sum(hypothesis_counts.values())
    if reference_total == 0 and hypothesis_total == 0:
        return 1.0
    true_positive = sum(
        min(reference_counts[character], hypothesis_counts[character])
        for character in punctuation
    )
    if true_positive == 0:
        return 0.0
    precision = true_positive / hypothesis_total if hypothesis_total else 0.0
    recall = true_positive / reference_total if reference_total else 0.0
    return 2 * precision * recall / (precision + recall)


def _label_accuracy(
    pairs: list[tuple[VoiceEvalPredictionV1, VoiceGoldAnnotationV1]],
    getter: Callable[[VoiceEvalPredictionV1], VoiceIntentLabel | None],
) -> float:
    if not pairs:
        return 0.0
    return sum(getter(prediction) is annotation.intent for prediction, annotation in pairs) / len(
        pairs
    )


def _annotation_tags(annotation: VoiceGoldAnnotationV1) -> set[str]:
    tags = {annotation.intent.value, *annotation.prosody_tags}
    if annotation.proper_names:
        tags.add("proper_name")
    if len(annotation.expected_action_ids) > 1:
        tags.add("compound")
    if annotation.expected_abstention:
        tags.add("safe_abstention")
    return tags


def _safe_eval_path(eval_root: Path, relative_path: str) -> Path:
    root = eval_root.resolve()
    candidate = (root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise VoiceEvalError("Ścieżka próbki wychodzi poza katalog ewaluacji.") from exc
    return candidate
