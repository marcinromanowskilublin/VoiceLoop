from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CORPUS_SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


class CorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceKind(StrEnum):
    AUDIO_TRANSCRIPT = "audio_transcript"
    CURSOR_USER = "cursor_user"
    TECHNICAL_EXCLUDED = "technical_excluded"


class UtteranceOrigin(StrEnum):
    AUDIO = "audio"
    CURSOR_USER = "cursor_user"


class SpeakerStatus(StrEnum):
    SELF = "self"
    OTHER = "other"
    MULTI = "multi"
    UNKNOWN = "unknown"


class CorpusSplit(StrEnum):
    TRAIN = "train"
    HOLDOUT = "holdout"
    UNUSED = "unused"


class ProcessingLocation(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class NetworkScope(StrEnum):
    NONE = "none"
    LOOPBACK = "loopback"
    INTERNET = "internet"


class AudioDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SpeakerRole(StrEnum):
    SELF = "self"
    OTHER = "other"
    MULTI = "multi"
    UNKNOWN = "unknown"


class VoiceSampleState(StrEnum):
    CANDIDATE = "candidate"
    SELECTED = "selected"
    ANNOTATED = "annotated"
    EXCLUDED = "excluded"


class VoiceEvalSplit(StrEnum):
    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


class VoiceIntentLabel(StrEnum):
    QUESTION = "question"
    CONVERSATION = "conversation"
    TASK = "task"
    AMBIGUOUS = "ambiguous"
    CANCELLATION = "cancellation"
    BARGE_IN = "barge_in"


class ExpectedVoiceOutcome(StrEnum):
    RESPOND = "respond"
    EXECUTE = "execute"
    CLARIFY = "clarify"
    REJECT = "reject"
    IGNORE = "ignore"


class JournalCategory(StrEnum):
    GOAL = "goal"
    DECISION = "decision"
    OPEN_PROBLEM = "open_problem"
    ARCHITECTURE_CHANGE = "architecture_change"
    TO_VERIFY = "to_verify"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ProvenanceV1(CorpusModel):
    schema_version: Literal[1] = 1
    source_system: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=500)
    source_record_id: str | None = Field(default=None, max_length=500)
    session_id: str | None = Field(default=None, max_length=500)
    request_id: str | None = Field(default=None, max_length=500)
    captured_start: datetime
    captured_end: datetime
    processing_location: ProcessingLocation = ProcessingLocation.LOCAL
    network_scope: NetworkScope = NetworkScope.NONE
    audio_direction: AudioDirection = AudioDirection.UNKNOWN
    device_name: str = Field(default="", max_length=500)
    device_type: str = Field(default="", max_length=100)
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    active_app: str = Field(default="", max_length=500)
    active_window: str = Field(default="", max_length=1000)
    retry_of: str | None = Field(default=None, max_length=500)
    duplicate_of: str | None = Field(default=None, max_length=500)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_capture_range(self) -> ProvenanceV1:
        if self.captured_end < self.captured_start:
            raise ValueError("Koniec przechwycenia nie może poprzedzać początku.")
        if self.retry_of and self.retry_of == self.source_record_id:
            raise ValueError("Rekord nie może być retry samego siebie.")
        if self.duplicate_of and self.duplicate_of == self.source_record_id:
            raise ValueError("Rekord nie może być duplikatem samego siebie.")
        return self


class AudioAssetRefV1(CorpusModel):
    schema_version: Literal[1] = 1
    relative_path: str = Field(min_length=1, max_length=1000)
    source_chunk_id: str = Field(min_length=1, max_length=500)
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(default="audio/wav", min_length=1, max_length=100)
    sample_rate_hz: int = Field(default=16000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=32)
    duration_seconds: float = Field(gt=0.0, le=3600.0)
    source_offset_start_seconds: float = Field(default=0.0, ge=0.0)
    source_offset_end_seconds: float = Field(gt=0.0)
    decoder_version: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_audio_asset(self) -> AudioAssetRefV1:
        path = self.relative_path.replace("\\", "/")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Ścieżka audio musi być względna i zamknięta w katalogu korpusu.")
        if self.source_offset_end_seconds <= self.source_offset_start_seconds:
            raise ValueError("Koniec fragmentu audio musi następować po początku.")
        return self


class VoiceAudioSourceV1(CorpusModel):
    schema_version: Literal[1] = 1
    source_id: str = Field(min_length=1, max_length=500)
    chunk_id: str = Field(min_length=1, max_length=500)
    path: str = Field(min_length=1, max_length=2000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    modified_at: datetime
    captured_start: datetime
    captured_end: datetime
    device_name: str = Field(default="", max_length=500)
    device_type: str = Field(default="", max_length=100)
    audio_direction: AudioDirection = AudioDirection.UNKNOWN
    source_offset_start_seconds: float = Field(default=0.0, ge=0.0)
    source_offset_end_seconds: float = Field(default=0.0, ge=0.0)
    included: bool = True
    exclude_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_source_range(self) -> VoiceAudioSourceV1:
        if self.included and self.captured_end <= self.captured_start:
            raise ValueError("Źródłowe audio musi mieć dodatni przedział czasu.")
        if not self.included and self.captured_end < self.captured_start:
            raise ValueError("Koniec wykluczonego źródła nie może poprzedzać początku.")
        if (
            self.source_offset_end_seconds
            and self.source_offset_end_seconds <= self.source_offset_start_seconds
        ):
            raise ValueError("Offset końca źródła musi następować po początku.")
        return self


class VoiceSourceManifestV1(CorpusModel):
    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)
    range_start: datetime
    range_end: datetime
    sources: tuple[VoiceAudioSourceV1, ...] = ()
    included_source_count: int = Field(ge=0)
    excluded_source_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_manifest(self) -> VoiceSourceManifestV1:
        if self.range_end <= self.range_start:
            raise ValueError("Koniec zakresu manifestu musi następować po początku.")
        included = sum(source.included for source in self.sources)
        if included != self.included_source_count:
            raise ValueError("Liczba dołączonych źródeł nie zgadza się z manifestem.")
        if len(self.sources) - included != self.excluded_source_count:
            raise ValueError("Liczba wykluczonych źródeł nie zgadza się z manifestem.")
        return self


class VoiceEvalSampleV1(CorpusModel):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1, max_length=160)
    provenance: ProvenanceV1
    audio: AudioAssetRefV1 | None = None
    observed_text: str = Field(default="", max_length=8000)
    tags: tuple[str, ...] = ()
    duplicate_group_id: str | None = Field(default=None, max_length=160)
    duplicate_type: str | None = Field(default=None, max_length=80)
    dedupe_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    dedupe_reasons: tuple[str, ...] = ()
    state: VoiceSampleState = VoiceSampleState.CANDIDATE
    split: VoiceEvalSplit | None = None
    exclusion_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_sample_state(self) -> VoiceEvalSampleV1:
        if self.state in {VoiceSampleState.SELECTED, VoiceSampleState.ANNOTATED}:
            if self.audio is None:
                raise ValueError("Wybrana próbka musi mieć zamrożone audio.")
            if self.split is None:
                raise ValueError("Wybrana próbka musi mieć przypisany split.")
        if self.state is VoiceSampleState.EXCLUDED and not self.exclusion_reason:
            raise ValueError("Wykluczona próbka wymaga powodu.")
        if self.state is not VoiceSampleState.EXCLUDED and self.exclusion_reason:
            raise ValueError("Powód wykluczenia jest dozwolony tylko dla wykluczonej próbki.")
        return self


class VoiceGoldAnnotationV1(CorpusModel):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1, max_length=160)
    audio_clip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    literal_text: str = Field(min_length=1, max_length=8000)
    punctuated_text: str = Field(min_length=1, max_length=8000)
    intent: VoiceIntentLabel
    prosody_tags: tuple[str, ...] = ()
    proper_names: tuple[str, ...] = ()
    speaker_role: SpeakerRole
    speaker_confirmed: bool
    expected_outcome: ExpectedVoiceOutcome
    expected_action_ids: tuple[str, ...] = ()
    expected_step_args: tuple[dict[str, Any], ...] = ()
    expected_abstention: bool = False
    annotator: str = Field(min_length=1, max_length=160)
    approved_at: datetime

    @model_validator(mode="after")
    def validate_gold_annotation(self) -> VoiceGoldAnnotationV1:
        if not self.speaker_confirmed or self.speaker_role is not SpeakerRole.SELF:
            raise ValueError("Próbka wymaga ręcznego potwierdzenia własnego mówcy.")
        if self.expected_outcome is ExpectedVoiceOutcome.EXECUTE and not self.expected_action_ids:
            raise ValueError("Wynik execute wymaga co najmniej jednej oczekiwanej akcji.")
        if self.expected_step_args and len(self.expected_step_args) != len(
            self.expected_action_ids
        ):
            raise ValueError("Argumenty muszą odpowiadać wszystkim oczekiwanym akcjom.")
        if self.expected_abstention and self.expected_outcome is ExpectedVoiceOutcome.EXECUTE:
            raise ValueError("Abstencja i wykonanie nie mogą być oczekiwane jednocześnie.")
        return self


class ProsodyFeaturesV1(CorpusModel):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1, max_length=160)
    available: bool
    reason: str | None = Field(default=None, max_length=500)
    duration_seconds: float = Field(ge=0.0)
    voiced_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    f0_median_hz: float | None = Field(default=None, gt=0.0)
    f0_range_semitones: float | None = Field(default=None, ge=0.0)
    final_f0_delta_semitones: float | None = None
    final_f0_slope_semitones_per_second: float | None = None
    rms_mean: float | None = Field(default=None, ge=0.0)
    rms_peak: float | None = Field(default=None, ge=0.0)
    final_rms_delta: float | None = None
    pause_count: int = Field(default=0, ge=0)
    pause_total_seconds: float = Field(default=0.0, ge=0.0)
    silence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    words_per_second: float | None = Field(default=None, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_availability(self) -> ProsodyFeaturesV1:
        if not self.available and not self.reason:
            raise ValueError("Niedostępna prozodia wymaga powodu.")
        return self


class VoiceEvalPredictionV1(CorpusModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1, max_length=160)
    sample_id: str = Field(min_length=1, max_length=160)
    transcript_text: str = Field(default="", max_length=8000)
    transcript_provider: str = Field(default="deepgram", max_length=80)
    transcript_processing_location: ProcessingLocation = ProcessingLocation.REMOTE
    transcript_network_scope: NetworkScope = NetworkScope.INTERNET
    transcript_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    transcript_words: tuple[dict[str, Any], ...] = ()
    prosody: ProsodyFeaturesV1 | None = None
    textual_label: VoiceIntentLabel | None = None
    textual_score: float | None = Field(default=None, ge=0.0, le=1.0)
    prosodic_label: VoiceIntentLabel | None = None
    prosodic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_label: VoiceIntentLabel | None = None
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_processing_location: ProcessingLocation = ProcessingLocation.LOCAL
    semantic_network_scope: NetworkScope = NetworkScope.LOOPBACK
    fused_label: VoiceIntentLabel | None = None
    fused_score: float | None = Field(default=None, ge=0.0, le=1.0)
    routing: dict[str, Any] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()


class VoiceEvalRunManifestV1(CorpusModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1, max_length=160)
    created_at: datetime = Field(default_factory=utc_now)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, Any] = Field(default_factory=dict)
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class VoiceEvalMetricsV1(CorpusModel):
    schema_version: Literal[1] = 1
    run_id: str
    sample_count: int = Field(ge=0)
    development_count: int = Field(ge=0)
    holdout_count: int = Field(ge=0)
    annotated_count: int = Field(ge=0)
    mean_wer: float = Field(ge=0.0)
    mean_cer: float = Field(ge=0.0)
    mean_punctuation_f1: float = Field(ge=0.0, le=1.0)
    question_mark_accuracy: float = Field(ge=0.0, le=1.0)
    textual_macro_accuracy: float = Field(ge=0.0, le=1.0)
    prosodic_macro_accuracy: float = Field(ge=0.0, le=1.0)
    semantic_macro_accuracy: float = Field(ge=0.0, le=1.0)
    fused_macro_accuracy: float = Field(ge=0.0, le=1.0)
    question_intonation_recall: float = Field(ge=0.0, le=1.0)
    routing_exact_plan_accuracy: float = Field(ge=0.0, le=1.0)
    routing_topk_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    routing_mean_reciprocal_rank: float = Field(default=0.0, ge=0.0, le=1.0)
    safe_abstention_recall: float = Field(ge=0.0, le=1.0)
    unsafe_resolution_count: int = Field(ge=0)
    unavailable_prosody_count: int = Field(ge=0)
    fused_accuracy_by_tag: dict[str, float] = Field(default_factory=dict)
    wer_by_tag: dict[str, float] = Field(default_factory=dict)
    quality_gate_passed: bool
    quality_gate_failures: tuple[str, ...] = ()


class ProperNameEntryV1(CorpusModel):
    schema_version: Literal[1] = 1
    canonical: str = Field(min_length=1, max_length=200)
    aliases: tuple[str, ...] = ()
    common_stt_errors: tuple[str, ...] = ()
    pronunciation_hint: str = Field(default="", max_length=500)
    category: str = Field(default="tool", max_length=80)
    evidence_sample_ids: tuple[str, ...] = ()
    approved: bool = False


class ProperNameLexiconV1(CorpusModel):
    schema_version: Literal[1] = 1
    created_at: datetime = Field(default_factory=utc_now)
    entries: tuple[ProperNameEntryV1, ...] = ()

    @model_validator(mode="after")
    def validate_unique_names(self) -> ProperNameLexiconV1:
        names = [entry.canonical.casefold() for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("Nazwy kanoniczne w słowniku muszą być unikalne.")
        return self


class ProjectJournalCandidateV1(CorpusModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1, max_length=160)
    category: JournalCategory
    summary: str = Field(min_length=1, max_length=2000)
    evidence_utterance_ids: tuple[str, ...]
    source_session_id: str = Field(min_length=1, max_length=500)
    source_timestamp: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: ReviewStatus = ReviewStatus.PENDING
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_journal_review(self) -> ProjectJournalCandidateV1:
        if not self.evidence_utterance_ids:
            raise ValueError("Kandydat dziennika wymaga dowodu źródłowego.")
        if self.status is ReviewStatus.PENDING and self.reviewed_at is not None:
            raise ValueError("Oczekujący kandydat nie może mieć daty decyzji.")
        if self.status is not ReviewStatus.PENDING and self.reviewed_at is None:
            raise ValueError("Rozpatrzony kandydat wymaga daty decyzji.")
        return self


class SourceManifestRecord(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    source_id: str
    kind: SourceKind
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    modified_at: datetime
    word_count: int = Field(ge=0)
    date_start: date | None = None
    date_end: date | None = None
    included: bool = True
    exclude_reason: str | None = None


class SourceManifest(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    manifest_id: str
    created_at: datetime = Field(default_factory=utc_now)
    sources: list[SourceManifestRecord]
    included_word_count: int = Field(ge=0)
    excluded_word_count: int = Field(ge=0)
    unique_source_count: int = Field(ge=0)


class SpeakerDecision(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_status: Literal["self", "other"]
    decided_at: datetime = Field(default_factory=utc_now)


class SpeakerDecisionFile(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    decisions: list[SpeakerDecision] = Field(default_factory=list)


class UtteranceRecord(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    utterance_id: str
    source_id: str
    origin: UtteranceOrigin
    session_id: str
    captured_at: datetime | None = None
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(max_length=20000)
    speaker_status: SpeakerStatus
    speaker_ids: tuple[int, ...] = ()
    is_near_duplicate: bool = False
    duplicate_of: str | None = None
    pii_flags: tuple[str, ...] = ()
    redacted: bool = False
    quarantine_reason: str | None = None
    split: CorpusSplit | None = None


class QuarantineRecord(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    utterance_id: str
    source_id: str
    origin: UtteranceOrigin
    session_id: str
    captured_at: datetime | None = None
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_status: SpeakerStatus
    pii_flags: tuple[str, ...] = ()
    reason: str


class SttErrorType(StrEnum):
    NONE = "none"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"
    FLEXION = "flexion"
    COMPOUND = "compound"
    LOW_CONFIDENCE = "low_confidence"


class ExpectedIntent(StrEnum):
    TASK = "task"
    CONVERSATION = "conversation"
    AMBIGUOUS = "ambiguous"


class RoutingEvalRecord(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    example_id: str
    base_example_id: str | None = None
    gold_text: str = Field(min_length=1, max_length=500)
    stt_text: str = Field(min_length=1, max_length=500)
    stt_confidence: float = Field(ge=0.0, le=1.0)
    error_type: SttErrorType = SttErrorType.NONE
    expected_action_id: str | None = None
    expected_action_ids: tuple[str, ...] = ()
    expected_step_args: tuple[dict[str, Any], ...] = ()
    expected_intent: ExpectedIntent
    ambiguous: bool = False
    compound: bool = False
    speaker_ids: tuple[int, ...] = (0,)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_ambiguity(self) -> RoutingEvalRecord:
        if self.ambiguous and self.expected_intent is not ExpectedIntent.AMBIGUOUS:
            raise ValueError("Niejednoznaczny przykład musi mieć intent=ambiguous.")
        expected_actions = self.plan_action_ids
        if self.expected_intent is ExpectedIntent.TASK and not expected_actions:
            raise ValueError("Przykład zadania wymaga oczekiwanej akcji.")
        if (
            self.expected_action_id
            and self.expected_action_ids
            and self.expected_action_id != self.expected_action_ids[0]
        ):
            raise ValueError("expected_action_id musi wskazywać pierwszy krok planu.")
        if self.expected_step_args and len(self.expected_step_args) != len(expected_actions):
            raise ValueError("Argumenty muszą odpowiadać wszystkim oczekiwanym krokom.")
        if (
            len(set(self.speaker_ids)) != len(self.speaker_ids)
            or tuple(sorted(self.speaker_ids)) != self.speaker_ids
            or any(speaker_id < 0 for speaker_id in self.speaker_ids)
        ):
            raise ValueError("speaker_ids muszą być unikalne, posortowane i nieujemne.")
        if len(self.gold_text.split()) > 25 or len(self.stt_text.split()) > 25:
            raise ValueError("Przykład STT/routingu może mieć najwyżej 25 słów.")
        return self

    @property
    def plan_action_ids(self) -> tuple[str, ...]:
        if self.expected_action_ids:
            return self.expected_action_ids
        return (self.expected_action_id,) if self.expected_action_id else ()


class RoutingScoreCard(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    example_id: str
    base_example_id: str | None
    expected_action_id: str | None
    predicted_top1: str | None
    topk_action_ids: tuple[str, ...]
    scores: tuple[float, ...]
    margin_top2: float | None
    hit_at_1: bool
    hit_at_k: bool
    below_min_score: bool
    stt_gate_blocked: bool
    expected_ambiguous: bool
    predicted_ambiguous: bool


class RoutingMetrics(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    sample_count: int = Field(ge=0)
    base_example_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    expected_action_count: int = Field(ge=0)
    catalog_coverage: float = Field(ge=0.0, le=1.0)
    top1_accuracy: float = Field(ge=0.0, le=1.0)
    topk_recall: float = Field(ge=0.0, le=1.0)
    mean_margin: float = Field(ge=-2.0, le=2.0)
    ambiguity_precision: float = Field(ge=0.0, le=1.0)
    ambiguity_recall: float = Field(ge=0.0, le=1.0)
    stt_gate_precision: float = Field(ge=0.0, le=1.0)
    stt_gate_recall: float = Field(ge=0.0, le=1.0)
    quality_gate_passed: bool
    quality_gate_failures: tuple[str, ...] = ()


class RoutingV2ScoreCard(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    example_id: str
    base_example_id: str | None
    expected_action_id: str | None
    expected_action_ids: tuple[str, ...] = ()
    expected_step_args: tuple[dict[str, Any], ...] = ()
    predicted_action_ids: tuple[str, ...] = ()
    predicted_step_args: tuple[dict[str, Any], ...] = ()
    topk_action_ids: tuple[str, ...] = ()
    top1_action_id: str | None = None
    top1_combined_score: float | None = Field(default=None, ge=0.0, le=1.0)
    margin_top2: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: Literal["resolved", "clarify", "unsupported"]
    reason: str | None = None
    expected_abstention: bool = False
    abstained: bool = False
    arguments_match: bool = False
    exact_plan_match: bool = False
    hit_at_1: bool = False
    hit_at_k: bool = False
    unsafe_resolution: bool = False


class RoutingV2RuntimeConfig(CorpusModel):
    schema_version: Literal[1] = 1
    candidate_limit: int = Field(ge=2, le=20)
    execute_min_score: float = Field(ge=0.0, le=1.0)
    execute_min_margin: float = Field(ge=0.0, le=1.0)
    stt_threshold: float = Field(ge=0.0, le=1.0)
    max_subtasks: int = Field(ge=1, le=100)
    embedding_model: str = Field(min_length=1, max_length=500)
    embedding_dimension: int = Field(ge=1)
    capability_collection: str = Field(min_length=1, max_length=500)
    catalog_hash: str = Field(min_length=1, max_length=100)
    vector_distance: Literal["cosine"] = "cosine"
    taxonomy_version: str = Field(default="routing-taxonomy-v2", min_length=1, max_length=100)
    document_format_version: str = Field(
        default="capability-document-v2:routing-taxonomy-v2",
        min_length=1,
        max_length=100,
    )
    query_format_version: str = Field(
        default="capability-query-v2:routing-taxonomy-v2",
        min_length=1,
        max_length=100,
    )
    vector_fusion: str = Field(default="weighted-rrf-v1", min_length=1, max_length=100)
    rank_fusion_k: int = Field(default=60, ge=1, le=1000)
    vector_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic": 1.0,
            "intent": 1.0,
            "target_context": 1.0,
        }
    )
    resolver_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "vector": 0.60,
            "lexical": 0.35,
            "arguments": 0.05,
        }
    )
    missing_space_policy: str = Field(
        default="coverage-adjusted-observed-cosine-v1",
        min_length=1,
        max_length=100,
    )
    minimum_vector_coverage: float = Field(default=2 / 3, ge=0.0, le=1.0)
    single_candidate_policy: str = Field(
        default="clarify-without-comparator-v1",
        min_length=1,
        max_length=100,
    )
    implementation_version: Literal["routing-v2-runtime-7"] = "routing-v2-runtime-7"

    @model_validator(mode="after")
    def validate_algorithm_contract(self) -> RoutingV2RuntimeConfig:
        expected_vectors = {"semantic", "intent", "target_context"}
        if set(self.vector_weights) != expected_vectors:
            raise ValueError("vector_weights must define the three capability spaces")
        expected_resolver = {"vector", "lexical", "arguments"}
        if set(self.resolver_weights) != expected_resolver:
            raise ValueError("resolver_weights must define vector, lexical and arguments")
        for weights, name in (
            (self.vector_weights, "vector_weights"),
            (self.resolver_weights, "resolver_weights"),
        ):
            if any(not 0.0 <= float(value) <= 1.0 for value in weights.values()):
                raise ValueError(f"{name} must contain values in range 0..1")
            if sum(float(value) for value in weights.values()) <= 0.0:
                raise ValueError(f"{name} must contain a positive total weight")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class RoutingV2Metrics(CorpusModel):
    schema_version: Literal[2] = 2
    catalog_hash: str
    runtime_config: RoutingV2RuntimeConfig
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=0)
    base_example_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    expected_action_count: int = Field(ge=0)
    catalog_coverage: float = Field(ge=0.0, le=1.0)
    resolved_accuracy: float = Field(ge=0.0, le=1.0)
    topk_recall: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_margin_top2: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_calibration_error: float = Field(default=0.0, ge=0.0, le=1.0)
    safe_abstention_recall: float = Field(ge=0.0, le=1.0)
    unsafe_resolution_count: int = Field(ge=0)
    quality_gate_passed: bool
    quality_gate_failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_runtime_fingerprint(self) -> RoutingV2Metrics:
        if self.runtime_config.catalog_hash != self.catalog_hash:
            raise ValueError("Runtime config i raport mają różne hashe katalogu.")
        if self.runtime_config.fingerprint() != self.runtime_fingerprint:
            raise ValueError("Nieprawidłowy fingerprint konfiguracji routingu.")
        return self


class RoutingCalibrationMode(StrEnum):
    OFF = "off"
    REPORT_ONLY = "report_only"


class RoutingCalibrationSetRole(StrEnum):
    REPRESENTATIVE = "representative"
    CHALLENGE = "challenge"


class RoutingCalibrationSafetyOutcome(StrEnum):
    SAFE_EXECUTE_CORRECT = "safe_execute_correct"
    SAFE_ABSTAIN = "safe_abstain"
    MISSED_ACTION = "missed_action"
    UNSAFE_RESOLUTION = "unsafe_resolution"


class RoutingCalibrationArtifactStatus(StrEnum):
    READY = "ready"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_REPORTABLE = "not_reportable"


class RoutingCalibrationObservationV1(CorpusModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=120)
    dataset_id: str = Field(min_length=1, max_length=160)
    source_record_id: str = Field(min_length=1, max_length=160)
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observed_at: datetime = Field(default_factory=utc_now)
    source: str = Field(min_length=1, max_length=80)
    normalized_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(min_length=1, max_length=100)
    session_id: str | None = Field(default=None, max_length=120)
    split_group_override: str | None = Field(default=None, max_length=160)
    group_id: str = Field(min_length=1, max_length=120)
    set_role: RoutingCalibrationSetRole = RoutingCalibrationSetRole.REPRESENTATIVE
    subtask_index: int = Field(ge=0, le=99)
    subtask_count: int = Field(ge=1, le=100)
    decision: Literal["resolved", "clarify", "unsupported"]
    predicted_action_id: str | None = Field(default=None, max_length=120)
    expected_action_id: str | None = Field(default=None, max_length=120)
    ranking_score: float | None = Field(default=None, ge=0.0, le=1.0)
    margin_top2: float | None = Field(default=None, ge=0.0, le=1.0)
    vector_score: float | None = Field(default=None, ge=0.0, le=1.0)
    lexical_score: float | None = Field(default=None, ge=0.0, le=1.0)
    argument_score: float | None = Field(default=None, ge=0.0, le=1.0)
    vector_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    stt_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_count: int = Field(ge=0, le=100)
    eligible: bool | None = None
    rejection_reasons: tuple[str, ...] = ()
    p_action_correct: float | None = Field(default=None, ge=0.0, le=1.0)
    action_correct: bool | None = None
    action_sequence_correct: bool | None = None
    exact_plan_correct: bool | None = None
    safety_outcome: RoutingCalibrationSafetyOutcome | None = None
    label_source: str | None = Field(default=None, max_length=120)
    snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "subtask_index": self.subtask_index,
            "dataset_id": self.dataset_id,
            "source_record_id": self.source_record_id,
            "manifest_sha256": self.manifest_sha256,
            "observed_at": self.observed_at.isoformat(),
            "source": self.source,
            "normalized_text_sha256": self.normalized_text_sha256,
            "runtime_fingerprint": self.runtime_fingerprint,
            "catalog_hash": self.catalog_hash,
            "session_id": self.session_id,
            "group_id": self.group_id,
            "set_role": self.set_role.value,
            "subtask_count": self.subtask_count,
            "decision": self.decision,
            "predicted_action_id": self.predicted_action_id,
            "ranking_score": self.ranking_score,
            "margin_top2": self.margin_top2,
            "vector_score": self.vector_score,
            "lexical_score": self.lexical_score,
            "argument_score": self.argument_score,
            "vector_coverage": self.vector_coverage,
            "stt_confidence": self.stt_confidence,
            "candidate_count": self.candidate_count,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "p_action_correct": self.p_action_correct,
        }

    def recompute_snapshot_sha256(self) -> str:
        payload = json.dumps(
            self.snapshot_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @model_validator(mode="after")
    def validate_labels(self) -> RoutingCalibrationObservationV1:
        if self.subtask_index >= self.subtask_count:
            raise ValueError("subtask_index must be lower than subtask_count")
        if self.exact_plan_correct is not None and self.action_sequence_correct is None:
            raise ValueError("exact_plan_correct requires action_sequence_correct label")
        if self.action_sequence_correct is False and self.exact_plan_correct is True:
            raise ValueError("exact_plan_correct cannot be true when action_sequence is false")
        if self.action_correct is not None and self.expected_action_id:
            if not self.predicted_action_id:
                raise ValueError(
                    "predicted_action_id is required when expected_action_id "
                    "and action_correct are set"
                )
            expected_match = self.predicted_action_id == self.expected_action_id
            if bool(self.action_correct) != expected_match:
                raise ValueError(
                    "action_correct must match predicted_action_id == expected_action_id"
                )
        if self.snapshot_sha256 is not None:
            expected_snapshot = self.recompute_snapshot_sha256()
            if self.snapshot_sha256 != expected_snapshot:
                raise ValueError("snapshot_sha256 must match immutable observation payload")
        return self


class RoutingCalibrationCoefficientsV1(CorpusModel):
    a: float
    b: float

    @model_validator(mode="after")
    def validate_coefficients(self) -> RoutingCalibrationCoefficientsV1:
        if not math.isfinite(self.a) or self.a < 0.0:
            raise ValueError("coefficient a must be finite and monotonic (a >= 0)")
        if not math.isfinite(self.b):
            raise ValueError("coefficient b must be finite")
        return self


class RoutingCalibrationClassCountsV1(CorpusModel):
    successes: int = Field(ge=0)
    errors: int = Field(ge=0)


class RoutingCalibrationCredibilityGatesV1(CorpusModel):
    point_ece_le_0_10: bool
    ci_upper_ece_le_0_15: bool
    brier_improves_over_raw: bool
    logloss_not_worse_than_intercept: bool
    passed: bool

    @model_validator(mode="after")
    def validate_passed_consistency(self) -> RoutingCalibrationCredibilityGatesV1:
        expected = bool(
            self.point_ece_le_0_10
            and self.ci_upper_ece_le_0_15
            and self.brier_improves_over_raw
            and self.logloss_not_worse_than_intercept
        )
        if self.passed != expected:
            raise ValueError(
                "credibility_gates.passed must equal conjunction of all component gates"
            )
        return self


class RoutingCalibrationArtifactV1(CorpusModel):
    schema_version: Literal[1] = 1
    target_version: Literal["action_correct.v1"] = "action_correct.v1"
    feature_version: Literal["ranking_score.v1"] = "ranking_score.v1"
    model_version: Literal["platt_sigmoid_monotonic_v1"] = "platt_sigmoid_monotonic_v1"
    status: RoutingCalibrationArtifactStatus
    created_at: datetime = Field(default_factory=utc_now)
    coefficients: RoutingCalibrationCoefficientsV1
    class_counts: RoutingCalibrationClassCountsV1
    group_count: int = Field(ge=0)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_window_start: datetime | None = None
    training_window_end: datetime | None = None
    evaluation_window_start: datetime | None = None
    evaluation_window_end: datetime | None = None
    train_cutoff: datetime | None = None
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(min_length=1, max_length=100)
    train_component_ids: tuple[str, ...] = ()
    evaluation_component_ids: tuple[str, ...] = ()
    excluded_component_ids: tuple[str, ...] = ()
    train_group_tokens: tuple[str, ...] = ()
    credibility_gates: RoutingCalibrationCredibilityGatesV1 | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_windows(self) -> RoutingCalibrationArtifactV1:
        for start, end in (
            (self.training_window_start, self.training_window_end),
            (self.evaluation_window_start, self.evaluation_window_end),
        ):
            if start is not None and end is not None and end < start:
                raise ValueError("dataset window end cannot precede start")
        if any(
            not math.isfinite(value)
            for value in self.metrics.values()
        ):
            raise ValueError("artifact metrics must be finite numbers")
        if self.status is RoutingCalibrationArtifactStatus.READY:
            if self.credibility_gates is None or not self.credibility_gates.passed:
                raise ValueError("READY artifact requires passed credibility gates")
            if (
                not self.train_component_ids
                or not self.evaluation_component_ids
                or not self.train_group_tokens
                or self.train_cutoff is None
            ):
                raise ValueError("READY artifact requires split manifest and cutoff")
            train_set = set(self.train_component_ids)
            eval_set = set(self.evaluation_component_ids)
            excluded_set = set(self.excluded_component_ids)
            if (
                len(train_set) != len(self.train_component_ids)
                or len(eval_set) != len(self.evaluation_component_ids)
                or len(excluded_set) != len(self.excluded_component_ids)
            ):
                raise ValueError("READY artifact component manifests must be duplicate-free")
            if (
                train_set.intersection(eval_set)
                or train_set.intersection(excluded_set)
                or eval_set.intersection(excluded_set)
            ):
                raise ValueError(
                    "READY artifact requires disjoint train/evaluation/excluded components"
                )
            if (
                self.training_window_start is None
                or self.training_window_end is None
                or self.evaluation_window_start is None
                or self.evaluation_window_end is None
            ):
                raise ValueError(
                    "READY artifact requires complete training/evaluation windows"
                )
            if not (
                self.training_window_start
                <= self.training_window_end
                <= self.train_cutoff
                < self.evaluation_window_start
                <= self.evaluation_window_end
            ):
                raise ValueError(
                    "READY artifact requires training_window_end <= train_cutoff < "
                    "evaluation_window_start"
                )
            if self.class_counts.successes < 20 or self.class_counts.errors < 20:
                raise ValueError(
                    "READY artifact requires at least 20 successes and 20 errors"
                )
            if self.group_count < 200:
                raise ValueError("READY artifact requires at least 200 independent groups")
            if self.group_count != len(train_set):
                raise ValueError(
                    "READY artifact group_count must match unique train components"
                )
            metric_keys = (
                "point_ece",
                "ece_ci95_upper",
                "brier_delta_vs_raw",
                "brier_delta_vs_raw_ci95_upper",
                "log_loss_delta_vs_intercept",
                "log_loss_delta_vs_intercept_ci95_upper",
            )
            if any(key not in self.metrics for key in metric_keys):
                raise ValueError("READY artifact metrics must include all gate-critical values")
            point_ece = float(self.metrics["point_ece"])
            ece_ci95_upper = float(self.metrics["ece_ci95_upper"])
            brier_delta_vs_raw = float(self.metrics["brier_delta_vs_raw"])
            brier_delta_vs_raw_ci95_upper = float(
                self.metrics["brier_delta_vs_raw_ci95_upper"]
            )
            log_loss_delta_vs_intercept = float(
                self.metrics["log_loss_delta_vs_intercept"]
            )
            log_loss_delta_vs_intercept_ci95_upper = float(
                self.metrics["log_loss_delta_vs_intercept_ci95_upper"]
            )
            expected_point_gate = point_ece <= 0.10
            expected_ci_gate = ece_ci95_upper <= 0.15
            expected_brier_gate = (
                brier_delta_vs_raw < 0.0 and brier_delta_vs_raw_ci95_upper < 0.0
            )
            expected_logloss_gate = (
                log_loss_delta_vs_intercept <= 0.0
                and log_loss_delta_vs_intercept_ci95_upper <= 0.0
            )
            if self.credibility_gates.point_ece_le_0_10 != expected_point_gate:
                raise ValueError("READY artifact point ECE gate inconsistent with metrics")
            if self.credibility_gates.ci_upper_ece_le_0_15 != expected_ci_gate:
                raise ValueError("READY artifact ECE upper CI gate inconsistent with metrics")
            if self.credibility_gates.brier_improves_over_raw != expected_brier_gate:
                raise ValueError("READY artifact Brier gate inconsistent with metrics")
            if (
                self.credibility_gates.logloss_not_worse_than_intercept
                != expected_logloss_gate
            ):
                raise ValueError("READY artifact logloss gate inconsistent with metrics")
        return self

    def canonical_payload(self, *, include_fingerprint: bool) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_fingerprint:
            payload.pop("artifact_fingerprint", None)
        return payload

    def recompute_fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_payload(include_fingerprint=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class RoutingCalibrationReportV1(CorpusModel):
    schema_version: Literal[1] = 1
    status: str = Field(min_length=1, max_length=80)
    artifact_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    sample_count: int = Field(ge=0)
    representative_sample_count: int = Field(default=0, ge=0)
    challenge_sample_count: int = Field(default=0, ge=0)
    group_count: int = Field(ge=0)
    has_independent_holdout: bool = False
    brier_score: float | None = None
    log_loss: float | None = None
    equal_mass_ece: float | None = None
    equal_mass_ece_bins: int | None = Field(default=None, ge=1, le=10)
    ece_ci95_lower: float | None = None
    ece_ci95_upper: float | None = None
    brier_ci95_lower: float | None = None
    brier_ci95_upper: float | None = None
    log_loss_ci95_lower: float | None = None
    log_loss_ci95_upper: float | None = None
    calibration_intercept: float | None = None
    calibration_slope: float | None = None
    selective_risk_by_coverage: dict[str, float | None] = Field(default_factory=dict)
    raw_brier_score: float | None = None
    raw_log_loss: float | None = None
    intercept_only_log_loss: float | None = None
    brier_delta_vs_raw: float | None = None
    brier_delta_vs_raw_ci95_lower: float | None = None
    brier_delta_vs_raw_ci95_upper: float | None = None
    log_loss_delta_vs_intercept: float | None = None
    log_loss_delta_vs_intercept_ci95_lower: float | None = None
    log_loss_delta_vs_intercept_ci95_upper: float | None = None
    unsafe_event_count: int = Field(default=0, ge=0)
    safety_labeled_plan_count: int = Field(default=0, ge=0)
    unsafe_component_count: int = Field(default=0, ge=0)
    safety_labeled_component_count: int = Field(default=0, ge=0)
    unsafe_zero_event_upper_bound_95: float | None = None
    calibration_intercept_ci95_lower: float | None = None
    calibration_intercept_ci95_upper: float | None = None
    calibration_slope_ci95_lower: float | None = None
    calibration_slope_ci95_upper: float | None = None
    selective_diagnostics_ci95: dict[str, dict[str, float | None]] = Field(
        default_factory=dict
    )
    credibility_gates: RoutingCalibrationCredibilityGatesV1 | None = None

class MemoryRetrievalEvalRecordV1(CorpusModel):
    schema_version: Literal[1] = 1
    example_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=2000)
    expected_source_ids: tuple[str, ...] = Field(min_length=1)
    expected_vector_names: tuple[
        Literal["semantic", "topic", "intent", "decision", "person_context"],
        ...,
    ] = ()
    source_filter: str | None = Field(default=None, max_length=80)


class MemoryRetrievalScoreCardV1(CorpusModel):
    schema_version: Literal[1] = 1
    example_id: str
    query: str
    expected_source_ids: tuple[str, ...]
    predicted_source_ids: tuple[str, ...]
    expected_vector_names: tuple[str, ...] = ()
    observed_vector_names: tuple[str, ...] = ()
    first_relevant_rank: int | None = Field(default=None, ge=1)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    hit_at_k: bool
    provenance_complete: bool
    vector_evidence_complete: bool


class MemoryRetrievalRuntimeConfigV1(CorpusModel):
    schema_version: Literal[1] = 1
    collection: str = Field(min_length=1, max_length=500)
    embedding_model: str = Field(min_length=1, max_length=500)
    query_format_version: str = Field(min_length=1, max_length=100)
    fusion_method: Literal["weighted-rrf-v1"] = "weighted-rrf-v1"
    vector_weights: dict[str, float]
    adaptive_query_weights: bool
    min_score: float = Field(ge=0.0, le=1.0)
    rrf_k: int = Field(ge=1, le=1000)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class MemoryRetrievalMetricsV1(CorpusModel):
    schema_version: Literal[1] = 1
    sample_count: int = Field(ge=0)
    k: int = Field(ge=1, le=100)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    mean_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    provenance_coverage: float = Field(ge=0.0, le=1.0)
    vector_evidence_coverage: float = Field(ge=0.0, le=1.0)
    runtime_config: MemoryRetrievalRuntimeConfigV1 | None = None
    runtime_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_runtime_fingerprint(self) -> MemoryRetrievalMetricsV1:
        if (self.runtime_config is None) != (self.runtime_fingerprint is None):
            raise ValueError("runtime_config i runtime_fingerprint muszą występować razem")
        if (
            self.runtime_config is not None
            and self.runtime_config.fingerprint() != self.runtime_fingerprint
        ):
            raise ValueError("Nieprawidłowy fingerprint konfiguracji retrieval pamięci")
        return self


class StyleProfile(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    profile_id: str
    built_from_manifest_id: str
    utterance_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    preferred_reply_words: tuple[int, int]
    directness: float = Field(ge=0.0, le=1.0)
    grammatical_form: Literal["ty", "mixed", "pan"]
    detail_tolerance: Literal["low", "medium", "high"]
    question_style: Literal["few", "clarifying", "many"]
    maps_to_conversation_style: Literal["default", "concise", "max_iq"]
    enabled: bool = False
    notes: str = Field(default="", max_length=1000)

    def prompt_instruction(self) -> str:
        low, high = self.preferred_reply_words
        return (
            "Prywatna preferencja stylu użytkownika: odpowiadaj po polsku, "
            f"zwykle w zakresie {low}–{high} słów; "
            f"poziom szczegółu={self.detail_tolerance}; "
            f"styl pytań={self.question_style}. "
            "To jest preferencja formy, nie źródło faktów."
        )


class StyleEvaluationReport(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    profile_id: str
    holdout_utterance_count: int = Field(ge=0)
    length_interval_overlap: float = Field(ge=0.0, le=1.0)
    directness_delta: float = Field(ge=0.0, le=1.0)
    grammatical_form_matches: bool
    passes_quality_gate: bool


class MemoryCandidateKind(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"


class CandidateStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class MemoryCandidateCreate(CorpusModel):
    candidate_id: str
    kind: MemoryCandidateKind
    proposed_content: str = Field(min_length=1, max_length=10000)
    content_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    manifest_id: str | None = Field(default=None, max_length=100)
    evidence_utterance_ids: tuple[str, ...] = ()
    status: CandidateStatus = CandidateStatus.PENDING
    block_reason: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> MemoryCandidateCreate:
        if self.status is CandidateStatus.BLOCKED and not self.block_reason:
            raise ValueError("Zablokowany kandydat wymaga powodu.")
        if self.status is CandidateStatus.PENDING and self.block_reason:
            raise ValueError("Oczekujący kandydat nie może mieć block_reason.")
        return self


class MemoryCandidate(MemoryCandidateCreate):
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    decided_at: datetime | None = None
    memory_id: int | None = None


class MemoryCandidateApprovalRequest(CorpusModel):
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorpusRunReport(CorpusModel):
    schema_version: Literal[1] = CORPUS_SCHEMA_VERSION
    manifest_id: str
    extracted_count: int = Field(ge=0)
    clean_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    clean_word_count: int = Field(ge=0)
    technical_excluded_count: int = Field(ge=0)
