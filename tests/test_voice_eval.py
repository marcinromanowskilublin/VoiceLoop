import json
import math
import sqlite3
import struct
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from voiceloop.corpus import cli as corpus_cli
from voiceloop.corpus.cli import build_parser
from voiceloop.corpus.journal import extract_project_journal_candidates
from voiceloop.corpus.pipeline import CorpusPaths
from voiceloop.corpus.proper_names import (
    apply_proper_name_lexicon,
    build_proper_name_lexicon,
)
from voiceloop.corpus.prosody import analyze_prosody
from voiceloop.corpus.reliability import build_action_reliability_report
from voiceloop.corpus.schema import (
    AudioAssetRefV1,
    AudioDirection,
    ExpectedVoiceOutcome,
    ProcessingLocation,
    ProperNameLexiconV1,
    ProvenanceV1,
    SpeakerRole,
    SpeakerStatus,
    UtteranceOrigin,
    UtteranceRecord,
    VoiceEvalPredictionV1,
    VoiceEvalSampleV1,
    VoiceEvalSplit,
    VoiceGoldAnnotationV1,
    VoiceIntentLabel,
    VoiceSampleState,
)
from voiceloop.corpus.storage import (
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from voiceloop.corpus.voice_eval import (
    VoiceEvalError,
    backup_voice_eval_artifacts,
    build_voice_source_manifest,
    detect_speech_segments,
    discover_meeting_input_files,
    discover_screenpipe_audio_files,
    inventory_meeting_audio,
    mark_cross_channel_duplicates,
    merge_voice_candidates,
    parse_meeting_clip_timestamp,
    refill_voice_development_samples,
    select_voice_eval_samples,
    validate_voice_eval_dataset,
)
from voiceloop.corpus.voice_metrics import (
    DeepgramReplayCache,
    VoiceNoSpeechError,
    evaluate_voice_dataset,
    punctuation_f1,
)
from voiceloop.corpus.voice_review import render_voice_annotation_review
from voiceloop.models import TranscriptEnvelopeV1
from voiceloop.screenpipe import ScreenpipeAudioChunk


def _provenance(index: int = 0) -> ProvenanceV1:
    start = datetime(2026, 8, 1, 12, tzinfo=UTC) + timedelta(minutes=index)
    return ProvenanceV1(
        source_system="screenpipe",
        source_id=f"source-{index}",
        source_record_id=f"record-{index}",
        session_id=f"session-{index // 5}",
        captured_start=start,
        captured_end=start + timedelta(seconds=2),
        processing_location=ProcessingLocation.LOCAL,
        audio_direction=AudioDirection.INPUT,
        speaker_role=SpeakerRole.SELF,
        source_sha256=f"{index + 1:064x}",
    )


def _audio(index: int = 0) -> AudioAssetRefV1:
    return AudioAssetRefV1(
        relative_path=f"audio/{index}.wav",
        source_chunk_id=f"chunk-{index}",
        original_sha256="a" * 64,
        clip_sha256=f"{index:064x}"[-64:],
        duration_seconds=2.0,
        source_offset_end_seconds=2.0,
    )


def _sample(index: int) -> VoiceEvalSampleV1:
    return VoiceEvalSampleV1(
        sample_id=f"sample-{index:03}",
        provenance=_provenance(index),
        audio=_audio(index),
        tags=(
            "question",
            "question_intonation_candidate",
            "task",
            "compound",
            "self_correction",
            "cancellation",
            "barge_in",
            "conversation_boundary",
            "proper_name",
            "adverse_audio",
            "non_action",
        ),
    )


def _selected_sample(index: int, split: VoiceEvalSplit) -> VoiceEvalSampleV1:
    return _sample(index).model_copy(
        update={
            "state": VoiceSampleState.SELECTED,
            "split": split,
        }
    )


def _annotation(sample_id: str) -> VoiceGoldAnnotationV1:
    return VoiceGoldAnnotationV1(
        sample_id=sample_id,
        audio_clip_sha256=(
            _audio(int(sample_id.removeprefix("sample-"))).clip_sha256
            if sample_id.startswith("sample-")
            else "a" * 64
        ),
        literal_text="Czy otworzysz VoiceLoop",
        punctuated_text="Czy otworzysz VoiceLoop?",
        intent=VoiceIntentLabel.QUESTION,
        prosody_tags=("question_intonation",),
        proper_names=("VoiceLoop",),
        speaker_role=SpeakerRole.SELF,
        speaker_confirmed=True,
        expected_outcome=ExpectedVoiceOutcome.RESPOND,
        annotator="user",
        approved_at=datetime.now(UTC),
    )


def _write_wav(path: Path, values: list[float], sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(
                struct.pack("<h", max(-32768, min(32767, round(value * 32767))))
                for value in values
            )
        )


def test_voice_manifest_contains_audio_metadata_without_transcript(tmp_path) -> None:
    audio = tmp_path / "chunk.wav"
    _write_wav(audio, [0.0] * 1600)
    start = datetime(2026, 8, 1, 12, tzinfo=UTC)
    manifest = build_voice_source_manifest(
        [
            ScreenpipeAudioChunk(
                chunk_id="chunk",
                file_path=audio,
                device_name="Microphone USB",
                device_type="Input",
                start_time=start.isoformat(),
                end_time=(start + timedelta(seconds=1)).isoformat(),
                text="tajna treść",
            )
        ],
        start=start,
        end=start + timedelta(days=1),
        screenpipe_root=tmp_path,
    )

    payload = manifest.model_dump_json()
    assert manifest.included_source_count == 1
    assert manifest.sources[0].audio_direction is AudioDirection.INPUT
    assert "tajna treść" not in payload


def test_voice_evaluation_defaults_to_development_split() -> None:
    args = build_parser().parse_args(["evaluate-voice-eval"])
    transcribe_args = build_parser().parse_args(["transcribe-voice-eval"])
    memory_args = build_parser().parse_args(["evaluate-memory-retrieval"])
    migration_args = build_parser().parse_args(
        ["build-memory-index-v2", "--confirm", "voiceloop_memory_v2"]
    )

    assert args.split == "development"
    assert args.confirm == ""
    assert transcribe_args.split == "development"
    assert transcribe_args.holdout_confirm == ""
    assert memory_args.k == 5
    assert migration_args.include_screenpipe is False


@pytest.mark.asyncio
async def test_deepgram_replay_caches_no_speech_result(tmp_path) -> None:
    class EmptyTranscriber:
        model = "nova-test"
        language = "pl"

        def __init__(self) -> None:
            self.calls = 0

        async def transcribe_envelope(self, _path):
            self.calls += 1
            return None

    sample = _sample(0)
    audio_path = tmp_path / "audio" / "0.wav"
    _write_wav(audio_path, [0.0] * 1600)
    transcriber = EmptyTranscriber()
    replay = DeepgramReplayCache(
        tmp_path / "cache",
        transcriber=transcriber,  # type: ignore[arg-type]
    )

    with pytest.raises(VoiceNoSpeechError, match="sample-000"):
        await replay.envelope_for(
            sample,
            eval_root=tmp_path,
            allow_remote=True,
            confirmation="DEEPGRAM_AUDIO_UPLOAD",
        )
    with pytest.raises(VoiceNoSpeechError, match="sample-000"):
        await replay.envelope_for(sample, eval_root=tmp_path)

    assert transcriber.calls == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_annotation_review_uses_relative_audio_and_exports_hash_bound_jsonl() -> None:
    envelope = TranscriptEnvelopeV1.from_text(
        "Włącz nasłuch.",
        confidence=0.91,
    )
    content = render_voice_annotation_review(
        [_sample(1)],
        download_filename="annotations-development-v1.jsonl",
        storage_namespace="development",
        prefill_envelopes={"sample-001": envelope},
    )

    assert "audio/1.wav" in content
    assert "Włącz nasłuch." in content
    assert '"deepgram_confidence":0.91' in content
    assert '"speaker_confirmed":true' not in content
    assert _audio(1).clip_sha256 in content
    assert "annotations-development-v1.jsonl" in content
    assert "voiceloop-voice-annotations-v1:development" in content
    assert "C:\\\\" not in content


def test_discovers_unindexed_screenpipe_files_and_prioritizes_input(tmp_path) -> None:
    input_path = tmp_path / "Microphone USB (input)_2026-08-07_19-38-50.mp4"
    output_path = tmp_path / "Speakers (output)_2026-08-07_19-38-51.mp4"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")

    chunks = discover_screenpipe_audio_files(
        tmp_path,
        start=datetime(2026, 8, 7, tzinfo=UTC),
        end=datetime(2026, 8, 8, tzinfo=UTC),
        limit=2,
    )

    assert len(chunks) == 2
    assert chunks[0].device_type == "Input"
    assert chunks[1].device_type == "Output"


def test_meeting_inventory_accepts_only_microphone_wavs(tmp_path) -> None:
    audio_dir = tmp_path / "meeting-20260818T154448Z-abc123" / "audio"
    input_path = audio_dir / "input-microphone-20260818T154448724919Z-a21437bf.wav"
    _write_wav(input_path, [0.2] * 1600)
    _write_wav(
        audio_dir / "output-loopback-20260818T154448724919Z-deadbeef.wav",
        [0.9] * 1600,
    )
    (audio_dir / "input-microphone-invalid.mp4").write_bytes(b"not-a-wav")

    chunks = discover_meeting_input_files(tmp_path)
    manifest = inventory_meeting_audio(tmp_path)

    assert parse_meeting_clip_timestamp("20260818T154448724919Z") == datetime(
        2026,
        8,
        18,
        15,
        44,
        48,
        724919,
        tzinfo=UTC,
    )
    assert len(chunks) == 1
    assert chunks[0].file_path.resolve() == input_path.resolve()
    assert manifest.included_source_count == 1
    assert manifest.sources[0].audio_direction is AudioDirection.INPUT


def test_merge_and_backup_voice_artifacts_are_safe_on_first_run(tmp_path) -> None:
    first = _sample(1)
    updated = _sample(1).model_copy(update={"tags": ("prosody_available",)})
    second = _sample(2)

    assert backup_voice_eval_artifacts(tmp_path) is None
    merged = merge_voice_candidates([first], [updated, second])
    assert [sample.sample_id for sample in merged] == ["sample-001", "sample-002"]
    assert merged[0].tags == ("prosody_available",)

    (tmp_path / "samples-v1.jsonl").write_text("samples\n", encoding="utf-8")
    backup_root = backup_voice_eval_artifacts(tmp_path)
    assert backup_root is not None
    assert (backup_root / "samples-v1.jsonl").read_text(encoding="utf-8") == "samples\n"


@pytest.mark.asyncio
async def test_prepare_meeting_voice_publishes_after_holdout_validation(
    tmp_path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "corpus"
    paths = CorpusPaths(data_root)
    meetings_root = tmp_path / "meetings"
    meeting_audio = (
        meetings_root
        / "meeting-20260818T154448Z-abc123"
        / "audio"
        / "input-microphone-20260818T154448724919Z-a21437bf.wav"
    )
    _write_wav(meeting_audio, [0.2] * 1600)
    development = _selected_sample(0, VoiceEvalSplit.DEVELOPMENT)
    holdout = _selected_sample(1, VoiceEvalSplit.HOLDOUT)
    write_jsonl(paths.voice_candidates, [development, holdout])
    write_jsonl(paths.voice_samples, [development, holdout])
    write_json(
        paths.voice_root / "transcription-report-development-v1.json",
        {"no_speech_sample_ids": [development.sample_id]},
    )

    def fake_build_voice_candidates(_manifest, *, eval_root, source_system, **_kwargs):
        assert source_system == "meeting_recorder"
        clip_path = eval_root / "audio" / "meeting.wav"
        _write_wav(clip_path, [0.3] * 16000)
        candidate = _sample(2)
        return [
            candidate.model_copy(
                update={
                    "provenance": candidate.provenance.model_copy(
                        update={
                            "source_system": "meeting_recorder",
                            "session_id": "meeting:abc123",
                        }
                    ),
                    "audio": candidate.audio.model_copy(
                        update={
                            "relative_path": "audio/meeting.wav",
                            "clip_sha256": sha256_file(clip_path),
                        }
                    ),
                }
            )
        ]

    monkeypatch.setattr(
        corpus_cli,
        "build_voice_candidates",
        fake_build_voice_candidates,
    )
    args = build_parser().parse_args(
        [
            "prepare-meeting-voice",
            "--meetings-root",
            str(meetings_root),
            "--data-root",
            str(data_root),
            "--development-count",
            "1",
            "--confirm",
            "SELF_AUDIO_ONLY",
        ]
    )

    assert await corpus_cli._run_async(args) == 0

    samples = read_jsonl(paths.voice_samples, VoiceEvalSampleV1)
    holdout_ids = {
        sample.sample_id
        for sample in samples
        if sample.split is VoiceEvalSplit.HOLDOUT
    }
    assert holdout_ids == {holdout.sample_id}
    assert any(sample.provenance.source_system == "meeting_recorder" for sample in samples)
    assert (paths.voice_root / "audio" / "meeting.wav").is_file()
    assert paths.voice_meeting_manifest.is_file()


@pytest.mark.asyncio
async def test_prepare_meeting_voice_does_not_publish_when_refill_fails(
    tmp_path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "corpus"
    paths = CorpusPaths(data_root)
    meetings_root = tmp_path / "meetings"
    meeting_audio = (
        meetings_root
        / "meeting-20260818T154448Z-abc123"
        / "audio"
        / "input-microphone-20260818T154448724919Z-a21437bf.wav"
    )
    _write_wav(meeting_audio, [0.2] * 1600)
    existing = _sample(0)
    write_jsonl(paths.voice_candidates, [existing])
    original_candidates = paths.voice_candidates.read_bytes()

    def fake_build_voice_candidates(_manifest, *, eval_root, **_kwargs):
        clip_path = eval_root / "audio" / "meeting.wav"
        _write_wav(clip_path, [0.3] * 16000)
        candidate = _sample(2)
        return [
            candidate.model_copy(
                update={
                    "audio": candidate.audio.model_copy(
                        update={
                            "relative_path": "audio/meeting.wav",
                            "clip_sha256": sha256_file(clip_path),
                        }
                    )
                }
            )
        ]

    monkeypatch.setattr(
        corpus_cli,
        "build_voice_candidates",
        fake_build_voice_candidates,
    )
    args = build_parser().parse_args(
        [
            "prepare-meeting-voice",
            "--meetings-root",
            str(meetings_root),
            "--data-root",
            str(data_root),
            "--confirm",
            "SELF_AUDIO_ONLY",
        ]
    )

    with pytest.raises(VoiceEvalError, match="Brak raportu development"):
        await corpus_cli._run_async(args)

    assert paths.voice_candidates.read_bytes() == original_candidates
    assert not paths.voice_meeting_manifest.exists()
    assert not (paths.voice_root / "audio" / "meeting.wav").exists()


def test_audio_asset_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="względna"):
        AudioAssetRefV1(
            relative_path="../secret.wav",
            source_chunk_id="chunk",
            original_sha256="a" * 64,
            clip_sha256="b" * 64,
            duration_seconds=1.0,
            source_offset_end_seconds=1.0,
        )


def test_gold_annotation_requires_confirmed_self_speaker() -> None:
    payload = _annotation("sample").model_dump()
    payload["speaker_confirmed"] = False

    with pytest.raises(ValueError, match="własnego mówcy"):
        VoiceGoldAnnotationV1.model_validate(payload)


def test_punctuation_f1_penalizes_missing_question_mark() -> None:
    assert punctuation_f1("Czy działa?", "Czy działa?") == 1.0
    assert punctuation_f1("Czy działa?", "Czy działa") == 0.0


@pytest.mark.asyncio
async def test_voice_metrics_include_topk_and_reciprocal_rank(tmp_path) -> None:
    audio_path = tmp_path / "audio" / "0.wav"
    _write_wav(audio_path, [0.0] * 1600)
    sample = _sample(0).model_copy(
        update={
            "audio": _audio(0).model_copy(
                update={"clip_sha256": sha256_file(audio_path)}
            )
        }
    )
    annotation = _annotation(sample.sample_id).model_copy(
        update={
            "audio_clip_sha256": sample.audio.clip_sha256,
            "literal_text": "Otwórz przeglądarkę",
            "punctuated_text": "Otwórz przeglądarkę.",
            "intent": VoiceIntentLabel.TASK,
            "prosody_tags": (),
            "expected_outcome": ExpectedVoiceOutcome.EXECUTE,
            "expected_action_ids": ("open_browser",),
        }
    )

    async def route(_request):
        decision = SimpleNamespace(
            decision=SimpleNamespace(value="resolved"),
            reason=None,
            margin_top2=0.1,
            candidates=(
                SimpleNamespace(action_id="open_url", combined_score=0.8),
                SimpleNamespace(action_id="open_browser", combined_score=0.7),
            ),
        )
        return SimpleNamespace(
            plan=SimpleNamespace(
                steps=(SimpleNamespace(action_id="open_browser", args={}),)
            ),
            decisions=(decision,),
        )

    _, metrics = await evaluate_voice_dataset(
        samples=[sample],
        annotations=[annotation],
        envelopes={
            sample.sample_id: TranscriptEnvelopeV1.from_text(
                "Otwórz przeglądarkę",
                confidence=0.98,
            )
        },
        eval_root=tmp_path,
        run_id="test-run",
        route=route,
        required_sample_count=1,
    )

    assert metrics.routing_exact_plan_accuracy == 1.0
    assert metrics.routing_topk_recall == 1.0
    assert metrics.routing_mean_reciprocal_rank == 0.5
    assert metrics.quality_gate_passed is True


def test_speech_segmentation_splits_on_long_silence(tmp_path) -> None:
    sample_rate = 16000
    tone = [
        0.4 * math.sin(2 * math.pi * 180 * index / sample_rate)
        for index in range(sample_rate)
    ]
    path = tmp_path / "speech.wav"
    _write_wav(path, tone + [0.0] * sample_rate + tone)

    segments = detect_speech_segments(path)

    assert len(segments) == 2
    assert segments[0][1] <= segments[1][0]


def test_deduplication_marks_identical_clip_hash() -> None:
    first = _sample(1)
    second = _sample(2).model_copy(
        update={
            "audio": _audio(2).model_copy(
                update={"clip_sha256": first.audio.clip_sha256}
            )
        }
    )

    deduplicated = mark_cross_channel_duplicates([first, second])

    duplicate = next(item for item in deduplicated if item.provenance.duplicate_of)
    assert duplicate.duplicate_type == "exact_audio_hash"
    assert duplicate.dedupe_confidence == 1.0


def test_selects_exact_30_90_split_without_group_leakage() -> None:
    samples = select_voice_eval_samples(
        [_sample(index) for index in range(120)],
        target=120,
        development_count=30,
    )

    assert len(samples) == 120
    assert sum(item.split is VoiceEvalSplit.DEVELOPMENT for item in samples) == 30
    assert sum(item.split is VoiceEvalSplit.HOLDOUT for item in samples) == 90
    assert all(item.state is VoiceSampleState.SELECTED for item in samples)


def test_split_keeps_segments_from_same_source_hash_together() -> None:
    candidates: list[VoiceEvalSampleV1] = []
    for index in range(120):
        source_group = index // 2
        candidates.append(
            _sample(index).model_copy(
                update={
                    "provenance": _provenance(index).model_copy(
                        update={"source_sha256": f"{source_group:064x}"}
                    )
                }
            )
        )

    samples = select_voice_eval_samples(candidates)
    splits_by_source: dict[str, set[VoiceEvalSplit]] = {}
    for sample in samples:
        splits_by_source.setdefault(sample.provenance.source_sha256, set()).add(
            sample.split
        )

    assert all(len(splits) == 1 for splits in splits_by_source.values())


def test_refill_replaces_no_speech_without_changing_holdout() -> None:
    candidates = [_sample(index) for index in range(125)]
    selected = select_voice_eval_samples(
        candidates[:120],
        target=120,
        development_count=30,
    )
    rejected_ids = {
        sample.sample_id
        for sample in selected
        if sample.split is VoiceEvalSplit.DEVELOPMENT
    }
    rejected_ids = set(sorted(rejected_ids)[:3])
    original_holdout_ids = {
        sample.sample_id
        for sample in selected
        if sample.split is VoiceEvalSplit.HOLDOUT
    }

    refilled = refill_voice_development_samples(
        candidates,
        selected,
        rejected_sample_ids=rejected_ids,
        development_count=30,
    )

    development_ids = {
        sample.sample_id
        for sample in refilled
        if sample.split is VoiceEvalSplit.DEVELOPMENT
    }
    holdout_ids = {
        sample.sample_id
        for sample in refilled
        if sample.split is VoiceEvalSplit.HOLDOUT
    }
    replacement_ids = {f"sample-{index:03d}" for index in range(120, 125)}
    assert len(refilled) == 120
    assert len(development_ids) == 30
    assert development_ids.isdisjoint(rejected_ids)
    assert holdout_ids == original_holdout_ids
    assert development_ids & replacement_ids


def test_selection_keeps_adverse_audio_as_controlled_minority() -> None:
    candidates: list[VoiceEvalSampleV1] = []
    for index in range(180):
        tags = ("adverse_audio",) if index < 60 else ("prosody_available",)
        duration = 0.6 if index % 3 == 0 else (2.5 if index % 3 == 1 else 1.2)
        candidates.append(
            _sample(index).model_copy(
                update={
                    "tags": tags,
                    "audio": _audio(index).model_copy(
                        update={"duration_seconds": duration}
                    ),
                }
            )
        )

    selected = select_voice_eval_samples(candidates)

    assert sum("adverse_audio" in item.tags for item in selected) == 15
    assert (
        sum(
            "adverse_audio" in item.tags
            for item in selected
            if item.split is VoiceEvalSplit.DEVELOPMENT
        )
        == 4
    )
    assert sum(item.audio.duration_seconds < 0.8 for item in selected if item.audio) >= 15
    assert sum(item.audio.duration_seconds >= 2.0 for item in selected if item.audio) >= 30


def test_dataset_validation_checks_annotations_and_tag_coverage() -> None:
    samples = select_voice_eval_samples(
        [_sample(index) for index in range(120)],
        target=120,
        development_count=30,
    )
    annotations = [_annotation(sample.sample_id) for sample in samples]

    report = validate_voice_eval_dataset(samples, annotations)

    assert report["valid"] is True
    assert report["tag_deficits"] == {}
    assert "temporal_concentration_over_50_percent" in report["warnings"]


def test_dataset_validation_binds_annotation_to_audio_hash() -> None:
    samples = select_voice_eval_samples(
        [_sample(index) for index in range(120)],
        target=120,
        development_count=30,
    )
    annotations = [_annotation(sample.sample_id) for sample in samples]
    annotations[0] = annotations[0].model_copy(
        update={"audio_clip_sha256": "f" * 64}
    )

    report = validate_voice_eval_dataset(samples, annotations)

    assert report["valid"] is False
    assert report["annotation_audio_mismatch_ids"] == [annotations[0].sample_id]


def test_prosody_extracts_rising_pitch(tmp_path) -> None:
    sample_rate = 16000
    duration = 2.0
    phase = 0.0
    values: list[float] = []
    for index in range(round(sample_rate * duration)):
        frequency = 150.0 + 70.0 * (index / (sample_rate * duration))
        phase += 2 * math.pi * frequency / sample_rate
        values.append(0.4 * math.sin(phase))
    path = tmp_path / "audio" / "0.wav"
    _write_wav(path, values)
    sample = _sample(0).model_copy(
        update={
            "audio": _audio(0).model_copy(
                update={
                    "clip_sha256": sha256_file(path),
                    "duration_seconds": duration,
                }
            )
        }
    )

    features = analyze_prosody(sample, eval_root=tmp_path, word_count=4)

    assert features.available is True
    assert features.final_f0_delta_semitones is not None
    assert features.final_f0_delta_semitones > 0


def test_proper_name_lexicon_requires_approval_before_replacement() -> None:
    annotation = _annotation("sample")
    prediction = VoiceEvalPredictionV1(
        run_id="run",
        sample_id="sample",
        transcript_text="Czy otworzysz Voice Look",
    )
    lexicon = build_proper_name_lexicon([annotation], [prediction])
    entry = lexicon.entries[0].model_copy(
        update={"common_stt_errors": ("Voice Look",), "approved": True}
    )

    corrected, changes = apply_proper_name_lexicon(
        prediction.transcript_text,
        ProperNameLexiconV1(entries=(entry,)),
    )

    assert corrected == "Czy otworzysz VoiceLoop"
    assert changes[0]["before"] == "Voice Look"


def test_journal_excludes_questions_and_keeps_explicit_decisions() -> None:
    base = {
        "source_id": "source",
        "origin": UtteranceOrigin.CURSOR_USER,
        "session_id": "session",
        "captured_at": datetime.now(UTC),
        "speaker_status": SpeakerStatus.SELF,
    }
    question_text = "Czy problemem jest routing?"
    decision_text = "Ustalamy, że Routing V2 pozostaje fail-closed."
    records = [
        UtteranceRecord(
            utterance_id="question",
            word_count=len(question_text.split()),
            char_count=len(question_text),
            text_sha256=sha256_text(question_text.casefold()),
            text=question_text,
            **base,
        ),
        UtteranceRecord(
            utterance_id="decision",
            word_count=len(decision_text.split()),
            char_count=len(decision_text),
            text_sha256=sha256_text(decision_text.casefold()),
            text=decision_text,
            **base,
        ),
    ]

    candidates = extract_project_journal_candidates(records)

    assert [item.evidence_utterance_ids for item in candidates] == [("decision",)]


def test_reliability_report_reconciles_statuses_without_exposing_text(tmp_path) -> None:
    database = tmp_path / "voice.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE commands (
                request_id TEXT PRIMARY KEY, source TEXT, input_text TEXT, status TEXT,
                intent TEXT, response_text TEXT, provider TEXT, model TEXT, error TEXT,
                plan_json TEXT, results_json TEXT, created_at TEXT, updated_at TEXT
            )
            """
        )
        now = datetime.now(UTC)
        plan = {
            "steps": [
                {
                    "action_id": "open_browser",
                    "confirmation_required": False,
                }
            ]
        }
        results = [{"action_id": "open_browser", "success": True}]
        connection.execute(
            "INSERT INTO commands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "request",
                "deepgram",
                "prywatna treść",
                "succeeded",
                "open_browser",
                None,
                "local",
                "model",
                None,
                json.dumps(plan),
                json.dumps(results),
                now.isoformat(),
                (now + timedelta(seconds=1)).isoformat(),
            ),
        )

    report = build_action_reliability_report(database)

    assert report["reconciled"] is True
    assert report["actions"]["open_browser"]["result_coverage"] == 1.0
    assert report["actions"]["open_browser"]["observed_success_rate"] == 1.0
    assert report["status_by_source"]["deepgram"] == {"succeeded": 1}
    assert report["status_by_program"]["browser"] == {"succeeded": 1}
    assert report["action_result_summary"]["result_coverage"] == 1.0
    assert report["action_result_summary"]["observed_success_rate"] == 1.0
    assert "prywatna treść" not in json.dumps(report, ensure_ascii=False)
