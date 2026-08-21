from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..actions import ActionRegistry
from ..behavior_digest import LocalBehaviorDigestClient
from ..capability_index import CapabilityIndex, CapabilityIndexError
from ..embeddings import (
    EmbeddingUnavailableError,
    OpenAICompatibleEmbeddingClient,
)
from ..memory import MemoryStore
from ..memory_vectorization import (
    MEMORY_QUERY_DOCUMENTS_VERSION,
    MEMORY_VECTOR_NAMES,
    memory_query_documents,
    memory_query_weights,
)
from ..models import TranscriptEnvelopeV1
from ..qdrant_memory import WEIGHTED_RRF_VERSION, QdrantVectorStore
from ..routing.service import RoutingV2Service
from ..screenpipe import ScreenpipeClient, ScreenpipeError
from ..screenpipe_deepgram import DeepgramFileError, DeepgramFileTranscriber
from ..screenpipe_memory import ScreenpipeVectorMemoryWorker
from ..settings import Settings
from ..tts import WindowsTTS
from .analysis import build_routing_eval_records
from .candidates import CandidateDecisionError, MemoryCandidateStore
from .inventory import build_manifest
from .journal import (
    approved_journal_entries,
    decide_journal_candidate,
    extract_project_journal_candidates,
)
from .local_only import LocalOnlyViolation, require_loopback_url
from .memory_eval import evaluate_memory_retrieval
from .pipeline import (
    CorpusPaths,
    corpus_scope_id,
    evaluate_routing,
    evaluate_routing_v2,
    run_pipeline,
)
from .proper_names import build_proper_name_lexicon
from .reliability import (
    ReliabilityReportError,
    build_action_reliability_report,
    render_action_reliability_report,
)
from .routing_calibration import (
    evaluate_routing_calibration,
    fit_routing_calibration,
    load_calibration_observations,
    validate_calibration_artifact_for_evaluation,
    validate_calibration_observations,
)
from .schema import (
    CandidateStatus,
    MemoryRetrievalEvalRecordV1,
    MemoryRetrievalRuntimeConfigV1,
    ProjectJournalCandidateV1,
    ProperNameLexiconV1,
    RoutingCalibrationArtifactStatus,
    RoutingCalibrationArtifactV1,
    RoutingCalibrationSetRole,
    RoutingEvalRecord,
    SourceKind,
    SourceManifest,
    SpeakerDecision,
    SpeakerDecisionFile,
    SpeakerRole,
    UtteranceRecord,
    VoiceEvalPredictionV1,
    VoiceEvalRunManifestV1,
    VoiceEvalSampleV1,
    VoiceGoldAnnotationV1,
    VoiceSourceManifestV1,
)
from .storage import (
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
    write_text,
)
from .voice_eval import (
    VoiceEvalError,
    backup_voice_eval_artifacts,
    build_voice_candidates,
    inventory_meeting_audio,
    inventory_screenpipe_audio,
    merge_voice_candidates,
    refill_voice_development_samples,
    select_voice_eval_samples,
    tag_voice_candidate_quality,
    validate_voice_eval_dataset,
)
from .voice_metrics import (
    DeepgramReplayCache,
    VoiceNoSpeechError,
    evaluate_voice_dataset,
    render_voice_report,
)
from .voice_review import render_voice_annotation_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voiceloop.corpus",
        description="Lokalny, offline pipeline prywatnego korpusu VoiceLoop.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser(
        "inventory",
        help="Zapisz manifest bez kopiowania treści.",
    )
    _source_arguments(inventory_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Uruchom deduplikację, bramki, eval, styl i kolejkę pamięci.",
    )
    _source_arguments(run_parser)
    run_parser.add_argument("--enable-style", action="store_true")
    run_parser.add_argument("--holdout-percent", type=int, default=20)

    status_parser = subparsers.add_parser(
        "source-status",
        help="Pokaż hashe i stan decyzji mówcy bez treści źródeł.",
    )
    status_parser.add_argument("--data-root", type=Path)

    approve_source_parser = subparsers.add_parser(
        "approve-audio-source",
        help="Jawnie oznacz konkretny hash źródła audio jako własny.",
    )
    approve_source_parser.add_argument("source_id")
    approve_source_parser.add_argument("--data-root", type=Path)
    approve_source_parser.add_argument("--confirm", required=True)

    revoke_source_parser = subparsers.add_parser(
        "revoke-audio-source",
        help="Cofnij decyzję mówcy dla źródła audio.",
    )
    revoke_source_parser.add_argument("source_id")
    revoke_source_parser.add_argument("--data-root", type=Path)

    eval_parser = subparsers.add_parser(
        "evaluate-routing",
        help="Oceń top-1/top-k i margines na lokalnym indeksie możliwości.",
    )
    eval_parser.add_argument("--data-root", type=Path)
    eval_parser.add_argument("--margin-threshold", type=float)
    eval_parser.add_argument(
        "--capability-collection",
        help="Oceń wskazaną wersjonowaną kolekcję Qdrant bez zmiany aktywnej konfiguracji.",
    )

    eval_v2_parser = subparsers.add_parser(
        "evaluate-routing-v2",
        help="Oceń segmentację, resolver, argumenty i bezpieczną abstencję Routing V2.",
    )
    eval_v2_parser.add_argument("--data-root", type=Path)
    eval_v2_parser.add_argument(
        "--capability-collection",
        help="Oceń wskazaną wersjonowaną kolekcję Qdrant bez zmiany aktywnej konfiguracji.",
    )

    personal_routing_parser = subparsers.add_parser(
        "evaluate-routing-personal",
        help="Oceń Routing V2 na odseparowanym holdoucie rzeczywistych polskich wzorców.",
    )
    personal_routing_parser.add_argument("--data-root", type=Path)
    personal_routing_parser.add_argument(
        "--capability-collection",
        help="Oceń wskazaną wersjonowaną kolekcję Qdrant bez zmiany aktywnej konfiguracji.",
    )

    calibration_validate_parser = subparsers.add_parser(
        "validate-routing-calibration",
        help="Sprawdź integralność i readiness lokalnego zbioru kalibracyjnego.",
    )
    calibration_validate_parser.add_argument("--data-root", type=Path)
    calibration_validate_parser.add_argument("--dataset", type=Path)
    calibration_validate_parser.add_argument("--train-cutoff")
    calibration_validate_parser.add_argument("--for-training", action="store_true")

    calibration_fit_parser = subparsers.add_parser(
        "fit-routing-calibration",
        help=(
            "Dopasuj monotoniczny Platt v1 i zapisz artefakt JSON. "
            "Personal holdout jest zabroniony."
        ),
    )
    calibration_fit_parser.add_argument("--data-root", type=Path)
    calibration_fit_parser.add_argument("--dataset", type=Path)
    calibration_fit_parser.add_argument("--artifact", type=Path)
    calibration_fit_parser.add_argument("--report", type=Path)
    calibration_fit_parser.add_argument("--train-cutoff")
    calibration_fit_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    calibration_fit_parser.add_argument("--bootstrap-seed", type=int, default=17)

    calibration_eval_parser = subparsers.add_parser(
        "evaluate-routing-calibration",
        help="Oceń artefakt kalibracji na wskazanym zbiorze i policz bootstrap CI.",
    )
    calibration_eval_parser.add_argument("--data-root", type=Path)
    calibration_eval_parser.add_argument("--dataset", type=Path)
    calibration_eval_parser.add_argument("--artifact", type=Path)
    calibration_eval_parser.add_argument("--report", type=Path)
    calibration_eval_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    calibration_eval_parser.add_argument("--bootstrap-seed", type=int, default=17)

    memory_eval_parser = subparsers.add_parser(
        "evaluate-memory-retrieval",
        help="Oceń lokalną pamięć pięciowektorową na ręcznym zestawie gold.",
    )
    memory_eval_parser.add_argument("--data-root", type=Path)
    memory_eval_parser.add_argument("--gold", type=Path)
    memory_eval_parser.add_argument("--collection")
    memory_eval_parser.add_argument("--k", type=int, default=5)

    memory_migration_parser = subparsers.add_parser(
        "build-memory-index-v2",
        help="Zbuduj odwracalną kolekcję pamięci V2 bez przełączania aktywnej kolekcji.",
    )
    memory_migration_parser.add_argument("--data-root", type=Path)
    memory_migration_parser.add_argument("--database", type=Path)
    memory_migration_parser.add_argument("--collection")
    memory_migration_parser.add_argument("--confirm", required=True)
    memory_migration_parser.add_argument(
        "--include-screenpipe",
        action="store_true",
        help="Po migracji semantic-only przetwórz również aktualne dane Screenpipe.",
    )

    list_parser = subparsers.add_parser(
        "list-candidates",
        help="Pokaż wyłącznie identyfikatory i statusy kandydatów.",
    )
    list_parser.add_argument("--data-root", type=Path)
    list_parser.add_argument(
        "--status",
        choices=[item.value for item in CandidateStatus],
        default=CandidateStatus.PENDING.value,
    )

    show_parser = subparsers.add_parser(
        "show-candidate",
        help="Pokaż treść jednego kandydata przed decyzją.",
    )
    show_parser.add_argument("candidate_id")
    show_parser.add_argument("--data-root", type=Path)

    approve_parser = subparsers.add_parser(
        "approve",
        help="Jawnie zatwierdź pojedynczego kandydata pamięci.",
    )
    approve_parser.add_argument("candidate_id")
    approve_parser.add_argument("--data-root", type=Path)
    approve_parser.add_argument("--content-sha256", required=True)
    approve_parser.add_argument(
        "--confirm",
        required=True,
        help="Musi być identyczne z candidate_id.",
    )

    reject_parser = subparsers.add_parser(
        "reject",
        help="Odrzuć pojedynczego kandydata pamięci.",
    )
    reject_parser.add_argument("candidate_id")
    reject_parser.add_argument("--data-root", type=Path)

    voice_inventory = subparsers.add_parser(
        "inventory-voice-eval",
        help="Zapisz lokalny manifest audio Screenpipe bez kopiowania treści.",
    )
    voice_inventory.add_argument("--start", required=True)
    voice_inventory.add_argument("--end", required=True)
    voice_inventory.add_argument("--screenpipe-root", type=Path)
    voice_inventory.add_argument("--max-results", type=int, default=2000)
    voice_inventory.add_argument("--data-root", type=Path)

    voice_candidates = subparsers.add_parser(
        "build-voice-candidates",
        help="Wytnij lokalne fragmenty mowy z zatwierdzonego kanału użytkownika.",
    )
    voice_candidates.add_argument("--screenpipe-root", type=Path)
    voice_candidates.add_argument("--max-candidates", type=int, default=360)
    voice_candidates.add_argument("--ffmpeg", default="ffmpeg")
    voice_candidates.add_argument("--confirm", required=True)
    voice_candidates.add_argument("--data-root", type=Path)

    voice_select = subparsers.add_parser(
        "select-voice-eval",
        help="Wybierz zamrożony zestaw głosowy i utwórz szablon adnotacji.",
    )
    voice_select.add_argument("--target", type=int, default=120)
    voice_select.add_argument("--development-count", type=int, default=30)
    voice_select.add_argument("--data-root", type=Path)

    voice_refill = subparsers.add_parser(
        "refill-voice-development",
        help="Zastąp lokalnie klipy bez mowy, nie otwierając ani nie zmieniając holdoutu.",
    )
    voice_refill.add_argument("--development-count", type=int, default=30)
    voice_refill.add_argument("--data-root", type=Path)

    voice_prepare_meetings = subparsers.add_parser(
        "prepare-meeting-voice",
        help="Dodaj własne nagrania mikrofonowe ze spotkań do development eval.",
    )
    voice_prepare_meetings.add_argument("--meetings-root", type=Path)
    voice_prepare_meetings.add_argument("--per-session-limit", type=int, default=80)
    voice_prepare_meetings.add_argument("--max-candidates", type=int, default=120)
    voice_prepare_meetings.add_argument("--development-count", type=int, default=30)
    voice_prepare_meetings.add_argument("--ffmpeg", default="ffmpeg")
    voice_prepare_meetings.add_argument("--confirm", required=True)
    voice_prepare_meetings.add_argument("--skip-refill", action="store_true")
    voice_prepare_meetings.add_argument("--data-root", type=Path)

    voice_validate = subparsers.add_parser(
        "validate-voice-eval",
        help="Sprawdź kompletność audio, splitów, adnotacji i pokrycia tagów.",
    )
    voice_validate.add_argument("--target", type=int, default=120)
    voice_validate.add_argument("--development-count", type=int, default=30)
    voice_validate.add_argument("--data-root", type=Path)

    voice_transcribe = subparsers.add_parser(
        "transcribe-voice-eval",
        help="Odtwórz Deepgram dla zamrożonego audio z cache po hashu.",
    )
    voice_transcribe.add_argument("--allow-remote", action="store_true")
    voice_transcribe.add_argument("--confirm", default="")
    voice_transcribe.add_argument(
        "--split",
        choices=("development", "holdout", "all"),
        default="development",
    )
    voice_transcribe.add_argument("--holdout-confirm", default="")
    voice_transcribe.add_argument("--data-root", type=Path)

    voice_evaluate = subparsers.add_parser(
        "evaluate-voice-eval",
        help="Porównaj tekst, prozodię, semantykę i pełny plan routingu.",
    )
    voice_evaluate.add_argument("--run-id")
    voice_evaluate.add_argument("--without-routing", action="store_true")
    voice_evaluate.add_argument(
        "--split",
        choices=("development", "holdout", "all"),
        default="development",
    )
    voice_evaluate.add_argument("--confirm", default="")
    voice_evaluate.add_argument("--data-root", type=Path)

    proper_names = subparsers.add_parser(
        "build-proper-names",
        help="Zbuduj kontrolowany słownik nazw z ręcznych adnotacji i błędów STT.",
    )
    proper_names.add_argument("--run-id")
    proper_names.add_argument("--data-root", type=Path)

    reliability = subparsers.add_parser(
        "report-actions",
        help="Zbuduj read-only raport niezawodności działań.",
    )
    reliability.add_argument("--database", type=Path)
    reliability.add_argument("--start")
    reliability.add_argument("--end")
    reliability.add_argument("--data-root", type=Path)

    journal_extract = subparsers.add_parser(
        "extract-project-journal",
        help="Utwórz kandydatów dziennika wyłącznie z czystego kanału użytkownika.",
    )
    journal_extract.add_argument("--data-root", type=Path)

    journal_list = subparsers.add_parser(
        "list-journal-candidates",
        help="Pokaż kandydatów dziennika bez automatycznej akceptacji.",
    )
    journal_list.add_argument("--data-root", type=Path)

    for command, help_text in (
        ("approve-journal", "Ręcznie zatwierdź wpis dziennika."),
        ("reject-journal", "Ręcznie odrzuć wpis dziennika."),
    ):
        journal_decision = subparsers.add_parser(command, help=help_text)
        journal_decision.add_argument("candidate_id")
        journal_decision.add_argument("--confirm", required=True)
        journal_decision.add_argument("--data-root", type=Path)

    local_parser = subparsers.add_parser(
        "validate-local-url",
        help="Sprawdź, czy endpoint wskazuje wyłącznie loopback.",
    )
    local_parser.add_argument("url")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-local-url":
            require_loopback_url(args.url)
            print(json.dumps({"local_only": True}))
            return 0
        return asyncio.run(_run_async(args))
    except (
        CandidateDecisionError,
        CapabilityIndexError,
        EmbeddingUnavailableError,
        DeepgramFileError,
        LocalOnlyViolation,
        ReliabilityReportError,
        ScreenpipeError,
        VoiceEvalError,
        ValueError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2


async def _run_async(args: argparse.Namespace) -> int:
    settings = Settings()
    root = (args.data_root or settings.data_dir / "corpus").resolve()
    paths = CorpusPaths(root)
    if args.command == "inventory-voice-eval":
        require_loopback_url(settings.screenpipe_base_url)
        start = _parse_cli_datetime(args.start)
        end = _parse_cli_datetime(args.end)
        screenpipe_root = (
            args.screenpipe_root or Path.home() / ".screenpipe" / "data"
        ).resolve()
        manifest = await inventory_screenpipe_audio(
            ScreenpipeClient(settings),
            start=start,
            end=end,
            screenpipe_root=screenpipe_root,
            max_results=args.max_results,
        )
        write_json(paths.voice_manifest, manifest)
        print(
            json.dumps(
                {
                    "manifest_id": manifest.manifest_id,
                    "included_sources": manifest.included_source_count,
                    "excluded_sources": manifest.excluded_source_count,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "build-voice-candidates":
        if args.confirm != "SELF_AUDIO_ONLY":
            raise VoiceEvalError(
                "--confirm musi mieć wartość SELF_AUDIO_ONLY po sprawdzeniu kanału."
            )
        if not paths.voice_manifest.is_file():
            raise VoiceEvalError("Najpierw uruchom inventory-voice-eval.")
        manifest = VoiceSourceManifestV1.model_validate_json(
            paths.voice_manifest.read_text(encoding="utf-8")
        )
        screenpipe_root = (
            args.screenpipe_root or Path.home() / ".screenpipe" / "data"
        ).resolve()
        candidates = await asyncio.to_thread(
            build_voice_candidates,
            manifest,
            eval_root=paths.voice_root,
            screenpipe_root=screenpipe_root,
            speaker_role=SpeakerRole.SELF,
            max_candidates=args.max_candidates,
            ffmpeg_executable=args.ffmpeg,
        )
        write_jsonl(paths.voice_candidates, candidates)
        print(
            json.dumps(
                {
                    "candidate_count": len(candidates),
                    "audio_root": str(paths.voice_root / "audio"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "select-voice-eval":
        candidates = read_jsonl(paths.voice_candidates, VoiceEvalSampleV1)
        candidates = tag_voice_candidate_quality(
            candidates,
            eval_root=paths.voice_root,
        )
        write_jsonl(paths.voice_candidates, candidates)
        samples = select_voice_eval_samples(
            candidates,
            target=args.target,
            development_count=args.development_count,
        )
        _write_voice_selection_artifacts(paths, samples)
        print(
            json.dumps(
                {
                    "sample_count": len(samples),
                    "development_count": args.development_count,
                    "holdout_count": len(samples) - args.development_count,
                    "annotation_template": str(paths.voice_annotation_template),
                    "annotation_review": str(paths.voice_annotation_review),
                    "development_review": str(
                        paths.voice_annotation_review_development
                    ),
                    "holdout_review": str(paths.voice_annotation_review_holdout),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "refill-voice-development":
        candidates = read_jsonl(paths.voice_candidates, VoiceEvalSampleV1)
        samples, envelopes, replacement_ids = _prepare_voice_development_refill(
            paths,
            candidates,
            development_count=args.development_count,
        )
        _write_voice_selection_artifacts(
            paths,
            samples,
            prefill_envelopes=envelopes,
        )
        print(
            json.dumps(
                {
                    "development_count": args.development_count,
                    "preserved_transcribed_count": (
                        args.development_count - len(replacement_ids)
                    ),
                    "replacement_count": len(replacement_ids),
                    "replacement_sample_ids": replacement_ids,
                    "holdout_unchanged": True,
                    "development_review": str(
                        paths.voice_annotation_review_development
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "prepare-meeting-voice":
        if args.confirm != "SELF_AUDIO_ONLY":
            raise VoiceEvalError(
                "--confirm musi mieć wartość SELF_AUDIO_ONLY po sprawdzeniu kanału."
            )
        meetings_root = (
            args.meetings_root or settings.data_dir / "meetings"
        ).resolve()
        manifest = inventory_meeting_audio(
            meetings_root,
            per_session_limit=args.per_session_limit,
        )
        existing_candidates = read_jsonl(
            paths.voice_candidates,
            VoiceEvalSampleV1,
        )
        with tempfile.TemporaryDirectory(prefix="voiceloop-meeting-voice-") as temp_dir:
            staging_root = Path(temp_dir)
            new_candidates = await asyncio.to_thread(
                build_voice_candidates,
                manifest,
                eval_root=staging_root,
                screenpipe_root=meetings_root,
                speaker_role=SpeakerRole.SELF,
                max_candidates=args.max_candidates,
                ffmpeg_executable=args.ffmpeg,
                source_system="meeting_recorder",
            )
            candidates = merge_voice_candidates(existing_candidates, new_candidates)
            samples: list[VoiceEvalSampleV1] | None = None
            envelopes: dict[str, TranscriptEnvelopeV1] = {}
            replacement_ids: list[str] = []
            if not args.skip_refill:
                samples, envelopes, replacement_ids = (
                    _prepare_voice_development_refill(
                        paths,
                        candidates,
                        development_count=args.development_count,
                    )
                )

            backup_root = backup_voice_eval_artifacts(paths.voice_root)
            _publish_voice_candidate_audio(
                new_candidates,
                source_root=staging_root,
                destination_root=paths.voice_root,
            )
            write_json(paths.voice_meeting_manifest, manifest)
            write_jsonl(paths.voice_candidates, candidates)
            if samples is not None:
                _write_voice_selection_artifacts(
                    paths,
                    samples,
                    prefill_envelopes=envelopes,
                )

        print(
            json.dumps(
                {
                    "backup_root": str(backup_root) if backup_root else None,
                    "included_sources": manifest.included_source_count,
                    "excluded_sources": manifest.excluded_source_count,
                    "new_candidate_count": len(new_candidates),
                    "candidate_count": len(candidates),
                    "replacement_count": len(replacement_ids),
                    "replacement_sample_ids": replacement_ids,
                    "holdout_checked": not args.skip_refill,
                    "holdout_unchanged": True if not args.skip_refill else None,
                    "development_review": (
                        str(paths.voice_annotation_review_development)
                        if samples is not None
                        else None
                    ),
                    "meeting_manifest": str(paths.voice_meeting_manifest),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "validate-voice-eval":
        samples = read_jsonl(paths.voice_samples, VoiceEvalSampleV1)
        annotations = _load_voice_annotations(paths, split="all")
        report = validate_voice_eval_dataset(
            samples,
            annotations,
            target=args.target,
            development_count=args.development_count,
        )
        write_json(paths.voice_validation, report)
        print(
            json.dumps(
                {
                    "valid": report["valid"],
                    "sample_count": report["sample_count"],
                    "development_count": report["development_count"],
                    "holdout_count": report["holdout_count"],
                    "annotation_count": report["annotation_count"],
                    "missing_annotation_count": len(report["missing_annotation_ids"]),
                    "tag_deficits": report["tag_deficits"],
                    "date_counts": report["date_counts"],
                    "errors": report["errors"],
                    "warnings": report["warnings"],
                    "full_report": str(paths.voice_validation),
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["valid"] else 3
    if args.command == "transcribe-voice-eval":
        samples = read_jsonl(paths.voice_samples, VoiceEvalSampleV1)
        if args.split in {"holdout", "all"}:
            if args.holdout_confirm != "HOLDOUT_FINAL_EVALUATION":
                raise VoiceEvalError(
                    "Replay holdoutu wymaga --holdout-confirm "
                    "HOLDOUT_FINAL_EVALUATION."
                )
        if args.split != "all":
            samples = [
                sample
                for sample in samples
                if sample.split is not None and sample.split.value == args.split
            ]
        transcriber = DeepgramFileTranscriber(settings)
        replay = DeepgramReplayCache(paths.voice_cache, transcriber=transcriber)
        envelopes = _load_voice_envelopes(
            paths.voice_transcripts,
            required=False,
        )
        no_speech_ids: list[str] = []
        for sample in samples:
            try:
                envelope = await replay.envelope_for(
                    sample,
                    eval_root=paths.voice_root,
                    allow_remote=args.allow_remote,
                    confirmation=args.confirm,
                )
            except VoiceNoSpeechError:
                no_speech_ids.append(sample.sample_id)
                continue
            envelopes[sample.sample_id] = envelope
        transcript_rows: list[dict] = [
            (
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "envelope": envelope.model_dump(mode="json"),
                }
            )
            for sample_id, envelope in sorted(envelopes.items())
        ]
        write_jsonl(paths.voice_transcripts, transcript_rows)
        transcribed_samples = [
            sample for sample in samples if sample.sample_id in envelopes
        ]
        review_path = {
            "development": paths.voice_annotation_review_development,
            "holdout": paths.voice_annotation_review_holdout,
            "all": paths.voice_annotation_review,
        }[args.split]
        annotation_filename = {
            "development": paths.voice_annotations_development.name,
            "holdout": paths.voice_annotations_holdout.name,
            "all": paths.voice_annotations.name,
        }[args.split]
        write_text(
            review_path,
            render_voice_annotation_review(
                transcribed_samples,
                download_filename=annotation_filename,
                storage_namespace=args.split,
                prefill_envelopes=envelopes,
            ),
        )
        report_path = paths.voice_root / f"transcription-report-{args.split}-v1.json"
        report = {
            "schema_version": 1,
            "requested_in_split": len(samples),
            "transcribed_in_split": len(transcribed_samples),
            "no_speech_count": len(no_speech_ids),
            "no_speech_sample_ids": no_speech_ids,
            "cached_total": len(transcript_rows),
            "split": args.split,
            "review": str(review_path),
        }
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    **report,
                    "report": str(report_path),
                },
                ensure_ascii=False,
            )
        )
        return 3 if no_speech_ids else 0
    if args.command == "evaluate-voice-eval":
        samples = read_jsonl(paths.voice_samples, VoiceEvalSampleV1)
        annotations = _load_voice_annotations(paths, split=args.split)
        envelopes = _load_voice_envelopes(paths.voice_transcripts)
        if args.split in {"holdout", "all"} and args.confirm != "HOLDOUT_FINAL_EVALUATION":
            raise VoiceEvalError(
                "Odczyt holdoutu wymaga --confirm HOLDOUT_FINAL_EVALUATION."
            )
        if args.split != "all":
            samples = [
                sample
                for sample in samples
                if sample.split is not None and sample.split.value == args.split
            ]
        selected_ids = {sample.sample_id for sample in samples}
        annotations = [
            annotation
            for annotation in annotations
            if annotation.sample_id in selected_ids
        ]
        envelopes = {
            sample_id: envelope
            for sample_id, envelope in envelopes.items()
            if sample_id in selected_ids
        }
        run_id = args.run_id or (
            f"{args.split}-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        run_root = paths.voice_runs / run_id
        route = None
        index = None
        runtime_configuration: dict = {
            "deepgram_model": settings.deepgram_model,
            "deepgram_language": settings.deepgram_language,
            "stt_processing_location": "remote",
            "stt_network_scope": "internet",
            "prosody_version": "autocorrelation-v1",
            "prosody_processing_location": "local",
            "semantic_processing_location": "local",
            "semantic_network_scope": "loopback",
            "routing_enabled": not args.without_routing,
            "split": args.split,
        }
        if not args.without_routing:
            require_loopback_url(
                settings.local_embeddings_base_url or settings.lm_studio_base_url
            )
            require_loopback_url(settings.qdrant_url)
            registry = ActionRegistry(
                settings,
                MemoryStore(root / "offline.db"),
                WindowsTTS(),
            )
            definitions = registry.capability_catalog()["voiceloop_actions"]
            embeddings = OpenAICompatibleEmbeddingClient(
                base_url=(
                    settings.local_embeddings_base_url or settings.lm_studio_base_url
                ),
                api_key=(
                    settings.local_embeddings_api_key or settings.lm_studio_api_key
                ),
                model=settings.local_embeddings_model,
                timeout_seconds=settings.local_embeddings_timeout_seconds,
                enabled=settings.local_embeddings_enabled,
            )
            index = CapabilityIndex(
                settings,
                embeddings=embeddings,
                definitions=definitions,
            )
            await index.start()
            routing_v2 = RoutingV2Service(
                settings,
                capability_index=index,
                definitions=definitions,
            )
            route = routing_v2.evaluate
            runtime_configuration["routing"] = routing_v2.runtime_config()
        try:
            predictions, metrics = await evaluate_voice_dataset(
                samples=samples,
                annotations=annotations,
                envelopes=envelopes,
                eval_root=paths.voice_root,
                run_id=run_id,
                route=route,
                required_sample_count={
                    "development": 30,
                    "holdout": 90,
                    "all": 120,
                }[args.split],
            )
        finally:
            if index is not None:
                await index.close()
        write_jsonl(run_root / "predictions.jsonl", predictions)
        write_json(run_root / "metrics.json", metrics)
        write_text(run_root / "report.md", render_voice_report(metrics))
        configuration_json = json.dumps(
            runtime_configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        write_json(
            run_root / "manifest.json",
            VoiceEvalRunManifestV1(
                run_id=run_id,
                dataset_sha256=sha256_file(paths.voice_samples),
                annotations_sha256=sha256_text(
                    "\n".join(
                        annotation.model_dump_json()
                        for annotation in sorted(
                            annotations,
                            key=lambda item: item.sample_id,
                        )
                    )
                ),
                splits_sha256=sha256_file(paths.voice_splits),
                configuration=runtime_configuration,
                configuration_fingerprint=sha256_text(configuration_json),
            ),
        )
        print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
        return 0 if metrics.quality_gate_passed else 3
    if args.command == "build-proper-names":
        annotations = _load_voice_annotations(paths, split="all")
        run_root = _resolve_voice_run(paths, args.run_id)
        predictions = read_jsonl(
            run_root / "predictions.jsonl",
            VoiceEvalPredictionV1,
        )
        existing = (
            ProperNameLexiconV1.model_validate_json(
                paths.proper_names.read_text(encoding="utf-8")
            )
            if paths.proper_names.is_file()
            else None
        )
        lexicon = build_proper_name_lexicon(
            annotations,
            predictions,
            existing=existing,
        )
        write_json(paths.proper_names, lexicon)
        print(
            json.dumps(
                {
                    "entry_count": len(lexicon.entries),
                    "approved_count": sum(entry.approved for entry in lexicon.entries),
                    "path": str(paths.proper_names),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "report-actions":
        database = (args.database or settings.data_dir / "voiceloop.db").resolve()
        report = build_action_reliability_report(
            database,
            start=_parse_cli_datetime(args.start) if args.start else None,
            end=_parse_cli_datetime(args.end) if args.end else None,
        )
        write_json(paths.action_reliability_json, report)
        write_text(
            paths.action_reliability_markdown,
            render_action_reliability_report(report),
        )
        print(
            json.dumps(
                {
                    "command_count": report["command_count"],
                    "confirmation_count": report["confirmation_count"],
                    "duplicate_or_retry_candidates": report[
                        "duplicate_or_retry_candidates"
                    ],
                    "event_count": report["event_count"],
                    "reconciled": report["reconciled"],
                    "json_report": str(paths.action_reliability_json),
                    "markdown_report": str(paths.action_reliability_markdown),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "extract-project-journal":
        records = read_jsonl(paths.clean, UtteranceRecord)
        existing = read_jsonl(
            paths.journal_candidates,
            ProjectJournalCandidateV1,
        )
        existing_by_id = {item.candidate_id: item for item in existing}
        for item in extract_project_journal_candidates(records):
            existing_by_id.setdefault(item.candidate_id, item)
        candidates = sorted(existing_by_id.values(), key=lambda item: item.candidate_id)
        write_jsonl(paths.journal_candidates, candidates)
        write_jsonl(paths.journal_entries, approved_journal_entries(candidates))
        print(json.dumps({"candidate_count": len(candidates)}))
        return 0
    if args.command == "list-journal-candidates":
        candidates = read_jsonl(
            paths.journal_candidates,
            ProjectJournalCandidateV1,
        )
        print(
            json.dumps(
                [
                    {
                        "candidate_id": item.candidate_id,
                        "category": item.category.value,
                        "status": item.status.value,
                    }
                    for item in candidates
                ],
                ensure_ascii=False,
            )
        )
        return 0
    if args.command in {"approve-journal", "reject-journal"}:
        candidates = read_jsonl(
            paths.journal_candidates,
            ProjectJournalCandidateV1,
        )
        decided = decide_journal_candidate(
            candidates,
            candidate_id=args.candidate_id,
            approve=args.command == "approve-journal",
            confirmation=args.confirm,
        )
        write_jsonl(paths.journal_candidates, decided)
        write_jsonl(paths.journal_entries, approved_journal_entries(decided))
        item = next(
            candidate for candidate in decided if candidate.candidate_id == args.candidate_id
        )
        print(
            json.dumps(
                {
                    "candidate_id": item.candidate_id,
                    "status": item.status.value,
                }
            )
        )
        return 0
    if args.command == "inventory":
        manifest = build_manifest(
            audio_transcript=args.audio,
            cursor_projects_root=args.cursor_root,
        )
        write_json(paths.manifest, manifest)
        print(
            json.dumps(
                {
                    "manifest_id": manifest.manifest_id,
                    "sources": manifest.unique_source_count,
                    "included_words": manifest.included_word_count,
                    "excluded_words": manifest.excluded_word_count,
                }
            )
        )
        return 0
    if args.command in {
        "source-status",
        "approve-audio-source",
        "revoke-audio-source",
    }:
        if not paths.manifest.is_file():
            raise ValueError("Najpierw uruchom komendę inventory.")
        manifest = SourceManifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))
        decisions = _load_speaker_decisions(paths.speaker_decisions)
        if args.command == "source-status":
            trusted = {item.source_id for item in decisions.decisions}
            audio_sources = [
                {
                    "source_id": source.source_id,
                    "sha256": source.sha256,
                    "word_count": source.word_count,
                    "speaker_approved": source.source_id in trusted,
                }
                for source in manifest.sources
                if source.kind is SourceKind.AUDIO_TRANSCRIPT
            ]
            print(json.dumps(audio_sources, ensure_ascii=False))
            return 0
        if args.command == "approve-audio-source":
            if args.confirm != args.source_id:
                raise ValueError("--confirm musi być identyczne z source_id.")
            source = next(
                (
                    item
                    for item in manifest.sources
                    if item.source_id == args.source_id and item.kind is SourceKind.AUDIO_TRANSCRIPT
                ),
                None,
            )
            if source is None:
                raise ValueError("Nie znaleziono źródła audio w bieżącym manifeście.")
            remaining = [item for item in decisions.decisions if item.source_id != source.source_id]
            remaining.append(
                SpeakerDecision(
                    source_id=source.source_id,
                    source_sha256=source.sha256,
                    speaker_status="self",
                )
            )
            write_json(
                paths.speaker_decisions,
                SpeakerDecisionFile(decisions=remaining),
            )
            await _activate_decision_scope(paths, manifest, remaining)
            print(json.dumps({"source_id": source.source_id, "speaker": "self"}))
            return 0
        remaining = [item for item in decisions.decisions if item.source_id != args.source_id]
        write_json(
            paths.speaker_decisions,
            SpeakerDecisionFile(decisions=remaining),
        )
        await _activate_decision_scope(paths, manifest, remaining)
        print(json.dumps({"source_id": args.source_id, "speaker": "unknown"}))
        return 0
    if args.command == "run":
        registry = ActionRegistry(settings, MemoryStore(root / "offline.db"), WindowsTTS())
        definitions = registry.capability_catalog()["voiceloop_actions"]
        decisions = _load_speaker_decisions(paths.speaker_decisions)
        report = await run_pipeline(
            paths=paths,
            audio_transcript=args.audio,
            cursor_projects_root=args.cursor_root,
            capability_definitions=definitions,
            style_enabled=args.enable_style,
            holdout_percent=args.holdout_percent,
            trusted_audio_source_ids={
                item.source_id for item in decisions.decisions if item.speaker_status == "self"
            },
        )
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command in {
        "evaluate-routing",
        "evaluate-routing-v2",
        "evaluate-routing-personal",
    }:
        require_loopback_url(settings.local_embeddings_base_url or settings.lm_studio_base_url)
        require_loopback_url(settings.qdrant_url)
        eval_settings = (
            settings.model_copy(
                update={"qdrant_capability_collection": args.capability_collection}
            )
            if args.capability_collection
            else settings
        )
        registry = ActionRegistry(eval_settings, MemoryStore(root / "offline.db"), WindowsTTS())
        definitions = registry.capability_catalog()["voiceloop_actions"]
        embeddings = OpenAICompatibleEmbeddingClient(
            base_url=(
                eval_settings.local_embeddings_base_url or eval_settings.lm_studio_base_url
            ),
            api_key=(
                eval_settings.local_embeddings_api_key or eval_settings.lm_studio_api_key
            ),
            model=eval_settings.local_embeddings_model,
            timeout_seconds=eval_settings.local_embeddings_timeout_seconds,
            enabled=eval_settings.local_embeddings_enabled,
        )
        index = CapabilityIndex(
            eval_settings,
            embeddings=embeddings,
            definitions=definitions,
        )
        await index.start()
        try:
            expected_action_ids = {
                str(definition["id"])
                for definition in definitions
                if str(definition.get("id") or "") != "speak_text"
            }
            if args.command in {
                "evaluate-routing-v2",
                "evaluate-routing-personal",
            }:
                if args.command == "evaluate-routing-personal":
                    records = read_jsonl(
                        paths.routing_personal_holdout,
                        RoutingEvalRecord,
                    )
                    if not records or any(
                        "personal_holdout" not in record.tags for record in records
                    ):
                        raise ValueError(
                            "Personal holdout musi być niepusty i oznaczony tagiem "
                            "personal_holdout."
                        )
                    expected_action_ids = {
                        action_id
                        for record in records
                        for action_id in record.plan_action_ids
                    }
                else:
                    records = build_routing_eval_records(definitions)
                    write_jsonl(paths.eval_set, records)
                routing_v2 = RoutingV2Service(
                    eval_settings,
                    capability_index=index,
                    definitions=definitions,
                )
                cards, metrics = await evaluate_routing_v2(
                    records=records,
                    route=routing_v2.evaluate,
                    stt_threshold=eval_settings.stt_min_action_confidence,
                    expected_action_ids=expected_action_ids,
                    catalog_hash=index.catalog_hash,
                    runtime_config=routing_v2.runtime_config,
                )
                if args.command == "evaluate-routing-personal":
                    score_path = paths.routing_personal_scores
                    metrics_path = paths.routing_personal_metrics
                else:
                    score_path = paths.routing_v2_scores
                    metrics_path = paths.routing_v2_metrics
                if (
                    args.capability_collection
                    and args.capability_collection
                    != settings.qdrant_capability_collection
                ):
                    collection_label = re.sub(
                        r"[^a-zA-Z0-9._-]+",
                        "-",
                        args.capability_collection,
                    ).strip("-")[:120] or "next"
                    if args.command == "evaluate-routing-personal":
                        score_path = (
                            paths.routing_personal_scores.parent
                            / f"routing-personal-scores-v1-{collection_label}.jsonl"
                        )
                        metrics_path = (
                            paths.routing_personal_metrics.parent
                            / f"routing-personal-metrics-v1-{collection_label}.json"
                        )
                    else:
                        score_path = (
                            paths.routing_v2_scores.parent
                            / f"routing-v2-scores-{collection_label}.jsonl"
                        )
                        metrics_path = (
                            paths.routing_v2_metrics.parent
                            / f"routing-v2-metrics-{collection_label}.json"
                        )
                write_jsonl(score_path, cards)
                write_json(metrics_path, metrics)
            else:
                records = read_jsonl(paths.eval_set, RoutingEvalRecord)
                cards, metrics = await evaluate_routing(
                    records=records,
                    search=index.search,
                    min_score=eval_settings.capability_match_min_score,
                    margin_threshold=(
                        args.margin_threshold
                        if args.margin_threshold is not None
                        else eval_settings.corpus_routing_margin_threshold
                    ),
                    stt_threshold=eval_settings.stt_min_action_confidence,
                    expected_action_ids=expected_action_ids,
                )
                write_jsonl(paths.routing_scores, cards)
                write_json(paths.routing_metrics, metrics)
        finally:
            await index.close()
        print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
        return 0 if metrics.quality_gate_passed else 3

    if args.command == "validate-routing-calibration":
        dataset_path = (
            args.dataset.resolve()
            if args.dataset is not None
            else paths.routing_calibration_observations.resolve()
        )
        _ensure_regular_dataset_file(dataset_path)
        observations = load_calibration_observations(dataset_path)
        train_cutoff = (
            _parse_cli_datetime(args.train_cutoff) if args.train_cutoff else None
        )
        report = validate_calibration_observations(
            observations,
            train_cutoff=train_cutoff,
            for_training=bool(args.for_training),
        )
        report.update(
            {
                "dataset_path": str(dataset_path),
                "for_training": bool(args.for_training),
            }
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["temporal_component_leakage_count"] == 0 else 3

    if args.command == "fit-routing-calibration":
        dataset_path = (
            args.dataset.resolve()
            if args.dataset is not None
            else paths.routing_calibration_observations.resolve()
        )
        _ensure_regular_dataset_file(dataset_path)
        if "routing-personal-holdout" in dataset_path.name.casefold():
            raise ValueError(
                "Personal holdout jest zabroniony dla treningu kalibracji."
            )
        observations = load_calibration_observations(dataset_path)
        representative_rows = [
            row
            for row in observations
            if row.set_role is RoutingCalibrationSetRole.REPRESENTATIVE
        ]
        if representative_rows:
            runtime_fingerprint = _unique_required_value(
                [
                    row.runtime_fingerprint
                    for row in representative_rows
                    if row.runtime_fingerprint
                ],
                field_name="runtime_fingerprint",
            )
            catalog_hash = _unique_required_value(
                [row.catalog_hash for row in representative_rows if row.catalog_hash],
                field_name="catalog_hash",
            )
        else:
            # Pusty lokalny store ma dać jawny artefakt insufficient_data,
            # a nie wyjątek ani pozorne prawdopodobieństwo.
            runtime_fingerprint = "0" * 64
            catalog_hash = "unknown"
        artifact, report = fit_routing_calibration(
            observations,
            runtime_fingerprint=runtime_fingerprint,
            catalog_hash=catalog_hash,
            train_cutoff=(
                _parse_cli_datetime(args.train_cutoff) if args.train_cutoff else None
            ),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        artifact_path = (
            (args.artifact or paths.routing_calibration_artifact)
            if args.artifact is not None
            else paths.routing_calibration_artifact
        ).resolve()
        report_path = (
            (args.report or paths.routing_calibration_report)
            if args.report is not None
            else paths.routing_calibration_report
        ).resolve()
        _assert_distinct_paths(dataset_path, artifact_path, report_path)
        write_json(artifact_path, artifact)
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": artifact.status.value,
                    "artifact": str(artifact_path),
                    "report": str(report_path),
                    "artifact_fingerprint": artifact.artifact_fingerprint,
                    "sample_count": report.sample_count,
                    "group_count": report.group_count,
                    "credibility_passed": (
                        report.credibility_gates.passed
                        if report.credibility_gates is not None
                        else False
                    ),
                },
                ensure_ascii=False,
            )
        )
        return (
            0
            if artifact.status is RoutingCalibrationArtifactStatus.READY
            else 3
        )

    if args.command == "evaluate-routing-calibration":
        dataset_path = (
            args.dataset.resolve()
            if args.dataset is not None
            else paths.routing_calibration_observations.resolve()
        )
        _ensure_regular_dataset_file(dataset_path)
        artifact_path = (
            (args.artifact or paths.routing_calibration_artifact)
            if args.artifact is not None
            else paths.routing_calibration_artifact
        ).resolve()
        if not artifact_path.is_file():
            raise ValueError("Brak artefaktu kalibracji.")
        artifact = RoutingCalibrationArtifactV1.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        observations = load_calibration_observations(dataset_path)
        has_independent_holdout = validate_calibration_artifact_for_evaluation(
            artifact, observations
        )
        report = evaluate_routing_calibration(
            observations,
            artifact=artifact,
            require_independent_holdout=True,
            independent_holdout_available=has_independent_holdout,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        report_path = (
            (args.report or paths.routing_calibration_report)
            if args.report is not None
            else paths.routing_calibration_report
        ).resolve()
        _assert_distinct_paths(dataset_path, artifact_path, report_path)
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report.status,
                    "dataset_path": str(dataset_path),
                    "artifact_path": str(artifact_path),
                    "report": str(report_path),
                    "sample_count": report.sample_count,
                    "brier_score": report.brier_score,
                    "equal_mass_ece": report.equal_mass_ece,
                    "ece_ci95_upper": report.ece_ci95_upper,
                    "credibility_passed": (
                        report.credibility_gates.passed
                        if report.credibility_gates is not None
                        else False
                    ),
                },
                ensure_ascii=False,
            )
        )
        return (
            0
            if report.credibility_gates is not None and report.credibility_gates.passed
            else 3
        )

    if args.command == "build-memory-index-v2":
        require_loopback_url(settings.local_embeddings_base_url or settings.lm_studio_base_url)
        require_loopback_url(settings.qdrant_url)
        target_collection = (
            args.collection or settings.qdrant_memory_next_collection or ""
        ).strip()
        if not target_collection:
            raise ValueError(
                "Podaj --collection albo QDRANT_MEMORY_NEXT_COLLECTION."
            )
        if target_collection == settings.qdrant_collection:
            raise ValueError("Kolekcja migracyjna musi różnić się od aktywnej kolekcji.")
        if args.confirm != target_collection:
            raise ValueError("--confirm musi być identyczne z nazwą kolekcji migracyjnej.")
        migration_settings = settings.model_copy(
            update={
                "qdrant_collection": target_collection,
                "vector_memory_prune_enabled": False,
            }
        )
        database_path = (
            args.database or migration_settings.data_dir / "voiceloop.db"
        ).resolve()
        memory = MemoryStore(database_path)
        await memory.initialize()
        embeddings = OpenAICompatibleEmbeddingClient(
            base_url=(
                migration_settings.local_embeddings_base_url
                or migration_settings.lm_studio_base_url
            ),
            api_key=(
                migration_settings.local_embeddings_api_key
                or migration_settings.lm_studio_api_key
            ),
            model=migration_settings.local_embeddings_model,
            timeout_seconds=migration_settings.local_embeddings_timeout_seconds,
            enabled=migration_settings.local_embeddings_enabled,
        )
        qdrant = QdrantVectorStore(migration_settings)
        digester = LocalBehaviorDigestClient(
            base_url=migration_settings.lm_studio_base_url,
            api_key=migration_settings.lm_studio_api_key,
            model=(
                migration_settings.behavior_digest_model
                or migration_settings.lm_studio_model
            ),
            timeout_seconds=migration_settings.behavior_digest_timeout_seconds,
            enabled=migration_settings.behavior_digest_enabled,
        )
        worker = ScreenpipeVectorMemoryWorker(
            settings=migration_settings,
            screenpipe=ScreenpipeClient(migration_settings),
            memory=memory,
            embeddings=embeddings,
            qdrant=qdrant,
            digester=digester,
        )
        try:
            legacy_count = await worker.migrate_legacy_memories()
            rebuilt_count = (
                await worker.index_recent_activity()
                if args.include_screenpipe
                else 0
            )
        finally:
            await qdrant.close()
        print(
            json.dumps(
                {
                    "collection": target_collection,
                    "active_collection_unchanged": settings.qdrant_collection,
                    "legacy_semantic_only_count": legacy_count,
                    "screenpipe_rebuilt_count": rebuilt_count,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "evaluate-memory-retrieval":
        require_loopback_url(settings.local_embeddings_base_url or settings.lm_studio_base_url)
        require_loopback_url(settings.qdrant_url)
        gold_path = (args.gold or paths.memory_retrieval_eval).resolve()
        records = read_jsonl(gold_path, MemoryRetrievalEvalRecordV1)
        if not records:
            raise ValueError(
                f"Brak ręcznych przykładów gold pamięci w pliku: {gold_path}"
            )
        eval_settings = (
            settings.model_copy(update={"qdrant_collection": args.collection})
            if args.collection
            else settings
        )
        embeddings = OpenAICompatibleEmbeddingClient(
            base_url=(
                eval_settings.local_embeddings_base_url or eval_settings.lm_studio_base_url
            ),
            api_key=(
                eval_settings.local_embeddings_api_key or eval_settings.lm_studio_api_key
            ),
            model=eval_settings.local_embeddings_model,
            timeout_seconds=eval_settings.local_embeddings_timeout_seconds,
            enabled=eval_settings.local_embeddings_enabled,
        )
        vector_store = QdrantVectorStore(eval_settings)

        async def search_memory(record: MemoryRetrievalEvalRecordV1, limit: int):
            documents = memory_query_documents(record.query)
            vector_names = tuple(
                name for name in MEMORY_VECTOR_NAMES if name in documents
            )
            vectors = await embeddings.embed_queries(
                [documents[name] for name in vector_names]
            )
            if len(vectors) != len(vector_names):
                raise EmbeddingUnavailableError(
                    "embedding count mismatch for memory retrieval evaluation"
                )
            return await vector_store.search(
                query_vectors=dict(zip(vector_names, vectors, strict=True)),
                query_weights=memory_query_weights(
                    record.query,
                    adaptive=eval_settings.vector_memory_adaptive_query_weights,
                    base_weights=eval_settings.vector_memory_weights,
                ),
                vector_names=MEMORY_VECTOR_NAMES,
                source=record.source_filter,
                limit=limit,
                min_score=eval_settings.vector_memory_min_score,
                rrf_k=eval_settings.vector_memory_rrf_k,
            )

        try:
            cards, metrics = await evaluate_memory_retrieval(
                records=records,
                search=search_memory,
                k=args.k,
            )
        finally:
            await vector_store.close()
        runtime_config = MemoryRetrievalRuntimeConfigV1(
            collection=eval_settings.qdrant_collection,
            embedding_model=(
                embeddings._resolved_model
                or embeddings.configured_model
                or "unresolved-local-embedding"
            ),
            query_format_version=MEMORY_QUERY_DOCUMENTS_VERSION,
            fusion_method=WEIGHTED_RRF_VERSION,
            vector_weights=eval_settings.vector_memory_weights,
            adaptive_query_weights=(
                eval_settings.vector_memory_adaptive_query_weights
            ),
            min_score=eval_settings.vector_memory_min_score,
            rrf_k=eval_settings.vector_memory_rrf_k,
        )
        metrics = metrics.model_copy(
            update={
                "runtime_config": runtime_config,
                "runtime_fingerprint": runtime_config.fingerprint(),
            }
        )
        score_path = paths.memory_retrieval_scores
        metrics_path = paths.memory_retrieval_metrics
        if args.collection and args.collection != settings.qdrant_collection:
            collection_label = re.sub(
                r"[^a-zA-Z0-9._-]+",
                "-",
                args.collection,
            ).strip("-")[:120] or "next"
            score_path = score_path.with_name(
                f"memory-retrieval-scores-{collection_label}.jsonl"
            )
            metrics_path = metrics_path.with_name(
                f"memory-retrieval-metrics-{collection_label}.json"
            )
        write_jsonl(score_path, cards)
        write_json(metrics_path, metrics)
        print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
        return 0

    store = MemoryCandidateStore(paths.candidates_database)
    await store.initialize()
    if args.command == "list-candidates":
        candidates = await store.list(status=CandidateStatus(args.status), limit=500)
        print(
            json.dumps(
                [
                    {
                        "candidate_id": item.candidate_id,
                        "kind": item.kind.value,
                        "status": item.status.value,
                    }
                    for item in candidates
                ],
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "show-candidate":
        candidate = await store.get(args.candidate_id)
        if candidate is None:
            raise CandidateDecisionError("Nie znaleziono kandydata pamięci.")
        print(json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False))
        return 0
    if args.command == "approve":
        if args.confirm != args.candidate_id:
            raise CandidateDecisionError("--confirm musi być identyczne z candidate_id.")
        memory = MemoryStore(settings.data_dir / "voiceloop.db")
        await memory.initialize()
        candidate, item = await store.approve(
            args.candidate_id,
            memory,
            expected_content_sha256=args.content_sha256,
        )
        print(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status.value,
                    "memory_id": item.id,
                }
            )
        )
        return 0
    if args.command == "reject":
        candidate = await store.reject(args.candidate_id)
        print(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status.value,
                }
            )
        )
        return 0
    raise ValueError("Nieznana komenda.")


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--audio",
        type=Path,
        help="Opcjonalny lokalny plik transkrypcji audio.",
    )
    parser.add_argument(
        "--cursor-root",
        type=Path,
        help="Opcjonalny lokalny katalog projektów Cursor.",
    )
    parser.add_argument("--data-root", type=Path)


def _unique_required_value(values: list[str], *, field_name: str) -> str:
    unique = sorted({value for value in values if value})
    if not unique:
        raise ValueError(f"Brak wartosci {field_name} w zbiorze.")
    if len(unique) != 1:
        raise ValueError(f"Mieszane wartosci {field_name} w zbiorze: {unique}")
    return unique[0]


def _assert_distinct_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("dataset/artifact/report muszą wskazywać różne ścieżki.")


def _ensure_regular_dataset_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Nie znaleziono datasetu kalibracji: {path}")
    if not path.is_file():
        raise ValueError(f"Dataset kalibracji musi być plikiem: {path}")


def _load_speaker_decisions(path: Path) -> SpeakerDecisionFile:
    if not path.is_file():
        return SpeakerDecisionFile()
    try:
        return SpeakerDecisionFile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Nieprawidłowy plik decyzji mówcy.") from exc


async def _activate_decision_scope(
    paths: CorpusPaths,
    manifest: SourceManifest,
    decisions: list[SpeakerDecision],
) -> None:
    store = MemoryCandidateStore(paths.candidates_database)
    await store.initialize()
    trusted = {item.source_id for item in decisions if item.speaker_status == "self"}
    await store.set_active_scope(corpus_scope_id(manifest.manifest_id, trusted))


def _parse_cli_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Nieprawidłowa data lub timestamp: {value!r}.") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _load_voice_envelopes(
    path: Path,
    *,
    required: bool = True,
) -> dict[str, TranscriptEnvelopeV1]:
    if not path.is_file():
        if required:
            raise VoiceEvalError(
                "Brak transcripts-v1.jsonl; uruchom transcribe-voice-eval."
            )
        return {}
    result: dict[str, TranscriptEnvelopeV1] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                payload = json.loads(raw_line)
                sample_id = str(payload["sample_id"])
                envelope = TranscriptEnvelopeV1.model_validate(payload["envelope"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise VoiceEvalError(
                    f"Nieprawidłowy transcript cache, linia {line_number}."
                ) from exc
            if sample_id in result:
                raise VoiceEvalError(f"Powtórzony transcript próbki: {sample_id}.")
            result[sample_id] = envelope
    return result


def _resolve_voice_run(paths: CorpusPaths, run_id: str | None) -> Path:
    if run_id:
        run_root = paths.voice_runs / run_id
        if not run_root.is_dir():
            raise VoiceEvalError(f"Nie znaleziono uruchomienia: {run_id}.")
        return run_root
    if not paths.voice_runs.is_dir():
        raise VoiceEvalError("Brak uruchomień evaluate-voice-eval.")
    runs = sorted(
        (item for item in paths.voice_runs.iterdir() if item.is_dir()),
        key=lambda item: item.name,
    )
    if not runs:
        raise VoiceEvalError("Brak uruchomień evaluate-voice-eval.")
    return runs[-1]


def _voice_annotation_template(
    samples: list[VoiceEvalSampleV1],
) -> list[dict]:
    return [
        {
            "schema_version": 1,
            "sample_id": sample.sample_id,
            "audio_clip_sha256": sample.audio.clip_sha256 if sample.audio else None,
            "literal_text": sample.observed_text,
            "punctuated_text": sample.observed_text,
            "intent": None,
            "prosody_tags": [],
            "proper_names": [],
            "speaker_role": None,
            "speaker_confirmed": False,
            "expected_outcome": None,
            "expected_action_ids": [],
            "expected_step_args": [],
            "expected_abstention": False,
            "annotator": "",
            "approved_at": None,
        }
        for sample in samples
    ]


def _prepare_voice_development_refill(
    paths: CorpusPaths,
    candidates: list[VoiceEvalSampleV1],
    *,
    development_count: int,
) -> tuple[
    list[VoiceEvalSampleV1],
    dict[str, TranscriptEnvelopeV1],
    list[str],
]:
    if development_count < 1:
        raise VoiceEvalError("Liczba próbek development musi być dodatnia.")
    report_path = paths.voice_root / "transcription-report-development-v1.json"
    if not report_path.is_file():
        raise VoiceEvalError(
            "Brak raportu development; uruchom najpierw transcribe-voice-eval."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rejected_ids = {
        str(sample_id) for sample_id in report.get("no_speech_sample_ids", ())
    }
    if not rejected_ids:
        raise VoiceEvalError("Raport nie zawiera klipów development bez mowy.")
    current_samples = read_jsonl(paths.voice_samples, VoiceEvalSampleV1)
    if not current_samples:
        raise VoiceEvalError("Brak zamrożonego zestawu voice eval do uzupełnienia.")
    previous_holdout_ids = {
        sample.sample_id
        for sample in current_samples
        if sample.split is not None and sample.split.value == "holdout"
    }
    if not previous_holdout_ids:
        raise VoiceEvalError("Zamrożony zestaw nie zawiera holdoutu.")
    samples = refill_voice_development_samples(
        candidates,
        current_samples,
        rejected_sample_ids=rejected_ids,
        development_count=development_count,
    )
    holdout_ids = {
        sample.sample_id
        for sample in samples
        if sample.split is not None and sample.split.value == "holdout"
    }
    if holdout_ids != previous_holdout_ids:
        raise VoiceEvalError("Refill naruszył zamrożony holdout.")
    current_ids = {sample.sample_id for sample in current_samples}
    replacement_ids = [
        sample.sample_id
        for sample in samples
        if sample.split is not None
        and sample.split.value == "development"
        and sample.sample_id not in current_ids
    ]
    envelopes = _load_voice_envelopes(paths.voice_transcripts, required=False)
    return samples, envelopes, replacement_ids


def _publish_voice_candidate_audio(
    candidates: list[VoiceEvalSampleV1],
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    pending: list[tuple[Path, Path, str]] = []
    for candidate in candidates:
        if candidate.audio is None:
            continue
        relative_path = Path(candidate.audio.relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise VoiceEvalError("Kandydat audio ma niebezpieczną ścieżkę względną.")
        source = source_root / relative_path
        destination = destination_root / relative_path
        expected_hash = candidate.audio.clip_sha256
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise VoiceEvalError(
                f"Nie udało się zweryfikować przygotowanego audio: {candidate.sample_id}."
            )
        if destination.is_file():
            if sha256_file(destination) != expected_hash:
                raise VoiceEvalError(
                    f"Docelowy plik audio ma inny hash: {candidate.sample_id}."
                )
            continue
        pending.append((source, destination, expected_hash))

    for source, destination, expected_hash in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != expected_hash:
            destination.unlink(missing_ok=True)
            raise VoiceEvalError(f"Nie udało się opublikować audio: {destination.name}.")


def _write_voice_selection_artifacts(
    paths: CorpusPaths,
    samples: list[VoiceEvalSampleV1],
    *,
    prefill_envelopes: dict[str, TranscriptEnvelopeV1] | None = None,
) -> None:
    development_samples = [
        sample
        for sample in samples
        if sample.split is not None and sample.split.value == "development"
    ]
    holdout_samples = [
        sample
        for sample in samples
        if sample.split is not None and sample.split.value == "holdout"
    ]
    write_jsonl(paths.voice_samples, samples)
    write_json(
        paths.voice_splits,
        {
            "schema_version": 1,
            "development": [sample.sample_id for sample in development_samples],
            "holdout": [sample.sample_id for sample in holdout_samples],
        },
    )
    write_jsonl(
        paths.voice_annotation_template,
        _voice_annotation_template(samples),
    )
    write_jsonl(
        paths.voice_annotation_template_development,
        _voice_annotation_template(development_samples),
    )
    write_jsonl(
        paths.voice_annotation_template_holdout,
        _voice_annotation_template(holdout_samples),
    )
    prefills = prefill_envelopes or {}
    write_text(
        paths.voice_annotation_review,
        render_voice_annotation_review(
            samples,
            prefill_envelopes=prefills,
        ),
    )
    write_text(
        paths.voice_annotation_review_development,
        render_voice_annotation_review(
            development_samples,
            download_filename=paths.voice_annotations_development.name,
            storage_namespace="development",
            prefill_envelopes=prefills,
        ),
    )
    write_text(
        paths.voice_annotation_review_holdout,
        render_voice_annotation_review(
            holdout_samples,
            download_filename=paths.voice_annotations_holdout.name,
            storage_namespace="holdout",
            prefill_envelopes=prefills,
        ),
    )


def _load_voice_annotations(
    paths: CorpusPaths,
    *,
    split: str,
) -> list[VoiceGoldAnnotationV1]:
    if split == "development" and paths.voice_annotations_development.is_file():
        records = read_jsonl(
            paths.voice_annotations_development,
            VoiceGoldAnnotationV1,
        )
    elif split == "holdout" and paths.voice_annotations_holdout.is_file():
        records = read_jsonl(
            paths.voice_annotations_holdout,
            VoiceGoldAnnotationV1,
        )
    elif paths.voice_annotations.is_file():
        records = read_jsonl(paths.voice_annotations, VoiceGoldAnnotationV1)
    else:
        records = [
            *read_jsonl(
                paths.voice_annotations_development,
                VoiceGoldAnnotationV1,
            ),
            *read_jsonl(
                paths.voice_annotations_holdout,
                VoiceGoldAnnotationV1,
            ),
        ]
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise VoiceEvalError("Pliki adnotacji zawierają powtórzone sample_id.")
    return records
