from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..router import normalize_text
from ..screenpipe import ScreenpipeAudioChunk, ScreenpipeClient
from .prosody import analyze_prosody
from .schema import (
    AudioAssetRefV1,
    AudioDirection,
    NetworkScope,
    ProvenanceV1,
    SpeakerRole,
    VoiceAudioSourceV1,
    VoiceEvalSampleV1,
    VoiceEvalSplit,
    VoiceGoldAnnotationV1,
    VoiceSampleState,
    VoiceSourceManifestV1,
)
from .storage import sha256_file, sha256_text

_SUPPORTED_AUDIO_SUFFIXES = {".mp4", ".m4a", ".wav", ".webm"}
_SCREENPIPE_FILE_PATTERN = re.compile(
    r"^(?P<device>.+?)\s+\((?P<direction>input|output)\)_"
    r"(?P<day>20\d{2}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})"
    r"\.(?P<extension>mp4|m4a|wav|webm)$",
    re.IGNORECASE,
)
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
_CORRECTION_MARKERS = {
    "znaczy",
    "to znaczy",
    "nie,",
    "jednak",
    "poprawka",
    "właściwie",
    "wlasciwie",
}
_CANCEL_MARKERS = {
    "anuluj",
    "przerwij",
    "stop",
    "nieważne",
    "niewazne",
    "zostaw",
}
_KNOWN_PROPER_NAMES = {
    "voiceloop",
    "deepgram",
    "screenpipe",
    "cursor",
    "voiceattack",
    "qdrant",
    "gemini",
    "venice",
    "youtube",
    "chrome",
    "lm studio",
}


class VoiceEvalError(RuntimeError):
    pass


def parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VoiceEvalError(f"Nieprawidłowy timestamp audio: {value!r}.") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def infer_audio_direction(device_name: str, device_type: str) -> AudioDirection:
    combined = f"{device_type} {device_name}".casefold()
    if any(marker in combined for marker in ("input", "microphone", "mikrofon", "mic")):
        return AudioDirection.INPUT
    if any(
        marker in combined
        for marker in ("output", "speaker", "głośnik", "glosnik", "loopback", "monitor")
    ):
        return AudioDirection.OUTPUT
    return AudioDirection.UNKNOWN


def safe_screenpipe_audio_path(path: Path, *, screenpipe_root: Path) -> Path:
    root = screenpipe_root.expanduser().resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise VoiceEvalError("Plik audio jest poza bezpiecznym katalogiem Screenpipe.") from exc
    if resolved.suffix.casefold() not in _SUPPORTED_AUDIO_SUFFIXES:
        raise VoiceEvalError(f"Nieobsługiwany format audio: {resolved.suffix}.")
    return resolved


async def inventory_screenpipe_audio(
    client: ScreenpipeClient,
    *,
    start: datetime,
    end: datetime,
    screenpipe_root: Path,
    max_results: int = 2000,
) -> VoiceSourceManifestV1:
    if end <= start:
        raise VoiceEvalError("Koniec inwentaryzacji musi następować po początku.")
    chunks = await client.audio_chunks(start=start, end=end, max_results=max_results)
    known_paths = {
        str(
            safe_screenpipe_audio_path(chunk.file_path, screenpipe_root=screenpipe_root)
        ).casefold()
        for chunk in chunks
        if _is_safe_audio_path(chunk.file_path, screenpipe_root=screenpipe_root)
    }
    if len(chunks) < max_results:
        chunks.extend(
            discover_screenpipe_audio_files(
                screenpipe_root,
                start=start,
                end=end,
                limit=max_results - len(chunks),
                excluded_paths=known_paths,
            )
        )
    return build_voice_source_manifest(
        chunks,
        start=start,
        end=end,
        screenpipe_root=screenpipe_root,
    )


def discover_screenpipe_audio_files(
    screenpipe_root: Path,
    *,
    start: datetime,
    end: datetime,
    limit: int,
    excluded_paths: set[str] | None = None,
) -> list[ScreenpipeAudioChunk]:
    if limit <= 0:
        return []
    root = screenpipe_root.expanduser().resolve()
    local_timezone = datetime.now().astimezone().tzinfo or UTC
    excluded_paths = excluded_paths or set()
    candidates: list[tuple[datetime, Path, str, str]] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.casefold() not in _SUPPORTED_AUDIO_SUFFIXES:
            continue
        if str(path.resolve()).casefold() in excluded_paths:
            continue
        match = _SCREENPIPE_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        try:
            captured = datetime.strptime(
                f"{match.group('day')} {match.group('time')}",
                "%Y-%m-%d %H-%M-%S",
            ).replace(tzinfo=local_timezone)
            captured = captured.astimezone(UTC)
        except ValueError:
            continue
        if not (start <= captured < end):
            continue
        candidates.append(
            (
                captured,
                path,
                match.group("device"),
                match.group("direction").capitalize(),
            )
        )
    candidates.sort(
        key=lambda item: (
            item[3].casefold() != "input",
            sha256_text(f"{item[0].date().isoformat()}:{item[1].name}"),
        )
    )
    return [
        ScreenpipeAudioChunk(
            chunk_id=f"file:{sha256_text(str(path.resolve()).casefold())[:24]}",
            file_path=path,
            device_name=device_name,
            device_type=device_type,
            start_time=captured.isoformat(),
            end_time=(captured + timedelta(seconds=30)).isoformat(),
            text="",
        )
        for captured, path, device_name, device_type in candidates[:limit]
    ]


def build_voice_source_manifest(
    chunks: list[ScreenpipeAudioChunk],
    *,
    start: datetime,
    end: datetime,
    screenpipe_root: Path,
) -> VoiceSourceManifestV1:
    records: list[VoiceAudioSourceV1] = []
    seen_chunk_ids: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: (item.start_time, item.chunk_id)):
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        try:
            captured_start = parse_timestamp(chunk.start_time)
            captured_end = parse_timestamp(chunk.end_time)
            path = safe_screenpipe_audio_path(chunk.file_path, screenpipe_root=screenpipe_root)
            stat = path.stat()
            file_hash = sha256_file(path)
            included = captured_end > captured_start and stat.st_size > 0
            exclude_reason = None if included else "empty_or_invalid_range"
            path_text = str(path)
        except (VoiceEvalError, OSError) as exc:
            captured_start = _safe_timestamp(chunk.start_time, fallback=start)
            captured_end = _safe_timestamp(chunk.end_time, fallback=captured_start)
            path_text = str(chunk.file_path)
            file_hash = hashlib.sha256(path_text.encode("utf-8")).hexdigest()
            stat = None
            included = False
            exclude_reason = str(exc)[:500]
        source_id = str(
            uuid5(
                NAMESPACE_URL,
                f"voiceloop-voice-source:{chunk.chunk_id}:{file_hash}:{captured_start.isoformat()}",
            )
        )
        records.append(
            VoiceAudioSourceV1(
                source_id=source_id,
                chunk_id=chunk.chunk_id,
                path=path_text,
                sha256=file_hash,
                byte_size=stat.st_size if stat is not None else 0,
                modified_at=(
                    datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                    if stat is not None
                    else captured_start
                ),
                captured_start=captured_start,
                captured_end=captured_end,
                device_name=chunk.device_name,
                device_type=chunk.device_type,
                audio_direction=infer_audio_direction(chunk.device_name, chunk.device_type),
                source_offset_start_seconds=chunk.start_offset_seconds,
                source_offset_end_seconds=chunk.end_offset_seconds,
                included=included,
                exclude_reason=exclude_reason,
            )
        )
    fingerprint = json.dumps(
        [
            {
                "source_id": record.source_id,
                "sha256": record.sha256,
                "included": record.included,
                "exclude_reason": record.exclude_reason,
            }
            for record in records
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    included_count = sum(record.included for record in records)
    return VoiceSourceManifestV1(
        manifest_id=sha256_text(fingerprint)[:20],
        range_start=start,
        range_end=end,
        sources=tuple(records),
        included_source_count=included_count,
        excluded_source_count=len(records) - included_count,
    )


def build_voice_candidates(
    manifest: VoiceSourceManifestV1,
    *,
    eval_root: Path,
    screenpipe_root: Path,
    speaker_role: SpeakerRole,
    max_candidates: int = 360,
    ffmpeg_executable: str = "ffmpeg",
) -> list[VoiceEvalSampleV1]:
    if max_candidates < 1:
        return []
    candidates: list[VoiceEvalSampleV1] = []
    sources = [
        source
        for source in manifest.sources
        if source.included and source.audio_direction is AudioDirection.INPUT
    ]
    for source in _spread_sources(sources):
        if len(candidates) >= max_candidates:
            break
        path = safe_screenpipe_audio_path(Path(source.path), screenpipe_root=screenpipe_root)
        remaining = max_candidates - len(candidates)
        candidates.extend(
            _segment_source(
                source,
                path=path,
                eval_root=eval_root,
                speaker_role=speaker_role,
                max_segments=remaining,
                ffmpeg_executable=ffmpeg_executable,
            )
        )
    return mark_cross_channel_duplicates(candidates)


def select_voice_eval_samples(
    candidates: list[VoiceEvalSampleV1],
    *,
    target: int = 120,
    development_count: int = 30,
) -> list[VoiceEvalSampleV1]:
    if target < 1:
        raise VoiceEvalError("Docelowa liczba próbek musi być dodatnia.")
    if development_count < 1 or development_count >= target:
        raise VoiceEvalError("Split development musi być dodatni i mniejszy od celu.")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.audio is not None
        and candidate.provenance.speaker_role is SpeakerRole.SELF
        and candidate.provenance.duplicate_of is None
        and candidate.provenance.retry_of is None
        and candidate.state is not VoiceSampleState.EXCLUDED
    ]
    ranked = _rank_by_coverage(eligible, target=target)
    if len(ranked) < target:
        raise VoiceEvalError(
            f"Po bramkach prywatności i deduplikacji jest {len(ranked)} próbek; "
            f"wymagane {target}."
        )
    selected = ranked[:target]
    development_ids = _development_sample_ids(selected, development_count)
    return [
        sample.model_copy(
            update={
                "state": VoiceSampleState.SELECTED,
                "split": (
                    VoiceEvalSplit.DEVELOPMENT
                    if sample.sample_id in development_ids
                    else VoiceEvalSplit.HOLDOUT
                ),
            }
        )
        for sample in selected
    ]


def refill_voice_development_samples(
    candidates: list[VoiceEvalSampleV1],
    current_samples: list[VoiceEvalSampleV1],
    *,
    rejected_sample_ids: set[str],
    development_count: int = 30,
) -> list[VoiceEvalSampleV1]:
    """Replace unusable development clips without opening or changing the holdout."""

    holdout = [
        sample
        for sample in current_samples
        if sample.split is VoiceEvalSplit.HOLDOUT
    ]
    development = [
        sample
        for sample in current_samples
        if sample.split is VoiceEvalSplit.DEVELOPMENT
        and sample.sample_id not in rejected_sample_ids
    ]
    if len(development) > development_count:
        raise VoiceEvalError("Zachowanych próbek development jest więcej niż docelowy limit.")
    needed = development_count - len(development)
    if needed == 0:
        return [*development, *holdout]

    used_ids = {sample.sample_id for sample in current_samples}
    blocked_groups = {_voice_split_group(sample) for sample in holdout}
    eligible = [
        candidate
        for candidate in candidates
        if candidate.audio is not None
        and candidate.audio.duration_seconds >= 0.8
        and candidate.provenance.speaker_role is SpeakerRole.SELF
        and candidate.provenance.duplicate_of is None
        and candidate.provenance.retry_of is None
        and candidate.state is not VoiceSampleState.EXCLUDED
        and candidate.sample_id not in used_ids
        and candidate.sample_id not in rejected_sample_ids
        and _voice_split_group(candidate) not in blocked_groups
    ]
    preferred = [
        candidate for candidate in eligible if "adverse_audio" not in candidate.tags
    ]
    preferred_ids = {candidate.sample_id for candidate in preferred}
    fallback = [
        candidate
        for candidate in eligible
        if candidate.sample_id not in preferred_ids
    ]
    ranked = [
        *_rank_speech_replacements(preferred),
        *_rank_speech_replacements(fallback),
    ]
    if len(ranked) < needed:
        raise VoiceEvalError(
            f"Brakuje zamienników development: dostępne {len(ranked)}, potrzebne {needed}."
        )
    replacements = [
        sample.model_copy(
            update={
                "state": VoiceSampleState.SELECTED,
                "split": VoiceEvalSplit.DEVELOPMENT,
            }
        )
        for sample in ranked[:needed]
    ]
    return [*development, *replacements, *holdout]


def tag_voice_candidate_quality(
    candidates: list[VoiceEvalSampleV1],
    *,
    eval_root: Path,
) -> list[VoiceEvalSampleV1]:
    tagged: list[VoiceEvalSampleV1] = []
    for sample in candidates:
        normalized_sample = sample.model_copy(
            update={
                "provenance": sample.provenance.model_copy(
                    update={"network_scope": NetworkScope.NONE}
                )
            }
        )
        if {"prosody_available", "adverse_audio"} & set(sample.tags):
            tagged.append(normalized_sample)
            continue
        prosody = analyze_prosody(normalized_sample, eval_root=eval_root)
        tags = set(normalized_sample.tags)
        tags.add("prosody_available" if prosody.available else "adverse_audio")
        tagged.append(
            normalized_sample.model_copy(update={"tags": tuple(sorted(tags))})
        )
    return tagged


def validate_voice_eval_dataset(
    samples: list[VoiceEvalSampleV1],
    annotations: list[VoiceGoldAnnotationV1],
    *,
    target: int = 120,
    development_count: int = 30,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids = [sample.sample_id for sample in samples]
    if len(samples) != target:
        errors.append(f"sample_count:{len(samples)}!=target:{target}")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate_sample_ids")
    development = sum(sample.split is VoiceEvalSplit.DEVELOPMENT for sample in samples)
    holdout = sum(sample.split is VoiceEvalSplit.HOLDOUT for sample in samples)
    if development != development_count:
        errors.append(f"development_count:{development}!={development_count}")
    if holdout != target - development_count:
        errors.append(f"holdout_count:{holdout}!={target - development_count}")
    annotation_ids = [annotation.sample_id for annotation in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append("duplicate_annotation_ids")
    missing_annotations = sorted(set(sample_ids) - set(annotation_ids))
    unknown_annotations = sorted(set(annotation_ids) - set(sample_ids))
    if missing_annotations:
        errors.append(f"missing_annotations:{len(missing_annotations)}")
    if unknown_annotations:
        errors.append(f"unknown_annotations:{len(unknown_annotations)}")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    annotation_audio_mismatches = sorted(
        annotation.sample_id
        for annotation in annotations
        if annotation.sample_id in sample_by_id
        and sample_by_id[annotation.sample_id].audio is not None
        and annotation.audio_clip_sha256
        != sample_by_id[annotation.sample_id].audio.clip_sha256
    )
    if annotation_audio_mismatches:
        errors.append(
            f"annotation_audio_hash_mismatch:{len(annotation_audio_mismatches)}"
        )
    split_by_group: dict[str, set[VoiceEvalSplit]] = {}
    for sample in samples:
        if sample.audio is None:
            errors.append(f"missing_audio:{sample.sample_id}")
        group = sample.duplicate_group_id or (
            f"source-sha256:{sample.provenance.source_sha256}"
        )
        if sample.split is not None:
            split_by_group.setdefault(group, set()).add(sample.split)
    if any(len(splits) > 1 for splits in split_by_group.values()):
        errors.append("duplicate_group_crosses_splits")
    tag_counts = Counter(tag for sample in samples for tag in sample.tags)
    date_counts = Counter(
        sample.provenance.captured_start.date().isoformat() for sample in samples
    )
    if samples and max(date_counts.values(), default=0) / len(samples) > 0.50:
        warnings.append("temporal_concentration_over_50_percent")
    if samples and len(date_counts) < 5:
        warnings.append("fewer_than_5_capture_dates")
    for annotation in annotations:
        if annotation.intent.value == "question":
            tag_counts["question"] += 1
        elif annotation.intent.value == "task":
            tag_counts["task"] += 1
        elif annotation.intent.value == "cancellation":
            tag_counts["cancellation"] += 1
        elif annotation.intent.value == "barge_in":
            tag_counts["barge_in"] += 1
        if annotation.intent.value in {"conversation", "ambiguous", "question"}:
            tag_counts["conversation_boundary"] += 1
        if annotation.expected_outcome.value != "execute":
            tag_counts["non_action"] += 1
        if len(annotation.expected_action_ids) > 1:
            tag_counts["compound"] += 1
        if annotation.proper_names:
            tag_counts["proper_name"] += 1
        for tag in set(annotation.prosody_tags):
            if tag == "question_intonation":
                tag_counts["question_intonation_candidate"] += 1
            else:
                tag_counts[tag] += 1
    required_tags = {
        "question": 30,
        "question_intonation_candidate": 15,
        "task": 30,
        "compound": 20,
        "self_correction": 20,
        "cancellation": 15,
        "barge_in": 15,
        "conversation_boundary": 20,
        "proper_name": 20,
        "adverse_audio": 15,
        "non_action": 20,
    }
    deficits = {
        tag: minimum - tag_counts.get(tag, 0)
        for tag, minimum in required_tags.items()
        if tag_counts.get(tag, 0) < minimum
    }
    if deficits:
        errors.append("tag_coverage_incomplete")
    return {
        "schema_version": 1,
        "valid": not errors,
        "sample_count": len(samples),
        "development_count": development,
        "holdout_count": holdout,
        "annotation_count": len(annotations),
        "missing_annotation_ids": missing_annotations,
        "unknown_annotation_ids": unknown_annotations,
        "annotation_audio_mismatch_ids": annotation_audio_mismatches,
        "tag_counts": dict(sorted(tag_counts.items())),
        "tag_deficits": deficits,
        "date_counts": dict(sorted(date_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def derive_voice_tags(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    words = set(normalized.split())
    tags: set[str] = set()
    if words & _QUESTION_WORDS or text.rstrip().endswith("?"):
        tags.add("question")
    if words & _TASK_WORDS:
        tags.add("task")
    else:
        tags.update({"conversation_boundary", "non_action"})
    if any(marker in normalized for marker in _CORRECTION_MARKERS):
        tags.add("self_correction")
    if any(marker in normalized for marker in _CANCEL_MARKERS):
        tags.add("cancellation")
    if any(marker in f" {normalized} " for marker in (" i ", " potem ", " następnie ")):
        tags.add("compound")
    if any(name in normalized for name in _KNOWN_PROPER_NAMES):
        tags.add("proper_name")
    return tuple(sorted(tags))


def mark_cross_channel_duplicates(
    samples: list[VoiceEvalSampleV1],
    *,
    overlap_threshold: float = 0.8,
) -> list[VoiceEvalSampleV1]:
    result: list[VoiceEvalSampleV1] = []
    canonical: list[VoiceEvalSampleV1] = []
    canonical_by_hash: dict[str, VoiceEvalSampleV1] = {}
    for sample in sorted(samples, key=lambda item: item.provenance.captured_start):
        clip_hash = sample.audio.clip_sha256 if sample.audio is not None else ""
        exact_duplicate = canonical_by_hash.get(clip_hash) if clip_hash else None
        if exact_duplicate is not None:
            group_id = exact_duplicate.duplicate_group_id or exact_duplicate.sample_id
            result = [
                item.model_copy(update={"duplicate_group_id": group_id})
                if item.sample_id == exact_duplicate.sample_id
                else item
                for item in result
            ]
            result.append(
                sample.model_copy(
                    update={
                        "duplicate_group_id": group_id,
                        "duplicate_type": "exact_audio_hash",
                        "dedupe_confidence": 1.0,
                        "dedupe_reasons": ("identical_clip_sha256",),
                        "provenance": sample.provenance.model_copy(
                            update={"duplicate_of": exact_duplicate.sample_id}
                        ),
                    }
                )
            )
            continue
        duplicate: VoiceEvalSampleV1 | None = None
        for other in reversed(canonical[-30:]):
            if sample.provenance.audio_direction == other.provenance.audio_direction:
                continue
            overlap = _time_overlap_ratio(sample.provenance, other.provenance)
            if overlap >= overlap_threshold:
                duplicate = other
                break
        if duplicate is None:
            canonical.append(sample)
            if clip_hash:
                canonical_by_hash[clip_hash] = sample
            result.append(sample)
            continue
        group_id = duplicate.duplicate_group_id or duplicate.sample_id
        result = [
            item.model_copy(update={"duplicate_group_id": group_id})
            if item.sample_id == duplicate.sample_id
            else item
            for item in result
        ]
        result.append(
            sample.model_copy(
                update={
                    "duplicate_group_id": group_id,
                    "duplicate_type": "cross_channel_overlap",
                    "dedupe_confidence": 0.9,
                    "dedupe_reasons": ("time_overlap", "different_audio_direction"),
                    "provenance": sample.provenance.model_copy(
                        update={"duplicate_of": duplicate.sample_id}
                    ),
                }
            )
        )
    return result


def _segment_source(
    source: VoiceAudioSourceV1,
    *,
    path: Path,
    eval_root: Path,
    speaker_role: SpeakerRole,
    max_segments: int,
    ffmpeg_executable: str,
) -> list[VoiceEvalSampleV1]:
    if max_segments < 1:
        return []
    audio_dir = eval_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voiceloop-voice-") as temp_dir:
        decoded = Path(temp_dir) / "decoded.wav"
        _decode_audio(
            path,
            output=decoded,
            ffmpeg_executable=ffmpeg_executable,
        )
        decoded_duration = wav_duration_seconds(decoded)
        if source.source_offset_end_seconds > source.source_offset_start_seconds:
            segments = [
                (
                    max(0.0, source.source_offset_start_seconds - 0.15),
                    min(decoded_duration, source.source_offset_end_seconds + 0.15),
                )
            ]
        else:
            segments = detect_speech_segments(decoded)[:max_segments]
        decoder_version = _ffmpeg_version(ffmpeg_executable)
        samples: list[VoiceEvalSampleV1] = []
        for ordinal, (start_offset, end_offset) in enumerate(segments, start=1):
            end_offset = min(end_offset, decoded_duration)
            if end_offset <= start_offset:
                continue
            sample_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"voiceloop-voice-sample:{source.source_id}:{start_offset:.3f}:{end_offset:.3f}",
                )
            )
            clip_path = audio_dir / f"{sample_id}.wav"
            _extract_wav_segment(decoded, clip_path, start_offset, end_offset)
            duration = wav_duration_seconds(clip_path)
            relative_path = clip_path.relative_to(eval_root).as_posix()
            if source.source_offset_end_seconds > source.source_offset_start_seconds:
                captured_start = source.captured_start + timedelta(
                    seconds=start_offset - source.source_offset_start_seconds
                )
                captured_end = source.captured_end + timedelta(
                    seconds=end_offset - source.source_offset_end_seconds
                )
            else:
                captured_start = source.captured_start + timedelta(seconds=start_offset)
                captured_end = source.captured_start + timedelta(seconds=end_offset)
            provenance = ProvenanceV1(
                source_system="screenpipe",
                source_id=source.source_id,
                source_record_id=f"{source.chunk_id}:{ordinal}",
                session_id=f"screenpipe:{source.captured_start.date().isoformat()}",
                captured_start=captured_start,
                captured_end=captured_end,
                audio_direction=source.audio_direction,
                device_name=source.device_name,
                device_type=source.device_type,
                speaker_role=speaker_role,
                source_sha256=source.sha256,
            )
            sample = VoiceEvalSampleV1(
                sample_id=sample_id,
                provenance=provenance,
                audio=AudioAssetRefV1(
                    relative_path=relative_path,
                    source_chunk_id=source.chunk_id,
                    original_sha256=source.sha256,
                    clip_sha256=sha256_file(clip_path),
                    duration_seconds=duration,
                    source_offset_start_seconds=start_offset,
                    source_offset_end_seconds=end_offset,
                    decoder_version=decoder_version,
                ),
                tags=(),
                state=VoiceSampleState.CANDIDATE,
            )
            prosody = analyze_prosody(sample, eval_root=eval_root)
            samples.append(
                sample.model_copy(
                    update={
                        "tags": (
                            ("prosody_available",)
                            if prosody.available
                            else ("adverse_audio",)
                        )
                    }
                )
            )
        return samples


def detect_speech_segments(
    wav_path: Path,
    *,
    frame_seconds: float = 0.03,
    min_speech_seconds: float = 0.45,
    max_speech_seconds: float = 25.0,
    split_silence_seconds: float = 0.55,
    padding_seconds: float = 0.15,
) -> list[tuple[float, float]]:
    samples, sample_rate = _read_pcm16_mono(wav_path)
    if not samples:
        return []
    frame_size = max(1, round(sample_rate * frame_seconds))
    rms_values: list[float] = []
    for index in range(0, len(samples), frame_size):
        frame = samples[index : index + frame_size]
        if not frame:
            continue
        rms_values.append(math.sqrt(sum(value * value for value in frame) / len(frame)))
    if not rms_values:
        return []
    ordered = sorted(rms_values)
    noise = ordered[max(0, int(len(ordered) * 0.2) - 1)]
    peak = max(rms_values)
    threshold = max(noise * 2.5, peak * 0.08, 100.0)
    active = [value >= threshold for value in rms_values]
    max_silent_frames = max(1, round(split_silence_seconds / frame_seconds))
    segments: list[tuple[float, float]] = []
    start_frame: int | None = None
    last_active = 0
    for index, is_active in enumerate(active):
        if is_active:
            if start_frame is None:
                start_frame = index
            last_active = index
        elif start_frame is not None and index - last_active > max_silent_frames:
            _append_segment(
                segments,
                start_frame,
                last_active + 1,
                frame_seconds=frame_seconds,
                min_seconds=min_speech_seconds,
                max_seconds=max_speech_seconds,
                padding_seconds=padding_seconds,
                total_seconds=len(samples) / sample_rate,
            )
            start_frame = None
    if start_frame is not None:
        _append_segment(
            segments,
            start_frame,
            last_active + 1,
            frame_seconds=frame_seconds,
            min_seconds=min_speech_seconds,
            max_seconds=max_speech_seconds,
            padding_seconds=padding_seconds,
            total_seconds=len(samples) / sample_rate,
        )
    return segments


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _decode_audio(path: Path, *, output: Path, ffmpeg_executable: str) -> None:
    executable = shutil.which(ffmpeg_executable)
    if executable is None:
        raise VoiceEvalError("Brak ffmpeg w PATH; nie można zamrozić audio.")
    result = subprocess.run(
        [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0 or not output.is_file():
        raise VoiceEvalError(f"ffmpeg nie zdekodował audio: {result.stderr[:500]}")


def _ffmpeg_version(ffmpeg_executable: str) -> str:
    executable = shutil.which(ffmpeg_executable)
    if executable is None:
        return ""
    try:
        result = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout.splitlines() or [""])[0][:200]


def _extract_wav_segment(source: Path, target: Path, start: float, end: float) -> None:
    with wave.open(str(source), "rb") as reader:
        frame_rate = reader.getframerate()
        start_frame = max(0, round(start * frame_rate))
        end_frame = min(reader.getnframes(), round(end * frame_rate))
        reader.setpos(start_frame)
        frames = reader.readframes(max(0, end_frame - start_frame))
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(frame_rate)
            writer.writeframes(frames)


def _read_pcm16_mono(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise VoiceEvalError("Analiza segmentów wymaga PCM 16-bit mono.")
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = [
        int.from_bytes(raw[index : index + 2], byteorder="little", signed=True)
        for index in range(0, len(raw) - 1, 2)
    ]
    return samples, sample_rate


def _append_segment(
    segments: list[tuple[float, float]],
    start_frame: int,
    end_frame: int,
    *,
    frame_seconds: float,
    min_seconds: float,
    max_seconds: float,
    padding_seconds: float,
    total_seconds: float,
) -> None:
    start = max(0.0, start_frame * frame_seconds - padding_seconds)
    end = min(total_seconds, end_frame * frame_seconds + padding_seconds)
    if end - start < min_seconds:
        return
    while end - start > max_seconds:
        segments.append((start, start + max_seconds))
        start += max_seconds
    if end - start >= min_seconds:
        segments.append((start, end))


def _rank_by_coverage(
    samples: list[VoiceEvalSampleV1],
    *,
    target: int,
) -> list[VoiceEvalSampleV1]:
    tag_counts: Counter[str] = Counter()
    ordered = sorted(
        samples,
        key=lambda item: (
            item.provenance.captured_start.date().isoformat(),
            item.provenance.device_name.casefold(),
            item.sample_id,
        ),
    )
    short = [item for item in ordered if item.audio and item.audio.duration_seconds < 0.8]
    medium = [
        item
        for item in ordered
        if item.audio and 0.8 <= item.audio.duration_seconds < 2.0
    ]
    long = [item for item in ordered if item.audio and item.audio.duration_seconds >= 2.0]
    adverse = [item for item in ordered if "adverse_audio" in item.tags]
    adverse_quota = min(len(adverse), max(1, round(target * 0.125)))
    selected = [*_spread_samples(adverse)[:adverse_quota]]
    selected_ids = {item.sample_id for item in selected}
    duration_groups = (
        (short, max(1, round(target * 0.125))),
        (long, max(1, round(target * 0.25))),
        (medium, target),
    )
    for group, desired_total in duration_groups:
        current = sum(item in group for item in selected)
        preferred_group = [
            item for item in group if "adverse_audio" not in item.tags
        ]
        needed = min(
            max(0, desired_total - current),
            target - len(selected),
        )
        selected.extend(
            item
            for item in _spread_samples(preferred_group)
            if item.sample_id not in selected_ids
        )
        if needed < len(selected) - len(selected_ids):
            selected = selected[: len(selected_ids) + needed]
        selected_ids = {item.sample_id for item in selected}
        if len(selected) >= target:
            break
    remaining = [item for item in ordered if item.sample_id not in selected_ids]
    tag_counts.update(tag for item in selected for tag in set(item.tags))
    required = {
        "question": 30,
        "question_intonation_candidate": 15,
        "task": 30,
        "compound": 20,
        "self_correction": 20,
        "cancellation": 15,
        "barge_in": 15,
        "conversation_boundary": 20,
        "proper_name": 20,
        "adverse_audio": 15,
        "non_action": 20,
    }
    while remaining:
        best = max(
            remaining,
            key=lambda item: (
                int(
                    "adverse_audio" not in item.tags
                    or tag_counts["adverse_audio"] < required["adverse_audio"]
                ),
                sum(
                    max(0, required.get(tag, 0) - tag_counts[tag])
                    for tag in set(item.tags)
                ),
                -sum(tag_counts[tag] for tag in set(item.tags)),
                item.sample_id,
            ),
        )
        remaining.remove(best)
        selected.append(best)
        tag_counts.update(set(best.tags))
    return selected


def _development_sample_ids(
    samples: list[VoiceEvalSampleV1],
    target_count: int,
) -> set[str]:
    groups: dict[str, list[VoiceEvalSampleV1]] = {}
    for sample in samples:
        group_key = sample.duplicate_group_id or (
            f"source-sha256:{sample.provenance.source_sha256}"
        )
        groups.setdefault(group_key, []).append(sample)
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: sha256_text(group[0].provenance.session_id or group[0].sample_id),
    )
    target_adverse = round(
        sum("adverse_audio" in sample.tags for sample in samples)
        * target_count
        / max(1, len(samples))
    )
    reachable: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for group_index, group in enumerate(ordered_groups):
        group_adverse = sum("adverse_audio" in sample.tags for sample in group)
        for (current_count, current_adverse), chosen in sorted(
            tuple(reachable.items()),
            reverse=True,
        ):
            new_count = current_count + len(group)
            new_adverse = current_adverse + group_adverse
            state = (new_count, new_adverse)
            if new_count > target_count or state in reachable:
                continue
            reachable[state] = (*chosen, group_index)
    candidates = [
        (abs(adverse_count - target_adverse), adverse_count, chosen)
        for (count, adverse_count), chosen in reachable.items()
        if count == target_count
    ]
    if not candidates:
        raise VoiceEvalError("Nie można utworzyć dokładnego splitu bez rozdzielenia grup.")
    _, _, chosen_groups = min(candidates)
    return {
        sample.sample_id
        for group_index in chosen_groups
        for sample in ordered_groups[group_index]
    }


def _voice_split_group(sample: VoiceEvalSampleV1) -> str:
    return sample.duplicate_group_id or f"source-sha256:{sample.provenance.source_sha256}"


def _rank_speech_replacements(
    samples: list[VoiceEvalSampleV1],
) -> list[VoiceEvalSampleV1]:
    groups: dict[str, list[VoiceEvalSampleV1]] = {}
    for sample in samples:
        day = sample.provenance.captured_start.date().isoformat()
        groups.setdefault(day, []).append(sample)
    for group in groups.values():
        group.sort(
            key=lambda item: (
                -min(item.audio.duration_seconds if item.audio else 0.0, 8.0),
                item.sample_id,
            )
        )
    ordered_days = sorted(groups, key=sha256_text)
    ranked: list[VoiceEvalSampleV1] = []
    index = 0
    while True:
        added = False
        for day in ordered_days:
            group = groups[day]
            if index < len(group):
                ranked.append(group[index])
                added = True
        if not added:
            return ranked
        index += 1


def _spread_sources(sources: list[VoiceAudioSourceV1]) -> list[VoiceAudioSourceV1]:
    groups: dict[tuple[str, str], list[VoiceAudioSourceV1]] = {}
    for source in sources:
        key = (
            source.captured_start.date().isoformat(),
            source.device_name.casefold(),
        )
        groups.setdefault(key, []).append(source)
    for group in groups.values():
        group.sort(key=lambda item: (item.captured_start, item.source_id))
    ordered_keys = sorted(
        groups,
        key=lambda key: sha256_text(":".join(key)),
    )
    spread: list[VoiceAudioSourceV1] = []
    index = 0
    while True:
        added = False
        for key in ordered_keys:
            group = groups[key]
            if index < len(group):
                spread.append(group[index])
                added = True
        if not added:
            return spread
        index += 1


def _spread_samples(samples: list[VoiceEvalSampleV1]) -> list[VoiceEvalSampleV1]:
    groups: dict[str, list[VoiceEvalSampleV1]] = {}
    for sample in samples:
        day = sample.provenance.captured_start.date().isoformat()
        groups.setdefault(day, []).append(sample)
    for group in groups.values():
        group.sort(
            key=lambda item: sha256_text(
                f"{item.provenance.session_id}:{item.sample_id}"
            )
        )
    ordered_days = sorted(groups, key=sha256_text)
    spread: list[VoiceEvalSampleV1] = []
    index = 0
    while True:
        added = False
        for day in ordered_days:
            group = groups[day]
            if index < len(group):
                spread.append(group[index])
                added = True
        if not added:
            return spread
        index += 1


def _time_overlap_ratio(left: ProvenanceV1, right: ProvenanceV1) -> float:
    overlap_start = max(left.captured_start, right.captured_start)
    overlap_end = min(left.captured_end, right.captured_end)
    overlap = max(0.0, (overlap_end - overlap_start).total_seconds())
    shortest = min(
        (left.captured_end - left.captured_start).total_seconds(),
        (right.captured_end - right.captured_start).total_seconds(),
    )
    return overlap / shortest if shortest > 0 else 0.0


def _safe_timestamp(value: str, *, fallback: datetime) -> datetime:
    try:
        return parse_timestamp(value)
    except VoiceEvalError:
        return fallback


def _is_safe_audio_path(path: Path, *, screenpipe_root: Path) -> bool:
    try:
        safe_screenpipe_audio_path(path, screenpipe_root=screenpipe_root)
    except VoiceEvalError:
        return False
    return True
