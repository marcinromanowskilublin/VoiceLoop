from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .schema import SourceKind, SourceManifest, SourceManifestRecord
from .storage import sha256_text, word_count

_ISO_DAY_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_EXCLUDED_PATH_PARTS = {"subagents", "terminals", "mcps"}


def build_manifest(
    *,
    audio_transcript: Path | None,
    cursor_projects_root: Path | None,
) -> SourceManifest:
    records: list[SourceManifestRecord] = []
    if audio_transcript is not None and audio_transcript.is_file():
        records.append(
            inspect_text_source(
                audio_transcript,
                kind=SourceKind.AUDIO_TRANSCRIPT,
            )
        )
    if cursor_projects_root is not None and cursor_projects_root.is_dir():
        records.extend(inspect_cursor_sources(cursor_projects_root))

    seen_hashes: dict[str, str] = {}
    deduplicated: list[SourceManifestRecord] = []
    for record in sorted(records, key=lambda item: (item.kind.value, item.path.casefold())):
        duplicate_of = seen_hashes.get(record.sha256)
        if duplicate_of:
            record = record.model_copy(
                update={
                    "included": False,
                    "exclude_reason": f"duplicate_source:{duplicate_of}",
                }
            )
        elif record.included:
            seen_hashes[record.sha256] = record.source_id
        deduplicated.append(record)

    included_words = sum(item.word_count for item in deduplicated if item.included)
    excluded_words = sum(item.word_count for item in deduplicated if not item.included)
    fingerprint = json.dumps(
        [
            {
                "source_id": item.source_id,
                "sha256": item.sha256,
                "included": item.included,
                "exclude_reason": item.exclude_reason,
            }
            for item in deduplicated
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SourceManifest(
        manifest_id=sha256_text(fingerprint)[:20],
        sources=deduplicated,
        included_word_count=included_words,
        excluded_word_count=excluded_words,
        unique_source_count=sum(item.included for item in deduplicated),
    )


def inspect_cursor_sources(root: Path) -> list[SourceManifestRecord]:
    records: list[SourceManifestRecord] = []
    for path in sorted(root.rglob("*.jsonl")):
        lower_parts = {part.casefold() for part in path.parts}
        if lower_parts & _EXCLUDED_PATH_PARTS:
            continue
        if "agent-transcripts" not in lower_parts:
            continue
        records.append(inspect_text_source(path, kind=SourceKind.CURSOR_USER))
    return records


def inspect_text_source(path: Path, *, kind: SourceKind) -> SourceManifestRecord:
    resolved = path.resolve()
    raw_bytes = resolved.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    days = _extract_days(raw_text)
    stat = resolved.stat()
    excluded = _is_overlap_chunk(resolved)
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    return SourceManifestRecord(
        source_id=str(uuid5(NAMESPACE_URL, f"voiceloop-corpus:{kind.value}:{file_hash}")),
        kind=kind,
        path=str(resolved),
        sha256=file_hash,
        byte_size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        word_count=word_count(raw_text),
        date_start=min(days) if days else None,
        date_end=max(days) if days else None,
        included=not excluded,
        exclude_reason="overlap_chunk" if excluded else None,
    )


def _extract_days(text: str) -> list[date]:
    days: set[date] = set()
    for value in _ISO_DAY_PATTERN.findall(text):
        try:
            days.add(date.fromisoformat(value))
        except ValueError:
            continue
    return sorted(days)


def _is_overlap_chunk(path: Path) -> bool:
    lower_parts = [part.casefold() for part in path.parts]
    return (
        "chunks" in lower_parts
        and "output" in lower_parts
        and path.name.casefold().startswith("chunk_")
    )
