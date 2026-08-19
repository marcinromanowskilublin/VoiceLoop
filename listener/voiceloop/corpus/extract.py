from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..router import normalize_text
from .schema import (
    SourceKind,
    SourceManifest,
    SpeakerStatus,
    UtteranceOrigin,
    UtteranceRecord,
)
from .storage import sha256_text, word_count

_USER_QUERY_PATTERN = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)
_TIMESTAMP_PATTERN = re.compile(r"<timestamp>\s*(.*?)\s*</timestamp>", re.DOTALL)
_AUTO_BLOCK_NAMES = (
    "user_info",
    "git_status",
    "system_reminder",
    "agent_transcripts",
    "rules",
    "image_files",
    "attached_files",
)
_AUTO_BLOCK_PATTERN = re.compile(
    rf"<({'|'.join(_AUTO_BLOCK_NAMES)})\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_STRUCTURED_LINE_PATTERN = re.compile(
    r"^\s*(?:[{}\[\]]|Traceback\b|File \".*\", line \d+|[A-Za-z_][\w.-]*\s*[=:])"
)
_CODE_MARKERS = (
    "```",
    "diff --git",
    "*** begin patch",
    "import ",
    "def ",
    "class ",
    "function ",
    "const ",
    "let ",
    "var ",
)
_AUDIO_HEADER_PATTERN = re.compile(r"^\s*#")
_ISO_DAY_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


@dataclass(frozen=True)
class ExtractionResult:
    records: list[UtteranceRecord]
    technical_excluded_count: int
    invalid_record_count: int


class SourceChangedError(RuntimeError):
    pass


def extract_from_manifest(
    manifest: SourceManifest,
    *,
    trusted_audio_source_ids: set[str] | None = None,
) -> ExtractionResult:
    trusted_audio_source_ids = trusted_audio_source_ids or set()
    records: list[UtteranceRecord] = []
    technical_count = 0
    invalid_count = 0
    for source in manifest.sources:
        if not source.included:
            continue
        path = Path(source.path)
        snapshot = path.read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != source.sha256:
            raise SourceChangedError(
                f"Źródło {source.source_id} zmieniło się po utworzeniu manifestu."
            )
        text = snapshot.decode("utf-8", errors="replace")
        if source.kind is SourceKind.CURSOR_USER:
            result = _extract_cursor_text(text, path, source.source_id)
        elif source.kind is SourceKind.AUDIO_TRANSCRIPT:
            result = _extract_audio_text(
                text,
                source.source_id,
                speaker_status=(
                    SpeakerStatus.SELF
                    if source.source_id in trusted_audio_source_ids
                    else SpeakerStatus.UNKNOWN
                ),
            )
        else:
            continue
        records.extend(result.records)
        technical_count += result.technical_excluded_count
        invalid_count += result.invalid_record_count
    return ExtractionResult(records, technical_count, invalid_count)


def extract_cursor_file(path: Path, source_id: str) -> ExtractionResult:
    snapshot = path.read_bytes()
    return _extract_cursor_text(
        snapshot.decode("utf-8", errors="replace"),
        path,
        source_id,
    )


def _extract_cursor_text(
    text: str,
    path: Path,
    source_id: str,
) -> ExtractionResult:
    records: list[UtteranceRecord] = []
    technical_count = 0
    invalid_count = 0
    session_id = _cursor_session_id(path)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            invalid_count += 1
            continue
        if payload.get("role") != "user":
            continue
        full_text = _message_text(payload)
        if not full_text:
            continue
        query_match = _USER_QUERY_PATTERN.search(full_text)
        user_text = (
            query_match.group(1).strip()
            if query_match
            else _strip_auto_blocks(full_text)
        )
        if not user_text or user_text.startswith("[Previous conversation summary]"):
            continue
        if _is_technical_or_long(user_text):
            technical_count += 1
            continue
        captured_at = _message_timestamp(full_text)
        records.append(
            _utterance(
                source_id=source_id,
                origin=UtteranceOrigin.CURSOR_USER,
                session_id=session_id,
                ordinal=line_number,
                text=user_text,
                speaker_status=SpeakerStatus.SELF,
                captured_at=captured_at,
            )
        )
    return ExtractionResult(records, technical_count, invalid_count)


def extract_audio_file(
    path: Path,
    source_id: str,
    *,
    speaker_status: SpeakerStatus = SpeakerStatus.UNKNOWN,
) -> ExtractionResult:
    snapshot = path.read_bytes()
    return _extract_audio_text(
        snapshot.decode("utf-8", errors="replace"),
        source_id,
        speaker_status=speaker_status,
    )


def _extract_audio_text(
    text: str,
    source_id: str,
    *,
    speaker_status: SpeakerStatus,
) -> ExtractionResult:
    current_day = "unknown"
    blocks: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        value = " ".join(part.strip() for part in buffer if part.strip()).strip()
        buffer.clear()
        if value:
            blocks.extend((current_day, chunk) for chunk in _split_long_text(value))

    for line in text.splitlines():
        day_match = _ISO_DAY_PATTERN.search(line)
        if _AUDIO_HEADER_PATTERN.match(line):
            flush()
            if day_match:
                current_day = day_match.group(1)
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()

    records = [
        _utterance(
            source_id=source_id,
            origin=UtteranceOrigin.AUDIO,
            session_id=f"audio:{day}",
            ordinal=index,
            text=value,
            speaker_status=speaker_status,
            captured_at=None,
        )
        for index, (day, value) in enumerate(blocks, start=1)
    ]
    return ExtractionResult(records, 0, 0)


def _cursor_session_id(path: Path) -> str:
    stable_path_hash = sha256_text(str(path.resolve()).casefold())[:20]
    return f"cursor:{stable_path_hash}"


def _utterance(
    *,
    source_id: str,
    origin: UtteranceOrigin,
    session_id: str,
    ordinal: int,
    text: str,
    speaker_status: SpeakerStatus,
    captured_at: datetime | None,
) -> UtteranceRecord:
    compact = " ".join(text.split())[:20000]
    normalized = normalize_text(compact)
    text_hash = sha256_text(normalized)
    return UtteranceRecord(
        utterance_id=str(
            uuid5(
                NAMESPACE_URL,
                f"voiceloop-corpus:{source_id}:{session_id}:{ordinal}:{text_hash}",
            )
        ),
        source_id=source_id,
        origin=origin,
        session_id=session_id,
        captured_at=captured_at,
        word_count=word_count(compact),
        char_count=len(compact),
        text_sha256=text_hash,
        text=compact,
        speaker_status=speaker_status,
    )


def _message_text(payload: dict) -> str:
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = payload.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _strip_auto_blocks(text: str) -> str:
    without_blocks = _AUTO_BLOCK_PATTERN.sub("", text)
    without_timestamp = _TIMESTAMP_PATTERN.sub("", without_blocks)
    return without_timestamp.strip()


def _message_timestamp(text: str) -> datetime | None:
    match = _TIMESTAMP_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _is_technical_or_long(text: str) -> bool:
    if word_count(text) > 500:
        return True
    lowered = text.casefold()
    if any(marker in lowered for marker in _CODE_MARKERS):
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    structured = sum(bool(_STRUCTURED_LINE_PATTERN.match(line)) for line in lines)
    return structured / len(lines) >= 0.35


def _split_long_text(text: str, *, max_words: int = 350) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [
        " ".join(words[offset : offset + max_words])
        for offset in range(0, len(words), max_words)
    ]
