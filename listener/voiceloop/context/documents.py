"""Explicit document ingest into the timeline. No watcher and no vectors."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..corpus.privacy import redact_text
from ..memory import MemoryStore
from .schema import ContextEventV1

ALLOWED_DOCUMENT_SUFFIXES = frozenset({".md", ".txt", ".rst", ".markdown"})
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        ".codex-tmp",
        "data",
    }
)
SECRET_FILE_NAMES = frozenset(
    {".env", ".env.local", "credentials.json", "secrets.json", "id_rsa"}
)
SECRET_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx"})


class DocumentAccessError(RuntimeError):
    """A stored reference no longer resolves inside the configured roots."""


@dataclass(frozen=True)
class DocumentIngestReport:
    scanned: int
    stored_events: int
    skipped: int
    skipped_secret: int = 0
    skipped_unreadable: int = 0


class DocumentTimelineIngestor:
    """Stores a digest plus a reference; the full text stays on disk."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        roots: list[Path],
        allowed_suffixes: frozenset[str] | None = None,
        max_files: int = 100,
        max_bytes: int = 65_536,
        digest_chars: int = 2000,
        ttl_days: int = 14,
    ) -> None:
        self.memory = memory
        self.roots = [root.resolve() for root in roots]
        self.allowed_suffixes = allowed_suffixes or ALLOWED_DOCUMENT_SUFFIXES
        self.max_files = max(1, min(int(max_files), 1000))
        self.max_bytes = max(1, min(int(max_bytes), 262_144))
        self.digest_chars = max(200, min(int(digest_chars), 20000))
        self.ttl_days = max(0, min(int(ttl_days), 3650))

    async def ingest(self, *, observed_at: datetime | None = None) -> DocumentIngestReport:
        moment = _aware(observed_at or datetime.now(UTC))
        paths = await asyncio.to_thread(self._collect)
        stored = 0
        reasons: dict[str, int] = {}
        for path in paths:
            event, reason = await asyncio.to_thread(self._event_for_path, path, moment)
            if event is None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            await self.memory.upsert_context_event(event)
            stored += 1
        return DocumentIngestReport(
            scanned=len(paths),
            stored_events=stored,
            skipped=sum(reasons.values()),
            skipped_secret=reasons.get("secret", 0),
            skipped_unreadable=reasons.get("unreadable", 0),
        )

    async def load_full_text(self, event: ContextEventV1) -> str:
        """Re-read on demand. Validates against current roots, not stored metadata."""

        raw_path = str(event.metadata.get("document_path") or "")
        if not raw_path:
            raise DocumentAccessError("Zdarzenie nie ma odnośnika do dokumentu.")
        path = Path(raw_path)
        if not self._within_roots(path) or path.suffix.lower() not in self.allowed_suffixes:
            raise DocumentAccessError("Odnośnik wskazuje poza dozwolone korzenie.")
        text = await asyncio.to_thread(self._read_text, path)
        if text is None:
            raise DocumentAccessError("Nie można odczytać dokumentu z dysku.")
        redacted, _ = _redact_document(text)
        return redacted

    def _collect(self) -> list[Path]:
        """Prune skipped directories while walking so `.git` is never read."""

        found: list[Path] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            for current, directories, file_names in os.walk(root):
                directories[:] = [
                    name for name in directories if name not in SKIPPED_DIRECTORIES
                ]
                for file_name in sorted(file_names):
                    if len(found) >= self.max_files:
                        return found
                    path = Path(current) / file_name
                    if self._allowed(path):
                        found.append(path)
        return found

    def _allowed(self, path: Path) -> bool:
        name = path.name.lower()
        if name in SECRET_FILE_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
            return False
        return path.suffix.lower() in self.allowed_suffixes

    def _within_roots(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(resolved.is_relative_to(root) for root in self.roots)

    def _read_text(self, path: Path) -> str | None:
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if len(raw) > self.max_bytes or b"\x00" in raw[:4096]:
            return None
        return raw.decode("utf-8", errors="replace").strip() or None

    def _event_for_path(
        self,
        path: Path,
        observed_at: datetime,
    ) -> tuple[ContextEventV1 | None, str]:
        """Return the event, or `None` with the reason the document was skipped."""

        try:
            stat = path.stat()
        except OSError:
            return None, "unreadable"
        text = self._read_text(path)
        if text is None:
            return None, "unreadable"
        redacted, flags = _redact_document(text)
        if "secret" in flags or "pesel" in flags:
            # Redaction already removed the match, but a file that carries one
            # secret may carry others in shapes the patterns do not cover.
            return None, "secret"
        digest, truncated = _digest(redacted, limit=self.digest_chars)
        resolved = path.resolve()
        source_id = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
        # The timeline axis is when the document changed, not when we scanned it.
        # Scan time would move on every run and rewrite the event content hash.
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        event = ContextEventV1.from_observation(
            source="local_document",
            source_id=source_id,
            started_at=min(modified_at, observed_at),
            event_type="document",
            window_title=_document_title(redacted, fallback=path.name),
            text=digest,
            metadata={
                "relative_name": path.name,
                "suffix": resolved.suffix.lower(),
                # Reference for on-demand reads. Metadata never enters the prompt,
                # which serializes source, time, trust, title and content only.
                "document_path": str(resolved),
                "path_sha256": source_id,
                "byte_size": stat.st_size,
                "digest_truncated": truncated,
                "digest_method": "deterministic_document_v1",
                "redaction_flags": flags,
                "observed_at": observed_at.isoformat(),
                "vectorize": False,
            },
            sensitivity="private",
            expires_at=(
                observed_at + timedelta(days=self.ttl_days) if self.ttl_days else None
            ),
        )
        return event, ""


def _digest(text: str, *, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    boundary = head.rfind("\n")
    if boundary > limit // 2:
        head = head[:boundary]
    return f"{head.rstrip()}\n[...]", True


def _document_title(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        heading = line.strip()
        if heading.startswith("#"):
            stripped = heading.lstrip("#").strip()
            if stripped:
                return stripped[:1000]
        if heading:
            break
    return fallback[:1000]


def _redact_document(text: str) -> tuple[str, list[str]]:
    """Redact per line so headings and lists survive the privacy pass."""

    lines: list[str] = []
    flags: list[str] = []
    for line in text.splitlines():
        redacted, line_flags = redact_text(line)
        lines.append(redacted)
        for flag in line_flags:
            if flag not in flags:
                flags.append(flag)
    return "\n".join(lines).strip(), flags


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
