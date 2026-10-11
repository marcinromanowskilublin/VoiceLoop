"""Explicit document ingest into the timeline. No watcher and no vectors."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
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
        result = await asyncio.to_thread(self._secure_read, Path(raw_path))
        if result is None:
            raise DocumentAccessError("Odnośnik wskazuje poza dozwolone korzenie.")
        _, _, _, text = result
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

    def _secure_read(
        self,
        path: Path,
    ) -> tuple[Path, float, int, str] | None:
        candidate = Path(os.path.abspath(path))
        root = next(
            (root for root in self.roots if candidate.is_relative_to(root)),
            None,
        )
        if root is None or not self._allowed(candidate):
            return None
        if os.name == "nt":
            return self._secure_read_windows(candidate, root)
        relative = candidate.relative_to(root)
        if not relative.parts:
            return None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptors: list[int] = []
        try:
            current = os.open(root, directory_flags | nofollow)
            descriptors.append(current)
            for component in relative.parts[:-1]:
                current = os.open(
                    component,
                    directory_flags | nofollow,
                    dir_fd=current,
                )
                descriptors.append(current)
            file_descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | nofollow,
                dir_fd=current,
            )
            descriptors.append(file_descriptor)
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                return None
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(file_descriptor, min(65_536, self.max_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > self.max_bytes:
                    return None
            raw = b"".join(chunks)
        except OSError:
            return None
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
        if b"\x00" in raw[:4096]:
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        return (candidate, file_stat.st_mtime, file_stat.st_size, text) if text else None

    def _secure_read_windows(
        self,
        path: Path,
        root: Path,
    ) -> tuple[Path, float, int, str] | None:
        import pywintypes
        import win32con
        import win32file

        file_flag_open_reparse_point = 0x00200000
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        handle = None
        try:
            handle = win32file.CreateFile(
                str(path),
                win32con.GENERIC_READ,
                (
                    win32con.FILE_SHARE_READ
                    | win32con.FILE_SHARE_WRITE
                    | win32con.FILE_SHARE_DELETE
                ),
                None,
                win32con.OPEN_EXISTING,
                file_flag_open_reparse_point,
                None,
            )
            information = win32file.GetFileInformationByHandle(handle)
            attributes = int(information[0])
            if attributes & (
                win32con.FILE_ATTRIBUTE_DIRECTORY | reparse_flag
            ):
                return None
            final_path = str(win32file.GetFinalPathNameByHandle(handle, 0))
            if final_path.startswith("\\\\?\\"):
                final_path = final_path[4:]
            normalized_root = os.path.normcase(os.path.abspath(root))
            normalized_final = os.path.normcase(os.path.abspath(final_path))
            try:
                common = os.path.commonpath((normalized_root, normalized_final))
            except ValueError:
                return None
            if common != normalized_root:
                return None
            chunks: list[bytes] = []
            size = 0
            while size <= self.max_bytes:
                _, chunk = win32file.ReadFile(
                    handle,
                    min(65_536, self.max_bytes + 1 - size),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            raw = b"".join(chunks)
            modified_timestamp = information[3].timestamp()
            byte_size = (int(information[5]) << 32) | int(information[6])
        except (OSError, pywintypes.error):
            return None
        finally:
            if handle is not None:
                handle.Close()
        if len(raw) > self.max_bytes or b"\x00" in raw[:4096]:
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        return (
            (Path(final_path), modified_timestamp, byte_size, text)
            if text
            else None
        )

    def _event_for_path(
        self,
        path: Path,
        observed_at: datetime,
    ) -> tuple[ContextEventV1 | None, str]:
        """Return the event, or `None` with the reason the document was skipped."""

        result = self._secure_read(path)
        if result is None:
            return None, "unreadable"
        resolved, modified_timestamp, byte_size, text = result
        if not self._allowed(resolved):
            return None, "secret"
        redacted, flags = _redact_document(text)
        if "secret" in flags or "pesel" in flags:
            # Redaction already removed the match, but a file that carries one
            # secret may carry others in shapes the patterns do not cover.
            return None, "secret"
        digest, truncated = _digest(redacted, limit=self.digest_chars)
        source_id = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
        # The timeline axis is when the document changed, not when we scanned it.
        # Scan time would move on every run and rewrite the event content hash.
        modified_at = datetime.fromtimestamp(modified_timestamp, tz=UTC)
        event = ContextEventV1.from_observation(
            source="local_document",
            source_id=source_id,
            started_at=min(modified_at, observed_at),
            event_type="document",
            window_title=_document_title(redacted, fallback=resolved.name),
            text=digest,
            metadata={
                "relative_name": path.name,
                "suffix": resolved.suffix.lower(),
                # Reference for on-demand reads. Metadata never enters the prompt,
                # which serializes source, time, trust, title and content only.
                "document_path": str(resolved),
                "path_sha256": source_id,
                "byte_size": byte_size,
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
