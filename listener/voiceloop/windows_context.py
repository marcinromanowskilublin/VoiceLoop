from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .memory import MemoryStore
from .screenpipe import ScreenpipeClient, ScreenpipeElement
from .settings import Settings

LOGGER = logging.getLogger("voiceloop.windows_context")
PROJECT_MARKERS = (".git", "pyproject.toml", "package.json", "Cargo.toml")


def _now() -> datetime:
    return datetime.now(UTC)


def normalize_local_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass(frozen=True)
class LocalPathRecord:
    name: str
    path: str
    is_dir: bool
    root: str
    first_seen: datetime
    last_seen: datetime
    exists: bool
    is_project: bool = False
    project_type: str = ""


@dataclass(frozen=True)
class LocalPathEvent:
    event: str
    path: str
    timestamp: datetime
    previous_path: str = ""


@dataclass(frozen=True)
class CachedScreenpipeElement:
    element: ScreenpipeElement
    seen_at: datetime


class WindowsContextService:
    """Session-only context over explicitly configured local roots."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        screenpipe: ScreenpipeClient,
        *,
        roots: list[Path] | None = None,
    ) -> None:
        self.enabled = settings.windows_context_enabled
        self.timeline_projection_enabled = (
            settings.context_timeline_windows_projection_enabled
        )
        self.timeline_ttl_days = settings.context_timeline_auto_ttl_days
        self.memory = memory
        self.screenpipe = screenpipe
        self.roots = roots if roots is not None else self._configured_roots(settings)
        self.reconcile_seconds = max(10, settings.windows_context_reconcile_seconds)
        self.context_ttl_seconds = max(10, settings.windows_context_ttl_seconds)
        self.retention_days = max(1, settings.windows_context_retention_days)
        self.max_records = max(10, settings.windows_context_max_records)
        self.records: dict[str, LocalPathRecord] = {}
        self.journal: deque[LocalPathEvent] = deque(
            maxlen=max(10, settings.windows_context_journal_limit)
        )
        self.screenpipe_elements: deque[CachedScreenpipeElement] = deque(
            maxlen=max(10, settings.windows_context_screenpipe_cache_limit)
        )
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _configured_roots(settings: Settings) -> list[Path]:
        raw = settings.windows_context_roots.strip()
        candidates = (
            [Path(os.path.expandvars(value.strip())).expanduser() for value in raw.split(";")]
            if raw
            else [Path.home() / "Desktop", settings.project_root]
        )
        roots: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = normalize_local_path(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            roots.append(Path(os.path.abspath(candidate)))
        return roots

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        await self.reconcile()
        self._task = asyncio.create_task(self._run(), name="windows-context")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            from watchfiles import awatch
        except ImportError:
            awatch = None
        while True:
            try:
                valid_roots = [root for root in self.roots if root.is_dir()]
                if awatch is None or not valid_roots:
                    await asyncio.sleep(self.reconcile_seconds)
                else:
                    async for _changes in awatch(
                        *valid_roots,
                        step=500,
                        debounce=1000,
                        rust_timeout=int(self.reconcile_seconds * 1000),
                    ):
                        await self.reconcile()
                        break
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Windows context watcher failed; retrying reconciliation")
                await asyncio.sleep(self.reconcile_seconds)

    async def reconcile(self) -> None:
        snapshots: list[tuple[Path, Path, bool]] = []
        for root in self.roots:
            snapshots.extend(await asyncio.to_thread(self._snapshot_root, root))
        now = _now()
        current = {
            normalize_local_path(path): (path, root, is_dir)
            for path, root, is_dir in snapshots
        }
        prior_existing = {key for key, record in self.records.items() if record.exists}

        for key, (path, root, is_dir) in current.items():
            previous = self.records.get(key)
            project_type = self._project_type(path) if is_dir else ""
            record = LocalPathRecord(
                name=path.name,
                path=str(path),
                is_dir=is_dir,
                root=str(root),
                first_seen=previous.first_seen if previous else now,
                last_seen=now,
                exists=True,
                is_project=bool(project_type),
                project_type=project_type,
            )
            self.records[key] = record
            event = "create" if previous is None or not previous.exists else "modify"
            if event == "create":
                self.journal.append(LocalPathEvent(event=event, path=str(path), timestamp=now))
                await self.memory.upsert_path_event(
                    path=path,
                    display_name=path.name,
                    root=root,
                    is_dir=is_dir,
                    event=event,
                    exists=True,
                    is_project=record.is_project,
                    project_type=project_type,
                    seen_at=now,
                )

        for key in prior_existing - current.keys():
            previous = self.records[key]
            deleted = LocalPathRecord(
                **{**previous.__dict__, "last_seen": now, "exists": False}
            )
            self.records[key] = deleted
            self.journal.append(LocalPathEvent(event="delete", path=previous.path, timestamp=now))
            await self.memory.upsert_path_event(
                path=Path(previous.path),
                display_name=previous.name,
                root=Path(previous.root),
                is_dir=previous.is_dir,
                event="delete",
                exists=False,
                is_project=previous.is_project,
                project_type=previous.project_type,
                seen_at=now,
            )

        await self.memory.prune_path_history(
            retention_days=self.retention_days,
            max_records=self.max_records,
            now=now,
        )
        if self.timeline_projection_enabled:
            try:
                from .context.projection import project_windows_projects

                await project_windows_projects(
                    self.memory,
                    list(self.records.values()),
                    observed_at=now,
                    ttl_days=self.timeline_ttl_days,
                )
            except Exception:
                LOGGER.exception("Windows project timeline projection failed")
        await self._refresh_screenpipe(now)
        self._expire_session_context(now)

    @staticmethod
    def _snapshot_root(root: Path) -> list[tuple[Path, Path, bool]]:
        if not root.is_dir():
            return []
        try:
            return [(entry, root, entry.is_dir()) for entry in root.iterdir()]
        except OSError:
            return []

    @staticmethod
    def _project_type(path: Path) -> str:
        for marker in PROJECT_MARKERS:
            if (path / marker).exists():
                return marker
        try:
            if any(path.glob("*.sln")):
                return "sln"
        except OSError:
            pass
        return ""

    async def _refresh_screenpipe(self, now: datetime) -> None:
        if not self.screenpipe.enabled:
            return
        try:
            elements = await self.screenpipe.recent_elements(limit=50)
        except Exception:
            LOGGER.debug("Screenpipe recent_elements refresh failed", exc_info=True)
            return
        for element in elements:
            self.screenpipe_elements.append(
                CachedScreenpipeElement(element=element, seen_at=now)
            )

    def _expire_session_context(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.context_ttl_seconds)
        for key, record in list(self.records.items()):
            if not record.exists and record.last_seen < cutoff:
                del self.records[key]
        while self.screenpipe_elements and self.screenpipe_elements[0].seen_at < cutoff:
            self.screenpipe_elements.popleft()
