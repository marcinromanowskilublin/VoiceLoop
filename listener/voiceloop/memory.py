from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import (
    ActionResult,
    CommandPlan,
    CommandRequest,
    CommandStatus,
    CommandView,
    MemoryCreate,
    MemoryItem,
    TranscriptEnvelopeV1,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class VectorMemoryHit:
    source: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    score: float
    created_at: datetime


@dataclass(frozen=True)
class StoredVectorMemory:
    source: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float]
    created_at: datetime


@dataclass(frozen=True)
class ScreenpipeTranscript:
    chunk_id: str
    meeting_id: int
    device_name: str
    device_type: str
    start_time: str
    end_time: str
    text: str
    source: str
    created_at: datetime


@dataclass(frozen=True)
class MeetingSession:
    session_id: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    title: str
    audio_dir: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MeetingTranscriptSegment:
    id: int
    session_id: str
    segment_key: str
    channel: str
    speaker_label: str
    speaker_id: int | None
    device_name: str
    start_time: datetime
    end_time: datetime
    text: str
    transcript: TranscriptEnvelopeV1 | None
    emotions: tuple[dict[str, Any], ...]
    source: str
    created_at: datetime


@dataclass(frozen=True)
class MeetingAudioFile:
    id: int
    session_id: str
    chunk_id: str
    channel: str
    device_name: str
    start_time: datetime
    end_time: datetime
    source_path: str
    archived_path: str
    created_at: datetime


class MemoryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        def _initialize() -> None:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS commands (
                        request_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        status TEXT NOT NULL,
                        intent TEXT,
                        response_text TEXT,
                        provider TEXT,
                        model TEXT,
                        error TEXT,
                        plan_json TEXT,
                        results_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS command_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        detail_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(request_id) REFERENCES commands(request_id)
                    );

                    CREATE TABLE IF NOT EXISTS command_transcripts (
                        request_id TEXT PRIMARY KEY,
                        segment_id TEXT NOT NULL,
                        transcript_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(request_id) REFERENCES commands(request_id)
                    );

                    CREATE TABLE IF NOT EXISTS conversation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL,
                        content TEXT NOT NULL,
                        sensitivity TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS app_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS screenpipe_meeting_jobs (
                        meeting_id INTEGER PRIMARY KEY,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        processed_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS screenpipe_transcripts (
                        chunk_id TEXT PRIMARY KEY,
                        meeting_id INTEGER NOT NULL,
                        device_name TEXT NOT NULL,
                        device_type TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        text TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS meeting_sessions (
                        session_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        title TEXT NOT NULL DEFAULT '',
                        audio_dir TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS meeting_transcript_segments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        segment_key TEXT NOT NULL UNIQUE,
                        channel TEXT NOT NULL,
                        speaker_label TEXT NOT NULL,
                        speaker_id INTEGER,
                        device_name TEXT NOT NULL DEFAULT '',
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        text TEXT NOT NULL,
                        transcript_json TEXT,
                        emotion_json TEXT,
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES meeting_sessions(session_id)
                    );

                    CREATE TABLE IF NOT EXISTS meeting_audio_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        device_name TEXT NOT NULL DEFAULT '',
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        archived_path TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(session_id, chunk_id),
                        FOREIGN KEY(session_id) REFERENCES meeting_sessions(session_id)
                    );

                    CREATE TABLE IF NOT EXISTS vector_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        embedding_json TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(source, source_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_commands_created
                        ON commands(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_command_events_request
                        ON command_events(request_id, created_at ASC);
                    CREATE INDEX IF NOT EXISTS idx_command_events_status
                        ON command_events(status, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_conversation_created
                        ON conversation(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_memories_kind
                        ON memories(kind, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_screenpipe_transcripts_created
                        ON screenpipe_transcripts(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_meeting_sessions_started
                        ON meeting_sessions(started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_meeting_segments_session_time
                        ON meeting_transcript_segments(session_id, start_time ASC);
                    CREATE INDEX IF NOT EXISTS idx_meeting_audio_session_time
                        ON meeting_audio_files(session_id, start_time ASC);
                    CREATE INDEX IF NOT EXISTS idx_vector_memories_source
                        ON vector_memories(source, created_at DESC);
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(meeting_transcript_segments)"
                    ).fetchall()
                }
                if "emotion_json" not in columns:
                    connection.execute(
                        "ALTER TABLE meeting_transcript_segments "
                        "ADD COLUMN emotion_json TEXT"
                    )

        await asyncio.to_thread(_initialize)

    async def create_command(self, request: CommandRequest) -> CommandView:
        created = request.created_at.isoformat()
        input_text = (request.text or request.command_id or "").strip()

        def _create() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO commands (
                        request_id, source, input_text, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.request_id,
                        request.source.value,
                        input_text,
                        CommandStatus.RECEIVED.value,
                        created,
                        created,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO command_events(request_id, status, detail_json, created_at)
                    VALUES (?, ?, '{}', ?)
                    """,
                    (
                        request.request_id,
                        CommandStatus.RECEIVED.value,
                        created,
                    ),
                )
                if request.transcript is not None:
                    connection.execute(
                        """
                        INSERT INTO command_transcripts(
                            request_id, segment_id, transcript_json, created_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(request_id) DO UPDATE SET
                            segment_id = excluded.segment_id,
                            transcript_json = excluded.transcript_json,
                            created_at = excluded.created_at
                        """,
                        (
                            request.request_id,
                            request.transcript.segment_id,
                            request.transcript.model_dump_json(),
                            created,
                        ),
                    )

        await asyncio.to_thread(_create)
        command = await self.get_command(request.request_id)
        if command is None:
            raise RuntimeError("command was not persisted")
        return command

    async def update_command(
        self,
        request_id: str,
        *,
        status: CommandStatus | None = None,
        plan: CommandPlan | None = None,
        results: list[ActionResult] | None = None,
        error: str | None = None,
    ) -> CommandView | None:
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [_now_iso()]

        if status is not None:
            assignments.append("status = ?")
            values.append(status.value)
        if plan is not None:
            assignments.extend(
                [
                    "intent = ?",
                    "response_text = ?",
                    "provider = ?",
                    "model = ?",
                    "plan_json = ?",
                ]
            )
            values.extend(
                [
                    plan.intent,
                    plan.response_text,
                    plan.provider,
                    plan.model,
                    plan.model_dump_json(),
                ]
            )
        if results is not None:
            assignments.append("results_json = ?")
            values.append(json.dumps([result.model_dump(mode="json") for result in results]))
        if error is not None:
            assignments.append("error = ?")
            values.append(error[:4000])

        values.append(request_id)

        def _update() -> None:
            with self._connect() as connection:
                cursor = connection.execute(
                    f"UPDATE commands SET {', '.join(assignments)} WHERE request_id = ?",
                    values,
                )
                if cursor.rowcount > 0 and status is not None:
                    detail = {
                        "has_plan": plan is not None,
                        "result_count": len(results) if results is not None else None,
                        "has_error": error is not None,
                    }
                    connection.execute(
                        """
                        INSERT INTO command_events(
                            request_id, status, detail_json, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            request_id,
                            status.value,
                            json.dumps(detail, separators=(",", ":")),
                            values[0],
                        ),
                    )

        await asyncio.to_thread(_update)
        return await self.get_command(request_id)

    async def get_command(self, request_id: str) -> CommandView | None:
        def _get() -> sqlite3.Row | None:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT * FROM commands WHERE request_id = ?", (request_id,)
                ).fetchone()

        row = await asyncio.to_thread(_get)
        return self._command_from_row(row) if row else None

    async def recent_commands(self, limit: int = 30) -> list[CommandView]:
        safe_limit = max(1, min(limit, 200))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT * FROM commands ORDER BY created_at DESC LIMIT ?", (safe_limit,)
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [self._command_from_row(row) for row in rows]

    async def command_events(self, request_id: str) -> list[dict[str, Any]]:
        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT status, detail_json, created_at
                    FROM command_events
                    WHERE request_id = ?
                    ORDER BY id ASC
                    """,
                    (request_id,),
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        events: list[dict[str, Any]] = []
        for row in rows:
            try:
                detail = json.loads(row["detail_json"] or "{}")
            except json.JSONDecodeError:
                detail = {}
            events.append(
                {
                    "status": row["status"],
                    "detail": detail if isinstance(detail, dict) else {},
                    "created_at": row["created_at"],
                }
            )
        return events

    async def add_message(self, role: str, content: str, request_id: str | None = None) -> None:
        def _add() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO conversation(request_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (request_id, role, content[:20000], _now_iso()),
                )

        await asyncio.to_thread(_add)

    async def recent_messages(self, limit: int = 12) -> list[dict[str, str]]:
        safe_limit = max(1, min(limit, 100))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT role, content FROM (
                        SELECT id, role, content
                        FROM conversation
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (safe_limit,),
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def create_memory(self, item: MemoryCreate) -> MemoryItem:
        now = _now_iso()

        def _create() -> int:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO memories(kind, content, sensitivity, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (item.kind, item.content, item.sensitivity, item.source, now, now),
                )
                return int(cursor.lastrowid)

        memory_id = await asyncio.to_thread(_create)
        memories = await self.list_memories(limit=1, memory_id=memory_id)
        return memories[0]

    async def list_memories(
        self, limit: int = 100, kind: str | None = None, memory_id: int | None = None
    ) -> list[MemoryItem]:
        safe_limit = max(1, min(limit, 500))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                if memory_id is not None:
                    return connection.execute(
                        "SELECT * FROM memories WHERE id = ?", (memory_id,)
                    ).fetchall()
                if kind:
                    return connection.execute(
                        """
                        SELECT * FROM memories
                        WHERE kind = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (kind, safe_limit),
                    ).fetchall()
                return connection.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (safe_limit,)
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [
            MemoryItem(
                id=row["id"],
                kind=row["kind"],
                content=row["content"],
                sensitivity=row["sensitivity"],
                source=row["source"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def delete_memory(self, memory_id: int) -> bool:
        def _delete() -> bool:
            with self._connect() as connection:
                cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0

        return await asyncio.to_thread(_delete)

    async def has_vector_memory(self, *, source: str, source_id: str) -> bool:
        def _has() -> bool:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT 1 FROM vector_memories
                    WHERE source = ? AND source_id = ?
                    """,
                    (source, source_id),
                ).fetchone()
                return row is not None

        return await asyncio.to_thread(_has)

    async def vector_memory_metadata(
        self,
        *,
        source: str,
        source_id: str,
    ) -> dict[str, Any] | None:
        def _metadata() -> dict[str, Any] | None:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT metadata_json FROM vector_memories
                    WHERE source = ? AND source_id = ?
                    """,
                    (source, source_id),
                ).fetchone()
            if row is None:
                return None
            try:
                value = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}

        return await asyncio.to_thread(_metadata)

    async def upsert_vector_memory(
        self,
        *,
        source: str,
        source_id: str,
        title: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not embedding:
            return
        now = _now_iso()
        safe_embedding = [float(value) for value in embedding]
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        embedding_json = json.dumps(safe_embedding, separators=(",", ":"))

        def _upsert() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO vector_memories(
                        source, source_id, title, content, metadata_json,
                        embedding_json, dimension, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, source_id) DO UPDATE SET
                        title = excluded.title,
                        content = excluded.content,
                        metadata_json = excluded.metadata_json,
                        embedding_json = excluded.embedding_json,
                        dimension = excluded.dimension,
                        updated_at = excluded.updated_at
                    """,
                    (
                        source[:80],
                        source_id[:200],
                        title[:500],
                        content[:4000],
                        metadata_json[:8000],
                        embedding_json,
                        len(safe_embedding),
                        now,
                        now,
                    ),
                )

        await asyncio.to_thread(_upsert)

    async def search_vector_memories(
        self,
        query_embedding: list[float],
        *,
        limit: int = 8,
        source: str | None = None,
        min_score: float = 0.15,
        candidate_limit: int = 2000,
    ) -> list[VectorMemoryHit]:
        if not query_embedding:
            return []
        safe_limit = max(1, min(limit, 30))
        safe_candidate_limit = max(safe_limit, min(candidate_limit, 10000))
        query = [float(value) for value in query_embedding]

        def _rows() -> list[sqlite3.Row]:
            with self._connect() as connection:
                if source:
                    return connection.execute(
                        """
                        SELECT * FROM vector_memories
                        WHERE source = ?
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (source, safe_candidate_limit),
                    ).fetchall()
                return connection.execute(
                    """
                    SELECT * FROM vector_memories
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (safe_candidate_limit,),
                ).fetchall()

        rows = await asyncio.to_thread(_rows)
        hits: list[VectorMemoryHit] = []
        for row in rows:
            try:
                embedding = [float(value) for value in json.loads(row["embedding_json"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            score = self._cosine_similarity(query, embedding)
            if score < min_score:
                continue
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            hits.append(
                VectorMemoryHit(
                    source=row["source"],
                    source_id=row["source_id"],
                    title=row["title"],
                    content=row["content"],
                    metadata=metadata if isinstance(metadata, dict) else {},
                    score=score,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:safe_limit]

    async def list_stored_vector_memories(
        self,
        *,
        limit: int = 10000,
    ) -> list[StoredVectorMemory]:
        safe_limit = max(1, min(limit, 100000))

        def _rows() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM vector_memories
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        rows = await asyncio.to_thread(_rows)
        memories: list[StoredVectorMemory] = []
        for row in rows:
            try:
                embedding = [float(value) for value in json.loads(row["embedding_json"])]
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            memories.append(
                StoredVectorMemory(
                    source=row["source"],
                    source_id=row["source_id"],
                    title=row["title"],
                    content=row["content"],
                    metadata=metadata if isinstance(metadata, dict) else {},
                    embedding=embedding,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return memories

    async def prune_expired_vector_memories(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        cutoff = now or datetime.now(UTC)

        def _prune() -> int:
            expired_ids: list[int] = []
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, metadata_json FROM vector_memories"
                ).fetchall()
                for row in rows:
                    try:
                        metadata = json.loads(row["metadata_json"] or "{}")
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    expires_at_raw = str(metadata.get("expires_at") or "").strip()
                    if not expires_at_raw:
                        continue
                    try:
                        expires_at = datetime.fromisoformat(expires_at_raw)
                    except ValueError:
                        continue
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=UTC)
                    if expires_at <= cutoff:
                        expired_ids.append(int(row["id"]))
                if expired_ids:
                    connection.executemany(
                        "DELETE FROM vector_memories WHERE id = ?",
                        [(row_id,) for row_id in expired_ids],
                    )
            return len(expired_ids)

        return await asyncio.to_thread(_prune)

    async def get_state(self, key: str) -> str | None:
        def _get() -> str | None:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value FROM app_state WHERE key = ?",
                    (key,),
                ).fetchone()
                return str(row["value"]) if row else None

        return await asyncio.to_thread(_get)

    async def set_state(self, key: str, value: str) -> None:
        now = _now_iso()

        def _set() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO app_state(key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )

        await asyncio.to_thread(_set)

    async def create_meeting_session(
        self,
        *,
        session_id: str,
        started_at: datetime,
        audio_dir: Path,
        title: str = "",
    ) -> MeetingSession:
        now = _now_iso()

        def _create() -> sqlite3.Row:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO meeting_sessions(
                        session_id, status, started_at, ended_at, title,
                        audio_dir, created_at, updated_at
                    ) VALUES (?, 'active', ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        started_at.isoformat(),
                        title[:500],
                        str(audio_dir),
                        now,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM meeting_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                assert row is not None
                return row

        return self._meeting_session_from_row(await asyncio.to_thread(_create))

    async def update_meeting_session(
        self,
        session_id: str,
        *,
        status: str,
        ended_at: datetime | None = None,
    ) -> MeetingSession | None:
        def _update() -> sqlite3.Row | None:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE meeting_sessions
                    SET status = ?,
                        ended_at = COALESCE(?, ended_at),
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        status[:40],
                        ended_at.isoformat() if ended_at is not None else None,
                        _now_iso(),
                        session_id,
                    ),
                )
                return connection.execute(
                    "SELECT * FROM meeting_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

        row = await asyncio.to_thread(_update)
        return self._meeting_session_from_row(row) if row is not None else None

    async def get_meeting_session(self, session_id: str) -> MeetingSession | None:
        def _get() -> sqlite3.Row | None:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT * FROM meeting_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

        row = await asyncio.to_thread(_get)
        return self._meeting_session_from_row(row) if row is not None else None

    async def active_meeting_session(self) -> MeetingSession | None:
        def _get() -> sqlite3.Row | None:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM meeting_sessions
                    WHERE status IN ('active', 'finalizing')
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                ).fetchone()

        row = await asyncio.to_thread(_get)
        return self._meeting_session_from_row(row) if row is not None else None

    async def list_meeting_sessions(self, *, limit: int = 20) -> list[MeetingSession]:
        safe_limit = max(1, min(limit, 200))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM meeting_sessions
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [self._meeting_session_from_row(row) for row in rows]

    async def save_meeting_transcript_segment(
        self,
        *,
        session_id: str,
        segment_key: str,
        channel: str,
        speaker_label: str,
        speaker_id: int | None,
        device_name: str,
        start_time: datetime,
        end_time: datetime,
        text: str,
        transcript: TranscriptEnvelopeV1 | None,
        source: str,
    ) -> bool:
        def _save() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO meeting_transcript_segments(
                        session_id, segment_key, channel, speaker_label,
                        speaker_id, device_name, start_time, end_time, text,
                        transcript_json, emotion_json, source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    ON CONFLICT(segment_key) DO NOTHING
                    """,
                    (
                        session_id,
                        segment_key[:500],
                        channel[:40],
                        speaker_label[:200],
                        speaker_id,
                        device_name[:500],
                        start_time.isoformat(),
                        end_time.isoformat(),
                        text[:100000],
                        transcript.model_dump_json() if transcript is not None else None,
                        source[:80],
                        _now_iso(),
                    ),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(_save)

    async def save_meeting_segment_emotions(
        self,
        *,
        segment_key: str,
        emotions: list[dict[str, Any]],
    ) -> bool:
        safe_emotions = [
            {
                "name": str(item.get("name") or "")[:80],
                "score": float(item.get("score") or 0.0),
            }
            for item in emotions
            if item.get("name")
        ]
        payload = json.dumps(
            safe_emotions[:8],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def _save() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE meeting_transcript_segments
                    SET emotion_json = ?
                    WHERE segment_key = ?
                    """,
                    (payload, segment_key[:500]),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(_save)

    async def list_meeting_transcript_segments(
        self,
        session_id: str,
        *,
        limit: int = 5000,
    ) -> list[MeetingTranscriptSegment]:
        safe_limit = max(1, min(limit, 20000))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM meeting_transcript_segments
                    WHERE session_id = ?
                    ORDER BY start_time ASC, id ASC
                    LIMIT ?
                    """,
                    (session_id, safe_limit),
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [
            MeetingTranscriptSegment(
                id=int(row["id"]),
                session_id=row["session_id"],
                segment_key=row["segment_key"],
                channel=row["channel"],
                speaker_label=row["speaker_label"],
                speaker_id=row["speaker_id"],
                device_name=row["device_name"],
                start_time=datetime.fromisoformat(row["start_time"]),
                end_time=datetime.fromisoformat(row["end_time"]),
                text=row["text"],
                transcript=(
                    TranscriptEnvelopeV1.model_validate_json(row["transcript_json"])
                    if row["transcript_json"]
                    else None
                ),
                emotions=self._emotions_from_row(row),
                source=row["source"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _emotions_from_row(row: sqlite3.Row) -> tuple[dict[str, Any], ...]:
        try:
            raw = row["emotion_json"]
        except (IndexError, KeyError):
            return ()
        if not raw:
            return ()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return ()
        if not isinstance(value, list):
            return ()
        emotions: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                continue
            emotions.append({"name": name, "score": score})
        return tuple(emotions)

    async def save_meeting_audio_file(
        self,
        *,
        session_id: str,
        chunk_id: str,
        channel: str,
        device_name: str,
        start_time: datetime,
        end_time: datetime,
        source_path: Path,
        archived_path: Path,
    ) -> bool:
        def _save() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO meeting_audio_files(
                        session_id, chunk_id, channel, device_name,
                        start_time, end_time, source_path, archived_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, chunk_id) DO NOTHING
                    """,
                    (
                        session_id,
                        chunk_id[:1000],
                        channel[:40],
                        device_name[:500],
                        start_time.isoformat(),
                        end_time.isoformat(),
                        str(source_path),
                        str(archived_path),
                        _now_iso(),
                    ),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(_save)

    async def has_meeting_audio_file(self, session_id: str, chunk_id: str) -> bool:
        def _has() -> bool:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT 1 FROM meeting_audio_files
                    WHERE session_id = ? AND chunk_id = ?
                    """,
                    (session_id, chunk_id),
                ).fetchone()
                return row is not None

        return await asyncio.to_thread(_has)

    async def list_meeting_audio_files(
        self,
        session_id: str,
        *,
        limit: int = 5000,
    ) -> list[MeetingAudioFile]:
        safe_limit = max(1, min(limit, 20000))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM meeting_audio_files
                    WHERE session_id = ?
                    ORDER BY start_time ASC, id ASC
                    LIMIT ?
                    """,
                    (session_id, safe_limit),
                ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [
            MeetingAudioFile(
                id=int(row["id"]),
                session_id=row["session_id"],
                chunk_id=row["chunk_id"],
                channel=row["channel"],
                device_name=row["device_name"],
                start_time=datetime.fromisoformat(row["start_time"]),
                end_time=datetime.fromisoformat(row["end_time"]),
                source_path=row["source_path"],
                archived_path=row["archived_path"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def has_screenpipe_meeting_job(self, meeting_id: int) -> bool:
        def _has() -> bool:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT 1 FROM screenpipe_meeting_jobs WHERE meeting_id = ?",
                    (meeting_id,),
                ).fetchone()
                return row is not None

        return await asyncio.to_thread(_has)

    async def mark_screenpipe_meeting_job(
        self,
        meeting_id: int,
        *,
        status: str,
        reason: str,
    ) -> None:
        def _mark() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO screenpipe_meeting_jobs(meeting_id, status, reason, processed_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(meeting_id) DO UPDATE SET
                        status = excluded.status,
                        reason = excluded.reason,
                        processed_at = excluded.processed_at
                    """,
                    (meeting_id, status[:40], reason[:500], _now_iso()),
                )

        await asyncio.to_thread(_mark)

    async def save_screenpipe_transcript(
        self,
        *,
        chunk_id: str,
        meeting_id: int,
        device_name: str,
        device_type: str,
        start_time: str,
        end_time: str,
        text: str,
        source: str,
    ) -> None:
        def _save() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO screenpipe_transcripts(
                        chunk_id, meeting_id, device_name, device_type,
                        start_time, end_time, text, source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO NOTHING
                    """,
                    (
                        chunk_id,
                        meeting_id,
                        device_name[:500],
                        device_type[:40],
                        start_time,
                        end_time,
                        text[:100000],
                        source[:40],
                        _now_iso(),
                    ),
                )

        await asyncio.to_thread(_save)

    async def has_screenpipe_transcript(self, chunk_id: str) -> bool:
        def _has() -> bool:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT 1 FROM screenpipe_transcripts WHERE chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                return row is not None

        return await asyncio.to_thread(_has)

    async def list_screenpipe_transcripts(
        self,
        *,
        limit: int = 500,
    ) -> list[ScreenpipeTranscript]:
        safe_limit = max(1, min(limit, 5000))

        def _rows() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM screenpipe_transcripts
                    ORDER BY start_time DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        rows = await asyncio.to_thread(_rows)
        return [
            ScreenpipeTranscript(
                chunk_id=row["chunk_id"],
                meeting_id=row["meeting_id"],
                device_name=row["device_name"],
                device_type=row["device_type"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                text=row["text"],
                source=row["source"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def prune_screenpipe_transcripts(self, *, retention_days: int) -> int:
        retention_days = max(1, min(retention_days, 365))

        def _prune() -> int:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM screenpipe_transcripts
                    WHERE datetime(created_at) < datetime('now', ?)
                    """,
                    (f"-{retention_days} days",),
                )
                return cursor.rowcount

        return await asyncio.to_thread(_prune)

    @staticmethod
    def _meeting_session_from_row(row: sqlite3.Row) -> MeetingSession:
        return MeetingSession(
            session_id=row["session_id"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
            title=row["title"],
            audio_dir=row["audio_dir"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _command_from_row(row: sqlite3.Row) -> CommandView:
        plan = CommandPlan.model_validate_json(row["plan_json"]) if row["plan_json"] else None
        result_data = json.loads(row["results_json"] or "[]")
        results = [ActionResult.model_validate(item) for item in result_data]
        return CommandView(
            request_id=row["request_id"],
            source=row["source"],
            input_text=row["input_text"],
            status=CommandStatus(row["status"]),
            intent=row["intent"],
            response_text=row["response_text"],
            provider=row["provider"],
            model=row["model"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            plan=plan,
            results=results,
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right, strict=True):
            dot += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return dot / ((left_norm**0.5) * (right_norm**0.5))
