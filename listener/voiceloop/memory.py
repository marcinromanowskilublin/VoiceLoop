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
                    CREATE INDEX IF NOT EXISTS idx_conversation_created
                        ON conversation(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_memories_kind
                        ON memories(kind, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_screenpipe_transcripts_created
                        ON screenpipe_transcripts(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_vector_memories_source
                        ON vector_memories(source, created_at DESC);
                    """
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
                connection.execute(
                    f"UPDATE commands SET {', '.join(assignments)} WHERE request_id = ?",
                    values,
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
