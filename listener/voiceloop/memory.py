from __future__ import annotations

import asyncio
import json
import sqlite3
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

                    CREATE INDEX IF NOT EXISTS idx_commands_created
                        ON commands(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_conversation_created
                        ON conversation(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_memories_kind
                        ON memories(kind, created_at DESC);
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
