from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .context.schema import ContextEpisodeV1, ContextEventV1, ForegroundConfidence
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

LOGGER = logging.getLogger("voiceloop.memory")


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


@dataclass(frozen=True)
class PathHistoryItem:
    normalized_path: str
    display_name: str
    root: str
    is_dir: bool
    first_seen: datetime
    last_seen: datetime
    last_event: str
    exists: bool
    deleted_at: datetime | None
    is_project: bool
    project_type: str


class MemoryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._context_fts_enabled = False

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
                        session_id TEXT,
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

                    CREATE TABLE IF NOT EXISTS local_path_history (
                        normalized_path TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        root TEXT NOT NULL,
                        is_dir INTEGER NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        last_event TEXT NOT NULL,
                        exists_flag INTEGER NOT NULL,
                        deleted_at TEXT,
                        is_project INTEGER NOT NULL DEFAULT 0,
                        project_type TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS context_events (
                        event_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        app_name TEXT NOT NULL DEFAULT '',
                        process_name TEXT NOT NULL DEFAULT '',
                        window_title TEXT NOT NULL DEFAULT '',
                        is_foreground INTEGER,
                        foreground_confidence TEXT NOT NULL DEFAULT 'unknown',
                        focus_duration_ms INTEGER,
                        text TEXT NOT NULL DEFAULT '',
                        content_hash TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        sensitivity TEXT NOT NULL DEFAULT 'private',
                        expires_at TEXT,
                        deleted_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(source, source_id)
                    );

                    CREATE TABLE IF NOT EXISTS context_episodes (
                        episode_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        summary TEXT NOT NULL,
                        source_event_ids_json TEXT NOT NULL,
                        app_names_json TEXT NOT NULL DEFAULT '[]',
                        person_ids_json TEXT NOT NULL DEFAULT '[]',
                        project_ids_json TEXT NOT NULL DEFAULT '[]',
                        vector_spaces_json TEXT NOT NULL DEFAULT '[]',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        content_hash TEXT NOT NULL,
                        sensitivity TEXT NOT NULL DEFAULT 'private',
                        expires_at TEXT,
                        deleted_at TEXT,
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
                    CREATE INDEX IF NOT EXISTS idx_local_path_history_seen
                        ON local_path_history(last_seen DESC);
                    CREATE INDEX IF NOT EXISTS idx_local_path_history_name
                        ON local_path_history(display_name);
                    CREATE INDEX IF NOT EXISTS idx_local_path_history_project
                        ON local_path_history(is_project, last_seen DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_events_time
                        ON context_events(started_at DESC, ended_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_events_source
                        ON context_events(source, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_events_app
                        ON context_events(app_name, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_events_hash
                        ON context_events(content_hash);
                    CREATE INDEX IF NOT EXISTS idx_context_episodes_time
                        ON context_episodes(started_at DESC, ended_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_episodes_source
                        ON context_episodes(source, started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_episodes_hash
                        ON context_episodes(content_hash);
                    """
                )
                try:
                    connection.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS context_events_fts
                        USING fts5(
                            event_id UNINDEXED,
                            text,
                            window_title,
                            app_name,
                            tokenize='unicode61 remove_diacritics 2'
                        )
                        """
                    )
                    connection.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS context_episodes_fts
                        USING fts5(
                            episode_id UNINDEXED,
                            title,
                            summary,
                            app_names,
                            tokenize='unicode61 remove_diacritics 2'
                        )
                        """
                    )
                except sqlite3.OperationalError as exc:
                    self._context_fts_enabled = False
                    LOGGER.warning("SQLite FTS5 unavailable for context events: %s", exc)
                else:
                    self._context_fts_enabled = True
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
                conversation_columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(conversation)"
                    ).fetchall()
                }
                if "session_id" not in conversation_columns:
                    connection.execute(
                        "ALTER TABLE conversation ADD COLUMN session_id TEXT"
                    )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_session
                    ON conversation(session_id, id DESC)
                    """
                )

        await asyncio.to_thread(_initialize)

    @property
    def context_fts_enabled(self) -> bool:
        return self._context_fts_enabled

    async def upsert_context_event(self, event: ContextEventV1) -> ContextEventV1:
        """Persist one canonical timeline event and keep the FTS projection in sync."""

        metadata_json = json.dumps(
            event.metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def _upsert() -> sqlite3.Row:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO context_events(
                        event_id, source, source_id, event_type, started_at, ended_at,
                        app_name, process_name, window_title, is_foreground,
                        foreground_confidence, focus_duration_ms, text, content_hash,
                        metadata_json, sensitivity, expires_at, deleted_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, source_id) DO UPDATE SET
                        event_type = excluded.event_type,
                        started_at = excluded.started_at,
                        ended_at = excluded.ended_at,
                        app_name = excluded.app_name,
                        process_name = excluded.process_name,
                        window_title = excluded.window_title,
                        is_foreground = excluded.is_foreground,
                        foreground_confidence = excluded.foreground_confidence,
                        focus_duration_ms = excluded.focus_duration_ms,
                        text = excluded.text,
                        content_hash = excluded.content_hash,
                        metadata_json = excluded.metadata_json,
                        sensitivity = excluded.sensitivity,
                        expires_at = excluded.expires_at,
                        deleted_at = excluded.deleted_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        event.event_id,
                        event.source,
                        event.source_id,
                        event.event_type,
                        event.started_at.isoformat(),
                        event.ended_at.isoformat() if event.ended_at else None,
                        event.app_name,
                        event.process_name,
                        event.window_title,
                        (
                            None
                            if event.is_foreground is None
                            else int(event.is_foreground)
                        ),
                        event.foreground_confidence.value,
                        event.focus_duration_ms,
                        event.text,
                        event.content_hash,
                        metadata_json,
                        event.sensitivity,
                        event.expires_at.isoformat() if event.expires_at else None,
                        event.deleted_at.isoformat() if event.deleted_at else None,
                        event.created_at.isoformat(),
                        event.updated_at.isoformat(),
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM context_events
                    WHERE source = ? AND source_id = ?
                    """,
                    (event.source, event.source_id),
                ).fetchone()
                assert row is not None
                if self._context_fts_enabled:
                    connection.execute(
                        "DELETE FROM context_events_fts WHERE event_id = ?",
                        (row["event_id"],),
                    )
                    if row["deleted_at"] is None:
                        connection.execute(
                            """
                            INSERT INTO context_events_fts(
                                event_id, text, window_title, app_name
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                row["event_id"],
                                row["text"],
                                row["window_title"],
                                row["app_name"],
                            ),
                        )
                return row

        return self._context_event_from_row(await asyncio.to_thread(_upsert))

    async def get_context_event(self, event_id: str) -> ContextEventV1 | None:
        def _get() -> sqlite3.Row | None:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT * FROM context_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()

        row = await asyncio.to_thread(_get)
        return self._context_event_from_row(row) if row is not None else None

    async def search_context_events(
        self,
        *,
        query: str = "",
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
        app_name: str | None = None,
        is_foreground: bool | None = None,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[ContextEventV1]:
        safe_limit = max(1, min(int(limit), 500))
        fts_query = self._context_fts_query(query)

        def _search() -> list[sqlite3.Row]:
            joins: list[str] = []
            conditions = ["e.deleted_at IS NULL"]
            parameters: list[Any] = []
            if fts_query and self._context_fts_enabled:
                joins.append(
                    "JOIN context_events_fts f ON f.event_id = e.event_id"
                )
                conditions.append("context_events_fts MATCH ?")
                parameters.append(fts_query)
            elif query.strip():
                tokens = self._context_query_tokens(query)
                for token in tokens:
                    pattern = f"%{token}%"
                    conditions.append(
                        "(e.text LIKE ? OR e.window_title LIKE ? OR e.app_name LIKE ?)"
                    )
                    parameters.extend((pattern, pattern, pattern))
            if start is not None:
                conditions.append("COALESCE(e.ended_at, e.started_at) >= ?")
                parameters.append(self._aware_iso(start))
            if end is not None:
                conditions.append("e.started_at <= ?")
                parameters.append(self._aware_iso(end))
            if source:
                conditions.append("e.source = ?")
                parameters.append(source)
            if app_name:
                conditions.append("LOWER(e.app_name) = LOWER(?)")
                parameters.append(app_name)
            if is_foreground is not None:
                conditions.append("e.is_foreground = ?")
                parameters.append(int(is_foreground))
            if not include_expired:
                conditions.append("(e.expires_at IS NULL OR e.expires_at > ?)")
                parameters.append(_now_iso())
            order = (
                "bm25(context_events_fts) ASC, e.started_at DESC"
                if fts_query and self._context_fts_enabled
                else "e.started_at DESC"
            )
            sql = (
                "SELECT e.* FROM context_events e "
                + " ".join(joins)
                + " WHERE "
                + " AND ".join(conditions)
                + f" ORDER BY {order} LIMIT ?"
            )
            parameters.append(safe_limit)
            with self._connect() as connection:
                return connection.execute(sql, parameters).fetchall()

        rows = await asyncio.to_thread(_search)
        return [self._context_event_from_row(row) for row in rows]

    async def tombstone_context_event(
        self,
        event_id: str,
        *,
        deleted_at: datetime | None = None,
    ) -> bool:
        timestamp = self._aware_iso(deleted_at or datetime.now(UTC))

        def _delete() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE context_events
                    SET deleted_at = ?, updated_at = ?
                    WHERE event_id = ? AND deleted_at IS NULL
                    """,
                    (timestamp, timestamp, event_id),
                )
                if cursor.rowcount and self._context_fts_enabled:
                    connection.execute(
                        "DELETE FROM context_events_fts WHERE event_id = ?",
                        (event_id,),
                    )
                return cursor.rowcount > 0

        return await asyncio.to_thread(_delete)

    async def upsert_context_episode(
        self,
        episode: ContextEpisodeV1,
    ) -> ContextEpisodeV1:
        def dump(value: object) -> str:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

        def _upsert() -> sqlite3.Row:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO context_episodes(
                        episode_id, source, source_id, started_at, ended_at,
                        title, summary, source_event_ids_json, app_names_json,
                        person_ids_json, project_ids_json, vector_spaces_json,
                        metadata_json, content_hash, sensitivity, expires_at,
                        deleted_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, source_id) DO UPDATE SET
                        started_at = excluded.started_at,
                        ended_at = excluded.ended_at,
                        title = excluded.title,
                        summary = excluded.summary,
                        source_event_ids_json = excluded.source_event_ids_json,
                        app_names_json = excluded.app_names_json,
                        person_ids_json = excluded.person_ids_json,
                        project_ids_json = excluded.project_ids_json,
                        vector_spaces_json = excluded.vector_spaces_json,
                        metadata_json = excluded.metadata_json,
                        content_hash = excluded.content_hash,
                        sensitivity = excluded.sensitivity,
                        expires_at = excluded.expires_at,
                        deleted_at = excluded.deleted_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        episode.episode_id,
                        episode.source,
                        episode.source_id,
                        episode.started_at.isoformat(),
                        episode.ended_at.isoformat(),
                        episode.title,
                        episode.summary,
                        dump(episode.source_event_ids),
                        dump(episode.app_names),
                        dump(episode.person_ids),
                        dump(episode.project_ids),
                        dump(episode.vector_spaces),
                        dump(episode.metadata),
                        episode.content_hash,
                        episode.sensitivity,
                        episode.expires_at.isoformat() if episode.expires_at else None,
                        episode.deleted_at.isoformat() if episode.deleted_at else None,
                        episode.created_at.isoformat(),
                        episode.updated_at.isoformat(),
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM context_episodes
                    WHERE source = ? AND source_id = ?
                    """,
                    (episode.source, episode.source_id),
                ).fetchone()
                assert row is not None
                if self._context_fts_enabled:
                    connection.execute(
                        "DELETE FROM context_episodes_fts WHERE episode_id = ?",
                        (row["episode_id"],),
                    )
                    if row["deleted_at"] is None:
                        app_names = " ".join(
                            self._json_string_tuple(row["app_names_json"])
                        )
                        connection.execute(
                            """
                            INSERT INTO context_episodes_fts(
                                episode_id, title, summary, app_names
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                row["episode_id"],
                                row["title"],
                                row["summary"],
                                app_names,
                            ),
                        )
                return row

        return self._context_episode_from_row(await asyncio.to_thread(_upsert))

    async def search_context_episodes(
        self,
        *,
        query: str = "",
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
        limit: int = 20,
        include_expired: bool = False,
    ) -> list[ContextEpisodeV1]:
        safe_limit = max(1, min(int(limit), 200))
        fts_query = self._context_fts_query(query)

        def _search() -> list[sqlite3.Row]:
            joins: list[str] = []
            conditions = ["e.deleted_at IS NULL"]
            parameters: list[Any] = []
            if fts_query and self._context_fts_enabled:
                joins.append(
                    "JOIN context_episodes_fts f ON f.episode_id = e.episode_id"
                )
                conditions.append("context_episodes_fts MATCH ?")
                parameters.append(fts_query)
            elif query.strip():
                for token in self._context_query_tokens(query):
                    pattern = f"%{token}%"
                    conditions.append("(e.title LIKE ? OR e.summary LIKE ?)")
                    parameters.extend((pattern, pattern))
            if start is not None:
                conditions.append("e.ended_at >= ?")
                parameters.append(self._aware_iso(start))
            if end is not None:
                conditions.append("e.started_at <= ?")
                parameters.append(self._aware_iso(end))
            if source:
                conditions.append("e.source = ?")
                parameters.append(source)
            if not include_expired:
                conditions.append("(e.expires_at IS NULL OR e.expires_at > ?)")
                parameters.append(_now_iso())
            order = (
                "bm25(context_episodes_fts) ASC, e.started_at DESC"
                if fts_query and self._context_fts_enabled
                else "e.started_at DESC"
            )
            sql = (
                "SELECT e.* FROM context_episodes e "
                + " ".join(joins)
                + " WHERE "
                + " AND ".join(conditions)
                + f" ORDER BY {order} LIMIT ?"
            )
            parameters.append(safe_limit)
            with self._connect() as connection:
                return connection.execute(sql, parameters).fetchall()

        rows = await asyncio.to_thread(_search)
        return [self._context_episode_from_row(row) for row in rows]

    async def tombstone_context_episode(
        self,
        episode_id: str,
        *,
        deleted_at: datetime | None = None,
    ) -> bool:
        timestamp = self._aware_iso(deleted_at or datetime.now(UTC))

        def _delete() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE context_episodes
                    SET deleted_at = ?, updated_at = ?
                    WHERE episode_id = ? AND deleted_at IS NULL
                    """,
                    (timestamp, timestamp, episode_id),
                )
                if cursor.rowcount and self._context_fts_enabled:
                    connection.execute(
                        "DELETE FROM context_episodes_fts WHERE episode_id = ?",
                        (episode_id,),
                    )
                return cursor.rowcount > 0

        return await asyncio.to_thread(_delete)

    async def prune_expired_context_records(
        self,
        *,
        now: datetime | None = None,
        dry_run: bool = True,
    ) -> dict[str, int]:
        cutoff = self._aware_iso(now or datetime.now(UTC))

        def _prune() -> dict[str, int]:
            with self._connect() as connection:
                event_rows = connection.execute(
                    """
                    SELECT event_id FROM context_events
                    WHERE deleted_at IS NULL
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (cutoff,),
                ).fetchall()
                episode_rows = connection.execute(
                    """
                    SELECT episode_id FROM context_episodes
                    WHERE deleted_at IS NULL
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (cutoff,),
                ).fetchall()
                if dry_run:
                    return {
                        "events": len(event_rows),
                        "episodes": len(episode_rows),
                    }
                connection.execute(
                    """
                    UPDATE context_events
                    SET deleted_at = ?, updated_at = ?
                    WHERE deleted_at IS NULL
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (cutoff, cutoff, cutoff),
                )
                connection.execute(
                    """
                    UPDATE context_episodes
                    SET deleted_at = ?, updated_at = ?
                    WHERE deleted_at IS NULL
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (cutoff, cutoff, cutoff),
                )
                if self._context_fts_enabled:
                    connection.executemany(
                        "DELETE FROM context_events_fts WHERE event_id = ?",
                        ((row["event_id"],) for row in event_rows),
                    )
                    connection.executemany(
                        "DELETE FROM context_episodes_fts WHERE episode_id = ?",
                        ((row["episode_id"],) for row in episode_rows),
                    )
                return {
                    "events": len(event_rows),
                    "episodes": len(episode_rows),
                }

        return await asyncio.to_thread(_prune)

    async def upsert_path_event(
        self,
        *,
        path: Path,
        display_name: str,
        root: Path,
        is_dir: bool,
        event: str,
        exists: bool,
        is_project: bool = False,
        project_type: str = "",
        seen_at: datetime | None = None,
    ) -> None:
        now = seen_at or datetime.now(UTC)
        normalized = os.path.normcase(os.path.abspath(path))
        deleted_at = now.isoformat() if not exists else None

        def _upsert() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO local_path_history(
                        normalized_path, display_name, root, is_dir, first_seen,
                        last_seen, last_event, exists_flag, deleted_at,
                        is_project, project_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(normalized_path) DO UPDATE SET
                        display_name = excluded.display_name,
                        root = excluded.root,
                        is_dir = excluded.is_dir,
                        last_seen = excluded.last_seen,
                        last_event = excluded.last_event,
                        exists_flag = excluded.exists_flag,
                        deleted_at = excluded.deleted_at,
                        is_project = excluded.is_project,
                        project_type = excluded.project_type
                    """,
                    (
                        normalized,
                        display_name[:500],
                        str(root)[:4000],
                        int(is_dir),
                        now.isoformat(),
                        now.isoformat(),
                        event[:20],
                        int(exists),
                        deleted_at,
                        int(is_project),
                        project_type[:80],
                    ),
                )

        await asyncio.to_thread(_upsert)

    async def list_path_history(self, *, limit: int = 100) -> list[PathHistoryItem]:
        safe_limit = max(1, min(limit, 1000))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT * FROM local_path_history ORDER BY last_seen DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()

        return [self._path_history_from_row(row) for row in await asyncio.to_thread(_list)]

    async def search_path_history(
        self, query: str, *, limit: int = 20
    ) -> list[PathHistoryItem]:
        safe_query = query.strip()
        if not safe_query:
            return []
        safe_limit = max(1, min(limit, 100))
        escaped = safe_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        def _search() -> list[sqlite3.Row]:
            with self._connect() as connection:
                return connection.execute(
                    """
                    SELECT * FROM local_path_history
                    WHERE display_name LIKE ? ESCAPE '\\'
                       OR normalized_path LIKE ? ESCAPE '\\'
                    ORDER BY is_project DESC, last_seen DESC
                    LIMIT ?
                    """,
                    (f"%{escaped}%", f"%{escaped}%", safe_limit),
                ).fetchall()

        return [self._path_history_from_row(row) for row in await asyncio.to_thread(_search)]

    async def prune_path_history(
        self,
        *,
        retention_days: int,
        max_records: int,
        now: datetime | None = None,
    ) -> int:
        cutoff = (now or datetime.now(UTC)) - timedelta(days=max(1, retention_days))
        safe_max = max(10, max_records)

        def _prune() -> int:
            with self._connect() as connection:
                old = connection.execute(
                    """
                    DELETE FROM local_path_history
                    WHERE last_seen < ? AND exists_flag = 0
                    """,
                    (cutoff.isoformat(),),
                ).rowcount
                excess = connection.execute(
                    """
                    DELETE FROM local_path_history
                    WHERE normalized_path IN (
                        SELECT normalized_path FROM local_path_history
                        ORDER BY exists_flag DESC, last_seen DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (safe_max,),
                ).rowcount
                return old + excess

        return await asyncio.to_thread(_prune)

    @staticmethod
    def _path_history_from_row(row: sqlite3.Row) -> PathHistoryItem:
        return PathHistoryItem(
            normalized_path=row["normalized_path"],
            display_name=row["display_name"],
            root=row["root"],
            is_dir=bool(row["is_dir"]),
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            last_event=row["last_event"],
            exists=bool(row["exists_flag"]),
            deleted_at=(
                datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None
            ),
            is_project=bool(row["is_project"]),
            project_type=row["project_type"],
        )

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

    async def add_message(
        self,
        role: str,
        content: str,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        def _add() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO conversation(
                        request_id, session_id, role, content, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        session_id[:120] if session_id else None,
                        role,
                        content[:20000],
                        _now_iso(),
                    ),
                )

        await asyncio.to_thread(_add)

    async def recent_messages(
        self,
        limit: int = 12,
        *,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        safe_limit = max(1, min(limit, 100))

        def _list() -> list[sqlite3.Row]:
            with self._connect() as connection:
                where = "WHERE session_id = ?" if session_id else ""
                parameters: tuple[Any, ...] = (
                    (session_id, safe_limit) if session_id else (safe_limit,)
                )
                return connection.execute(
                    f"""
                    SELECT role, content FROM (
                        SELECT id, role, content
                        FROM conversation
                        {where}
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    parameters,
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
        min_score: float = 0.0,
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
    def _aware_iso(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("context time filters must be timezone-aware")
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _context_query_tokens(query: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[\w-]+", query.casefold(), flags=re.UNICODE)
            if len(token) > 1
        ][:16]

    @classmethod
    def _context_fts_query(cls, query: str) -> str:
        tokens = cls._context_query_tokens(query)
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    @staticmethod
    def _context_event_from_row(row: sqlite3.Row) -> ContextEventV1:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        raw_foreground = row["is_foreground"]
        return ContextEventV1(
            event_id=row["event_id"],
            source=row["source"],
            source_id=row["source_id"],
            event_type=row["event_type"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
            app_name=row["app_name"],
            process_name=row["process_name"],
            window_title=row["window_title"],
            is_foreground=(
                None if raw_foreground is None else bool(raw_foreground)
            ),
            foreground_confidence=ForegroundConfidence(row["foreground_confidence"]),
            focus_duration_ms=row["focus_duration_ms"],
            text=row["text"],
            content_hash=row["content_hash"],
            metadata=metadata if isinstance(metadata, dict) else {},
            sensitivity=row["sensitivity"],
            expires_at=(
                datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            ),
            deleted_at=(
                datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _json_string_tuple(raw: str) -> tuple[str, ...]:
        try:
            value = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return ()
        if not isinstance(value, list):
            return ()
        return tuple(str(item) for item in value if str(item).strip())

    @classmethod
    def _context_episode_from_row(cls, row: sqlite3.Row) -> ContextEpisodeV1:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return ContextEpisodeV1(
            episode_id=row["episode_id"],
            source=row["source"],
            source_id=row["source_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]),
            title=row["title"],
            summary=row["summary"],
            source_event_ids=cls._json_string_tuple(row["source_event_ids_json"]),
            app_names=cls._json_string_tuple(row["app_names_json"]),
            person_ids=cls._json_string_tuple(row["person_ids_json"]),
            project_ids=cls._json_string_tuple(row["project_ids_json"]),
            vector_spaces=cls._json_string_tuple(row["vector_spaces_json"]),
            metadata=metadata if isinstance(metadata, dict) else {},
            content_hash=row["content_hash"],
            sensitivity=row["sensitivity"],
            expires_at=(
                datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            ),
            deleted_at=(
                datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

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
