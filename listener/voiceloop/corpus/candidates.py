from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ..memory import MemoryStore
from ..models import MemoryCreate, MemoryItem
from ..router import normalize_text
from .privacy import redact_text, sensitive_reason
from .schema import (
    CandidateStatus,
    CorpusSplit,
    MemoryCandidate,
    MemoryCandidateCreate,
    MemoryCandidateKind,
    SpeakerStatus,
    UtteranceRecord,
)
from .storage import sha256_text

_PREFERENCE_PATTERN = re.compile(
    r"\b(?:wolę|preferuję|lubię|nie lubię|chcę,?\s+żeby|"
    r"odpowiadaj|nie chcę|zawsze proszę)\b",
    re.IGNORECASE,
)
_FACT_PATTERN = re.compile(
    r"\b(?:moim celem jest|pracuję nad|mój projekt|używam na co dzień)\b",
    re.IGNORECASE,
)
_BLOCKED_CONTENT = "[BLOCKED_SENSITIVE_CONTENT]"


class CandidateStoreError(RuntimeError):
    pass


class CandidateDecisionError(CandidateStoreError):
    pass


class MemoryCandidateStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._decision_lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        def initialize_sync() -> None:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_candidates (
                        candidate_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        proposed_content TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        manifest_id TEXT,
                        evidence_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL,
                        block_reason TEXT,
                        created_at TEXT NOT NULL,
                        decided_at TEXT,
                        memory_id INTEGER
                    );
                    CREATE INDEX IF NOT EXISTS idx_memory_candidates_status
                        ON memory_candidates(status, created_at DESC);
                    CREATE TABLE IF NOT EXISTS corpus_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(memory_candidates)"
                    ).fetchall()
                }
                if "content_sha256" not in columns:
                    connection.execute(
                        "ALTER TABLE memory_candidates ADD COLUMN content_sha256 TEXT"
                    )
                if "manifest_id" not in columns:
                    connection.execute(
                        "ALTER TABLE memory_candidates ADD COLUMN manifest_id TEXT"
                    )
                rows = connection.execute(
                    """
                    SELECT candidate_id, proposed_content, status, block_reason
                    FROM memory_candidates
                    """
                ).fetchall()
                for row in rows:
                    original = str(row["proposed_content"])
                    redacted, _ = redact_text(original)
                    detected_reason = sensitive_reason(original)
                    existing_reason = (
                        str(row["block_reason"])
                        if row["block_reason"]
                        else None
                    )
                    block_reason = detected_reason or existing_reason
                    safe_content = _BLOCKED_CONTENT if block_reason else redacted
                    safe_status = (
                        CandidateStatus.BLOCKED.value
                        if block_reason
                        else str(row["status"])
                    )
                    connection.execute(
                        """
                        UPDATE memory_candidates
                        SET proposed_content = ?, content_sha256 = ?,
                            status = ?, block_reason = ?
                        WHERE candidate_id = ?
                        """,
                        (
                            safe_content,
                            sha256_text(safe_content),
                            safe_status,
                            block_reason,
                            str(row["candidate_id"]),
                        ),
                    )

        await asyncio.to_thread(initialize_sync)

    async def upsert(self, candidate: MemoryCandidateCreate) -> MemoryCandidate:
        active_scope = await self.active_scope()
        candidate = _sanitize_candidate(candidate, active_scope=active_scope)
        content_hash = sha256_text(candidate.proposed_content)
        created_at = _now_iso()

        def upsert_sync() -> None:
            with self._connect() as connection:
                existing = connection.execute(
                    """
                    SELECT kind, content_sha256, status
                    FROM memory_candidates WHERE candidate_id = ?
                    """,
                    (candidate.candidate_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["content_sha256"]) != content_hash
                        or str(existing["kind"]) != candidate.kind.value
                    ):
                        raise CandidateStoreError(
                            "Identyfikator kandydata jest związany z inną treścią."
                        )
                    if str(existing["status"]) == CandidateStatus.PENDING.value:
                        connection.execute(
                            """
                            UPDATE memory_candidates
                            SET evidence_json = ?, manifest_id = ?,
                                status = ?, block_reason = ?
                            WHERE candidate_id = ?
                            """,
                            (
                                json.dumps(candidate.evidence_utterance_ids),
                                candidate.manifest_id,
                                candidate.status.value,
                                candidate.block_reason,
                                candidate.candidate_id,
                            ),
                        )
                    return
                connection.execute(
                    """
                    INSERT INTO memory_candidates (
                        candidate_id, kind, proposed_content, content_sha256,
                        manifest_id, evidence_json, status, block_reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        candidate.kind.value,
                        candidate.proposed_content,
                        content_hash,
                        candidate.manifest_id,
                        json.dumps(candidate.evidence_utterance_ids),
                        candidate.status.value,
                        candidate.block_reason,
                        created_at,
                    ),
                )

        await asyncio.to_thread(upsert_sync)
        stored = await self.get(candidate.candidate_id)
        if stored is None:
            raise CandidateStoreError("Nie udało się zapisać kandydata pamięci.")
        return stored

    async def get(self, candidate_id: str) -> MemoryCandidate | None:
        def get_sync() -> sqlite3.Row | None:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT * FROM memory_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()

        row = await asyncio.to_thread(get_sync)
        return _row_to_candidate(row) if row is not None else None

    async def list(
        self,
        *,
        status: CandidateStatus | None = CandidateStatus.PENDING,
        limit: int = 100,
    ) -> list[MemoryCandidate]:
        safe_limit = max(1, min(limit, 500))

        def list_sync() -> list[sqlite3.Row]:
            with self._connect() as connection:
                if status is None:
                    return connection.execute(
                        """
                        SELECT * FROM memory_candidates
                        ORDER BY created_at DESC LIMIT ?
                        """,
                        (safe_limit,),
                    ).fetchall()
                return connection.execute(
                    """
                    SELECT * FROM memory_candidates
                    WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (status.value, safe_limit),
                ).fetchall()

        rows = await asyncio.to_thread(list_sync)
        return [_row_to_candidate(row) for row in rows]

    async def reject(self, candidate_id: str) -> MemoryCandidate:
        async with self._decision_lock:
            candidate = await self.get(candidate_id)
            if candidate is None:
                raise CandidateDecisionError("Nie znaleziono kandydata pamięci.")
            if candidate.status is CandidateStatus.APPROVED:
                raise CandidateDecisionError("Zatwierdzonego kandydata nie można odrzucić.")
            if candidate.status is CandidateStatus.BLOCKED:
                raise CandidateDecisionError("Zablokowany kandydat nie podlega decyzji.")
            if candidate.status is CandidateStatus.REJECTED:
                return candidate
            changed = await self._set_decision(
                candidate_id,
                CandidateStatus.REJECTED,
            )
            if not changed:
                raise CandidateDecisionError(
                    "Stan kandydata zmienił się podczas odrzucania."
                )
            updated = await self.get(candidate_id)
            if updated is None:
                raise CandidateStoreError("Nie udało się odczytać decyzji.")
            return updated

    async def approve(
        self,
        candidate_id: str,
        memory: MemoryStore,
        *,
        expected_content_sha256: str,
    ) -> tuple[MemoryCandidate, MemoryItem]:
        async with self._decision_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM memory_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                if row is None:
                    raise CandidateDecisionError(
                        "Nie znaleziono kandydata pamięci."
                    )
                candidate = _row_to_candidate(row)
                scope_row = connection.execute(
                    "SELECT value FROM corpus_state WHERE key = 'active_scope'"
                ).fetchone()
                active_scope = (
                    str(scope_row["value"]) if scope_row is not None else None
                )
                if candidate.content_sha256 != expected_content_sha256:
                    raise CandidateDecisionError(
                        "Hash treści nie odpowiada kandydatowi pokazanemu "
                        "do akceptacji."
                    )
                if not active_scope or candidate.manifest_id != active_scope:
                    _block_candidate_sync(
                        connection,
                        candidate_id,
                        "stale_manifest",
                    )
                    connection.commit()
                    raise CandidateDecisionError(
                        "Kandydat nie należy do bieżącego zakresu korpusu."
                    )
                if candidate.status in {
                    CandidateStatus.BLOCKED,
                    CandidateStatus.REJECTED,
                }:
                    raise CandidateDecisionError(
                        f"Kandydat ma status {candidate.status.value} "
                        "i nie może być zatwierdzony."
                    )
                redacted_content, _ = redact_text(candidate.proposed_content)
                block_reason = sensitive_reason(candidate.proposed_content)
                if block_reason or redacted_content != candidate.proposed_content:
                    _block_candidate_sync(
                        connection,
                        candidate_id,
                        block_reason or "pii_changed_after_review",
                    )
                    connection.commit()
                    raise CandidateDecisionError(
                        "Kandydat nie przeszedł ponownej bramki prywatności."
                    )
                if candidate.status is CandidateStatus.APPROVED and candidate.memory_id:
                    existing_by_id = await memory.list_memories(
                        limit=1,
                        memory_id=candidate.memory_id,
                    )
                    if existing_by_id:
                        connection.commit()
                        return candidate, existing_by_id[0]

                existing = await memory.list_memories(
                    limit=500,
                    kind=candidate.kind.value,
                )
                normalized = normalize_text(candidate.proposed_content)
                item = next(
                    (
                        memory_item
                        for memory_item in existing
                        if normalize_text(memory_item.content) == normalized
                    ),
                    None,
                )
                if item is None:
                    item = await memory.create_memory(
                        MemoryCreate(
                            kind=candidate.kind.value,
                            content=candidate.proposed_content,
                            sensitivity="private",
                            source="corpus_approved",
                        )
                    )
                cursor = connection.execute(
                    """
                    UPDATE memory_candidates
                    SET status = 'approved', decided_at = ?, memory_id = ?
                    WHERE candidate_id = ? AND status = 'pending'
                      AND manifest_id = ? AND content_sha256 = ?
                    """,
                    (
                        _now_iso(),
                        item.id,
                        candidate_id,
                        active_scope,
                        expected_content_sha256,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CandidateDecisionError(
                        "Stan kandydata zmienił się podczas zatwierdzania."
                    )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
            updated = await self.get(candidate_id)
            if updated is None:
                raise CandidateStoreError("Nie udało się odczytać zatwierdzenia.")
            return updated, item

    async def block_pending_not_manifest(self, manifest_id: str) -> int:
        def block_sync() -> int:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE memory_candidates
                    SET proposed_content = ?, content_sha256 = ?,
                        status = 'blocked', block_reason = 'stale_manifest',
                        decided_at = ?, memory_id = NULL
                    WHERE status = 'pending'
                      AND (manifest_id IS NULL OR manifest_id != ?)
                    """,
                    (
                        _BLOCKED_CONTENT,
                        sha256_text(_BLOCKED_CONTENT),
                        _now_iso(),
                        manifest_id,
                    ),
                )
                return max(0, int(cursor.rowcount))

        return await asyncio.to_thread(block_sync)

    async def set_active_scope(self, scope_id: str) -> int:
        if not scope_id:
            raise CandidateStoreError("Aktywny scope korpusu nie może być pusty.")

        def set_sync() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO corpus_state (key, value, updated_at)
                    VALUES ('active_scope', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (scope_id, _now_iso()),
                )

        await asyncio.to_thread(set_sync)
        return await self.block_pending_not_manifest(scope_id)

    async def active_scope(self) -> str | None:
        def get_sync() -> str | None:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value FROM corpus_state WHERE key = 'active_scope'"
                ).fetchone()
                return str(row["value"]) if row is not None else None

        return await asyncio.to_thread(get_sync)

    async def health(self) -> tuple[bool, str]:
        try:
            pending = len(await self.list(status=CandidateStatus.PENDING, limit=500))
        except Exception as exc:
            return False, f"niedostępna: {type(exc).__name__}"
        return True, f"kolejka kandydatów: {pending} oczekujących"

    async def _set_decision(
        self,
        candidate_id: str,
        status: CandidateStatus,
        *,
        memory_id: int | None = None,
    ) -> bool:
        decided_at = _now_iso()

        def set_sync() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE memory_candidates
                    SET status = ?, decided_at = ?, memory_id = ?
                    WHERE candidate_id = ? AND status = 'pending'
                    """,
                    (status.value, decided_at, memory_id, candidate_id),
                )
                return cursor.rowcount == 1

        return await asyncio.to_thread(set_sync)

    async def _block_candidate(self, candidate_id: str, reason: str) -> None:
        def block_sync() -> None:
            with self._connect() as connection:
                _block_candidate_sync(connection, candidate_id, reason)

        await asyncio.to_thread(block_sync)


def extract_memory_candidates(
    records: list[UtteranceRecord],
    *,
    max_candidates: int = 200,
    manifest_id: str | None = None,
) -> list[MemoryCandidateCreate]:
    candidates: list[MemoryCandidateCreate] = []
    seen: set[str] = set()
    for record in records:
        if len(candidates) >= max_candidates:
            break
        if (
            record.speaker_status is not SpeakerStatus.SELF
            or record.is_near_duplicate
            or record.quarantine_reason
            or record.split is CorpusSplit.HOLDOUT
            or record.word_count > 80
        ):
            continue
        kind = _candidate_kind(record.text)
        if kind is None:
            continue
        normalized = normalize_text(record.text)
        candidate_id = sha256_text(f"{kind.value}:{normalized}")[:24]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        block_reason = sensitive_reason(record.text)
        status = CandidateStatus.BLOCKED if block_reason else CandidateStatus.PENDING
        candidates.append(
            MemoryCandidateCreate(
                candidate_id=candidate_id,
                kind=kind,
                proposed_content=record.text,
                manifest_id=manifest_id,
                evidence_utterance_ids=(record.utterance_id,),
                status=status,
                block_reason=block_reason,
            )
        )
    return candidates


def _sanitize_candidate(
    candidate: MemoryCandidateCreate,
    *,
    active_scope: str | None,
) -> MemoryCandidateCreate:
    redacted_content, _ = redact_text(candidate.proposed_content)
    detected_reason = sensitive_reason(candidate.proposed_content)
    declared_reason = (
        candidate.block_reason
        if candidate.status is CandidateStatus.BLOCKED
        else None
    )
    scope_id = candidate.manifest_id or active_scope
    scope_reason = None if scope_id else "missing_active_scope"
    block_reason = detected_reason or declared_reason or scope_reason
    safe_content = _BLOCKED_CONTENT if block_reason else redacted_content
    return candidate.model_copy(
        update={
            "proposed_content": safe_content,
            "content_sha256": sha256_text(safe_content),
            "manifest_id": scope_id,
            "status": (
                CandidateStatus.BLOCKED
                if block_reason
                else CandidateStatus.PENDING
            ),
            "block_reason": block_reason,
        }
    )


def _candidate_kind(text: str) -> MemoryCandidateKind | None:
    if _PREFERENCE_PATTERN.search(text):
        return MemoryCandidateKind.PREFERENCE
    if _FACT_PATTERN.search(text):
        return MemoryCandidateKind.FACT
    return None


def _block_candidate_sync(
    connection: sqlite3.Connection,
    candidate_id: str,
    reason: str,
) -> None:
    connection.execute(
        """
        UPDATE memory_candidates
        SET proposed_content = ?, content_sha256 = ?,
            status = 'blocked', block_reason = ?,
            decided_at = ?, memory_id = NULL
        WHERE candidate_id = ?
        """,
        (
            _BLOCKED_CONTENT,
            sha256_text(_BLOCKED_CONTENT),
            reason,
            _now_iso(),
            candidate_id,
        ),
    )


def _row_to_candidate(row: sqlite3.Row) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=str(row["candidate_id"]),
        kind=MemoryCandidateKind(str(row["kind"])),
        proposed_content=str(row["proposed_content"]),
        content_sha256=str(row["content_sha256"]),
        manifest_id=str(row["manifest_id"]) if row["manifest_id"] else None,
        evidence_utterance_ids=tuple(json.loads(row["evidence_json"])),
        status=CandidateStatus(str(row["status"])),
        block_reason=str(row["block_reason"]) if row["block_reason"] else None,
        created_at=datetime.fromisoformat(str(row["created_at"])),
        decided_at=(
            datetime.fromisoformat(str(row["decided_at"]))
            if row["decided_at"]
            else None
        ),
        memory_id=int(row["memory_id"]) if row["memory_id"] is not None else None,
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
