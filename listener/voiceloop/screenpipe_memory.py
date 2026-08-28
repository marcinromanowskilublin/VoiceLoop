from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from .behavior_digest import BehaviorDigestError, DigestedMemory, LocalBehaviorDigestClient
from .corpus.privacy import redact_text
from .embeddings import (
    EmbeddingUnavailableError,
    OpenAICompatibleEmbeddingClient,
    embedding_prefix_metadata,
)
from .memory import MemoryStore
from .memory_vectorization import MEMORY_DOCUMENT_SCHEMA_VERSION, memory_query_documents
from .qdrant_memory import QdrantMemoryError, QdrantUnavailableError, QdrantVectorStore
from .screenpipe import ScreenpipeClient, ScreenpipeContext, ScreenpipeError
from .settings import Settings

LOGGER = logging.getLogger("voiceloop.screenpipe_memory")
ACTIVITY_BUCKET_MINUTES = 10

# Zmierzone na kolekcji produkcyjnej (1090 kubełków Screenpipe). Poprzedni próg 0.92
# porównywał surową treść jako `search_query` z dokumentem zapisanym jako
# `search_document` wraz z nagłówkiem szablonu. Tekst identyczny bajt w bajt
# osiągał w tym układzie najwyżej 0.898, więc bramka nie mogła zadziałać nigdy —
# i nie zadziałała: 48% kolekcji to nadmiarowe kopie, największa grupa liczy 146
# punktów. Po zrównaniu przestrzeni ten sam tekst wraca dokładnie na 1.000,
# a dokumenty faktycznie różne mają p99 = 0.874. Stąd 0.97: osiągalne dla
# duplikatu, z zapasem nad rozkładem treści nowych.
ACTIVITY_DUPLICATE_MIN_SCORE = 0.97
OCR_NOISE_LINES = {
    "add this tab to bookmarks",
    "close",
    "copy message",
    "cursor",
    "edit",
    "exit full screen",
    "file",
    "fork chat",
    "help",
    "maximize",
    "minimize",
    "open new tab menu",
    "restore",
    "show sidebar",
    "thumbs down",
    "thumbs up",
    "view",
}


class ScreenpipeVectorMemoryWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        screenpipe: ScreenpipeClient,
        memory: MemoryStore,
        embeddings: OpenAICompatibleEmbeddingClient,
        qdrant: QdrantVectorStore | None = None,
        digester: LocalBehaviorDigestClient | None = None,
    ) -> None:
        self.settings = settings
        self.screenpipe = screenpipe
        self.memory = memory
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.digester = digester
        self.enabled = settings.screenpipe_vector_memory_enabled
        self.poll_seconds = max(
            30,
            (
                settings.behavior_digest_poll_seconds
                if settings.behavior_digest_enabled
                else settings.screenpipe_vector_poll_seconds
            ),
        )
        self.recent_minutes = max(
            5,
            (
                settings.behavior_digest_recent_minutes
                if settings.behavior_digest_enabled
                else settings.screenpipe_vector_recent_minutes
            ),
        )
        self.max_contexts = max(1, min(settings.behavior_digest_max_contexts, 50))
        self.dual_write = settings.qdrant_dual_write
        self.min_digest_confidence = max(
            0.0,
            min(float(settings.behavior_digest_min_confidence), 1.0),
        )
        ttl_days = max(0, int(settings.vector_memory_ttl_days))
        self.ttl_seconds = ttl_days * 86400 if ttl_days else None
        self.prune_enabled = bool(settings.vector_memory_prune_enabled)
        self.prune_interval_seconds = max(
            300,
            int(settings.vector_memory_prune_interval_seconds),
        )
        self._last_prune_monotonic = 0.0
        self._task: asyncio.Task[None] | None = None
        self._last_error: str | None = None
        self._last_indexed = 0
        self._migration_done = False

    async def start(self) -> None:
        if not self.enabled or not self.embeddings.enabled:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="screenpipe-vector-memory")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączone w konfiguracji"
        if not self.embeddings.enabled:
            return False, "lokalne embeddings są wyłączone"
        if self._last_error:
            return False, self._last_error
        if self._task and not self._task.done():
            return True, f"działa, ostatnio dodano: {self._last_indexed}"
        return False, "nie wystartował"

    async def _run(self) -> None:
        backoff_seconds = float(self.poll_seconds)
        while True:
            try:
                if not self._migration_done:
                    await self.migrate_legacy_memories()
                    self._migration_done = True
                await self.prune_expired_memories()
                self._last_indexed = await self.index_recent_activity()
                self._last_error = None
                backoff_seconds = float(self.poll_seconds)
            except asyncio.CancelledError:
                raise
            except QdrantUnavailableError as exc:
                # Awaria magazynu ≠ „duplikatu nie ma". Lepiej stracić kubełek
                # niż dopisać kopię, gdy nie umiemy sprawdzić, czy już leży.
                self._last_error = str(exc)[:500]
                LOGGER.warning("Screenpipe vector memory paused: %s", exc)
                backoff_seconds = min(max(backoff_seconds * 2, self.poll_seconds), 600.0)
            except ScreenpipeError as exc:
                self._last_error = str(exc)[:500]
                LOGGER.warning("Screenpipe vector memory paused: %s", exc)
                backoff_seconds = min(max(backoff_seconds * 2, self.poll_seconds), 600.0)
            except Exception as exc:
                self._last_error = str(exc)[:500]
                LOGGER.exception("Screenpipe vector memory indexing failed")
                backoff_seconds = min(max(backoff_seconds * 2, self.poll_seconds), 600.0)
            await asyncio.sleep(backoff_seconds)

    async def prune_expired_memories(self, *, force: bool = False) -> int:
        if not self.prune_enabled or self.qdrant is None:
            return 0
        now_monotonic = time.monotonic()
        if (
            not force
            and self._last_prune_monotonic
            and now_monotonic - self._last_prune_monotonic
            < self.prune_interval_seconds
        ):
            return 0
        qdrant_removed = await self.qdrant.prune_expired(dry_run=False)
        sqlite_removed = await self.memory.prune_expired_vector_memories()
        self._last_prune_monotonic = now_monotonic
        return qdrant_removed + sqlite_removed

    async def index_recent_activity(self) -> int:
        if self.qdrant is None or self.digester is None:
            return await self._index_legacy_activity()

        items = await self.screenpipe.recent_text_activity(
            minutes=self.recent_minutes,
            limit=self.max_contexts,
        )
        indexed = 0
        if items:
            now = self._activity_bucket(datetime.now(UTC))
            title = f"Aktywność użytkownika {now.isoformat()}"
            item_blocks = [
                block
                for item in items
                if (block := self._activity_item_content(item))
            ]
            content = "\n\n".join(item_blocks)[:20000]
            if not content.strip():
                indexed += await self._index_meeting_transcripts()
                return indexed
            if await self._looks_like_duplicate(content):
                LOGGER.info("Skipping duplicate Screenpipe activity bucket: %s", title)
                indexed += await self._index_meeting_transcripts()
                return indexed
            source_id = f"activity:{now.isoformat()}"
            metadata: dict[str, Any] = {
                "timestamp": now.isoformat(),
                "apps": sorted({item.app_name for item in items if item.app_name}),
                "windows": sorted({item.window_name for item in items if item.window_name})[:30],
                "content_types": sorted(
                    {item.content_type for item in items if item.content_type}
                ),
            }
            historical = await self._related_history(content)
            if historical:
                metadata["related_history"] = historical[:3]
            if await self._digest_and_store(
                source="screenpipe_behavior",
                source_id=source_id,
                title=title,
                content=content,
                metadata=metadata,
                memory_type="behavior_digest",
            ):
                indexed += 1

        indexed += await self._index_meeting_transcripts()
        return indexed

    async def _index_legacy_activity(self) -> int:
        contexts = await self.screenpipe.recent_activity(minutes=self.recent_minutes)
        documents: list[tuple[ScreenpipeContext, str, str, str]] = []
        for context in contexts:
            source_id = self._source_id(context)
            if await self.memory.has_vector_memory(
                source="screenpipe_activity",
                source_id=source_id,
            ):
                continue
            title = self._title(context)
            content = self._content(context)
            if content:
                documents.append((context, source_id, title, content))

        if not documents:
            return 0

        vectors = await self.embeddings.embed_documents([document[3] for document in documents])
        if len(vectors) != len(documents):
            raise EmbeddingUnavailableError("embedding count mismatch for Screenpipe documents")

        indexed = 0
        for (context, source_id, title, content), vector in zip(documents, vectors, strict=True):
            expires_at = self._expires_at()
            await self.memory.upsert_vector_memory(
                source="screenpipe_activity",
                source_id=source_id,
                title=title,
                content=content,
                embedding=vector,
                metadata={
                    "app_name": context.app_name,
                    "window_name": context.window_name,
                    "browser_url": context.browser_url,
                    "timestamp": context.timestamp,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "expires_at": expires_at.isoformat() if expires_at is not None else "",
                    **embedding_prefix_metadata("document"),
                },
            )
            indexed += 1
        return indexed

    async def _index_meeting_transcripts(self) -> int:
        if self.qdrant is None:
            return 0
        transcripts = await self.memory.list_screenpipe_transcripts(limit=500)
        grouped: dict[int, list[Any]] = defaultdict(list)
        for transcript in transcripts:
            if transcript.text.strip():
                grouped[transcript.meeting_id].append(transcript)

        indexed = 0
        for meeting_id, chunks in grouped.items():
            source_id = f"meeting:{meeting_id}"
            ordered = sorted(chunks, key=lambda item: item.start_time)
            full_content = "\n".join(
                (
                    f"[{item.start_time}–{item.end_time}] "
                    f"{item.device_type}/{item.device_name}: {item.text}"
                )
                for item in ordered
            )
            content_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest()
            if await self.qdrant.has_memory(
                source="screenpipe_meeting",
                source_id=source_id,
                content_hash=content_hash,
            ):
                continue
            content = full_content[:20000]
            metadata = {
                "meeting_id": meeting_id,
                "start_time": ordered[0].start_time,
                "end_time": ordered[-1].end_time,
                "devices": sorted({item.device_name for item in ordered}),
                "content_hash": content_hash,
            }
            if await self._digest_and_store(
                source="screenpipe_meeting",
                source_id=source_id,
                title=f"Rozmowa lub wizyta {meeting_id}",
                content=content,
                metadata=metadata,
                memory_type="meeting",
                content_hash=content_hash,
            ):
                indexed += 1
        return indexed

    async def _related_history(self, content: str) -> list[str]:
        if self.qdrant is None or not content.strip():
            return []
        try:
            safe_content = redact_text(content)[0]
            query_documents = memory_query_documents(safe_content[:2000])
            semantic_query = query_documents.get("semantic")
            if not semantic_query:
                return []
            query_vectors = await self.embeddings.embed_queries([semantic_query])
            if len(query_vectors) != 1:
                return []
            hits = await self.qdrant.search(
                query_vectors[0],
                limit=6,
                min_score=0.0,
                vector_names=("semantic",),
            )
        except (EmbeddingUnavailableError, QdrantMemoryError, AttributeError):
            return []
        # `hit.score` to fusion_score, a przy jednej osi jest funkcją samej rangi:
        # pierwszy wynik zawsze dostaje 1.000, choćby leżał daleko. Podajemy
        # rzeczywisty cosinus, bo ta liczba trafia do kontekstu modelu.
        return [
            f"{hit.title}: {hit.content[:1200]} "
            f"(cosinus={hit.vector_scores.get('semantic', hit.score):.3f})"
            for hit in hits
        ]

    async def _digest_and_store(
        self,
        *,
        source: str,
        source_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        memory_type: str,
        content_hash: str | None = None,
    ) -> bool:
        if self.qdrant is None or self.digester is None:
            return False
        source_content_hash = content_hash or hashlib.sha256(content.encode("utf-8")).hexdigest()
        safe_title, title_flags = redact_text(title)
        safe_content, content_flags = redact_text(content)
        safe_metadata, metadata_flags = self._redact_metadata(metadata)
        privacy_flags = set(title_flags) | set(content_flags) | set(metadata_flags)
        fallback_used = False
        try:
            digest = await self.digester.digest(
                title=safe_title,
                content=safe_content,
                metadata=safe_metadata,
            )
        except BehaviorDigestError as exc:
            LOGGER.warning(
                "Local behavior digest failed; using deterministic fallback: %s",
                exc,
            )
            fallback_used = True
            digest = self.digester.fallback(title=safe_title, content=safe_content)
        digest, digest_flags = self._redact_digest(digest)
        privacy_flags.update(digest_flags)
        if digest.confidence < self.min_digest_confidence:
            LOGGER.info(
                "Skipping low-confidence digest source=%s source_id=%s confidence=%.2f",
                source,
                source_id,
                digest.confidence,
            )
            return False
        vector_documents = digest.vector_documents(
            title=safe_title,
            raw_content=safe_content,
        )
        if not vector_documents:
            return False
        vector_names = tuple(vector_documents)
        texts = [vector_documents[name] for name in vector_names]
        vectors = await self.embeddings.embed_documents(texts)
        if len(vectors) != len(vector_names):
            raise EmbeddingUnavailableError("embedding count mismatch for named vectors")
        named_vectors = dict(zip(vector_names, vectors, strict=True))
        semantic_vector = named_vectors.get("semantic")
        if semantic_vector and await self._semantic_duplicate_exists(
            source=source,
            semantic_vector=semantic_vector,
        ):
            LOGGER.info(
                "Skipping semantic duplicate source=%s source_id=%s",
                source,
                source_id,
            )
            return False
        digest_model = self._component_model(self.digester)
        if fallback_used:
            digest_model = "deterministic-fallback-v2"
        embedding_model = self._component_model(self.embeddings)
        source_time = (
            safe_metadata.get("timestamp")
            or safe_metadata.get("start_time")
            or safe_metadata.get("end_time")
            or ""
        )
        provenance = {
            "source": source,
            "source_id": source_id,
            "time": source_time,
            "confidence": digest.confidence,
            "model": digest_model,
            "embedding_model": embedding_model,
            "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
        }
        enriched_metadata = {
            **safe_metadata,
            "people": digest.people,
            "observations": digest.observations,
            "confidence": digest.confidence,
            "content_hash": source_content_hash,
            "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
            "vector_spaces": list(vector_names),
            "vector_profile": "dynamic_named_subset_v2",
            "privacy_redactions": sorted(privacy_flags),
            "provenance": provenance,
        }
        expires_at = self._expires_at()
        if expires_at is not None:
            enriched_metadata["expires_at"] = expires_at.isoformat()
        if digest_model:
            enriched_metadata["model"] = digest_model
        if source_time:
            enriched_metadata["time"] = source_time
        await self.qdrant.upsert_memory(
            source=source,
            source_id=source_id,
            title=safe_title,
            content=digest.summary,
            vectors=named_vectors,
            metadata=enriched_metadata,
            memory_type=memory_type,
            content_hash=source_content_hash,
            expires_at=expires_at,
        )
        if self.dual_write and "semantic" in named_vectors:
            await self.memory.upsert_vector_memory(
                source=source,
                source_id=source_id,
                title=safe_title,
                content=digest.summary,
                embedding=named_vectors["semantic"],
                metadata=enriched_metadata,
            )
        return True

    @classmethod
    def _redact_metadata(
        cls,
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        value, flags = cls._redact_value(metadata)
        return (value if isinstance(value, dict) else {}), flags

    @classmethod
    def _redact_value(cls, value: Any) -> tuple[Any, list[str]]:
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            flags: list[str] = []
            for key, item in value.items():
                safe_item, item_flags = cls._redact_value(item)
                result[str(key)] = safe_item
                flags.extend(item_flags)
            return result, list(dict.fromkeys(flags))
        if isinstance(value, list | tuple | set):
            result_items: list[Any] = []
            flags = []
            for item in value:
                safe_item, item_flags = cls._redact_value(item)
                result_items.append(safe_item)
                flags.extend(item_flags)
            return result_items, list(dict.fromkeys(flags))
        return value, []

    @staticmethod
    def _redact_digest(digest: DigestedMemory) -> tuple[DigestedMemory, list[str]]:
        updates: dict[str, Any] = {}
        flags: list[str] = []
        for field_name in ("summary", "topic", "intent", "decision", "person_context"):
            safe_value, value_flags = redact_text(str(getattr(digest, field_name) or ""))
            updates[field_name] = safe_value
            flags.extend(value_flags)
        safe_people: list[str] = []
        for person in digest.people:
            safe_value, value_flags = redact_text(person)
            if safe_value:
                safe_people.append(safe_value)
            flags.extend(value_flags)
        safe_observations: list[str] = []
        for observation in digest.observations:
            safe_value, value_flags = redact_text(observation)
            if safe_value:
                safe_observations.append(safe_value)
            flags.extend(value_flags)
        updates["people"] = safe_people
        updates["observations"] = safe_observations
        return digest.model_copy(update=updates), list(dict.fromkeys(flags))

    @staticmethod
    def _component_model(component: Any) -> str:
        for attribute in ("_resolved_model", "configured_model", "model"):
            value = getattr(component, attribute, None)
            if value is not None and str(value).strip():
                return str(value).strip()[:500]
        return ""

    async def migrate_legacy_memories(self) -> int:
        if self.qdrant is None or not self.qdrant.enabled:
            return 0
        memories = await self.memory.list_stored_vector_memories(limit=100000)
        state_key = "qdrant_legacy_migration_v2_semantic_only"
        migration_done = await self.memory.get_state(state_key) == "done"
        if migration_done and (
            not memories
            or await self.qdrant.has_memory(
                source=memories[0].source,
                source_id=memories[0].source_id,
            )
        ):
            return 0
        migrated = 0
        for item in memories:
            if not item.embedding:
                continue
            existing_payload: dict[str, Any] | None = None
            payload_reader = getattr(self.qdrant, "get_memory_payload", None)
            if callable(payload_reader):
                existing_payload = await payload_reader(
                    source=item.source,
                    source_id=item.source_id,
                )
            if existing_payload is not None:
                if existing_payload.get("memory_type") != "legacy":
                    continue
            elif await self.qdrant.has_memory(source=item.source, source_id=item.source_id):
                continue
            safe_metadata, privacy_flags = self._redact_metadata(item.metadata)
            item_content_hash = hashlib.sha256(item.content.encode("utf-8")).hexdigest()
            await self.qdrant.upsert_memory(
                source=item.source,
                source_id=item.source_id,
                title=redact_text(item.title)[0],
                content=redact_text(item.content)[0],
                vectors={"semantic": item.embedding},
                metadata={
                    **safe_metadata,
                    "content_hash": item_content_hash,
                    "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
                    "vector_spaces": ["semantic"],
                    "vector_profile": "legacy_semantic_only_v2",
                    "privacy_redactions": privacy_flags,
                    "provenance": {
                        "source": item.source,
                        "source_id": item.source_id,
                        "time": item.created_at.isoformat(),
                        "confidence": safe_metadata.get("confidence"),
                        "model": safe_metadata.get("model"),
                        "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
                    },
                },
                memory_type="legacy",
                content_hash=item_content_hash,
                ttl_seconds=self.ttl_seconds,
            )
            migrated += 1
        await self.memory.set_state(state_key, "done")
        return migrated

    def _expires_at(self) -> datetime | None:
        if self.ttl_seconds is None:
            return None
        return datetime.now(UTC) + timedelta(seconds=self.ttl_seconds)

    @staticmethod
    def _source_id(context: ScreenpipeContext) -> str:
        raw = "\n".join(
            [
                context.timestamp,
                context.app_name,
                context.window_name,
                context.browser_url,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _title(context: ScreenpipeContext) -> str:
        if context.app_name and context.window_name:
            return f"{context.app_name}: {context.window_name}"
        return context.window_name or context.app_name or "Screenpipe activity"

    @staticmethod
    def _content(context: ScreenpipeContext) -> str:
        parts = [
            f"Aplikacja: {context.app_name}" if context.app_name else "",
            f"Okno: {context.window_name}" if context.window_name else "",
            f"URL: {context.browser_url}" if context.browser_url else "",
            f"Czas: {context.timestamp}" if context.timestamp else "",
        ]
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _activity_bucket(value: datetime) -> datetime:
        minute = value.minute - (value.minute % ACTIVITY_BUCKET_MINUTES)
        return value.replace(minute=minute, second=0, microsecond=0)

    @classmethod
    def _activity_item_content(cls, item: Any) -> str:
        text = cls._clean_ocr_text(str(getattr(item, "text", "") or ""))
        if not text:
            return ""
        parts = [
            f"[{getattr(item, 'timestamp', '')}]",
            f"typ={getattr(item, 'content_type', '')}",
            f"aplikacja={getattr(item, 'app_name', '')}",
            f"okno={getattr(item, 'window_name', '')}",
        ]
        browser_url = str(getattr(item, "browser_url", "") or "").strip()
        if browser_url:
            parts.append(f"url={browser_url}")
        return f"{'; '.join(part for part in parts if part)}\n{text}".strip()

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        cleaned_lines: list[str] = []
        seen_recent: set[str] = set()
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            normalized = line.casefold()
            if normalized in OCR_NOISE_LINES:
                continue
            if normalized.startswith(("app://", "file://", "vscode-file://")):
                continue
            if re.fullmatch(r"[|/\\\-–—_.: ]+", line):
                continue
            if len(line) <= 2 and not any(character.isdigit() for character in line):
                continue
            if normalized in seen_recent:
                continue
            cleaned_lines.append(line)
            seen_recent.add(normalized)
            if len(seen_recent) > 200:
                seen_recent.clear()
        return "\n".join(cleaned_lines).strip()

    async def _looks_like_duplicate(self, content: str) -> bool:
        """Czy ten kubełek aktywności już jest w pamięci, sprawdzone na tanio.

        Sam odcisk treści, bez wektorów. To właśnie powtórzony bajt w bajt kubełek
        zapchał kolekcję (największa grupa identycznych punktów liczyła 146), a
        wykrycie go nie wymaga ani jednego wywołania modelu.

        Parafrazy tu nie sprawdzamy, i to jest decyzja, nie przeoczenie. Przed
        digestem mamy surowy OCR, a w Qdrancie leży podsumowanie od modelu owinięte
        w szablon. To dwa różne rodzaje tekstu, więc żaden próg dla tej pary nie
        jest skalibrowany — a to dokładnie ten błąd unieruchomił poprzednią wersję.
        Porównanie semantyczne robimy po digeście, w `_semantic_duplicate_exists`,
        gdzie oba teksty są tego samego rodzaju i gdzie i tak mamy już gotowy wektor.
        """

        if self.qdrant is None or not content.strip():
            return False
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        try:
            return await self.qdrant.has_content_hash(
                content_hash=content_hash,
                source="screenpipe_behavior",
            )
        except AttributeError as exc:
            raise QdrantUnavailableError(
                "Qdrant store does not support content-hash duplicate checks."
            ) from exc

    async def _semantic_duplicate_exists(
        self,
        *,
        source: str,
        semantic_vector: list[float],
    ) -> bool:
        """Czy w pamięci leży już dokument o tym samym znaczeniu.

        Wektor jest ten sam, którym zaraz zapiszemy punkt, więc porównanie idzie
        dokument do dokumentu w identycznej przestrzeni i nie kosztuje dodatkowego
        embeddingu. Dopiero w tym układzie próg da się skalibrować: tekst
        identyczny wraca dokładnie na 1.000, a dokumenty faktycznie różne mają
        p99 = 0.874.

        Przy niedostępności Qdranta rzuca `QdrantUnavailableError` — wcześniej
        zwracaliśmy `False` i zapisywaliśmy punkt mimo braku sprawdzenia.
        """

        if self.qdrant is None or not semantic_vector:
            return False
        try:
            hits = await self.qdrant.search(
                semantic_vector,
                limit=1,
                source=source,
                min_score=ACTIVITY_DUPLICATE_MIN_SCORE,
                vector_names=("semantic",),
            )
        except QdrantUnavailableError:
            raise
        except AttributeError:
            return False
        return bool(hits)
