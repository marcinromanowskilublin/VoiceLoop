from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from .behavior_digest import BehaviorDigestError, LocalBehaviorDigestClient
from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .memory import MemoryStore
from .qdrant_memory import VECTOR_NAMES, QdrantMemoryError, QdrantVectorStore
from .screenpipe import ScreenpipeClient, ScreenpipeContext
from .settings import Settings

LOGGER = logging.getLogger("voiceloop.screenpipe_memory")
CORE_VECTOR_NAMES = ("semantic", "intent", "person_context")
MEETING_VECTOR_NAMES = VECTOR_NAMES


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
        while True:
            try:
                if not self._migration_done:
                    await self.migrate_legacy_memories()
                    self._migration_done = True
                self._last_indexed = await self.index_recent_activity()
                self._last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)[:500]
                LOGGER.exception("Screenpipe vector memory indexing failed")
            await asyncio.sleep(self.poll_seconds)

    async def index_recent_activity(self) -> int:
        if self.qdrant is None or self.digester is None:
            return await self._index_legacy_activity()

        items = await self.screenpipe.recent_text_activity(
            minutes=self.recent_minutes,
            limit=self.max_contexts,
        )
        indexed = 0
        if items:
            now = datetime.now(UTC).replace(second=0, microsecond=0)
            title = f"Aktywność użytkownika {now.isoformat()}"
            content = "\n\n".join(
                (
                    f"[{item.timestamp}] typ={item.content_type}; "
                    f"aplikacja={item.app_name}; okno={item.window_name}; "
                    f"url={item.browser_url}\n{item.text}"
                ).strip()
                for item in items
            )[:20000]
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
                content = (
                    f"{content}\n\nPOWIĄZANE WCZEŚNIEJSZE OBSERWACJE:\n"
                    + "\n---\n".join(historical)
                )[:20000]
            await self._digest_and_store(
                source="screenpipe_behavior",
                source_id=source_id,
                title=title,
                content=content,
                metadata=metadata,
                memory_type="behavior_digest",
                vector_names=CORE_VECTOR_NAMES,
            )
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

        vectors = await self.embeddings.embed_texts([document[3] for document in documents])
        if len(vectors) != len(documents):
            raise EmbeddingUnavailableError("embedding count mismatch for Screenpipe documents")

        indexed = 0
        for (context, source_id, title, content), vector in zip(documents, vectors, strict=True):
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
            if await self.qdrant.has_memory(
                source="screenpipe_meeting",
                source_id=source_id,
            ):
                continue
            ordered = sorted(chunks, key=lambda item: item.start_time)
            content = "\n".join(
                (
                    f"[{item.start_time}–{item.end_time}] "
                    f"{item.device_type}/{item.device_name}: {item.text}"
                )
                for item in ordered
            )[:20000]
            metadata = {
                "meeting_id": meeting_id,
                "start_time": ordered[0].start_time,
                "end_time": ordered[-1].end_time,
                "devices": sorted({item.device_name for item in ordered}),
            }
            await self._digest_and_store(
                source="screenpipe_meeting",
                source_id=source_id,
                title=f"Rozmowa lub wizyta {meeting_id}",
                content=content,
                metadata=metadata,
                memory_type="meeting",
                vector_names=MEETING_VECTOR_NAMES,
            )
            indexed += 1
        return indexed

    async def _related_history(self, content: str) -> list[str]:
        if self.qdrant is None or not content.strip():
            return []
        try:
            query = await self.embeddings.embed_query(content[:2000])
            hits = await self.qdrant.search(query, limit=6, min_score=0.30)
        except (EmbeddingUnavailableError, QdrantMemoryError, AttributeError):
            return []
        return [
            f"{hit.title}: {hit.content[:1200]} (score={hit.score:.3f})"
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
        vector_names: tuple[str, ...],
    ) -> None:
        if self.qdrant is None or self.digester is None:
            return
        try:
            digest = await self.digester.digest(
                title=title,
                content=content,
                metadata=metadata,
            )
        except BehaviorDigestError as exc:
            LOGGER.warning(
                "Local behavior digest failed; using deterministic fallback: %s",
                exc,
            )
            digest = self.digester.fallback(title=title, content=content)
        vector_documents = digest.vector_documents(title=title, raw_content=content)
        texts = [vector_documents[name] for name in vector_names]
        vectors = await self.embeddings.embed_documents(texts)
        if len(vectors) != len(vector_names):
            raise EmbeddingUnavailableError("embedding count mismatch for named vectors")
        named_vectors = dict(zip(vector_names, vectors, strict=True))
        profile = "five_vectors_meeting" if vector_names == MEETING_VECTOR_NAMES else "three_vectors_core"
        enriched_metadata = {
            **metadata,
            "people": digest.people,
            "observations": digest.observations,
            "confidence": digest.confidence,
            "vector_profile": profile,
        }
        await self.qdrant.upsert_memory(
            source=source,
            source_id=source_id,
            title=title,
            content=digest.summary,
            vectors=named_vectors,
            metadata=enriched_metadata,
            memory_type=memory_type,
        )
        if self.dual_write:
            await self.memory.upsert_vector_memory(
                source=source,
                source_id=source_id,
                title=title,
                content=digest.summary,
                embedding=named_vectors["semantic"],
                metadata=enriched_metadata,
            )

    async def migrate_legacy_memories(self) -> int:
        if self.qdrant is None or not self.qdrant.enabled:
            return 0
        memories = await self.memory.list_stored_vector_memories(limit=100000)
        migration_done = await self.memory.get_state("qdrant_legacy_migration_v1") == "done"
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
            if await self.qdrant.has_memory(source=item.source, source_id=item.source_id):
                continue
            await self.qdrant.upsert_memory(
                source=item.source,
                source_id=item.source_id,
                title=item.title,
                content=item.content,
                vectors={name: item.embedding for name in VECTOR_NAMES},
                metadata=item.metadata,
                memory_type="legacy",
            )
            migrated += 1
        await self.memory.set_state("qdrant_legacy_migration_v1", "done")
        return migrated

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
