from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..embeddings import (
    EMBEDDING_PREFIX_POLICY_VERSION,
    OpenAICompatibleEmbeddingClient,
)
from ..memory import MemoryStore
from ..memory_vectorization import (
    MEMORY_DOCUMENT_SCHEMA_VERSION,
    memory_vector_documents,
)
from ..qdrant_memory import QdrantVectorStore
from .schema import (
    ContextEpisodeV1,
    ContextIndexOperation,
)


@dataclass(frozen=True)
class EpisodeVectorizationResult:
    episode_id: str
    vector_spaces: tuple[str, ...]
    indexed: bool
    reason: str = "indexed"
    index_fingerprint: str | None = None
    index_profile: str | None = None


@dataclass(frozen=True)
class EpisodeIndexingReport:
    collection: str
    considered: int
    indexed: int
    already_indexed: int
    would_index: int
    dry_run: bool
    deleted: int = 0
    not_requested: int = 0
    failed: int = 0
    outbox_pending: int = 0


class ContextEpisodeIndexError(RuntimeError):
    pass


def validate_episode_collection_isolation(
    *,
    primary_collection: str,
    episode_collection: str,
) -> None:
    primary = primary_collection.strip()
    candidate = episode_collection.strip()
    if not candidate:
        raise ContextEpisodeIndexError("Nazwa kolekcji epizodów nie może być pusta.")
    if candidate == primary:
        raise ContextEpisodeIndexError(
            "Kolekcja epizodów musi być oddzielona od aktywnej kolekcji pamięci."
        )


class ContextEpisodeVectorizer:
    """Indexes reviewed episode meaning, never raw timeline events."""

    def __init__(
        self,
        *,
        embeddings: OpenAICompatibleEmbeddingClient,
        qdrant: QdrantVectorStore,
        memory: MemoryStore | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.memory = memory

    def available(self) -> bool:
        return (
            self.embeddings.enabled
            and self.qdrant.enabled
            and self.embeddings.accepts_private_text()
            and self.qdrant.accepts_private_data()
        )

    async def index_episode(
        self,
        episode: ContextEpisodeV1,
        *,
        topic: str = "",
        intent: str = "",
        decision: str = "",
        person_context: str = "",
        source_event_hashes: dict[str, str] | None = None,
        vector_index_hash: str | None = None,
        index_profile: str | None = None,
    ) -> EpisodeVectorizationResult:
        if episode.metadata.get("vectorize") is False:
            return EpisodeVectorizationResult(
                episode_id=episode.episode_id,
                vector_spaces=(),
                indexed=False,
                reason="not_requested",
            )
        if not self.available():
            return EpisodeVectorizationResult(
                episode_id=episode.episode_id,
                vector_spaces=(),
                indexed=False,
                reason="provider_unavailable",
            )
        documents = memory_vector_documents(
            summary=episode.summary,
            topic=topic,
            intent=intent,
            decision=decision,
            person_context=person_context,
            redact=True,
        )
        vector_names = tuple(documents)
        vectors = await self.embeddings.embed_documents(
            [documents[name] for name in vector_names]
        )
        if len(vectors) != len(vector_names):
            raise RuntimeError("embedding count mismatch for context episode")
        resolved_source_hashes = dict(source_event_hashes or {})
        if source_event_hashes is None and self.memory is not None:
            for event_id in episode.source_event_ids:
                event = await self.memory.get_context_event(event_id)
                if event is not None:
                    resolved_source_hashes[event_id] = event.content_hash
        resolved_profile = index_profile or await _embedding_index_profile(
            self.embeddings
        )
        resolved_index_hash = vector_index_hash or _vector_index_hash(
            episode=episode,
            documents=documents,
            source_event_hashes=resolved_source_hashes,
            index_profile=resolved_profile,
        )
        try:
            await self.qdrant.upsert_memory(
                source=episode.source,
                source_id=episode.source_id,
                title=episode.title,
                content=episode.summary,
                vectors=dict(zip(vector_names, vectors, strict=True)),
                metadata={
                    **episode.metadata,
                    "episode_id": episode.episode_id,
                    "source_event_hashes": resolved_source_hashes,
                    "vector_index_hash": resolved_index_hash,
                    "index_profile": resolved_profile,
                    "started_at": episode.started_at.isoformat(),
                    "ended_at": episode.ended_at.isoformat(),
                    "time": episode.started_at.isoformat(),
                    "source_event_ids": list(episode.source_event_ids),
                    "app_names": list(episode.app_names),
                    "person_ids": list(episode.person_ids),
                    "project_ids": list(episode.project_ids),
                    "vector_spaces": list(vector_names),
                    "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
                    "provenance": {
                        "source": episode.source,
                        "source_id": episode.source_id,
                        "time": episode.started_at.isoformat(),
                        "confidence": 0.8,
                        "model": "context-episode-v1",
                        "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
                    },
                },
                memory_type="context_episode",
                content_hash=episode.content_hash,
                expires_at=episode.expires_at,
            )
        except Exception as exc:
            if self.memory is not None:
                await self.memory.mark_context_index_failed(
                    episode_id=episode.episode_id,
                    operation=ContextIndexOperation.UPSERT,
                    desired_revision=episode.content_hash,
                    error_type=type(exc).__name__,
                )
            raise
        if self.memory is not None:
            applied = await self.memory.mark_context_index_applied(
                episode_id=episode.episode_id,
                operation=ContextIndexOperation.UPSERT,
                desired_revision=episode.content_hash,
                index_fingerprint=resolved_index_hash,
                index_profile=resolved_profile,
                vector_spaces=vector_names,
            )
            if not applied:
                return EpisodeVectorizationResult(
                    episode_id=episode.episode_id,
                    vector_spaces=(),
                    indexed=False,
                    reason="superseded",
                    index_fingerprint=resolved_index_hash,
                    index_profile=resolved_profile,
                )
        return EpisodeVectorizationResult(
            episode_id=episode.episode_id,
            vector_spaces=vector_names,
            indexed=True,
            index_fingerprint=resolved_index_hash,
            index_profile=resolved_profile,
        )


class ContextEpisodeIndexer:
    """Explicit, idempotent indexer for the isolated episode collection."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        embeddings: OpenAICompatibleEmbeddingClient,
        qdrant: QdrantVectorStore,
        primary_collection: str,
    ) -> None:
        validate_episode_collection_isolation(
            primary_collection=primary_collection,
            episode_collection=qdrant.collection_name,
        )
        self.memory = memory
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.vectorizer = ContextEpisodeVectorizer(
            embeddings=embeddings,
            qdrant=qdrant,
            memory=memory,
        )

    async def index(
        self,
        *,
        limit: int = 200,
        offset: int = 0,
        dry_run: bool = True,
    ) -> EpisodeIndexingReport:
        episodes = await self.memory.list_context_episodes(
            limit=limit,
            offset=offset,
        )
        vectorizable = [
            episode
            for episode in episodes
            if episode.metadata.get("vectorize") is not False
        ]
        not_requested = len(episodes) - len(vectorizable)
        if dry_run:
            return EpisodeIndexingReport(
                collection=self.qdrant.collection_name,
                considered=len(episodes),
                indexed=0,
                already_indexed=0,
                would_index=len(vectorizable),
                dry_run=True,
                not_requested=not_requested,
            )
        if not self.vectorizer.available():
            raise ContextEpisodeIndexError(
                "Indexer wymaga lokalnych embeddingów i lokalnego Qdrant."
            )

        index_profile = await _embedding_index_profile(self.embeddings)
        deleted = await self._apply_delete_jobs(
            limit=limit,
            index_profile=index_profile,
        )
        collection_ready = await self.qdrant.collection_exists()
        indexed = 0
        already_indexed = 0
        for episode in vectorizable:
            aspects = {
                "topic": _metadata_text(episode.metadata.get("topic")),
                "intent": _metadata_text(episode.metadata.get("intent")),
                "decision": _metadata_text(episode.metadata.get("decision")),
                "person_context": _metadata_text(
                    episode.metadata.get("person_context")
                    or episode.metadata.get("people")
                ),
            }
            source_event_hashes = await self._complete_source_hashes(episode)
            documents = memory_vector_documents(
                summary=episode.summary,
                redact=True,
                **aspects,
            )
            vector_index_hash = _vector_index_hash(
                episode=episode,
                documents=documents,
                source_event_hashes=source_event_hashes,
                index_profile=index_profile,
            )
            if collection_ready and await self.qdrant.has_memory(
                source=episode.source,
                source_id=episode.source_id,
                content_hash=episode.content_hash,
                metadata_match={"vector_index_hash": vector_index_hash},
            ):
                applied = await self.memory.mark_context_index_applied(
                    episode_id=episode.episode_id,
                    operation=ContextIndexOperation.UPSERT,
                    desired_revision=episode.content_hash,
                    index_fingerprint=vector_index_hash,
                    index_profile=index_profile,
                    vector_spaces=tuple(documents),
                )
                if applied:
                    already_indexed += 1
                continue
            result = await self.vectorizer.index_episode(
                episode,
                **aspects,
                source_event_hashes=source_event_hashes,
                vector_index_hash=vector_index_hash,
                index_profile=index_profile,
            )
            if result.reason == "superseded":
                continue
            if not result.indexed:
                raise ContextEpisodeIndexError(
                    f"Nie udało się zindeksować epizodu: {result.reason}."
                )
            collection_ready = True
            indexed += 1
        index_summary = await self.memory.context_index_summary()
        return EpisodeIndexingReport(
            collection=self.qdrant.collection_name,
            considered=len(episodes),
            indexed=indexed,
            already_indexed=already_indexed,
            would_index=0,
            dry_run=False,
            deleted=deleted,
            not_requested=not_requested,
            outbox_pending=index_summary.get("outbox_ready", 0),
        )

    async def _apply_delete_jobs(
        self,
        *,
        limit: int,
        index_profile: str,
    ) -> int:
        jobs = await self.memory.list_context_index_jobs(
            operation=ContextIndexOperation.DELETE,
            limit=limit,
        )
        deleted = 0
        for job in jobs:
            episode = await self.memory.get_context_episode(job.episode_id)
            if episode is None:
                await self.memory.mark_context_index_failed(
                    episode_id=job.episode_id,
                    operation=job.operation,
                    desired_revision=job.desired_revision,
                    error_type="MissingCanonicalEpisode",
                )
                continue
            try:
                await self.qdrant.delete_memory(
                    source=episode.source,
                    source_id=episode.source_id,
                )
            except Exception as exc:
                await self.memory.mark_context_index_failed(
                    episode_id=job.episode_id,
                    operation=job.operation,
                    desired_revision=job.desired_revision,
                    error_type=type(exc).__name__,
                )
                raise
            fingerprint = hashlib.sha256(
                (
                    f"context-index-delete-v1:{job.episode_id}:"
                    f"{job.desired_revision}:{index_profile}"
                ).encode()
            ).hexdigest()
            if await self.memory.mark_context_index_applied(
                episode_id=job.episode_id,
                operation=job.operation,
                desired_revision=job.desired_revision,
                index_fingerprint=fingerprint,
                index_profile=index_profile,
            ):
                deleted += 1
        return deleted

    async def _complete_source_hashes(
        self,
        episode: ContextEpisodeV1,
    ) -> dict[str, str]:
        now = datetime.now(UTC)
        source_hashes: dict[str, str] = {}
        for event_id in episode.source_event_ids:
            event = await self.memory.get_context_event(event_id)
            if (
                event is None
                or event.deleted_at is not None
                or (event.expires_at is not None and event.expires_at <= now)
            ):
                raise ContextEpisodeIndexError(
                    f"Epizod {episode.episode_id} ma brakujące lub nieaktywne źródło."
                )
            source_hashes[event_id] = event.content_hash
        return source_hashes


def _metadata_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | tuple | set):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _vector_index_hash(
    *,
    episode: ContextEpisodeV1,
    documents: dict[str, str],
    source_event_hashes: dict[str, str],
    index_profile: str,
) -> str:
    payload = {
        "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
        "index_profile": index_profile,
        "episode_content_hash": episode.content_hash,
        "expires_at": episode.expires_at.isoformat() if episode.expires_at else None,
        "documents": documents,
        "source_event_hashes": source_event_hashes,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _embedding_index_profile(
    embeddings: OpenAICompatibleEmbeddingClient,
) -> str:
    resolver = getattr(embeddings, "resolve_model", None)
    if callable(resolver):
        resolved = resolver()
        model = str(await resolved if inspect.isawaitable(resolved) else resolved)
    else:
        model = str(
            getattr(embeddings, "configured_model", None)
            or getattr(embeddings, "_resolved_model", None)
            or type(embeddings).__name__
        )
    return (
        f"{MEMORY_DOCUMENT_SCHEMA_VERSION}:"
        f"{EMBEDDING_PREFIX_POLICY_VERSION}:{model}"
    )[:240]
