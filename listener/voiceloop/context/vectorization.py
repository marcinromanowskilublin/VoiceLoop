from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..embeddings import OpenAICompatibleEmbeddingClient
from ..memory import MemoryStore
from ..memory_vectorization import (
    MEMORY_DOCUMENT_SCHEMA_VERSION,
    memory_vector_documents,
)
from ..qdrant_memory import QdrantVectorStore
from .schema import ContextEpisodeV1


@dataclass(frozen=True)
class EpisodeVectorizationResult:
    episode_id: str
    vector_spaces: tuple[str, ...]
    indexed: bool
    reason: str = "indexed"


@dataclass(frozen=True)
class EpisodeIndexingReport:
    collection: str
    considered: int
    indexed: int
    already_indexed: int
    would_index: int
    dry_run: bool


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
    ) -> EpisodeVectorizationResult:
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
        source_event_hashes = {}
        if self.memory is not None:
            for event_id in episode.source_event_ids:
                event = await self.memory.get_context_event(event_id)
                if event is not None:
                    source_event_hashes[event_id] = event.content_hash
        await self.qdrant.upsert_memory(
            source=episode.source,
            source_id=episode.source_id,
            title=episode.title,
            content=episode.summary,
            vectors=dict(zip(vector_names, vectors, strict=True)),
            metadata={
                **episode.metadata,
                "episode_id": episode.episode_id,
                "source_event_hashes": source_event_hashes,
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
        if self.memory is not None:
            await self.memory.upsert_context_episode(
                episode.model_copy(update={"vector_spaces": vector_names})
            )
        return EpisodeVectorizationResult(
            episode_id=episode.episode_id,
            vector_spaces=vector_names,
            indexed=True,
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
        if dry_run:
            return EpisodeIndexingReport(
                collection=self.qdrant.collection_name,
                considered=len(episodes),
                indexed=0,
                already_indexed=0,
                would_index=len(episodes),
                dry_run=True,
            )
        if not self.vectorizer.available():
            raise ContextEpisodeIndexError(
                "Indexer wymaga lokalnych embeddingów i lokalnego Qdrant."
            )

        collection_ready = await self.qdrant.collection_exists()
        indexed = 0
        already_indexed = 0
        for episode in episodes:
            if collection_ready and await self.qdrant.has_memory(
                source=episode.source,
                source_id=episode.source_id,
                content_hash=episode.content_hash,
            ):
                already_indexed += 1
                continue
            result = await self.vectorizer.index_episode(
                episode,
                topic=_metadata_text(episode.metadata.get("topic")),
                intent=_metadata_text(episode.metadata.get("intent")),
                decision=_metadata_text(episode.metadata.get("decision")),
                person_context=_metadata_text(
                    episode.metadata.get("person_context")
                    or episode.metadata.get("people")
                ),
            )
            if not result.indexed:
                raise ContextEpisodeIndexError(
                    f"Nie udało się zindeksować epizodu: {result.reason}."
                )
            collection_ready = True
            indexed += 1
        return EpisodeIndexingReport(
            collection=self.qdrant.collection_name,
            considered=len(episodes),
            indexed=indexed,
            already_indexed=already_indexed,
            would_index=0,
            dry_run=False,
        )


def _metadata_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | tuple | set):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return ""
