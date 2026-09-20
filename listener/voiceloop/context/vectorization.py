from __future__ import annotations

from dataclasses import dataclass

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
        await self.qdrant.upsert_memory(
            source=episode.source,
            source_id=episode.source_id,
            title=episode.title,
            content=episode.summary,
            vectors=dict(zip(vector_names, vectors, strict=True)),
            metadata={
                **episode.metadata,
                "episode_id": episode.episode_id,
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
