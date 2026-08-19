from __future__ import annotations

import hashlib
import json
import logging
import re

from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .memory import MemoryStore
from .memory_vectorization import (
    MEMORY_DOCUMENT_SCHEMA_VERSION,
    memory_vector_documents,
)
from .models import MemoryCreate, MemoryItem
from .qdrant_memory import QdrantMemoryError, QdrantVectorStore

LOGGER = logging.getLogger("voiceloop.manual_memory")
MANUAL_MEMORY_SOURCE = "manual_memory"


class ManualMemoryService:
    """Keeps manual SQLite memories searchable in a strictly local vector store."""

    def __init__(
        self,
        *,
        memory: MemoryStore,
        embeddings: OpenAICompatibleEmbeddingClient,
        qdrant: QdrantVectorStore,
    ) -> None:
        self.memory = memory
        self.embeddings = embeddings
        self.qdrant = qdrant

    async def create(self, item: MemoryCreate) -> MemoryItem:
        stored = await self.memory.create_memory(item)
        try:
            await self.index(stored)
        except (EmbeddingUnavailableError, QdrantMemoryError) as exc:
            LOGGER.warning(
                "Manual memory %s saved in SQLite but not indexed: %s",
                stored.id,
                exc,
            )
        return stored

    async def delete(self, memory_id: int) -> bool:
        existing = await self.memory.list_memories(limit=1, memory_id=memory_id)
        if not existing:
            return False
        if self.private_backend_available():
            await self.qdrant.delete_memory(
                source=MANUAL_MEMORY_SOURCE,
                source_id=str(memory_id),
            )
        return await self.memory.delete_memory(memory_id)

    async def sync(self, *, limit: int = 500) -> int:
        if not self.private_backend_available():
            return 0
        indexed = 0
        for item in reversed(await self.memory.list_memories(limit=limit)):
            if await self.index(item):
                indexed += 1
        return indexed

    async def index(self, item: MemoryItem) -> bool:
        if not self.private_backend_available():
            return False
        content_hash = self._content_hash(item)
        if await self.qdrant.has_memory(
            source=MANUAL_MEMORY_SOURCE,
            source_id=str(item.id),
            content_hash=content_hash,
        ):
            return False
        documents = self._documents(item)
        vector_names = tuple(documents)
        vectors = await self.embeddings.embed_documents(
            [documents[name] for name in vector_names]
        )
        if len(vectors) != len(vector_names):
            raise EmbeddingUnavailableError(
                "embedding count mismatch for manual memory documents"
            )
        embedding_model = (
            self.embeddings._resolved_model
            or self.embeddings.configured_model
            or ""
        )
        metadata = {
            "manual_memory_id": item.id,
            "kind": item.kind,
            "sensitivity": item.sensitivity,
            "authoring_source": item.source,
            "content_hash": content_hash,
            "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
            "vector_spaces": list(vector_names),
            "vector_profile": "manual_dynamic_subset_v1",
            "processing_location": "local",
            "network_scope": "loopback",
            "embedding_model": embedding_model,
            "provenance": {
                "source": MANUAL_MEMORY_SOURCE,
                "source_id": str(item.id),
                "time": item.updated_at.isoformat(),
                "confidence": 1.0,
                "model": "explicit-user-memory",
                "embedding_model": embedding_model,
                "schema": MEMORY_DOCUMENT_SCHEMA_VERSION,
            },
        }
        await self.qdrant.upsert_memory(
            source=MANUAL_MEMORY_SOURCE,
            source_id=str(item.id),
            title=f"Pamięć {item.kind} #{item.id}",
            content=item.content,
            vectors=dict(zip(vector_names, vectors, strict=True)),
            metadata=metadata,
            memory_type="manual",
            content_hash=content_hash,
        )
        return True

    def private_backend_available(self) -> bool:
        return (
            self.embeddings.enabled
            and self.qdrant.enabled
            and self.embeddings.accepts_private_text()
            and self.qdrant.accepts_private_data()
        )

    @staticmethod
    def _content_hash(item: MemoryItem) -> str:
        payload = {
            "id": item.id,
            "kind": item.kind,
            "content": item.content,
            "sensitivity": item.sensitivity,
            "source": item.source,
            "updated_at": item.updated_at.isoformat(),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _documents(cls, item: MemoryItem) -> dict[str, str]:
        kind = item.kind.casefold()
        content = item.content.strip()
        intent = ""
        decision = ""
        person_context = ""
        if re.search(r"prefer|cel|goal|plan|intenc|zamiar|chc[eę]", kind + " " + content):
            intent = content
        if re.search(r"decyz|decision|ustalen|wyb[oó]r|postanow", kind + " " + content):
            decision = content
        if re.search(
            r"osob|person|kontakt|relac|profil|zdrow|medic|"
            r"\b(?:mam|jestem|m[oó]j|moja|moje|lubi[eę])\b",
            kind + " " + content,
        ):
            person_context = content
        return memory_vector_documents(
            summary=content,
            topic=f"Kategoria pamięci: {item.kind}. Treść: {content}",
            intent=intent,
            decision=decision,
            person_context=person_context,
            redact=False,
        )
