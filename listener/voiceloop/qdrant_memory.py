from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from .memory import VectorMemoryHit
from .settings import Settings

VECTOR_NAMES = (
    "semantic",
    "topic",
    "intent",
    "decision",
    "person_context",
)
VECTOR_WEIGHTS = {
    "semantic": 0.40,
    "topic": 0.20,
    "intent": 0.15,
    "decision": 0.15,
    "person_context": 0.10,
}


class QdrantMemoryError(RuntimeError):
    pass


class QdrantVectorStore:
    """Local Qdrant-backed memory with five independent named-vector spaces."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.enabled = settings.qdrant_enabled
        self.url = settings.qdrant_url.rstrip("/")
        self.collection_name = settings.qdrant_collection
        self.timeout_seconds = max(1.0, settings.qdrant_timeout_seconds)
        self._dimension: int | None = None
        api_key = (
            settings.qdrant_api_key.get_secret_value().strip()
            if settings.qdrant_api_key
            else None
        )
        self.client = client or AsyncQdrantClient(
            url=self.url,
            api_key=api_key or None,
            timeout=self.timeout_seconds,
        )

    async def close(self) -> None:
        await self.client.close()

    async def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączony w konfiguracji"
        try:
            collections = await self.client.get_collections()
        except Exception as exc:
            return False, f"niedostępny: {type(exc).__name__}"
        names = {item.name for item in collections.collections}
        if self.collection_name not in names:
            return True, f"API działa; kolekcja {self.collection_name} oczekuje na pierwszy wektor"
        try:
            count = await self.client.count(
                collection_name=self.collection_name,
                exact=False,
            )
        except Exception as exc:
            return False, f"kolekcja niedostępna: {type(exc).__name__}"
        dimension = f", wymiar {self._dimension}" if self._dimension else ""
        return True, f"{self.collection_name}: {count.count} punktów{dimension}"

    async def ensure_collection(self, dimension: int) -> None:
        if not self.enabled:
            return
        if dimension <= 0:
            raise QdrantMemoryError("Wymiar embeddingu musi być dodatni.")
        if self._dimension is not None:
            if dimension != self._dimension:
                raise QdrantMemoryError(
                    f"Embedding ma wymiar {dimension}, kolekcja oczekuje {self._dimension}."
                )
            return
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                vectors = {
                    name: models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    )
                    for name in VECTOR_NAMES
                }
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors,
                    on_disk_payload=True,
                )
                for field in ("source", "source_id", "memory_type", "person_id", "visit_id"):
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            else:
                info = await self.client.get_collection(self.collection_name)
                configured = info.config.params.vectors
                if not isinstance(configured, dict):
                    raise QdrantMemoryError(
                        "Kolekcja Qdrant nie używa wymaganych named vectors."
                    )
                missing = set(VECTOR_NAMES) - set(configured)
                invalid = {
                    name
                    for name in VECTOR_NAMES
                    if name in configured and configured[name].size != dimension
                }
                if missing or invalid:
                    details = ", ".join(sorted(missing | invalid))
                    raise QdrantMemoryError(
                        f"Nieprawidłowy schemat Qdrant dla wektorów: {details}."
                    )
            self._dimension = dimension
        except QdrantMemoryError:
            raise
        except Exception as exc:
            raise QdrantMemoryError(
                f"Nie udało się przygotować kolekcji Qdrant: {type(exc).__name__}"
            ) from exc

    async def has_memory(self, *, source: str, source_id: str) -> bool:
        if not self.enabled:
            return False
        point_id = self._point_id(source, source_id)
        try:
            records = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            return False
        return bool(records)

    async def upsert_memory(
        self,
        *,
        source: str,
        source_id: str,
        title: str,
        content: str,
        vectors: dict[str, list[float]],
        metadata: dict[str, Any] | None = None,
        memory_type: str = "observation",
    ) -> None:
        if not self.enabled:
            return
        clean_vectors = {
            name: [float(value) for value in vector]
            for name, vector in vectors.items()
            if name in VECTOR_NAMES and vector
        }
        if not clean_vectors:
            return
        dimensions = {len(vector) for vector in clean_vectors.values()}
        if len(dimensions) != 1:
            raise QdrantMemoryError("Named vectors mają różne wymiary.")
        dimension = dimensions.pop()
        await self.ensure_collection(dimension)
        if self._dimension is not None and dimension != self._dimension:
            raise QdrantMemoryError(
                f"Embedding ma wymiar {dimension}, kolekcja oczekuje {self._dimension}."
            )

        details = dict(metadata or {})
        payload: dict[str, Any] = {
            "source": source[:80],
            "source_id": source_id[:200],
            "title": title[:1000],
            "content": content[:20000],
            "memory_type": memory_type[:80],
            "metadata": details,
            "created_at": datetime.now(UTC).isoformat(),
        }
        for key in ("person_id", "visit_id", "meeting_id", "speaker", "timestamp"):
            value = details.get(key)
            if value is not None and str(value).strip():
                payload[key] = str(value)[:500]

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=self._point_id(source, source_id),
                        vector=clean_vectors,
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            raise QdrantMemoryError(
                f"Nie udało się zapisać pamięci w Qdrant: {type(exc).__name__}"
            ) from exc

    async def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 8,
        source: str | None = None,
        min_score: float = 0.15,
        vector_names: tuple[str, ...] = VECTOR_NAMES,
    ) -> list[VectorMemoryHit]:
        if not self.enabled or not query_embedding:
            return []
        safe_limit = max(1, min(limit, 30))
        query_filter = None
        if source:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    )
                ]
            )

        async def _query(name: str):
            return name, await self.client.query_points(
                collection_name=self.collection_name,
                query=[float(value) for value in query_embedding],
                using=name,
                query_filter=query_filter,
                limit=max(safe_limit * 3, 10),
                with_payload=True,
                with_vectors=False,
            )

        requests = [_query(name) for name in vector_names if name in VECTOR_WEIGHTS]
        responses = await asyncio.gather(*requests, return_exceptions=True)
        aggregate: dict[str, dict[str, Any]] = {}
        for response in responses:
            if isinstance(response, Exception):
                continue
            name, result = response
            weight = VECTOR_WEIGHTS[name]
            for point in result.points:
                key = str(point.id)
                entry = aggregate.setdefault(
                    key,
                    {
                        "weighted_score": 0.0,
                        "weight": 0.0,
                        "payload": point.payload or {},
                    },
                )
                entry["weighted_score"] += float(point.score) * weight
                entry["weight"] += weight

        hits: list[VectorMemoryHit] = []
        for entry in aggregate.values():
            if not entry["weight"]:
                continue
            score = entry["weighted_score"] / entry["weight"]
            if score < min_score:
                continue
            payload = entry["payload"] if isinstance(entry["payload"], dict) else {}
            metadata = payload.get("metadata")
            created_at_raw = str(payload.get("created_at") or "")
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                created_at = datetime.now(UTC)
            hits.append(
                VectorMemoryHit(
                    source=str(payload.get("source") or ""),
                    source_id=str(payload.get("source_id") or ""),
                    title=str(payload.get("title") or ""),
                    content=str(payload.get("content") or ""),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    score=score,
                    created_at=created_at,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:safe_limit]

    @staticmethod
    def _point_id(source: str, source_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"voiceloop:{source}:{source_id}"))
