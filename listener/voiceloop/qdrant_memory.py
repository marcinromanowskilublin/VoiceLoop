from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from .corpus.local_only import LocalOnlyViolation, require_loopback_url
from .memory import VectorMemoryHit
from .memory_vectorization import MEMORY_VECTOR_NAMES, MEMORY_VECTOR_WEIGHTS
from .settings import Settings

LOGGER = logging.getLogger(__name__)

VECTOR_NAMES = MEMORY_VECTOR_NAMES
VECTOR_WEIGHTS = MEMORY_VECTOR_WEIGHTS
WEIGHTED_RRF_VERSION = "weighted-rrf-v1"
DEFAULT_RRF_K = 60


class QdrantMemoryError(RuntimeError):
    pass


class QdrantUnavailableError(QdrantMemoryError):
    """Qdrant nie odpowiedział — nie mylić z brakiem punktu.

    `False` / `None` znaczy „nie ma". Ten wyjątek znaczy „nie wiem". Worker
    indeksujący musi przy nim pominąć przebieg, a nie zapisać kolejną kopię.
    """


@dataclass(frozen=True)
class QdrantMemoryHit(VectorMemoryHit):
    """Backward-compatible memory hit enriched with per-space retrieval evidence."""

    vector_scores: dict[str, float] = field(default_factory=dict)
    vector_ranks: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, dict[str, float | int]] = field(default_factory=dict)
    fusion_score: float = 0.0
    fusion_method: str = WEIGHTED_RRF_VERSION


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

    def accepts_private_data(self) -> bool:
        if not self.enabled:
            return False
        try:
            require_loopback_url(self.url)
        except LocalOnlyViolation:
            return False
        return True

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
                for field in (
                    "source",
                    "source_id",
                    "memory_type",
                    "person_id",
                    "visit_id",
                    "meeting_id",
                    "content_hash",
                ):
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="expires_at",
                    field_schema=models.PayloadSchemaType.DATETIME,
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
                for field in (
                    "source",
                    "source_id",
                    "memory_type",
                    "person_id",
                    "visit_id",
                    "meeting_id",
                    "content_hash",
                ):
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="expires_at",
                    field_schema=models.PayloadSchemaType.DATETIME,
                )
            self._dimension = dimension
        except QdrantMemoryError:
            raise
        except Exception as exc:
            raise QdrantMemoryError(
                f"Nie udało się przygotować kolekcji Qdrant: {type(exc).__name__}"
            ) from exc

    async def has_memory(
        self,
        *,
        source: str,
        source_id: str,
        content_hash: str | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        point_id = self._point_id(source, source_id)
        try:
            records = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=content_hash is not None,
                with_vectors=False,
            )
        except Exception as exc:
            LOGGER.warning(
                "has_memory: Qdrant niedostępny (%s) — nie mylić z brakiem punktu",
                type(exc).__name__,
            )
            raise QdrantUnavailableError(
                f"Nie udało się sprawdzić pamięci w Qdrant: {type(exc).__name__}"
            ) from exc
        if not records:
            return False
        if content_hash is None:
            return True
        payload = getattr(records[0], "payload", None)
        if not isinstance(payload, dict):
            return False
        stored_hash = payload.get("content_hash")
        if not stored_hash:
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                stored_hash = metadata.get("content_hash")
        return str(stored_hash or "") == content_hash

    async def has_content_hash(
        self,
        *,
        content_hash: str,
        source: str | None = None,
    ) -> bool:
        """Czy ta sama treść już leży w kolekcji, niezależnie od source_id.

        `has_memory` wymaga znanego `source_id`, a kubełki aktywności Screenpipe
        mają go ze znacznika czasu, więc za każdym razem inny. Bez pytania o sam
        odcisk treści identyczny kubełek zawsze wygląda na nowy.

        Przy awarii Qdranta rzuca `QdrantUnavailableError`, a nie `False`.
        `False` znaczy wyłącznie „tej treści nie ma".
        """

        if not self.enabled or not content_hash.strip():
            return False
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="content_hash",
                match=models.MatchValue(value=content_hash),
            )
        ]
        if source:
            conditions.append(
                models.FieldCondition(key="source", match=models.MatchValue(value=source))
            )
        try:
            records, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=conditions),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:
            LOGGER.warning(
                "has_content_hash: Qdrant niedostępny (%s) — nie mylić z brakiem treści",
                type(exc).__name__,
            )
            raise QdrantUnavailableError(
                f"Nie udało się sprawdzić odcisku treści w Qdrant: {type(exc).__name__}"
            ) from exc
        return bool(records)

    async def get_memory_payload(
        self,
        *,
        source: str,
        source_id: str,
    ) -> dict[str, Any] | None:
        """Odczyt payloadu bez wektorów.

        `None` znaczy „punktu nie ma". Niedostępny magazyn to
        `QdrantUnavailableError` — wcześniej obie sytuacje wyglądały tak samo.
        """

        if not self.enabled:
            return None
        try:
            records = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._point_id(source, source_id)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            LOGGER.warning(
                "get_memory_payload: Qdrant niedostępny (%s) — nie mylić z brakiem punktu",
                type(exc).__name__,
            )
            raise QdrantUnavailableError(
                f"Nie udało się odczytać payloadu z Qdrant: {type(exc).__name__}"
            ) from exc
        if not records:
            return None
        payload = getattr(records[0], "payload", None)
        return dict(payload) if isinstance(payload, dict) else None

    async def delete_memory(self, *, source: str, source_id: str) -> None:
        if not self.enabled:
            return
        try:
            if not await self.client.collection_exists(self.collection_name):
                return
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[self._point_id(source, source_id)]
                ),
                wait=True,
            )
        except Exception as exc:
            raise QdrantMemoryError(
                f"Nie udało się usunąć pamięci z Qdrant: {type(exc).__name__}"
            ) from exc

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
        content_hash: str | None = None,
        ttl_seconds: int | float | None = None,
        expires_at: datetime | str | None = None,
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
        now = datetime.now(UTC)
        resolved_content_hash = str(
            content_hash
            or details.get("content_hash")
            or hashlib.sha256(content.encode("utf-8")).hexdigest()
        ).strip()
        details["content_hash"] = resolved_content_hash
        resolved_expiration = self._expiration_time(
            now=now,
            expires_at=expires_at or details.get("expires_at"),
            ttl_seconds=ttl_seconds if ttl_seconds is not None else details.get("ttl_seconds"),
        )
        if resolved_expiration is not None:
            details["expires_at"] = resolved_expiration.isoformat()

        provenance = details.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        payload: dict[str, Any] = {
            "source": source[:80],
            "source_id": source_id[:200],
            "title": title[:1000],
            "content": content[:20000],
            "memory_type": memory_type[:80],
            "content_hash": resolved_content_hash,
            "metadata": details,
            "provenance": provenance,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        if resolved_expiration is not None:
            payload["expires_at"] = resolved_expiration.isoformat()
        for key in (
            "person_id",
            "visit_id",
            "meeting_id",
            "speaker",
            "timestamp",
            "time",
            "confidence",
            "model",
            "schema",
        ):
            value = details.get(key, provenance.get(key))
            if value is not None and str(value).strip():
                payload[key] = value if isinstance(value, int | float) else str(value)[:500]

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
        query_embedding: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        *,
        query_vectors: Mapping[str, Sequence[float]] | None = None,
        limit: int = 8,
        source: str | None = None,
        min_score: float = 0.0,
        vector_names: tuple[str, ...] = VECTOR_NAMES,
        query_weights: Mapping[str, float] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[QdrantMemoryHit]:
        if query_embedding is not None and query_vectors is not None:
            raise QdrantMemoryError(
                "Podaj pojedynczy query_embedding albo mapę query_vectors, nie oba."
            )
        query_input = query_vectors if query_vectors is not None else query_embedding
        if not self.enabled or not query_input:
            return []
        if rrf_k < 1:
            raise QdrantMemoryError("Stała RRF musi być dodatnia.")
        safe_limit = max(1, min(limit, 30))
        selected_names = tuple(
            dict.fromkeys(name for name in vector_names if name in VECTOR_WEIGHTS)
        )
        if isinstance(query_input, Mapping):
            normalized_query_vectors = {
                name: [float(value) for value in query_input[name]]
                for name in selected_names
                if name in query_input and query_input[name]
            }
        else:
            shared_vector = [float(value) for value in query_input]
            normalized_query_vectors = {
                name: shared_vector for name in selected_names if shared_vector
            }
        if not normalized_query_vectors:
            return []

        configured_weights = query_weights or VECTOR_WEIGHTS
        effective_weights = {
            name: float(configured_weights.get(name, VECTOR_WEIGHTS[name]))
            for name in normalized_query_vectors
            if math.isfinite(float(configured_weights.get(name, VECTOR_WEIGHTS[name])))
            and float(configured_weights.get(name, VECTOR_WEIGHTS[name])) > 0
        }
        normalized_query_vectors = {
            name: vector
            for name, vector in normalized_query_vectors.items()
            if name in effective_weights
        }
        if not normalized_query_vectors:
            return []

        must_conditions: list[models.Condition] = []
        if source:
            must_conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source),
                )
            )
        query_filter = models.Filter(
            must=must_conditions or None,
            must_not=[
                models.FieldCondition(
                    key="expires_at",
                    range=models.DatetimeRange(lte=datetime.now(UTC)),
                )
            ],
        )

        async def _query(name: str):
            return name, await self.client.query_points(
                collection_name=self.collection_name,
                query=normalized_query_vectors[name],
                using=name,
                query_filter=query_filter,
                limit=max(safe_limit * 3, 10),
                score_threshold=min_score,
                with_payload=True,
                with_vectors=False,
            )

        requests = [_query(name) for name in normalized_query_vectors]
        responses = await asyncio.gather(*requests, return_exceptions=True)
        failures = [item for item in responses if isinstance(item, Exception)]
        if failures and len(failures) == len(responses):
            # Wszystkie osie milczą — to awaria magazynu, nie pusty wynik wyszukiwania.
            # Pusta lista wyglądałaby jak „nic nie znaleziono" i otwierała zapis kopii.
            raise QdrantUnavailableError(
                "Qdrant nie odpowiedział w żadnej przestrzeni wektorowej: "
                f"{type(failures[0]).__name__}"
            ) from failures[0]
        aggregate: dict[str, dict[str, Any]] = {}
        answering_spaces: set[str] = set()
        contributing_spaces: set[str] = set()
        # Bramka `min_score` była lata ustawiona na wartość, której żaden wynik nie
        # naruszał. Zapisujemy najniższy cosinus, jaki przeszedł, żeby dało się to
        # zobaczyć w danych zamiast zgadywać.
        lowest_kept = float("inf")
        for response in responses:
            if isinstance(response, Exception):
                continue
            name, result = response
            answering_spaces.add(name)
            weight = effective_weights[name]
            for rank, point in enumerate(result.points, start=1):
                similarity = float(point.score)
                lowest_kept = min(lowest_kept, similarity)
                contributing_spaces.add(name)
                key = str(point.id)
                entry = aggregate.setdefault(
                    key,
                    {
                        "rrf_score": 0.0,
                        "payload": point.payload or {},
                        "evidence": {},
                    },
                )
                contribution = weight / (rrf_k + rank)
                entry["rrf_score"] += contribution
                entry["evidence"][name] = {
                    "score": similarity,
                    "rank": rank,
                    "weight": weight,
                    "rrf_contribution": contribution,
                }

        # Mianownik z osi, które faktycznie coś zwróciły. Oś, która odpowiedziała
        # pustą listą, podnosiła wcześniej maksimum, choć żaden dokument nie mógł
        # z niej dostać punktów — wynik 1.0 stawał się nieosiągalny bez powodu.
        # Mianownik zostaje stały dla całego zapytania, więc kolejność wyników się
        # nie zmienia; naprawiamy czytelność liczby, nie ranking.
        maximum_rrf = sum(
            effective_weights[name] / (rrf_k + 1) for name in contributing_spaces
        )
        if maximum_rrf <= 0:
            return []

        hits: list[QdrantMemoryHit] = []
        for entry in aggregate.values():
            evidence = entry["evidence"]
            if not evidence:
                continue
            fusion_score = min(float(entry["rrf_score"]) / maximum_rrf, 1.0)
            evidence_weight = sum(float(item["weight"]) for item in evidence.values())
            weighted_similarity = (
                sum(
                    float(item["score"]) * float(item["weight"])
                    for item in evidence.values()
                )
                / evidence_weight
            )
            payload = entry["payload"] if isinstance(entry["payload"], dict) else {}
            raw_metadata = payload.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            vector_scores = {
                name: float(item["score"]) for name, item in evidence.items()
            }
            vector_ranks = {name: int(item["rank"]) for name, item in evidence.items()}
            # Bez tego niski `fusion_score` jest nieczytelny: nie wiadomo, czy
            # dokument wypadł słabo, czy po prostu nie ma wektora w połowie osi.
            # W kolekcji produkcyjnej `decision` istnieje dla 55% punktów, więc to
            # nie jest przypadek brzegowy.
            stored_spaces = metadata.get("vector_spaces")
            available = (
                [str(name) for name in stored_spaces if str(name) in VECTOR_WEIGHTS]
                if isinstance(stored_spaces, list)
                else []
            )
            reachable = [name for name in available if name in contributing_spaces]
            metadata["retrieval_evidence"] = {
                "method": WEIGHTED_RRF_VERSION,
                "fusion_score": fusion_score,
                "weighted_similarity": weighted_similarity,
                "spaces": evidence,
                "coverage": {
                    "queried": sorted(normalized_query_vectors),
                    "answered": sorted(answering_spaces),
                    "contributed": sorted(contributing_spaces),
                    "document_spaces": available,
                    "matched_of_reachable": (
                        f"{len(evidence)}/{len(reachable)}" if reachable else ""
                    ),
                },
                "gate": {
                    "min_score": min_score,
                    "lowest_kept": None if lowest_kept == float("inf") else lowest_kept,
                    "bound": min_score > 0.0 and lowest_kept < min_score + 0.01,
                },
            }
            created_at_raw = str(payload.get("created_at") or "")
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                created_at = datetime.now(UTC)
            hits.append(
                QdrantMemoryHit(
                    source=str(payload.get("source") or ""),
                    source_id=str(payload.get("source_id") or ""),
                    title=str(payload.get("title") or ""),
                    content=str(payload.get("content") or ""),
                    metadata=metadata,
                    score=fusion_score,
                    created_at=created_at,
                    vector_scores=vector_scores,
                    vector_ranks=vector_ranks,
                    evidence=evidence,
                    fusion_score=fusion_score,
                )
            )
        hits.sort(
            key=lambda hit: (
                hit.fusion_score,
                len(hit.evidence),
                float(hit.metadata["retrieval_evidence"]["weighted_similarity"]),
            ),
            reverse=True,
        )
        return hits[:safe_limit]

    async def prune_expired(
        self,
        *,
        now: datetime | None = None,
        dry_run: bool = True,
    ) -> int:
        """Count or delete expired points using only the indexed payload timestamp."""

        if not self.enabled:
            return 0
        cutoff = self._as_utc(now or datetime.now(UTC))
        expired_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="expires_at",
                    range=models.DatetimeRange(lte=cutoff),
                )
            ]
        )
        try:
            count = await self.client.count(
                collection_name=self.collection_name,
                count_filter=expired_filter,
                exact=True,
            )
            expired_count = int(count.count)
            if dry_run or expired_count <= 0:
                return expired_count
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=expired_filter),
                wait=True,
            )
            return expired_count
        except Exception as exc:
            raise QdrantMemoryError(
                f"Nie udało się usunąć wygasłych pamięci: {type(exc).__name__}"
            ) from exc

    @classmethod
    def _expiration_time(
        cls,
        *,
        now: datetime,
        expires_at: object,
        ttl_seconds: object,
    ) -> datetime | None:
        if expires_at is not None and str(expires_at).strip():
            if isinstance(expires_at, datetime):
                return cls._as_utc(expires_at)
            try:
                return cls._as_utc(datetime.fromisoformat(str(expires_at).replace("Z", "+00:00")))
            except ValueError as exc:
                raise QdrantMemoryError("Nieprawidłowy expires_at pamięci.") from exc
        if ttl_seconds is None or str(ttl_seconds).strip() == "":
            return None
        if isinstance(ttl_seconds, bool):
            raise QdrantMemoryError("TTL pamięci musi być dodatnią liczbą sekund.")
        try:
            seconds = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise QdrantMemoryError("TTL pamięci musi być dodatnią liczbą sekund.") from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise QdrantMemoryError("TTL pamięci musi być dodatnią liczbą sekund.")
        return cls._as_utc(now) + timedelta(seconds=seconds)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _point_id(source: str, source_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"voiceloop:{source}:{source_id}"))
