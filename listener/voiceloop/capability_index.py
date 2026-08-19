from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .models import SubtaskEmbeddingV1, SubtaskV1
from .routing.taxonomy import TAXONOMY_VERSION
from .routing.vector_documents import (
    CAPABILITY_DOCUMENT_FORMAT_VERSION,
    COMMAND_QUERY_FORMAT_VERSION,
    SUBTASK_QUERY_FORMAT_VERSION,
    VECTOR_NAMES,
    CapabilityDocuments,
    capability_documents,
    command_documents,
    subtask_documents,
)
from .settings import Settings

CAPABILITY_VECTOR_NAMES = VECTOR_NAMES
CAPABILITY_VECTOR_WEIGHTS = {name: 1.0 for name in CAPABILITY_VECTOR_NAMES}
RANK_FUSION_K = 60
RANK_FUSION_VERSION = "weighted-rrf-v1"
QUERY_EMBEDDING_CACHE_SIZE = 256
CAPABILITY_PAYLOAD_INDEXES = ("action_id", "catalog_hash")


class CapabilityIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapabilityMatch:
    action_id: str
    label: str
    description: str
    score: float
    vector_scores: dict[str, float]
    risk: str
    confirmation_required: bool
    available_in_voiceattack: bool
    vector_ranks: dict[str, int] = field(default_factory=dict)
    coverage: float | None = None

    def __post_init__(self) -> None:
        observed = set(self.vector_scores) & set(CAPABILITY_VECTOR_NAMES)
        derived = len(observed) / len(CAPABILITY_VECTOR_NAMES)
        coverage = derived if self.coverage is None else float(self.coverage)
        object.__setattr__(self, "coverage", max(0.0, min(coverage, derived, 1.0)))

    @property
    def missing_vector_names(self) -> tuple[str, ...]:
        return tuple(name for name in CAPABILITY_VECTOR_NAMES if name not in self.vector_scores)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "description": self.description,
            "score": round(self.score, 6),
            "vector_scores": {name: round(score, 6) for name, score in self.vector_scores.items()},
            "vector_ranks": dict(self.vector_ranks),
            "coverage": round(float(self.coverage or 0.0), 6),
            "missing_vector_names": list(self.missing_vector_names),
            "risk": self.risk,
            "confirmation_required": self.confirmation_required,
            "available_in_voiceattack": self.available_in_voiceattack,
        }


@dataclass(frozen=True)
class CapabilitySearchResult:
    query_documents: CapabilityDocuments
    matches: list[CapabilityMatch]
    catalog_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_documents": self.query_documents.as_dict(),
            "matches": [match.as_dict() for match in self.matches],
            "catalog_hash": self.catalog_hash,
        }


@dataclass(frozen=True)
class SubtaskCapabilitySearch:
    subtask: SubtaskV1
    embedding: SubtaskEmbeddingV1
    result: CapabilitySearchResult
    query_format_version: str = SUBTASK_QUERY_FORMAT_VERSION


class CapabilityIndex:
    taxonomy_version = TAXONOMY_VERSION
    document_format_version = CAPABILITY_DOCUMENT_FORMAT_VERSION
    query_format_version = SUBTASK_QUERY_FORMAT_VERSION
    vector_fusion = RANK_FUSION_VERSION
    vector_weights = CAPABILITY_VECTOR_WEIGHTS
    rank_fusion_k = RANK_FUSION_K

    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: OpenAICompatibleEmbeddingClient,
        definitions: list[dict[str, Any]],
        client: AsyncQdrantClient | None = None,
        query_cache_size: int = QUERY_EMBEDDING_CACHE_SIZE,
    ) -> None:
        self.enabled = (
            settings.qdrant_enabled
            and settings.capability_embeddings_enabled
            and embeddings.enabled
        )
        self.embeddings = embeddings
        self.collection_name = settings.qdrant_capability_collection
        self.default_limit = max(1, min(settings.capability_match_limit, 20))
        self.min_score = max(-1.0, min(float(settings.capability_match_min_score), 1.0))
        self.definitions = sorted(
            (
                dict(item)
                for item in definitions
                if str(item.get("id") or "").strip() and item.get("id") != "speak_text"
            ),
            key=lambda item: str(item["id"]),
        )
        self._definitions_by_id = {str(item["id"]): item for item in self.definitions}
        self.catalog_hash = _catalog_hash(self.definitions)
        self._dimension: int | None = None
        self._ready = False
        self._last_error: str | None = None
        self._query_cache_size = max(1, min(int(query_cache_size), 4096))
        self._query_embedding_cache: OrderedDict[
            tuple[str, int, str, str, str],
            tuple[float, ...],
        ] = OrderedDict()
        api_key = (
            settings.qdrant_api_key.get_secret_value().strip() if settings.qdrant_api_key else None
        )
        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url.rstrip("/"),
            api_key=api_key or None,
            timeout=max(1.0, settings.qdrant_timeout_seconds),
        )

    @property
    def ready(self) -> bool:
        return self._ready

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            await close()

    def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączony w konfiguracji"
        if self._ready:
            return (
                True,
                f"{self.collection_name}: {len(self.definitions)} możliwości; "
                f"katalog {self.catalog_hash}",
            )
        return False, self._last_error or "indeks nie został przygotowany"

    async def start(self) -> None:
        if not self.enabled:
            return
        if not self.definitions:
            self._last_error = "katalog możliwości jest pusty"
            raise CapabilityIndexError(self._last_error)
        try:
            model_name = await self._embedding_model_name()
            existing_dimension = await self._prepare_existing_collection()
            if existing_dimension is not None and await self._catalog_payload_is_current(
                model_name=model_name,
                dimension=existing_dimension,
            ):
                # These records prove that a compatible catalog upsert completed
                # previously; stale generations can now be removed safely.
                await self._delete_stale_catalog_points()
                self._dimension = existing_dimension
                self._ready = True
                self._last_error = None
                return

            documents_by_id = {
                str(item["id"]): capability_documents(item) for item in self.definitions
            }
            texts: list[str] = []
            for item in self.definitions:
                texts.extend(documents_by_id[str(item["id"])].as_dict().values())
            embedded = await self.embeddings.embed_documents(texts)
            expected_count = len(self.definitions) * len(CAPABILITY_VECTOR_NAMES)
            if len(embedded) != expected_count:
                raise CapabilityIndexError(
                    f"Embedding zwrócił {len(embedded)} z {expected_count} wektorów."
                )
            if any(not vector for vector in embedded):
                raise CapabilityIndexError("Embedding zwrócił pusty wektor możliwości.")
            dimensions = {len(vector) for vector in embedded if vector}
            if len(dimensions) != 1:
                raise CapabilityIndexError("Embeddingi możliwości mają różne wymiary.")
            dimension = dimensions.pop()
            if existing_dimension is None:
                await self._ensure_collection(dimension)
            elif dimension != existing_dimension:
                raise CapabilityIndexError(
                    "Wymiar embeddingu nie pasuje do istniejącego indeksu możliwości."
                )

            points: list[models.PointStruct] = []
            offset = 0
            for definition in self.definitions:
                action_id = str(definition["id"])
                vectors = {
                    name: embedded[offset + index]
                    for index, name in enumerate(CAPABILITY_VECTOR_NAMES)
                }
                offset += len(CAPABILITY_VECTOR_NAMES)
                points.append(
                    models.PointStruct(
                        id=_point_id(action_id),
                        vector=vectors,
                        payload={
                            "action_id": action_id,
                            "label": str(definition.get("label") or ""),
                            "description": str(definition.get("description") or ""),
                            "risk": str(definition.get("risk") or "low"),
                            "confirmation_required": bool(definition.get("confirmation_required")),
                            "available_in_voiceattack": bool(
                                definition.get("available_in_voiceattack")
                            ),
                            "catalog_hash": self.catalog_hash,
                            "embedding_model": model_name,
                            "embedding_dimension": dimension,
                            "document_format": CAPABILITY_DOCUMENT_FORMAT_VERSION,
                        },
                    )
                )
            await self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
            )
            await self._delete_stale_catalog_points()
            self._dimension = dimension
            self._query_embedding_cache.clear()
            self._ready = True
            self._last_error = None
        except (CapabilityIndexError, EmbeddingUnavailableError) as exc:
            self._ready = False
            self._last_error = str(exc)
            raise CapabilityIndexError(str(exc)) from exc
        except Exception as exc:
            self._ready = False
            self._last_error = f"Nie udało się przygotować indeksu: {type(exc).__name__}"
            raise CapabilityIndexError(self._last_error) from exc

    async def search(
        self,
        text: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> CapabilitySearchResult:
        documents = command_documents(text)
        if not documents.semantic:
            return CapabilitySearchResult(documents, [], self.catalog_hash)
        self._require_ready()
        query_vectors, _ = await self._embed_query_documents(
            [documents],
            format_version=COMMAND_QUERY_FORMAT_VERSION,
        )
        return await self._search_with_vectors(
            documents,
            query_vectors,
            limit=limit,
            min_score=min_score,
        )

    async def search_subtasks(
        self,
        subtasks: list[SubtaskV1] | tuple[SubtaskV1, ...],
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[SubtaskCapabilitySearch]:
        if not subtasks:
            return []
        self._require_ready()
        documents = [subtask_documents(subtask) for subtask in subtasks]
        query_vectors, model_name = await self._embed_query_documents(
            documents,
            format_version=SUBTASK_QUERY_FORMAT_VERSION,
        )

        searches = []
        embeddings: list[SubtaskEmbeddingV1] = []
        for index, (subtask, document) in enumerate(zip(subtasks, documents, strict=True)):
            offset = index * len(CAPABILITY_VECTOR_NAMES)
            vectors = query_vectors[offset : offset + len(CAPABILITY_VECTOR_NAMES)]
            dimension = len(vectors[0])
            embedding = SubtaskEmbeddingV1(
                subtask_id=subtask.subtask_id,
                semantic=tuple(vectors[0]),
                intent=tuple(vectors[1]),
                target_context=tuple(vectors[2]),
                embedding_model=model_name,
                dimension=dimension,
                format_version=SUBTASK_QUERY_FORMAT_VERSION,
                normalized_text_sha256=SubtaskEmbeddingV1.text_hash(subtask.normalized_text),
            )
            embeddings.append(embedding)
            searches.append(
                self._search_with_vectors(
                    document,
                    vectors,
                    limit=limit,
                    min_score=min_score,
                )
            )

        results = await asyncio.gather(*searches)
        return [
            SubtaskCapabilitySearch(
                subtask=subtask,
                embedding=embedding,
                result=result,
            )
            for subtask, embedding, result in zip(
                subtasks,
                embeddings,
                results,
                strict=True,
            )
        ]

    async def _embed_query_documents(
        self,
        documents: list[CapabilityDocuments],
        *,
        format_version: str,
    ) -> tuple[list[list[float]], str]:
        if self._dimension is None:
            raise CapabilityIndexError("Indeks możliwości nie ma ustalonego wymiaru.")
        model_name = await self._embedding_model_name()
        entries: list[tuple[tuple[str, int, str, str, str], str]] = []
        for document in documents:
            values = document.as_dict()
            for vector_name in CAPABILITY_VECTOR_NAMES:
                text = values[vector_name]
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                key = (
                    model_name,
                    self._dimension,
                    format_version,
                    vector_name,
                    text_hash,
                )
                entries.append((key, text))

        resolved: dict[tuple[str, int, str, str, str], tuple[float, ...]] = {}
        missing: OrderedDict[tuple[str, int, str, str, str], str] = OrderedDict()
        for key, text in entries:
            cached = self._query_embedding_cache.pop(key, None)
            if cached is not None:
                self._query_embedding_cache[key] = cached
                resolved[key] = cached
            elif key not in missing:
                missing[key] = text

        if missing:
            embedded = await self.embeddings.embed_queries(list(missing.values()))
            self._validate_query_vectors(embedded, expected_count=len(missing))
            for key, vector in zip(missing, embedded, strict=True):
                cached_vector = tuple(float(value) for value in vector)
                resolved[key] = cached_vector
                self._query_embedding_cache[key] = cached_vector
                while len(self._query_embedding_cache) > self._query_cache_size:
                    self._query_embedding_cache.popitem(last=False)

        vectors = [list(resolved[key]) for key, _ in entries]
        self._validate_query_vectors(vectors, expected_count=len(entries))
        return vectors, model_name

    def _require_ready(self) -> None:
        if not self.enabled:
            raise CapabilityIndexError("Indeks możliwości jest wyłączony.")
        if not self._ready:
            raise CapabilityIndexError(self._last_error or "Indeks możliwości nie jest gotowy.")

    def _validate_query_vectors(
        self,
        vectors: list[list[float]],
        *,
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count or any(not vector for vector in vectors):
            raise CapabilityIndexError(
                f"Nie udało się utworzyć {expected_count} wektorów zapytania."
            )
        if self._dimension is not None and any(
            len(vector) != self._dimension for vector in vectors
        ):
            raise CapabilityIndexError("Wymiar zapytania nie pasuje do indeksu możliwości.")

    async def _embedding_model_name(self) -> str:
        resolve_model = getattr(self.embeddings, "resolve_model", None)
        if callable(resolve_model):
            model = await resolve_model()
            if str(model).strip():
                return str(model).strip()
        for attribute in ("configured_model", "_resolved_model"):
            model = getattr(self.embeddings, attribute, None)
            if str(model or "").strip():
                return str(model).strip()
        return "unknown-local-embedding-model"

    async def _search_with_vectors(
        self,
        documents: CapabilityDocuments,
        query_vectors: list[list[float]],
        *,
        limit: int | None,
        min_score: float | None,
    ) -> CapabilitySearchResult:
        safe_limit = max(1, min(limit or self.default_limit, 20))
        threshold = self.min_score if min_score is None else max(-1.0, min(min_score, 1.0))
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="catalog_hash",
                    match=models.MatchValue(value=self.catalog_hash),
                )
            ]
        )
        candidate_limit = min(
            len(self.definitions),
            max(safe_limit * 4, 20),
        )

        async def _query(name: str, vector: list[float]):
            result = await self.client.query_points(
                collection_name=self.collection_name,
                query=[float(value) for value in vector],
                using=name,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=True,
                with_vectors=False,
            )
            return name, result

        responses = await asyncio.gather(
            *(
                _query(name, query_vectors[index])
                for index, name in enumerate(CAPABILITY_VECTOR_NAMES)
            ),
            return_exceptions=True,
        )
        failed_vectors = [
            name
            for name, response in zip(
                CAPABILITY_VECTOR_NAMES,
                responses,
                strict=True,
            )
            if isinstance(response, Exception)
        ]
        if failed_vectors:
            raise CapabilityIndexError(
                "Qdrant nie zwrócił wyników ze wszystkich przestrzeni: "
                f"{', '.join(failed_vectors)}."
            )
        aggregate: dict[str, dict[str, Any]] = {}
        for response in responses:
            if isinstance(response, Exception):
                continue
            name, result = response
            weight = CAPABILITY_VECTOR_WEIGHTS[name]
            ranked_points = sorted(
                result.points,
                key=lambda point: (
                    -float(point.score),
                    str(
                        ((point.payload or {}).get("action_id") or point.id)
                        if isinstance(point.payload, dict)
                        else point.id
                    ),
                ),
            )
            for rank, point in enumerate(ranked_points, start=1):
                payload = point.payload if isinstance(point.payload, dict) else {}
                action_id = str(payload.get("action_id") or "")
                if action_id not in self._definitions_by_id:
                    continue
                entry = aggregate.setdefault(
                    action_id,
                    {
                        "rank_fusion_score": 0.0,
                        "vector_scores": {},
                        "vector_ranks": {},
                        "payload": payload,
                    },
                )
                if name in entry["vector_ranks"]:
                    continue
                score = float(point.score)
                entry["rank_fusion_score"] += weight / (RANK_FUSION_K + rank)
                entry["vector_scores"][name] = score
                entry["vector_ranks"][name] = rank

        max_rank_fusion_score = sum(
            weight / (RANK_FUSION_K + 1)
            for weight in CAPABILITY_VECTOR_WEIGHTS.values()
        )
        matches: list[CapabilityMatch] = []
        for action_id, entry in aggregate.items():
            score = float(entry["rank_fusion_score"]) / max_rank_fusion_score
            if score < threshold:
                continue
            payload = entry["payload"]
            definition = self._definitions_by_id[action_id]
            matches.append(
                CapabilityMatch(
                    action_id=action_id,
                    label=str(
                        payload.get("label")
                        or definition.get("label")
                        or definition.get("description")
                        or action_id
                    ),
                    description=str(
                        payload.get("description") or definition.get("description") or ""
                    ),
                    score=score,
                    vector_scores={
                        name: float(entry["vector_scores"][name])
                        for name in CAPABILITY_VECTOR_NAMES
                        if name in entry["vector_scores"]
                    },
                    risk=str(payload.get("risk") or definition.get("risk") or "low"),
                    confirmation_required=bool(
                        payload.get("confirmation_required")
                        or definition.get("confirmation_required")
                    ),
                    available_in_voiceattack=bool(
                        payload.get("available_in_voiceattack")
                        or definition.get("available_in_voiceattack")
                    ),
                    vector_ranks={
                        name: int(entry["vector_ranks"][name])
                        for name in CAPABILITY_VECTOR_NAMES
                        if name in entry["vector_ranks"]
                    },
                    coverage=len(entry["vector_scores"]) / len(CAPABILITY_VECTOR_NAMES),
                )
            )
        matches.sort(key=lambda match: (-match.score, match.action_id))
        return CapabilitySearchResult(
            query_documents=documents,
            matches=matches[:safe_limit],
            catalog_hash=self.catalog_hash,
        )

    async def _ensure_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise CapabilityIndexError("Wymiar embeddingu musi być dodatni.")
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    name: models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    )
                    for name in CAPABILITY_VECTOR_NAMES
                },
                on_disk_payload=True,
            )
            for field_name in CAPABILITY_PAYLOAD_INDEXES:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            return
        info = await self.client.get_collection(self.collection_name)
        self._collection_dimension(info, expected_dimension=dimension)
        await self._ensure_payload_indexes(info)

    async def _prepare_existing_collection(self) -> int | None:
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            return None
        info = await self.client.get_collection(self.collection_name)
        dimension = self._collection_dimension(info)
        await self._ensure_payload_indexes(info)
        return dimension

    def _collection_dimension(
        self,
        info: Any,
        *,
        expected_dimension: int | None = None,
    ) -> int:
        configured = info.config.params.vectors
        if not isinstance(configured, dict):
            raise CapabilityIndexError("Kolekcja możliwości nie używa wymaganych named vectors.")
        missing = set(CAPABILITY_VECTOR_NAMES) - set(configured)
        invalid = {
            name
            for name in CAPABILITY_VECTOR_NAMES
            if name in configured
            and (
                int(configured[name].size) <= 0
                or (
                    expected_dimension is not None
                    and configured[name].size != expected_dimension
                )
                or configured[name].distance != models.Distance.COSINE
            )
        }
        dimensions = {
            int(configured[name].size)
            for name in CAPABILITY_VECTOR_NAMES
            if name in configured and name not in invalid
        }
        if len(dimensions) != 1:
            invalid.update(name for name in CAPABILITY_VECTOR_NAMES if name in configured)
        if missing or invalid:
            details = ", ".join(sorted(missing | invalid))
            raise CapabilityIndexError(f"Nieprawidłowy schemat indeksu możliwości: {details}.")
        return dimensions.pop()

    async def _ensure_payload_indexes(self, info: Any) -> None:
        payload_schema = getattr(info, "payload_schema", None)
        indexed_fields = set(payload_schema) if isinstance(payload_schema, dict) else set()
        for field_name in CAPABILITY_PAYLOAD_INDEXES:
            if field_name in indexed_fields:
                index_info = payload_schema[field_name]
                data_type = getattr(index_info, "data_type", None)
                if (
                    data_type is not None
                    and data_type != models.PayloadSchemaType.KEYWORD
                ):
                    raise CapabilityIndexError(
                        f"Indeks payload {field_name} nie jest typu keyword."
                    )
                continue
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def _catalog_payload_is_current(
        self,
        *,
        model_name: str,
        dimension: int,
    ) -> bool:
        retrieve = getattr(self.client, "retrieve", None)
        if not callable(retrieve):
            return False
        records = await retrieve(
            collection_name=self.collection_name,
            ids=[_point_id(str(definition["id"])) for definition in self.definitions],
            with_payload=True,
            with_vectors=False,
        )
        if len(records) != len(self.definitions):
            return False
        expected_action_ids = set(self._definitions_by_id)
        matched_action_ids: set[str] = set()
        for record in records:
            payload = record.payload if isinstance(record.payload, dict) else {}
            action_id = str(payload.get("action_id") or "")
            if action_id not in expected_action_ids:
                return False
            if (
                payload.get("catalog_hash") != self.catalog_hash
                or payload.get("embedding_model") != model_name
                or payload.get("embedding_dimension") != dimension
                or payload.get("document_format") != CAPABILITY_DOCUMENT_FORMAT_VERSION
            ):
                return False
            matched_action_ids.add(action_id)
        return matched_action_ids == expected_action_ids

    async def _delete_stale_catalog_points(self) -> None:
        delete = getattr(self.client, "delete", None)
        if not callable(delete):
            return
        stale_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="catalog_hash",
                    match=models.MatchValue(value=self.catalog_hash),
                )
            ]
        )
        await delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=stale_filter),
            wait=True,
        )


def _catalog_hash(definitions: list[dict[str, Any]]) -> str:
    relevant = [
        {
            "id": item.get("id"),
            "label": item.get("label"),
            "description": item.get("description"),
            "args_schema": item.get("args_schema"),
            "risk": item.get("risk"),
            "confirmation_required": item.get("confirmation_required"),
            "execution_layer": item.get("execution_layer"),
            "routing_examples": item.get("routing_examples"),
            "available_in_voiceattack": item.get("available_in_voiceattack"),
        }
        for item in definitions
    ]
    encoded = json.dumps(
        {
            "document_format": CAPABILITY_DOCUMENT_FORMAT_VERSION,
            "definitions": relevant,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _point_id(action_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"voiceloop:capability:{action_id}"))
