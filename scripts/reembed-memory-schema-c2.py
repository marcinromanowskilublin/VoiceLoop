"""C2: przeliczenie wektorów w voiceloop_memory na schemat memory-documents-v2.

Snapshot musiał istnieć wcześniej. Napływ Screenpipe ma być wyłączony.
Skrypt tylko odczytuje payload, buduje dokumenty produkcyjnym
`memory_vector_documents`, wektoryzuje przez `embed_documents` i zapisuje
te same punkty z poprawionymi wektorami oraz uczciwą etykietą schematu.

Użycie:
  python scripts/reembed-memory-schema-c2.py --dry-run
  python scripts/reembed-memory-schema-c2.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "listener")]

from voiceloop.embeddings import (
    EmbeddingUnavailableError,
    OpenAICompatibleEmbeddingClient,
)
from voiceloop.memory_vectorization import (
    MEMORY_DOCUMENT_SCHEMA_VERSION,
    MEMORY_VECTOR_NAMES,
    memory_vector_documents,
)
from voiceloop.qdrant_memory import QdrantVectorStore
from voiceloop.settings import get_settings
from voiceloop.threshold_measure import (
    EXACT_RECONSTRUCTION,
    classify_reconstruction,
    measure_document_neighbourhood,
)

PROFILE = "reembedded_documents_v2"
PROVENANCE_MARK = "c2-reembed-2026-08-26"
BATCH_POINTS = 16


def _build_documents(payload: dict[str, Any]) -> dict[str, str]:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    observations = [
        str(item).strip()
        for item in (metadata.get("observations") or [])
        if str(item).strip()
    ]
    people = [
        str(item).strip()
        for item in (metadata.get("people") or [])
        if str(item).strip()
    ]
    # content w behavior_digest to już summary digesta; w legacy — surowa treść.
    # Topic/intent/decision nie leżą w payloadzie starszych punktów — budujemy
    # tylko to, co da się uczciwie odtworzyć. Reszty osi nie udajemy.
    return memory_vector_documents(
        summary=str(payload.get("content") or ""),
        person_context="; ".join(people),
        observations=observations,
        redact=False,
    )


def _updated_payload(
    payload: dict[str, Any],
    *,
    vector_names: list[str],
    embedding_model: str | None,
) -> dict[str, Any]:
    updated = dict(payload)
    metadata = dict(updated.get("metadata") or {})
    provenance = dict(metadata.get("provenance") or updated.get("provenance") or {})
    now = datetime.now(UTC).isoformat()
    metadata["schema"] = MEMORY_DOCUMENT_SCHEMA_VERSION
    metadata["vector_profile"] = PROFILE
    metadata["vector_spaces"] = list(vector_names)
    if embedding_model:
        metadata["embedding_model"] = embedding_model
        provenance["embedding_model"] = embedding_model
    provenance["schema"] = MEMORY_DOCUMENT_SCHEMA_VERSION
    provenance["reembed"] = PROVENANCE_MARK
    provenance["reembedded_at"] = now
    metadata["provenance"] = provenance
    updated["metadata"] = metadata
    updated["schema"] = MEMORY_DOCUMENT_SCHEMA_VERSION
    updated["updated_at"] = now
    # created_at zostaje z oryginału — upsert przez ten skrypt nie udaje nowej pamięci.
    return updated


async def _scroll_all(store: QdrantVectorStore) -> list[Any]:
    collected: list[Any] = []
    offset = None
    while True:
        batch, offset = await store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        collected.extend(batch)
        if offset is None:
            break
    return collected


async def _embed_batch(
    embeddings: OpenAICompatibleEmbeddingClient,
    documents_by_point: list[tuple[Any, dict[str, str]]],
) -> list[dict[str, list[float]]]:
    names_order = list(MEMORY_VECTOR_NAMES)
    flat_texts: list[str] = []
    owners: list[tuple[int, str]] = []
    for index, (_, documents) in enumerate(documents_by_point):
        for name in names_order:
            text = documents.get(name)
            if text:
                flat_texts.append(text)
                owners.append((index, name))
    if not flat_texts:
        return [{} for _ in documents_by_point]
    vectors = await embeddings.embed_documents(flat_texts)
    if len(vectors) != len(flat_texts):
        raise EmbeddingUnavailableError(
            f"embedding count mismatch: {len(vectors)} vs {len(flat_texts)}"
        )
    result: list[dict[str, list[float]]] = [{} for _ in documents_by_point]
    for (point_index, name), vector in zip(owners, vectors, strict=True):
        result[point_index][name] = vector
    return result


async def run(*, apply: bool, limit: int | None, sample_verify: int) -> int:
    settings = get_settings()
    store = QdrantVectorStore(settings)
    embeddings = OpenAICompatibleEmbeddingClient(
        base_url=(settings.local_embeddings_base_url or settings.lm_studio_base_url),
        api_key=settings.local_embeddings_api_key or settings.lm_studio_api_key,
        model=settings.local_embeddings_model,
        timeout_seconds=settings.local_embeddings_timeout_seconds,
        enabled=settings.local_embeddings_enabled,
    )
    if not store.enabled or not embeddings.enabled:
        print("Qdrant albo embeddings wyłączone — stop.")
        return 1

    points = await _scroll_all(store)
    if limit is not None:
        points = points[:limit]
    print(f"Punktów do przerobu: {len(points)}")
    print(f"Tryb: {'APPLY (zapis)' if apply else 'DRY-RUN (bez zapisu)'}")

    rebuilt = 0
    skipped = 0
    written = 0
    verify_docs: list[tuple[str, str]] = []

    for start in range(0, len(points), BATCH_POINTS):
        batch = points[start : start + BATCH_POINTS]
        prepared: list[tuple[Any, dict[str, str]]] = []
        for point in batch:
            payload = point.payload if isinstance(point.payload, dict) else {}
            documents = _build_documents(payload)
            if not documents:
                skipped += 1
                continue
            prepared.append((point, documents))
            rebuilt += 1

        if not prepared:
            continue

        named_vectors = await _embed_batch(embeddings, prepared)
        model_name = await embeddings.resolve_model()

        upserts = []
        for (point, documents), vectors in zip(prepared, named_vectors, strict=True):
            if not vectors:
                skipped += 1
                continue
            payload = point.payload if isinstance(point.payload, dict) else {}
            new_payload = _updated_payload(
                payload,
                vector_names=list(vectors),
                embedding_model=model_name,
            )
            if "semantic" in documents and len(verify_docs) < sample_verify:
                verify_docs.append((str(point.id), documents["semantic"]))
            if apply:
                upserts.append(
                    {
                        "id": point.id,
                        "vector": vectors,
                        "payload": new_payload,
                    }
                )

        if apply and upserts:
            from qdrant_client import models

            await store.client.upsert(
                collection_name=store.collection_name,
                points=[
                    models.PointStruct(
                        id=item["id"],
                        vector=item["vector"],
                        payload=item["payload"],
                    )
                    for item in upserts
                ],
                wait=True,
            )
            written += len(upserts)
            print(f"  zapisano {written}/{len(points)} ...")

    print(f"Przebudowane dokumenty: {rebuilt}")
    print(f"Pominięte (pusty dokument): {skipped}")
    print(f"Zapisane punkty: {written}")

    if apply and verify_docs:
        print(f"Weryfikacja odtworzenia na {len(verify_docs)} punktach...")
        identical, _ = await measure_document_neighbourhood(
            client=store.client,
            collection=store.collection_name,
            embeddings=embeddings,
            documents=verify_docs,
            depth=5,
        )
        verdict = classify_reconstruction(identical=identical)
        exact = sum(1 for score in identical if score >= EXACT_RECONSTRUCTION)
        print(f"  exact {exact}/{len(identical)} (≥{EXACT_RECONSTRUCTION})")
        print(f"  [{verdict.status}] {verdict.message}")

    await store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-verify", type=int, default=40)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return asyncio.run(
        run(apply=args.apply, limit=args.limit, sample_verify=args.sample_verify)
    )


if __name__ == "__main__":
    raise SystemExit(main())
