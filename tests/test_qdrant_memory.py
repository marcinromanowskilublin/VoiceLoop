from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voiceloop.qdrant_memory import VECTOR_NAMES, QdrantVectorStore
from voiceloop.settings import Settings


@pytest.mark.asyncio
async def test_qdrant_creates_five_named_vectors_and_upserts(tmp_path) -> None:
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(return_value=True),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
    )
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    vectors = {name: [1.0, 0.0, 0.0] for name in VECTOR_NAMES}
    await store.upsert_memory(
        source="screenpipe_behavior",
        source_id="activity:1",
        title="Aktywność",
        content="Praca nad VoiceLoop.",
        vectors=vectors,
        metadata={"person_id": "person-1"},
    )

    config = client.create_collection.await_args.kwargs["vectors_config"]
    assert tuple(config) == VECTOR_NAMES
    assert {item.size for item in config.values()} == {3}
    point = client.upsert.await_args.kwargs["points"][0]
    assert set(point.vector) == set(VECTOR_NAMES)
    assert point.payload["person_id"] == "person-1"


@pytest.mark.asyncio
async def test_qdrant_search_fuses_named_vector_scores(tmp_path) -> None:
    payload = {
        "source": "screenpipe_behavior",
        "source_id": "activity:1",
        "title": "VoiceLoop",
        "content": "Praca nad pamięcią.",
        "metadata": {"app": "Cursor"},
        "created_at": "2026-08-10T00:00:00+00:00",
    }

    async def query_points(**kwargs):
        score = 0.9 if kwargs["using"] == "semantic" else 0.7
        return SimpleNamespace(
            points=[SimpleNamespace(id="point-1", score=score, payload=payload)]
        )

    client = SimpleNamespace(query_points=query_points)
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    hits = await store.search(
        [1.0, 0.0, 0.0],
        vector_names=("semantic", "topic"),
        limit=1,
    )

    assert len(hits) == 1
    assert hits[0].source_id == "activity:1"
    assert hits[0].metadata["app"] == "Cursor"
    assert hits[0].score == pytest.approx((0.9 * 0.4 + 0.7 * 0.2) / 0.6)
