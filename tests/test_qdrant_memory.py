from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voiceloop.qdrant_memory import (
    VECTOR_NAMES,
    WEIGHTED_RRF_VERSION,
    QdrantUnavailableError,
    QdrantVectorStore,
)
from voiceloop.settings import Settings


@pytest.mark.asyncio
async def test_qdrant_creates_schema_and_upserts_partial_named_vectors(tmp_path) -> None:
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

    vectors = {
        "semantic": [1.0, 0.0, 0.0],
        "decision": [0.0, 1.0, 0.0],
    }
    await store.upsert_memory(
        source="screenpipe_behavior",
        source_id="activity:1",
        title="Aktywność",
        content="Praca nad VoiceLoop.",
        vectors=vectors,
        metadata={"person_id": "person-1", "content_hash": "hash-v1"},
        ttl_seconds=3600,
    )

    config = client.create_collection.await_args.kwargs["vectors_config"]
    assert tuple(config) == VECTOR_NAMES
    assert {item.size for item in config.values()} == {3}
    point = client.upsert.await_args.kwargs["points"][0]
    assert set(point.vector) == {"semantic", "decision"}
    assert point.payload["person_id"] == "person-1"
    assert point.payload["content_hash"] == "hash-v1"
    assert point.payload["metadata"]["content_hash"] == "hash-v1"
    assert point.payload["expires_at"] == point.payload["metadata"]["expires_at"]
    assert point.payload["embedding_prefix_policy"] == "nomic-search-prefixes-v1"
    assert point.payload["embedding_input_kind"] == "document"
    assert point.payload["metadata"]["embedding_prefix"] == "search_document:"


@pytest.mark.asyncio
async def test_qdrant_deletes_manual_memory_by_stable_point_id(tmp_path) -> None:
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        delete=AsyncMock(),
    )
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    await store.delete_memory(source="manual_memory", source_id="42")

    selector = client.delete.await_args.kwargs["points_selector"]
    assert selector.points == [store._point_id("manual_memory", "42")]
    assert client.delete.await_args.kwargs["wait"] is True


def test_qdrant_private_data_requires_loopback_endpoint(tmp_path) -> None:
    local = QdrantVectorStore(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            qdrant_url="http://127.0.0.1:6333",
        ),
        client=SimpleNamespace(),  # type: ignore[arg-type]
    )
    remote = QdrantVectorStore(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            qdrant_url="https://vectors.example.com",
        ),
        client=SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert local.accepts_private_data() is True
    assert remote.accepts_private_data() is False


@pytest.mark.asyncio
async def test_qdrant_ensures_payload_indexes_on_existing_collection(tmp_path) -> None:
    vectors = {name: SimpleNamespace(size=3) for name in VECTOR_NAMES}
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        get_collection=AsyncMock(
            return_value=SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=vectors),
                )
            )
        ),
        create_payload_index=AsyncMock(),
    )
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    await store.ensure_collection(3)

    indexed_fields = {
        call.kwargs["field_name"]
        for call in client.create_payload_index.await_args_list
    }
    assert indexed_fields == {
        "source",
        "source_id",
        "memory_type",
        "person_id",
        "visit_id",
        "meeting_id",
        "content_hash",
        "embedding_prefix_policy",
        "embedding_input_kind",
        "expires_at",
    }


@pytest.mark.asyncio
async def test_qdrant_search_uses_distinct_vectors_and_rrf_evidence(tmp_path) -> None:
    queries: dict[str, list[float]] = {}

    def payload(source_id: str) -> dict[str, object]:
        return {
            "source": "screenpipe_behavior",
            "source_id": source_id,
            "title": "VoiceLoop",
            "content": "Praca nad pamięcią.",
            "metadata": {"app": "Cursor"},
            "created_at": "2026-08-10T00:00:00+00:00",
        }

    async def query_points(**kwargs):
        name = kwargs["using"]
        queries[name] = kwargs["query"]
        if name == "semantic":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id="point-a", score=0.95, payload=payload("activity:a")),
                    SimpleNamespace(id="point-b", score=0.90, payload=payload("activity:b")),
                ]
            )
        return SimpleNamespace(
            points=[
                SimpleNamespace(id="point-b", score=0.80, payload=payload("activity:b")),
            ]
        )

    client = SimpleNamespace(query_points=query_points)
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    hits = await store.search(
        query_vectors={
            "semantic": [1.0, 0.0, 0.0],
            "topic": [0.0, 1.0, 0.0],
        },
        vector_names=("semantic", "topic"),
        limit=2,
    )

    assert queries == {
        "semantic": [1.0, 0.0, 0.0],
        "topic": [0.0, 1.0, 0.0],
    }
    assert [hit.source_id for hit in hits] == ["activity:b", "activity:a"]
    assert hits[0].metadata["app"] == "Cursor"
    assert hits[0].fusion_method == WEIGHTED_RRF_VERSION
    assert hits[0].vector_scores == {"semantic": 0.9, "topic": 0.8}
    assert hits[0].vector_ranks == {"semantic": 2, "topic": 1}
    assert hits[0].evidence["topic"]["rrf_contribution"] > 0
    assert "topic" not in hits[1].evidence
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["retrieval_evidence"]["spaces"] == hits[0].evidence


@pytest.mark.asyncio
async def test_qdrant_search_keeps_single_vector_compatibility(tmp_path) -> None:
    queries: dict[str, list[float]] = {}

    async def query_points(**kwargs):
        queries[kwargs["using"]] = kwargs["query"]
        return SimpleNamespace(points=[])

    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(query_points=query_points),  # type: ignore[arg-type]
    )

    await store.search(
        [1.0, 2.0, 3.0],
        vector_names=("semantic", "decision"),
    )

    assert queries == {
        "semantic": [1.0, 2.0, 3.0],
        "decision": [1.0, 2.0, 3.0],
    }


@pytest.mark.asyncio
async def test_qdrant_has_memory_compares_content_hash(tmp_path) -> None:
    client = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                SimpleNamespace(payload={"content_hash": "current-hash"}),
            ]
        )
    )
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    assert await store.has_memory(source="meeting", source_id="1") is True
    assert (
        await store.has_memory(
            source="meeting",
            source_id="1",
            content_hash="current-hash",
        )
        is True
    )
    assert (
        await store.has_memory(
            source="meeting",
            source_id="1",
            content_hash="changed-hash",
        )
        is False
    )


@pytest.mark.asyncio
async def test_qdrant_finds_content_hash_without_knowing_source_id(tmp_path) -> None:
    scroll = AsyncMock(return_value=([SimpleNamespace(id="point-a")], None))
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(scroll=scroll),  # type: ignore[arg-type]
    )

    assert (
        await store.has_content_hash(
            content_hash="abc",
            source="screenpipe_behavior",
        )
        is True
    )
    conditions = scroll.await_args.kwargs["scroll_filter"].must
    assert [item.key for item in conditions] == ["content_hash", "source"]
    assert conditions[0].match.value == "abc"
    assert scroll.await_args.kwargs["with_vectors"] is False

    scroll.return_value = ([], None)
    assert await store.has_content_hash(content_hash="abc") is False
    assert await store.has_content_hash(content_hash="   ") is False


@pytest.mark.asyncio
async def test_qdrant_has_memory_raises_when_store_is_down(tmp_path) -> None:
    """Awaria magazynu nie może wyglądać jak brak punktu."""

    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("down"))),
    )

    with pytest.raises(QdrantUnavailableError):
        await store.has_memory(source="meeting", source_id="1")


@pytest.mark.asyncio
async def test_qdrant_has_content_hash_raises_when_store_is_down(tmp_path) -> None:
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(scroll=AsyncMock(side_effect=RuntimeError("down"))),
    )

    with pytest.raises(QdrantUnavailableError):
        await store.has_content_hash(content_hash="abc")


@pytest.mark.asyncio
async def test_qdrant_get_memory_payload_raises_when_store_is_down(tmp_path) -> None:
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("down"))),
    )

    with pytest.raises(QdrantUnavailableError):
        await store.get_memory_payload(source="meeting", source_id="1")


@pytest.mark.asyncio
async def test_qdrant_search_raises_when_every_axis_fails(tmp_path) -> None:
    """Pusta lista przy całkowitej awarii wyglądałaby jak „nic nie znaleziono"."""

    async def boom(**_kwargs):
        raise RuntimeError("down")

    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(query_points=boom),
    )

    with pytest.raises(QdrantUnavailableError):
        await store.search([1.0, 0.0, 0.0], vector_names=("semantic",))


@pytest.mark.asyncio
async def test_qdrant_empty_axis_does_not_make_top_score_unreachable(tmp_path) -> None:
    """Oś, która odpowiedziała pustą listą, nie może zabierać punktów innym.

    Wcześniej mianownik brał wszystkie osie, które nie rzuciły wyjątkiem, więc
    dokument pierwszy w jedynej działającej osi dostawał ułamek zamiast 1.0 —
    a przy 55% pokrycia osi `decision` w kolekcji produkcyjnej to reguła,
    nie wyjątek.
    """

    def payload() -> dict[str, object]:
        return {
            "source": "screenpipe_behavior",
            "source_id": "activity:a",
            "title": "VoiceLoop",
            "content": "Praca nad pamięcią.",
            "metadata": {"vector_spaces": ["semantic"]},
            "created_at": "2026-08-10T00:00:00+00:00",
        }

    async def query_points(**kwargs):
        if kwargs["using"] == "semantic":
            return SimpleNamespace(
                points=[SimpleNamespace(id="point-a", score=0.91, payload=payload())]
            )
        return SimpleNamespace(points=[])

    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(query_points=query_points),  # type: ignore[arg-type]
    )

    hits = await store.search(
        query_vectors={name: [1.0, 0.0, 0.0] for name in VECTOR_NAMES},
        vector_names=VECTOR_NAMES,
    )

    assert len(hits) == 1
    assert hits[0].fusion_score == pytest.approx(1.0)
    coverage = hits[0].metadata["retrieval_evidence"]["coverage"]
    assert coverage["answered"] == sorted(VECTOR_NAMES)
    assert coverage["contributed"] == ["semantic"]
    assert coverage["document_spaces"] == ["semantic"]
    assert coverage["matched_of_reachable"] == "1/1"


@pytest.mark.asyncio
async def test_qdrant_reports_whether_the_score_gate_bound_anything(tmp_path) -> None:
    """Bramka, której nie widać w danych, obiecuje jakość, której nie dostarcza."""

    async def query_points(**kwargs):
        assert kwargs["score_threshold"] == 0.0
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-a",
                    score=0.44,
                    payload={
                        "source": "screenpipe_behavior",
                        "source_id": "activity:a",
                        "title": "VoiceLoop",
                        "content": "Praca.",
                        "metadata": {},
                        "created_at": "2026-08-10T00:00:00+00:00",
                    },
                )
            ]
        )

    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(query_points=query_points),  # type: ignore[arg-type]
    )

    hits = await store.search(query_vectors={"semantic": [1.0, 0.0]}, vector_names=("semantic",))
    gate = hits[0].metadata["retrieval_evidence"]["gate"]

    assert gate["min_score"] == 0.0
    assert gate["lowest_kept"] == pytest.approx(0.44)
    assert gate["bound"] is False


def test_default_score_gate_is_off_because_it_never_bound(tmp_path) -> None:
    # 0.15 leżało poniżej najniższego cosinusa, jaki model produkuje dla tych
    # danych (0.441), więc nie odrzucało niczego. Zero mówi to wprost.
    assert Settings(voiceloop_data_dir=str(tmp_path)).vector_memory_min_score == 0.0


@pytest.mark.asyncio
async def test_qdrant_coverage_separates_weak_match_from_missing_vector(tmp_path) -> None:
    def payload(spaces: list[str]) -> dict[str, object]:
        return {
            "source": "screenpipe_behavior",
            "source_id": "activity:" + "-".join(spaces),
            "title": "VoiceLoop",
            "content": "Praca nad pamięcią.",
            "metadata": {"vector_spaces": spaces},
            "created_at": "2026-08-10T00:00:00+00:00",
        }

    async def query_points(**kwargs):
        if kwargs["using"] == "semantic":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id="full", score=0.9, payload=payload(["semantic", "topic"])),
                    SimpleNamespace(id="thin", score=0.8, payload=payload(["semantic"])),
                ]
            )
        return SimpleNamespace(
            points=[SimpleNamespace(id="full", score=0.7, payload=payload(["semantic", "topic"]))]
        )

    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=SimpleNamespace(query_points=query_points),  # type: ignore[arg-type]
    )

    hits = await store.search(
        query_vectors={"semantic": [1.0, 0.0], "topic": [0.0, 1.0]},
        vector_names=("semantic", "topic"),
    )
    by_id = {hit.source_id: hit for hit in hits}

    thin = by_id["activity:semantic"].metadata["retrieval_evidence"]["coverage"]
    full = by_id["activity:semantic-topic"].metadata["retrieval_evidence"]["coverage"]
    # Dokument bez wektora `topic` wykorzystał wszystko, co miał; dokument z obu
    # osiami też. Sam `fusion_score` tego nie odróżnia, więc pokrycie musi.
    assert thin["matched_of_reachable"] == "1/1"
    assert full["matched_of_reachable"] == "2/2"
    assert thin["document_spaces"] == ["semantic"]


@pytest.mark.asyncio
async def test_qdrant_prune_expired_is_dry_run_by_default(tmp_path) -> None:
    client = SimpleNamespace(
        count=AsyncMock(return_value=SimpleNamespace(count=2)),
        delete=AsyncMock(),
    )
    store = QdrantVectorStore(
        Settings(voiceloop_data_dir=str(tmp_path)),
        client=client,  # type: ignore[arg-type]
    )

    assert await store.prune_expired() == 2
    client.delete.assert_not_awaited()

    assert await store.prune_expired(dry_run=False) == 2
    client.delete.assert_awaited_once()
    selector = client.delete.await_args.kwargs["points_selector"]
    assert selector.filter.must[0].key == "expires_at"
