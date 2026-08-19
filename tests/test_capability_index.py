from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import models as qdrant_models

from voiceloop.capability_index import (
    CAPABILITY_VECTOR_NAMES,
    CapabilityIndex,
    CapabilityIndexError,
    capability_documents,
    command_documents,
)
from voiceloop.models import SubtaskV1
from voiceloop.routing.vector_documents import (
    CAPABILITY_DOCUMENT_FORMAT_VERSION,
    SUBTASK_QUERY_FORMAT_VERSION,
    subtask_documents,
)
from voiceloop.settings import Settings


class FakeEmbeddings:
    enabled = True
    configured_model = "fake-embedding"

    def __init__(self) -> None:
        self.document_texts: list[str] = []
        self.query_texts: list[str] = []
        self.document_batches: list[list[str]] = []
        self.query_batches: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts = list(texts)
        self.document_batches.append(list(texts))
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_texts = list(texts)
        self.query_batches.append(list(texts))
        return [[1.0, float(index), 0.0] for index, _ in enumerate(texts)]


def definitions() -> list[dict]:
    return [
        {
            "id": "close_window_under_cursor",
            "label": "zamykać okno pod kursorem",
            "description": "Zamyka okno wskazane kursorem.",
            "args_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "risk": "medium",
            "confirmation_required": True,
            "execution_layer": 1,
            "available_in_voiceattack": True,
        },
        {
            "id": "open_browser",
            "label": "otwierać przeglądarkę",
            "description": "Otwiera domyślną przeglądarkę.",
            "args_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "risk": "low",
            "confirmation_required": False,
            "execution_layer": 1,
            "available_in_voiceattack": True,
        },
    ]


def collection_info(*, payload_schema: dict | None = None):
    configured = {
        name: SimpleNamespace(size=3, distance=qdrant_models.Distance.COSINE)
        for name in CAPABILITY_VECTOR_NAMES
    }
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=configured)),
        payload_schema=payload_schema or {},
    )


def test_command_is_represented_by_three_distinct_documents() -> None:
    documents = command_documents("Zamknij okno UI Vision, otwórz Chrome i kartę z YouTubem.")

    assert documents.semantic.startswith("Zamknij okno")
    assert "zamknąć" in documents.intent
    assert "otworzyć" in documents.intent
    assert "UI.Vision" in documents.target_context
    assert "YouTube" in documents.target_context
    assert len(set(documents.as_dict().values())) == 3


def test_subtask_v1_query_documents_use_structured_fields() -> None:
    subtask = SubtaskV1(
        text="wyszukaj pogodę dla Krakowa",
        source_text="wyszukaj pogodę dla Krakowa",
        normalized_text="wyszukaj pogodę dla Krakowa",
        start_char=0,
        end_char=29,
        order=0,
        operation="search",
        target="web",
        raw_arguments={"tail": "pogodę dla Krakowa"},
        segmentation_confidence=0.95,
    )

    documents = subtask_documents(subtask)

    assert documents.semantic == subtask.text
    assert "search" in documents.intent
    assert subtask.text in documents.intent
    assert "web" in documents.target_context
    assert "tail: pogodę dla Krakowa" in documents.target_context
    assert subtask.text in documents.target_context
    assert len(set(documents.as_dict().values())) == 3


def test_capability_documents_separate_operation_and_target() -> None:
    documents = capability_documents(definitions()[0])

    assert "Zamyka okno" in documents.semantic
    assert "zamknąć" in documents.intent
    assert "okno" in documents.target_context
    assert "VoiceAttack" in documents.target_context
    assert len(set(documents.as_dict().values())) == 3


@pytest.mark.asyncio
async def test_index_creates_three_named_vectors_and_upserts_catalog(tmp_path) -> None:
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
        close=AsyncMock(),
    )
    embeddings = FakeEmbeddings()
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )

    await index.start()

    config = client.create_collection.await_args.kwargs["vectors_config"]
    assert tuple(config) == CAPABILITY_VECTOR_NAMES
    assert {item.size for item in config.values()} == {3}
    points = client.upsert.await_args.kwargs["points"]
    assert len(points) == 2
    assert all(set(point.vector) == set(CAPABILITY_VECTOR_NAMES) for point in points)
    assert all(point.payload["catalog_hash"] == index.catalog_hash for point in points)
    assert all(point.payload["embedding_model"] == "fake-embedding" for point in points)
    assert all(point.payload["embedding_dimension"] == 3 for point in points)
    assert all(
        point.payload["document_format"] == CAPABILITY_DOCUMENT_FORMAT_VERSION
        for point in points
    )
    assert len(embeddings.document_texts) == 6
    assert index.health()[0] is True


@pytest.mark.asyncio
async def test_search_uses_weighted_rank_fusion_and_preserves_cosines(tmp_path) -> None:
    payloads = {
        "open_browser": {
            "action_id": "open_browser",
            "label": "otwierać przeglądarkę",
            "description": "Otwiera domyślną przeglądarkę.",
            "risk": "low",
            "confirmation_required": False,
            "available_in_voiceattack": True,
        },
        "close_window_under_cursor": {
            "action_id": "close_window_under_cursor",
            "label": "zamykać okno pod kursorem",
            "description": "Zamyka okno wskazane kursorem.",
            "risk": "medium",
            "confirmation_required": True,
            "available_in_voiceattack": True,
        },
    }
    score_map = {
        "semantic": {"open_browser": 0.90, "close_window_under_cursor": 0.60},
        "intent": {"open_browser": 0.90, "close_window_under_cursor": 0.70},
        "target_context": {
            "open_browser": 0.75,
            "close_window_under_cursor": 0.90,
        },
    }

    async def query_points(**kwargs):
        vector_name = kwargs["using"]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=action_id,
                    score=score,
                    payload={
                        **payloads[action_id],
                        "catalog_hash": index.catalog_hash,
                    },
                )
                for action_id, score in score_map[vector_name].items()
            ]
        )

    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
        query_points=query_points,
        close=AsyncMock(),
    )
    embeddings = FakeEmbeddings()
    index = CapabilityIndex(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            capability_match_min_score=0.99,
        ),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    await index.start()

    result = await index.search("Otwórz Chrome.", limit=5)

    assert [match.action_id for match in result.matches] == ["open_browser"]
    expected_rank_score = ((1 / 61) + (1 / 61) + (1 / 62)) / (3 / 61)
    assert result.matches[0].score == pytest.approx(expected_rank_score)
    assert result.matches[0].vector_scores == {
        "semantic": pytest.approx(0.90),
        "intent": pytest.approx(0.90),
        "target_context": pytest.approx(0.75),
    }
    assert result.matches[0].vector_ranks == {
        "semantic": 1,
        "intent": 1,
        "target_context": 2,
    }
    assert result.matches[0].coverage == pytest.approx(1.0)
    assert set(result.matches[0].vector_scores) == set(CAPABILITY_VECTOR_NAMES)
    assert len(embeddings.query_texts) == 3
    assert len(set(embeddings.query_texts)) == 3


@pytest.mark.asyncio
async def test_search_union_keeps_missing_space_explicit(tmp_path) -> None:
    payloads = {
        action_id: {
            "action_id": action_id,
            "label": action_id,
            "description": action_id,
        }
        for action_id in ("open_browser", "close_window_under_cursor")
    }
    score_map = {
        "semantic": {"open_browser": 0.91},
        "intent": {"open_browser": 0.87},
        "target_context": {"close_window_under_cursor": 0.99},
    }

    async def query_points(**kwargs):
        vector_name = kwargs["using"]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=action_id,
                    score=score,
                    payload={
                        **payloads[action_id],
                        "catalog_hash": index.catalog_hash,
                    },
                )
                for action_id, score in score_map[vector_name].items()
            ]
        )

    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
        query_points=query_points,
        close=AsyncMock(),
    )
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    await index.start()

    result = await index.search("otwórz Chrome", min_score=-1.0)

    assert [match.action_id for match in result.matches] == [
        "open_browser",
        "close_window_under_cursor",
    ]
    browser = result.matches[0]
    assert browser.vector_scores == {"semantic": 0.91, "intent": 0.87}
    assert browser.vector_ranks == {"semantic": 1, "intent": 1}
    assert browser.coverage == pytest.approx(2 / 3)
    assert browser.missing_vector_names == ("target_context",)
    assert browser.score == pytest.approx(2 / 3)
    assert "target_context" not in browser.as_dict()["vector_scores"]


@pytest.mark.asyncio
async def test_search_fails_closed_when_one_vector_space_is_unavailable(
    tmp_path,
) -> None:
    async def query_points(**kwargs):
        if kwargs["using"] == "intent":
            raise RuntimeError("intent vector unavailable")
        return SimpleNamespace(points=[])

    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
        query_points=query_points,
        close=AsyncMock(),
    )
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    await index.start()

    with pytest.raises(CapabilityIndexError, match="wszystkich przestrzeni"):
        await index.search("otwórz Chrome")


@pytest.mark.asyncio
async def test_search_reports_unavailable_index_and_honors_qdrant_flag(tmp_path) -> None:
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(close=AsyncMock())
    unavailable = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(CapabilityIndexError, match="nie jest gotowy"):
        await unavailable.search("otwórz przeglądarkę")

    disabled = CapabilityIndex(
        Settings(
            voiceloop_data_dir=str(tmp_path),
            qdrant_enabled=False,
        ),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    with pytest.raises(CapabilityIndexError, match="wyłączony"):
        await disabled.search("otwórz przeglądarkę")
    assert disabled.enabled is False


@pytest.mark.asyncio
async def test_subtasks_are_embedded_separately_in_one_batch(tmp_path) -> None:
    async def query_points(**_kwargs):
        return SimpleNamespace(points=[])

    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
        query_points=query_points,
        close=AsyncMock(),
    )
    embeddings = FakeEmbeddings()
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    await index.start()
    subtasks = [
        SubtaskV1(
            text="otwórz Chrome",
            source_text="otwórz Chrome",
            normalized_text="otwórz Chrome",
            start_char=0,
            end_char=13,
            order=0,
            operation="open",
            target="browser",
            raw_arguments={"tail": "Chrome"},
            segmentation_confidence=0.95,
        ),
        SubtaskV1(
            text="zamknij UI Vision",
            source_text="zamknij UI Vision",
            normalized_text="zamknij UI Vision",
            start_char=16,
            end_char=33,
            order=1,
            operation="close",
            target="UI Vision",
            segmentation_confidence=0.95,
        ),
    ]

    results = await index.search_subtasks(subtasks)

    assert len(results) == 2
    assert len(embeddings.query_texts) == 6
    assert results[0].embedding.subtask_id == subtasks[0].subtask_id
    assert results[1].embedding.subtask_id == subtasks[1].subtask_id
    assert results[0].embedding.dimension == 3
    assert results[0].query_format_version == SUBTASK_QUERY_FORMAT_VERSION
    assert "open" in results[0].result.query_documents.intent
    assert "otwórz Chrome" in results[0].result.query_documents.intent
    assert "browser" in results[0].result.query_documents.target_context
    assert "tail: Chrome" in results[0].result.query_documents.target_context
    assert (
        results[0].embedding.normalized_text_sha256 != results[1].embedding.normalized_text_sha256
    )


@pytest.mark.asyncio
async def test_query_embedding_cache_uses_hashed_versioned_keys(tmp_path) -> None:
    async def query_points(**_kwargs):
        return SimpleNamespace(points=[])

    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=False),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(),
        query_points=query_points,
        close=AsyncMock(),
    )
    embeddings = FakeEmbeddings()
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    await index.start()
    subtask = SubtaskV1(
        text="zapamiętaj prywatny sekret",
        source_text="zapamiętaj prywatny sekret",
        normalized_text="zapamiętaj prywatny sekret",
        start_char=0,
        end_char=27,
        order=0,
        operation="remember",
        target=None,
        raw_arguments={"tail": "prywatny sekret"},
        segmentation_confidence=0.95,
    )

    await index.search_subtasks([subtask])
    await index.search_subtasks([subtask])

    assert len(embeddings.query_batches) == 1
    assert len(index._query_embedding_cache) == 3
    for key in index._query_embedding_cache:
        model_name, dimension, format_version, vector_name, text_hash = key
        assert model_name == "fake-embedding"
        assert dimension == 3
        assert format_version == SUBTASK_QUERY_FORMAT_VERSION
        assert vector_name in CAPABILITY_VECTOR_NAMES
        assert len(text_hash) == 64
        assert "sekret" not in repr(key)


@pytest.mark.asyncio
async def test_existing_collection_rejects_non_cosine_distance(tmp_path) -> None:
    configured = {name: SimpleNamespace(size=3, distance="Dot") for name in CAPABILITY_VECTOR_NAMES}
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        get_collection=AsyncMock(
            return_value=SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors=configured))
            )
        ),
    )
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(CapabilityIndexError, match="Nieprawidłowy schemat"):
        await index._ensure_collection(3)


@pytest.mark.asyncio
async def test_start_skips_unchanged_versioned_catalog(tmp_path) -> None:
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        get_collection=AsyncMock(
            return_value=collection_info(
                payload_schema={
                    "action_id": SimpleNamespace(),
                    "catalog_hash": SimpleNamespace(),
                }
            )
        ),
        create_payload_index=AsyncMock(),
        retrieve=AsyncMock(),
        upsert=AsyncMock(),
        delete=AsyncMock(),
        close=AsyncMock(),
    )
    embeddings = FakeEmbeddings()
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=embeddings,  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )
    client.retrieve.return_value = [
        SimpleNamespace(
            payload={
                "action_id": definition["id"],
                "catalog_hash": index.catalog_hash,
                "embedding_model": "fake-embedding",
                "embedding_dimension": 3,
                "document_format": CAPABILITY_DOCUMENT_FORMAT_VERSION,
            }
        )
        for definition in definitions()
    ]

    await index.start()

    assert index.ready is True
    assert index._dimension == 3
    assert embeddings.document_batches == []
    client.upsert.assert_not_awaited()
    client.create_payload_index.assert_not_awaited()
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_deletes_stale_hash_only_after_successful_upsert(tmp_path) -> None:
    events: list[str] = []

    async def upsert(**_kwargs):
        events.append("upsert")

    async def delete(**kwargs):
        events.append("delete")
        selector = kwargs["points_selector"]
        condition = selector.filter.must_not[0]
        assert condition.key == "catalog_hash"
        assert condition.match.value == index.catalog_hash

    client = SimpleNamespace(
        collection_exists=AsyncMock(side_effect=[False, False]),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=upsert,
        delete=delete,
        close=AsyncMock(),
    )
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )

    await index.start()

    assert events == ["upsert", "delete"]


@pytest.mark.asyncio
async def test_start_does_not_delete_stale_hash_when_upsert_fails(tmp_path) -> None:
    client = SimpleNamespace(
        collection_exists=AsyncMock(side_effect=[False, False]),
        create_collection=AsyncMock(),
        create_payload_index=AsyncMock(),
        upsert=AsyncMock(side_effect=RuntimeError("upsert failed")),
        delete=AsyncMock(),
        close=AsyncMock(),
    )
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(CapabilityIndexError, match="Nie udało się przygotować"):
        await index.start()

    client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_collection_gets_missing_payload_indexes(tmp_path) -> None:
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        get_collection=AsyncMock(
            return_value=collection_info(
                payload_schema={"action_id": SimpleNamespace()}
            )
        ),
        create_payload_index=AsyncMock(),
    )
    index = CapabilityIndex(
        Settings(voiceloop_data_dir=str(tmp_path)),
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        definitions=definitions(),
        client=client,  # type: ignore[arg-type]
    )

    await index._ensure_collection(3)

    client.create_payload_index.assert_awaited_once()
    assert client.create_payload_index.await_args.kwargs["field_name"] == "catalog_hash"
