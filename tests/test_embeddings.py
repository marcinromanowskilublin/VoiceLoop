import pytest

from voiceloop.embeddings import (
    EMBEDDING_DOCUMENT_PREFIX,
    EMBEDDING_PREFIX_POLICY_VERSION,
    EMBEDDING_QUERY_PREFIX,
    OpenAICompatibleEmbeddingClient,
    embedding_prefix_metadata,
    with_embedding_prefix,
)


def test_embedding_prefix_is_applied_once() -> None:
    assert with_embedding_prefix("VoiceLoop memory", EMBEDDING_QUERY_PREFIX) == (
        "search_query: VoiceLoop memory"
    )
    assert with_embedding_prefix("search_query: VoiceLoop memory", EMBEDDING_QUERY_PREFIX) == (
        "search_query: VoiceLoop memory"
    )
    assert with_embedding_prefix("SEARCH_QUERY: VoiceLoop memory", EMBEDDING_QUERY_PREFIX) == (
        "SEARCH_QUERY: VoiceLoop memory"
    )


def test_embedding_prefix_metadata_names_the_policy() -> None:
    assert embedding_prefix_metadata("document") == {
        "embedding_prefix_policy": EMBEDDING_PREFIX_POLICY_VERSION,
        "embedding_input_kind": "document",
        "embedding_prefix": EMBEDDING_DOCUMENT_PREFIX,
    }
    assert embedding_prefix_metadata("query")["embedding_prefix"] == EMBEDDING_QUERY_PREFIX


@pytest.mark.asyncio
async def test_embedding_client_routes_queries_and_documents_through_policy() -> None:
    class RecordingClient(OpenAICompatibleEmbeddingClient):
        seen: list[str]

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.seen = list(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

    client = RecordingClient(
        base_url="http://127.0.0.1:1234/v1",
        api_key=None,
        model="nomic",
        timeout_seconds=1.0,
    )

    await client.embed_queries(["Co robilem?", "search_query: juz gotowe"])
    assert client.seen == ["search_query: Co robilem?", "search_query: juz gotowe"]

    await client.embed_documents(["Notatka", "search_document: juz gotowe"])
    assert client.seen == ["search_document: Notatka", "search_document: juz gotowe"]
