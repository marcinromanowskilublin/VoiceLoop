from unittest.mock import AsyncMock

import pytest

from voiceloop.settings import Settings
from voiceloop.web_search import WebSearchClient, WebSearchError, WebSearchResult


@pytest.mark.asyncio
async def test_web_search_falls_back_to_duckduckgo_when_brave_unavailable(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="brave",
        web_search_fallback_provider="duckduckgo",
    )
    client = WebSearchClient(settings)
    client._search_duckduckgo = AsyncMock(
        return_value=[
            WebSearchResult(
                title="Duck result",
                url="https://duck.example/result",
                snippet="fallback",
                provider="duckduckgo",
            )
        ]
    )

    results = await client.search("python", limit=1)

    assert results[0].provider == "duckduckgo"
    client._search_duckduckgo.assert_awaited_once()


@pytest.mark.asyncio
async def test_web_search_prefers_primary_provider_when_available(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="brave",
        web_search_fallback_provider="duckduckgo",
        web_search_api_key="test-key",
    )
    client = WebSearchClient(settings)
    client._search_brave = AsyncMock(
        return_value=[
            WebSearchResult(
                title="Brave result",
                url="https://brave.example/result",
                snippet="primary",
                provider="brave",
            )
        ]
    )
    client._search_duckduckgo = AsyncMock(return_value=[])

    results = await client.search("python", limit=1)

    assert results[0].provider == "brave"
    client._search_brave.assert_awaited_once()
    client._search_duckduckgo.assert_not_awaited()


@pytest.mark.asyncio
async def test_web_search_raises_after_all_providers_fail(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="brave",
        web_search_fallback_provider="duckduckgo",
    )
    client = WebSearchClient(settings)
    client._search_brave = AsyncMock(side_effect=WebSearchError("brave down"))
    client._search_duckduckgo = AsyncMock(side_effect=WebSearchError("duck down"))

    with pytest.raises(WebSearchError) as exc_info:
        await client.search("python", limit=1)

    message = str(exc_info.value)
    assert "brave:" in message
    assert "duckduckgo:" in message


@pytest.mark.asyncio
async def test_web_search_returns_empty_when_fallback_has_no_hits(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="brave",
        web_search_fallback_provider="duckduckgo",
    )
    client = WebSearchClient(settings)
    client._search_brave = AsyncMock(side_effect=WebSearchError("brave down"))
    client._search_duckduckgo = AsyncMock(return_value=[])

    results = await client.search("very specific obscure query", limit=1)

    assert results == []


@pytest.mark.asyncio
async def test_web_search_gemini_parses_grounded_links(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="gemini",
        web_search_api_key="gem-key",
        web_search_gemini_model="gemini-2.5-flash",
    )
    client = WebSearchClient(settings)
    client._post_json = AsyncMock(
        return_value={
            "candidates": [
                {
                    "content": {"parts": [{"text": "Szybkie podsumowanie AI."}]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {
                                "web": {
                                    "title": "AI source one",
                                    "uri": "https://example.com/ai-1",
                                }
                            },
                            {
                                "web": {
                                    "title": "AI source two",
                                    "uri": "https://example.com/ai-2",
                                }
                            },
                        ]
                    },
                }
            ]
        }
    )

    results = await client.search("ai news", limit=2)

    assert len(results) == 2
    assert results[0].provider == "gemini"
    assert results[0].url == "https://example.com/ai-1"
    assert "podsumowanie" in results[0].snippet


@pytest.mark.asyncio
async def test_web_search_falls_back_when_gemini_key_missing(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="gemini",
        web_search_fallback_provider="duckduckgo",
    )
    client = WebSearchClient(settings)
    client._search_duckduckgo = AsyncMock(
        return_value=[
            WebSearchResult(
                title="Duck fallback",
                url="https://duck.example/fallback",
                snippet="fallback",
                provider="duckduckgo",
            )
        ]
    )

    results = await client.search("ai news", limit=1)

    assert len(results) == 1
    assert results[0].provider == "duckduckgo"
    client._search_duckduckgo.assert_awaited_once()


@pytest.mark.asyncio
async def test_web_search_uses_venice_with_cloud_key(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="venice",
        cloud_llm_api_key="venice-key",
        cloud_llm_base_url="https://api.venice.ai/api/v1",
    )
    client = WebSearchClient(settings)
    client._post_json = AsyncMock(
        return_value={
            "query": "ai news",
            "results": [
                {
                    "title": "AI News",
                    "url": "https://example.com/ai-news",
                    "content": "Top AI updates.",
                    "date": "2026-08-10",
                }
            ],
        }
    )

    results = await client.search("ai news", limit=1)

    assert len(results) == 1
    assert results[0].provider == "venice"
    assert results[0].url == "https://example.com/ai-news"
    headers = client._post_json.await_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer venice-key"


@pytest.mark.asyncio
async def test_web_search_falls_back_from_venice_quota_to_gemini(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="venice",
        web_search_fallback_provider="duckduckgo",
        cloud_llm_api_key="venice-key",
        gemini_api_key="gemini-key",
    )
    client = WebSearchClient(settings)
    client._search_venice = AsyncMock(side_effect=WebSearchError("HTTP 402"))
    client._search_gemini = AsyncMock(
        return_value=[
            WebSearchResult(
                title="Gemini source",
                url="https://example.com/current",
                snippet="Aktualne dane.",
                provider="gemini",
            )
        ]
    )
    client._search_duckduckgo = AsyncMock(return_value=[])

    results = await client.search("aktualne informacje", limit=1)

    assert results[0].provider == "gemini"
    client._search_gemini.assert_awaited_once()
    client._search_duckduckgo.assert_not_awaited()


@pytest.mark.asyncio
async def test_web_search_falls_back_when_venice_key_missing(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="venice",
        web_search_fallback_provider="duckduckgo",
        cloud_llm_api_key="",
        web_search_api_key="",
        gemini_api_key="",
    )
    client = WebSearchClient(settings)
    client._search_duckduckgo = AsyncMock(
        return_value=[
            WebSearchResult(
                title="Duck fallback",
                url="https://duck.example/fallback",
                snippet="fallback",
                provider="duckduckgo",
            )
        ]
    )

    results = await client.search("ai news", limit=1)

    assert len(results) == 1
    assert results[0].provider == "duckduckgo"
    client._search_duckduckgo.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_endpoint_in_documentation_reports_match_and_access(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    client = WebSearchClient(settings)
    fake_results = [
        WebSearchResult(
            title="API docs",
            url="https://docs.example.com/reference",
            snippet="Reference",
            provider="venice",
        ),
        WebSearchResult(
            title="OpenAPI file",
            url="https://docs.example.com/openapi.json",
            snippet="OpenAPI",
            provider="venice",
        ),
    ]
    client.search = AsyncMock(return_value=fake_results)  # type: ignore[method-assign]
    client._probe_endpoint_source = AsyncMock(
        side_effect=[
            {
                "title": "API docs",
                "url": "https://docs.example.com/reference",
                "provider": "venice",
                "matched": True,
                "match_source": "content",
                "access_channel": "web",
            },
            {
                "title": "OpenAPI file",
                "url": "https://docs.example.com/openapi.json",
                "provider": "venice",
                "matched": False,
                "match_source": "content_no_match",
                "access_channel": "download",
            },
        ]
    )

    report = await client.inspect_endpoint_in_documentation(
        api_name="example",
        endpoint="/v1/items",
        limit=5,
    )

    assert report["endpoint_found"] is True
    assert report["matched_source"]["url"] == "https://docs.example.com/reference"
    assert report["documentation_access"]["availability"] == "web_and_download"
    assert report["documentation_access"]["web_count"] == 1
    assert report["documentation_access"]["download_count"] == 1
