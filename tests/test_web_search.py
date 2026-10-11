import ipaddress
from unittest.mock import AsyncMock, Mock, patch

import pytest

from voiceloop.settings import Settings
from voiceloop.web_search import (
    MAX_INSPECTION_BYTES,
    WebSearchClient,
    WebSearchError,
    WebSearchResult,
    _InspectionResponse,
    _InspectionTarget,
    _is_public_unicast,
    _pinned_request_sync,
)


@pytest.mark.asyncio
async def test_web_search_health_is_configuration_only(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        web_search_provider="gemini",
        gemini_api_key="gem-key",
    )
    client = WebSearchClient(settings)
    client.search = AsyncMock(side_effect=AssertionError("health must not search"))

    ok, detail = await client.health()

    assert ok is True
    assert "skonfigurowane" in detail
    client.search.assert_not_awaited()


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
    request = client._post_json.await_args.kwargs
    assert request["params"] == {}
    assert request["headers"]["x-goog-api-key"] == "gem-key"


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:secret@example.com/docs",
        "http://127.0.0.1/private",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/private",
    ],
)
async def test_document_inspection_rejects_unsafe_urls(tmp_path, url) -> None:
    client = WebSearchClient(Settings(voiceloop_data_dir=str(tmp_path)))

    with pytest.raises(WebSearchError):
        await client._validate_inspection_url(url)


@pytest.mark.asyncio
async def test_document_inspection_validates_every_redirect(tmp_path) -> None:
    client = WebSearchClient(Settings(voiceloop_data_dir=str(tmp_path)))
    client._resolve_public_addresses = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            frozenset({"93.184.216.34"}),
            WebSearchError("blocked redirect"),
        ]
    )
    client._pinned_request = AsyncMock(  # type: ignore[method-assign]
        return_value=_InspectionResponse(
            status_code=302,
            headers={"location": "http://127.0.0.1/private"},
            content=b"",
        )
    )

    with pytest.raises(WebSearchError, match="blocked redirect"):
        await client._fetch_text_for_inspection("https://docs.example/start")

    client._pinned_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_document_inspection_rejects_oversized_response(tmp_path) -> None:
    client = WebSearchClient(Settings(voiceloop_data_dir=str(tmp_path)))
    client._resolve_public_addresses = AsyncMock(  # type: ignore[method-assign]
        return_value=frozenset({"93.184.216.34"})
    )
    client._pinned_request = AsyncMock(  # type: ignore[method-assign]
        side_effect=WebSearchError("Dokumentacja przekroczyła limit rozmiaru.")
    )

    with pytest.raises(WebSearchError, match="limit rozmiaru"):
        await client._fetch_text_for_inspection("https://docs.example/large")


class _FakeSocket:
    def __init__(self, peer) -> None:
        self.peer = peer
        self.connected_to = None
        self.closed = False

    def settimeout(self, _timeout) -> None:
        return None

    def connect(self, destination) -> None:
        self.connected_to = destination

    def getpeername(self):
        return self.peer

    def close(self) -> None:
        self.closed = True


def _inspection_target() -> _InspectionTarget:
    return _InspectionTarget(
        url="http://docs.example/reference",
        scheme="http",
        hostname="docs.example",
        port=80,
        request_target="/reference",
        host_header="docs.example",
        addresses=("93.184.216.34",),
    )


@pytest.mark.parametrize("peer", [(), ("127.0.0.1", 80), ("93.184.216.35", 80)])
def test_pinned_inspection_fails_closed_before_http_request(peer) -> None:
    sock = _FakeSocket(peer)
    connection_factory = Mock(side_effect=AssertionError("HTTP request must not start"))

    with (
        patch("voiceloop.web_search.socket.socket", return_value=sock),
        patch("voiceloop.web_search.http.client.HTTPConnection", connection_factory),
        pytest.raises(WebSearchError),
    ):
        _pinned_request_sync(_inspection_target(), {}, 1.0)

    connection_factory.assert_not_called()
    assert sock.closed is True


def test_pinned_inspection_enforces_body_limit() -> None:
    sock = _FakeSocket(("93.184.216.34", 80))

    class Response:
        status = 200
        headers = {}

        @staticmethod
        def read(_limit):
            return b"x" * (MAX_INSPECTION_BYTES + 1)

        @staticmethod
        def getheaders():
            return []

    connection = Mock()
    connection.getresponse.return_value = Response()

    with (
        patch("voiceloop.web_search.socket.socket", return_value=sock),
        patch(
            "voiceloop.web_search.http.client.HTTPConnection",
            return_value=connection,
        ),
        pytest.raises(WebSearchError, match="limit rozmiaru"),
    ):
        _pinned_request_sync(_inspection_target(), {}, 1.0)

    connection.request.assert_called_once()


@pytest.mark.parametrize(
    "address",
    [
        "224.0.0.1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "64:ff9b:1::7f00:1",
        "2002:7f00:1::",
        "2001:0000:4136:e378:8000:63bf:3fff:fdd2",
    ],
)
def test_inspection_rejects_multicast_translation_and_transition_addresses(
    address,
) -> None:
    assert _is_public_unicast(ipaddress.ip_address(address)) is False
