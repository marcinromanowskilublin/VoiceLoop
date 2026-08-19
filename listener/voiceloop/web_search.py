from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import httpx

from .settings import Settings

DOWNLOADABLE_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".doc",
    ".docx",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".txt",
    ".csv",
    ".md",
)

INSPECTABLE_DOWNLOAD_EXTENSIONS = (
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".txt",
    ".md",
    ".csv",
)

BINARY_CONTENT_TYPE_MARKERS = (
    "application/pdf",
    "application/zip",
    "application/octet-stream",
    "application/gzip",
    "application/x-gzip",
    "application/x-7z-compressed",
    "application/vnd",
    "image/",
    "audio/",
    "video/",
)


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    provider: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
        }


class WebSearchClient:
    """Read-only internet search client for fast fact lookup."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.web_search_enabled
        self.provider = self._normalize_provider(
            settings.web_search_provider or "duckduckgo"
        )
        self.fallback_provider = self._normalize_provider(
            settings.web_search_fallback_provider or "duckduckgo"
        )
        self.api_key = settings.web_search_api_key
        self.gemini_api_key = settings.gemini_api_key or settings.web_search_api_key
        self.venice_api_key = settings.cloud_llm_api_key or settings.web_search_api_key
        self.venice_base_url = (
            (settings.cloud_llm_base_url or "https://api.venice.ai/api/v1")
            .strip()
            .rstrip("/")
        )
        self.gemini_model = (
            settings.web_search_gemini_model or "gemini-3.6-flash"
        ).strip()
        self.timeout_seconds = max(
            1.0,
            min(settings.web_search_timeout_seconds, 30.0),
        )
        self.max_results = max(1, min(settings.web_search_max_results, 10))

    async def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączone w konfiguracji"
        chain = " -> ".join(self._provider_chain())
        try:
            results = await self.search("aktualne informacje", limit=1)
        except WebSearchError as exc:
            return False, f"{exc} (chain: {chain})"
        active_provider = results[0].provider if results else self.provider
        return True, f"provider={active_provider}; chain={chain}"

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> list[WebSearchResult]:
        if not self.enabled:
            raise WebSearchError("Wyszukiwanie internetowe jest wyłączone.")
        cleaned_query = query.strip()
        if not cleaned_query:
            raise WebSearchError("Zapytanie do wyszukiwania jest puste.")
        result_limit = max(1, min(limit or self.max_results, self.max_results, 10))
        errors: list[str] = []
        at_least_one_provider_responded = False
        for provider in self._provider_chain():
            try:
                results = await self._search_with_provider(
                    provider,
                    cleaned_query,
                    limit=result_limit,
                )
            except WebSearchError as exc:
                errors.append(f"{provider}: {exc}")
                continue
            at_least_one_provider_responded = True
            if results:
                return results[:result_limit]
        if at_least_one_provider_responded:
            return []
        if errors:
            raise WebSearchError("; ".join(errors))
        return []

    async def inspect_endpoint_in_documentation(
        self,
        *,
        api_name: str,
        endpoint: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        clean_api_name = re.sub(r"\s+", " ", api_name).strip()
        clean_endpoint = endpoint.strip().strip("\"'`")
        if len(clean_api_name) < 2:
            raise WebSearchError("Nazwa API jest zbyt krótka.")
        if len(clean_endpoint) < 1:
            raise WebSearchError("Endpoint jest pusty.")

        docs_query = f"{clean_api_name} API documentation {clean_endpoint}"
        result_limit = max(1, min(limit, self.max_results, 10))
        search_results = await self.search(docs_query, limit=result_limit)
        docs_access = self._documentation_access(search_results)

        checked_sources: list[dict[str, Any]] = []
        matched_source: dict[str, Any] = {}
        for item in search_results[:result_limit]:
            probe = await self._probe_endpoint_source(item, clean_endpoint)
            checked_sources.append(probe)
            if probe.get("matched") and not matched_source:
                matched_source = probe

        return {
            "query": docs_query,
            "api_name": clean_api_name,
            "endpoint": clean_endpoint,
            "endpoint_found": bool(matched_source),
            "matched_source": matched_source,
            "checked_sources": checked_sources,
            "documentation_access": docs_access,
            "results": [item.to_dict() for item in search_results],
        }

    @staticmethod
    def _normalize_provider(value: str | None) -> str:
        aliases = {
            "ddg": "duckduckgo",
            "duck": "duckduckgo",
            "brave_search": "brave",
            "google": "gemini",
            "gemini_search": "gemini",
            "gemini-google-search": "gemini",
            "veniceai": "venice",
        }
        normalized = (value or "").strip().casefold()
        if not normalized:
            return "duckduckgo"
        return aliases.get(normalized, normalized)

    def _provider_chain(self) -> list[str]:
        chain: list[str] = []
        if self.provider == "duckduckgo":
            if self._gemini_api_key_value():
                chain.append("gemini")
            elif self._venice_api_key_value():
                chain.append("venice")
        if self.provider not in chain:
            chain.append(self.provider)
        if (
            self.provider == "venice"
            and self._gemini_api_key_value()
            and "gemini" not in chain
        ):
            chain.append("gemini")
        if self.fallback_provider and self.fallback_provider not in chain:
            chain.append(self.fallback_provider)
        if "duckduckgo" not in chain:
            chain.append("duckduckgo")
        return chain

    async def _search_with_provider(
        self,
        provider: str,
        query: str,
        *,
        limit: int,
    ) -> list[WebSearchResult]:
        if provider == "duckduckgo":
            return await self._search_duckduckgo(query, limit=limit)
        if provider == "brave":
            return await self._search_brave(query, limit=limit)
        if provider == "gemini":
            return await self._search_gemini(query, limit=limit)
        if provider == "venice":
            return await self._search_venice(query, limit=limit)
        raise WebSearchError(f"Nieznany provider wyszukiwania: {provider}")

    def _api_key_value(self) -> str:
        return self._secret_value(self.api_key)

    def _venice_api_key_value(self) -> str:
        return self._secret_value(self.venice_api_key)

    def _gemini_api_key_value(self) -> str:
        return self._secret_value(self.gemini_api_key)

    @staticmethod
    def _secret_value(secret: Any) -> str:
        if secret is None:
            return ""
        get_secret_value = getattr(secret, "get_secret_value", None)
        if callable(get_secret_value):
            return str(get_secret_value()).strip()
        return str(secret).strip()

    async def _probe_endpoint_source(
        self,
        result: WebSearchResult,
        endpoint: str,
    ) -> dict[str, Any]:
        url = result.url.strip()
        endpoint_variants = self._endpoint_variants(endpoint)
        access_channel = self._documentation_channel_for_url(url)
        decoded_url = self._normalize_text_for_match(url)
        if endpoint_variants and any(variant in decoded_url for variant in endpoint_variants):
            return {
                "title": result.title,
                "url": url,
                "provider": result.provider,
                "matched": True,
                "match_source": "url",
                "access_channel": access_channel,
            }

        if access_channel == "download" and not self._is_inspectable_download_url(url):
            return {
                "title": result.title,
                "url": url,
                "provider": result.provider,
                "matched": False,
                "match_source": "not_checked",
                "access_channel": access_channel,
                "reason": "download_binary_not_checked",
            }

        try:
            text, content_type = await self._fetch_text_for_inspection(url)
        except WebSearchError as exc:
            return {
                "title": result.title,
                "url": url,
                "provider": result.provider,
                "matched": False,
                "match_source": "error",
                "access_channel": access_channel,
                "reason": str(exc),
            }

        if self._is_binary_content_type(content_type):
            return {
                "title": result.title,
                "url": url,
                "provider": result.provider,
                "matched": False,
                "match_source": "not_checked",
                "access_channel": access_channel,
                "reason": "binary_content_type",
            }

        normalized_text = self._normalize_text_for_match(text)
        matched = bool(
            endpoint_variants
            and any(variant in normalized_text for variant in endpoint_variants)
        )
        return {
            "title": result.title,
            "url": url,
            "provider": result.provider,
            "matched": matched,
            "match_source": "content" if matched else "content_no_match",
            "access_channel": access_channel,
            "content_type": content_type,
        }

    async def _fetch_text_for_inspection(self, url: str) -> tuple[str, str]:
        timeout = max(2.0, min(self.timeout_seconds * 2.0, 20.0))
        headers = {
            "Accept": (
                "text/html,application/json,text/plain,text/markdown,application/yaml,"
                "application/x-yaml,application/xml;q=0.9,*/*;q=0.8"
            )
        }
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"Nie udało się pobrać dokumentacji z {url}: {exc}") from exc

        if response.status_code >= 400:
            raise WebSearchError(
                f"Dokumentacja zwróciła HTTP {response.status_code} dla {url}."
            )
        content_type = str(response.headers.get("content-type", "")).casefold()
        raw = response.content[:1_000_000]
        encoding = response.encoding or "utf-8"
        try:
            text = raw.decode(encoding, errors="ignore")
        except (LookupError, UnicodeError):
            text = raw.decode("utf-8", errors="ignore")
        return text, content_type

    @staticmethod
    def _normalize_text_for_match(value: str) -> str:
        normalized = unquote(value).casefold().replace("\\/", "/")
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _endpoint_variants(endpoint: str) -> set[str]:
        raw = endpoint.strip().strip("\"'`")
        if not raw:
            return set()

        variants = {
            WebSearchClient._normalize_text_for_match(raw),
        }
        if "?" in raw:
            variants.add(WebSearchClient._normalize_text_for_match(raw.split("?", 1)[0]))

        expanded = set(variants)
        for value in list(expanded):
            trimmed = value.strip()
            if not trimmed:
                continue
            if trimmed.startswith("/"):
                variants.add(trimmed[1:])
            else:
                variants.add("/" + trimmed)
            variants.add(trimmed.replace("/", r"\/"))
        return {variant for variant in variants if variant}

    @staticmethod
    def _documentation_channel_for_url(url: str) -> str:
        lower_url = unquote(url).casefold()
        if any(
            marker in lower_url
            for marker in (
                "swagger.json",
                "openapi.json",
                "openapi.yaml",
                "openapi.yml",
                "/api-spec",
                "download=",
            )
        ):
            return "download"
        return (
            "download"
            if any(lower_url.endswith(ext) for ext in DOWNLOADABLE_EXTENSIONS)
            else "web"
        )

    @staticmethod
    def _is_inspectable_download_url(url: str) -> bool:
        lower_url = unquote(url).casefold()
        return any(lower_url.endswith(ext) for ext in INSPECTABLE_DOWNLOAD_EXTENSIONS)

    @staticmethod
    def _is_binary_content_type(content_type: str) -> bool:
        lowered = content_type.casefold()
        if not lowered:
            return False
        if any(marker in lowered for marker in BINARY_CONTENT_TYPE_MARKERS):
            return True
        textual_hints = (
            "text/",
            "application/json",
            "application/xml",
            "application/yaml",
            "application/x-yaml",
            "application/javascript",
            "application/x-javascript",
            "application/ld+json",
            "application/xhtml+xml",
        )
        return not any(hint in lowered for hint in textual_hints)

    def _documentation_access(
        self,
        results: list[WebSearchResult],
    ) -> dict[str, Any]:
        web_urls: list[str] = []
        downloadable_urls: list[str] = []
        for item in results:
            channel = self._documentation_channel_for_url(item.url)
            if channel == "download":
                downloadable_urls.append(item.url)
            else:
                web_urls.append(item.url)

        if web_urls and downloadable_urls:
            availability = "web_and_download"
            label = "strona i plik do pobrania"
        elif web_urls:
            availability = "web"
            label = "strona"
        elif downloadable_urls:
            availability = "download"
            label = "plik do pobrania"
        else:
            availability = "unknown"
            label = "nieznana"

        return {
            "availability": availability,
            "availability_label": label,
            "web_count": len(web_urls),
            "download_count": len(downloadable_urls),
            "web_urls": web_urls[:5],
            "download_urls": downloadable_urls[:5],
        }

    async def _search_duckduckgo(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[WebSearchResult]:
        payload = await self._get_json(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            },
            headers={"Accept": "application/json"},
        )
        if not isinstance(payload, dict):
            return []

        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()

        def add_result(title: str, url: str, snippet: str) -> None:
            clean_title = title.strip()
            clean_url = url.strip()
            clean_snippet = snippet.strip()
            if not clean_url or clean_url in seen_urls:
                return
            if not clean_title:
                clean_title = clean_url
            seen_urls.add(clean_url)
            results.append(
                WebSearchResult(
                    title=clean_title[:200],
                    url=clean_url[:1000],
                    snippet=clean_snippet[:700],
                    provider="duckduckgo",
                )
            )

        abstract_url = str(payload.get("AbstractURL") or "")
        abstract_text = str(payload.get("AbstractText") or "")
        heading = str(payload.get("Heading") or "")
        if abstract_url and (abstract_text or heading):
            add_result(heading or "DuckDuckGo", abstract_url, abstract_text)

        for item in payload.get("Results", []):
            if not isinstance(item, dict):
                continue
            add_result(
                str(item.get("Text") or ""),
                str(item.get("FirstURL") or ""),
                str(item.get("Text") or ""),
            )
            if len(results) >= limit:
                return results

        def collect_related(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                nested = item.get("Topics")
                if isinstance(nested, list):
                    collect_related([topic for topic in nested if isinstance(topic, dict)])
                add_result(
                    str(item.get("Text") or ""),
                    str(item.get("FirstURL") or ""),
                    str(item.get("Text") or ""),
                )
                if len(results) >= limit:
                    return

        related = payload.get("RelatedTopics")
        if isinstance(related, list):
            collect_related([topic for topic in related if isinstance(topic, dict)])

        if not results:
            answer = str(payload.get("Answer") or payload.get("Definition") or "").strip()
            answer_url = str(
                payload.get("AnswerURL") or payload.get("DefinitionURL") or ""
            ).strip()
            if answer and answer_url:
                add_result("Szybka odpowiedź", answer_url, answer)
        return results[:limit]

    async def _search_brave(self, query: str, *, limit: int) -> list[WebSearchResult]:
        token = self._api_key_value()
        if not token:
            raise WebSearchError("Brave Search wymaga WEB_SEARCH_API_KEY.")
        payload = await self._get_json(
            "https://api.search.brave.com/res/v1/web/search",
            params={
                "q": query,
                "count": str(limit),
                "safesearch": "moderate",
                "search_lang": "pl",
            },
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": token,
            },
        )
        if not isinstance(payload, dict):
            return []
        web = payload.get("web")
        if not isinstance(web, dict):
            return []
        items = web.get("results")
        if not isinstance(items, list):
            return []
        results: list[WebSearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("description") or "").strip()
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=(title or url)[:200],
                    url=url[:1000],
                    snippet=snippet[:700],
                    provider="brave",
                )
            )
            if len(results) >= limit:
                break
        return results

    async def _search_gemini(self, query: str, *, limit: int) -> list[WebSearchResult]:
        token = self._gemini_api_key_value()
        if not token:
            raise WebSearchError("Gemini wymaga GEMINI_API_KEY lub WEB_SEARCH_API_KEY.")
        payload = await self._post_json(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.gemini_model}:generateContent"
            ),
            params={"key": token},
            headers={"Content-Type": "application/json"},
            json_payload={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "Znajdź najnowsze i wiarygodne informacje dla zapytania: "
                                    f"{query}. Zwróć krótkie podsumowanie i oprzyj je na "
                                    "wynikach z internetu."
                                )
                            }
                        ],
                    }
                ],
                "tools": [{"google_search": {}}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
            },
        )
        if not isinstance(payload, dict):
            return []
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return []
        candidate = next(
            (item for item in candidates if isinstance(item, dict)),
            {},
        )
        answer_text = self._extract_candidate_text(candidate)
        summary_snippet = answer_text[:700] if answer_text else ""
        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()

        def add_result(title: str, url: str) -> None:
            clean_url = url.strip()
            if not clean_url or clean_url in seen_urls:
                return
            seen_urls.add(clean_url)
            clean_title = title.strip() or clean_url
            results.append(
                WebSearchResult(
                    title=clean_title[:200],
                    url=clean_url[:1000],
                    snippet=summary_snippet,
                    provider="gemini",
                )
            )

        grounding_metadata = candidate.get("groundingMetadata")
        if isinstance(grounding_metadata, dict):
            chunks = grounding_metadata.get("groundingChunks")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    web = chunk.get("web")
                    if not isinstance(web, dict):
                        continue
                    add_result(
                        str(web.get("title") or ""),
                        str(web.get("uri") or ""),
                    )
                    if len(results) >= limit:
                        return results[:limit]

        citation_metadata = candidate.get("citationMetadata")
        if isinstance(citation_metadata, dict):
            sources = citation_metadata.get("citationSources")
            if isinstance(sources, list):
                for source in sources:
                    if not isinstance(source, dict):
                        continue
                    add_result(
                        str(source.get("title") or ""),
                        str(source.get("uri") or ""),
                    )
                    if len(results) >= limit:
                        return results[:limit]
        return results[:limit]

    async def _search_venice(self, query: str, *, limit: int) -> list[WebSearchResult]:
        token = self._venice_api_key_value()
        if not token:
            raise WebSearchError(
                "Venice search wymaga CLOUD_LLM_API_KEY lub WEB_SEARCH_API_KEY."
            )
        payload = await self._post_json(
            f"{self.venice_base_url}/augment/search",
            params={},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json_payload={"query": query},
        )
        if not isinstance(payload, dict):
            return []
        items = payload.get("results")
        if not isinstance(items, list):
            return []
        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(item.get("title") or "").strip() or url
            snippet = str(item.get("content") or "").strip()
            date = str(item.get("date") or "").strip()
            if date and snippet:
                snippet = f"{date} — {snippet}"
            elif date:
                snippet = date
            results.append(
                WebSearchResult(
                    title=title[:200],
                    url=url[:1000],
                    snippet=snippet[:700],
                    provider="venice",
                )
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _extract_candidate_text(candidate: dict[str, Any]) -> str:
        content = candidate.get("content")
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()

    async def _get_json(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"Wyszukiwanie internetowe niedostępne: {exc}") from exc
        if response.status_code >= 400:
            raise WebSearchError(
                f"Wyszukiwanie internetowe zwróciło HTTP {response.status_code}."
            )
        try:
            return response.json()
        except ValueError as exc:
            raise WebSearchError("Wyszukiwanie internetowe zwróciło niepoprawny JSON.") from exc

    async def _post_json(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
        json_payload: dict[str, Any],
    ) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    params=params,
                    headers=headers,
                    json=json_payload,
                )
        except httpx.HTTPError as exc:
            raise WebSearchError(f"Wyszukiwanie internetowe niedostępne: {exc}") from exc
        if response.status_code >= 400:
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    error = payload.get("error")
                    if isinstance(error, dict):
                        detail = str(error.get("message") or "").strip()
            except ValueError:
                detail = ""
            if detail:
                raise WebSearchError(
                    "Wyszukiwanie internetowe zwróciło "
                    f"HTTP {response.status_code}: {detail}"
                )
            raise WebSearchError(
                f"Wyszukiwanie internetowe zwróciło HTTP {response.status_code}."
            )
        try:
            return response.json()
        except ValueError as exc:
            raise WebSearchError("Wyszukiwanie internetowe zwróciło niepoprawny JSON.") from exc
