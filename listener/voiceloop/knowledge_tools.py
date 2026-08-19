from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from .conversation_telemetry import ConversationTelemetry
from .events import EventBus
from .models import ToolObservation
from .router import normalize_text
from .settings import Settings
from .web_search import WebSearchClient, WebSearchError

_EXPLICIT_WEB = re.compile(
    r"\b(?:sprawdz|wyszukaj|poszukaj|zweryfikuj)\b.*\b(?:internet|sie[cć]|web|online)\b"
)
_CURRENT_INFORMATION = re.compile(
    r"\b(?:dzis\w*|teraz|obecnie|aktualn\w*|najnowsz\w*|"
    r"pogod\w*|kurs\w*|cen\w*|wiadomosc\w*|news\w*|"
    r"wersj\w*|status\w*|notowani\w*|premier\w*)\b"
)
_PERSONAL_CONTEXT = re.compile(
    r"\b(?:o mnie|moj\w*\s+pamiec\w*|co pamietasz|nasza rozmowa|powiedzialem ci)\b"
)
_SCREEN_CONTEXT = re.compile(
    r"\b(?:ekran\w*|okn\w*|przycisk\w*|formularz\w*|zaznaczon\w*|"
    r"pole\b.{0,40}\baktywn\w*)\b"
)


@dataclass(frozen=True, slots=True)
class KnowledgeLookup:
    observations: tuple[ToolObservation, ...] = ()
    error: str | None = None
    from_cache: bool = False


class KnowledgeToolOrchestrator:
    """Bounded read-only web lookup for managed conversational turns."""

    def __init__(
        self,
        *,
        settings: Settings,
        web_search: WebSearchClient,
        events: EventBus,
        telemetry: ConversationTelemetry | None = None,
    ) -> None:
        self.enabled = bool(settings.knowledge_tools_enabled)
        self.web_search = web_search
        self.events = events
        self.telemetry = telemetry
        self.max_results = max(1, min(settings.knowledge_tools_max_results, 5))
        self.timeout_seconds = max(
            1.0,
            min(settings.knowledge_tools_timeout_seconds, 30.0),
        )
        self.cache_ttl_seconds = max(
            0.0,
            min(settings.knowledge_tools_cache_ttl_seconds, 3600.0),
        )
        self._cache: dict[str, tuple[float, tuple[ToolObservation, ...]]] = {}
        self._lock = asyncio.Lock()

    def should_search(self, text: str) -> bool:
        if not self.enabled or not self.web_search.enabled:
            return False
        normalized = normalize_text(text)
        if (
            not normalized
            or _PERSONAL_CONTEXT.search(normalized)
            or _SCREEN_CONTEXT.search(normalized)
        ):
            return False
        return bool(
            _EXPLICIT_WEB.search(normalized)
            or _CURRENT_INFORMATION.search(normalized)
        )

    async def lookup(self, *, request_id: str, text: str) -> KnowledgeLookup:
        query = self._query(text)
        cached = await self._cached(query)
        if cached is not None:
            await self.events.publish(
                "knowledge.search.completed",
                {
                    "request_id": request_id,
                    "query": query,
                    "count": len(cached),
                    "from_cache": True,
                    "sources": [item.model_dump(mode="json") for item in cached],
                },
            )
            return KnowledgeLookup(observations=cached, from_cache=True)

        if self.telemetry is not None:
            await self.telemetry.mark_request(request_id, "tool_started")
        await self.events.publish(
            "knowledge.search.started",
            {"request_id": request_id, "query": query},
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                results = await self.web_search.search(
                    query,
                    limit=self.max_results,
                )
        except TimeoutError:
            error = "Wyszukiwanie internetowe przekroczyło limit czasu."
            await self._publish_failure(request_id, query, error)
            return KnowledgeLookup(error=error)
        except WebSearchError as exc:
            error = str(exc)
            await self._publish_failure(request_id, query, error)
            return KnowledgeLookup(error=error)

        seen_urls: set[str] = set()
        observations: list[ToolObservation] = []
        for result in results:
            url = result.url.strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            observations.append(
                ToolObservation(
                    query=query,
                    title=result.title.strip(),
                    url=url,
                    snippet=result.snippet.strip(),
                    provider=result.provider.strip(),
                )
            )
            if len(observations) >= self.max_results:
                break
        frozen = tuple(observations)
        if not frozen:
            error = "Nie znaleziono aktualnych źródeł dla tego pytania."
            await self._publish_failure(request_id, query, error)
            return KnowledgeLookup(error=error)
        async with self._lock:
            self._cache[normalize_text(query)] = (time.monotonic(), frozen)
            self._prune_cache_locked()
        if self.telemetry is not None:
            await self.telemetry.mark_request(
                request_id,
                "tool_completed",
                metadata={
                    "knowledge_source_count": len(frozen),
                    "sources": [
                        {
                            "title": item.title,
                            "url": item.url,
                            "provider": item.provider,
                        }
                        for item in frozen[:3]
                    ],
                },
            )
        await self.events.publish(
            "knowledge.search.completed",
            {
                "request_id": request_id,
                "query": query,
                "count": len(frozen),
                "from_cache": False,
                "sources": [item.model_dump(mode="json") for item in frozen],
            },
        )
        return KnowledgeLookup(observations=frozen)

    async def _publish_failure(
        self,
        request_id: str,
        query: str,
        error: str,
    ) -> None:
        if self.telemetry is not None:
            await self.telemetry.mark_request(
                request_id,
                "tool_completed",
                metadata={"knowledge_error": error[:300]},
            )
        await self.events.publish(
            "knowledge.search.failed",
            {"request_id": request_id, "query": query, "error": error[:500]},
        )

    async def _cached(
        self,
        query: str,
    ) -> tuple[ToolObservation, ...] | None:
        if self.cache_ttl_seconds <= 0:
            return None
        key = normalize_text(query)
        async with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            created, observations = item
            if time.monotonic() - created > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            return observations

    def _prune_cache_locked(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, (created, _) in self._cache.items()
            if now - created > self.cache_ttl_seconds
        ]
        for key in expired:
            self._cache.pop(key, None)
        if len(self._cache) > 100:
            oldest = sorted(self._cache, key=lambda key: self._cache[key][0])
            for key in oldest[: len(self._cache) - 100]:
                self._cache.pop(key, None)

    @staticmethod
    def _query(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        cleaned = re.sub(
            r"^\s*(?:asystencie|asystent|venice|voiceloop)[\s,.:;!?-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:sprawdź|sprawdz|wyszukaj|poszukaj)\s+"
            r"(?:to\s+)?(?:w\s+)?(?:internecie|sieci|webie|online)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return " ".join(cleaned.split())[:500] or text.strip()[:500]
