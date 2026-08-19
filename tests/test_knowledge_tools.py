from types import SimpleNamespace
from unittest.mock import AsyncMock

from voiceloop.knowledge_tools import KnowledgeToolOrchestrator
from voiceloop.settings import Settings
from voiceloop.web_search import WebSearchResult


class SearchStub:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, limit: int):
        self.calls += 1
        return [
            WebSearchResult(
                title="Źródło A",
                url="https://example.com/a",
                snippet=f"Aktualna odpowiedź dla {query}",
                provider="test",
            ),
            WebSearchResult(
                title="Duplikat",
                url="https://example.com/a",
                snippet="Ten sam URL",
                provider="test",
            ),
            WebSearchResult(
                title="Źródło B",
                url="https://example.org/b",
                snippet="Drugie potwierdzenie",
                provider="test",
            ),
        ][:limit]


async def test_current_question_searches_deduplicates_and_caches(tmp_path) -> None:
    search = SearchStub()
    events = SimpleNamespace(publish=AsyncMock())
    tools = KnowledgeToolOrchestrator(
        settings=Settings(voiceloop_data_dir=str(tmp_path)),
        web_search=search,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
    )

    assert tools.should_search("Jaka jest dziś pogoda w Warszawie?")
    first = await tools.lookup(
        request_id="request-1",
        text="Sprawdź w internecie aktualną pogodę w Warszawie",
    )
    second = await tools.lookup(
        request_id="request-2",
        text="Sprawdź w internecie aktualną pogodę w Warszawie",
    )

    assert [item.url for item in first.observations] == [
        "https://example.com/a",
        "https://example.org/b",
    ]
    assert second.from_cache is True
    assert search.calls == 1


def test_personal_and_stable_questions_do_not_force_web(tmp_path) -> None:
    tools = KnowledgeToolOrchestrator(
        settings=Settings(voiceloop_data_dir=str(tmp_path)),
        web_search=SearchStub(),  # type: ignore[arg-type]
        events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
    )

    assert not tools.should_search("Co pamiętasz o mnie teraz?")
    assert not tools.should_search("Dlaczego niebo jest niebieskie?")
    assert tools.should_search("Jaka jest najnowsza wersja Pythona?")


async def test_empty_search_is_reported_as_unverified_instead_of_cached(tmp_path) -> None:
    class EmptySearch:
        enabled = True

        async def search(self, query: str, *, limit: int):
            return []

    events = SimpleNamespace(publish=AsyncMock())
    tools = KnowledgeToolOrchestrator(
        settings=Settings(voiceloop_data_dir=str(tmp_path)),
        web_search=EmptySearch(),  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
    )

    result = await tools.lookup(
        request_id="request-empty",
        text="Jaka jest aktualna cena bitcoina?",
    )

    assert result.observations == ()
    assert result.error == "Nie znaleziono aktualnych źródeł dla tego pytania."
    assert any(
        call.args[0] == "knowledge.search.failed"
        for call in events.publish.await_args_list
    )
