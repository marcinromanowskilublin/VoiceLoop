import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from voiceloop.knowledge_tools import KnowledgeToolOrchestrator
from voiceloop.settings import Settings


def test_conversation_quality_dataset_has_sixty_balanced_cases(tmp_path) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "conversation-quality-v1.json"
    )
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) == 60
    assert len({case["id"] for case in cases}) == 60
    categories = {case["category"] for case in cases}
    assert categories == {
        "stable_knowledge",
        "current_web",
        "follow_up",
        "personal_context",
        "screen_question",
        "command_or_question",
    }
    assert all(sum(case["category"] == category for case in cases) == 10 for category in categories)

    web_search = SimpleNamespace(enabled=True)
    tools = KnowledgeToolOrchestrator(
        settings=Settings(voiceloop_data_dir=str(tmp_path)),
        web_search=web_search,  # type: ignore[arg-type]
        events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
    )
    for case in cases:
        assert tools.should_search(case["text"]) is case["expect_search"], case["id"]
