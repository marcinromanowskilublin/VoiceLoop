from unittest.mock import AsyncMock

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.memory import MemoryStore
from voiceloop.models import PlanStep, RiskLevel
from voiceloop.screenpipe import ScreenpipeContext
from voiceloop.settings import Settings
from voiceloop.tts import WindowsTTS
from voiceloop.web_search import WebSearchResult


def test_policy_cannot_lower_registered_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    step = PlanStep(action_id="run_uivision_macro", risk=RiskLevel.LOW)

    secured = registry.enforce_policy(step)

    assert secured.risk is RiskLevel.MEDIUM
    assert secured.confirmation_required is True


def test_unknown_action_is_rejected(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    with pytest.raises(ValueError, match="unknown action"):
        registry.enforce_policy(PlanStep(action_id="powershell_anything"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "macro",
    [
        "../outside.json",
        r"..\outside.json",
        "nested/macro.json",
        r"nested\macro.json",
        "..json",
    ],
)
async def test_uivision_macro_rejects_path_traversal(tmp_path, macro) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    with pytest.raises(ValueError, match="Nieprawidłowa nazwa makra"):
        await registry._run_uivision_macro({"macro": macro})


@pytest.mark.asyncio
async def test_uivision_macro_requires_runtime_sync(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        uivision_home=str(tmp_path / "runtime"),
    )
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    with pytest.raises(FileNotFoundError, match="nie jest zsynchronizowane"):
        await registry._run_uivision_macro({"macro": "voiceloop_notatka.json"})


@pytest.mark.asyncio
async def test_uivision_macro_must_be_on_project_allowlist(tmp_path) -> None:
    runtime_macros = tmp_path / "runtime" / "macros"
    runtime_macros.mkdir(parents=True)
    (runtime_macros / "runtime-only.json").write_text("{}", encoding="utf-8")
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        uivision_home=str(tmp_path / "runtime"),
    )
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    with pytest.raises(FileNotFoundError, match="allowliście projektu"):
        await registry._run_uivision_macro({"macro": "runtime-only.json"})


def test_describe_active_window_is_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    step = registry.enforce_policy(PlanStep(action_id="describe_active_window"))

    assert step.risk is RiskLevel.LOW
    assert step.confirmation_required is False
    definition = next(
        item for item in registry.definitions() if item["id"] == "describe_active_window"
    )
    assert definition["args_schema"]["properties"] == {}


def test_minimize_window_actions_are_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    for action_id in ("minimize_active_window", "minimize_all_windows"):
        step = registry.enforce_policy(PlanStep(action_id=action_id))
        assert step.risk is RiskLevel.LOW
        assert step.confirmation_required is False
        assert registry.has_action(action_id)


def test_chat_split_actions_are_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    for action_id in ("open_chat", "open_gpt_chat", "open_gemini_chat"):
        step = registry.enforce_policy(PlanStep(action_id=action_id))
        assert step.risk is RiskLevel.LOW
        assert step.confirmation_required is False
        assert registry.has_action(action_id)


def test_copy_actions_are_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    for action_id in (
        "copy_selected_text",
        "copy_number_under_cursor",
        "copy_sentence_under_cursor",
    ):
        step = registry.enforce_policy(PlanStep(action_id=action_id))
        assert step.risk is RiskLevel.LOW
        assert step.confirmation_required is False
        assert registry.has_action(action_id)


def test_search_web_action_is_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    step = registry.enforce_policy(PlanStep(action_id="search_web"))

    assert step.risk is RiskLevel.LOW
    assert step.confirmation_required is False
    assert registry.has_action("search_web")


def test_remember_last_source_requires_confirmation(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    step = registry.enforce_policy(PlanStep(action_id="remember_last_source"))

    assert step.risk is RiskLevel.MEDIUM
    assert step.confirmation_required is True
    assert registry.has_action("remember_last_source")


def test_safe_paste_action_requires_confirmation(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    step = registry.enforce_policy(PlanStep(action_id="paste_text_safe"))

    assert step.risk is RiskLevel.MEDIUM
    assert step.confirmation_required is True
    assert registry.has_action("describe_text_target")


def test_action_layers_expose_native_uia_rpa_priority(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    definitions = {item["id"]: item for item in registry.definitions()}
    assert definitions["minimize_active_window"]["execution_layer"] == 1
    assert definitions["paste_text_safe"]["execution_layer"] == 2
    assert definitions["run_uivision_macro"]["execution_layer"] == 3


@pytest.mark.asyncio
async def test_minimize_active_window_uses_win32(tmp_path, monkeypatch) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_minimize_active_window_sync",
        staticmethod(lambda: ("Zminimalizowałem okno „Notatnik”.", {"window_title": "Notatnik"})),
    )

    message, data = await registry._minimize_active_window({})

    assert "Notatnik" in message
    assert data["window_title"] == "Notatnik"


@pytest.mark.asyncio
async def test_copy_selected_text_uses_helper(tmp_path, monkeypatch) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_copy_selected_text_sync",
        staticmethod(lambda: ("Skopiowałem zaznaczony tekst.", {"text": "abc"})),
    )

    message, data = await registry._copy_selected_text({})

    assert "Skopiowałem" in message
    assert data["text"] == "abc"


@pytest.mark.asyncio
async def test_search_web_returns_results(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    registry.web_search.search = AsyncMock(
        return_value=[
            WebSearchResult(
                title="AI news",
                url="https://example.com/ai-news",
                snippet="Szybkie podsumowanie",
                provider="duckduckgo",
            )
        ]
    )

    message, data = await registry._search_web({"query": "ai news", "limit": 3})

    assert "Znalazłem 1 wynik" in message
    assert data["query"] == "ai news"
    assert data["results"][0]["url"] == "https://example.com/ai-news"


@pytest.mark.asyncio
async def test_search_web_checks_endpoint_in_api_docs(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    registry.web_search.inspect_endpoint_in_documentation = AsyncMock(
        return_value={
            "query": "stripe API documentation /v1/customers",
            "api_name": "stripe",
            "endpoint": "/v1/customers",
            "endpoint_found": True,
            "matched_source": {
                "url": "https://docs.stripe.com/api/customers",
                "matched": True,
                "match_source": "content",
            },
            "checked_sources": [{"url": "https://docs.stripe.com/api/customers"}],
            "documentation_access": {
                "availability": "web_and_download",
                "availability_label": "strona i plik do pobrania",
            },
            "results": [
                {
                    "title": "Stripe API",
                    "url": "https://docs.stripe.com/api/customers",
                    "snippet": "Customers API",
                    "provider": "venice",
                }
            ],
        }
    )

    message, data = await registry._search_web(
        {
            "query": "stripe API documentation /v1/customers",
            "limit": 5,
            "api_name": "stripe",
            "endpoint": "/v1/customers",
        }
    )

    assert "stripe" in message.casefold()
    assert "endpoint" in message.casefold()
    assert "dokumentacja" in message.casefold()
    assert data["endpoint_check"]["endpoint_found"] is True
    assert data["results"][0]["url"] == "https://docs.stripe.com/api/customers"


@pytest.mark.asyncio
async def test_remember_last_source_saves_latest_web_result(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    registry = ActionRegistry(
        settings,
        memory,
        WindowsTTS(),
    )
    registry.web_search.search = AsyncMock(
        return_value=[
            WebSearchResult(
                title="API docs",
                url="https://docs.example.com/api",
                snippet="Reference docs",
                provider="venice",
            )
        ]
    )

    await registry._search_web({"query": "example api docs", "limit": 3})
    message, data = await registry._remember_last_source({"index": 1})
    memories = await memory.list_memories(limit=5, kind="web_source")

    assert "Zapamiętałem źródło" in message
    assert data["url"] == "https://docs.example.com/api"
    assert memories
    assert "https://docs.example.com/api" in memories[0].content


@pytest.mark.asyncio
async def test_remember_last_source_requires_prior_search(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    registry = ActionRegistry(
        settings,
        memory,
        WindowsTTS(),
    )

    with pytest.raises(RuntimeError, match="Brak ostatnich źródeł"):
        await registry._remember_last_source({})


@pytest.mark.asyncio
async def test_describe_text_target_uses_helper(tmp_path, monkeypatch) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_text_target_info_sync",
        staticmethod(
            lambda: {
                "window_title": "ChatGPT - Chrome",
                "process_name": "chrome.exe",
                "field_name": "Message",
                "is_editable": True,
                "looks_like_address_bar": False,
                "safe_for_typing": True,
            }
        ),
    )

    message, data = await registry._describe_text_target({})

    assert "bezpiecznie" in message
    assert data["safe_for_typing"] is True


@pytest.mark.asyncio
async def test_safe_paste_rejects_address_bar(tmp_path, monkeypatch) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_text_target_info_sync",
        staticmethod(
            lambda: {
                "window_title": "Nowa karta - Google Chrome",
                "process_name": "chrome.exe",
                "field_name": "Address and search bar",
                "is_editable": True,
                "looks_like_address_bar": True,
                "safe_for_typing": False,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="pasek adresu"):
        await registry._paste_text_safe({"text": "hej"})


@pytest.mark.asyncio
async def test_safe_paste_accepts_matching_target(tmp_path, monkeypatch) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_text_target_info_sync",
        staticmethod(
            lambda: {
                "window_title": "ChatGPT - Google Chrome",
                "process_name": "chrome.exe",
                "field_name": "Message composer",
                "is_editable": True,
                "looks_like_address_bar": False,
                "safe_for_typing": True,
            }
        ),
    )
    clipboard_calls: list[str] = []
    monkeypatch.setattr(
        ActionRegistry,
        "_write_clipboard_text_sync",
        staticmethod(lambda text: clipboard_calls.append(text)),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_send_paste_shortcut_sync",
        staticmethod(lambda: None),
    )

    message, data = await registry._paste_text_safe(
        {"text": "Dzien dobry", "expected_window": "chatgpt"}
    )

    assert "Wkleiłem tekst" in message
    assert data["safe_for_typing"] is True
    assert clipboard_calls == ["Dzien dobry"]


def test_extract_number_from_text() -> None:
    assert ActionRegistry._extract_number_from_text("Kontakt: +48 601 234 567") == "+48 601 234 567"
    assert ActionRegistry._extract_number_from_text("Brak danych") is None


def test_extract_sentence_from_text() -> None:
    assert (
        ActionRegistry._extract_sentence_from_text("To jest pierwsze zdanie. Drugie zdanie!")
        == "To jest pierwsze zdanie."
    )
    assert ActionRegistry._extract_sentence_from_text("jedno") is None


@pytest.mark.asyncio
async def test_describe_recent_activity_reads_screenpipe(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    registry.screenpipe.recent_activity = AsyncMock(
        return_value=[
            ScreenpipeContext(
                app_name="Cursor.exe",
                window_name="VoiceLoop - Cursor",
                timestamp="2026-08-09T21:00:00Z",
            )
        ]
    )

    message, data = await registry._describe_recent_activity({"minutes": 60})

    assert "Cursor.exe" in message
    assert data["minutes"] == 60
    assert data["items"][0]["window_name"] == "VoiceLoop - Cursor"
