from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.memory import MemoryStore
from voiceloop.models import CommandPlan, PlanStep, RiskLevel
from voiceloop.qdrant_memory import QdrantUnavailableError
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
    step = PlanStep(
        action_id="run_uivision_macro",
        args={"macro": "test.json"},
        risk=RiskLevel.LOW,
    )

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
async def test_recall_prefers_five_space_vector_memory(tmp_path) -> None:
    class EmbeddingsStub:
        enabled = True

        def accepts_private_text(self) -> bool:
            return True

        async def embed_queries(self, documents):
            return [
                [float(index), 0.0, 1.0]
                for index, _document in enumerate(documents, start=1)
            ]

    class QdrantStub:
        enabled = True

        def __init__(self) -> None:
            self.kwargs = {}

        def accepts_private_data(self) -> bool:
            return True

        async def search(self, **kwargs):
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    source="screenpipe_meeting",
                    source_id="meeting:2",
                    title="Decyzja",
                    content="Ustalono wdrożenie Qdrant V2.",
                    metadata={"retrieval_evidence": {"spaces": {"decision": {}}}},
                    score=0.88,
                    created_at=datetime.now(UTC),
                )
            ]

    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    qdrant = QdrantStub()
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        memory,
        WindowsTTS(),
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    message, payload = await registry._recall(
        {"query": "Jaką decyzję ustaliliśmy w sprawie Qdrant?"}
    )

    assert "pamięci semantycznej" in message
    assert payload["retrieval"] == "vector_v2"
    assert payload["items"][0]["source_id"] == "meeting:2"
    assert len(qdrant.kwargs["query_vectors"]) == 5


@pytest.mark.asyncio
async def test_recall_falls_back_to_sqlite_when_qdrant_is_unavailable(tmp_path) -> None:
    class EmbeddingsStub:
        enabled = True

        def accepts_private_text(self) -> bool:
            return True

        async def embed_queries(self, documents):
            return [
                [1.0, 0.0, 0.0] if index == 0 else [0.0, float(index), 0.0]
                for index, _document in enumerate(documents)
            ]

    class QdrantStub:
        enabled = True

        def accepts_private_data(self) -> bool:
            return True

        async def search(self, **_kwargs):
            raise QdrantUnavailableError("down")

    memory = MemoryStore(tmp_path / "voice.db")
    await memory.initialize()
    await memory.upsert_vector_memory(
        source="manual_memory",
        source_id="fallback-1",
        title="Fallback SQLite",
        content="Lokalny wpis dostępny bez Qdranta.",
        embedding=[1.0, 0.0, 0.0],
        metadata={"origin": "sqlite"},
    )
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        memory,
        WindowsTTS(),
        embeddings=EmbeddingsStub(),  # type: ignore[arg-type]
        qdrant=QdrantStub(),  # type: ignore[arg-type]
    )

    message, payload = await registry._recall({"query": "fallback"})

    assert "pamięci semantycznej" in message
    assert payload["retrieval"] == "vector_v2"
    assert payload["items"][0]["source_id"] == "fallback-1"


def test_policy_rejects_missing_required_action_argument(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    with pytest.raises(
        ValueError,
        match="missing_required_argument:query",
    ):
        registry.enforce_policy(PlanStep(action_id="search_web"))


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

    for action_id in (
        "minimize_active_window",
        "minimize_window_under_cursor",
        "minimize_all_windows",
    ):
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


def test_capability_catalog_separates_voiceattack_and_native_actions(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    catalog = registry.capability_catalog()
    va_ids = {item["id"] for item in catalog["voiceattack_actions"]}
    all_ids = {item["id"] for item in catalog["voiceloop_actions"]}

    assert "rename_under_cursor" in va_ids
    assert "paste_text_safe" in all_ids
    assert "list_capabilities" not in all_ids
    assert all("label" in item for item in catalog["voiceloop_actions"])


def test_copy_actions_are_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    for action_id in (
        "copy_selected_text",
        "copy_text_under_cursor",
        "copy_email_under_cursor",
        "copy_number_under_cursor",
        "copy_sentence_under_cursor",
        "select_sentence_under_cursor",
        "select_paragraph_under_cursor",
    ):
        step = registry.enforce_policy(PlanStep(action_id=action_id))
        assert step.risk is RiskLevel.LOW
        assert step.confirmation_required is False
        assert registry.has_action(action_id)


def test_close_window_under_cursor_requires_confirmation(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    step = registry.enforce_policy(PlanStep(action_id="close_window_under_cursor"))

    assert step.risk is RiskLevel.MEDIUM
    assert step.confirmation_required is True


def test_search_web_action_is_registered_as_low_risk(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    step = registry.enforce_policy(PlanStep(action_id="search_web", args={"query": "VoiceLoop"}))

    assert step.risk is RiskLevel.LOW
    assert step.confirmation_required is False
    assert registry.has_action("search_web")


@pytest.mark.asyncio
async def test_open_folder_allows_only_this_pc(tmp_path, monkeypatch) -> None:
    import os

    launched: list[str] = []
    monkeypatch.setattr(
        os,
        "startfile",
        lambda target: launched.append(target),
        raising=False,
    )
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    message, data = await registry._open_folder({"folder_id": "this_pc"})

    assert "Ten komputer" in message
    assert data == {"folder_id": "this_pc"}
    assert launched == ["shell:MyComputerFolder"]
    with pytest.raises(ValueError, match="allowlisty"):
        await registry._open_folder({"folder_id": r"C:\Windows"})


@pytest.mark.asyncio
async def test_open_app_allows_only_whatsapp(tmp_path, monkeypatch) -> None:
    import os

    launched: list[str] = []
    monkeypatch.setattr(
        os,
        "startfile",
        lambda target: launched.append(target),
        raising=False,
    )
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    message, data = await registry._open_app({"app_id": "whatsapp"})

    assert "WhatsApp" in message
    assert data == {"app_id": "whatsapp"}
    assert launched == ["whatsapp:"]
    with pytest.raises(ValueError, match="allowlisty"):
        await registry._open_app({"app_id": "autodesk"})


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

    step = registry.enforce_policy(PlanStep(action_id="paste_text_safe", args={"text": "test"}))

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
    assert definitions["close_window_under_cursor"]["execution_layer"] == 1
    assert definitions["copy_text_under_cursor"]["execution_layer"] == 2
    assert definitions["copy_email_under_cursor"]["execution_layer"] == 2
    assert definitions["select_sentence_under_cursor"]["execution_layer"] == 2
    assert definitions["select_paragraph_under_cursor"]["execution_layer"] == 2
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
async def test_minimize_window_under_cursor_uses_targeted_window(tmp_path, monkeypatch) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    registry = ActionRegistry(
        settings,
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_minimize_window_under_cursor_sync",
        staticmethod(
            lambda: (
                "Zminimalizowałem okno „Kalendarz” wskazywane kursorem.",
                {"window_title": "Kalendarz", "already_minimized": False},
            )
        ),
    )

    message, data = await registry._minimize_window_under_cursor({})

    assert "Kalendarz" in message
    assert data["already_minimized"] is False


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


def test_extract_emails_from_text_deduplicates_case_insensitively() -> None:
    assert ActionRegistry._extract_emails_from_text(
        "Kontakt: Marcin.R@example.com oraz marcin.r@EXAMPLE.com."
    ) == ["Marcin.R@example.com"]
    assert ActionRegistry._extract_emails_from_text("Brak danych") == []


def test_copy_email_under_cursor_writes_single_address(monkeypatch) -> None:
    clipboard: list[str] = []
    monkeypatch.setattr(
        ActionRegistry,
        "_text_candidates_under_cursor_sync",
        staticmethod(lambda: ["Napisz do Marcin.R@example.com"]),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_write_clipboard_text_sync",
        staticmethod(clipboard.append),
    )

    message, data = ActionRegistry._copy_email_under_cursor_sync()

    assert message == "Skopiowałem adres e-mail."
    assert data == {"email": "Marcin.R@example.com", "source": "cursor_text"}
    assert clipboard == ["Marcin.R@example.com"]


def test_copy_email_under_cursor_rejects_ambiguous_element(monkeypatch) -> None:
    monkeypatch.setattr(
        ActionRegistry,
        "_text_candidates_under_cursor_sync",
        staticmethod(lambda: ["a@example.com lub b@example.com"]),
    )

    with pytest.raises(RuntimeError, match="kilka adresów"):
        ActionRegistry._copy_email_under_cursor_sync()


def test_copy_text_under_cursor_uses_closest_accessible_text(monkeypatch) -> None:
    clipboard: list[str] = []
    monkeypatch.setattr(
        ActionRegistry,
        "_text_candidates_under_cursor_sync",
        staticmethod(lambda: ["  Tekst   wskazanego elementu  ", "Tekst rodzica"]),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_write_clipboard_text_sync",
        staticmethod(clipboard.append),
    )

    message, data = ActionRegistry._copy_text_under_cursor_sync()

    assert message == "Skopiowałem tekst spod kursora."
    assert data["text"] == "Tekst wskazanego elementu"
    assert clipboard == ["Tekst wskazanego elementu"]


def test_sentence_span_uses_cursor_offset() -> None:
    text = "Pierwsze zdanie. Drugie zdanie jest wskazane! Trzecie."

    span = ActionRegistry._sentence_span_at_offset(text, text.index("wskazane"))

    assert span is not None
    assert span[2] == "Drugie zdanie jest wskazane!"


@pytest.mark.parametrize(
    ("unit", "expected_message"),
    [
        ("sentence", "Zaznaczyłem zdanie pod kursorem."),
        ("paragraph", "Zaznaczyłem akapit pod kursorem."),
    ],
)
def test_select_text_under_cursor_reports_uia_result(
    monkeypatch,
    unit,
    expected_message,
) -> None:
    monkeypatch.setattr(
        ActionRegistry,
        "_select_uia_text_under_cursor_sync",
        staticmethod(lambda selected_unit: f"wybrany {selected_unit}"),
    )

    if unit == "sentence":
        message, data = ActionRegistry._select_sentence_under_cursor_sync()
    else:
        message, data = ActionRegistry._select_paragraph_under_cursor_sync()

    assert message == expected_message
    assert data["unit"] == unit
    assert data["text"] == f"wybrany {unit}"


def test_close_window_under_cursor_uses_graceful_wm_close(monkeypatch) -> None:
    closed: list[int] = []
    monkeypatch.setattr(
        ActionRegistry,
        "_window_info_sync",
        staticmethod(
            lambda hwnd: {
                "hwnd": 123,
                "process_id": 456,
                "window_title": "Dokument - Notatnik",
                "process_name": "notepad.exe",
                "class_name": "Notepad",
            }
        ),
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_post_close_window_sync",
        staticmethod(closed.append),
    )

    message, data = ActionRegistry._close_window_under_cursor_sync(
        {
            "expected_hwnd": 123,
            "expected_process_id": 456,
            "expected_window_title": "Dokument - Notatnik",
            "expected_process_name": "notepad.exe",
            "expected_class_name": "Notepad",
        }
    )

    assert "Dokument - Notatnik" in message
    assert data["mode"] == "wm_close"
    assert closed == [123]


@pytest.mark.asyncio
async def test_close_window_target_is_bound_before_confirmation(
    tmp_path,
    monkeypatch,
) -> None:
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )
    monkeypatch.setattr(
        registry,
        "_window_under_cursor_info_sync",
        lambda: {
            "hwnd": 123,
            "process_id": 456,
            "window_title": "Dokument - Notatnik",
            "process_name": "notepad.exe",
            "class_name": "Notepad",
        },
    )
    plan = CommandPlan(
        request_id="request",
        intent="task",
        steps=[PlanStep(action_id="close_window_under_cursor")],
    )

    await registry.bind_execution_targets(plan)

    assert plan.steps[0].args["expected_hwnd"] == 123
    assert plan.steps[0].args["expected_process_id"] == 456
    assert plan.steps[0].args["expected_window_title"] == "Dokument - Notatnik"


def test_close_window_rejects_changed_target_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        ActionRegistry,
        "_window_info_sync",
        staticmethod(
            lambda hwnd: {
                "hwnd": hwnd,
                "process_id": 999,
                "window_title": "Inne okno",
                "process_name": "other.exe",
                "class_name": "Other",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="Tożsamość"):
        ActionRegistry._close_window_under_cursor_sync(
            {
                "expected_hwnd": 123,
                "expected_process_id": 456,
                "expected_window_title": "Dokument - Notatnik",
                "expected_process_name": "notepad.exe",
                "expected_class_name": "Notepad",
            }
        )


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


def test_capability_catalog_exposes_communication_metadata(tmp_path) -> None:
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    actions = {
        item["id"]: item for item in registry.capability_catalog()["voiceloop_actions"]
    }
    open_app = actions["open_app"]

    assert open_app["category"] == "aplikacje"
    assert open_app["spoken_name"]
    assert "otwórz WhatsApp" in open_app["positive_examples"]
    assert open_app["result_reporting"]


@pytest.mark.asyncio
async def test_capability_answer_does_not_claim_close_by_app_name(tmp_path) -> None:
    registry = ActionRegistry(
        Settings(voiceloop_data_dir=str(tmp_path)),
        MemoryStore(tmp_path / "voice.db"),
        WindowsTTS(),
    )

    message, _catalog = await registry._list_capabilities(
        {"query": "Czy potrafisz zamknąć aplikację po samej nazwie?"}
    )

    assert message.startswith("Nie zamykam jeszcze")
    assert "pod kursorem" in message
