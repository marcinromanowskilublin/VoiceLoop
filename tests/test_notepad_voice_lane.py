from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voiceloop.actions import ActionRegistry
from voiceloop.assistant import AssistantService
from voiceloop.events import EventBus
from voiceloop.executor import CommandExecutor
from voiceloop.memory import MemoryStore
from voiceloop.models import (
    CommandAccepted,
    CommandPlan,
    CommandRequest,
    CommandSource,
    CommandStatus,
    PlanStep,
    RiskLevel,
)
from voiceloop.notepad_document import NotepadIdentity, NotepadSnapshot
from voiceloop.router import confirmation_decision, deterministic_plan
from voiceloop.settings import Settings
from voiceloop.tts import WindowsTTS
from voiceloop.voice_conversation import VoiceConversationCoordinator


def _registry(tmp_path, **settings_kwargs) -> ActionRegistry:
    settings = Settings(voiceloop_data_dir=str(tmp_path), **settings_kwargs)
    return ActionRegistry(settings, MemoryStore(tmp_path / "voice.db"), WindowsTTS())


def _identity(**overrides) -> NotepadIdentity:
    data = dict(
        process_id=11,
        process_name="notepad.exe",
        hwnd=22,
        window_title="VoiceLoop demo - Notatnik",
        window_class="Notepad",
        control_type="Document",
        control_class="RichEditD2DPT",
        automation_id="editor",
        runtime_id="1,2,3",
        document_name="VoiceLoop demo",
    )
    data.update(overrides)
    return NotepadIdentity(**data)


def _snapshot(text: str = "stara tresc", **overrides) -> NotepadSnapshot:
    return NotepadSnapshot(identity=_identity(**overrides), text=text, control_count=1)


def test_copy_ignores_stale_clipboard_without_selection(tmp_path, monkeypatch) -> None:
    _registry(tmp_path)
    monkeypatch.setattr(
        "voiceloop.notepad_document.focused_selection_text",
        lambda: None,
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_read_clipboard_text_sync",
        staticmethod(lambda: "stary schowek"),
    )
    monkeypatch.setattr(ActionRegistry, "_send_copy_shortcut_sync", staticmethod(lambda: None))

    with pytest.raises(RuntimeError, match="wiarygodnego zaznaczenia"):
        ActionRegistry._copy_selected_text_sync()


def test_copy_selected_text_prefers_uia_selection(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(
        "voiceloop.notepad_document.focused_selection_text",
        lambda: "zaznaczenie uia",
    )
    monkeypatch.setattr(
        ActionRegistry,
        "_send_copy_shortcut_sync",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("clipboard copy"))),
    )

    message, data = ActionRegistry._copy_selected_text_sync()

    assert "zaznaczony tekst" in message.casefold() or "zaznaczenie" in message.casefold()
    assert data["text"] == "zaznaczenie uia"
    assert data["source"] == "uia_selection"
    assert registry.has_action("read_active_notepad")


def test_text_target_prefers_focus_over_cursor(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "voiceloop.notepad_document.is_notepad_process",
        lambda name: False,
    )
    monkeypatch.setattr(
        "voiceloop.notepad_document.focused_text_target",
        lambda: {
            "field_name": "Message",
            "control_type": "Edit",
            "automation_id": "composer",
            "class_name": "Edit",
            "is_editable": True,
            "target_source": "focused_control",
        },
    )
    win32 = SimpleNamespace(
        GetForegroundWindow=lambda: 7,
        GetWindowText=lambda hwnd: "ChatGPT",
        GetWindowThreadProcessId=lambda hwnd: (0, 9),
        OpenProcess=lambda *args, **kwargs: 1,
        CloseHandle=lambda handle: None,
        GetModuleFileNameEx=lambda handle, index: r"C:\chrome.exe",
        PROCESS_QUERY_INFORMATION=1,
        PROCESS_VM_READ=2,
        GetCursorPos=lambda: (_ for _ in ()).throw(AssertionError("cursor")),
    )
    monkeypatch.setattr("win32gui.GetForegroundWindow", win32.GetForegroundWindow)
    monkeypatch.setattr("win32gui.GetWindowText", win32.GetWindowText)
    monkeypatch.setattr("win32process.GetWindowThreadProcessId", win32.GetWindowThreadProcessId)
    monkeypatch.setattr("win32api.OpenProcess", win32.OpenProcess)
    monkeypatch.setattr("win32api.CloseHandle", win32.CloseHandle)
    monkeypatch.setattr("win32process.GetModuleFileNameEx", win32.GetModuleFileNameEx)
    monkeypatch.setattr("win32con.PROCESS_QUERY_INFORMATION", 1, raising=False)
    monkeypatch.setattr("win32con.PROCESS_VM_READ", 2, raising=False)

    info = ActionRegistry._text_target_info_sync()

    assert info["target_source"] == "focused_control"
    assert info["field_name"] == "Message"
    assert info["cursor_x"] is None


@pytest.mark.asyncio
async def test_paste_rejects_cursor_when_focus_differs(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(
        ActionRegistry,
        "_text_target_info_sync",
        staticmethod(
            lambda: {
                "window_title": "ChatGPT",
                "process_name": "chrome.exe",
                "field_name": "Address bar",
                "is_editable": True,
                "looks_like_address_bar": False,
                "safe_for_typing": True,
                "target_source": "cursor",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="pozycji myszy"):
        await registry._paste_text_safe({"text": "hej"})


def test_router_maps_notepad_read_and_literal_write() -> None:
    read = deterministic_plan(CommandRequest(text="Odczytaj notatkę"))
    write = deterministic_plan(
        CommandRequest(text="Wpisz w notatniku: VoiceLoop demo jeden")
    )
    help_plan = deterministic_plan(
        CommandRequest(text="Co możesz zrobić z tą notatką?")
    )

    assert read is not None and read.steps[0].action_id == "read_active_notepad"
    assert write is not None
    assert write.steps[0].action_id == "write_active_notepad"
    assert write.steps[0].args["text"] == "VoiceLoop demo jeden"
    assert help_plan is not None and help_plan.intent == "notepad_help"
    assert confirmation_decision("Potwierdzam zamianę") == "confirm"
    assert confirmation_decision("anuluj zadanie") == "cancel"
    assert confirmation_decision("otwórz przeglądarkę") is None


def test_document_text_is_not_parsed_as_command() -> None:
    plan = deterministic_plan(CommandRequest(text="usuń wszystkie pliki"))
    assert plan is None or not any(
        step.action_id == "write_active_notepad" for step in (plan.steps or [])
    )


def test_allowlist_blocks_out_of_scope_action(tmp_path) -> None:
    registry = _registry(tmp_path, notepad_voice_lane=True)
    with pytest.raises(ValueError, match="not allowed"):
        registry.enforce_policy(PlanStep(action_id="open_browser"))
    step = registry.enforce_policy(
        PlanStep(action_id="write_active_notepad", args={"text": "demo"})
    )
    assert step.confirmation_required is True
    assert step.risk is RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_write_bind_and_revalidate_detect_document_change(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    first = _snapshot("alfa")
    second = _snapshot("beta")
    states = {"current": first}
    monkeypatch.setattr(
        "voiceloop.notepad_document.resolve_active_notepad",
        lambda: states["current"],
    )
    request = CommandRequest(text="wpisz w notatniku: nowa tresc")
    plan = CommandPlan(
        request_id=request.request_id,
        intent="write_active_notepad",
        steps=[
            PlanStep(
                action_id="write_active_notepad",
                args={"text": "nowa tresc"},
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
            )
        ],
    )

    bound = await registry.bind_execution_targets(plan)
    assert bound.steps[0].args["expected_source_text"] == "alfa"
    assert "potwierdzam" in bound.response_text.casefold()

    states["current"] = second
    with pytest.raises(RuntimeError, match="Treść notatki zmieniła się"):
        await registry.revalidate_bound_targets(bound)


@pytest.mark.asyncio
async def test_write_reports_unverified_result_without_retry(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    calls = {"write": 0}

    def fake_write(text, *, expected=None, expected_source_text=None):
        calls["write"] += 1
        return (
            "Zapis w Notatniku nie potwierdził się odczytem. Sprawdź treść ręcznie.",
            {"verified": False, "verification": "mismatch", "actual_text": "inne"},
        )

    monkeypatch.setattr("voiceloop.notepad_document.write_active_notepad", fake_write)
    result = await registry.execute(
        PlanStep(action_id="write_active_notepad", args={"text": "oczekiwane"})
    )

    assert result.success is False
    assert result.data["verification"] == "mismatch"
    assert calls["write"] == 1


@pytest.mark.asyncio
async def test_voice_confirmation_cancel_and_expired(tmp_path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    registry = _registry(tmp_path)
    executor = CommandExecutor(
        memory=store,
        actions=registry,
        events=EventBus(),
        queue_limit=4,
    )
    await executor.start()
    monkeypatch.setattr(
        "voiceloop.notepad_document.resolve_active_notepad",
        lambda: _snapshot("stare"),
    )
    request = CommandRequest(text="wpisz w notatniku: demo")
    await store.create_command(request)
    plan = CommandPlan(
        request_id=request.request_id,
        intent="write_active_notepad",
        steps=[
            PlanStep(
                action_id="write_active_notepad",
                args={"text": "demo"},
                confirmation_required=True,
            )
        ],
    )
    waiting = await executor.submit(plan)
    assert waiting is not None
    assert waiting.status is CommandStatus.AWAITING_CONFIRMATION

    cancelled = await executor.cancel(request.request_id)
    assert cancelled is not None
    assert cancelled.status is CommandStatus.CANCELLED

    second = CommandRequest(text="wpisz w notatniku: druga")
    await store.create_command(second)
    second_plan = CommandPlan(
        request_id=second.request_id,
        intent="write_active_notepad",
        steps=[
            PlanStep(
                action_id="write_active_notepad",
                args={"text": "druga"},
                confirmation_required=True,
            )
        ],
    )
    await executor.submit(second_plan)
    import voiceloop.executor as executor_module

    original_ttl = executor_module.CONFIRMATION_TTL_SECONDS
    executor_module.CONFIRMATION_TTL_SECONDS = 0
    try:
        expired = await executor.confirm(second.request_id)
    finally:
        executor_module.CONFIRMATION_TTL_SECONDS = original_ttl
    assert expired is not None
    assert expired.status is CommandStatus.CANCELLED
    assert expired.error == "Potwierdzenie wygasło."
    await executor.close()


@pytest.mark.asyncio
async def test_stop_cancels_pending_and_does_not_promise_undo(tmp_path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    registry = _registry(tmp_path)
    executor = CommandExecutor(
        memory=store,
        actions=registry,
        events=EventBus(),
        queue_limit=4,
    )
    await executor.start()
    monkeypatch.setattr(
        "voiceloop.notepad_document.resolve_active_notepad",
        lambda: _snapshot("stare"),
    )
    request = CommandRequest(text="wpisz w notatniku: demo")
    await store.create_command(request)
    await executor.submit(
        CommandPlan(
            request_id=request.request_id,
            intent="write_active_notepad",
            steps=[
                PlanStep(
                    action_id="write_active_notepad",
                    args={"text": "demo"},
                    confirmation_required=True,
                )
            ],
        )
    )
    await executor.stop_all()
    command = await store.get_command(request.request_id)
    assert command is not None
    assert command.status is CommandStatus.CANCELLED
    assert "nie cofam" in (command.error or "").casefold()
    await executor.close()


@pytest.mark.asyncio
async def test_assistant_confirm_and_duplicate_transcript(tmp_path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    registry = _registry(tmp_path, notepad_voice_lane=True)
    monkeypatch.setattr(
        "voiceloop.notepad_document.resolve_active_notepad",
        lambda: _snapshot("stare"),
    )
    written: list[str] = []

    def fake_write(text, *, expected=None, expected_source_text=None):
        written.append(text)
        return (
            "Zastąpiłem treść w Notatniku „VoiceLoop demo” i sprawdziłem odczyt.",
            {"verified": True, "verification": "matched", "actual_text": text},
        )

    monkeypatch.setattr("voiceloop.notepad_document.write_active_notepad", fake_write)
    executor = CommandExecutor(
        memory=store,
        actions=registry,
        events=EventBus(),
        queue_limit=4,
    )
    await executor.start()
    assistant = AssistantService(
        memory=store,
        events=EventBus(),
        executor=executor,
        n8n=SimpleNamespace(route=AsyncMock(return_value=None)),
        model_router=SimpleNamespace(),
        screen=object(),
        tts=WindowsTTS(),
        embeddings=None,
        qdrant=None,
        action_definitions=registry.definitions(),
        dedupe_seconds=2.0,
        vector_context_limit=0,
        routing_v2=None,
    )

    first = await assistant.handle(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Wpisz w notatniku: VoiceLoop demo jeden",
            transcript_confidence=0.99,
            managed_voice_turn=True,
        )
    )
    assert first.plan is not None
    assert first.plan.confirmation_required is True
    duplicate = await assistant.handle(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Wpisz w notatniku: VoiceLoop demo jeden",
            transcript_confidence=0.99,
            managed_voice_turn=True,
        )
    )
    assert duplicate.duplicate is True

    blocked = await assistant.handle(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Otwórz przeglądarkę",
            transcript_confidence=0.99,
            managed_voice_turn=True,
        )
    )
    assert blocked.plan is not None
    assert blocked.plan.provider == "voice_allowlist"
    assert not blocked.plan.steps

    confirmed = await assistant.handle(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="potwierdzam",
            transcript_confidence=0.99,
            managed_voice_turn=True,
        )
    )
    assert written == ["VoiceLoop demo jeden"]
    assert "sprawdziłem odczyt" in (confirmed.plan.response_text or "").casefold()
    await executor.close()


@pytest.mark.asyncio
async def test_assistant_cancel_by_voice_without_write(tmp_path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    await store.initialize()
    registry = _registry(tmp_path)
    monkeypatch.setattr(
        "voiceloop.notepad_document.resolve_active_notepad",
        lambda: _snapshot("stare"),
    )
    monkeypatch.setattr(
        "voiceloop.notepad_document.write_active_notepad",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("write")),
    )
    executor = CommandExecutor(
        memory=store,
        actions=registry,
        events=EventBus(),
        queue_limit=4,
    )
    await executor.start()
    assistant = AssistantService(
        memory=store,
        events=EventBus(),
        executor=executor,
        n8n=SimpleNamespace(route=AsyncMock(return_value=None)),
        model_router=SimpleNamespace(),
        screen=object(),
        tts=WindowsTTS(),
        embeddings=None,
        qdrant=None,
        action_definitions=registry.definitions(),
        dedupe_seconds=0,
        vector_context_limit=0,
    )
    await assistant.handle(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Wpisz w notatniku: nie zapisuj",
            transcript_confidence=0.99,
            managed_voice_turn=True,
        )
    )
    cancelled = await assistant.handle(
        CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="anuluj zadanie",
            transcript_confidence=0.99,
            managed_voice_turn=True,
        )
    )
    assert "anulowałam" in (cancelled.plan.response_text or "").casefold()
    await executor.close()


class DeepgramStub:
    def __init__(self) -> None:
        self.started = 0
        self.started_once = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def start_once(self, *, prefix: str = "", timeout_seconds: float = 30.0) -> None:
        self.started_once += 1

    async def start_conversation(self) -> None:
        self.started += 1

    async def wait_until_connected(self, *, timeout_seconds: float = 5.0) -> None:
        return None

    async def stop(self) -> None:
        self.stopped += 1


class TTSHold:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def stop(self) -> None:
        return None


class RecordingAssistant:
    def __init__(self) -> None:
        self.handled: list[str] = []
        self.conversation_active = True
        self.conversation_session_id = "session-1"
        self.interrupt_calls = 0

    async def begin_conversation_session(self, greeting: str) -> None:
        self.conversation_active = True

    async def handle(self, request):
        self.handled.append(request.text or "")
        confirmation = request.text and confirmation_decision(request.text)
        text = "potwierdzono" if confirmation == "confirm" else "przyjeto"
        return CommandAccepted(
            request_id=request.request_id,
            status=CommandStatus.SUCCEEDED,
            plan=CommandPlan(
                request_id=request.request_id,
                intent="voice_confirmation" if confirmation else "write_active_notepad",
                response_text=text,
                confidence=1.0,
            ),
        )

    async def interrupt(self, *, end_conversation: bool = False) -> None:
        self.interrupt_calls += 1


@pytest.mark.asyncio
async def test_deepgram_transcript_reaches_assistant_confirm() -> None:
    assistant = RecordingAssistant()
    tts = TTSHold()
    coordinator = VoiceConversationCoordinator(
        assistant=assistant,  # type: ignore[arg-type]
        deepgram=DeepgramStub(),  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
        events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
        greeting="Cześć",
        cooldown_ms=0,
        barge_in_after_ms=10_000,
    )
    await coordinator.start_conversation()
    await coordinator.handle_transcript("potwierdzam", confidence=0.99)
    assert assistant.handled == ["potwierdzam"]
    assert any("potwierdzono" in item for item in tts.spoken)
