import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voiceloop.models import (
    CommandAccepted,
    CommandPlan,
    CommandStatus,
    TranscriptEnvelopeV1,
)
from voiceloop.voice_conversation import (
    ConversationState,
    VoiceConversationCoordinator,
)


class TTSStub:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._hold = asyncio.Event()
        self._hold.set()
        self.stop_calls = 0

    async def speak(self, text: str) -> None:
        self.spoken.append(text)
        await self._hold.wait()

    async def stop(self) -> None:
        self.stop_calls += 1
        self._hold.set()


class DeepgramStub:
    def __init__(self) -> None:
        self.started = 0
        self.started_once = 0
        self.conversation_started = 0
        self.stopped = 0
        self.ready_waits = 0

    async def start(self) -> None:
        self.started += 1

    async def start_once(self, *, prefix: str = "", timeout_seconds: float = 30.0) -> None:
        self.started_once += 1

    async def start_conversation(self) -> None:
        self.conversation_started += 1

    async def wait_until_connected(self, *, timeout_seconds: float = 5.0) -> None:
        self.ready_waits += 1

    async def stop(self) -> None:
        self.stopped += 1


class AssistantStub:
    def __init__(self) -> None:
        self.greetings: list[str] = []
        self.handled: list[tuple[str, float | None]] = []
        self.handled_session_ids: list[str | None] = []
        self.interrupt_calls = 0
        self.conversation_active = False
        self.conversation_session_id: str | None = None
        self._session_counter = 0

    async def begin_conversation_session(self, greeting: str) -> None:
        self.greetings.append(greeting)
        self.conversation_active = True
        self._session_counter += 1
        self.conversation_session_id = f"session-{self._session_counter}"

    async def handle(self, request):
        self.handled.append((request.text or "", request.transcript_confidence))
        self.handled_session_ids.append(request.interaction_session_id)
        return CommandAccepted(
            request_id=request.request_id,
            status=CommandStatus.SUCCEEDED,
            plan=CommandPlan(
                request_id=request.request_id,
                intent="conversation",
                response_text="ok",
                confidence=1.0,
            ),
        )

    async def interrupt(self, *, end_conversation: bool = False) -> None:
        self.interrupt_calls += 1
        if end_conversation:
            self.conversation_active = False
            self.conversation_session_id = None


def test_conversation_speaks_once_then_listens_once() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć. W czym mogę pomóc?",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )

        result = await coordinator.start_conversation()
        assert result["status"] == "listening_once"
        assert tts.spoken == ["Cześć. W czym mogę pomóc?"]
        assert deepgram.started_once == 1
        assert deepgram.ready_waits == 1
        assert assistant.greetings == ["Cześć. W czym mogę pomóc?"]
        timing_calls = [
            call
            for call in events.publish.await_args_list
            if call.args and call.args[0] == "conversation.timing"
        ]
        assert len(timing_calls) == 1
        timing = timing_calls[0].args[1]
        assert timing["tts_ms"] >= 0
        assert timing["cooldown_ms"] >= 0
        assert timing["deepgram_reconnect_ms"] >= 0

        second = await coordinator.start_conversation()
        assert second["status"] == "busy"
        assert deepgram.started_once == 1

        await coordinator.handle_transcript("otwórz kalendarz", confidence=0.92)
        assert assistant.handled == [("otwórz kalendarz", 0.92)]
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert deepgram.started_once >= 2

        await coordinator.handle_transcript("kolejne", confidence=0.99)
        assert len(assistant.handled) == 2
        assert coordinator.state is ConversationState.LISTENING_ONCE

        next_turn = await coordinator.start_conversation()
        assert next_turn["status"] == "busy"

    asyncio.run(scenario())


def test_hard_stop_blocks_and_interrupts() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )
        await coordinator.start_conversation()
        stopped = await coordinator.hard_stop()
        assert stopped["status"] == "stopped"
        assert assistant.interrupt_calls == 1
        assert tts.spoken == ["Cześć"]
        blocked = await coordinator.start_conversation()
        assert blocked["status"] == "blocked"
        coordinator.clear_block()
        again = await coordinator.start_conversation()
        assert again["status"] == "listening_once"

    asyncio.run(scenario())


def test_interrupt_speech_cuts_and_resumes_listening() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )
        await coordinator.start_conversation()
        started = deepgram.started_once
        result = await coordinator.interrupt_speech()
        assert result["status"] == "listening_once"
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert assistant.conversation_active is True
        assert assistant.interrupt_calls == 1
        assert deepgram.started_once == started + 1

    asyncio.run(scenario())


def test_spoken_stop_skips_model_and_resumes() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )
        await coordinator.start_conversation()
        await coordinator.handle_transcript("stop", confidence=0.99)
        assert assistant.handled == []
        assert assistant.interrupt_calls == 1
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_turn_error_rearms_microphone_instead_of_killing_conversation() -> None:
    class FailingAssistant(AssistantStub):
        async def handle(self, request):
            raise RuntimeError("planner offline")

    async def scenario() -> None:
        assistant = FailingAssistant()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )
        await coordinator.start_conversation()
        started = deepgram.started_once

        await coordinator.handle_transcript("spróbuj odpowiedzieć", confidence=0.99)

        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert deepgram.started_once == started + 1
        events.publish.assert_any_call(
            "conversation.turn_error",
            {"error": "planner offline", "continue": True},
        )

    asyncio.run(scenario())


def test_listen_timeout_rearms_mic_while_listening() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )
        await coordinator.start_conversation()
        started = deepgram.started_once
        assert coordinator.state is ConversationState.LISTENING_ONCE

        await coordinator.handle_listen_timeout(reason="timeout")
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert deepgram.started_once == started + 1
        events.publish.assert_any_call(
            "conversation.listen_timeout",
            {"reason": "timeout", "state": "listening_once"},
        )

        # Timeout during speaking (barge-in) must not steal the turn.
        coordinator.state = ConversationState.SPEAKING
        before = deepgram.started_once
        await coordinator.handle_listen_timeout(reason="timeout")
        assert deepgram.started_once == before

        coordinator._barge_in_armed = True
        coordinator._accept_final = True
        await coordinator.handle_listen_timeout(reason="timeout")
        assert deepgram.started_once == before + 1
        assert coordinator._barge_in_armed is True
        assert coordinator._accept_final is True

    asyncio.run(scenario())


def test_failed_barge_rearm_stops_listener_and_disarms() -> None:
    class FailingReadyDeepgram(DeepgramStub):
        async def wait_until_connected(self, *, timeout_seconds: float = 5.0) -> None:
            self.ready_waits += 1
            raise RuntimeError("not connected")

    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = FailingReadyDeepgram()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10,
        )
        coordinator.state = ConversationState.SPEAKING
        coordinator._barge_in_armed = True
        coordinator._accept_final = True

        await coordinator.handle_listen_timeout(reason="one_shot_error")

        assert deepgram.started_once == 1
        assert deepgram.stopped == 2
        assert coordinator._barge_in_armed is False
        assert coordinator._accept_final is False

    asyncio.run(scenario())


def test_barge_in_after_grace_cuts_speech_and_handles_user() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        tts._hold.clear()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Długie powitanie testowe",
            cooldown_ms=0,
            barge_in_after_ms=50,
        )

        start_task = asyncio.create_task(coordinator.start_conversation())
        for _ in range(20):
            if coordinator._barge_in_armed:
                break
            await asyncio.sleep(0.01)
        assert coordinator.state is ConversationState.SPEAKING
        assert coordinator._barge_in_armed is True

        await coordinator.handle_interim("chcę o coś zapytać")
        assert tts.stop_calls == 0
        await coordinator.handle_interim("stop")
        assert tts.stop_calls >= 1

        await coordinator.handle_transcript("kontynuuj temat", confidence=0.95)
        # Allow speak() to finish after stop unblocked the hold.
        await asyncio.wait_for(start_task, timeout=2.0)

        assert ("kontynuuj temat", 0.95) in assistant.handled
        assert tts.spoken[-1] == "ok"
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_barge_in_ignores_transcript_matching_current_tts() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        tts._hold.clear()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="To jest długie powitanie testowe asystentki.",
            cooldown_ms=0,
            barge_in_after_ms=20,
        )

        start_task = asyncio.create_task(coordinator.start_conversation())
        await asyncio.sleep(0.04)
        await coordinator.handle_interim("długie powitanie testowe")
        starts_before_echo_final = deepgram.started_once
        await coordinator.handle_transcript(
            "długie powitanie testowe asystentki",
            confidence=0.9,
        )

        assert tts.stop_calls == 0
        assert coordinator._pending_barge_in is None
        assert deepgram.started_once == starts_before_echo_final + 1
        tts._hold.set()
        await asyncio.wait_for(start_task, timeout=2.0)
        assert assistant.handled == []
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_echo_filter_catches_short_asymmetric_deepgram_fragments() -> None:
    coordinator = VoiceConversationCoordinator(
        assistant=AssistantStub(),  # type: ignore[arg-type]
        deepgram=DeepgramStub(),  # type: ignore[arg-type]
        tts=TTSStub(),  # type: ignore[arg-type]
        events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
        greeting="Cześć",
    )
    coordinator._current_spoken_text = (
        "Obecnie do analizy prozodii potrzebny jest dźwięk. "
        "Minimalne wymagania określają, kto i kiedy mówi."
    )

    for fragment in (
        "Obecnie.",
        "Do analizy czego?",
        "Minimalne co?",
        "Kto i kiedy co?",
    ):
        assert coordinator._is_likely_tts_echo(fragment) is True

    assert coordinator._is_likely_tts_echo("Chcę teraz otworzyć kalendarz") is False


@pytest.mark.parametrize(
    ("heard", "expected_echo"),
    (
        ("Obecnie.", True),
        ("Do analizy czego?", True),
        ("Minimalne co?", True),
        ("Kto i kiedy co?", True),
        ("analizy prozodii potrzebny jest dźwięk", True),
        ("minimalne wymagania", True),
        ("określają kto i kiedy mówi", True),
        ("prozodii potrzebny", True),
        ("obecnie do analizy", True),
        ("potrzebny jest dźwięk", True),
        ("Chcę teraz otworzyć kalendarz", False),
        ("Nie, zamiast tego otwórz pocztę", False),
        ("Asystencie, sprawdź pogodę w Warszawie", False),
        ("Poczekaj, mam inne pytanie", False),
        ("To nie jest odpowiedź na moje pytanie", False),
        ("Która godzina jest teraz w Tokio", False),
        ("Otwórz WhatsApp i pokaż czat", False),
        ("Przerwij i posłuchaj nowego polecenia", False),
        ("Wyjaśnij poprzedni punkt znacznie prościej", False),
        ("A co z najnowszą wersją Pythona", False),
        ("Zmieniłem zdanie, zamknij wskazane okno", False),
        ("Mam na myśli zupełnie inne okno", False),
        ("Najpierw sprawdź źródła internetowe", False),
        ("Wróć do poprzedniego tematu rozmowy", False),
    ),
)
def test_overlap_replay_separates_tts_echo_from_real_barge_in(
    heard: str,
    expected_echo: bool,
) -> None:
    coordinator = VoiceConversationCoordinator(
        assistant=AssistantStub(),  # type: ignore[arg-type]
        deepgram=DeepgramStub(),  # type: ignore[arg-type]
        tts=TTSStub(),  # type: ignore[arg-type]
        events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
        greeting="Cześć",
        stream_reuse_enabled=True,
        hybrid_barge_in_enabled=True,
        hybrid_barge_in_grace_ms=350,
        barge_in_profile="auto",
    )
    coordinator._current_spoken_text = (
        "Obecnie do analizy prozodii potrzebny jest dźwięk. "
        "Minimalne wymagania określają, kto i kiedy mówi."
    )

    decision = coordinator._echo_decision(heard)

    assert decision.is_echo is expected_echo


def test_clean_overlap_signal_adaptively_reduces_barge_in_grace() -> None:
    coordinator = VoiceConversationCoordinator(
        assistant=AssistantStub(),  # type: ignore[arg-type]
        deepgram=DeepgramStub(),  # type: ignore[arg-type]
        tts=TTSStub(),  # type: ignore[arg-type]
        events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
        greeting="Cześć",
        stream_reuse_enabled=True,
        hybrid_barge_in_enabled=True,
        hybrid_barge_in_grace_ms=350,
        barge_in_profile="auto",
    )
    coordinator._current_spoken_text = "To jest aktualna odpowiedź asystentki."

    decision = coordinator._echo_decision("Otwórz kalendarz i pokaż jutrzejsze spotkania")

    assert decision.is_echo is False
    assert coordinator._effective_barge_in_delay_seconds(decision) <= 0.18


def test_post_tts_echo_tail_is_ignored_but_direct_address_is_accepted() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=TTSStub(),  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Do analizy prozodii potrzebny jest dźwięk.",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )

        await coordinator.start_conversation()
        starts_before_echo = deepgram.started_once
        await coordinator.handle_transcript("Do analizy czego?", confidence=0.9)

        assert assistant.handled == []
        assert deepgram.started_once == starts_before_echo + 1
        assert coordinator.state is ConversationState.LISTENING_ONCE

        await coordinator.handle_transcript(
            "Asystencie, do analizy czego?",
            confidence=0.9,
        )

        assert assistant.handled == [("Asystencie, do analizy czego?", 0.9)]

    asyncio.run(scenario())


def test_barge_listener_is_cleaned_when_readiness_fails() -> None:
    class FailingFirstReadyDeepgram(DeepgramStub):
        async def wait_until_connected(self, *, timeout_seconds: float = 5.0) -> None:
            self.ready_waits += 1
            if self.ready_waits == 1:
                raise RuntimeError("connection failed")

    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = FailingFirstReadyDeepgram()
        tts = TTSStub()
        tts._hold.clear()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Długie powitanie",
            cooldown_ms=0,
            barge_in_after_ms=10,
        )

        start_task = asyncio.create_task(coordinator.start_conversation())
        await asyncio.sleep(0.04)
        assert deepgram.stopped >= 2
        assert coordinator._barge_in_armed is False

        tts._hold.set()
        await asyncio.wait_for(start_task, timeout=2.0)
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_barge_listener_is_cleaned_if_start_is_cancelled_after_creation() -> None:
    class BlockingFirstStartDeepgram(DeepgramStub):
        def __init__(self) -> None:
            super().__init__()
            self.first_started = asyncio.Event()

        async def start_once(
            self,
            *,
            prefix: str = "",
            timeout_seconds: float = 30.0,
        ) -> None:
            self.started_once += 1
            if self.started_once == 1:
                self.first_started.set()
                await asyncio.Event().wait()

    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = BlockingFirstStartDeepgram()
        tts = TTSStub()
        tts._hold.clear()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Długie powitanie",
            cooldown_ms=0,
            barge_in_after_ms=10,
        )

        start_task = asyncio.create_task(coordinator.start_conversation())
        await asyncio.wait_for(deepgram.first_started.wait(), timeout=1.0)
        tts._hold.set()
        await asyncio.wait_for(start_task, timeout=2.0)

        assert deepgram.stopped >= 2
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_pause_requires_confirmation_ignores_speech_and_resumes_by_voice() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )
        await coordinator.start_conversation()

        await coordinator.handle_transcript(
            "Przerwij działanie na 2 minuty",
            confidence=0.99,
        )
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert tts.spoken[-1] == "Czy na pewno wstrzymać działanie na 2 minuty?"
        assert assistant.interrupt_calls == 0

        await coordinator.handle_transcript("Tak, potwierdzam.", confidence=0.99)
        assert coordinator.state is ConversationState.PAUSED
        assert tts.spoken[-1] == "Wstrzymuję działanie na 2 minuty."
        assert assistant.interrupt_calls == 1

        handled_before = list(assistant.handled)
        starts_before = deepgram.started_once
        await coordinator.handle_transcript(
            "rozmawiam teraz z kolegą",
            confidence=0.95,
        )
        assert coordinator.state is ConversationState.PAUSED
        assert assistant.handled == handled_before
        assert deepgram.started_once == starts_before + 1

        await coordinator.handle_transcript("wznów działanie", confidence=0.99)
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert tts.spoken[-1] == "Wznawiam działanie."
        assert coordinator._pause_task is None

    asyncio.run(scenario())


def test_pause_confirmation_and_resume_keep_persistent_deepgram_transport() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            stream_reuse_enabled=True,
            hybrid_barge_in_enabled=True,
            hybrid_barge_in_grace_ms=0,
        )
        await coordinator.start_conversation()

        await coordinator.handle_transcript(
            "Przerwij działanie na 2 minuty",
            confidence=0.99,
        )
        await coordinator.handle_transcript("potwierdź", confidence=0.99)
        await coordinator.handle_transcript("rozmowa w tle", confidence=0.80)
        await coordinator.handle_transcript("wznów działanie", confidence=0.99)

        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert deepgram.started_once == 0
        assert deepgram.stopped == 0
        assert tts.spoken[-1] == "Wznawiam działanie."

    asyncio.run(scenario())


def test_pause_command_is_handled_outside_managed_conversation() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )

        assert coordinator.should_handle_control_transcript(
            "Przerwij działanie na 5 minut"
        )
        await coordinator.handle_transcript(
            "Przerwij działanie na 5 minut",
            confidence=0.98,
        )

        assert assistant.conversation_active is True
        assert assistant.handled == []
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert tts.spoken == ["Czy na pewno wstrzymać działanie na 5 minut?"]

    asyncio.run(scenario())


def test_barge_from_control_prompt_can_speak_new_turn_reply() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10,
        )
        await coordinator.start_conversation()
        tts._hold.clear()

        control_task = asyncio.create_task(
            coordinator.handle_transcript(
                "Przerwij działanie na 25 godzin",
                confidence=0.99,
            )
        )
        for _ in range(20):
            if coordinator._barge_in_armed:
                break
            await asyncio.sleep(0.01)
        assert coordinator._barge_in_armed is True

        await coordinator.handle_transcript("opowiedz żart", confidence=0.97)
        await asyncio.wait_for(control_task, timeout=2.0)

        assert assistant.handled == [("opowiedz żart", 0.97)]
        assert tts.spoken[-1] == "ok"
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_multi_speaker_transcript_requires_direct_address() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
            ignore_multi_speaker=True,
        )
        await coordinator.start_conversation()

        await coordinator.handle_transcript(
            "rozmawiamy o projekcie",
            confidence=0.9,
            speaker_ids=(0, 1),
        )
        assert assistant.handled == []
        assert coordinator.state is ConversationState.LISTENING_ONCE

        await coordinator.handle_transcript(
            "Asystencie powiedz która godzina",
            confidence=0.9,
            speaker_ids=(0, 1),
        )
        assert assistant.handled == [("Asystencie powiedz która godzina", 0.9)]

    asyncio.run(scenario())


def test_idle_guard_requires_direct_address_after_followup_window() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
            direct_address_after_seconds=0,
        )
        await coordinator.start_conversation()
        coordinator._address_free_until = time.monotonic() - 1

        await coordinator.handle_transcript("rozmawiam z kolegą", confidence=0.9)
        assert assistant.handled == []
        assert coordinator.state is ConversationState.LISTENING_ONCE

        await coordinator.handle_transcript(
            "Asystencie opowiedz żart",
            confidence=0.95,
        )
        assert assistant.handled == [("Asystencie opowiedz żart", 0.95)]

    asyncio.run(scenario())


def test_pause_parser_supports_polish_time_units() -> None:
    second = VoiceConversationCoordinator._parse_pause_request(
        "Przerwij działanie na 1 sekundę"
    )
    minute = VoiceConversationCoordinator._parse_pause_request(
        "wstrzymaj pracę na 3 minuty"
    )
    hour = VoiceConversationCoordinator._parse_pause_request(
        "zatrzymaj asystenta na 5 godzin"
    )
    spoken_number = VoiceConversationCoordinator._parse_pause_request(
        "przerwij działanie na dwadzieścia dwie minuty"
    )

    assert second is not None and (second.seconds, second.display) == (1, "1 sekundę")
    assert minute is not None and (minute.seconds, minute.display) == (180, "3 minuty")
    assert hour is not None and (hour.seconds, hour.display) == (18_000, "5 godzin")
    assert spoken_number is not None
    assert (spoken_number.seconds, spoken_number.display) == (1320, "22 minuty")


def test_direct_address_requires_complete_wake_word() -> None:
    assert VoiceConversationCoordinator._is_direct_address(
        "Asystencie, opowiedz żart"
    )
    assert VoiceConversationCoordinator._is_direct_address("Venice powiedz więcej")
    assert not VoiceConversationCoordinator._is_direct_address(
        "Rozmawiam z asystentem kolegi"
    )


def test_transcribe_only_keeps_listening_without_assistant() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
        )

        result = await coordinator.start_transcribe_only()
        assert result["status"] == "transcribe_only"
        assert coordinator.should_route_to_assistant() is False
        assert coordinator.should_handle_control_transcript("stop") is False
        assert deepgram.started == 1
        assert assistant.handled == []

        await coordinator.handle_transcript("Asystencie, która godzina?", confidence=0.99)
        await coordinator.handle_interim("Asystencie")
        interrupted = await coordinator.interrupt_speech()
        stopped = await coordinator.hard_stop()

        assert assistant.handled == []
        assert tts.spoken == []
        assert interrupted["status"] == "transcribe_only"
        assert stopped["status"] == "transcribe_only"
        assert coordinator.health() == (True, "transcribe_only")

    asyncio.run(scenario())


def test_conversation_turns_share_session_id_and_reset_between_sessions() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        events = SimpleNamespace(publish=AsyncMock())
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            greeting="Cześć",
            cooldown_ms=0,
            barge_in_after_ms=10_000,
        )

        await coordinator.start_conversation()
        first_session_id = assistant.conversation_session_id
        assert first_session_id is not None
        await coordinator.handle_transcript("otwórz kalendarz", confidence=0.92)
        await coordinator.handle_transcript("otwórz chrome", confidence=0.91)
        assert assistant.handled_session_ids[-2:] == [first_session_id, first_session_id]

        await coordinator.hard_stop()
        assert assistant.conversation_session_id is None
        coordinator.clear_block()
        await coordinator.start_conversation()
        second_session_id = assistant.conversation_session_id
        assert second_session_id is not None
        assert second_session_id != first_session_id
        await coordinator.handle_transcript("otwórz youtube", confidence=0.93)
        assert assistant.handled_session_ids[-1] == second_session_id

    asyncio.run(scenario())


def test_persistent_stream_buffers_early_final_without_losing_first_words() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        tts._hold.clear()
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
            greeting="Dłuższe powitanie",
            cooldown_ms=0,
            stream_reuse_enabled=True,
            hybrid_barge_in_enabled=True,
            hybrid_barge_in_grace_ms=80,
        )

        start_task = asyncio.create_task(coordinator.start_conversation())
        for _ in range(20):
            if coordinator.state is ConversationState.SPEAKING:
                break
            await asyncio.sleep(0.005)
        await coordinator.handle_speech_started()
        await coordinator.handle_transcript(
            "Pierwsze słowo zostaje zachowane",
            confidence=0.96,
        )

        await asyncio.wait_for(start_task, timeout=2.0)
        assert assistant.handled[0] == ("Pierwsze słowo zostaje zachowane", 0.96)
        assert deepgram.started_once == 0
        assert deepgram.stopped == 0
        assert coordinator.state is ConversationState.LISTENING_ONCE

    asyncio.run(scenario())


def test_hybrid_stable_interim_cuts_tts_and_pause_alias_is_immediate() -> None:
    async def scenario() -> None:
        assistant = AssistantStub()
        deepgram = DeepgramStub()
        tts = TTSStub()
        tts._hold.clear()
        coordinator = VoiceConversationCoordinator(
            assistant=assistant,  # type: ignore[arg-type]
            deepgram=deepgram,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
            greeting="To jest długa odpowiedź testowa",
            cooldown_ms=0,
            stream_reuse_enabled=True,
            hybrid_barge_in_enabled=True,
            hybrid_barge_in_grace_ms=0,
            barge_in_stability_ms=50,
        )

        start_task = asyncio.create_task(coordinator.start_conversation())
        for _ in range(20):
            if coordinator.state is ConversationState.SPEAKING:
                break
            await asyncio.sleep(0.005)
        await coordinator.handle_interim("chcę otworzyć kalendarz")
        await asyncio.sleep(0.06)
        await coordinator.handle_interim("chcę otworzyć kalendarz teraz")
        await asyncio.wait_for(start_task, timeout=2.0)

        assert tts.stop_calls >= 1
        assert coordinator.state is ConversationState.LISTENING_ONCE
        assert VoiceConversationCoordinator._is_interrupt_phrase("pauza")
        assert VoiceConversationCoordinator._is_interrupt_phrase("poczekaj")
        assert VoiceConversationCoordinator._is_interrupt_phrase("chwila")

    asyncio.run(scenario())


def test_real_transcript_replay_preserves_first_words_in_hybrid_path() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "corpus"
        / "eval"
        / "voice-v1"
        / "transcripts-v1.jsonl"
    )
    if not path.is_file():
        pytest.skip("Lokalny prywatny replay głosowy nie jest dostępny.")
    envelopes = [
        TranscriptEnvelopeV1.model_validate(json.loads(line)["envelope"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    envelopes = [
        envelope
        for envelope in envelopes
        if envelope.normalized_text.strip()
        and not VoiceConversationCoordinator._is_interrupt_phrase(
            envelope.normalized_text
        )
    ][:20]
    if len(envelopes) < 20:
        pytest.skip("Replay wymaga co najmniej 20 realnych transkryptów.")

    async def scenario() -> None:
        legacy = VoiceConversationCoordinator(
            assistant=AssistantStub(),  # type: ignore[arg-type]
            deepgram=DeepgramStub(),  # type: ignore[arg-type]
            tts=TTSStub(),  # type: ignore[arg-type]
            events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
            greeting="Cześć",
            stream_reuse_enabled=False,
            hybrid_barge_in_enabled=False,
        )
        hybrid = VoiceConversationCoordinator(
            assistant=AssistantStub(),  # type: ignore[arg-type]
            deepgram=DeepgramStub(),  # type: ignore[arg-type]
            tts=TTSStub(),  # type: ignore[arg-type]
            events=SimpleNamespace(publish=AsyncMock()),  # type: ignore[arg-type]
            greeting="Cześć",
            stream_reuse_enabled=True,
            hybrid_barge_in_enabled=True,
            hybrid_barge_in_grace_ms=350,
        )
        legacy.state = ConversationState.SPEAKING
        hybrid.state = ConversationState.SPEAKING
        hybrid._current_spoken_text = "Alfa beta gamma delta epsilon."

        assert legacy.accepts_speaking_transcript() is False
        preserved = 0
        for envelope in envelopes:
            hybrid.state = ConversationState.SPEAKING
            hybrid._barge_in_armed = False
            hybrid._pending_barge_in = None
            await hybrid.handle_transcript(
                envelope.normalized_text,
                confidence=envelope.confidence_mean,
                speaker_ids=envelope.speaker_ids,
                transcript=envelope,
            )
            pending = hybrid._pending_barge_in
            assert pending is not None
            assert pending[0].split()[0] == envelope.normalized_text.split()[0]
            preserved += 1

        assert preserved == 20

    asyncio.run(scenario())
