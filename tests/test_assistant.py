import asyncio
from types import SimpleNamespace

from voiceloop.assistant import VOICE_RESULT_ACTIONS, AssistantService
from voiceloop.capability_index import CapabilityIndexError
from voiceloop.events import EventBus
from voiceloop.memory import MemoryStore
from voiceloop.models import (
    CommandPlan,
    CommandRequest,
    CommandSource,
    CommandStatus,
    PlanStep,
    TranscriptEnvelopeV1,
)
from voiceloop.routing.assembler import AssemblyResult, clarification_plan
from voiceloop.routing.segmenter import segment_command
from voiceloop.routing.service import RoutingV2Outcome


class ExecutorStub:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.stop_calls = 0

    async def submit(self, plan: CommandPlan):
        return await self.memory.update_command(
            plan.request_id,
            status=CommandStatus.SUCCEEDED,
            plan=plan,
            results=[],
        )

    async def stop_all(self) -> None:
        self.stop_calls += 1


class TTSStub:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stop_calls = 0

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def stop(self) -> None:
        self.stop_calls += 1


class N8nStub:
    def __init__(self) -> None:
        self.calls = 0

    async def route(self, request: CommandRequest):
        self.calls += 1


class RouterStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def plan(self, **kwargs) -> CommandPlan:
        self.calls.append(kwargs)
        request = kwargs["request"]
        return CommandPlan(
            request_id=request.request_id,
            intent="conversation",
            response_text=f"odpowiedz-{len(self.calls)}",
            confidence=1.0,
            provider="venice",
        )


class BlockingRouterStub:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def plan(self, **kwargs) -> CommandPlan:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _assistant(
    memory: MemoryStore,
    *,
    router,
    n8n: N8nStub,
    executor: ExecutorStub,
    tts: TTSStub,
    routing_v2=None,
    embeddings=None,
    qdrant=None,
    qdrant_shadow=None,
    vector_context_limit: int = 0,
) -> AssistantService:
    return AssistantService(
        memory=memory,
        events=EventBus(),
        executor=executor,  # type: ignore[arg-type]
        n8n=n8n,  # type: ignore[arg-type]
        model_router=router,  # type: ignore[arg-type]
        screen=object(),  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
        embeddings=embeddings,
        qdrant=qdrant,
        qdrant_shadow=qdrant_shadow,
        action_definitions=[],
        dedupe_seconds=0,
        vector_context_limit=vector_context_limit,
        routing_v2=routing_v2,
    )


def test_vector_memory_query_uses_five_distinct_spaces(tmp_path) -> None:
    class EmbeddingsStub:
        enabled = True

        def __init__(self) -> None:
            self.documents: list[str] = []

        def accepts_private_text(self) -> bool:
            return True

        async def embed_queries(self, documents):
            self.documents = list(documents)
            return [
                [float(index), 1.0, 0.0]
                for index, _document in enumerate(documents, start=1)
            ]

    class QdrantStub:
        enabled = True

        def __init__(self, collection_name: str, source_id: str) -> None:
            self.collection_name = collection_name
            self.source_id = source_id
            self.kwargs = {}

        def accepts_private_data(self) -> bool:
            return True

        async def search(self, **kwargs):
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    source="screenpipe_meeting",
                    source_id=self.source_id,
                    title="Ustalenia",
                    content="Ustalono wdrożenie weighted RRF.",
                    score=0.9,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-08-18"),
                    metadata={
                        "provenance": {
                            "time": "2026-08-18T12:00:00+00:00",
                            "confidence": 0.91,
                        },
                        "retrieval_evidence": {
                            "spaces": {
                                "semantic": {"rank": 1},
                                "decision": {"rank": 1},
                            }
                        },
                    },
                )
            ]

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        embeddings = EmbeddingsStub()
        qdrant = QdrantStub("voiceloop_memory", "meeting:7")
        qdrant_shadow = QdrantStub("voiceloop_memory_v2", "meeting:8")
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            embeddings=embeddings,
            qdrant=qdrant,
            qdrant_shadow=qdrant_shadow,
            vector_context_limit=5,
        )
        events = assistant.events.subscribe()
        next_event = asyncio.create_task(anext(events))
        await asyncio.sleep(0)

        contexts = await assistant._vector_memories_for_request(
            CommandRequest(text="Jaką decyzję ustaliliśmy w sprawie Qdrant?")
        )
        shadow_event = await asyncio.wait_for(next_event, timeout=1.0)
        await events.aclose()

        assert len(embeddings.documents) == 5
        assert len(set(embeddings.documents)) == 5
        assert set(qdrant.kwargs["query_vectors"]) == {
            "semantic",
            "topic",
            "intent",
            "decision",
            "person_context",
        }
        assert (
            qdrant.kwargs["query_weights"]["decision"]
            > qdrant.kwargs["query_weights"]["intent"]
        )
        assert "source_id=meeting:7" in contexts[0]
        assert "spaces=decision,semantic" in contexts[0]
        assert shadow_event["type"] == "memory.retrieval_shadow"
        assert shadow_event["payload"]["active_source_ids"] == ["meeting:7"]
        assert shadow_event["payload"]["shadow_source_ids"] == ["meeting:8"]
        assert shadow_event["payload"]["top1_match"] is False
        await assistant.close()

    asyncio.run(scenario())


def test_copy_actions_speak_only_their_execution_result() -> None:
    assert {
        "close_window_under_cursor",
        "copy_email_under_cursor",
        "copy_number_under_cursor",
        "copy_selected_text",
        "copy_sentence_under_cursor",
        "copy_text_under_cursor",
        "select_paragraph_under_cursor",
        "select_sentence_under_cursor",
    }.issubset(VOICE_RESULT_ACTIONS)


def test_venice_session_keeps_context_without_repeating_wake_word(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        executor = ExecutorStub(memory)
        tts = TTSStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=executor,
            tts=tts,
        )

        await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="Venice opowiedz o Qdrant",
            )
        )
        await asyncio.sleep(0)
        await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="A jak przechowuje kontekst?",
            )
        )
        await asyncio.sleep(0)

        assert assistant._conversation_active is True
        assert n8n.calls == 0
        assert router.calls[0]["history"] == []
        assert router.calls[0]["conversation_active"] is True
        assert router.calls[0]["conversation_style_override"] == "max_iq"
        assert router.calls[1]["history"] == [
            {"role": "user", "content": "Venice opowiedz o Qdrant"},
            {"role": "assistant", "content": "odpowiedz-1"},
        ]

        await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="koniec rozmowy",
            )
        )
        await asyncio.sleep(0)

        assert assistant._conversation_active is False
        assert assistant._conversation_history == []
        assert len(router.calls) == 2
        await assistant.close()

    asyncio.run(scenario())


def test_stop_cancels_model_generation_but_keeps_session_open(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = BlockingRouterStub()
        n8n = N8nStub()
        executor = ExecutorStub(memory)
        tts = TTSStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=executor,
            tts=tts,
        )
        conversation_request = CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Venice wyjaśnij szczegółowo pamięć",
        )
        generation = asyncio.create_task(assistant.handle(conversation_request))
        await router.started.wait()

        stopped = await assistant.handle(
            CommandRequest(source=CommandSource.DEEPGRAM, text="przerwij")
        )
        result = await asyncio.gather(generation, return_exceptions=True)
        command = await memory.get_command(conversation_request.request_id)

        assert isinstance(result[0], asyncio.CancelledError)
        assert command is not None
        assert command.status is CommandStatus.CANCELLED
        assert stopped.status is CommandStatus.SUCCEEDED
        assert executor.stop_calls >= 1
        assert tts.stop_calls >= 1
        assert assistant._conversation_active is True
        await assistant.close()

    asyncio.run(scenario())


def test_paraphrases_with_same_action_plan_are_deduplicated(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        executor = ExecutorStub(memory)
        tts = TTSStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=executor,
            tts=tts,
        )
        first_request = CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Kopiuj numer",
            transcript_confidence=0.95,
        )
        second_request = CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Skopiuj numer pod kursorem",
            transcript_confidence=0.95,
        )

        first = await assistant.handle(first_request)
        second = await assistant.handle(second_request)
        rejected_duplicate = await memory.get_command(second_request.request_id)

        assert first.duplicate is False
        assert second.duplicate is True
        assert second.request_id == first.request_id
        assert rejected_duplicate is not None
        assert rejected_duplicate.status is CommandStatus.REJECTED
        assert rejected_duplicate.error
        assert "Duplikat" in rejected_duplicate.error or "duplikat" in rejected_duplicate.error
        await assistant.close()

    asyncio.run(scenario())


def test_low_stt_confidence_blocks_deterministic_voice_action(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        executor = ExecutorStub(memory)
        tts = TTSStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=executor,
            tts=tts,
        )
        assistant.configure_stt_threshold(0.75)
        accepted = await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="otwórz kalendarz",
                transcript_confidence=0.2,
            )
        )
        await asyncio.sleep(0)
        assert accepted.plan is not None
        assert accepted.plan.provider == "stt_confidence_gate"
        assert accepted.plan.requires_clarification is True
        assert accepted.plan.steps == []
        assert tts.spoken
        spoken = tts.spoken[0].casefold()
        assert "polecenie" in spoken or "pewna" in spoken or "usłyszałam" in spoken
        await assistant.close()

    asyncio.run(scenario())


def test_missing_stt_confidence_blocks_deterministic_voice_action(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
        )

        accepted = await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="otwórz kalendarz",
            )
        )

        assert accepted.plan is not None
        assert accepted.plan.provider == "stt_confidence_gate"
        assert accepted.plan.steps == []
        assert router.calls == []
        await assistant.close()

    asyncio.run(scenario())


def test_multi_speaker_command_is_blocked_before_planning(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        executor = ExecutorStub(memory)
        tts = TTSStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=executor,
            tts=tts,
        )
        envelope = TranscriptEnvelopeV1.from_text(
            "otwórz kalendarz",
            confidence=0.96,
            speaker_ids=(0, 1),
        )

        accepted = await assistant.handle(
            CommandRequest.from_transcript(
                envelope,
                managed_voice_turn=True,
            )
        )

        assert accepted.plan is not None
        assert accepted.plan.provider == "speaker_gate"
        assert accepted.plan.requires_clarification is True
        assert accepted.plan.steps == []
        assert router.calls == []
        await assistant.close()

    asyncio.run(scenario())


def test_shadow_routing_never_replaces_legacy_plan(tmp_path) -> None:
    class RoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=True,
            routing_v2_execute=False,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = False

        async def evaluate(self, request):
            candidate_plan = CommandPlan(
                request_id=request.request_id,
                intent="task",
                response_text="shadow",
                confidence=0.95,
                steps=[
                    PlanStep(action_id="open_browser"),
                    PlanStep(action_id="open_url", args={"url": "https://www.youtube.com"}),
                ],
                provider="routing_v2",
            )
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(candidate_plan),
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {
                "request_id": request.request_id,
                "mode": "shadow",
                "v2_action_ids": [step.action_id for step in outcome.plan.steps],
                "v1_action_ids": [step.action_id for step in legacy_plan.steps]
                if legacy_plan
                else [],
            }

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=RoutingV2Stub(),
        )

        accepted = await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="otwórz Chrome i YouTube",
                transcript_confidence=0.96,
            )
        )

        assert accepted.plan is not None
        assert accepted.plan.provider == "compound_fast_path_guard"
        assert accepted.plan.steps == []
        await assistant.close()

    asyncio.run(scenario())


def test_canary_routing_keeps_legacy_plan_for_non_allowlisted_action(tmp_path) -> None:
    class CanaryRoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute=True,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = True
        live_execution_requested = True

        async def evaluate(self, request):
            candidate_plan = CommandPlan(
                request_id=request.request_id,
                intent="task",
                steps=[PlanStep(action_id="open_browser")],
                provider="routing_v2",
            )
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(candidate_plan),
            )

        def plan_execution_allowed(self, _plan):
            return False

        def activation_guard_plan(self, _request, _outcome):
            return None

        def shadow_payload(self, request, _outcome, *, legacy_plan):
            return {
                "request_id": request.request_id,
                "mode": "canary",
                "v1_action_ids": [step.action_id for step in legacy_plan.steps],
            }

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=CanaryRoutingV2Stub(),
        )

        accepted = await assistant.handle(CommandRequest(text="otwórz kalendarz"))

        assert accepted.plan is not None
        assert [step.action_id for step in accepted.plan.steps] == ["open_calendar"]
        assert accepted.plan.provider == "deterministic"
        await assistant.close()

    asyncio.run(scenario())


def test_live_routing_never_replaces_protected_capabilities_plan(tmp_path) -> None:
    class LiveRoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute=True,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = True

        async def evaluate(self, request):
            candidate_plan = CommandPlan(
                request_id=request.request_id,
                intent="task",
                response_text="Nieprawidłowy plan V2",
                confidence=0.99,
                steps=[PlanStep(action_id="open_browser")],
                provider="routing_v2",
            )
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(candidate_plan),
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "execute"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=LiveRoutingV2Stub(),
        )

        accepted = await assistant.handle(CommandRequest(text="co potrafisz"))

        assert accepted.plan is not None
        assert accepted.plan.intent == "list_capabilities"
        assert [step.action_id for step in accepted.plan.steps] == ["list_capabilities"]
        assert accepted.plan.provider == "deterministic"
        assert router.calls == []
        await assistant.close()

    asyncio.run(scenario())


def test_live_routing_cannot_bypass_speaker_gate(tmp_path) -> None:
    class LiveRoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute=True,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = True

        async def evaluate(self, request):
            candidate_plan = CommandPlan(
                request_id=request.request_id,
                intent="task",
                response_text="live",
                confidence=0.99,
                steps=[PlanStep(action_id="open_browser")],
                provider="routing_v2",
            )
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(candidate_plan),
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "execute"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=LiveRoutingV2Stub(),
        )

        accepted = await assistant.handle(
            CommandRequest.from_transcript(
                TranscriptEnvelopeV1.from_text(
                    "otwórz Chrome",
                    confidence=0.98,
                    speaker_ids=(0, 1),
                )
            )
        )

        assert accepted.plan is not None
        assert accepted.plan.provider == "speaker_gate"
        assert accepted.plan.steps == []
        await assistant.close()

    asyncio.run(scenario())


def test_voiceattack_command_id_keeps_fast_path_during_conversation(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=ExecutorStub(memory),
            tts=TTSStub(),
        )
        assistant._conversation_active = True

        accepted = await assistant.handle(
            CommandRequest(
                source=CommandSource.VOICEATTACK,
                command_id="open_browser",
            )
        )

        assert accepted.plan is not None
        assert accepted.plan.provider == "deterministic"
        assert [step.action_id for step in accepted.plan.steps] == ["open_browser"]
        assert router.calls == []
        assert n8n.calls == 0
        await assistant.close()

    asyncio.run(scenario())


def test_live_routing_block_is_final_and_never_falls_back(tmp_path) -> None:
    class BlockedRoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute=True,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = True
        live_execution_requested = True

        async def evaluate(self, request):
            reason = "clarify:low_top2_margin"
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(
                    clarification_plan(request, reason=reason),
                    reason,
                ),
                blocked_reason=reason,
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "execute"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=BlockedRoutingV2Stub(),
        )
        assistant._conversation_active = True

        accepted = await assistant.handle(CommandRequest(text="otwórz Chrome"))

        assert accepted.plan is not None
        assert accepted.plan.provider == "routing_v2_guard"
        assert accepted.plan.requires_clarification is True
        assert accepted.plan.steps == []
        assert router.calls == []
        assert n8n.calls == 0
        await assistant.close()

    asyncio.run(scenario())


def test_requested_live_mode_with_invalid_gate_blocks_legacy(tmp_path) -> None:
    class InvalidGateRoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute=True,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = False
        live_execution_requested = True

        async def evaluate(self, request):
            candidate_plan = CommandPlan(
                request_id=request.request_id,
                intent="task",
                steps=[PlanStep(action_id="open_browser")],
                provider="routing_v2",
            )
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(candidate_plan),
            )

        def activation_guard_plan(self, request, _outcome):
            return clarification_plan(
                request,
                reason="quality_gate:runtime_config_mismatch",
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "shadow"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        n8n = N8nStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=n8n,
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=InvalidGateRoutingV2Stub(),
        )

        accepted = await assistant.handle(CommandRequest(text="otwórz Chrome"))

        assert accepted.plan is not None
        assert accepted.plan.provider == "routing_v2_guard"
        assert accepted.plan.steps == []
        assert router.calls == []
        assert n8n.calls == 0
        await assistant.close()

    asyncio.run(scenario())


def test_live_routing_unavailability_fails_closed_for_commands(tmp_path) -> None:
    class UnavailableRoutingV2Stub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=False,
            routing_v2_execute=True,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = True
        live_execution_requested = True

        async def evaluate(self, _request):
            raise CapabilityIndexError("offline")

        def unavailable_outcome(self, request, *, reason):
            blocked_reason = f"routing_unavailable:{reason}"
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(
                    clarification_plan(request, reason=blocked_reason),
                    blocked_reason,
                ),
                blocked_reason=blocked_reason,
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "execute"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=UnavailableRoutingV2Stub(),
        )

        accepted = await assistant.handle(CommandRequest(text="otwórz Chrome"))

        assert accepted.plan is not None
        assert accepted.plan.provider == "routing_v2_guard"
        assert accepted.plan.steps == []
        assert router.calls == []
        await assistant.close()

    asyncio.run(scenario())


def test_begin_conversation_session_seeds_history(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
        )
        await assistant.begin_conversation_session("Cześć. W czym mogę pomóc?")
        assert assistant._conversation_active is True
        assert assistant._conversation_history == [
            {"role": "assistant", "content": "Cześć. W czym mogę pomóc?"}
        ]
        assert assistant.conversation_session_id is not None
        first_session_id = assistant.conversation_session_id
        await assistant.interrupt(end_conversation=True)
        assert assistant._conversation_active is False
        assert assistant.conversation_session_id is None
        await assistant.begin_conversation_session("Cześć ponownie.")
        assert assistant.conversation_session_id is not None
        assert assistant.conversation_session_id != first_session_id
        await assistant.close()

    asyncio.run(scenario())


def test_command_request_normalizes_interaction_session_id_whitespace() -> None:
    empty = CommandRequest(source=CommandSource.API, text="test", interaction_session_id="")
    whitespace = CommandRequest(source=CommandSource.API, text="test", interaction_session_id="   ")
    padded = CommandRequest(
        source=CommandSource.API,
        text="test",
        interaction_session_id="  external-id  ",
    )

    assert empty.interaction_session_id is None
    assert whitespace.interaction_session_id is None
    assert padded.interaction_session_id == "external-id"


def test_assistant_attaches_interaction_session_id_on_start_and_active_turns(tmp_path) -> None:
    class RoutingV2SessionStub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=True,
            routing_v2_execute=False,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = False
        seen_session_ids: list[str | None]

        def __init__(self) -> None:
            self.seen_session_ids = []

        async def evaluate(self, request):
            self.seen_session_ids.append(request.interaction_session_id)
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(None),
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "shadow"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        router = RouterStub()
        routing_stub = RoutingV2SessionStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=routing_stub,
        )
        explicit_start_id = "external-start-session"
        start = await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="Venice opowiedz krótko",
                interaction_session_id=explicit_start_id,
            )
        )
        assert start.plan is not None
        first_session_id = assistant.conversation_session_id
        assert first_session_id is not None
        assert first_session_id == explicit_start_id
        assert routing_stub.seen_session_ids[0] == explicit_start_id

        await assistant.handle(
            CommandRequest(
                source=CommandSource.PANEL,
                text="A teraz dodaj przykład",
                interaction_session_id=None,
            )
        )
        assert routing_stub.seen_session_ids[1] == explicit_start_id

        external_during_active = "external-during-active"
        await assistant.handle(
            CommandRequest(
                source=CommandSource.API,
                text="otwórz Chrome",
                interaction_session_id=external_during_active,
            )
        )
        assert routing_stub.seen_session_ids[2] == external_during_active
        assert assistant.conversation_session_id == explicit_start_id

        await assistant.interrupt(end_conversation=True)
        assert assistant.conversation_session_id is None

        explicit = "external-session-id"
        await assistant.handle(
            CommandRequest(
                source=CommandSource.API,
                text="otwórz youtube",
                interaction_session_id=explicit,
            )
        )
        assert routing_stub.seen_session_ids[3] == explicit
        await assistant.close()

    asyncio.run(scenario())


def test_assistant_uses_managed_id_when_whitespace_session_id_provided(tmp_path) -> None:
    class RoutingV2SessionStub:
        settings = SimpleNamespace(
            routing_v2_enabled=True,
            routing_v2_shadow_mode=True,
            routing_v2_execute=False,
            routing_v2_shadow_timeout_seconds=1.0,
        )
        execution_enabled = False
        seen_session_ids: list[str | None]

        def __init__(self) -> None:
            self.seen_session_ids = []

        async def evaluate(self, request):
            self.seen_session_ids.append(request.interaction_session_id)
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(None),
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {"request_id": request.request_id, "mode": "shadow"}

    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        routing_stub = RoutingV2SessionStub()
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
            routing_v2=routing_stub,
        )
        await assistant.handle(
            CommandRequest(
                source=CommandSource.DEEPGRAM,
                text="Venice opowiedz krótko",
                interaction_session_id="   ",
            )
        )
        first_managed_id = assistant.conversation_session_id
        assert first_managed_id is not None
        assert first_managed_id.strip() == first_managed_id
        assert routing_stub.seen_session_ids[0] == first_managed_id

        await assistant.handle(
            CommandRequest(
                source=CommandSource.PANEL,
                text="doprecyzuj proszę",
                interaction_session_id=" ",
            )
        )
        assert routing_stub.seen_session_ids[1] == first_managed_id
        await assistant.close()

    asyncio.run(scenario())


def test_assistant_keeps_execution_plan_identical_with_report_only_shadow(tmp_path) -> None:
    class RoutingStub:
        def __init__(self, calibration_status: str) -> None:
            self.calibration_status = calibration_status
            self.settings = SimpleNamespace(
                routing_v2_enabled=True,
                routing_v2_shadow_mode=True,
                routing_v2_execute=False,
                routing_v2_shadow_timeout_seconds=1.0,
            )
            self.execution_enabled = False
            self.live_execution_requested = False

        async def evaluate(self, request):
            candidate_plan = CommandPlan(
                request_id=request.request_id,
                intent="task",
                steps=[PlanStep(action_id="open_browser")],
                provider="routing_v2",
            )
            return RoutingV2Outcome(
                segmentation=segment_command(request.text or ""),
                assembly=AssemblyResult(candidate_plan),
            )

        def shadow_payload(self, request, outcome, *, legacy_plan):
            return {
                "request_id": request.request_id,
                "mode": "shadow",
                "calibration": {"status": self.calibration_status},
            }

    async def scenario() -> None:
        memory_off = MemoryStore(tmp_path / "off.db")
        memory_report = MemoryStore(tmp_path / "report.db")
        await memory_off.initialize()
        await memory_report.initialize()
        request = CommandRequest(source=CommandSource.API, text="otwórz kalendarz")

        assistant_off = _assistant(
            memory_off,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory_off),
            tts=TTSStub(),
            routing_v2=RoutingStub("off"),
        )
        assistant_report = _assistant(
            memory_report,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory_report),
            tts=TTSStub(),
            routing_v2=RoutingStub("ready"),
        )
        accepted_off = await assistant_off.handle(request.model_copy())
        accepted_report = await assistant_report.handle(request.model_copy())
        assert accepted_off.plan is not None and accepted_report.plan is not None
        assert [step.action_id for step in accepted_off.plan.steps] == [
            step.action_id for step in accepted_report.plan.steps
        ]
        assert accepted_off.plan.provider == accepted_report.plan.provider
        await assistant_off.close()
        await assistant_report.close()

    asyncio.run(scenario())


def test_conversation_context_sources_start_in_parallel(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        assistant = _assistant(
            memory,
            router=RouterStub(),
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
        )
        all_started = asyncio.Event()
        release = asyncio.Event()
        started: set[str] = set()

        async def gated(name: str, result):
            started.add(name)
            if len(started) == 3:
                all_started.set()
            await release.wait()
            return result

        class ScreenStub:
            async def capture(self, _request_id):
                return await gated("screen", None)

            @staticmethod
            def image_data_url(_snapshot):
                return None

        class KnowledgeStub:
            @staticmethod
            def should_search(_text):
                return True

            async def lookup(self, **_kwargs):
                return await gated(
                    "knowledge",
                    SimpleNamespace(observations=(), error=None),
                )

        async def vector_context(_request):
            return await gated("memory", [])

        assistant.screen = ScreenStub()  # type: ignore[assignment]
        assistant.knowledge_tools = KnowledgeStub()  # type: ignore[assignment]
        assistant._vector_memories_for_request = vector_context  # type: ignore[method-assign]
        request = CommandRequest(
            source=CommandSource.DEEPGRAM,
            text="Jaka jest dziś pogoda?",
            include_screen=True,
        )
        task = asyncio.create_task(
            assistant._create_plan(
                request,
                None,
                conversation_active=True,
            )
        )
        try:
            await asyncio.wait_for(all_started.wait(), timeout=0.5)
        finally:
            release.set()
        plan = await asyncio.wait_for(task, timeout=1.0)

        assert started == {"screen", "knowledge", "memory"}
        assert plan.intent == "conversation"

    asyncio.run(scenario())


def test_n8n_client_when_disabled() -> None:
    async def scenario() -> None:
        from voiceloop.n8n_client import N8nClient

        client = N8nClient(
            base_url="http://127.0.0.1:5678",
            webhook_url="http://127.0.0.1:5678/webhook/test",
            token=None,
            timeout_seconds=5.0,
            enabled=False,
        )
        ok, detail = await client.health()
        assert ok is True
        assert detail == "wyłączony"

        plan = await client.route(CommandRequest(source=CommandSource.PANEL, text="test"))
        assert plan is None

    asyncio.run(scenario())
