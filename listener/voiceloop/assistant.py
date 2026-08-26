from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime
from uuid import uuid4

from .capability_index import CapabilityIndex, CapabilityIndexError
from .conversation_telemetry import ConversationTelemetry
from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .events import EventBus
from .executor import CommandExecutor
from .knowledge_tools import KnowledgeToolOrchestrator
from .memory import MemoryStore
from .memory_vectorization import (
    MEMORY_VECTOR_NAMES,
    memory_query_documents,
    memory_query_weights,
)
from .model_router import ModelRouter, ModelUnavailableError, OpenAICompatiblePlanner
from .models import (
    CommandAccepted,
    CommandPlan,
    CommandRequest,
    CommandSource,
    CommandStatus,
    ToolObservation,
    TurnContext,
)
from .n8n_client import N8nClient, N8nUnavailableError
from .qdrant_memory import QdrantMemoryError, QdrantVectorStore
from .router import deterministic_plan, normalize_text
from .routing.service import RoutingV2Outcome, RoutingV2Service
from .screen import ScreenContextService
from .tts import WindowsTTS

LOGGER = logging.getLogger("voiceloop.assistant")

VOICE_RESULT_ACTIONS = {
    "close_window_under_cursor",
    "copy_email_under_cursor",
    "copy_number_under_cursor",
    "copy_selected_text",
    "copy_sentence_under_cursor",
    "copy_text_under_cursor",
    "create_note",
    "describe_active_window",
    "describe_recent_activity",
    "describe_text_target",
    "list_capabilities",
    "minimize_active_window",
    "minimize_all_windows",
    "recall",
    "remember",
    "remember_last_source",
    "search_web",
    "select_paragraph_under_cursor",
    "select_sentence_under_cursor",
    "rename_under_cursor",
}
PROTECTED_DETERMINISTIC_INTENTS = {
    "list_capabilities",
    "stop",
    "voice_test",
}
CONVERSATION_HISTORY_LIMIT = 24
CONVERSATION_RAW_HISTORY_LIMIT = 10
CONVERSATION_SUMMARY_MAX_CHARS = 2400
CONVERSATION_END_PHRASES = {
    "koniec rozmowy",
    "zakoncz rozmowe",
    "zamknij rozmowe",
    "koniec czatu",
    "zakoncz czat",
    "koniec",
    "zakoncz",
    "do widzenia",
    "na razie",
    "konczymy",
}


class AssistantService:
    def __init__(
        self,
        *,
        memory: MemoryStore,
        events: EventBus,
        executor: CommandExecutor,
        n8n: N8nClient,
        model_router: ModelRouter,
        screen: ScreenContextService,
        tts: WindowsTTS,
        embeddings: OpenAICompatibleEmbeddingClient | None,
        qdrant: QdrantVectorStore | None,
        action_definitions: list[dict[str, object]],
        dedupe_seconds: float,
        vector_context_limit: int,
        vector_query_adaptive_weights: bool = True,
        vector_query_weights: dict[str, float] | None = None,
        vector_memory_min_score: float = 0.15,
        vector_memory_rrf_k: int = 60,
        qdrant_shadow: QdrantVectorStore | None = None,
        capability_index: CapabilityIndex | None = None,
        routing_v2: RoutingV2Service | None = None,
        private_style_instruction: str | None = None,
        telemetry: ConversationTelemetry | None = None,
        knowledge_tools: KnowledgeToolOrchestrator | None = None,
    ) -> None:
        self.memory = memory
        self.events = events
        self.executor = executor
        self.n8n = n8n
        self.model_router = model_router
        self.screen = screen
        self.tts = tts
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.qdrant_shadow = qdrant_shadow
        self.capability_index = capability_index
        self.routing_v2 = routing_v2
        self.private_style_instruction = private_style_instruction
        self.telemetry = telemetry
        self.knowledge_tools = knowledge_tools
        self._latest_tool_observations: list[dict[str, object]] = []
        self.action_definitions = action_definitions
        self.dedupe_seconds = dedupe_seconds
        self.vector_context_limit = max(0, min(vector_context_limit, 30))
        self.vector_query_adaptive_weights = bool(vector_query_adaptive_weights)
        self.vector_query_weights = dict(vector_query_weights or {})
        self.vector_memory_min_score = max(0.0, min(float(vector_memory_min_score), 1.0))
        self.vector_memory_rrf_k = max(1, min(int(vector_memory_rrf_k), 1000))
        self._recent_inputs: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._recent_plan_signatures: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._speech_tasks: set[asyncio.Task[None]] = set()
        self._background_tasks: set[asyncio.Task] = set()
        self._active_handles: dict[asyncio.Task, str] = {}
        self._turn_lock = asyncio.Lock()
        self._conversation_active = False
        self._conversation_style = "default"
        self._conversation_history: list[dict[str, str]] = []
        self._conversation_session_id: str | None = None
        self._recent_action_summaries: list[str] = []
        self.stt_min_action_confidence = 0.75
        self.reject_multi_speaker_commands = True

    def configure_stt_threshold(self, value: float) -> None:
        self.stt_min_action_confidence = max(0.0, min(float(value), 1.0))

    def configure_speaker_gate(self, *, reject_multi_speaker: bool) -> None:
        self.reject_multi_speaker_commands = bool(reject_multi_speaker)

    @property
    def conversation_active(self) -> bool:
        return self._conversation_active

    @property
    def conversation_session_id(self) -> str | None:
        return self._conversation_session_id

    def _start_conversation_state(self) -> None:
        self._conversation_active = True
        self._conversation_style = "max_iq"
        self._conversation_history.clear()
        self._conversation_session_id = str(uuid4())

    def _end_conversation_state(self) -> None:
        self._conversation_active = False
        self._conversation_style = "default"
        self._conversation_history.clear()
        self._conversation_session_id = None

    async def begin_conversation_session(self, greeting: str) -> None:
        cleaned = (greeting or "").strip()
        self._start_conversation_state()
        if cleaned:
            self._conversation_history.append({"role": "assistant", "content": cleaned})
            await self.memory.add_message("assistant", cleaned, None)
        await self.events.publish(
            "conversation.started",
            {"style": self._conversation_style, "greeting": cleaned},
        )

    async def remember_managed_turn_result(
        self,
        *,
        request_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        self._remember_conversation_turn(
            user_text=user_text,
            assistant_text=assistant_text,
        )
        if assistant_text.strip():
            await self.memory.add_message("assistant", assistant_text.strip(), request_id)

    async def handle(self, request: CommandRequest) -> CommandAccepted:
        safety_plan = deterministic_plan(request)
        if safety_plan and safety_plan.intent == "stop":
            return await self._handle_stop_request(request, safety_plan)

        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_handles[current_task] = request.request_id
        try:
            async with self._turn_lock:
                return await self._handle_request(request, safety_plan)
        except asyncio.CancelledError:
            await self._mark_request_cancelled(request.request_id)
            raise
        finally:
            if current_task is not None:
                self._active_handles.pop(current_task, None)

    async def _handle_request(
        self,
        request: CommandRequest,
        safety_plan: CommandPlan | None,
    ) -> CommandAccepted:
        fingerprint = self._fingerprint(request)
        duplicate_id = self._find_duplicate(fingerprint)
        if duplicate_id:
            existing = await self.memory.get_command(duplicate_id)
            return CommandAccepted(
                request_id=duplicate_id,
                status=existing.status if existing else CommandStatus.RECEIVED,
                plan=existing.plan if existing else None,
                duplicate=True,
            )

        existing = await self.memory.get_command(request.request_id)
        if existing:
            return CommandAccepted(
                request_id=existing.request_id,
                status=existing.status,
                plan=existing.plan,
                duplicate=True,
            )

        await self.memory.create_command(request)
        self._remember_fingerprint(fingerprint, request.request_id)
        input_text = (request.text or request.command_id or "").strip()
        self._latest_tool_observations = []
        await self.memory.add_message("user", input_text, request.request_id)
        await self.memory.update_command(
            request.request_id,
            status=CommandStatus.PLANNING,
        )
        await self.events.publish(
            "command.received",
            {
                "request_id": request.request_id,
                "source": request.source.value,
                "text": input_text,
            },
        )
        if self.telemetry is not None:
            await self.telemetry.mark_request(request.request_id, "request_received")

        starts_conversation = self._starts_venice_conversation(input_text)
        if starts_conversation:
            self._start_conversation_state()
            await self.events.publish(
                "conversation.started",
                {"request_id": request.request_id, "style": self._conversation_style},
            )
        self._attach_interaction_session_id(
            request,
            starts_conversation=starts_conversation,
        )

        if self._conversation_active and self._ends_conversation(input_text):
            self._end_conversation_state()
            plan = CommandPlan(
                request_id=request.request_id,
                intent="conversation_end",
                response_text="Kończę rozmowę. Powiedz Asystencie, gdy chcesz zacząć nową.",
                confidence=1.0,
                provider="deterministic",
            )
            await self.events.publish(
                "conversation.ended",
                {"request_id": request.request_id},
            )
            return await self._submit_plan(request, plan)

        conversation_active = self._conversation_active
        conversation_history = list(self._conversation_history) if conversation_active else None
        gated_deterministic = self._gate_deterministic_for_voice(
            request,
            safety_plan,
            conversation_active=conversation_active,
        )
        route_in_background = bool(
            conversation_active
            and self.routing_v2 is not None
            and self.routing_v2.settings.routing_v2_shadow_mode
            and hasattr(self.routing_v2, "live_execution_requested")
            and not self.routing_v2.live_execution_requested
            and not OpenAICompatiblePlanner._is_explicit_action_request(input_text)
        )
        if route_in_background:
            routing_task = asyncio.create_task(
                self._evaluate_routing_v2(
                    request,
                    legacy_plan=gated_deterministic,
                ),
                name=f"routing-v2-shadow-{request.request_id}",
            )
            self._background_tasks.add(routing_task)
            routing_task.add_done_callback(self._background_tasks.discard)
            routing_v2_outcome = None
        else:
            routing_v2_outcome = await self._evaluate_routing_v2(
                request,
                legacy_plan=gated_deterministic,
            )
        protected_legacy_plan = bool(
            gated_deterministic is not None
            and (
                gated_deterministic.provider in {"speaker_gate", "stt_confidence_gate"}
                or gated_deterministic.intent in PROTECTED_DETERMINISTIC_INTENTS
            )
        )
        if (
            routing_v2_outcome is not None
            and self.routing_v2 is not None
            and not protected_legacy_plan
        ):
            plan_execution_allowed = getattr(
                self.routing_v2,
                "plan_execution_allowed",
                None,
            )
            may_execute_v2 = bool(
                self.routing_v2.execution_enabled
                and routing_v2_outcome.plan is not None
                and (
                    not callable(plan_execution_allowed)
                    or plan_execution_allowed(routing_v2_outcome.plan)
                )
            )
            if may_execute_v2:
                gated_deterministic = routing_v2_outcome.plan
            elif getattr(self.routing_v2, "live_execution_requested", False):
                activation_guard = self.routing_v2.activation_guard_plan(
                    request,
                    routing_v2_outcome,
                )
                if activation_guard is not None:
                    gated_deterministic = activation_guard
        plan = await self._create_plan(
            request,
            gated_deterministic,
            conversation_active=conversation_active,
            conversation_style=self._conversation_style,
            history_override=conversation_history,
        )
        plan = self._gate_planned_voice_action(request, plan)
        managed_execution = bool(
            request.managed_voice_turn
            and plan.steps
            and not plan.confirmation_required
            and not plan.requires_clarification
        )
        if conversation_active and not managed_execution:
            self._remember_conversation_turn(
                user_text=input_text,
                assistant_text=plan.clarification_question or plan.response_text,
            )
            if self._latest_tool_observations:
                sources = " | ".join(
                    (
                        f"{item.get('title', '')}: {item.get('url', '')}; "
                        f"{str(item.get('snippet', ''))[:300]}"
                    )
                    for item in self._latest_tool_observations[:3]
                )
                self._conversation_history.append(
                    {
                        "role": "assistant",
                        "content": f"Źródła poprzedniej odpowiedzi: {sources}"[:2400],
                    }
                )
        accepted = await self._submit_plan(request, plan)
        if plan.intent == "task" and plan.steps:
            self._remember_action_summary(plan)
        return accepted

    async def _handle_stop_request(
        self,
        request: CommandRequest,
        plan: CommandPlan,
    ) -> CommandAccepted:
        await self.interrupt()
        existing = await self.memory.get_command(request.request_id)
        if existing is not None:
            return CommandAccepted(
                request_id=existing.request_id,
                status=existing.status,
                plan=existing.plan,
                duplicate=True,
            )

        await self.memory.create_command(request)
        input_text = (request.text or request.command_id or "").strip()
        await self.memory.add_message("user", input_text, request.request_id)
        await self.memory.update_command(
            request.request_id,
            status=CommandStatus.SUCCEEDED,
            plan=plan,
            results=[],
        )
        await self.events.publish(
            "command.received",
            {
                "request_id": request.request_id,
                "source": request.source.value,
                "text": input_text,
            },
        )
        await self.events.publish(
            "command.completed",
            {
                "request_id": request.request_id,
                "status": CommandStatus.SUCCEEDED.value,
            },
        )
        return CommandAccepted(
            request_id=request.request_id,
            status=CommandStatus.SUCCEEDED,
            plan=plan,
        )

    async def _submit_plan(
        self,
        request: CommandRequest,
        plan: CommandPlan,
    ) -> CommandAccepted:
        voice_source = request.source in {
            CommandSource.DEEPGRAM,
            CommandSource.VOICEATTACK,
        }
        plan.managed_voice_turn = request.managed_voice_turn
        plan.speak_result = voice_source and any(
            step.action_id in VOICE_RESULT_ACTIONS for step in plan.steps
        )
        for step in plan.steps:
            if step.action_id == "list_capabilities" and "query" not in step.args:
                step.args["query"] = (request.text or "")[:500]
        plan_signature = self._plan_signature(plan)
        if plan_signature:
            duplicate_id = self._find_duplicate_plan(plan_signature)
            if duplicate_id:
                existing = await self.memory.get_command(duplicate_id)
                await self.memory.update_command(
                    request.request_id,
                    status=CommandStatus.REJECTED,
                    plan=plan,
                    error=f"Pominięto duplikat planu: {duplicate_id}",
                )
                await self.events.publish(
                    "command.duplicate",
                    {
                        "request_id": request.request_id,
                        "duplicate_of": duplicate_id,
                        "reason": "same_action_plan",
                    },
                )
                LOGGER.info(
                    "Skipping duplicate action plan request_id=%s duplicate_of=%s",
                    request.request_id,
                    duplicate_id,
                )
                return CommandAccepted(
                    request_id=duplicate_id,
                    status=existing.status if existing else CommandStatus.REJECTED,
                    plan=existing.plan if existing else plan,
                    duplicate=True,
                )
        command = await self.executor.submit(plan)
        if plan_signature:
            self._remember_plan_signature(plan_signature, request.request_id)
        if plan.response_text and not (
            request.managed_voice_turn
            and plan.steps
            and not plan.confirmation_required
            and not plan.requires_clarification
        ):
            await self.memory.add_message("assistant", plan.response_text, request.request_id)
        await self.events.publish(
            "command.planned",
            {
                "request_id": request.request_id,
                "plan": plan.model_dump(mode="json"),
            },
        )
        if plan.steps:
            voiceattack_steps = [
                step.action_id
                for step in plan.steps
                if self._action_available_in_voiceattack(step.action_id)
            ]
            await self.events.publish(
                "command.capability_match",
                {
                    "request_id": request.request_id,
                    "action_ids": [step.action_id for step in plan.steps],
                    "voiceattack_action_ids": voiceattack_steps,
                    "has_voiceattack_match": bool(voiceattack_steps),
                    "has_native_voiceloop_action": len(voiceattack_steps) < len(plan.steps),
                },
            )
        if voice_source and not request.managed_voice_turn:
            spoken = plan.clarification_question or plan.response_text
            if spoken and (
                plan.intent == "conversation"
                or not plan.speak_result
                or plan.confirmation_required
                or plan.requires_clarification
            ):
                task = asyncio.create_task(self.tts.speak(spoken))
                self._speech_tasks.add(task)
                task.add_done_callback(self._speech_tasks.discard)
        return CommandAccepted(
            request_id=request.request_id,
            status=command.status if command else CommandStatus.FAILED,
            plan=plan,
        )

    async def _create_plan(
        self,
        request: CommandRequest,
        deterministic: CommandPlan | None,
        *,
        conversation_active: bool = False,
        conversation_style: str = "default",
        history_override: list[dict[str, str]] | None = None,
    ) -> CommandPlan:
        text = (request.text or "").strip()
        explicit_action = OpenAICompatiblePlanner._is_explicit_action_request(text)
        if deterministic and (
            request.command_id
            or not conversation_active
            or deterministic.requires_clarification
            or deterministic.provider == "stt_confidence_gate"
            or deterministic.provider == "speaker_gate"
            or deterministic.provider == "compound_fast_path_guard"
            or deterministic.intent in PROTECTED_DETERMINISTIC_INTENTS
            or (explicit_action and deterministic.steps)
        ):
            return deterministic

        if text and explicit_action:
            await self._publish_capability_candidates(request.request_id, text)

        if not conversation_active:
            try:
                n8n_plan = await self.n8n.route(request)
                if n8n_plan:
                    return n8n_plan
            except N8nUnavailableError:
                pass

        if self.telemetry is not None:
            await self.telemetry.mark_request(request.request_id, "context_started")
        history_coro = (
            asyncio.sleep(0, result=history_override)
            if history_override is not None
            else self.memory.recent_messages(limit=12)
        )
        screen_coro = (
            self.screen.capture(request.request_id)
            if request.include_screen
            else asyncio.sleep(0, result=None)
        )
        knowledge_coro = (
            self.knowledge_tools.lookup(request_id=request.request_id, text=text)
            if (
                conversation_active
                and self.knowledge_tools is not None
                and self.knowledge_tools.should_search(text)
            )
            else asyncio.sleep(0, result=None)
        )
        (
            history,
            memories,
            memory_items,
            screen_snapshot,
            knowledge_lookup,
        ) = await asyncio.gather(
            history_coro,
            self._vector_memories_for_request(request),
            self.memory.list_memories(limit=30),
            screen_coro,
            knowledge_coro,
        )
        tool_observations = (
            list(knowledge_lookup.observations)
            if knowledge_lookup is not None
            else []
        )
        if knowledge_lookup is not None and knowledge_lookup.error:
            tool_observations.append(
                ToolObservation(
                    kind="web_search_error",
                    query=text[:500],
                    title="Nie udało się pobrać aktualnych źródeł",
                    snippet=knowledge_lookup.error[:1200],
                    provider="knowledge_tools",
                )
            )
            memories.append(
                "Aktualne wyszukiwanie nie powiodło się: "
                f"{knowledge_lookup.error[:500]}. "
                "Nie przedstawiaj świeżych danych jako sprawdzonych."
            )
        self._latest_tool_observations = [
            observation.model_dump(mode="json") for observation in tool_observations
        ]
        image_data_url = (
            self.screen.image_data_url(screen_snapshot)
            if screen_snapshot is not None
            else None
        )
        if screen_snapshot is not None:
            await self.events.publish(
                "screen.captured",
                {
                    "request_id": request.request_id,
                    "window_title": screen_snapshot.window_title,
                    "process_name": screen_snapshot.process_name,
                },
            )
        if not any("source=manual_memory;" in context for context in memories):
            memory_limit = 3 if memories else 30
            memories.extend(
                item.content for item in reversed(memory_items[:memory_limit])
            )
        if self._recent_action_summaries:
            memories.extend(self._recent_action_summaries[-5:])
        turn_context = TurnContext(
            question=text,
            session_id=request.interaction_session_id,
            recent_turns=list(history[-12:]),
            memories=list(memories[-20:]),
            tool_observations=tool_observations,
            local_time=datetime.now().astimezone().isoformat(),
            screen=screen_snapshot,
            sources={
                "history_messages": len(history),
                "memory_items": len(memories),
                "knowledge_sources": len(tool_observations),
            },
        )
        if self.telemetry is not None:
            await self.telemetry.mark_request(
                request.request_id,
                "context_ready",
                metadata={
                    "history_messages": len(history),
                    "memory_items": len(memories),
                    "screen_included": screen_snapshot is not None,
                    "knowledge_sources": len(tool_observations),
                },
            )
        try:
            if self.telemetry is not None:
                await self.telemetry.mark_request(request.request_id, "model_started")
            plan = await self.model_router.plan(
                request=request,
                history=turn_context.recent_turns,
                memories=turn_context.memories,
                screen=turn_context.screen,
                image_data_url=image_data_url,
                actions=self.action_definitions,
                conversation_active=conversation_active,
                conversation_style_override=conversation_style,
                private_style_instruction=self.private_style_instruction,
                tool_observations=turn_context.tool_observations,
                local_time=turn_context.local_time,
            )
            if self.telemetry is not None:
                await self.telemetry.mark_request(
                    request.request_id,
                    "model_completed",
                    metadata={"model_provider": plan.provider, "model": plan.model},
                )
            return plan
        except ModelUnavailableError as exc:
            await self.memory.update_command(
                request.request_id,
                status=CommandStatus.FAILED,
                error=str(exc),
            )
            raise

    async def _publish_capability_candidates(
        self,
        request_id: str,
        text: str,
    ) -> None:
        if self.capability_index is None or not self.capability_index.ready:
            return
        try:
            result = await self.capability_index.search(text)
        except (CapabilityIndexError, EmbeddingUnavailableError) as exc:
            LOGGER.warning("Capability retrieval unavailable: %s", exc)
            return
        except Exception:
            LOGGER.exception("Capability retrieval failed")
            return
        await self.events.publish(
            "command.capability_candidates",
            {
                "request_id": request_id,
                **result.as_dict(),
            },
        )

    async def _evaluate_routing_v2(
        self,
        request: CommandRequest,
        *,
        legacy_plan: CommandPlan | None,
    ) -> RoutingV2Outcome | None:
        service = self.routing_v2
        if service is None or not service.settings.routing_v2_enabled:
            return None
        if not (service.settings.routing_v2_shadow_mode or service.settings.routing_v2_execute):
            return None
        timeout = max(
            0.1,
            min(float(service.settings.routing_v2_shadow_timeout_seconds), 30.0),
        )
        try:
            async with asyncio.timeout(timeout):
                outcome = await service.evaluate(request)
        except TimeoutError:
            LOGGER.warning("Routing V2 timed out after %.2fs", timeout)
            await self.events.publish(
                "routing.v2.unavailable",
                {
                    "request_id": request.request_id,
                    "reason": "timeout",
                },
            )
            outcome = (
                service.unavailable_outcome(request, reason="timeout")
                if service.live_execution_requested
                else None
            )
            if outcome is None:
                return None
        except (CapabilityIndexError, EmbeddingUnavailableError) as exc:
            LOGGER.warning("Routing V2 unavailable: %s", exc)
            reason = type(exc).__name__
            await self.events.publish(
                "routing.v2.unavailable",
                {
                    "request_id": request.request_id,
                    "reason": reason,
                },
            )
            outcome = (
                service.unavailable_outcome(request, reason=reason)
                if service.live_execution_requested
                else None
            )
            if outcome is None:
                return None
        except Exception:
            LOGGER.exception("Routing V2 evaluation failed")
            await self.events.publish(
                "routing.v2.unavailable",
                {
                    "request_id": request.request_id,
                    "reason": "unexpected_error",
                },
            )
            outcome = (
                service.unavailable_outcome(request, reason="unexpected_error")
                if service.live_execution_requested
                else None
            )
            if outcome is None:
                return None

        envelope = request.transcript
        if envelope is not None:
            await self.events.publish(
                "routing.v2.envelope",
                {
                    "request_id": request.request_id,
                    "transcript": envelope.model_dump(mode="json"),
                },
            )
        await self.events.publish(
            "routing.v2.segments",
            {
                "request_id": request.request_id,
                "segmentation": outcome.segmentation.model_dump(mode="json"),
            },
        )
        await self.events.publish(
            "routing.v2.decisions",
            {
                "request_id": request.request_id,
                "decisions": [decision.model_dump(mode="json") for decision in outcome.decisions],
            },
        )
        await self.events.publish(
            "routing.v2.shadow",
            service.shadow_payload(
                request,
                outcome,
                legacy_plan=legacy_plan,
            ),
        )
        return outcome

    def _gate_deterministic_for_voice(
        self,
        request: CommandRequest,
        plan: CommandPlan | None,
        *,
        conversation_active: bool,
    ) -> CommandPlan | None:
        if (
            self.reject_multi_speaker_commands
            and request.source is CommandSource.DEEPGRAM
            and request.transcript is not None
            and len(set(request.transcript.speaker_ids)) > 1
        ):
            return CommandPlan(
                request_id=request.request_id,
                intent="conversation",
                response_text=(
                    "Nie wykonuję polecenia, bo transkrypcja zawiera więcej niż jednego mówcę."
                ),
                confidence=request.effective_transcript_confidence or 0.0,
                requires_clarification=True,
                clarification_question=("Powiedz polecenie ponownie, zaczynając od „Asystencie”."),
                provider="speaker_gate",
            )
        if plan is None:
            return None
        if request.command_id or not plan.steps:
            return plan
        if request.source is not CommandSource.DEEPGRAM:
            return plan
        confidence = request.effective_transcript_confidence
        if confidence is None or confidence < self.stt_min_action_confidence:
            return CommandPlan(
                request_id=request.request_id,
                intent="conversation",
                response_text=(
                    "Nie jestem pewna, czy dobrze usłyszałam polecenie. "
                    "Powiedz jeszcze raz albo doprecyzuj, co mam zrobić."
                ),
                confidence=float(confidence) if confidence is not None else 0.0,
                requires_clarification=True,
                clarification_question=("Czy to było polecenie do wykonania, czy zwykła rozmowa?"),
                provider="stt_confidence_gate",
            )
        from .model_router import OpenAICompatiblePlanner

        text = (request.text or "").strip()
        if conversation_active and not OpenAICompatiblePlanner._is_explicit_action_request(text):
            return None
        return plan

    def _gate_planned_voice_action(
        self,
        request: CommandRequest,
        plan: CommandPlan,
    ) -> CommandPlan:
        confidence = request.effective_transcript_confidence
        if (
            request.source is CommandSource.DEEPGRAM
            and plan.steps
            and (confidence is None or confidence < self.stt_min_action_confidence)
        ):
            return CommandPlan(
                request_id=request.request_id,
                intent="conversation",
                response_text="Nie wykonuję tej akcji, bo transkrypcja jest niepewna.",
                confidence=float(confidence) if confidence is not None else 0.0,
                requires_clarification=True,
                clarification_question="Jakie dokładnie polecenie mam wykonać?",
                provider="stt_confidence_gate",
            )
        return plan

    def _action_available_in_voiceattack(self, action_id: str) -> bool:
        return any(
            item.get("id") == action_id and bool(item.get("available_in_voiceattack"))
            for item in self.action_definitions
        )

    def _remember_action_summary(self, plan: CommandPlan) -> None:
        action_ids = [step.action_id for step in plan.steps]
        if not action_ids:
            return
        summary = (
            f"Ostatnia udana intencja zadania: {plan.intent}; akcje={','.join(action_ids[:6])}"
        )
        self._recent_action_summaries.append(summary)
        self._recent_action_summaries = self._recent_action_summaries[-8:]

    async def _vector_memories_for_request(self, request: CommandRequest) -> list[str]:
        if (
            self.embeddings is None
            or not self.embeddings.enabled
            or not self.embeddings.accepts_private_text()
            or self.vector_context_limit <= 0
        ):
            return []
        query = (request.text or request.command_id or "").strip()
        if not query:
            return []
        try:
            query_documents = memory_query_documents(query[:2000])
            vector_names = tuple(
                name for name in MEMORY_VECTOR_NAMES if name in query_documents
            )
            vectors = await self.embeddings.embed_queries(
                [query_documents[name] for name in vector_names]
            )
            if len(vectors) != len(vector_names):
                raise EmbeddingUnavailableError(
                    "embedding count mismatch for memory query documents"
                )
            query_vectors = dict(zip(vector_names, vectors, strict=True))
            semantic_embedding = query_vectors.get("semantic", [])
            query_weights = memory_query_weights(
                query,
                adaptive=self.vector_query_adaptive_weights,
                base_weights=self.vector_query_weights or None,
            )
            hits = []
            if (
                self.qdrant is not None
                and self.qdrant.enabled
                and self.qdrant.accepts_private_data()
            ):
                hits = await self.qdrant.search(
                    query_vectors=query_vectors,
                    limit=self.vector_context_limit,
                    min_score=self.vector_memory_min_score,
                    query_weights=query_weights,
                    rrf_k=self.vector_memory_rrf_k,
                )
            active_qdrant_hits = list(hits)
            if (
                self.qdrant_shadow is not None
                and self.qdrant_shadow.enabled
                and self.qdrant_shadow.accepts_private_data()
            ):
                try:
                    shadow_hits = await self.qdrant_shadow.search(
                        query_vectors=query_vectors,
                        limit=self.vector_context_limit,
                        min_score=self.vector_memory_min_score,
                        query_weights=query_weights,
                        rrf_k=self.vector_memory_rrf_k,
                    )
                except QdrantMemoryError as exc:
                    LOGGER.warning("Shadow vector memory unavailable: %s", exc)
                else:
                    active_ids = [hit.source_id for hit in active_qdrant_hits]
                    shadow_ids = [hit.source_id for hit in shadow_hits]
                    await self.events.publish(
                        "memory.retrieval_shadow",
                        {
                            "request_id": request.request_id,
                            "active_collection": getattr(
                                self.qdrant,
                                "collection_name",
                                "",
                            ),
                            "shadow_collection": getattr(
                                self.qdrant_shadow,
                                "collection_name",
                                "",
                            ),
                            "active_source_ids": active_ids,
                            "shadow_source_ids": shadow_ids,
                            "top1_match": bool(
                                active_ids
                                and shadow_ids
                                and active_ids[0] == shadow_ids[0]
                            ),
                            "overlap_count": len(set(active_ids) & set(shadow_ids)),
                        },
                    )
            if not hits and semantic_embedding:
                hits = await self.memory.search_vector_memories(
                    semantic_embedding,
                    limit=self.vector_context_limit,
                    min_score=self.vector_memory_min_score,
                )
        except EmbeddingUnavailableError as exc:
            LOGGER.warning("Vector memory unavailable: %s", exc)
            return []
        except QdrantMemoryError as exc:
            LOGGER.warning("Qdrant memory unavailable: %s", exc)
            return []
        except Exception:
            LOGGER.exception("Vector memory retrieval failed")
            return []
        contexts: list[str] = []
        for hit in hits:
            metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
            provenance = metadata.get("provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            evidence = metadata.get("retrieval_evidence")
            evidence = evidence if isinstance(evidence, dict) else {}
            spaces = evidence.get("spaces")
            space_names = (
                ",".join(sorted(str(name) for name in spaces))
                if isinstance(spaces, dict)
                else "semantic"
            )
            source_time = (
                provenance.get("time")
                or metadata.get("time")
                or metadata.get("timestamp")
                or ""
            )
            confidence = provenance.get("confidence", metadata.get("confidence"))
            contexts.append(
                (
                    f"Local vector memory score={hit.score:.3f}; "
                    f"spaces={space_names}; source={hit.source}; source_id={hit.source_id}; "
                    f"time={source_time or 'unknown'}; confidence={confidence}; "
                    f"title={hit.title}; content={hit.content[:900]}"
                )[:1600]
            )
        return contexts

    async def interrupt(self, *, end_conversation: bool = False) -> None:
        current_task = asyncio.current_task()
        active_tasks = [
            task
            for task in tuple(self._active_handles)
            if task is not current_task and not task.done()
        ]
        for task in active_tasks:
            task.cancel()
        speech_tasks = [task for task in tuple(self._speech_tasks) if not task.done()]
        for task in speech_tasks:
            task.cancel()
        background_tasks = [
            task for task in tuple(self._background_tasks) if not task.done()
        ]
        for task in background_tasks:
            task.cancel()

        await self.tts.stop()
        await self.executor.stop_all()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        if speech_tasks:
            await asyncio.gather(*speech_tasks, return_exceptions=True)
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        if end_conversation:
            self._end_conversation_state()
            await self.events.publish("conversation.ended", {"reason": "hard_stop"})
        await self.events.publish(
            "assistant.interrupted",
            {
                "active_requests": len(active_tasks),
                "speech_tasks": len(speech_tasks),
                "background_tasks": len(background_tasks),
                "conversation_ended": end_conversation,
            },
        )

    async def close(self) -> None:
        await self.interrupt(end_conversation=True)
        for task in tuple(self._speech_tasks):
            task.cancel()
        if self._speech_tasks:
            await asyncio.gather(*self._speech_tasks, return_exceptions=True)
        self._speech_tasks.clear()

    async def _mark_request_cancelled(self, request_id: str) -> None:
        command = await self.memory.get_command(request_id)
        if command is None or command.status in {
            CommandStatus.SUCCEEDED,
            CommandStatus.FAILED,
            CommandStatus.CANCELLED,
            CommandStatus.REJECTED,
        }:
            return
        await self.memory.update_command(
            request_id,
            status=CommandStatus.CANCELLED,
            error="Przerwano przez użytkownika.",
        )
        await self.events.publish(
            "command.cancelled",
            {"request_id": request_id, "reason": "user_interrupt"},
        )

    @staticmethod
    def _starts_venice_conversation(text: str) -> bool:
        normalized = normalize_text(text)
        first_word = normalized.split(maxsplit=1)[0] if normalized else ""
        return first_word in {"venice", "venive", "wenice"}

    @staticmethod
    def _ends_conversation(text: str) -> bool:
        return normalize_text(text) in CONVERSATION_END_PHRASES

    def _remember_conversation_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
    ) -> None:
        if user_text:
            self._conversation_history.append({"role": "user", "content": user_text})
        if assistant_text:
            self._conversation_history.append({"role": "assistant", "content": assistant_text})
        if len(self._conversation_history) <= CONVERSATION_HISTORY_LIMIT:
            return
        older = self._conversation_history[:-CONVERSATION_RAW_HISTORY_LIMIT]
        recent = self._conversation_history[-CONVERSATION_RAW_HISTORY_LIMIT:]
        summary_parts: list[str] = []
        for message in older:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if content.startswith("Wcześniejszy kontekst rozmowy:"):
                content = content.removeprefix("Wcześniejszy kontekst rozmowy:").strip()
            role = "Użytkownik" if message.get("role") == "user" else "Asystent"
            summary_parts.append(f"{role}: {content[:500]}")
        summary = " | ".join(summary_parts)[-CONVERSATION_SUMMARY_MAX_CHARS:]
        self._conversation_history = [
            {
                "role": "assistant",
                "content": f"Wcześniejszy kontekst rozmowy: {summary}",
            },
            *recent,
        ]

    def _attach_interaction_session_id(
        self,
        request: CommandRequest,
        *,
        starts_conversation: bool,
    ) -> None:
        if starts_conversation:
            if request.interaction_session_id:
                self._conversation_session_id = request.interaction_session_id
            else:
                request.interaction_session_id = self._conversation_session_id
            return
        if (
            self._conversation_active
            and self._conversation_session_id
            and request.interaction_session_id is None
        ):
            # During managed assistant conversation, keep one stable session ID.
            request.interaction_session_id = self._conversation_session_id

    def _fingerprint(self, request: CommandRequest) -> str:
        normalized = normalize_text(request.command_id or request.text or "")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _plan_signature(plan: CommandPlan) -> str | None:
        if not plan.steps:
            return None
        payload = [
            {
                "action_id": step.action_id,
                "args": step.args,
            }
            for step in plan.steps
        ]
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _find_duplicate_plan(self, signature: str) -> str | None:
        now = time.monotonic()
        self._prune_plan_signatures(now)
        entry = self._recent_plan_signatures.get(signature)
        window_seconds = max(self.dedupe_seconds * 3, 5.0)
        if entry and now - entry[0] <= window_seconds:
            return entry[1]
        return None

    def _remember_plan_signature(self, signature: str, request_id: str) -> None:
        now = time.monotonic()
        self._recent_plan_signatures[signature] = (now, request_id)
        self._recent_plan_signatures.move_to_end(signature)
        self._prune_plan_signatures(now)

    def _prune_plan_signatures(self, now: float) -> None:
        max_age = max(self.dedupe_seconds * 6, 15.0)
        while self._recent_plan_signatures:
            _, (timestamp, _) = next(iter(self._recent_plan_signatures.items()))
            if now - timestamp <= max_age and len(self._recent_plan_signatures) <= 500:
                break
            self._recent_plan_signatures.popitem(last=False)

    def _find_duplicate(self, fingerprint: str) -> str | None:
        now = time.monotonic()
        self._prune_fingerprints(now)
        entry = self._recent_inputs.get(fingerprint)
        if entry and now - entry[0] <= self.dedupe_seconds:
            return entry[1]
        return None

    def _remember_fingerprint(self, fingerprint: str, request_id: str) -> None:
        now = time.monotonic()
        self._recent_inputs[fingerprint] = (now, request_id)
        self._recent_inputs.move_to_end(fingerprint)
        self._prune_fingerprints(now)

    def _prune_fingerprints(self, now: float) -> None:
        max_age = max(self.dedupe_seconds * 4, 10.0)
        while self._recent_inputs:
            _, (timestamp, _) = next(iter(self._recent_inputs.items()))
            if now - timestamp <= max_age and len(self._recent_inputs) <= 500:
                break
            self._recent_inputs.popitem(last=False)
