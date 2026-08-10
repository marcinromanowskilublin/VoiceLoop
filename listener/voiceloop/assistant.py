from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict

from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .events import EventBus
from .executor import CommandExecutor
from .memory import MemoryStore
from .model_router import ModelRouter, ModelUnavailableError
from .models import (
    CommandAccepted,
    CommandPlan,
    CommandRequest,
    CommandSource,
    CommandStatus,
)
from .n8n_client import N8nClient, N8nUnavailableError
from .qdrant_memory import QdrantMemoryError, QdrantVectorStore
from .router import deterministic_plan, normalize_text
from .screen import ScreenContextService
from .tts import WindowsTTS

LOGGER = logging.getLogger("voiceloop.assistant")

VOICE_RESULT_ACTIONS = {
    "create_note",
    "describe_active_window",
    "describe_recent_activity",
    "describe_text_target",
    "minimize_active_window",
    "minimize_all_windows",
    "recall",
    "remember",
    "remember_last_source",
    "search_web",
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
        self.action_definitions = action_definitions
        self.dedupe_seconds = dedupe_seconds
        self.vector_context_limit = max(0, min(vector_context_limit, 30))
        self._recent_inputs: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._speech_tasks: set[asyncio.Task[None]] = set()

    async def handle(self, request: CommandRequest) -> CommandAccepted:
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

        safety_plan = deterministic_plan(request)
        if safety_plan and safety_plan.intent == "stop":
            await self.executor.stop_all()
            await self.tts.stop()
            await self.memory.update_command(
                request.request_id,
                status=CommandStatus.SUCCEEDED,
                plan=safety_plan,
                results=[],
            )
            return CommandAccepted(
                request_id=request.request_id,
                status=CommandStatus.SUCCEEDED,
                plan=safety_plan,
            )

        plan = await self._create_plan(request, safety_plan)
        voice_source = request.source in {
            CommandSource.DEEPGRAM,
            CommandSource.VOICEATTACK,
        }
        plan.speak_result = voice_source and any(
            step.action_id in VOICE_RESULT_ACTIONS for step in plan.steps
        )
        command = await self.executor.submit(plan)
        if plan.response_text:
            await self.memory.add_message("assistant", plan.response_text, request.request_id)
        await self.events.publish(
            "command.planned",
            {
                "request_id": request.request_id,
                "plan": plan.model_dump(mode="json"),
            },
        )
        if voice_source:
            spoken = plan.clarification_question or plan.response_text
            if spoken and (
                not plan.speak_result
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
    ) -> CommandPlan:
        try:
            n8n_plan = await self.n8n.route(request)
            if n8n_plan:
                return n8n_plan
        except N8nUnavailableError:
            pass

        if deterministic:
            return deterministic

        screen_snapshot = None
        image_data_url = None
        if request.include_screen:
            screen_snapshot = await self.screen.capture(request.request_id)
            image_data_url = self.screen.image_data_url(screen_snapshot)
            await self.events.publish(
                "screen.captured",
                {
                    "request_id": request.request_id,
                    "window_title": screen_snapshot.window_title,
                    "process_name": screen_snapshot.process_name,
                },
            )
        history = await self.memory.recent_messages(limit=12)
        memory_items = await self.memory.list_memories(limit=30)
        memories = [item.content for item in reversed(memory_items)]
        memories.extend(await self._vector_memories_for_request(request))
        try:
            return await self.model_router.plan(
                request=request,
                history=history,
                memories=memories,
                screen=screen_snapshot,
                image_data_url=image_data_url,
                actions=self.action_definitions,
            )
        except ModelUnavailableError as exc:
            await self.memory.update_command(
                request.request_id,
                status=CommandStatus.FAILED,
                error=str(exc),
            )
            raise

    async def _vector_memories_for_request(self, request: CommandRequest) -> list[str]:
        if self.embeddings is None or not self.embeddings.enabled or self.vector_context_limit <= 0:
            return []
        query = (request.text or request.command_id or "").strip()
        if not query:
            return []
        try:
            query_embedding = await self.embeddings.embed_query(query[:2000])
            hits = []
            if self.qdrant is not None and self.qdrant.enabled:
                hits = await self.qdrant.search(
                    query_embedding,
                    limit=self.vector_context_limit,
                )
            if not hits:
                hits = await self.memory.search_vector_memories(
                    query_embedding,
                    limit=self.vector_context_limit,
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
        return [
            (
                f"Screenpipe vector memory score={hit.score:.3f}; "
                f"title={hit.title}; content={hit.content[:900]}"
            )
            for hit in hits
        ]

    async def close(self) -> None:
        for task in tuple(self._speech_tasks):
            task.cancel()
        if self._speech_tasks:
            await asyncio.gather(*self._speech_tasks, return_exceptions=True)
        self._speech_tasks.clear()

    def _fingerprint(self, request: CommandRequest) -> str:
        normalized = normalize_text(request.command_id or request.text or "")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

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
