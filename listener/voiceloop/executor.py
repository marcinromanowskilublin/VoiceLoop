from __future__ import annotations

import asyncio
from contextlib import suppress

from .actions import ActionRegistry
from .conversation_telemetry import ConversationTelemetry
from .events import EventBus
from .memory import MemoryStore
from .models import ActionResult, CommandPlan, CommandStatus, CommandView, utc_now

CONFIRMATION_TTL_SECONDS = 300


class CommandExecutor:
    def __init__(
        self,
        *,
        memory: MemoryStore,
        actions: ActionRegistry,
        events: EventBus,
        queue_limit: int,
        telemetry: ConversationTelemetry | None = None,
    ) -> None:
        self.memory = memory
        self.actions = actions
        self.events = events
        self.telemetry = telemetry
        self.queue: asyncio.Queue[CommandPlan] = asyncio.Queue(maxsize=queue_limit)
        self.pending_confirmation: dict[str, CommandPlan] = {}
        self._completion_futures: dict[str, asyncio.Future[CommandView | None]] = {}
        self._state_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._current_request_id: str | None = None
        self._current_execution: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._stopping = False
            self._worker = asyncio.create_task(self._run(), name="voiceloop-executor")

    async def close(self) -> None:
        self._stopping = True
        await self.stop_all()
        if self._worker:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
        self._worker = None

    async def submit(self, plan: CommandPlan) -> CommandView | None:
        plan = await self.actions.bind_execution_targets(plan)
        for step in plan.steps:
            self.actions.enforce_policy(step)

        if plan.requires_clarification:
            async with self._state_lock:
                return await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.AWAITING_CONFIRMATION,
                    plan=plan,
                )

        if not plan.steps:
            async with self._state_lock:
                command = await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.SUCCEEDED,
                    plan=plan,
                    results=[],
                )
            await self.events.publish(
                "command.completed",
                {"request_id": plan.request_id, "status": CommandStatus.SUCCEEDED.value},
            )
            return command

        if plan.confirmation_required:
            async with self._state_lock:
                self.pending_confirmation[plan.request_id] = plan
                command = await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.AWAITING_CONFIRMATION,
                    plan=plan,
                )
            await self.events.publish(
                "command.confirmation_required",
                {"request_id": plan.request_id, "plan": plan.model_dump(mode="json")},
            )
            return command

        return await self._enqueue(plan)

    async def confirm(self, request_id: str) -> CommandView | None:
        async with self._state_lock:
            command = await self.memory.get_command(request_id)
            if command is None or command.status is not CommandStatus.AWAITING_CONFIRMATION:
                return command
            age_seconds = (utc_now() - command.updated_at).total_seconds()
            if age_seconds > CONFIRMATION_TTL_SECONDS:
                self.pending_confirmation.pop(request_id, None)
                expired = await self.memory.update_command(
                    request_id,
                    status=CommandStatus.CANCELLED,
                    error="Potwierdzenie wygasło.",
                )
                await self.events.publish(
                    "command.cancelled",
                    {"request_id": request_id, "reason": "confirmation_expired"},
                )
                return expired

            plan = self.pending_confirmation.pop(request_id, None)
            if plan is None:
                if command.plan is None or not command.plan.steps:
                    return command
                plan = command.plan
                for step in plan.steps:
                    self.actions.enforce_policy(step)
            return await self._enqueue_locked(plan)

    async def cancel(self, request_id: str) -> CommandView | None:
        async with self._state_lock:
            self.pending_confirmation.pop(request_id, None)
            execution = (
                self._current_execution
                if self._current_request_id == request_id
                else None
            )
            if execution is not None:
                execution.cancel()
            command = await self.memory.update_command(
                request_id,
                status=CommandStatus.CANCELLED,
                error="Anulowano przez użytkownika.",
            )
            self._resolve_completion(request_id, command)
        if execution is not None:
            await self.actions.stop()
        await self.events.publish(
            "command.cancelled",
            {"request_id": request_id},
        )
        return command

    async def stop_all(self) -> None:
        for request_id in tuple(self.pending_confirmation):
            await self.cancel(request_id)
        while True:
            async with self._state_lock:
                try:
                    plan = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.CANCELLED,
                    error="Anulowano przez STOP.",
                )
                command = await self.memory.get_command(plan.request_id)
                self._resolve_completion(plan.request_id, command)
            self.queue.task_done()
        async with self._state_lock:
            execution = self._current_execution
            if execution is not None:
                execution.cancel()
                if self._current_request_id is not None:
                    await self.memory.update_command(
                        self._current_request_id,
                        status=CommandStatus.CANCELLED,
                        error="Wykonanie przerwane.",
                    )
        if execution is not None:
            with suppress(asyncio.CancelledError):
                await execution
        await self.actions.stop()
        await self.events.publish("stop", {"status": "all_cancelled"})

    async def wait_for_completion(
        self,
        request_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> CommandView | None:
        command = await self.memory.get_command(request_id)
        if command is None or command.status in {
            CommandStatus.SUCCEEDED,
            CommandStatus.FAILED,
            CommandStatus.CANCELLED,
            CommandStatus.REJECTED,
            CommandStatus.AWAITING_CONFIRMATION,
        }:
            return command
        future = self._completion_future(request_id)
        if timeout_seconds is None:
            return await asyncio.shield(future)
        return await asyncio.wait_for(
            asyncio.shield(future),
            timeout=max(0.1, timeout_seconds),
        )

    async def _enqueue(self, plan: CommandPlan) -> CommandView | None:
        async with self._state_lock:
            return await self._enqueue_locked(plan)

    async def _enqueue_locked(self, plan: CommandPlan) -> CommandView | None:
        if self.queue.full():
            await self.memory.update_command(
                plan.request_id,
                status=CommandStatus.REJECTED,
                plan=plan,
                error="Kolejka jest pełna.",
            )
            raise RuntimeError("Kolejka poleceń jest pełna.")
        command = await self.memory.update_command(
            plan.request_id,
            status=CommandStatus.QUEUED,
            plan=plan,
        )
        self._completion_future(plan.request_id)
        try:
            self.queue.put_nowait(plan)
        except asyncio.QueueFull as exc:
            await self.memory.update_command(
                plan.request_id,
                status=CommandStatus.REJECTED,
                error="Kolejka zapełniła się podczas dodawania.",
            )
            raise RuntimeError("Kolejka poleceń jest pełna.") from exc
        await self.events.publish(
            "command.queued",
            {"request_id": plan.request_id, "plan": plan.model_dump(mode="json")},
        )
        return command

    async def _run(self) -> None:
        while not self._stopping:
            plan = await self.queue.get()
            try:
                execution = await self._claim_execution(plan)
                if execution is None:
                    continue
                try:
                    await execution
                except asyncio.CancelledError:
                    await self._finish_execution(
                        plan,
                        status=CommandStatus.CANCELLED,
                        error="Wykonanie przerwane.",
                    )
                except Exception as exc:
                    failed = await self._finish_execution(
                        plan,
                        status=CommandStatus.FAILED,
                        error=f"Błąd executora: {exc}",
                    )
                    if failed is not None:
                        await self.events.publish(
                            "command.completed",
                            {
                                "request_id": plan.request_id,
                                "status": CommandStatus.FAILED.value,
                                "error": str(exc),
                            },
                        )
                finally:
                    async with self._state_lock:
                        if self._current_execution is execution:
                            self._current_execution = None
                            self._current_request_id = None
            finally:
                self.queue.task_done()

    async def _claim_execution(
        self,
        plan: CommandPlan,
    ) -> asyncio.Task[None] | None:
        async with self._state_lock:
            command = await self.memory.get_command(plan.request_id)
            if command is None or command.status is not CommandStatus.QUEUED:
                return None
            await self.memory.update_command(
                plan.request_id,
                status=CommandStatus.EXECUTING,
                plan=plan,
            )
            execution = asyncio.create_task(
                self._execute_plan(plan),
                name=f"voiceloop-command-{plan.request_id}",
            )
            self._current_request_id = plan.request_id
            self._current_execution = execution
            return execution

    async def _finish_execution(
        self,
        plan: CommandPlan,
        *,
        status: CommandStatus,
        results: list[ActionResult] | None = None,
        error: str | None = None,
    ) -> CommandView | None:
        async with self._state_lock:
            command = await self.memory.get_command(plan.request_id)
            if command is None or command.status is not CommandStatus.EXECUTING:
                return None
            updated = await self.memory.update_command(
                plan.request_id,
                status=status,
                plan=plan,
                results=results,
                error=error,
            )
            self._resolve_completion(plan.request_id, updated)
            return updated

    async def _execute_plan(self, plan: CommandPlan) -> None:
        if self.telemetry is not None:
            await self.telemetry.mark_request(
                plan.request_id,
                "tool_started",
                metadata={"tool_action_count": len(plan.steps)},
            )
        await self.events.publish(
            "command.executing",
            {"request_id": plan.request_id, "plan": plan.model_dump(mode="json")},
        )

        results: list[ActionResult] = []
        successful_steps: set[str] = set()
        for step in plan.steps:
            if any(dependency not in successful_steps for dependency in step.depends_on):
                result = ActionResult(
                    action_id=step.action_id,
                    success=False,
                    message="Pominięto: zależność nie zakończyła się sukcesem.",
                )
            else:
                result = await self.actions.execute(step)
            results.append(result)
            await self.events.publish(
                "action.completed",
                {
                    "request_id": plan.request_id,
                    "step_id": step.id,
                    "result": result.model_dump(mode="json"),
                },
            )
            if not result.success:
                failed = await self._finish_execution(
                    plan,
                    status=CommandStatus.FAILED,
                    results=results,
                    error=result.message,
                )
                if failed is None:
                    return
                await self.events.publish(
                    "command.completed",
                    {
                        "request_id": plan.request_id,
                        "status": CommandStatus.FAILED.value,
                        "error": result.message,
                    },
                )
                if self.telemetry is not None:
                    await self.telemetry.mark_request(
                        plan.request_id,
                        "tool_completed",
                        metadata={"tool_status": CommandStatus.FAILED.value},
                    )
                if plan.speak_result and not plan.managed_voice_turn:
                    await self.actions.speak(
                        f"Nie udało się wykonać polecenia. {result.message}"[:1000]
                    )
                return
            successful_steps.add(step.id)

        succeeded = await self._finish_execution(
            plan,
            status=CommandStatus.SUCCEEDED,
            results=results,
        )
        if succeeded is None:
            return
        await self.events.publish(
            "command.completed",
            {
                "request_id": plan.request_id,
                "status": CommandStatus.SUCCEEDED.value,
                "results": [result.model_dump(mode="json") for result in results],
            },
        )
        if self.telemetry is not None:
            await self.telemetry.mark_request(
                plan.request_id,
                "tool_completed",
                metadata={"tool_status": CommandStatus.SUCCEEDED.value},
            )
        if plan.speak_result and not plan.managed_voice_turn:
            messages = [result.message.strip() for result in results if result.message.strip()]
            if messages:
                await self.actions.speak(" ".join(dict.fromkeys(messages))[:2000])

    def _completion_future(
        self,
        request_id: str,
    ) -> asyncio.Future[CommandView | None]:
        future = self._completion_futures.get(request_id)
        if future is None or future.cancelled():
            future = asyncio.get_running_loop().create_future()
            self._completion_futures[request_id] = future
        return future

    def _resolve_completion(
        self,
        request_id: str,
        command: CommandView | None,
    ) -> None:
        future = self._completion_futures.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(command)
