from __future__ import annotations

import asyncio
from contextlib import suppress

from .actions import ActionRegistry
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
    ) -> None:
        self.memory = memory
        self.actions = actions
        self.events = events
        self.queue: asyncio.Queue[CommandPlan] = asyncio.Queue(maxsize=queue_limit)
        self.pending_confirmation: dict[str, CommandPlan] = {}
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
        for step in plan.steps:
            self.actions.enforce_policy(step)

        if plan.requires_clarification:
            return await self.memory.update_command(
                plan.request_id,
                status=CommandStatus.AWAITING_CONFIRMATION,
                plan=plan,
            )

        if not plan.steps:
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
        return await self._enqueue(plan)

    async def cancel(self, request_id: str) -> CommandView | None:
        self.pending_confirmation.pop(request_id, None)
        if self._current_request_id == request_id and self._current_execution:
            self._current_execution.cancel()
            await self.actions.stop()
        command = await self.memory.update_command(
            request_id,
            status=CommandStatus.CANCELLED,
            error="Anulowano przez użytkownika.",
        )
        await self.events.publish(
            "command.cancelled",
            {"request_id": request_id},
        )
        return command

    async def stop_all(self) -> None:
        for request_id in tuple(self.pending_confirmation):
            await self.cancel(request_id)
        while True:
            try:
                plan = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self.memory.update_command(
                plan.request_id,
                status=CommandStatus.CANCELLED,
                error="Anulowano przez STOP.",
            )
            self.queue.task_done()
        if self._current_execution:
            self._current_execution.cancel()
            with suppress(asyncio.CancelledError):
                await self._current_execution
        await self.actions.stop()
        await self.events.publish("stop", {"status": "all_cancelled"})

    async def _enqueue(self, plan: CommandPlan) -> CommandView | None:
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
            self._current_request_id = plan.request_id
            self._current_execution = asyncio.create_task(
                self._execute_plan(plan),
                name=f"voiceloop-command-{plan.request_id}",
            )
            try:
                await self._current_execution
            except asyncio.CancelledError:
                await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.CANCELLED,
                    error="Wykonanie przerwane.",
                )
            except Exception as exc:
                await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.FAILED,
                    error=f"Błąd executora: {exc}",
                )
                await self.events.publish(
                    "command.completed",
                    {
                        "request_id": plan.request_id,
                        "status": CommandStatus.FAILED.value,
                        "error": str(exc),
                    },
                )
            finally:
                self._current_execution = None
                self._current_request_id = None
                self.queue.task_done()

    async def _execute_plan(self, plan: CommandPlan) -> None:
        await self.memory.update_command(
            plan.request_id,
            status=CommandStatus.EXECUTING,
            plan=plan,
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
                await self.memory.update_command(
                    plan.request_id,
                    status=CommandStatus.FAILED,
                    plan=plan,
                    results=results,
                    error=result.message,
                )
                await self.events.publish(
                    "command.completed",
                    {
                        "request_id": plan.request_id,
                        "status": CommandStatus.FAILED.value,
                        "error": result.message,
                    },
                )
                if plan.speak_result:
                    await self.actions.speak(
                        f"Nie udało się wykonać polecenia. {result.message}"[:1000]
                    )
                return
            successful_steps.add(step.id)

        await self.memory.update_command(
            plan.request_id,
            status=CommandStatus.SUCCEEDED,
            plan=plan,
            results=results,
        )
        await self.events.publish(
            "command.completed",
            {
                "request_id": plan.request_id,
                "status": CommandStatus.SUCCEEDED.value,
                "results": [result.model_dump(mode="json") for result in results],
            },
        )
        if plan.speak_result:
            messages = [result.message.strip() for result in results if result.message.strip()]
            if messages:
                await self.actions.speak(" ".join(dict.fromkeys(messages))[:2000])
