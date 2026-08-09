import asyncio

from voiceloop.events import EventBus
from voiceloop.executor import CommandExecutor
from voiceloop.memory import MemoryStore
from voiceloop.models import (
    ActionResult,
    CommandPlan,
    CommandRequest,
    CommandStatus,
    PlanStep,
)


class FakeActions:
    def enforce_policy(self, step: PlanStep) -> PlanStep:
        return step

    async def execute(self, step: PlanStep) -> ActionResult:
        return ActionResult(action_id=step.action_id, success=True, message="ok")

    async def stop(self) -> None:
        return None


async def wait_for_status(
    store: MemoryStore,
    request_id: str,
    expected: CommandStatus,
) -> None:
    command = None
    for _ in range(100):
        command = await store.get_command(request_id)
        if command and command.status is expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"Command did not reach {expected}; current={command.status if command else None}; "
        f"error={command.error if command else None}"
    )


async def test_executor_runs_plan(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="test")
    await store.create_command(request)
    executor = CommandExecutor(
        memory=store,
        actions=FakeActions(),  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    await executor.start()
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake")],
    )

    await executor.submit(plan)
    await wait_for_status(store, request.request_id, CommandStatus.SUCCEEDED)
    command = await store.get_command(request.request_id)
    await executor.close()

    assert command is not None
    assert command.results[0].success is True


async def test_executor_waits_for_confirmation(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="test")
    await store.create_command(request)
    executor = CommandExecutor(
        memory=store,
        actions=FakeActions(),  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    await executor.start()
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake", confirmation_required=True)],
    )

    waiting = await executor.submit(plan)
    assert waiting is not None
    assert waiting.status is CommandStatus.AWAITING_CONFIRMATION

    await executor.confirm(request.request_id)
    await wait_for_status(store, request.request_id, CommandStatus.SUCCEEDED)
    await executor.close()
