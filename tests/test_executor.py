import asyncio

import voiceloop.executor as executor_module
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
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.executed: list[PlanStep] = []

    def enforce_policy(self, step: PlanStep) -> PlanStep:
        return step

    async def bind_execution_targets(self, plan: CommandPlan) -> CommandPlan:
        return plan

    async def execute(self, step: PlanStep) -> ActionResult:
        self.executed.append(step)
        return ActionResult(action_id=step.action_id, success=True, message="ok")

    async def stop(self) -> None:
        return None

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


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
    actions = FakeActions()
    executor = CommandExecutor(
        memory=store,
        actions=actions,  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    await executor.start()
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake")],
        speak_result=True,
    )

    await executor.submit(plan)
    await wait_for_status(store, request.request_id, CommandStatus.SUCCEEDED)
    command = await store.get_command(request.request_id)
    await executor.close()

    assert command is not None
    assert command.results[0].success is True
    assert actions.spoken == ["ok"]


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

    # Confirmation must survive a core restart, where the in-memory map is empty.
    executor.pending_confirmation.clear()
    await executor.confirm(request.request_id)
    await wait_for_status(store, request.request_id, CommandStatus.SUCCEEDED)
    await executor.close()


async def test_executor_confirms_command_once_when_confirmed_concurrently(
    tmp_path,
    monkeypatch,
) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="test")
    await store.create_command(request)
    actions = FakeActions()
    executor = CommandExecutor(
        memory=store,
        actions=actions,  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake", confirmation_required=True)],
    )
    await executor.submit(plan)

    original_get_command = store.get_command
    awaiting_reads = 0
    concurrent_reads = asyncio.Event()
    release_reads = asyncio.Event()

    async def synchronized_get_command(request_id: str):
        nonlocal awaiting_reads
        command = await original_get_command(request_id)
        if command is not None and command.status is CommandStatus.AWAITING_CONFIRMATION:
            awaiting_reads += 1
            if awaiting_reads == 2:
                concurrent_reads.set()
            await release_reads.wait()
        return command

    async def release_synchronized_reads() -> None:
        try:
            await asyncio.wait_for(concurrent_reads.wait(), timeout=0.1)
        except TimeoutError:
            pass
        release_reads.set()

    monkeypatch.setattr(store, "get_command", synchronized_get_command)
    release_task = asyncio.create_task(release_synchronized_reads())
    confirmations = await asyncio.gather(
        executor.confirm(request.request_id),
        executor.confirm(request.request_id),
    )
    await release_task

    assert all(
        command is not None and command.status is CommandStatus.QUEUED
        for command in confirmations
    )
    assert executor.queue.qsize() == 1

    await executor.start()
    await wait_for_status(store, request.request_id, CommandStatus.SUCCEEDED)
    await executor.close()

    assert actions.executed == [plan.steps[0]]


async def test_executor_cancel_wins_concurrent_confirmation(tmp_path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="test")
    await store.create_command(request)
    actions = FakeActions()
    executor = CommandExecutor(
        memory=store,
        actions=actions,  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake", confirmation_required=True)],
    )
    await executor.submit(plan)

    original_get_command = store.get_command
    original_update_command = store.update_command
    confirmation_read = asyncio.Event()
    cancellation_written = asyncio.Event()
    release_confirmation = asyncio.Event()

    async def paused_get_command(request_id: str):
        command = await original_get_command(request_id)
        if (
            command is not None
            and command.status is CommandStatus.AWAITING_CONFIRMATION
            and not confirmation_read.is_set()
        ):
            confirmation_read.set()
            await release_confirmation.wait()
        return command

    async def tracked_update_command(request_id: str, **changes):
        command = await original_update_command(request_id, **changes)
        if changes.get("status") is CommandStatus.CANCELLED:
            cancellation_written.set()
        return command

    async def release_after_cancel_gets_a_turn() -> None:
        try:
            await asyncio.wait_for(cancellation_written.wait(), timeout=0.1)
        except TimeoutError:
            pass
        release_confirmation.set()

    monkeypatch.setattr(store, "get_command", paused_get_command)
    monkeypatch.setattr(store, "update_command", tracked_update_command)
    confirm_task = asyncio.create_task(executor.confirm(request.request_id))
    await confirmation_read.wait()
    cancel_task = asyncio.create_task(executor.cancel(request.request_id))
    release_task = asyncio.create_task(release_after_cancel_gets_a_turn())
    await asyncio.gather(confirm_task, cancel_task, release_task)

    await executor.start()
    await asyncio.wait_for(executor.queue.join(), timeout=1)
    command = await store.get_command(request.request_id)
    await executor.close()

    assert command is not None
    assert command.status is CommandStatus.CANCELLED
    assert actions.executed == []


async def test_executor_does_not_execute_cancelled_queued_command(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="test")
    await store.create_command(request)
    actions = FakeActions()
    executor = CommandExecutor(
        memory=store,
        actions=actions,  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake")],
    )
    queued = await executor.submit(plan)
    assert queued is not None
    assert queued.status is CommandStatus.QUEUED

    cancelled = await executor.cancel(request.request_id)
    assert cancelled is not None
    assert cancelled.status is CommandStatus.CANCELLED

    await executor.start()
    await asyncio.wait_for(executor.queue.join(), timeout=1)
    command = await store.get_command(request.request_id)
    await executor.close()

    assert command is not None
    assert command.status is CommandStatus.CANCELLED
    assert actions.executed == []


async def test_executor_rejects_expired_confirmation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "CONFIRMATION_TTL_SECONDS", -1)
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
    plan = CommandPlan(
        request_id=request.request_id,
        intent="test",
        confidence=1,
        steps=[PlanStep(action_id="fake", confirmation_required=True)],
    )
    await executor.submit(plan)

    expired = await executor.confirm(request.request_id)

    assert expired is not None
    assert expired.status is CommandStatus.CANCELLED
    assert expired.error == "Potwierdzenie wygasło."


async def test_managed_turn_waits_for_result_without_executor_speech(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="test", managed_voice_turn=True)
    await store.create_command(request)
    actions = FakeActions()
    executor = CommandExecutor(
        memory=store,
        actions=actions,  # type: ignore[arg-type]
        events=EventBus(),
        queue_limit=2,
    )
    await executor.start()
    plan = CommandPlan(
        request_id=request.request_id,
        intent="task",
        confidence=1,
        steps=[PlanStep(action_id="fake")],
        speak_result=True,
        managed_voice_turn=True,
    )

    await executor.submit(plan)
    completed = await asyncio.wait_for(
        executor.wait_for_completion(request.request_id),
        timeout=1,
    )
    await executor.close()

    assert completed is not None
    assert completed.status is CommandStatus.SUCCEEDED
    assert completed.results[0].message == "ok"
    assert actions.spoken == []
