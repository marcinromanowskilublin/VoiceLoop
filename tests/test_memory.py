from voiceloop.memory import MemoryStore
from voiceloop.models import (
    CommandPlan,
    CommandRequest,
    CommandStatus,
    MemoryCreate,
    PlanStep,
)


async def test_command_lifecycle(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text="otwórz kalendarz")
    await store.create_command(request)
    plan = CommandPlan(
        request_id=request.request_id,
        intent="open_calendar",
        response_text="Otwieram.",
        confidence=1,
        steps=[PlanStep(action_id="open_calendar")],
    )

    updated = await store.update_command(
        request.request_id,
        status=CommandStatus.QUEUED,
        plan=plan,
    )

    assert updated is not None
    assert updated.status is CommandStatus.QUEUED
    assert updated.plan == plan
    assert (await store.recent_commands())[0].request_id == request.request_id


async def test_memory_create_list_delete(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    created = await store.create_memory(
        MemoryCreate(kind="preference", content="Lubię ciemny motyw.")
    )

    items = await store.list_memories(kind="preference")
    deleted = await store.delete_memory(created.id)

    assert [item.content for item in items] == ["Lubię ciemny motyw."]
    assert deleted is True
    assert await store.list_memories() == []
