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


async def test_vector_memory_search(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()

    await store.upsert_vector_memory(
        source="screenpipe_activity",
        source_id="cursor",
        title="Cursor",
        content="Praca nad projektem VoiceLoop w Cursorze.",
        embedding=[1.0, 0.0, 0.0],
        metadata={"app_name": "Cursor"},
    )
    await store.upsert_vector_memory(
        source="screenpipe_activity",
        source_id="browser",
        title="Browser",
        content="Czytanie wiadomości w przeglądarce.",
        embedding=[0.0, 1.0, 0.0],
    )

    hits = await store.search_vector_memories([0.9, 0.1, 0.0], limit=1)

    assert len(hits) == 1
    assert hits[0].source_id == "cursor"
    assert hits[0].metadata["app_name"] == "Cursor"
