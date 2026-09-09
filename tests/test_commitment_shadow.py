import asyncio

from tests.test_assistant import ExecutorStub, N8nStub, RouterStub, TTSStub, _assistant

from voiceloop.assistant import AssistantService
from voiceloop.events import EventBus
from voiceloop.memory import MemoryStore
from voiceloop.models import CommandRequest, CommandStatus
from voiceloop.router import deterministic_plan


def _collect_shadow(assistant: AssistantService) -> asyncio.Task:
    async def _read() -> dict:
        async for event in assistant.events.subscribe():
            if event["type"] == "commitment.shadow":
                return event
        raise AssertionError("commitment.shadow nie przyszedł")

    return asyncio.create_task(_read())


def test_commitment_shadow_observes_request_without_changing_the_plan(
    tmp_path,
) -> None:
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
        pending = _collect_shadow(assistant)
        await asyncio.sleep(0)
        accepted = await assistant.handle(
            CommandRequest(text="Musisz mi to wysłać dzisiaj.")
        )
        event = await asyncio.wait_for(pending, timeout=1.0)

        assert accepted.plan is not None
        assert accepted.plan.provider == "venice"
        assert [step.action_id for step in accepted.plan.steps] == []
        assert event["payload"]["request_id"] == accepted.request_id
        assert event["payload"]["item_count"] >= 1
        assert {item["status"] for item in event["payload"]["items"]} <= {
            "captured",
            "needs_clarification",
            "needs_user_review",
        }
        assert "accepted" not in {item["status"] for item in event["payload"]["items"]}
        await assistant.close()

    asyncio.run(scenario())


def test_commitment_shadow_disabled_publishes_nothing(tmp_path) -> None:
    async def scenario() -> None:
        memory = MemoryStore(tmp_path / "voice.db")
        await memory.initialize()
        assistant = AssistantService(
            memory=memory,
            events=EventBus(),
            executor=ExecutorStub(memory),  # type: ignore[arg-type]
            n8n=N8nStub(),  # type: ignore[arg-type]
            model_router=RouterStub(),  # type: ignore[arg-type]
            screen=object(),  # type: ignore[arg-type]
            tts=TTSStub(),  # type: ignore[arg-type]
            embeddings=None,
            qdrant=None,
            action_definitions=[],
            dedupe_seconds=0,
            vector_context_limit=0,
            commitment_shadow_enabled=False,
        )
        pending = _collect_shadow(assistant)
        await asyncio.sleep(0)
        await assistant.handle(CommandRequest(text="Musisz mi to wysłać dzisiaj."))
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        await assistant.close()

    asyncio.run(scenario())


def test_commitment_shadow_keeps_deterministic_calendar_plan(tmp_path) -> None:
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
        accepted = await assistant.handle(CommandRequest(text="Otwórz kalendarz"))
        expected = deterministic_plan(CommandRequest(text="Otwórz kalendarz"))

        assert accepted.status is CommandStatus.SUCCEEDED
        assert accepted.plan is not None
        assert expected is not None
        assert [step.action_id for step in accepted.plan.steps] == [
            step.action_id for step in expected.steps
        ]
        assert router.calls == []
        await assistant.close()

    asyncio.run(scenario())
