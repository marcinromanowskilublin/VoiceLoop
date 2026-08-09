from voiceloop.model_router import ModelRouter, ModelUnavailableError
from voiceloop.models import CommandPlan, CommandRequest


class StubPlanner:
    def __init__(self, provider: str, confidence: float = 1.0, fail: bool = False) -> None:
        self.provider = provider
        self.confidence = confidence
        self.fail = fail
        self.received_memories: list[str] | None = None

    async def plan(self, **kwargs) -> CommandPlan:
        self.received_memories = kwargs["memories"]
        if self.fail:
            raise ModelUnavailableError("offline")
        request = kwargs["request"]
        return CommandPlan(
            request_id=request.request_id,
            intent="answer",
            response_text=self.provider,
            confidence=self.confidence,
            provider=self.provider,
        )


async def test_router_uses_local_by_default() -> None:
    local = StubPlanner("local")
    cloud = StubPlanner("cloud")
    router = ModelRouter(local=local, cloud=cloud)  # type: ignore[arg-type]
    request = CommandRequest(text="test", allow_cloud=False)

    plan = await router.plan(
        request=request,
        history=[],
        memories=["sekret"],
        screen=None,
        image_data_url=None,
        actions=[],
    )

    assert plan.provider == "local"
    assert cloud.received_memories is None


async def test_cloud_escalation_does_not_receive_private_memory() -> None:
    local = StubPlanner("local", confidence=0.1)
    cloud = StubPlanner("cloud")
    router = ModelRouter(local=local, cloud=cloud)  # type: ignore[arg-type]
    request = CommandRequest(text="trudne zadanie", allow_cloud=True)

    plan = await router.plan(
        request=request,
        history=[],
        memories=["sekret"],
        screen=None,
        image_data_url=None,
        actions=[],
    )

    assert plan.provider == "cloud"
    assert cloud.received_memories == []
