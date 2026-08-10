from voiceloop.model_router import (
    ModelRouter,
    ModelUnavailableError,
    OpenAICompatiblePlanner,
)
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


async def test_cloud_primary_does_not_require_allow_cloud() -> None:
    cloud = StubPlanner("cloud")
    local = StubPlanner("local")
    router = ModelRouter(  # type: ignore[arg-type]
        local=cloud,
        cloud=local,
        fallback_requires_allow_cloud=False,
    )
    request = CommandRequest(text="test", allow_cloud=False)

    plan = await router.plan(
        request=request,
        history=[],
        memories=["kontekst"],
        screen=None,
        image_data_url=None,
        actions=[],
    )

    assert plan.provider == "cloud"
    assert cloud.received_memories == ["kontekst"]
    assert local.received_memories is None


async def test_cloud_primary_can_fallback_to_local_without_allow_cloud() -> None:
    cloud = StubPlanner("cloud", confidence=0.1)
    local = StubPlanner("local")
    router = ModelRouter(  # type: ignore[arg-type]
        local=cloud,
        cloud=local,
        fallback_requires_allow_cloud=False,
    )
    request = CommandRequest(text="test", allow_cloud=False)

    plan = await router.plan(
        request=request,
        history=[],
        memories=["kontekst"],
        screen=None,
        image_data_url=None,
        actions=[],
    )

    assert plan.provider == "local"
    assert local.received_memories == []


def test_conversation_style_switches_by_prefix() -> None:
    assert OpenAICompatiblePlanner._conversation_style("Venice opowiedz o AI") == "max_iq"
    assert OpenAICompatiblePlanner._conversation_style("venive wyjasnij to szeroko") == "max_iq"
    assert OpenAICompatiblePlanner._conversation_style("Asystencie jaka jest pogoda") == "concise"
    assert OpenAICompatiblePlanner._conversation_style("Powiedz cos o AI") == "default"


def test_strip_conversation_prefix_removes_wake_word() -> None:
    assert (
        OpenAICompatiblePlanner._strip_conversation_prefix(
            "Venice, powiedz mi jaka jest stolica Mozambiku"
        )
        == "powiedz mi jaka jest stolica Mozambiku"
    )
    assert (
        OpenAICompatiblePlanner._strip_conversation_prefix(
            "asystencie: jaka jest stolica mozambiku"
        )
        == "jaka jest stolica mozambiku"
    )


def test_style_controls_token_budget() -> None:
    assert OpenAICompatiblePlanner._max_tokens_for_style("max_iq") > 1600
    assert OpenAICompatiblePlanner._max_tokens_for_style("concise") < 1600
    assert OpenAICompatiblePlanner._max_tokens_for_style("default") == 1600


def test_layering_instruction_mentions_execution_priority() -> None:
    instruction = OpenAICompatiblePlanner._layering_instruction()
    assert "execution_layer=1" in instruction
    assert "execution_layer=2" in instruction
    assert "execution_layer=3" in instruction
    assert "fallback" in instruction
