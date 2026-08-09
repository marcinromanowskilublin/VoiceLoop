from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .models import CommandPlan, CommandRequest, PlanStep, RiskLevel, ScreenSnapshot


class ProposedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW
    confirmation_required: bool = False
    success_condition: str | None = None


class ProposedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    response_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_clarification: bool = False
    clarification_question: str | None = None
    steps: list[ProposedStep] = Field(default_factory=list, max_length=12)


class ModelUnavailableError(RuntimeError):
    pass


class OpenAICompatiblePlanner:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: SecretStr | None,
        model: str | None,
        timeout_seconds: float,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.configured_model = model
        self.timeout_seconds = timeout_seconds
        self._resolved_model: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            key = self.api_key.get_secret_value().strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(f"{self.provider} unavailable: {exc}") from exc
        payload = response.json()
        return [str(item["id"]) for item in payload.get("data", []) if item.get("id")]

    async def resolve_model(self) -> str:
        if self.configured_model:
            return self.configured_model
        if self._resolved_model:
            return self._resolved_model
        models = await self.list_models()
        if not models:
            raise ModelUnavailableError(f"{self.provider} has no loaded model")
        self._resolved_model = models[0]
        return self._resolved_model

    async def health(self) -> tuple[bool, str]:
        try:
            models = await self.list_models()
        except ModelUnavailableError as exc:
            return False, str(exc)
        if not models:
            return False, "API works, but no model is loaded"
        return True, f"loaded: {', '.join(models)}"

    async def plan(
        self,
        *,
        request: CommandRequest,
        history: list[dict[str, str]],
        memories: list[str],
        screen: ScreenSnapshot | None,
        image_data_url: str | None,
        actions: list[dict[str, Any]],
    ) -> CommandPlan:
        model = await self.resolve_model()
        action_ids = {item["id"] for item in actions}
        system_prompt = (
            "Jesteś planerem lokalnego asystenta Windows VoiceLoop. Odpowiadasz po polsku. "
            "Twórz wyłącznie plan z narzędzi podanych w ACTIONS. Nie wymyślaj nazw narzędzi, "
            "nie generuj poleceń shell i nie umieszczaj sekretów w odpowiedzi. "
            "Jeżeli brakuje danych, ustaw requires_clarification=true i zadaj jedno precyzyjne "
            "pytanie. Operacje wysyłania, publikowania, usuwania, zakupów, logowania lub zmiany "
            "kont zawsze oznacz jako high oraz confirmation_required=true. "
            "Dla zwykłej odpowiedzi bez akcji zwróć pustą listę steps."
        )
        context = {
            "request": {
                "text": request.text,
                "command_id": request.command_id,
                "source": request.source.value,
            },
            "actions": actions,
            "memories": memories[-20:],
            "screen": screen.model_dump(mode="json", exclude={"image_path"}) if screen else None,
        }
        user_text = "Zaplanuj wykonanie tej prośby:\n" + json.dumps(
            context, ensure_ascii=False, separators=(",", ":")
        )
        user_content: str | list[dict[str, Any]] = user_text
        if image_data_url:
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in history[-12:]:
            if message.get("role") in {"user", "assistant"} and message.get("content"):
                messages.append(
                    {"role": message["role"], "content": str(message["content"])[:6000]}
                )
        messages.append({"role": "user", "content": user_content})

        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1600,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "voiceloop_command_plan",
                    "strict": True,
                    "schema": ProposedPlan.model_json_schema(),
                },
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                )
                if response.status_code == 400 and image_data_url:
                    body["messages"][-1]["content"] = user_text
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=body,
                    )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                proposed = ProposedPlan.model_validate_json(content)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise ModelUnavailableError(
                f"{self.provider} planning failed ({exc.response.status_code}): {detail}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ModelUnavailableError(f"{self.provider} planning failed: {exc}") from exc

        accepted_steps: list[tuple[int, ProposedStep, PlanStep]] = []
        index_to_id: dict[int, str] = {}
        for index, proposed_step in enumerate(proposed.steps):
            if proposed_step.action_id not in action_ids:
                continue
            step = PlanStep(
                action_id=proposed_step.action_id,
                args=proposed_step.args,
                risk=proposed_step.risk,
                confirmation_required=proposed_step.confirmation_required,
                success_condition=proposed_step.success_condition,
            )
            index_to_id[index] = step.id
            accepted_steps.append((index, proposed_step, step))
        for _, original, step in accepted_steps:
            step.depends_on = [
                index_to_id[dependency]
                for dependency in original.depends_on
                if dependency in index_to_id
            ]
        steps = [step for _, _, step in accepted_steps]

        return CommandPlan(
            request_id=request.request_id,
            intent=proposed.intent,
            response_text=proposed.response_text,
            confidence=proposed.confidence,
            requires_clarification=proposed.requires_clarification,
            clarification_question=proposed.clarification_question,
            steps=steps,
            provider=self.provider,
            model=model,
        )


class ModelRouter:
    def __init__(
        self,
        local: OpenAICompatiblePlanner,
        cloud: OpenAICompatiblePlanner | None = None,
    ) -> None:
        self.local = local
        self.cloud = cloud

    async def plan(
        self,
        *,
        request: CommandRequest,
        history: list[dict[str, str]],
        memories: list[str],
        screen: ScreenSnapshot | None,
        image_data_url: str | None,
        actions: list[dict[str, Any]],
    ) -> CommandPlan:
        local_error: Exception | None = None
        try:
            local_plan = await self.local.plan(
                request=request,
                history=history,
                memories=memories,
                screen=screen,
                image_data_url=image_data_url,
                actions=actions,
            )
            if local_plan.confidence >= 0.45 or not request.allow_cloud or self.cloud is None:
                return local_plan
        except ModelUnavailableError as exc:
            local_error = exc
            if not request.allow_cloud or self.cloud is None:
                raise

        if self.cloud is not None and request.allow_cloud:
            return await self.cloud.plan(
                request=request,
                history=[],
                memories=[],
                screen=None,
                image_data_url=None,
                actions=actions,
            )

        if local_error:
            raise ModelUnavailableError(str(local_error))
        raise ModelUnavailableError("no model provider is available")
