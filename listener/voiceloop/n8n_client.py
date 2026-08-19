from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr, ValidationError

from .models import CommandPlan, CommandRequest, PlanStep, RiskLevel


class N8nUnavailableError(RuntimeError):
    pass


class N8nClient:
    def __init__(
        self,
        *,
        base_url: str,
        webhook_url: str,
        token: SecretStr | None,
        timeout_seconds: float,
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.webhook_url = webhook_url
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            value = self.token.get_secret_value().strip()
            if value:
                headers["X-VoiceLoop-Token"] = value
        return headers

    async def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "wyłączony"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/healthz")
                if response.status_code >= 400:
                    response = await client.get(self.base_url)
                response.raise_for_status()
            return True, f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            return False, str(exc)

    async def route(self, request: CommandRequest) -> CommandPlan | None:
        if not self.enabled:
            return None
        payload = request.model_dump(mode="json")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.webhook_url,
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                data: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise N8nUnavailableError(str(exc)) from exc

        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None
        if "json" in data and isinstance(data["json"], dict):
            data = data["json"]

        if data.get("steps"):
            try:
                plan = CommandPlan.model_validate(data)
                plan.request_id = request.request_id
                plan.provider = "n8n"
                return plan
            except ValidationError:
                return None

        action_id = str(data.get("action_id") or "").strip()
        if not action_id or action_id in {"unknown", "none", "no_action"}:
            return None
        risk_value = data.get("risk", RiskLevel.LOW.value)
        try:
            risk = RiskLevel(risk_value)
        except ValueError:
            risk = RiskLevel.MEDIUM
        step = PlanStep(
            action_id=action_id,
            args=data.get("args") if isinstance(data.get("args"), dict) else {},
            risk=risk,
            confirmation_required=bool(data.get("confirmation_required", False)),
        )
        return CommandPlan(
            request_id=request.request_id,
            intent=str(data.get("intent") or action_id),
            response_text=str(data.get("response_text") or ""),
            confidence=float(data.get("confidence", 1.0)),
            steps=[step],
            provider="n8n",
        )
