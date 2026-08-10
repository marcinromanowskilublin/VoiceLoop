from __future__ import annotations

import json
import re
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
        if self.configured_model and self.configured_model not in models:
            return False, f"configured model is unavailable: {self.configured_model}"
        selected = self.configured_model or models[0]
        return True, f"ready: {selected}"

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
        conversation_style = self._conversation_style(request.text or "")
        request_text = self._strip_conversation_prefix((request.text or "").strip())
        if not request_text:
            request_text = (request.text or "").strip()
        system_prompt = (
            "Jesteś planerem lokalnego asystenta Windows VoiceLoop. Odpowiadasz po polsku. "
            "Twórz wyłącznie plan z narzędzi podanych w ACTIONS. Nie wymyślaj nazw narzędzi, "
            "nie generuj poleceń shell i nie umieszczaj sekretów w odpowiedzi. "
            "Jeżeli brakuje danych, ustaw requires_clarification=true i zadaj jedno precyzyjne "
            "pytanie. Operacje wysyłania, publikowania, usuwania, zakupów, logowania lub zmiany "
            "kont zawsze oznacz jako high oraz confirmation_required=true. "
            "Dla zwykłej odpowiedzi bez akcji zwróć pustą listę steps."
        )
        system_prompt = f"{system_prompt} {self._layering_instruction()}"
        style_instruction = self._style_instruction(conversation_style)
        if style_instruction:
            system_prompt = f"{system_prompt} {style_instruction}"
        context = {
            "request": {
                "text": request_text,
                "command_id": request.command_id,
                "source": request.source.value,
                "conversation_style": conversation_style,
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
            "max_tokens": self._max_tokens_for_style(conversation_style),
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
                try:
                    response.raise_for_status()
                    payload = response.json()
                    content = payload["choices"][0]["message"]["content"]
                    proposed = self._coerce_proposed_plan(str(content))
                except httpx.HTTPStatusError as exc:
                    if not self._is_lmstudio_grammar_error(exc):
                        raise
                    schema_json = json.dumps(ProposedPlan.model_json_schema(), ensure_ascii=False)
                    fallback_body = {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    f"{system_prompt} "
                                    "Zwróć WYŁĄCZNIE poprawny obiekt JSON bez markdown i bez "
                                    "dodatkowego tekstu, zgodny ze schematem: "
                                    f"{schema_json}"
                                ),
                            },
                            *messages[1:],
                        ],
                        "temperature": 0,
                        "max_tokens": self._max_tokens_for_style(conversation_style),
                        "stream": False,
                    }
                    fallback_response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=fallback_body,
                    )
                    fallback_response.raise_for_status()
                    fallback_payload = fallback_response.json()
                    fallback_content = fallback_payload["choices"][0]["message"]["content"]
                    proposed = self._coerce_proposed_plan(str(fallback_content))
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

        intent = proposed.intent
        response_text = proposed.response_text
        confidence = proposed.confidence
        if self._should_force_conversation(
            request,
            proposed,
            steps,
            request_text=request_text,
        ):
            intent = "conversation"
            steps = []
            confidence = min(confidence, 0.75)
            if self._is_generic_plan_response(response_text):
                response_text = (
                    "Rozumiem. Odpowiadam tekstowo i nie wykonuję żadnej akcji."
                )
        if intent == "conversation" and not steps and self._is_generic_plan_response(response_text):
            response_text = "Jasne. Odpowiadam tekstowo i czekam na Twoje pytanie."

        return CommandPlan(
            request_id=request.request_id,
            intent=intent,
            response_text=response_text,
            confidence=confidence,
            requires_clarification=proposed.requires_clarification,
            clarification_question=proposed.clarification_question,
            steps=steps,
            provider=self.provider,
            model=model,
        )

    @staticmethod
    def _is_lmstudio_grammar_error(exc: httpx.HTTPStatusError) -> bool:
        if exc.response.status_code != 400:
            return False
        detail = (exc.response.text or "").lower()
        return "failed to initialize samplers" in detail and "empty grammar stack" in detail

    @staticmethod
    def _extract_json_payload(content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            json.loads(cleaned)
            return cleaned
        except ValueError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response does not contain JSON object")
        candidate = cleaned[start : end + 1]
        json.loads(candidate)
        return candidate

    @staticmethod
    def _parse_relaxed_proposed_plan(content: str) -> ProposedPlan:
        payload = OpenAICompatiblePlanner._extract_json_payload(content)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Model JSON payload is not an object")
        normalized = {
            "intent": str(data.get("intent") or "conversation"),
            "response_text": str(
                data.get("response_text") or data.get("clarification_question") or ""
            )[:2000],
            "confidence": max(0.0, min(float(data.get("confidence", 0.6)), 1.0)),
            "requires_clarification": bool(data.get("requires_clarification", False)),
            "clarification_question": (
                str(data.get("clarification_question"))
                if data.get("clarification_question") is not None
                else None
            ),
            "steps": data.get("steps") if isinstance(data.get("steps"), list) else [],
        }
        if normalized["clarification_question"] and not normalized["response_text"]:
            normalized["response_text"] = normalized["clarification_question"]
        if normalized["clarification_question"]:
            normalized["requires_clarification"] = True
        return ProposedPlan.model_validate(normalized)

    @staticmethod
    def _coerce_proposed_plan(content: str) -> ProposedPlan:
        try:
            return ProposedPlan.model_validate_json(content)
        except ValueError:
            pass
        try:
            return OpenAICompatiblePlanner._parse_relaxed_proposed_plan(content)
        except ValueError:
            pass

        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        safe_text = text[:2000] or "Nie udało się uzyskać poprawnej odpowiedzi modelu."
        return ProposedPlan(
            intent="conversation",
            response_text=safe_text,
            confidence=0.35,
            requires_clarification=False,
            clarification_question=None,
            steps=[],
        )

    @staticmethod
    def _should_force_conversation(
        request: CommandRequest,
        proposed: ProposedPlan,
        steps: list[PlanStep],
        *,
        request_text: str | None = None,
    ) -> bool:
        text = (request_text if request_text is not None else (request.text or "")).strip().lower()
        if request.command_id or not text or not steps:
            return False
        if OpenAICompatiblePlanner._is_explicit_action_request(text):
            return False
        if proposed.requires_clarification:
            return False
        return (
            "?" in text
            or any(
                token in text
                for token in (
                    "gdzie",
                    "kiedy",
                    "dlaczego",
                    "jak ",
                    "co ",
                    "czy ",
                    "opowiedz",
                    "wytlumacz",
                )
            )
            or proposed.intent in {"plan", "execute_request"}
        )

    @staticmethod
    def _is_explicit_action_request(text: str) -> bool:
        return bool(
            re.search(
                (
                    r"\b("
                    r"otworz|uruchom|wlacz|włącz|zamknij|stop|przerwij|"
                    r"zapisz|dodaj|utworz|stworz|zapamietaj|remember|"
                    r"run|execute|open|launch|click|macro|uivision|"
                    r"kalendarz|przegladark|notatk|chat"
                    r")\b"
                ),
                text,
            )
        )

    @staticmethod
    def _is_generic_plan_response(text: str) -> bool:
        normalized = (text or "").strip().lower()
        return (
            not normalized
            or normalized.startswith("zaplanowano wykonanie")
            or normalized.startswith("zaplanuję wykonanie")
        )

    @staticmethod
    def _conversation_style(text: str) -> str:
        cleaned = (text or "").strip().casefold()
        if not cleaned:
            return "default"
        if re.match(r"^(?:venice|venive|wenice)\b", cleaned):
            return "max_iq"
        if re.match(r"^(?:asystencie|asystent|assistant)\b", cleaned):
            return "concise"
        return "default"

    @staticmethod
    def _strip_conversation_prefix(text: str) -> str:
        if not text:
            return text
        return re.sub(
            r"^\s*(?:venice|venive|wenice|asystencie|asystent|assistant)\b[\s,.:;!?-]*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    @staticmethod
    def _style_instruction(style: str) -> str:
        if style == "max_iq":
            return (
                "Gdy odpowiedź ma intent='conversation' i brak kroków, napisz możliwie "
                "najbardziej szczegółowo, długo i merytorycznie. Użyj konkretów, "
                "uporządkowanej struktury, krótkiego uzasadnienia i praktycznych przykładów. "
                "Nie skracaj odpowiedzi do jednego zdania."
            )
        if style == "concise":
            return (
                "Gdy odpowiedź ma intent='conversation' i brak kroków, odpowiadaj zwięźle: "
                "najpierw konkret w 1-2 zdaniach, potem opcjonalnie 1 krótkie doprecyzowanie."
            )
        return ""

    @staticmethod
    def _layering_instruction() -> str:
        return (
            "Stosuj warstwy wykonania: najpierw execution_layer=1 (natywne akcje systemowe), "
            "potem execution_layer=2 (UI Automation), a execution_layer=3 (UI.Vision/RPA) "
            "wyłącznie jako fallback, gdy niższe warstwy nie mają odpowiedniej akcji. "
            "Dla złożonych poleceń głosowych rozbij zadanie na krótkie kroki i używaj tylko "
            "zdefiniowanych akcji. Nie planuj bezpośrednich kliknięć po pikselach."
        )

    @staticmethod
    def _max_tokens_for_style(style: str) -> int:
        if style == "max_iq":
            return 2600
        if style == "concise":
            return 900
        return 1600


class ModelRouter:
    def __init__(
        self,
        local: OpenAICompatiblePlanner,
        cloud: OpenAICompatiblePlanner | None = None,
        *,
        fallback_requires_allow_cloud: bool = True,
    ) -> None:
        self.local = local
        self.cloud = cloud
        self.fallback_requires_allow_cloud = fallback_requires_allow_cloud

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
            fallback_allowed = (
                self.cloud is not None
                and (request.allow_cloud or not self.fallback_requires_allow_cloud)
            )
            if local_plan.confidence >= 0.45 or not fallback_allowed:
                return local_plan
        except ModelUnavailableError as exc:
            local_error = exc
            fallback_allowed = (
                self.cloud is not None
                and (request.allow_cloud or not self.fallback_requires_allow_cloud)
            )
            if not fallback_allowed:
                raise

        if self.cloud is not None and (
            request.allow_cloud or not self.fallback_requires_allow_cloud
        ):
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
