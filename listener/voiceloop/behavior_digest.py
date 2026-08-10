from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class BehaviorDigestError(RuntimeError):
    pass


class DigestedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=4000)
    topic: str = Field(max_length=1000)
    intent: str = Field(max_length=1500)
    decision: str = Field(default="", max_length=2000)
    person_context: str = Field(default="", max_length=2000)
    people: list[str] = Field(default_factory=list, max_length=30)
    observations: list[str] = Field(default_factory=list, max_length=30)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def vector_documents(self, *, title: str, raw_content: str) -> dict[str, str]:
        summary = self.summary.strip() or raw_content[:4000]
        topic = self.topic.strip() or title
        intent = self.intent.strip() or f"Zaobserwowana aktywność: {summary}"
        decision = self.decision.strip() or f"Brak jawnej decyzji. Kontekst: {summary}"
        people = ", ".join(item.strip() for item in self.people if item.strip())
        person_context = self.person_context.strip()
        if not person_context:
            person_context = (
                f"Osoby: {people}. {summary}"
                if people
                else f"Brak jawnie rozpoznanej osoby. Kontekst użytkownika: {summary}"
            )
        return {
            "semantic": f"{summary}\n\nMateriał źródłowy:\n{raw_content[:6000]}",
            "topic": f"Temat: {topic}\nTytuł: {title}",
            "intent": f"Intencja lub cel: {intent}\nKontekst: {summary}",
            "decision": f"Decyzje, ustalenia lub następne kroki: {decision}",
            "person_context": f"Kontekst osoby lub relacji: {person_context}",
        }


class LocalBehaviorDigestClient:
    """Structured, local-only behavioral analysis through LM Studio."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str | None,
        timeout_seconds: float,
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.configured_model = model
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self._resolved_model: str | None = None
        self._supports_json_schema: bool | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            key = self.api_key.get_secret_value().strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    async def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączone w konfiguracji"
        try:
            model = await self.resolve_model()
        except BehaviorDigestError as exc:
            return False, str(exc)
        return True, f"lokalny model: {model}"

    async def resolve_model(self) -> str:
        if self.configured_model:
            return self.configured_model
        if self._resolved_model:
            return self._resolved_model
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BehaviorDigestError(f"LM Studio niedostępne: {exc}") from exc
        models = [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict)
            and item.get("id")
            and "embed" not in str(item["id"]).casefold()
        ]
        if not models:
            raise BehaviorDigestError("LM Studio nie ma załadowanego modelu instruct.")
        self._resolved_model = models[0]
        return self._resolved_model

    async def digest(
        self,
        *,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> DigestedMemory:
        if not self.enabled:
            return self.fallback(title=title, content=content)
        model = await self.resolve_model()
        system_prompt = (
            "Analizujesz wyłącznie lokalne dane aktywności jednego użytkownika. "
            "Wyodrębnij obserwowalne zachowania, temat, prawdopodobny cel, jawne decyzje "
            "oraz osoby lub relacje. Nie wymyślaj faktów, których nie ma w materiale. "
            "Wnioski niepewne oznacz niskim confidence. Zwróć wyłącznie jeden krótki "
            "obiekt JSON z dokładnie tymi polami: "
            '{"summary":"tekst","topic":"tekst","intent":"tekst","decision":"tekst",'
            '"person_context":"tekst","people":["tekst"],"observations":["tekst"],'
            '"confidence":0.0}. Używaj pustego tekstu, gdy brak danych. Maksymalnie pięć '
            "krótkich observations. Nie używaj pól title, themes, goals, decisions ani relations."
        )
        context = {
            "title": title[:1000],
            "content": content[:20000],
            "metadata": metadata or {},
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 900,
            "stream": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "voiceloop_behavior_digest",
                "strict": True,
                "schema": DigestedMemory.model_json_schema(),
            },
        }
        if self._supports_json_schema is not False:
            body["response_format"] = response_format
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                )
                if response.status_code == 400 and "response_format" in body:
                    self._supports_json_schema = False
                    body.pop("response_format", None)
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=body,
                    )
                elif response.status_code < 400 and "response_format" in body:
                    self._supports_json_schema = True
                response.raise_for_status()
                payload = response.json()
                content_text = str(payload["choices"][0]["message"]["content"])
            digest_payload = json.loads(self._extract_json(content_text))
            if not isinstance(digest_payload, dict):
                raise ValueError("Analiza zachowania nie jest obiektem JSON.")
            return self._coerce_digest(
                digest_payload,
                title=title,
                source_content=content,
            )
        except httpx.HTTPStatusError as exc:
            raise BehaviorDigestError(
                f"LM Studio zwróciło HTTP {exc.response.status_code}."
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise BehaviorDigestError(
                "LM Studio nie zwróciło poprawnej analizy zachowania."
            ) from exc

    @classmethod
    def _coerce_digest(
        cls,
        payload: dict[str, Any],
        *,
        title: str,
        source_content: str,
    ) -> DigestedMemory:
        themes = cls._list_of_text(payload.get("themes") or payload.get("theme"))
        goals = cls._list_of_text(
            payload.get("goals") or payload.get("goal") or payload.get("objective")
        )
        decisions = cls._list_of_text(
            payload.get("decisions")
            or payload.get("decision")
            or payload.get("action_items")
        )
        relations = cls._list_of_text(
            payload.get("relations")
            or payload.get("person_context")
            or payload.get("entities")
        )
        observations = cls._list_of_text(
            payload.get("observations") or payload.get("events")
        )

        summary = cls._text(payload.get("summary") or payload.get("overview"))
        if not summary:
            summary = "; ".join((themes + goals + decisions + observations)[:8])
        summary = (summary or source_content.strip() or title.strip() or "Brak treści.")[:4000]
        topic = (
            cls._text(payload.get("topic"))
            or (themes[0] if themes else "")
            or title.strip()
            or "Aktywność użytkownika"
        )[:1000]
        intent = (
            cls._text(payload.get("intent"))
            or (goals[0] if goals else "")
            or f"Kontynuowanie aktywności: {topic}"
        )[:1500]
        decision = "; ".join(decisions)[:2000]
        person_context = "; ".join(relations)[:2000]

        people: list[str] = []
        raw_people = payload.get("people")
        if isinstance(raw_people, list):
            for item in raw_people:
                value = cls._text(item)
                if value and value not in people:
                    people.append(value[:300])
        elif raw_people is not None:
            value = cls._text(raw_people)
            if value:
                people.append(value[:300])

        return DigestedMemory(
            summary=summary,
            topic=topic,
            intent=intent,
            decision=decision,
            person_context=person_context,
            people=people[:30],
            observations=[item[:1000] for item in observations[:10]],
            confidence=cls._confidence(payload),
        )

    @classmethod
    def _text(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            parts: list[str] = []
            for key in (
                "name",
                "title",
                "summary",
                "description",
                "text",
                "content",
                "goal",
                "decision",
                "relation",
            ):
                part = cls._text(value.get(key))
                if part and part not in parts:
                    parts.append(part)
            timestamp = cls._text(value.get("timestamp") or value.get("time"))
            text = " — ".join(parts)
            return f"[{timestamp}] {text}".strip() if timestamp and text else text
        if isinstance(value, list):
            return "; ".join(item for item in (cls._text(entry) for entry in value) if item)
        return ""

    @classmethod
    def _list_of_text(cls, value: Any) -> list[str]:
        values = value if isinstance(value, list) else [value]
        results: list[str] = []
        for item in values:
            text = cls._text(item)
            if text and text not in results:
                results.append(text)
        return results

    @classmethod
    def _confidence(cls, payload: dict[str, Any]) -> float:
        values: list[float] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                raw = value.get("confidence")
                if isinstance(raw, int | float) and not isinstance(raw, bool):
                    values.append(float(raw))
                for nested in value.values():
                    if isinstance(nested, list):
                        collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)

        collect(payload)
        if not values:
            return 0.5
        return max(0.0, min(sum(values) / len(values), 1.0))

    @staticmethod
    def fallback(*, title: str, content: str) -> DigestedMemory:
        summary = content.strip()[:4000] or title.strip() or "Brak treści."
        return DigestedMemory(
            summary=summary,
            topic=title.strip() or "Aktywność użytkownika",
            intent=f"Kontynuowanie aktywności opisanej jako: {title.strip()}",
            decision="",
            person_context="",
            people=[],
            observations=[summary[:1000]],
            confidence=0.25,
        )

    @staticmethod
    def _extract_json(content: str) -> str:
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
        if start < 0 or end <= start:
            raise ValueError("Brak obiektu JSON w odpowiedzi.")
        candidate = cleaned[start : end + 1]
        json.loads(candidate)
        return candidate
