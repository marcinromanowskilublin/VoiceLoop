from __future__ import annotations

import asyncio
import os
import re
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .memory import MemoryStore
from .models import ActionResult, MemoryCreate, PlanStep, RiskLevel
from .settings import Settings
from .tts import WindowsTTS

ActionHandler = Callable[[dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]
RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


@dataclass(frozen=True)
class ActionSpec:
    id: str
    description: str
    args_schema: dict[str, Any]
    risk: RiskLevel
    confirmation_required: bool
    handler: ActionHandler

    def public_definition(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "args_schema": self.args_schema,
            "risk": self.risk.value,
            "confirmation_required": self.confirmation_required,
        }


class ActionRegistry:
    def __init__(self, settings: Settings, memory: MemoryStore, tts: WindowsTTS) -> None:
        self.settings = settings
        self.memory = memory
        self.tts = tts
        self._current_process: asyncio.subprocess.Process | None = None
        self._specs: dict[str, ActionSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._register(
            ActionSpec(
                id="open_calendar",
                description="Otwiera lokalny kalendarz Windows.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_calendar,
            )
        )
        self._register(
            ActionSpec(
                id="open_browser",
                description="Otwiera domyślną przeglądarkę.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_browser,
            )
        )
        self._register(
            ActionSpec(
                id="open_url",
                description="Otwiera jawny adres HTTP lub HTTPS.",
                args_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_url,
            )
        )
        self._register(
            ActionSpec(
                id="open_chat",
                description="Otwiera stronę ChatGPT.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._open_chat,
            )
        )
        self._register(
            ActionSpec(
                id="create_note",
                description="Uruchamia UI.Vision i wpisuje przekazaną treść do Notatnika.",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=False,
                handler=self._create_note,
            )
        )
        self._register(
            ActionSpec(
                id="run_uivision_macro",
                description="Uruchamia istniejące, dozwolone makro UI.Vision.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "macro": {"type": "string"},
                        "var1": {"type": "string"},
                        "var2": {"type": "string"},
                        "var3": {"type": "string"},
                    },
                    "required": ["macro"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._run_uivision_macro,
            )
        )
        self._register(
            ActionSpec(
                id="remember",
                description="Zapisuje fakt lub preferencję w lokalnej pamięci.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "kind": {"type": "string"},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.MEDIUM,
                confirmation_required=True,
                handler=self._remember,
            )
        )
        self._register(
            ActionSpec(
                id="recall",
                description="Wyszukuje pasujące wpisy lokalnej pamięci.",
                args_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._recall,
            )
        )
        self._register(
            ActionSpec(
                id="speak_text",
                description="Wypowiada krótki tekst po polsku przez lokalny głos Windows.",
                args_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk=RiskLevel.LOW,
                confirmation_required=False,
                handler=self._speak_text,
            )
        )

    def _register(self, spec: ActionSpec) -> None:
        self._specs[spec.id] = spec

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.public_definition() for spec in self._specs.values()]

    def has_action(self, action_id: str) -> bool:
        return action_id in self._specs

    def enforce_policy(self, step: PlanStep) -> PlanStep:
        spec = self._specs.get(step.action_id)
        if spec is None:
            raise ValueError(f"unknown action: {step.action_id}")
        if RISK_ORDER[spec.risk] > RISK_ORDER[step.risk]:
            step.risk = spec.risk
        step.confirmation_required = (
            step.confirmation_required or spec.confirmation_required or step.risk is RiskLevel.HIGH
        )
        return step

    async def execute(self, step: PlanStep) -> ActionResult:
        spec = self._specs.get(step.action_id)
        if spec is None:
            return ActionResult(
                action_id=step.action_id,
                success=False,
                message=f"Nieznana akcja: {step.action_id}",
            )
        started = time.perf_counter()
        try:
            message, data = await spec.handler(step.args)
            success = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc)
            data = {}
            success = False
        duration_ms = int((time.perf_counter() - started) * 1000)
        return ActionResult(
            action_id=step.action_id,
            success=success,
            message=message,
            data=data,
            duration_ms=duration_ms,
        )

    async def stop(self) -> None:
        await self.tts.stop()
        process = self._current_process
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._current_process = None

    async def _open_calendar(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        await asyncio.to_thread(os.startfile, "outlookcal:")
        return "Otwarto kalendarz.", {}

    async def _open_browser(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        opened = await asyncio.to_thread(webbrowser.open, "about:blank", 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć przeglądarki.")
        return "Otwarto przeglądarkę.", {}

    async def _open_chat(self, _: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        opened = await asyncio.to_thread(webbrowser.open, "https://chatgpt.com", 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć strony.")
        return "Otwarto ChatGPT.", {"url": "https://chatgpt.com"}

    async def _open_url(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        url = str(args.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or len(url) > 2048:
            raise ValueError("Dozwolone są wyłącznie poprawne adresy HTTP/HTTPS.")
        opened = await asyncio.to_thread(webbrowser.open, url, 2)
        if not opened:
            raise RuntimeError("Nie udało się otworzyć adresu.")
        return "Otwarto adres.", {"url": url}

    async def _create_note(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = str(args.get("text") or "").strip()
        if not text:
            raise ValueError("Treść notatki jest pusta.")
        return await self._invoke_uivision(
            macro="voiceloop_notatka.json",
            var1=text[:8000],
            var2="",
            var3="",
        )

    async def _run_uivision_macro(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        macro = str(args.get("macro") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-/]{1,160}\.json", macro):
            raise ValueError("Nieprawidłowa nazwa makra.")
        project_macro = self.settings.project_root / "uivision" / "macros" / macro
        runtime_macro = self.settings.ui_vision_home_path / "macros" / macro
        if not project_macro.exists() and not runtime_macro.exists():
            raise FileNotFoundError(f"Makro nie istnieje: {macro}")
        return await self._invoke_uivision(
            macro=macro,
            var1=str(args.get("var1") or "")[:8000],
            var2=str(args.get("var2") or "")[:8000],
            var3=str(args.get("var3") or "")[:8000],
        )

    async def _invoke_uivision(
        self, *, macro: str, var1: str, var2: str, var3: str
    ) -> tuple[str, dict[str, Any]]:
        runner = self.settings.project_root / "scripts" / "run-uivision.ps1"
        if not runner.exists():
            raise FileNotFoundError(f"Brak runnera UI.Vision: {runner}")
        self._current_process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            "-Macro",
            macro,
            "-Var1",
            var1,
            "-Var2",
            var2,
            "-Var3",
            var3,
            "-TimeoutSeconds",
            str(self.settings.uivision_timeout_seconds),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                self._current_process.communicate(),
                timeout=self.settings.uivision_timeout_seconds + 10,
            )
        except TimeoutError as exc:
            await self.stop()
            raise TimeoutError("UI.Vision przekroczył limit czasu.") from exc
        finally:
            process = self._current_process
            self._current_process = None
        if process is None:
            raise RuntimeError("Proces UI.Vision został przerwany.")
        output = stdout.decode("utf-8", errors="replace").strip()
        error = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(error or output or f"UI.Vision exit code {process.returncode}")
        return output or f"Makro {macro} zakończone.", {"macro": macro}

    async def _remember(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ValueError("Brak treści do zapamiętania.")
        item = await self.memory.create_memory(
            MemoryCreate(
                kind=str(args.get("kind") or "fact")[:50],
                content=content,
                sensitivity="private",
                source="assistant",
            )
        )
        return "Zapamiętano.", {"memory_id": item.id}

    async def _recall(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        query = str(args.get("query") or "").casefold().strip()
        if not query:
            raise ValueError("Brak zapytania do pamięci.")
        items = await self.memory.list_memories(limit=200)
        matched = [item for item in items if query in item.content.casefold()][:10]
        return (
            f"Znaleziono {len(matched)} wpisów.",
            {"items": [item.model_dump(mode="json") for item in matched]},
        )

    async def _speak_text(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = str(args.get("text") or "").strip()
        if not text:
            raise ValueError("Brak tekstu do wypowiedzenia.")
        await self.tts.speak(text)
        return "Wypowiedziano tekst.", {}
