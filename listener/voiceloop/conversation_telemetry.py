from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .events import EventBus
from .models import ConversationTraceSnapshot, ConversationTurnTrace

LOGGER = logging.getLogger("voiceloop.conversation_telemetry")

_METRIC_PHASES: dict[str, tuple[str, str]] = {
    "stt_ms": ("speech_started", "speech_end"),
    "context_ms": ("context_started", "context_ready"),
    "model_ms": ("model_started", "model_completed"),
    "tool_ms": ("tool_started", "tool_completed"),
    "speech_to_first_audio_ms": ("speech_end", "tts_first_audio"),
    "listening_resume_ms": ("tts_completed", "listening_ready"),
    "barge_stop_ms": ("barge_detected", "tts_stopped"),
}


class ConversationTelemetry:
    """Keeps a bounded, correlated view of recent voice turns."""

    def __init__(
        self,
        events: EventBus,
        *,
        max_traces: int = 200,
        storage_path: Path | None = None,
    ) -> None:
        self.events = events
        self.max_traces = max(20, max_traces)
        self.storage_path = storage_path
        self._traces: dict[tuple[str, int], ConversationTurnTrace] = {}
        self._started: dict[tuple[str, int], float] = {}
        self._request_keys: dict[str, tuple[str, int]] = {}
        self._order: deque[tuple[str, int]] = deque()
        self._lock = asyncio.Lock()
        self._persisted: set[tuple[str, int]] = set()
        self._load_persisted()

    async def begin(
        self,
        *,
        session_id: str,
        turn_id: int,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_perf: float | None = None,
        initial_phases_ms: dict[str, int] | None = None,
    ) -> ConversationTurnTrace:
        key = (session_id, turn_id)
        async with self._lock:
            trace = ConversationTurnTrace(
                session_id=session_id,
                turn_id=turn_id,
                request_id=request_id,
                metadata=dict(metadata or {}),
                phases_ms={"turn_started": 0, **dict(initial_phases_ms or {})},
            )
            self._traces[key] = trace
            self._started[key] = started_perf if started_perf is not None else time.perf_counter()
            if request_id:
                self._request_keys[request_id] = key
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            self._trim_locked()
            payload = trace.model_dump(mode="json")
        await self.events.publish("conversation.trace", payload)
        return trace.model_copy(deep=True)

    async def mark(
        self,
        *,
        session_id: str,
        turn_id: int,
        phase: str,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurnTrace | None:
        key = (session_id, turn_id)
        async with self._lock:
            trace = self._traces.get(key)
            started = self._started.get(key)
            if trace is None or started is None:
                return None
            trace.phases_ms[phase] = max(0, round((time.perf_counter() - started) * 1000))
            if request_id:
                trace.request_id = request_id
                self._request_keys[request_id] = key
            if metadata:
                trace.metadata.update(metadata)
            trace.metrics_ms = self._metrics_for(trace)
            payload = trace.model_dump(mode="json")
        await self.events.publish("conversation.trace", payload)
        return trace.model_copy(deep=True)

    async def mark_request(
        self,
        request_id: str,
        phase: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurnTrace | None:
        async with self._lock:
            key = self._request_keys.get(request_id)
        if key is None:
            return None
        return await self.mark(
            session_id=key[0],
            turn_id=key[1],
            phase=phase,
            request_id=request_id,
            metadata=metadata,
        )

    async def finish(
        self,
        *,
        session_id: str,
        turn_id: int,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurnTrace | None:
        key = (session_id, turn_id)
        should_persist = False
        async with self._lock:
            trace = self._traces.get(key)
            started = self._started.get(key)
            if trace is None or started is None:
                return None
            trace.phases_ms["turn_completed"] = max(
                0,
                round((time.perf_counter() - started) * 1000),
            )
            trace.completed_at = datetime.now(UTC)
            trace.status = status[:40] or "completed"
            if metadata:
                trace.metadata.update(metadata)
            trace.metrics_ms = self._metrics_for(trace)
            payload = trace.model_dump(mode="json")
            if self.storage_path is not None and key not in self._persisted:
                self._persisted.add(key)
                should_persist = True
        await self.events.publish("conversation.trace", payload)
        if should_persist:
            try:
                await asyncio.to_thread(self._append_persisted, payload)
            except OSError as exc:
                LOGGER.warning("Could not persist conversation trace: %s", exc)
        return trace.model_copy(deep=True)

    async def snapshot(self, *, limit: int = 20) -> ConversationTraceSnapshot:
        safe_limit = max(1, min(limit, self.max_traces))
        async with self._lock:
            keys = list(self._order)[-safe_limit:]
            traces = [
                self._traces[key].model_copy(deep=True)
                for key in reversed(keys)
                if key in self._traces
            ]
            aggregate_source = [
                trace for trace in self._traces.values() if trace.metrics_ms
            ]
            aggregates = self._aggregates(aggregate_source)
        return ConversationTraceSnapshot(traces=traces, aggregates=aggregates)

    async def quality_report(self) -> dict[str, Any]:
        snapshot = await self.snapshot(limit=self.max_traces)
        thresholds = {
            "speech_to_first_audio_ms": {"p50": 2500, "p95": 4000},
            "listening_resume_ms": {"p50": 150, "p95": 300},
            "barge_stop_ms": {"p50": 350, "p95": 700},
        }
        checks: dict[str, dict[str, Any]] = {}
        for name, limits in thresholds.items():
            aggregate = snapshot.aggregates.get(name, {})
            count = int(aggregate.get("count", 0))
            checks[name] = {
                "count": count,
                "p50": aggregate.get("p50"),
                "p95": aggregate.get("p95"),
                "ready": count >= 20,
                "passed": bool(
                    count >= 20
                    and int(aggregate.get("p50", limits["p50"] + 1)) <= limits["p50"]
                    and int(aggregate.get("p95", limits["p95"] + 1)) <= limits["p95"]
                ),
                "limits": limits,
            }
        modes = {
            str(trace.metadata.get("conversation_mode", "unknown"))
            for trace in snapshot.traces
        }
        mode_aggregates = {
            mode: self._aggregates(
                [
                    trace
                    for trace in snapshot.traces
                    if str(trace.metadata.get("conversation_mode", "unknown")) == mode
                ]
            )
            for mode in sorted(modes)
        }
        return {
            "sample_count": len(snapshot.traces),
            "sample_count_by_mode": {
                mode: sum(
                    str(trace.metadata.get("conversation_mode", "unknown")) == mode
                    for trace in snapshot.traces
                )
                for mode in sorted(modes)
            },
            "baseline_ready": len(snapshot.traces) >= 20,
            "passed": bool(checks and all(item["passed"] for item in checks.values())),
            "checks": checks,
            "mode_aggregates": mode_aggregates,
            "note": (
                "Bramki audio wymagają co najmniej 20 realnych próbek każdej metryki; "
                "false-interrupt i utratę słów ocenia replay/odsłuch."
            ),
        }

    def health(self) -> tuple[bool, str]:
        return True, f"recent_traces={len(self._traces)}"

    def _load_persisted(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            lines = self.storage_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            LOGGER.warning("Could not load conversation traces: %s", exc)
            return
        for raw in lines[-self.max_traces :]:
            try:
                trace = ConversationTurnTrace.model_validate_json(raw)
            except (ValueError, TypeError):
                continue
            key = (trace.session_id, trace.turn_id)
            self._traces[key] = trace
            self._order.append(key)
            self._persisted.add(key)
            if trace.request_id:
                self._request_keys[trace.request_id] = key
        self._trim_locked()

    def _append_persisted(self, payload: dict[str, Any]) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    def _trim_locked(self) -> None:
        while len(self._order) > self.max_traces:
            old_key = self._order.popleft()
            old_trace = self._traces.pop(old_key, None)
            if old_trace is not None and old_trace.request_id:
                self._request_keys.pop(old_trace.request_id, None)
            self._started.pop(old_key, None)

    @staticmethod
    def _metrics_for(trace: ConversationTurnTrace) -> dict[str, int]:
        metrics: dict[str, int] = {}
        for metric, (start_phase, end_phase) in _METRIC_PHASES.items():
            start = trace.phases_ms.get(start_phase)
            end = trace.phases_ms.get(end_phase)
            if start is not None and end is not None and end >= start:
                metrics[metric] = end - start
        return metrics

    @classmethod
    def _aggregates(
        cls,
        traces: list[ConversationTurnTrace],
    ) -> dict[str, dict[str, int]]:
        names = sorted({name for trace in traces for name in trace.metrics_ms})
        aggregates: dict[str, dict[str, int]] = {}
        for name in names:
            values = sorted(
                trace.metrics_ms[name]
                for trace in traces
                if name in trace.metrics_ms
            )
            if not values:
                continue
            aggregates[name] = {
                "count": len(values),
                "p50": cls._percentile(values, 0.50),
                "p95": cls._percentile(values, 0.95),
            }
        return aggregates

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> int:
        if len(values) == 1:
            return values[0]
        rank = max(0, math.ceil(percentile * len(values)) - 1)
        return values[min(rank, len(values) - 1)]
