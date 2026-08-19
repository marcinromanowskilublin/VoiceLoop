from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..router import normalize_text


class ReliabilityReportError(RuntimeError):
    pass


def build_action_reliability_report(
    database_path: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    rows = _read_command_rows(database_path, start=start, end=end)
    event_rows = _read_command_events(database_path, start=start, end=end)
    statuses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    models: Counter[str] = Counter()
    status_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_provider: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_program: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    action_totals: Counter[str] = Counter()
    action_successes: Counter[str] = Counter()
    action_failures: Counter[str] = Counter()
    durations: list[float] = []
    confirmation_count = 0
    malformed_count = 0
    duplicate_candidates = 0
    previous_by_key: dict[tuple[str, str], datetime] = {}

    for row in rows:
        status = str(row["status"] or "unknown")
        source = str(row["source"] or "unknown")
        provider = str(row["provider"] or "unknown")
        model = str(row["model"] or "unknown")
        statuses[status] += 1
        sources[source] += 1
        providers[provider] += 1
        models[model] += 1
        intent = str(row["intent"] or "unknown")
        status_by_intent[intent][status] += 1
        status_by_source[source][status] += 1
        status_by_provider[provider][status] += 1
        created = _parse_datetime(row["created_at"])
        updated = _parse_datetime(row["updated_at"])
        if created is not None and updated is not None and updated >= created:
            durations.append((updated - created).total_seconds())
        normalized = normalize_text(str(row["input_text"] or ""))
        key = (source, normalized)
        previous = previous_by_key.get(key)
        if previous is not None and created is not None:
            if 0.0 <= (created - previous).total_seconds() <= 10.0:
                duplicate_candidates += 1
        if created is not None:
            previous_by_key[key] = created
        try:
            plan = json.loads(row["plan_json"]) if row["plan_json"] else {}
            results = json.loads(row["results_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            malformed_count += 1
            continue
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if isinstance(steps, list):
            confirmation_count += int(
                any(
                    isinstance(step, dict) and step.get("confirmation_required") is True
                    for step in steps
                )
            )
            for step in steps:
                if not isinstance(step, dict):
                    continue
                action_id = str(step.get("action_id") or "unknown")
                action_totals[action_id] += 1
                status_by_action[action_id][status] += 1
                status_by_program[_program_bucket(step)][status] += 1
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                action_id = str(result.get("action_id") or "unknown")
                success = result.get("success")
                if success is True:
                    action_successes[action_id] += 1
                elif success is False:
                    action_failures[action_id] += 1

    actions: dict[str, dict[str, int | float]] = {}
    for action_id in sorted(action_totals):
        total = action_totals[action_id]
        succeeded = action_successes[action_id]
        failed = action_failures[action_id]
        actions[action_id] = {
            "planned": total,
            "succeeded": succeeded,
            "failed": failed,
            "observed_result_count": succeeded + failed,
            "result_coverage": (succeeded + failed) / total if total else 0.0,
            "observed_success_rate": (
                succeeded / (succeeded + failed) if succeeded + failed else 0.0
            ),
        }
    planned_step_count = sum(action_totals.values())
    observed_result_count = sum(action_successes.values()) + sum(action_failures.values())
    succeeded_result_count = sum(action_successes.values())
    failed_result_count = sum(action_failures.values())
    phase_durations: dict[str, list[float]] = defaultdict(list)
    events_by_request: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for event in event_rows:
        events_by_request[str(event["request_id"])].append(event)
    for events in events_by_request.values():
        for current, following in zip(events, events[1:], strict=False):
            current_at = _parse_datetime(current["created_at"])
            following_at = _parse_datetime(following["created_at"])
            if current_at is None or following_at is None or following_at < current_at:
                continue
            phase_durations[str(current["status"])].append(
                (following_at - current_at).total_seconds()
            )
    phase_latency = {
        status: {
            "count": len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        }
        for status, values in sorted(phase_durations.items())
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "range_start": start.isoformat() if start else None,
        "range_end": end.isoformat() if end else None,
        "command_count": len(rows),
        "status_count": dict(sorted(statuses.items())),
        "source_count": dict(sorted(sources.items())),
        "provider_count": dict(sorted(providers.items())),
        "model_count": dict(sorted(models.items())),
        "status_by_intent": _nested_counts(status_by_intent),
        "status_by_source": _nested_counts(status_by_source),
        "status_by_provider": _nested_counts(status_by_provider),
        "status_by_program": _nested_counts(status_by_program),
        "status_by_action": _nested_counts(status_by_action),
        "confirmation_count": confirmation_count,
        "duplicate_or_retry_candidates": duplicate_candidates,
        "malformed_record_count": malformed_count,
        "event_count": len(event_rows),
        "phase_latency_seconds": phase_latency,
        "latency_seconds": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": max(durations) if durations else 0.0,
        },
        "actions": actions,
        "action_result_summary": {
            "planned_step_count": planned_step_count,
            "observed_result_count": observed_result_count,
            "succeeded_result_count": succeeded_result_count,
            "failed_result_count": failed_result_count,
            "result_coverage": (
                observed_result_count / planned_step_count if planned_step_count else 0.0
            ),
            "observed_success_rate": (
                succeeded_result_count / observed_result_count
                if observed_result_count
                else 0.0
            ),
        },
        "reconciled": sum(statuses.values()) == len(rows),
        "limitations": [
            "Historyczne dane nie zawierają pełnego TranscriptEnvelopeV1.",
            "Historyczne retry są kandydatami wykrytymi czasowo, nie potwierdzoną relacją.",
            (
                "created_at→updated_at jest czasem całkowitym dla rekordów sprzed "
                "wprowadzenia command_events."
            ),
        ],
    }


def render_action_reliability_report(report: dict[str, Any]) -> str:
    lines = [
        "# VoiceLoop — niezawodność działań",
        "",
        f"- Komendy: {report['command_count']}",
        f"- Potwierdzenia: {report['confirmation_count']}",
        f"- Kandydaci retry/duplikatów: {report['duplicate_or_retry_candidates']}",
        f"- Rekordy uszkodzone: {report['malformed_record_count']}",
        f"- Zdarzenia faz: {report['event_count']}",
        f"- Uzgodnienie sum: {'OK' if report['reconciled'] else 'BŁĄD'}",
        f"- Latencja p50/p95: {report['latency_seconds']['p50']:.3f}s / "
        f"{report['latency_seconds']['p95']:.3f}s",
        f"- Kroki z wynikiem: {report['action_result_summary']['observed_result_count']}/"
        f"{report['action_result_summary']['planned_step_count']} "
        f"({report['action_result_summary']['result_coverage']:.3f})",
        "- Sukces wśród kroków z wynikiem: "
        f"{report['action_result_summary']['observed_success_rate']:.3f}",
        "",
        "## Statusy",
    ]
    lines.extend(
        f"- {status}: {count}"
        for status, count in report["status_count"].items()
    )
    lines.extend(("", "## Akcje"))
    for action_id, metrics in report["actions"].items():
        lines.append(
            f"- {action_id}: plan={metrics['planned']}, "
            f"sukces={metrics['succeeded']}, błąd={metrics['failed']}, "
            f"pokrycie_wynikiem={metrics['result_coverage']:.3f}, "
            f"success_rate_observed={metrics['observed_success_rate']:.3f}"
        )
    return "\n".join(lines) + "\n"


def _read_command_rows(
    database_path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
) -> list[sqlite3.Row]:
    resolved = database_path.resolve(strict=True)
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            clauses: list[str] = []
            values: list[str] = []
            if start is not None:
                clauses.append("created_at >= ?")
                values.append(start.isoformat())
            if end is not None:
                clauses.append("created_at < ?")
                values.append(end.isoformat())
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            return connection.execute(
                f"SELECT * FROM commands {where} ORDER BY created_at ASC",
                values,
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise ReliabilityReportError("Nie można odczytać bazy komend w trybie read-only.") from exc


def _read_command_events(
    database_path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
) -> list[sqlite3.Row]:
    resolved = database_path.resolve(strict=True)
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'command_events'
                """
            ).fetchone()
            if exists is None:
                return []
            clauses: list[str] = []
            values: list[str] = []
            if start is not None:
                clauses.append("created_at >= ?")
                values.append(start.isoformat())
            if end is not None:
                clauses.append("created_at < ?")
                values.append(end.isoformat())
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            return connection.execute(
                f"""
                SELECT request_id, status, created_at
                FROM command_events
                {where}
                ORDER BY request_id ASC, id ASC
                """,
                values,
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise ReliabilityReportError("Nie można odczytać zdarzeń komend.") from exc


def _parse_datetime(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _nested_counts(
    values: dict[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        key: dict(sorted(counts.items()))
        for key, counts in sorted(values.items())
    }


def _program_bucket(step: dict[str, Any]) -> str:
    action_id = str(step.get("action_id") or "unknown")
    args = step.get("args")
    if not isinstance(args, dict):
        args = {}
    for key in ("process_name", "app_name", "application"):
        value = str(args.get(key) or "").strip()
        if value:
            return Path(value).name.casefold()[:80]
    url_value = str(args.get("url") or "").strip()
    if url_value:
        hostname = urlparse(url_value).hostname
        if hostname:
            return hostname.casefold()[:80]
    if action_id.startswith("open_browser"):
        return "browser"
    if action_id.startswith("open_"):
        return action_id.removeprefix("open_")[:80]
    return "unknown"
