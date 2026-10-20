from __future__ import annotations

import hashlib
import html
import json
import math
import re
import tempfile
import time
import unicodedata
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient

from ..actions import ActionRegistry
from ..events import EventBus
from ..executor import CommandExecutor
from ..memory import MemoryStore
from ..models import (
    CommandPlan,
    CommandRequest,
    CommandStatus,
    PlanStep,
    RiskLevel,
)
from ..qdrant_memory import QdrantVectorStore
from ..settings import Settings
from .dual_track import DualTrackRun, DualTrackRunner
from .evaluation import (
    ContextRetrievalEvalRecordV1,
    evaluate_context_dual_track,
    evaluate_context_retrieval,
)
from .retrieval import TimeFirstRetriever
from .schema import ContextEpisodeV1, ContextEventV1
from .vectorization import ContextEpisodeIndexer

DEMO_NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)
EMBEDDING_MODEL = "deterministic-lexical-hash-v1"
EMBEDDING_DIMENSION = 192


@dataclass(frozen=True)
class DemoEpisode:
    source_id: str
    title: str
    summary: str
    started_at: datetime
    topic: str = ""
    intent: str = ""
    decision: str = ""
    person_context: str = ""


class DeterministicEmbeddingClient:
    """Offline lexical embeddings for reproducible integration evidence only."""

    enabled = True
    configured_model = EMBEDDING_MODEL

    def accepts_private_text(self) -> bool:
        return True

    async def health(self) -> tuple[bool, str]:
        return True, f"model: {EMBEDDING_MODEL}"

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [_lexical_vector(text) for text in texts]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_lexical_vector(text) for text in texts]


class _NoopTTS:
    async def speak(self, _text: str) -> None:
        return None

    async def stop(self) -> None:
        return None


async def build_context_evidence_demo(
    *,
    output_dir: Path,
    revision: str,
) -> dict[str, Any]:
    """Run a no-network SQLite + local-Qdrant evidence scenario."""

    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voiceloop-evidence-") as raw_workdir:
        workdir = Path(raw_workdir)
        settings = Settings(
            voiceloop_data_dir=str(workdir / "data"),
            qdrant_collection="demo_serving_memory",
            qdrant_context_episode_collection="demo_context_episodes",
            qdrant_url="http://127.0.0.1:6333",
            local_embeddings_enabled=True,
        )
        memory = MemoryStore(workdir / "voiceloop.db")
        await memory.initialize()
        episodes = await _seed_demo_timeline(memory)
        embeddings = DeterministicEmbeddingClient()
        client = AsyncQdrantClient(location=":memory:")
        qdrant = QdrantVectorStore(
            settings.model_copy(
                update={
                    "qdrant_collection": settings.qdrant_context_episode_collection,
                }
            ),
            client=client,
        )
        indexer = ContextEpisodeIndexer(
            memory=memory,
            embeddings=embeddings,  # type: ignore[arg-type]
            qdrant=qdrant,
            primary_collection=settings.qdrant_collection,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Payload indexes have no effect in the local Qdrant.*",
            )
            first_index = await indexer.index(limit=100, dry_run=False)
        second_index = await indexer.index(limit=100, dry_run=False)
        point_count = await qdrant.count_points()
        runner = DualTrackRunner(
            memory=memory,
            embeddings=embeddings,  # type: ignore[arg-type]
            qdrant=qdrant,
            max_inflight=1,
            timeout_seconds=5.0,
            trace_retention_days=30,
        )
        records = _evaluation_records()
        runs: dict[str, DualTrackRun] = {}
        dual_wall_ms: dict[str, float] = {}

        async def retrieve_dual(record: ContextRetrievalEvalRecordV1, limit: int):
            started = time.perf_counter()
            run = await runner.run(
                record.query,
                request_id=f"public-demo:{record.example_id}",
                now=record.observed_at or DEMO_NOW,
                limit=limit,
                persist=False,
            )
            dual_wall_ms[record.example_id] = round(
                (time.perf_counter() - started) * 1000,
                3,
            )
            runs[record.example_id] = run
            return run

        dual_report = await evaluate_context_dual_track(
            records=records,
            retrieve=retrieve_dual,
            k=3,
        )
        fts_retriever = TimeFirstRetriever(memory=memory)
        fts_wall_ms: dict[str, float] = {}

        async def retrieve_fts(record: ContextRetrievalEvalRecordV1, limit: int):
            started = time.perf_counter()
            pack = await fts_retriever.retrieve(
                record.query,
                now=record.observed_at or DEMO_NOW,
                limit=limit,
            )
            fts_wall_ms[record.example_id] = round(
                (time.perf_counter() - started) * 1000,
                3,
            )
            return pack

        fts_scores, fts_metrics = await evaluate_context_retrieval(
            records=records,
            retrieve=retrieve_fts,
            k=3,
        )

        async def baseline_from_runs(record: ContextRetrievalEvalRecordV1, _limit: int):
            return runs[record.example_id].baseline

        async def cascade_from_runs(record: ContextRetrievalEvalRecordV1, _limit: int):
            return runs[record.example_id].cascade

        baseline_scores, _ = await evaluate_context_retrieval(
            records=records,
            retrieve=baseline_from_runs,
            k=3,
        )
        cascade_scores, _ = await evaluate_context_retrieval(
            records=records,
            retrieve=cascade_from_runs,
            k=3,
        )
        fts_public_metrics = _public_metrics(records, fts_scores, fts_metrics)
        rrf_public_metrics = _public_metrics(
            records,
            baseline_scores,
            dual_report.baseline,
        )
        cascade_public_metrics = _public_metrics(
            records,
            cascade_scores,
            dual_report.cascade,
        )
        scenarios = await _demonstration_scenarios(runner)
        action_boundary = await _demonstrate_action_boundary(settings, memory)
        report = {
            "schema_version": 1,
            "evidence_pack": "context-evidence-v1",
            "source_revision": revision,
            "generated_at": datetime.now(UTC).isoformat(),
            "observation_time": DEMO_NOW.isoformat(),
            "data_class": "public_synthetic_only",
            "network_used": False,
            "storage": {
                "canonical": "SQLite",
                "derived_index": "Qdrant local mode",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "sqlite_bytes": (workdir / "voiceloop.db").stat().st_size,
                "qdrant_points": point_count,
            },
            "ingest": {
                "events": len(episodes),
                "episodes": len(episodes),
                "first_index": asdict(first_index),
                "second_index": asdict(second_index),
                "idempotent": (
                    first_index.indexed == len(episodes)
                    and second_index.indexed == 0
                    and second_index.already_indexed == len(episodes)
                ),
            },
            "evaluation": {
                "records": len(records),
                "fts_only": fts_public_metrics,
                "rrf5": rrf_public_metrics,
                "cascade_2_2_1": cascade_public_metrics,
                "arbiter": {
                    "verdict_counts": dual_report.verdict_counts,
                    "labeled_verdict_count": dual_report.labeled_verdict_count,
                    "verdict_accuracy": (
                        dual_report.verdict_accuracy
                        if dual_report.labeled_verdict_count
                        else None
                    ),
                    "time_boundary_accuracy": dual_report.time_boundary_accuracy,
                },
                "latency_ms": {
                    "fts_only": _latency_summary(fts_wall_ms.values()),
                    "dual_track_wall": _latency_summary(dual_wall_ms.values()),
                    "rrf5_track": _latency_summary(
                        run.trace.baseline_duration_ms for run in runs.values()
                    ),
                    "cascade_track": _latency_summary(
                        run.trace.cascade_duration_ms for run in runs.values()
                    ),
                },
                "failures": {
                    "fts_only": _failure_rows(records, fts_scores),
                    "rrf5": _failure_rows(records, baseline_scores),
                    "cascade_2_2_1": _failure_rows(records, cascade_scores),
                },
                "promotion_allowed": False,
            },
            "scenarios": scenarios,
            "action_boundary": action_boundary,
            "limitations": [
                "Synthetic data is not evidence of private-data quality.",
                (
                    "Lexical hash embeddings make the run deterministic; "
                    "they are not a model benchmark."
                ),
                "Qdrant runs in the real qdrant-client local engine, not a network server.",
                (
                    "Latency is runner-specific and must not be generalized "
                    "to target Windows hardware."
                ),
                (
                    "Agreement means the same top canonical episode version; "
                    "the remaining ranked lists may differ."
                ),
                "The arbiter remains shadow-only and cannot authorize an action.",
            ],
        }
        _assert_demo_contract(report)
        (resolved_output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (resolved_output / "report.md").write_text(
            render_evidence_markdown(report),
            encoding="utf-8",
        )
        (resolved_output / "report.html").write_text(
            render_evidence_html(report),
            encoding="utf-8",
        )
        await qdrant.close()
        return report


def render_evidence_markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    rows = (
        ("FTS only", evaluation["fts_only"], evaluation["latency_ms"]["fts_only"]),
        ("RRF5", evaluation["rrf5"], evaluation["latency_ms"]["rrf5_track"]),
        (
            "Cascade 2+2+1",
            evaluation["cascade_2_2_1"],
            evaluation["latency_ms"]["cascade_track"],
        ),
    )
    lines = [
        "# VoiceLoop public context evidence v1",
        "",
        f"- Source revision: `{report['source_revision']}`",
        "- Data: public synthetic only",
        "- Network: not used",
        f"- SQLite events/episodes: {report['ingest']['events']}",
        f"- Qdrant points: {report['storage']['qdrant_points']}",
        f"- Idempotent second index: {str(report['ingest']['idempotent']).lower()}",
        "- Promotion allowed: `false`",
        "",
        "## Retrieval comparison",
        "",
        (
            "| Track | Recall@3 | MRR | Negative abstention | "
            "Result provenance | Smoke P50 ms | Smoke P95 ms |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics, latency in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    _number(metrics["recall_at_k"]),
                    _number(metrics["mean_reciprocal_rank"]),
                    _number(metrics["negative_abstention_accuracy"]),
                    _number(metrics["provenance_coverage_on_results"]),
                    _number(latency["p50"]),
                    _number(latency["p95"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Observable outcomes",
            "",
            "| Scenario | Verdict | Baseline | Cascade | Safety behavior |",
            "|---|---|---:|---:|---|",
        ]
    )
    for scenario in report["scenarios"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    scenario["name"],
                    scenario["verdict"],
                    str(scenario["baseline_count"]),
                    str(scenario["cascade_count"]),
                    scenario["safety_behavior"],
                )
            )
            + " |"
        )
    action = report["action_boundary"]
    lines.extend(
        [
            "",
            "## Action boundary",
            "",
            f"- Proposed action: `{action['action_id']}`",
            f"- Risk raised locally: `{action['risk_before']}` -> `{action['risk_after']}`",
            f"- Confirmation required: `{str(action['confirmation_required']).lower()}`",
            f"- Executor status: `{action['executor_status']}`",
            f"- Handler calls: `{action['handler_calls']}`",
            f"- Handler executed: `{str(action['executed']).lower()}`",
            "",
            "## Failure analysis",
            "",
        ]
    )
    for track, failures in evaluation["failures"].items():
        if failures:
            lines.append(f"- **{track}:** " + "; ".join(failures))
        else:
            lines.append(f"- **{track}:** no misses in this synthetic fixture")
    lines.extend(
        [
            "",
            "## What this does not prove",
            "",
            *(f"- {item}" for item in report["limitations"]),
            "",
        ]
    )
    return "\n".join(lines)


def render_evidence_html(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    metric_rows = []
    for label, metrics, latency in (
        ("FTS only", evaluation["fts_only"], evaluation["latency_ms"]["fts_only"]),
        ("RRF5", evaluation["rrf5"], evaluation["latency_ms"]["rrf5_track"]),
        (
            "Cascade 2+2+1",
            evaluation["cascade_2_2_1"],
            evaluation["latency_ms"]["cascade_track"],
        ),
    ):
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{_number(metrics['recall_at_k'])}</td>"
            f"<td>{_number(metrics['mean_reciprocal_rank'])}</td>"
            f"<td>{_number(metrics['negative_abstention_accuracy'])}</td>"
            f"<td>{_number(latency['p50'])}</td>"
            f"<td>{_number(latency['p95'])}</td>"
            "</tr>"
        )
    scenario_cards = []
    for scenario in report["scenarios"]:
        scenario_cards.append(
            '<article class="scenario">'
            f"<span>{html.escape(scenario['name'])}</span>"
            f"<strong>{html.escape(scenario['verdict'])}</strong>"
            f"<p>{html.escape(scenario['safety_behavior'])}</p>"
            "</article>"
        )
    action = report["action_boundary"]
    qdrant_points = report["storage"]["qdrant_points"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VoiceLoop context evidence v1</title>
  <style>
    :root {{ color-scheme: dark; --bg: #08111f; --panel: #111c2f;
      --line: #334155; --text: #f8fafc; --muted: #94a3b8;
      --green: #6ee7b7; --blue: #7dd3fc; --amber: #fcd34d; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at 10% 0, #12304a, var(--bg) 38%);
      color: var(--text); font: 16px/1.5 Inter, Segoe UI, sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 48px 28px 80px; }}
    h1 {{ max-width: 720px; margin: 10px 0; font-size: 46px; line-height: 1.05; }}
    h2 {{ margin-top: 38px; }} p {{ color: #cbd5e1; }}
    .eyebrow {{ color: var(--green); font-weight: 800; letter-spacing: .09em; }}
    .lede {{ max-width: 820px; color: var(--muted); }}
    .cards, .scenarios {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .card, .scenario {{ padding: 18px; border: 1px solid var(--line);
      border-radius: 18px; background: rgba(17, 28, 47, .9); }}
    .card span, .scenario span {{ display: block; color: var(--muted);
      font-size: 12px; font-weight: 800; text-transform: uppercase; }}
    .card strong, .scenario strong {{ display: block; margin-top: 7px;
      color: var(--blue); font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden;
      border: 1px solid var(--line); border-radius: 16px; background: var(--panel); }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }} th {{ color: var(--muted); }}
    .boundary {{ border-left: 4px solid var(--amber); }}
    code {{ color: var(--green); }} .truth {{ color: var(--green); }}
    @media (max-width: 760px) {{
      h1 {{ font-size: 34px; }} .cards, .scenarios {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body><main>
  <div class="eyebrow">PUBLIC SYNTHETIC EVIDENCE · NO NETWORK</div>
  <h1>Context retrieval you can reproduce locally.</h1>
  <p class="lede">A deterministic run over real SQLite and the Qdrant local
    engine. It exercises ingest, provenance checks, RRF5, cascade 2+2+1,
    abstention, conflict handling and the local action policy.</p>
  <section class="cards">
    <div class="card"><span>Canonical store</span><strong>SQLite</strong></div>
    <div class="card"><span>Derived index</span>
      <strong>{qdrant_points} points</strong></div>
    <div class="card"><span>Second index</span><strong>idempotent</strong></div>
    <div class="card"><span>Serving promotion</span><strong>false</strong></div>
  </section>
  <h2>Retrieval comparison</h2>
  <table>
    <thead><tr><th>Track</th><th>Recall@3</th><th>MRR</th>
      <th>Negative abstention</th><th>Smoke P50 ms</th>
      <th>Smoke P95 ms</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
  <h2>Observable outcomes</h2>
  <section class="scenarios">{''.join(scenario_cards)}</section>
  <h2>Action boundary</h2>
  <article class="card boundary">
    <span>Policy result</span>
    <strong><code>{html.escape(action['action_id'])}</code> requires confirmation</strong>
    <p>Risk: <code>{action['risk_before']}</code> to <code>{action['risk_after']}</code>.
      Executor: <code>{action['executor_status']}</code>.
      Handler calls: <b class="truth">{action['handler_calls']}</b>.</p>
  </article>
  <p class="lede">Synthetic lexical embeddings make this run repeatable; they
    are not evidence of private-data quality or target-hardware latency.
    Source revision: <code>{html.escape(report['source_revision'])}</code>.</p>
</main></body>
</html>
"""


async def _seed_demo_timeline(memory: MemoryStore) -> list[ContextEpisodeV1]:
    seeds = (
        DemoEpisode(
            source_id="atlas-brief",
            title="Atlas daily brief",
            summary=(
                "Projekt Atlas lokalne wdrożenie pilota omówiono w zespole. "
                "To syntetyczne streszczenie kontekstu."
            ),
            started_at=DEMO_NOW - timedelta(days=1, hours=3),
        ),
        DemoEpisode(
            source_id="atlas-decision",
            title="Atlas decision",
            summary="Spotkanie Atlas dotyczyło harmonogramu i granic bezpieczeństwa.",
            started_at=DEMO_NOW - timedelta(days=1, hours=2),
            topic="projekt Atlas",
            decision="zatwierdziliśmy lokalne wdrożenie pilota Atlas w piątek",
        ),
        DemoEpisode(
            source_id="atlas-owner",
            title="Atlas risk owner",
            summary="Marta przygotowała analizę ryzyka dla projektu Atlas.",
            started_at=DEMO_NOW - timedelta(days=1, hours=1),
            topic="projekt Atlas",
            intent="analiza ryzyka przed pilotażem",
            person_context="Marta odpowiada za analizę ryzyka",
        ),
        DemoEpisode(
            source_id="borealis-intent",
            title="Borealis workshop",
            summary="Warsztat Borealis ustalił zakres kolejnego eksperymentu.",
            started_at=DEMO_NOW - timedelta(days=1, minutes=30),
            topic="projekt Borealis",
            intent="ograniczyć koszt ręcznego przeglądu",
        ),
        DemoEpisode(
            source_id="orion-topic",
            title="Orion research note",
            summary="Notatka Orion opisuje provenance i abstencję wyszukiwania.",
            started_at=DEMO_NOW - timedelta(days=1, minutes=15),
            topic="projekt Orion provenance abstencja",
        ),
    )
    episodes: list[ContextEpisodeV1] = []
    for seed in seeds:
        event = ContextEventV1.from_observation(
            source="public_demo",
            source_id=f"event:{seed.source_id}",
            started_at=seed.started_at,
            ended_at=seed.started_at + timedelta(minutes=10),
            event_type="synthetic_demo",
            text=seed.summary,
            metadata={"fixture": "context-evidence-v1"},
        )
        stored_event = await memory.upsert_context_event(event)
        episode = ContextEpisodeV1(
            episode_id=f"demo:{seed.source_id}",
            source="public_demo",
            source_id=seed.source_id,
            started_at=seed.started_at,
            ended_at=seed.started_at + timedelta(minutes=10),
            title=seed.title,
            summary=seed.summary,
            source_event_ids=(stored_event.event_id,),
            metadata={
                "fixture": "context-evidence-v1",
                "topic": seed.topic,
                "intent": seed.intent,
                "decision": seed.decision,
                "person_context": seed.person_context,
            },
        )
        episodes.append(await memory.upsert_context_episode(episode))
    return episodes


def _evaluation_records() -> list[ContextRetrievalEvalRecordV1]:
    return [
        ContextRetrievalEvalRecordV1(
            example_id="decision",
            query="Co zatwierdziliśmy wczoraj w projekcie Atlas?",
            expected_source_ids=("atlas-decision",),
            expected_axes=("decision",),
            observed_at=DEMO_NOW,
            category="decision",
            risk="high",
        ),
        ContextRetrievalEvalRecordV1(
            example_id="person",
            query="Kto wczoraj odpowiadał za analizę ryzyka Atlas?",
            expected_source_ids=("atlas-owner",),
            expected_axes=("person_context",),
            observed_at=DEMO_NOW,
            category="person_context",
            risk="medium",
        ),
        ContextRetrievalEvalRecordV1(
            example_id="intent",
            query="Jaki był wczoraj cel warsztatu Borealis?",
            expected_source_ids=("borealis-intent",),
            expected_axes=("intent",),
            observed_at=DEMO_NOW,
            category="intent",
        ),
        ContextRetrievalEvalRecordV1(
            example_id="topic",
            query="O czym była wczoraj notatka Orion?",
            expected_source_ids=("orion-topic",),
            expected_axes=("topic",),
            observed_at=DEMO_NOW,
            category="topic",
        ),
        ContextRetrievalEvalRecordV1(
            example_id="negative-time",
            query="Co ustaliliśmy przedwczoraj o projekcie Atlas?",
            observed_at=DEMO_NOW,
            category="negative",
            risk="high",
        ),
    ]


async def _demonstration_scenarios(
    runner: DualTrackRunner,
) -> list[dict[str, Any]]:
    raw = (
        (
            "agreement",
            "Kto wczoraj odpowiadał za analizę ryzyka Atlas?",
            3,
            "record agreement; no serving switch",
        ),
        (
            "conflict",
            "Co ustaliliśmy wczoraj o lokalnym wdrożeniu projektu Atlas?",
            1,
            "record conflict for human review",
        ),
        (
            "abstention",
            "Co ustaliliśmy przedwczoraj o projekcie Atlas?",
            3,
            "return no bounded evidence",
        ),
        (
            "unbounded",
            "Co ustaliliśmy o projekcie Atlas?",
            3,
            "reject unbounded dual-track measurement",
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, query, limit, safety_behavior in raw:
        result = await runner.run(
            query,
            request_id=f"public-scenario:{name}",
            now=DEMO_NOW,
            limit=limit,
            persist=False,
        )
        rows.append(
            {
                "name": name,
                "query": query,
                "verdict": result.trace.verdict,
                "reason": result.trace.reason,
                "baseline_count": result.trace.baseline_count,
                "cascade_count": result.trace.cascade_count,
                "overlap_count": result.trace.overlap_count,
                "baseline_sources": [
                    item.source_id for item in result.baseline.items
                ],
                "cascade_sources": [
                    item.source_id for item in result.cascade.items
                ],
                "safety_behavior": safety_behavior,
            }
        )
    return rows


def _public_metrics(
    records: list[ContextRetrievalEvalRecordV1],
    scores: list[Any],
    contract_metrics: Any,
) -> dict[str, Any]:
    positive = [
        score
        for record, score in zip(records, scores, strict=True)
        if record.expected_source_ids
    ]
    negative = [
        score
        for record, score in zip(records, scores, strict=True)
        if not record.expected_source_ids
    ]
    returned = [score for score in scores if score.predicted_source_ids]
    return {
        "schema_version": 1,
        "sample_count": len(records),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "k": contract_metrics.k,
        "recall_at_k": _ratio(score.hit_at_k for score in positive),
        "mean_reciprocal_rank": _average(
            score.reciprocal_rank for score in positive
        ),
        "negative_abstention_accuracy": _ratio(
            score.abstention_correct for score in negative
        ),
        "provenance_coverage_on_results": _ratio(
            score.provenance_complete for score in returned
        ),
        "reserve_axis_coverage": contract_metrics.reserve_axis_coverage,
    }


async def _demonstrate_action_boundary(
    settings: Settings,
    memory: MemoryStore,
) -> dict[str, Any]:
    registry = ActionRegistry(
        settings,
        memory,
        _NoopTTS(),  # type: ignore[arg-type]
    )
    handler_calls = 0

    async def guarded_handler(_args: dict[str, Any]):
        nonlocal handler_calls
        handler_calls += 1
        return "unexpected", {}

    spec = registry._specs["remember"]
    registry._specs["remember"] = replace(spec, handler=guarded_handler)
    proposed = PlanStep(
        action_id="remember",
        args={"content": "synthetic public demo"},
        risk=RiskLevel.LOW,
        confirmation_required=False,
    )
    request = CommandRequest(
        request_id="public-demo-confirmation",
        text="Zapamiętaj syntetyczny publiczny przykład.",
    )
    await memory.create_command(request)
    executor = CommandExecutor(
        memory=memory,
        actions=registry,
        events=EventBus(),
        queue_limit=1,
    )
    command = await executor.submit(
        CommandPlan(
            request_id=request.request_id,
            intent="remember",
            response_text="Propozycja wymaga lokalnego potwierdzenia.",
            confidence=1.0,
            steps=[proposed.model_copy(deep=True)],
            provider="public_demo",
        )
    )
    if command is None or command.plan is None:
        raise RuntimeError("Executor did not persist the confirmation plan.")
    gated = command.plan.steps[0]
    return {
        "action_id": gated.action_id,
        "risk_before": proposed.risk.value,
        "risk_after": gated.risk.value,
        "confirmation_required": gated.confirmation_required,
        "executor_status": command.status.value,
        "pending_confirmation": request.request_id in executor.pending_confirmation,
        "handler_calls": handler_calls,
        "executed": handler_calls > 0,
        "note": "CommandExecutor persisted awaiting_confirmation and did not call the handler.",
    }


def _failure_rows(
    records: list[ContextRetrievalEvalRecordV1],
    scores: list[Any],
) -> list[str]:
    failures: list[str] = []
    for record, score in zip(records, scores, strict=True):
        if record.expected_source_ids and not score.hit_at_k:
            failures.append(f"{record.example_id}:miss")
        elif not record.expected_source_ids and not score.abstention_correct:
            failures.append(f"{record.example_id}:false-positive")
    return failures


def _latency_summary(values: Any) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50": 0.0, "p95": 0.0}
    return {
        "count": len(ordered),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    rank = max(0, math.ceil(percentile * len(values)) - 1)
    return values[min(rank, len(values) - 1)]


def _ratio(values: Any) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _assert_demo_contract(report: dict[str, Any]) -> None:
    scenarios = {item["name"]: item for item in report["scenarios"]}
    required = {"agreement", "conflict", "abstention", "unbounded"}
    if set(scenarios) != required:
        raise RuntimeError("Public demo scenarios are incomplete.")
    if scenarios["agreement"]["verdict"] != "agreement":
        raise RuntimeError("Agreement scenario did not agree.")
    if scenarios["conflict"]["verdict"] != "conflict":
        raise RuntimeError("Conflict scenario did not conflict.")
    if scenarios["abstention"]["baseline_count"] != 0 or scenarios[
        "abstention"
    ]["cascade_count"] != 0:
        raise RuntimeError("Bounded abstention returned evidence.")
    if scenarios["unbounded"]["reason"] != "unbounded_rejected":
        raise RuntimeError("Unbounded scenario was not rejected.")
    if not report["ingest"]["idempotent"]:
        raise RuntimeError("Episode indexer was not idempotent.")
    action = report["action_boundary"]
    if (
        not action["confirmation_required"]
        or action["executor_status"] != CommandStatus.AWAITING_CONFIRMATION.value
        or not action["pending_confirmation"]
        or action["handler_calls"] != 0
        or action["executed"]
    ):
        raise RuntimeError("Action boundary did not remain confirmation-only.")
    if report["evaluation"]["promotion_allowed"]:
        raise RuntimeError("Public evidence must never promote retrieval.")


def _lexical_vector(text: str) -> list[float]:
    payload = text.split("\n", 1)[-1]
    normalized = unicodedata.normalize("NFKD", payload.casefold())
    plain = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    tokens = re.findall(r"[a-z0-9-]+", plain)
    features = [
        *tokens,
        *(
            f"{left}_{right}"
            for left, right in zip(tokens, tokens[1:], strict=False)
        ),
    ]
    vector = [0.0] * EMBEDDING_DIMENSION
    for feature in features:
        digest = hashlib.sha256(feature.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _number(value: Any) -> str:
    return f"{float(value):.3f}"
