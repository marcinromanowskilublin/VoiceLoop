import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from voiceloop.context.dual_track import DualTrackRun, DualTrackTraceV1
from voiceloop.context.schema import (
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
)
from voiceloop.context.vectorization import EpisodeIndexingReport
from voiceloop.corpus import cli


@pytest.mark.asyncio
async def test_report_context_retrieval_runs_real_dual_track_contract(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "example_id": "positive",
                        "query": "Co ustaliliśmy wczoraj?",
                        "expected_source_ids": ["source-1"],
                        "observed_at": "2026-10-19T12:00:00Z",
                        "expected_verdict": "agreement",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "example_id": "negative",
                        "query": "Nieistniejący zapis z wczoraj",
                        "expected_source_ids": [],
                        "observed_at": "2026-10-19T12:00:00Z",
                        "expected_verdict": "uncertain",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class EmbeddingsStub:
        def __init__(self, **_kwargs) -> None:
            pass

        async def health(self):
            return True, "synthetic embeddings"

    class QdrantStub:
        def __init__(self, settings) -> None:
            self.collection_name = settings.qdrant_collection

        async def health(self):
            return True, "synthetic qdrant"

        async def close(self):
            return None

    class RunnerStub:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self, query, *, request_id, **_kwargs):
            positive = request_id.endswith("positive")
            items = (
                (
                    ContextItemV1(
                        source="synthetic",
                        source_id="source-1",
                            scope=ContextScope.EPISODIC,
                        content="Syntetyczny wynik.",
                            trust=ContextTrust.DERIVED,
                        selection_reason="semantic_scout:rrf5",
                    ),
                )
                if positive
                else ()
            )
            baseline = ContextPackV1(question=query, items=items)
            cascade = ContextPackV1(question=query, items=items)
            trace = DualTrackTraceV1(
                trace_id=("1" if positive else "2") * 24,
                created_at=datetime(2026, 10, 19, tzinfo=UTC),
                expires_at=datetime(2026, 10, 19, tzinfo=UTC)
                + timedelta(days=30),
                bounded=True,
                verdict="agreement" if positive else "uncertain",
                reason="top_match" if positive else "no_candidates",
                baseline_count=len(items),
                cascade_count=len(items),
                overlap_count=len(items),
                top_match=positive,
            )
            return DualTrackRun(
                baseline=baseline,
                cascade=cascade,
                trace=trace,
            )

    monkeypatch.setattr(cli, "OpenAICompatibleEmbeddingClient", EmbeddingsStub)
    monkeypatch.setattr(cli, "QdrantVectorStore", QdrantStub)
    monkeypatch.setattr(cli, "DualTrackRunner", RunnerStub)
    args = cli.build_parser().parse_args(
        [
            "report-context-retrieval",
            "--gold",
            str(gold),
            "--database",
            str(tmp_path / "voice.db"),
            "--eval-now",
            "2026-10-19T12:00:00+00:00",
        ]
    )

    code = await cli._run_async(args)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["sample_count"] == 2
    assert payload["baseline"]["recall_at_k"] == 1.0
    assert payload["cascade"]["recall_at_k"] == 1.0
    assert payload["verdict_accuracy"] == 1.0
    assert payload["promotion_allowed"] is False
    assert payload["provider_health"]["embeddings"]["ok"] is True


@pytest.mark.asyncio
async def test_report_adaptive_context_retrieval_is_shadow_only(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    gold = tmp_path / "adaptive-gold.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "example_id": "positive",
                        "query": "Co ustaliliśmy wczoraj?",
                        "expected_source_ids": ["source-1"],
                        "observed_at": "2026-10-19T12:00:00Z",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "example_id": "negative",
                        "query": "Nieistniejący zapis z wczoraj",
                        "expected_source_ids": [],
                        "observed_at": "2026-10-19T12:00:00Z",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class EmbeddingsStub:
        def __init__(self, **_kwargs) -> None:
            pass

        async def health(self):
            return True, "synthetic embeddings"

    class QdrantStub:
        def __init__(self, settings) -> None:
            self.collection_name = settings.qdrant_collection

        async def health(self):
            return True, "synthetic qdrant"

        async def close(self):
            return None

    class AdaptiveStub:
        def __init__(self, **_kwargs) -> None:
            pass

        async def retrieve(self, query, *, risk, **_kwargs):
            positive = "Nieistniejący" not in query
            items = (
                (
                    ContextItemV1(
                        source="synthetic",
                        source_id="source-1",
                        scope=ContextScope.EPISODIC,
                        content="Syntetyczny wynik.",
                        trust=ContextTrust.DERIVED,
                        selection_reason="adaptive_hybrid:fts",
                    ),
                )
                if positive
                else ()
            )
            return SimpleNamespace(
                pack=ContextPackV1(question=query, items=items),
                policy=SimpleNamespace(
                    policy_id="a" * 16,
                    query_class="factual",
                ),
                trace=SimpleNamespace(
                    stop_reason="exact_evidence_sufficient",
                    vector_attempted=False,
                    vector_count=0,
                ),
            )

    monkeypatch.setattr(cli, "OpenAICompatibleEmbeddingClient", EmbeddingsStub)
    monkeypatch.setattr(cli, "QdrantVectorStore", QdrantStub)
    monkeypatch.setattr(cli, "AdaptiveHybridRetriever", AdaptiveStub)
    args = cli.build_parser().parse_args(
        [
            "report-adaptive-context-retrieval",
            "--gold",
            str(gold),
            "--database",
            str(tmp_path / "voice.db"),
            "--domain",
            "windows",
            "--eval-now",
            "2026-10-19T12:00:00+00:00",
        ]
    )

    code = await cli._run_async(args)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["domain"] == "windows"
    assert payload["metrics"]["recall_at_k"] == 1.0
    assert payload["metrics"]["abstention_accuracy"] == 1.0
    assert payload["vector_expansion_rate"] == 0.0
    assert payload["promotion_allowed"] is False


@pytest.mark.asyncio
async def test_context_prune_apply_targets_isolated_episode_collection(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    collections: list[str] = []

    class QdrantStub:
        def __init__(self, settings) -> None:
            collections.append(settings.qdrant_collection)

        async def close(self):
            return None

    class LifecycleStub:
        def __init__(self, *, memory, qdrant) -> None:
            assert memory is not None
            assert qdrant is not None

        async def prune_expired(self, *, dry_run):
            assert dry_run is False
            return type(
                "Report",
                (),
                {
                    "events": 0,
                    "episodes": 0,
                    "qdrant_deleted": 0,
                    "dry_run": False,
                },
            )()

    monkeypatch.setattr(cli, "QdrantVectorStore", QdrantStub)
    monkeypatch.setattr(cli, "ContextLifecycleService", LifecycleStub)
    args = cli.build_parser().parse_args(
        [
            "prune-context-timeline",
            "--database",
            str(tmp_path / "voice.db"),
            "--apply",
        ]
    )

    code = await cli._run_async(args)
    capsys.readouterr()

    assert code == 0
    assert collections == ["voiceloop_context_episodes_v1"]


@pytest.mark.asyncio
async def test_context_index_cli_drains_all_ready_batches() -> None:
    class IndexerStub:
        def __init__(self) -> None:
            self.calls = 0

        async def index(self, **_kwargs):
            self.calls += 1
            remaining = {1: 3, 2: 1, 3: 0}[self.calls]
            indexed = {1: 2, 2: 2, 3: 1}[self.calls]
            return EpisodeIndexingReport(
                collection="episodes",
                considered=indexed,
                indexed=indexed,
                already_indexed=0,
                would_index=0,
                dry_run=False,
                outbox_pending=remaining,
            )

    indexer = IndexerStub()

    report = await cli._run_context_indexer(
        indexer,  # type: ignore[arg-type]
        limit=2,
        offset=0,
        dry_run=False,
    )

    assert indexer.calls == 3
    assert report.indexed == 5
    assert report.outbox_pending == 0
