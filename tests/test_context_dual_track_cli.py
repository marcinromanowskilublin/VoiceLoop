import json
from datetime import UTC, datetime, timedelta

import pytest

from voiceloop.context.dual_track import DualTrackRun, DualTrackTraceV1
from voiceloop.context.schema import ContextItemV1, ContextPackV1
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
                        content="Syntetyczny wynik.",
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
