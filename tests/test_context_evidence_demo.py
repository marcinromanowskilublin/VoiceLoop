import json

import pytest

from voiceloop.context.evidence_demo import build_context_evidence_demo


@pytest.mark.asyncio
async def test_public_context_evidence_runs_real_sqlite_and_qdrant(tmp_path) -> None:
    output = tmp_path / "artifact"

    report = await build_context_evidence_demo(
        output_dir=output,
        revision="test-revision",
    )

    assert report["source_revision"] == "test-revision"
    assert report["network_used"] is False
    assert report["storage"]["canonical"] == "SQLite"
    assert report["storage"]["derived_index"] == "Qdrant local mode"
    assert report["storage"]["qdrant_points"] == 5
    assert report["ingest"]["events"] == 5
    assert report["ingest"]["episodes"] == 5
    assert report["ingest"]["idempotent"] is True
    assert report["ingest"]["second_index"]["already_indexed"] == 5
    assert report["evaluation"]["promotion_allowed"] is False

    scenarios = {item["name"]: item for item in report["scenarios"]}
    assert scenarios["agreement"]["verdict"] == "agreement"
    assert scenarios["conflict"]["verdict"] == "conflict"
    assert scenarios["abstention"]["baseline_count"] == 0
    assert scenarios["abstention"]["cascade_count"] == 0
    assert scenarios["unbounded"]["reason"] == "unbounded_rejected"

    action = report["action_boundary"]
    assert action["risk_before"] == "low"
    assert action["risk_after"] == "medium"
    assert action["confirmation_required"] is True
    assert action["executed"] is False

    stored = json.loads((output / "report.json").read_text(encoding="utf-8"))
    markdown = (output / "report.md").read_text(encoding="utf-8")
    html = (output / "report.html").read_text(encoding="utf-8")
    assert stored == report
    assert "## Failure analysis" in markdown
    assert "Synthetic data is not evidence" in markdown
    assert "PUBLIC SYNTHETIC EVIDENCE" in html
    assert "Qdrant local" in html
