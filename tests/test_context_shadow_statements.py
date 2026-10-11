from datetime import UTC, datetime

import pytest

from voiceloop.context.shadow_statements import (
    AtomicClaimProposal,
    analyze_temporal_statement_shadow,
)
from voiceloop.memory import MemoryStore
from voiceloop.models import CommandRequest

TEXT = "a jeszcze miesiąc temu mówiliśmy, że będzie wspaniale"
NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)


def test_compound_temporal_statement_is_a_prediction_candidate() -> None:
    analysis = analyze_temporal_statement_shadow(
        TEXT,
        request_id="request-1",
        now=NOW,
    )

    assert analysis is not None
    assert analysis.raw_text == TEXT
    assert analysis.provenance_request_id == "request-1"
    assert analysis.epistemic_role == "prediction"
    assert analysis.relevance_gate == "candidate"
    assert analysis.preferred_axes == ("semantic", "topic")
    assert analysis.utterance_time.start is not None
    assert analysis.utterance_time.end is not None
    assert analysis.utterance_time.precision == "approximate"
    assert analysis.utterance_time.source == "relative_text"
    assert analysis.content_time.start is None
    assert analysis.content_time.end is None
    assert analysis.content_time.precision == "unknown"
    assert analysis.content_time.relation == "future_relative_to_utterance"
    assert analysis.query_plan().used_time_filter is True
    assert "claim_axis_not_available" in analysis.axis_insufficiency
    assert TEXT not in str(analysis.telemetry())


@pytest.mark.parametrize(
    "trigger",
    [
        {"comparative_question": True},
        {"conflict": True},
        {"decision_context": True},
        {"action_impact": True},
    ],
)
def test_compound_statement_is_investigated_only_with_material_trigger(trigger) -> None:
    analysis = analyze_temporal_statement_shadow(
        TEXT,
        request_id="request-2",
        now=NOW,
        **trigger,
    )

    assert analysis is not None
    assert analysis.relevance_gate == "investigate"


def test_controller_approves_only_claim_grounded_in_raw_span() -> None:
    grounded = AtomicClaimProposal(
        claim="Rozmówcy przewidywali dobry wynik.",
        evidence_span="będzie wspaniale",
        role="prediction",
    )
    unsupported = AtomicClaimProposal(
        claim="Projekt był ukończony.",
        evidence_span="projekt był ukończony",
        role="prediction",
    )

    analysis = analyze_temporal_statement_shadow(
        TEXT,
        request_id="request-3",
        now=NOW,
        llm_proposals=(grounded, unsupported),
    )

    assert analysis is not None
    assert analysis.approved_claims == (grounded,)
    assert analysis.gap == ""


@pytest.mark.asyncio
async def test_shadow_sql_stores_time_roles_without_duplicate_raw_text(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    request = CommandRequest(text=TEXT)
    await store.create_command(request)
    analysis = analyze_temporal_statement_shadow(
        TEXT,
        request_id=request.request_id,
        now=NOW,
    )
    assert analysis is not None

    await store.save_context_cascade_shadow_temporal_roles(analysis)

    with store._connect() as connection:
        command = connection.execute(
            "SELECT input_text FROM commands WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()
        shadow = connection.execute(
            """
            SELECT * FROM context_cascade_shadow_telemetry
            WHERE request_id = ?
            """,
            (request.request_id,),
        ).fetchone()
        shadow_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(context_cascade_shadow_telemetry)"
            ).fetchall()
        }

    assert command["input_text"] == TEXT
    assert shadow["utterance_start"] is not None
    assert shadow["utterance_end"] is not None
    assert shadow["content_start"] is None
    assert shadow["content_end"] is None
    assert shadow["content_relation"] == "future_relative_to_utterance"
    assert shadow["epistemic_role"] == "prediction"
    assert shadow["relevance_gate"] == "candidate"
    assert not {"raw_text", "claim_text", "query_text"} & shadow_columns
