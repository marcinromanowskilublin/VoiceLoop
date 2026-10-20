import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from voiceloop.context.retrieval import build_time_first_query
from voiceloop.context.shadow_statements import (
    AtomicClaimProposal,
    analyze_temporal_statement_shadow,
    build_temporal_shadow_context,
    shadow_relevance_gate,
)
from voiceloop.memory import MemoryStore
from voiceloop.models import CommandRequest

TEXT = "a jeszcze miesiąc temu mówiliśmy, że będzie wspaniale"
NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("query", "relation", "min_days", "max_days", "confidence"),
    [
        (
            "Co ustaliliśmy miesiąc temu?",
            "approximately_one_month_before_observation",
            23,
            37,
            0.65,
        ),
        (
            "O czym rozmawialiśmy w zeszłym miesiącu?",
            "approximately_one_month_before_observation",
            23,
            37,
            0.65,
        ),
        (
            "Co ustaliliśmy 2 miesiące temu?",
            "approximately_2_months_before_observation",
            46,
            74,
            0.65,
        ),
        (
            "O czym rozmawialiśmy tydzień temu?",
            "approximately_one_week_before_observation",
            5,
            9,
            0.7,
        ),
        (
            "Co było w zeszłym tygodniu?",
            "approximately_one_week_before_observation",
            5,
            9,
            0.7,
        ),
        (
            "Co omawialiśmy kilka dni temu?",
            "approximately_several_days_before_observation",
            2,
            7,
            0.6,
        ),
        (
            "O czym pisaliśmy parę tygodni temu?",
            "approximately_several_weeks_before_observation",
            14,
            35,
            0.55,
        ),
    ],
)
def test_build_time_first_query_recognizes_approximate_intervals(
    query: str,
    relation: str,
    min_days: int,
    max_days: int,
    confidence: float,
) -> None:
    plan = build_time_first_query(query, now=NOW)

    assert plan.used_time_filter is True
    assert plan.time_precision == "approximate"
    assert plan.time_source == "relative_text"
    assert plan.time_relation == relation
    assert plan.time_confidence == confidence
    assert plan.start == NOW - timedelta(days=max_days)
    assert plan.end == NOW - timedelta(days=min_days)


def test_temporal_shadow_context_shares_plan_with_time_first_query() -> None:
    direct_plan = build_time_first_query(TEXT, now=NOW)
    shadow_plan, statement = build_temporal_shadow_context(
        TEXT,
        request_id="req-shared",
        now=NOW,
    )

    assert statement is not None
    assert shadow_plan.used_time_filter is True
    assert shadow_plan.start == direct_plan.start
    assert shadow_plan.end == direct_plan.end
    assert shadow_plan.time_precision == "approximate"
    assert shadow_plan.time_relation == direct_plan.time_relation
    assert shadow_plan.preferred_axes == ("semantic", "topic")
    assert direct_plan.preferred_axes == ("semantic", "topic")
    assert statement.utterance_time.start == direct_plan.start
    assert statement.utterance_time.end == direct_plan.end


def test_compound_temporal_statement_supports_opinion_and_week_window() -> None:
    analysis = analyze_temporal_statement_shadow(
        "Tydzień temu uważaliśmy, że temat wymaga rewizji, ale teraz jest konflikt.",
        request_id="req-week-opinion",
        now=NOW,
        conflict=True,
    )

    assert analysis is not None
    assert analysis.epistemic_role == "opinion"
    assert analysis.relevance_gate == "investigate"
    assert (
        analysis.utterance_time.relation
        == "approximately_one_week_before_observation"
    )
    assert analysis.utterance_time.start == NOW - timedelta(days=9)
    assert analysis.utterance_time.end == NOW - timedelta(days=5)
    assert (
        analysis.content_time.relation == "contemporaneous_with_utterance"
    )
    assert analysis.content_time.precision == "unknown"


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


def test_non_epistemic_context_stays_context_only() -> None:
    assert shadow_relevance_gate(has_epistemic_content=False) == "context_only"


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
    opaque_id = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:24]

    with store._connect() as connection:
        command = connection.execute(
            "SELECT input_text FROM commands WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()
        shadow = connection.execute(
            """
            SELECT * FROM context_cascade_shadow_telemetry
            WHERE telemetry_id = ?
            """,
            (opaque_id,),
        ).fetchone()
        raw_id_count = connection.execute(
            """
            SELECT COUNT(*) FROM context_cascade_shadow_telemetry
            WHERE telemetry_id = ?
            """,
            (request.request_id,),
        ).fetchone()[0]
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
    assert raw_id_count == 0
    assert not {"request_id", "raw_text", "claim_text", "query_text"} & shadow_columns


@pytest.mark.asyncio
async def test_shadow_schema_migration_hashes_legacy_request_id(tmp_path) -> None:
    database = tmp_path / "voice.db"
    store = MemoryStore(database)
    await store.initialize()
    request_id = "diagnosis-cancer"
    with store._connect() as connection:
        connection.execute("DROP INDEX idx_context_cascade_shadow_expires")
        connection.execute("DROP TABLE context_cascade_shadow_telemetry")
        connection.execute(
            """
            CREATE TABLE context_cascade_shadow_telemetry (
                request_id TEXT PRIMARY KEY,
                utterance_start TEXT,
                utterance_end TEXT,
                utterance_precision TEXT NOT NULL,
                utterance_source TEXT NOT NULL,
                utterance_confidence REAL NOT NULL,
                content_start TEXT,
                content_end TEXT,
                content_relation TEXT NOT NULL,
                content_precision TEXT NOT NULL,
                content_source TEXT NOT NULL,
                content_confidence REAL NOT NULL,
                epistemic_role TEXT NOT NULL,
                relevance_gate TEXT NOT NULL,
                approved_claim_count INTEGER NOT NULL DEFAULT 0,
                gap_present INTEGER NOT NULL DEFAULT 0,
                axis_insufficiency_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO context_cascade_shadow_telemetry
            VALUES (?, NULL, NULL, 'unknown', 'relative_text', 0.5,
                    NULL, NULL, 'unknown', 'unknown', 'semantic_relation', 0.5,
                    'opinion', 'candidate', 0, 1, '[]', ?, ?)
            """,
            (
                request_id,
                NOW.isoformat(),
                (NOW + timedelta(days=30)).isoformat(),
            ),
        )

    restarted = MemoryStore(database)
    await restarted.initialize()

    opaque_id = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:24]
    with restarted._connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(context_cascade_shadow_telemetry)"
            ).fetchall()
        }
        rows = connection.execute(
            "SELECT telemetry_id FROM context_cascade_shadow_telemetry"
        ).fetchall()
    assert "request_id" not in columns
    assert [row["telemetry_id"] for row in rows] == [opaque_id]
