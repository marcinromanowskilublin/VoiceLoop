import time

from voiceloop.conversation_telemetry import ConversationTelemetry
from voiceloop.events import EventBus


async def test_turn_trace_correlates_phases_and_builds_aggregates() -> None:
    telemetry = ConversationTelemetry(EventBus(), max_traces=20)

    for turn_id in range(1, 21):
        started = time.perf_counter() - (turn_id / 1000)
        await telemetry.begin(
            session_id="session-1",
            turn_id=turn_id,
            request_id=f"request-{turn_id}",
            started_perf=started,
            initial_phases_ms={"speech_started": 0},
        )
        await telemetry.mark_request(f"request-{turn_id}", "speech_end")
        await telemetry.mark_request(f"request-{turn_id}", "context_started")
        await telemetry.mark_request(f"request-{turn_id}", "context_ready")
        await telemetry.finish(
            session_id="session-1",
            turn_id=turn_id,
            status="listening_once",
        )

    snapshot = await telemetry.snapshot(limit=20)

    assert len(snapshot.traces) == 20
    assert snapshot.traces[0].turn_id == 20
    assert snapshot.traces[0].request_id == "request-20"
    assert snapshot.traces[0].metrics_ms["stt_ms"] >= 15
    assert snapshot.aggregates["stt_ms"]["count"] == 20
    assert snapshot.aggregates["stt_ms"]["p95"] >= snapshot.aggregates["stt_ms"]["p50"]
    report = await telemetry.quality_report()
    assert report["baseline_ready"] is True
    assert report["passed"] is False


async def test_unknown_request_marker_is_ignored() -> None:
    telemetry = ConversationTelemetry(EventBus())

    assert await telemetry.mark_request("missing", "model_started") is None


async def test_completed_traces_are_available_after_restart(tmp_path) -> None:
    path = tmp_path / "conversation-traces.jsonl"
    telemetry = ConversationTelemetry(EventBus(), storage_path=path)
    await telemetry.begin(session_id="session-persisted", turn_id=1)
    await telemetry.finish(
        session_id="session-persisted",
        turn_id=1,
        status="listening_once",
    )

    restored = ConversationTelemetry(EventBus(), storage_path=path)
    snapshot = await restored.snapshot(limit=5)

    assert snapshot.traces[0].session_id == "session-persisted"
    assert snapshot.traces[0].status == "listening_once"
