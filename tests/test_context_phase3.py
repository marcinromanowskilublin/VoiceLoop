from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from voiceloop.commitments.schema import CommitmentItem, CommitmentType
from voiceloop.context.documents import DocumentAccessError, DocumentTimelineIngestor
from voiceloop.context.evaluation import (
    ContextRetrievalEvalRecordV1,
    compare_context_retrieval_shadow,
)
from voiceloop.context.projection import MemoryTimelineMigrator, project_windows_projects
from voiceloop.context.review import (
    commitment_review_event,
    live_screen_item,
    question_is_deictic,
)
from voiceloop.context.schema import (
    ContextItemV1,
    ContextPackV1,
    ContextScope,
    ContextTrust,
)
from voiceloop.memory import MemoryStore
from voiceloop.models import CommandRequest, MemoryCreate, ScreenSnapshot


@pytest.mark.asyncio
async def test_document_ingest_skips_secrets_and_generated_dirs(tmp_path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "ustalenie.md").write_text("Ustaliliśmy termin na piątek.", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret-value-12345678", encoding="utf-8")
    (root / "leak.txt").write_text("api_key=abcdefghijklmnop", encoding="utf-8")
    nested = root / "node_modules"
    nested.mkdir()
    (nested / "skip.md").write_text("Nie zapisuj wygenerowanego katalogu.", encoding="utf-8")
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()

    report = await DocumentTimelineIngestor(memory=store, roots=[root]).ingest(
        observed_at=datetime(2026, 9, 21, tzinfo=UTC)
    )

    assert report.stored_events == 1
    assert (report.skipped, report.skipped_secret) == (1, 1)
    hits = await store.search_context_events(query="piątek", source="local_document")
    assert len(hits) == 1
    assert "TOKEN" not in hits[0].text
    assert hits[0].metadata["vectorize"] is False


@pytest.mark.asyncio
async def test_document_event_keeps_modification_time_and_structure(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    document = root / "plan.md"
    document.write_text(
        "# Plan\n\n- kontakt: kto@example.com\n- termin: piątek\n",
        encoding="utf-8",
    )
    modified_at = datetime(2026, 9, 1, 10, tzinfo=UTC)
    os.utime(document, (modified_at.timestamp(), modified_at.timestamp()))
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    ingestor = DocumentTimelineIngestor(memory=store, roots=[root])

    first = await ingestor.ingest(observed_at=datetime(2026, 9, 21, tzinfo=UTC))
    second = await ingestor.ingest(observed_at=datetime(2026, 9, 28, tzinfo=UTC))

    assert (first.stored_events, second.stored_events) == (1, 1)
    hits = await store.search_context_events(query="termin", source="local_document")
    assert len(hits) == 1
    event = hits[0]
    assert event.started_at == modified_at
    assert "\n" in event.text
    assert event.window_title == "Plan"
    assert "kto@example.com" not in event.text
    assert event.expires_at == datetime(2026, 10, 12, tzinfo=UTC)


@pytest.mark.asyncio
async def test_long_document_stores_digest_and_reads_full_text_on_demand(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    body = "\n".join(f"- ustalenie numer {index}" for index in range(400))
    (root / "dziennik.md").write_text(f"# Dziennik\n\n{body}\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    ingestor = DocumentTimelineIngestor(memory=store, roots=[root], digest_chars=500)

    await ingestor.ingest(observed_at=datetime(2026, 9, 21, tzinfo=UTC))
    event = (await store.search_context_events(query="Dziennik"))[0]
    full_text = await ingestor.load_full_text(event)

    assert event.metadata["digest_truncated"] is True
    assert len(event.text) < 600
    assert event.text.endswith("[...]")
    assert "ustalenie numer 399" not in event.text
    assert "ustalenie numer 399" in full_text


@pytest.mark.asyncio
async def test_full_read_refuses_a_reference_outside_the_roots(tmp_path) -> None:
    root = tmp_path / "docs"
    outside = tmp_path / "secret"
    root.mkdir()
    outside.mkdir()
    (root / "notatka.md").write_text("Notatka lokalna.", encoding="utf-8")
    (outside / "wykradzione.md").write_text("Prywatne dane.", encoding="utf-8")
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    ingestor = DocumentTimelineIngestor(memory=store, roots=[root])
    await ingestor.ingest(observed_at=datetime(2026, 9, 21, tzinfo=UTC))
    event = (await store.search_context_events(query="Notatka"))[0]
    tampered = event.model_copy(
        update={
            "metadata": {
                **event.metadata,
                "document_path": str(outside / "wykradzione.md"),
            }
        }
    )

    with pytest.raises(DocumentAccessError):
        await ingestor.load_full_text(tampered)


@pytest.mark.asyncio
async def test_windows_projection_keeps_paths_out_of_event_text(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    secret_path = r"C:\Users\marcin\private\VoiceLoop"
    first_seen = datetime(2026, 8, 30, 9, tzinfo=UTC)
    records = [
        SimpleNamespace(
            is_project=True,
            exists=True,
            name="VoiceLoop",
            project_type=".git",
            path=secret_path,
            first_seen=first_seen,
        ),
        SimpleNamespace(
            is_project=False,
            exists=True,
            name="notatka.txt",
            project_type="",
            path=r"C:\Users\marcin\notatka.txt",
            first_seen=first_seen,
        ),
    ]
    report = await project_windows_projects(
        store,
        records,
        observed_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
    )
    await project_windows_projects(
        store,
        records,
        observed_at=datetime(2026, 9, 22, 12, tzinfo=UTC),
    )

    assert report.stored_events == 1
    hits = await store.search_context_events(query="VoiceLoop", source="windows_context")
    assert len(hits) == 1
    event = hits[0]
    assert secret_path not in event.text
    assert event.metadata["vectorize"] is False
    assert event.started_at == first_seen
    assert event.metadata["last_seen"] == "2026-09-22T12:00:00+00:00"
    assert event.expires_at == datetime(2026, 10, 6, 12, tzinfo=UTC)


@pytest.mark.asyncio
async def test_memory_migration_is_idempotent_and_skips_secrets(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    await store.create_memory(MemoryCreate(kind="fact", content="Spotkanie jest w piątek."))
    await store.create_memory(
        MemoryCreate(kind="fact", content="api_key=abcdefghijklmnop")
    )
    migrator = MemoryTimelineMigrator(store)

    first = await migrator.migrate()
    second = await migrator.migrate()

    assert first.stored_events == 1
    assert first.skipped == 1
    assert second.stored_events == 1
    hits = await store.search_context_events(query="piątek", source="manual_memory")
    assert len(hits) == 1
    episodes = await store.search_context_episodes(query="piątek", source="manual_memory")
    assert len(episodes) == 1
    assert episodes[0].expires_at is None


@pytest.mark.asyncio
async def test_disabled_flag_keeps_live_screen_out_of_the_pack(tmp_path) -> None:
    """A captured screen must not reach the pack while the flag stays off."""

    from tests.test_assistant import ExecutorStub, N8nStub, RouterStub, TTSStub, _assistant

    class ScreenStub:
        def __init__(self) -> None:
            self.captures = 0

        async def capture(self, _request_id):
            self.captures += 1
            return ScreenSnapshot(
                window_title="Cursor",
                process_name="Cursor.exe",
                captured_at=datetime(2026, 9, 21, tzinfo=UTC),
            )

        @staticmethod
        def image_data_url(_snapshot):
            return None

    async def run(*, enabled: bool, include_screen: bool) -> tuple[list[str], int]:
        memory = MemoryStore(tmp_path / f"voice-{enabled}-{include_screen}.db")
        await memory.initialize()
        router = RouterStub()
        assistant = _assistant(
            memory,
            router=router,
            n8n=N8nStub(),
            executor=ExecutorStub(memory),
            tts=TTSStub(),
        )
        assistant.deictic_screen_enabled = enabled
        screen = ScreenStub()
        assistant.screen = screen  # type: ignore[assignment]
        await assistant._create_plan(
            CommandRequest(text="co jest na tym ekranie", include_screen=include_screen),
            None,
            conversation_active=True,
        )
        await assistant.close()
        return list(router.calls[-1]["memories"]), screen.captures

    off_memories, off_captures = await run(enabled=False, include_screen=True)
    on_memories, on_captures = await run(enabled=True, include_screen=False)

    assert off_captures == 1
    assert not any("source=live_screen" in entry for entry in off_memories)
    assert on_captures == 1
    assert any("source=live_screen" in entry for entry in on_memories)


def test_deictic_detector_ignores_common_polish_to() -> None:
    assert question_is_deictic("co jest na tym ekranie") is True
    assert question_is_deictic("to jest plan na jutro") is False
    item = live_screen_item(
        ScreenSnapshot(
            window_title="Cursor",
            process_name="Cursor.exe",
            captured_at=datetime(2026, 9, 21, tzinfo=UTC),
        )
    )
    assert item is not None
    assert item.scope is ContextScope.LIVE
    assert item.metadata["persisted"] is False
    assert item.expires_at == datetime(2026, 9, 21, 0, 5, tzinfo=UTC)


def test_commitment_review_event_is_not_executable() -> None:
    event = commitment_review_event(
        CommitmentItem(
            source_chunk_id="chunk-1",
            speaker="user",
            raw_text="Zadzwonię do Mikołaja jutro.",
            type=CommitmentType.PROMISE,
        ),
        request_id="req-1",
        observed_at=datetime(2026, 9, 21, tzinfo=UTC),
    )
    assert event is not None
    assert event.metadata["executable"] is False
    assert event.metadata["review_required"] is True
    assert event.event_type == "commitment_review"
    assert event.expires_at == datetime(2026, 10, 5, tzinfo=UTC)


@pytest.mark.asyncio
async def test_cli_ingests_documents_and_prunes_only_with_apply(tmp_path, capsys) -> None:
    from voiceloop.corpus.cli import _run_async, build_parser

    root = tmp_path / "notes"
    root.mkdir()
    (root / "ustalenie.md").write_text("Termin ustalony na piątek.", encoding="utf-8")
    database = tmp_path / "voice.db"
    store = MemoryStore(database)
    await store.initialize()
    await store.create_memory(MemoryCreate(kind="fact", content="Pamięć do migracji."))
    parser = build_parser()

    ingest_code = await _run_async(
        parser.parse_args(
            [
                "ingest-context-documents",
                "--root",
                str(root),
                "--database",
                str(database),
            ]
        )
    )
    ingest_out = json.loads(capsys.readouterr().out)
    migrate_code = await _run_async(
        parser.parse_args(
            ["migrate-memories-to-timeline", "--database", str(database)]
        )
    )
    migrate_out = json.loads(capsys.readouterr().out)
    prune_code = await _run_async(
        parser.parse_args(["prune-context-timeline", "--database", str(database)])
    )
    prune_out = json.loads(capsys.readouterr().out)

    assert (ingest_code, migrate_code, prune_code) == (0, 0, 0)
    assert ingest_out["stored_events"] == 1
    assert ingest_out["vectorized"] is False
    assert ingest_out["ttl_days"] == 14
    assert migrate_out["stored_episodes"] == 1
    assert migrate_out["ttl_days"] is None
    assert prune_out["dry_run"] is True
    assert prune_out["qdrant_deleted"] == 0
    assert len(await store.search_context_events(query="piątek")) == 1


@pytest.mark.asyncio
async def test_shadow_comparison_does_not_switch_on_a_regression() -> None:
    record = ContextRetrievalEvalRecordV1(
        example_id="q1",
        query="co ustaliliśmy",
        expected_source_ids=("ep-1",),
    )
    good = ContextItemV1(
        source="timeline",
        source_id="ep-1",
        scope=ContextScope.EPISODIC,
        content="Ustalenie z piątku.",
        trust=ContextTrust.DERIVED,
        selection_reason="timeline_episode",
    )

    async def baseline(record, k):
        return ContextPackV1(question=record.query, items=())

    async def candidate(record, k):
        return ContextPackV1(question=record.query, items=(good,))

    async def worse(record, k):
        return ContextPackV1(question=record.query, items=())

    improved = await compare_context_retrieval_shadow(
        records=[record],
        baseline=baseline,
        candidate=candidate,
    )
    regressed = await compare_context_retrieval_shadow(
        records=[record],
        baseline=candidate,
        candidate=worse,
    )

    assert improved.recall_delta > 0
    assert improved.candidate_recall_not_worse is True
    assert regressed.candidate_recall_not_worse is False
