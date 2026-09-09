from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from voiceloop.memory import MemoryStore
from voiceloop.screenpipe import (
    ScreenpipeClient,
    ScreenpipeElement,
    ScreenpipeError,
    _parse_elements,
)
from voiceloop.settings import Settings
from voiceloop.windows_context import WindowsContextService


@pytest.mark.asyncio
async def test_path_history_schema_upsert_delete_search_and_prune(tmp_path) -> None:
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    root = tmp_path / "root"
    path = root / "Projekt"
    old = datetime.now(UTC) - timedelta(days=100)
    await store.upsert_path_event(
        path=path,
        display_name="Projekt",
        root=root,
        is_dir=True,
        event="create",
        exists=True,
        is_project=True,
        project_type=".git",
        seen_at=old,
    )
    await store.upsert_path_event(
        path=path,
        display_name="Projekt",
        root=root,
        is_dir=True,
        event="delete",
        exists=False,
        is_project=True,
        project_type=".git",
        seen_at=old,
    )

    hits = await store.search_path_history("projekt")
    assert hits[0].exists is False
    assert hits[0].deleted_at == old
    assert hits[0].project_type == ".git"
    assert await store.prune_path_history(retention_days=90, max_records=100) == 1


@pytest.mark.asyncio
async def test_reconcile_uses_only_direct_entries_and_bounds_journal(tmp_path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    project = root / "VoiceLoop"
    project.mkdir()
    (project / ".git").mkdir()
    secret = project / "secret.txt"
    secret.write_text("TOP SECRET CONTENT", encoding="utf-8")
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        windows_context_journal_limit=3,
        screenpipe_enabled=False,
    )
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    service = WindowsContextService(
        settings,
        store,
        ScreenpipeClient(settings),
        roots=[root],
    )

    await service.reconcile()
    first_history = await store.list_path_history(limit=20)
    first_seen = first_history[0].last_seen
    await service.reconcile()
    unchanged_history = await store.list_path_history(limit=20)
    assert unchanged_history[0].last_seen == first_seen
    (root / "a.txt").write_text("one", encoding="utf-8")
    await service.reconcile()
    (root / "b.txt").write_text("two", encoding="utf-8")
    await service.reconcile()

    assert len(service.journal) == 3
    assert any(
        record.name == "VoiceLoop" and record.is_project
        for record in service.records.values()
    )
    history = await store.list_path_history(limit=20)
    serialized = repr(history)
    assert "TOP SECRET CONTENT" not in serialized
    assert "secret.txt" not in serialized


@pytest.mark.asyncio
async def test_screenpipe_context_is_session_only_and_lifecycle_is_safe(tmp_path) -> None:
    settings = Settings(
        voiceloop_data_dir=str(tmp_path),
        windows_context_reconcile_seconds=10,
    )
    store = MemoryStore(tmp_path / "voice.db")
    await store.initialize()
    client = ScreenpipeClient(settings)
    client.recent_elements = AsyncMock(
        return_value=[
            ScreenpipeElement(
                frame_id="1",
                element_id="2",
                text="OCR PRIVATE",
                role="button",
                bounds=(1, 2, 3, 4),
                timestamp="now",
            )
        ]
    )
    service = WindowsContextService(settings, store, client, roots=[])

    await service.start()
    assert service._task is not None
    assert service.screenpipe_elements[0].element.text == "OCR PRIVATE"
    with store._connect() as connection:
        dump = "\n".join(
            str(tuple(row))
            for row in connection.execute("SELECT * FROM local_path_history").fetchall()
        )
    assert "OCR PRIVATE" not in dump
    await service.stop()
    assert service._task is None


def test_screenpipe_element_parser_tolerates_payload_shapes() -> None:
    assert _parse_elements(
        {
            "elements": [
                {
                    "id": 7,
                    "text": "OK",
                    "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
                }
            ]
        },
        frame_id="9",
        limit=10,
    )[0].bounds == (1.0, 2.0, 4.0, 6.0)
    assert _parse_elements(
        [{"content": {"label": "Anuluj", "bbox": [5, 6, 7, 8]}}],
        frame_id="10",
        limit=10,
    )[0].text == "Anuluj"
    assert _parse_elements({"unexpected": True}, frame_id="1", limit=10) == []


@pytest.mark.asyncio
async def test_screenpipe_endpoint_errors_return_no_elements(tmp_path) -> None:
    client = ScreenpipeClient(Settings(voiceloop_data_dir=str(tmp_path)))
    client._recent_frame_ids = AsyncMock(return_value=["1"])
    client._get_json = AsyncMock(side_effect=ScreenpipeError("missing endpoint"))

    assert await client.recent_elements() == []
