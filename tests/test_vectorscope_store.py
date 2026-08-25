"""Testy zapisu na dysku i warstwy odtwarzalności.

Panel musi czytać także nagrania sprzed zmiany formatu — inaczej po każdej
przebudowie starsze eksperymenty przestają istnieć, a to przeczy sensowi
zapisywania warunków pomiaru.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vectorscope.store import RecordingStore, transcript_hash


def _store(tmp_path: Path) -> RecordingStore:
    return RecordingStore(tmp_path / "vectorscope")


def _write_legacy(store: RecordingStore, recording_id: str, payload: dict) -> Path:
    directory = store.root / recording_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------- skrót treści


def test_transcript_hash_is_stable_across_key_order() -> None:
    first = {"text": "kot", "words": [{"text": "kot", "start": 0.0, "end": 0.3}]}
    second = {"words": [{"end": 0.3, "start": 0.0, "text": "kot"}], "text": "kot"}
    assert transcript_hash(first) == transcript_hash(second)


def test_transcript_hash_changes_when_timing_changes() -> None:
    base = {"text": "kot", "words": [{"text": "kot", "start": 0.0, "end": 0.3}]}
    shifted = {"text": "kot", "words": [{"text": "kot", "start": 0.0, "end": 0.4}]}
    assert transcript_hash(base) != transcript_hash(shifted)


def test_transcript_hash_ignores_fields_that_do_not_affect_vectors() -> None:
    base = {"text": "kot", "words": [{"text": "kot", "start": 0.0, "end": 0.3}]}
    noisy = dict(base) | {"raw": {"cokolwiek": 1}, "utterances": []}
    assert transcript_hash(base) == transcript_hash(noisy)


# ------------------------------------------------------ tolerancja starych pól


def test_legacy_meta_is_readable_with_its_old_field_names(tmp_path: Path) -> None:
    """Regresja: starsze nagrania pokazywały 0 B obok poprawnego pliku audio."""

    store = _store(tmp_path)
    directory = _write_legacy(
        store,
        "20260824-080225-a144e3",
        {
            "id": "20260824-080225-a144e3",
            "created_at": "2026-08-24T08:02:25+00:00",
            "label": "Proba 2",
            "mime": "audio/wav",
            "bytes": 382754,
            "duration_seconds": 10.0,
            "mic_processing": False,
            "transcript_status": "ok",
            "language": "pl",
            "text": "Chory nie moze zasnac w nocy.",
            "word_count": 16,
        },
    )
    (directory / "audio.webm").write_bytes(b"x" * 10)

    meta = store.read_meta("20260824-080225-a144e3")
    assert meta.size_bytes == 382754
    assert meta.microphone_processing is False
    assert meta.transcript_language == "pl"
    assert meta.text_preview.startswith("Chory nie moze")
    assert meta.audio_file == "audio.webm"


def test_missing_audio_file_name_is_resolved_from_disk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    directory = _write_legacy(store, "rec-a", {"id": "rec-a", "label": "a"})
    (directory / "audio.wav").write_bytes(b"x")

    assert store.read_meta("rec-a").audio_file == "audio.wav"


def test_missing_audio_falls_back_to_a_default_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_legacy(store, "rec-b", {"id": "rec-b", "label": "b"})
    assert store.read_meta("rec-b").audio_file == "audio.webm"


def test_unknown_keys_do_not_break_reading(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_legacy(
        store,
        "rec-c",
        {"id": "rec-c", "label": "c", "cos_czego_nie_znamy": {"a": 1}, "levels": {"words": 3}},
    )
    meta = store.read_meta("rec-c")
    assert meta.id == "rec-c"
    assert meta.label == "c"


def test_defaults_describe_the_current_pipeline(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_legacy(store, "rec-d", {"id": "rec-d", "label": "d"})
    meta = store.read_meta("rec-d")
    assert meta.segmentation_version.startswith("vectorscope-segmentation")
    assert meta.vectorscope_version
    assert meta.transcript_status in {"pending", "ok", "error"}


# ------------------------------------------------------------ odtwarzalność


def test_timings_and_errors_are_recorded_for_the_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_legacy(store, "rec-e", {"id": "rec-e", "label": "e"})
    meta = store.read_meta("rec-e")

    meta.record_timing("transcribe", 1853.4839)
    meta.record_error("embedding", "LM Studio nie odpowiada")
    meta.record_embedding_run(
        level="word",
        prefix="search_document",
        model="nomic",
        dimension=768,
        fragment_count=17,
        over_context=0,
    )
    store.write_meta(meta)

    reloaded = store.read_meta("rec-e")
    assert reloaded.timings_ms["transcribe"] == pytest.approx(1853.48)
    assert reloaded.errors[0]["stage"] == "embedding"
    assert "at" in reloaded.errors[0]
    run = reloaded.embedding_runs["word|search_document"]
    assert run["dimension"] == 768
    assert run["normalized_on_disk"] is False, "wektory leżą surowe, nie znormalizowane"


def test_identifier_with_a_path_separator_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.read_meta("../../etc/passwd")
