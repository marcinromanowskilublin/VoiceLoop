from __future__ import annotations

from voiceloop.routing import resolver, segmenter, taxonomy, vector_documents
from voiceloop.routing.segmenter import segment_command
from voiceloop.routing.taxonomy import (
    ACTION_OPERATIONS,
    ACTION_TARGETS,
    CURSOR_MARKERS,
    TARGET_PATTERNS,
    operation_for_token,
)
from voiceloop.routing.validation import validate_arguments
from voiceloop.routing.vector_documents import command_documents, subtask_documents


def test_routing_layers_share_taxonomy_objects() -> None:
    assert segmenter.operation_for_token is taxonomy.operation_for_token
    assert segmenter.TARGET_PATTERNS is TARGET_PATTERNS
    assert resolver.ACTION_OPERATIONS is ACTION_OPERATIONS
    assert resolver.ACTION_TARGETS is ACTION_TARGETS
    assert resolver.CURSOR_MARKERS is CURSOR_MARKERS
    assert vector_documents.operation_document_label is taxonomy.operation_document_label
    assert vector_documents.target_document_label is taxonomy.target_document_label


def test_operation_synonym_is_shared_with_segmenter_and_documents() -> None:
    assert operation_for_token("odpal") == "open"

    segmentation = segment_command("odpal Chrome")
    subtask = segmentation.subtasks[0]
    documents = subtask_documents(subtask)

    assert subtask.operation == ACTION_OPERATIONS["open_browser"] == "open"
    assert subtask.target in ACTION_TARGETS["open_browser"]
    assert "open (otworzyć)" in documents.intent
    assert "browser (przeglądarka)" in documents.target_context


def test_cursor_target_and_markers_have_one_canonical_identity() -> None:
    segmentation = segment_command("skopiuj tekst pod kursorem")
    subtask = segmentation.subtasks[0]

    assert subtask.target == "text_under_cursor"
    assert subtask.target in ACTION_TARGETS["copy_text_under_cursor"]
    assert any(marker in taxonomy.CURSOR_MARKERS for marker in ("kursor", "mysz"))
    assert "text_under_cursor (tekst pod kursorem)" in subtask_documents(
        subtask
    ).target_context


def test_v1_command_documents_keep_polish_labels_from_taxonomy() -> None:
    documents = command_documents(
        "Zamknij okno pod kursorem, otwórz Chrome i kartę z YouTubem."
    )

    assert "zamknąć" in documents.intent
    assert "otworzyć" in documents.intent
    assert "okno pod kursorem" in documents.target_context
    assert "przeglądarka" in documents.target_context
    assert "YouTube" in documents.target_context


def test_allowlist_enum_rejects_unknown_app() -> None:
    errors = validate_arguments(
        {"app_id": "autodesk"},
        {
            "type": "object",
            "properties": {"app_id": {"type": "string", "enum": ["whatsapp"]}},
            "required": ["app_id"],
            "additionalProperties": False,
        },
    )

    assert "invalid_argument_enum:app_id" in errors
