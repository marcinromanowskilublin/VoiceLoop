from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import SubtaskV1
from ..router import normalize_text
from .taxonomy import (
    TAXONOMY_VERSION,
    action_operation_label,
    action_target_labels,
    match_document_operation_labels,
    match_document_target_labels,
    operation_document_label,
    target_document_label,
)

VECTOR_NAMES = ("semantic", "intent", "target_context")
CAPABILITY_DOCUMENT_FORMAT_VERSION = (
    f"capability-document-v2:{TAXONOMY_VERSION}"
)
COMMAND_QUERY_FORMAT_VERSION = f"capability-query-v1:{TAXONOMY_VERSION}"
SUBTASK_QUERY_FORMAT_VERSION = f"capability-query-v2:{TAXONOMY_VERSION}"


@dataclass(frozen=True, slots=True)
class CapabilityDocuments:
    semantic: str
    intent: str
    target_context: str

    def as_dict(self) -> dict[str, str]:
        return {
            "semantic": self.semantic,
            "intent": self.intent,
            "target_context": self.target_context,
        }


@dataclass(frozen=True, slots=True)
class VectorDocumentBuilder:
    capability_format_version: str = CAPABILITY_DOCUMENT_FORMAT_VERSION
    command_query_format_version: str = COMMAND_QUERY_FORMAT_VERSION
    subtask_query_format_version: str = SUBTASK_QUERY_FORMAT_VERSION

    def capability(self, definition: dict[str, Any]) -> CapabilityDocuments:
        action_id = str(definition.get("id") or "").strip()
        description = str(definition.get("description") or "").strip()
        label = str(definition.get("label") or description or action_id).strip()
        operation = action_operation_label(action_id)
        target_tokens = action_target_labels(action_id)
        args_schema = definition.get("args_schema")
        properties = args_schema.get("properties", {}) if isinstance(args_schema, dict) else {}
        argument_names = (
            sorted(str(name) for name in properties) if isinstance(properties, dict) else []
        )
        raw_examples = definition.get("routing_examples")
        examples = (
            _unique(str(example) for example in raw_examples)
            if isinstance(raw_examples, list | tuple)
            else []
        )
        target_parts = target_tokens + [f"argument {name}" for name in argument_names]
        availability = (
            "dostępna także w VoiceAttack"
            if bool(definition.get("available_in_voiceattack"))
            else "wykonywana natywnie przez VoiceLoop"
        )
        return CapabilityDocuments(
            semantic=". ".join(
                part
                for part in (
                    label,
                    description,
                    f"Przykładowe polecenia: {'; '.join(examples)}" if examples else "",
                )
                if part
            ),
            intent=f"Dozwolona operacja: {operation}. Akcja: {label}.",
            target_context=(
                f"Cel i kontekst akcji: {', '.join(target_parts) or label}; "
                f"{'przykłady: ' + '; '.join(examples) + '; ' if examples else ''}"
                f"{availability}."
            ),
        )

    def command_query(self, text: str) -> CapabilityDocuments:
        """Public/V1 query representation retained for API compatibility."""
        clean_text = _clean_text(text)
        normalized = normalize_text(clean_text)
        operations = match_document_operation_labels(normalized)
        targets = match_document_target_labels(normalized)
        return CapabilityDocuments(
            semantic=clean_text,
            intent=(
                f"Operacja użytkownika: {', '.join(operations)}."
                if operations
                else f"Intencja użytkownika: {clean_text}."
            ),
            target_context=(
                f"Cel i kontekst polecenia: {', '.join(targets)}."
                if targets
                else f"Cel polecenia użytkownika: {clean_text}."
            ),
        )

    def subtask_query(self, subtask: SubtaskV1) -> CapabilityDocuments:
        """V2 query representation built from the segmenter's structured output."""
        clean_text = _clean_text(subtask.text)
        operation = str(subtask.operation or "").strip()
        intent_operation = (
            f"{operation} ({operation_document_label(operation)})"
            if operation
            else "nieokreślona"
        )
        target = str(subtask.target or "").strip()
        target_label = target_document_label(target)
        target_context = (
            target
            if target and target_label == target
            else f"{target} ({target_label})"
            if target
            else "nieokreślony"
        )
        arguments = _render_arguments(subtask.raw_arguments)
        return CapabilityDocuments(
            semantic=clean_text,
            intent=f"Operacja użytkownika: {intent_operation}. Polecenie: {clean_text}.",
            target_context=(
                f"Cel polecenia: {target_context}. "
                f"Argumenty: {arguments or 'brak'}. Polecenie: {clean_text}."
            ),
        )


DEFAULT_VECTOR_DOCUMENT_BUILDER = VectorDocumentBuilder()


def capability_documents(definition: dict[str, Any]) -> CapabilityDocuments:
    return DEFAULT_VECTOR_DOCUMENT_BUILDER.capability(definition)


def command_documents(text: str) -> CapabilityDocuments:
    return DEFAULT_VECTOR_DOCUMENT_BUILDER.command_query(text)


def subtask_documents(subtask: SubtaskV1) -> CapabilityDocuments:
    return DEFAULT_VECTOR_DOCUMENT_BUILDER.subtask_query(subtask)


def subtask_query_documents(subtask: SubtaskV1) -> CapabilityDocuments:
    return subtask_documents(subtask)


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())[:2000]


def _render_arguments(arguments: dict[str, str]) -> str:
    rendered = "; ".join(
        f"{str(name).strip()}: {' '.join(str(value).strip().split())[:500]}"
        for name, value in sorted(arguments.items())
        if str(name).strip() and str(value).strip()
    )
    return rendered[:2000]


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
