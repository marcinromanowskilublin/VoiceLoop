from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..capability_index import SubtaskCapabilitySearch
from ..models import (
    ResolutionCandidateV1,
    ResolutionDecisionV1,
    ResolutionStatusV1,
    SubtaskV1,
)
from ..router import normalize_text
from .taxonomy import (
    ACTION_OPERATIONS,
    ACTION_TARGETS,
    CURSOR_MARKERS,
    UNDER_CURSOR_ACTIONS,
)
from .validation import validate_arguments
from .vector_documents import VECTOR_NAMES

_KNOWN_URLS = {
    "YouTube": "https://www.youtube.com",
}
_DESCRIBE_ACTIONS = {
    "describe_active_window",
    "describe_recent_activity",
    "describe_text_target",
}
_STOPWORDS = {
    "a",
    "do",
    "i",
    "mi",
    "na",
    "o",
    "oraz",
    "pod",
    "po",
    "proszę",
    "prosze",
    "ten",
    "to",
    "w",
    "z",
}
_ORDINALS = {
    "pierwsz": 1,
    "drug": 2,
    "trzec": 3,
    "czwart": 4,
    "piat": 5,
    "szost": 6,
    "siodm": 7,
    "osm": 8,
    "dziewiat": 9,
    "dziesiat": 10,
}
RESOLVER_WEIGHTS = {
    "vector": 0.60,
    "lexical": 0.35,
    "arguments": 0.05,
}
MINIMUM_VECTOR_COVERAGE = 2 / len(VECTOR_NAMES)
MISSING_SPACE_POLICY = "coverage-adjusted-observed-cosine-v1"
SINGLE_CANDIDATE_POLICY = "clarify-without-comparator-v1"


def resolve_subtasks(
    searches: Sequence[SubtaskCapabilitySearch],
    *,
    definitions: Sequence[dict[str, Any]],
    transcript_confidence: float | None,
    min_score: float,
    min_margin: float,
    stt_threshold: float,
) -> tuple[ResolutionDecisionV1, ...]:
    definitions_by_id = {
        str(definition.get("id") or ""): definition
        for definition in definitions
        if str(definition.get("id") or "")
    }
    return tuple(
        _resolve_one(
            search,
            definitions_by_id=definitions_by_id,
            transcript_confidence=transcript_confidence,
            min_score=min_score,
            min_margin=min_margin,
            stt_threshold=stt_threshold,
        )
        for search in searches
    )


def selected_candidate(
    decision: ResolutionDecisionV1,
) -> ResolutionCandidateV1 | None:
    if decision.decision is not ResolutionStatusV1.RESOLVED or decision.top1_action_id is None:
        return None
    return next(
        (
            candidate
            for candidate in decision.candidates
            if candidate.action_id == decision.top1_action_id
        ),
        None,
    )


def _resolve_one(
    search: SubtaskCapabilitySearch,
    *,
    definitions_by_id: dict[str, dict[str, Any]],
    transcript_confidence: float | None,
    min_score: float,
    min_margin: float,
    stt_threshold: float,
) -> ResolutionDecisionV1:
    candidates: list[ResolutionCandidateV1] = []
    for match in search.result.matches:
        definition = definitions_by_id.get(match.action_id)
        reasons: list[str] = []
        if definition is None:
            reasons.append("unknown_action")
            args: dict[str, Any] = {}
            argument_compatibility = 0.0
        else:
            args = _extract_arguments(match.action_id, search.subtask)
            schema_errors = validate_arguments(
                args,
                definition.get("args_schema"),
            )
            reasons.extend(schema_errors)
            reasons.extend(_context_rejections(match.action_id, search.subtask))
            argument_compatibility = 1.0 if not schema_errors else 0.0

        lexical_score = _lexical_score(
            search.subtask,
            match.action_id,
            definition or {},
        )
        observed_vector_scores = {
            name: max(
                -1.0,
                min(float(score), 1.0),
            )
            for name, score in match.vector_scores.items()
            if name in VECTOR_NAMES
        }
        evidence_coverage = len(observed_vector_scores) / len(VECTOR_NAMES)
        reported_coverage = max(
            0.0,
            min(float(getattr(match, "coverage", evidence_coverage)), 1.0),
        )
        coverage = min(reported_coverage, evidence_coverage)
        observed_cosine = (
            sum(max(0.0, score) for score in observed_vector_scores.values())
            / len(observed_vector_scores)
            if observed_vector_scores
            else 0.0
        )
        coverage_adjusted_vector_score = observed_cosine * coverage
        if coverage < MINIMUM_VECTOR_COVERAGE:
            reasons.append("insufficient_vector_coverage")
        observed_vector_ranks = {
            name: int(rank)
            for name, rank in getattr(match, "vector_ranks", {}).items()
            if name in observed_vector_scores and int(rank) > 0
        }
        combined_score = (
            coverage_adjusted_vector_score * RESOLVER_WEIGHTS["vector"]
            + lexical_score * RESOLVER_WEIGHTS["lexical"]
            + argument_compatibility * RESOLVER_WEIGHTS["arguments"]
        )
        if reasons:
            combined_score = 0.0
        candidates.append(
            ResolutionCandidateV1(
                action_id=match.action_id,
                vector_score=coverage_adjusted_vector_score,
                vector_scores=observed_vector_scores,
                vector_ranks=observed_vector_ranks,
                coverage=coverage,
                missing_vector_names=tuple(
                    name for name in VECTOR_NAMES if name not in observed_vector_scores
                ),
                lexical_score=lexical_score,
                argument_compatibility=argument_compatibility,
                combined_score=combined_score,
                extracted_args=args,
                eligible=not reasons,
                rejection_reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    candidates.sort(key=lambda item: (-item.combined_score, item.action_id))
    top = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin = (
        top.combined_score - runner_up.combined_score
        if top is not None and runner_up is not None
        else None
    )
    base = {
        "subtask_id": search.subtask.subtask_id,
        "candidates": tuple(candidates),
        "top1_action_id": top.action_id if top else None,
        "margin_top2": margin,
        "stt_confidence": transcript_confidence,
        "catalog_hash": search.result.catalog_hash,
    }

    if transcript_confidence is not None and transcript_confidence < stt_threshold:
        return ResolutionDecisionV1(
            **base,
            decision=ResolutionStatusV1.CLARIFY,
            reason="low_stt_confidence",
        )
    if top is None:
        return ResolutionDecisionV1(
            **base,
            decision=ResolutionStatusV1.UNSUPPORTED,
            reason="no_capability_candidates",
        )
    if not top.eligible:
        reason = top.rejection_reasons[0] if top.rejection_reasons else "candidate_rejected"
        status = (
            ResolutionStatusV1.CLARIFY
            if any(
                item.startswith(
                    (
                        "missing_required",
                        "invalid_",
                        "unexpected_argument",
                        "insufficient_vector_coverage",
                    )
                )
                for item in top.rejection_reasons
            )
            else ResolutionStatusV1.UNSUPPORTED
        )
        return ResolutionDecisionV1(
            **base,
            decision=status,
            reason=reason,
        )
    if top.combined_score < min_score:
        return ResolutionDecisionV1(
            **base,
            decision=ResolutionStatusV1.CLARIFY,
            reason="low_combined_score",
        )
    if runner_up is None:
        return ResolutionDecisionV1(
            **base,
            decision=ResolutionStatusV1.CLARIFY,
            reason="single_candidate_without_comparator",
        )
    if margin is None or margin < min_margin:
        return ResolutionDecisionV1(
            **base,
            decision=ResolutionStatusV1.CLARIFY,
            reason="low_top2_margin",
        )
    return ResolutionDecisionV1(
        **base,
        decision=ResolutionStatusV1.RESOLVED,
    )


def _context_rejections(action_id: str, subtask: SubtaskV1) -> list[str]:
    reasons: list[str] = []
    expected_operation = ACTION_OPERATIONS.get(action_id)
    if expected_operation is None or expected_operation != subtask.operation:
        reasons.append("operation_mismatch")

    normalized = normalize_text(subtask.source_text)
    if action_id in UNDER_CURSOR_ACTIONS and not any(
        marker in normalized for marker in CURSOR_MARKERS
    ):
        if subtask.operation in {"close", "minimize"} and subtask.target not in {
            "window_under_cursor",
            None,
        }:
            reasons.append("target_identity_not_supported")
        else:
            reasons.append("cursor_target_not_explicit")

    accepted_targets = ACTION_TARGETS.get(action_id)
    if (
        expected_operation == "open"
        and accepted_targets
        and subtask.target is None
    ):
        reasons.append("target_context_missing")
    if accepted_targets and subtask.target is not None and subtask.target not in accepted_targets:
        reasons.append("target_mismatch")
    if action_id == "copy_selected_text" and subtask.target != "selected_text":
        reasons.append("selected_text_not_explicit")
    expected_describe_action = _describe_action_for(
        normalized,
        target=subtask.target,
    )
    if action_id in _DESCRIBE_ACTIONS:
        if expected_describe_action is None:
            reasons.append("describe_context_missing")
        elif action_id != expected_describe_action:
            reasons.append("describe_context_mismatch")
    if action_id == "search_web" and expected_describe_action is not None:
        reasons.append("local_context_not_web_search")
    if (
        subtask.target == "UI Vision"
        and subtask.operation == "close"
        and action_id == "close_window_under_cursor"
    ):
        reasons.append("target_identity_not_supported")
    if action_id == "remember_last_source" and not any(
        marker in normalized for marker in ("zrodl", "link", "wynik", "ostatn", "poprzedn")
    ):
        reasons.append("source_context_missing")
    return list(dict.fromkeys(reasons))


def _lexical_score(
    subtask: SubtaskV1,
    action_id: str,
    definition: dict[str, Any],
) -> float:
    expected_operation = ACTION_OPERATIONS.get(action_id)
    operation_score = 1.0 if expected_operation == subtask.operation else 0.0
    accepted_targets = ACTION_TARGETS.get(action_id)
    if accepted_targets is None or subtask.target is None:
        target_score = 0.5
    else:
        target_score = 1.0 if subtask.target in accepted_targets else 0.0

    query_tokens = _tokens(subtask.normalized_text)
    texts = [
        action_id.replace("_", " "),
        str(definition.get("label") or ""),
        str(definition.get("description") or ""),
        *(str(item) for item in definition.get("routing_examples") or ()),
    ]
    overlap_score = max(
        (_token_overlap(query_tokens, _tokens(text)) for text in texts if text.strip()),
        default=0.0,
    )
    return max(
        0.0,
        min(operation_score * 0.5 + target_score * 0.3 + overlap_score * 0.2, 1.0),
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in _STOPWORDS
    }


def _token_overlap(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _extract_arguments(action_id: str, subtask: SubtaskV1) -> dict[str, Any]:
    text = subtask.source_text.strip()
    tail = str(subtask.raw_arguments.get("tail") or "").strip()
    normalized_tail = normalize_text(tail)
    if action_id == "open_url":
        explicit = re.search(r"https?://[^\s<>'\"]+", text, flags=re.IGNORECASE)
        if explicit:
            return {"url": explicit.group(0).rstrip(".,;:!?)]}")}
        bare_domain = re.search(
            r"(?<![\w.-])([a-z0-9-]+\s*\.\s*(?:pl|com|org|net|io|ai))\b",
            text,
            flags=re.IGNORECASE,
        )
        if bare_domain:
            domain = re.sub(r"\s+", "", bare_domain.group(1)).lower()
            return {"url": f"https://{domain}"}
        known = _KNOWN_URLS.get(str(subtask.target or ""))
        return {"url": known} if known else {}
    if action_id == "open_folder":
        if subtask.target == "this_pc" or re.search(
            r"\b(?:moj|ten)\s+komputer\b|\b(?:eksplorator\w*|explorer)\b",
            normalize_text(text),
        ):
            return {"folder_id": "this_pc"}
        return {}
    if action_id == "open_app":
        if subtask.target == "WhatsApp" or re.search(
            r"\b(?:whatsapp|whats\s*app|whatsap)\b",
            normalize_text(text),
        ):
            return {"app_id": "whatsapp"}
        return {}
    if action_id == "search_web":
        query = re.sub(
            r"^(?:w\s+(?:internecie|necie|sieci)|online)\s*[:,-]?\s*",
            "",
            tail,
            flags=re.IGNORECASE,
        ).strip()
        return {"query": query, "limit": 5} if query else {}
    if action_id == "paste_text_safe":
        match = re.match(
            r"^\s*(?:wpisz|napisz|wklej|wstaw)(?:\s+to)?"
            r"(?:\s+do\s+(chatgpt|gpt|gemini|cursora|cursor))?"
            r"\s*[:,-]?\s+(?:tekst|tre(?:s|ś)ć)\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return {}
        args: dict[str, Any] = {"text": match.group(2).strip()}
        target = normalize_text(match.group(1) or "")
        expected_window = {
            "chatgpt": "chatgpt",
            "gpt": "chatgpt",
            "gemini": "gemini",
            "cursor": "cursor",
            "cursora": "cursor",
        }.get(target)
        if expected_window:
            args["expected_window"] = expected_window
        return args
    if action_id == "create_note":
        content = re.sub(
            r"^(?:(?:mi\s+)?notatk\w*|(?:w|do)\s+(?:notatnik\w*|notes\w*))"
            r"\s*[:,-]?\s*",
            "",
            tail,
            flags=re.IGNORECASE,
        ).strip()
        content = re.sub(
            r"^(?:o\s+tre(?:s|ś)ci|że|ze|list\w*)\s*[:,-]?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        return {"text": content} if content else {}
    if action_id == "run_uivision_macro":
        match = re.search(
            r"(?<![A-Za-z0-9_.-])([A-Za-z0-9][A-Za-z0-9_.-]*\.json)\b",
            text,
            flags=re.IGNORECASE,
        )
        return {"macro": match.group(1)} if match else {}
    if action_id == "remember":
        content = re.sub(
            r"^(?:(?:w|do)\s+pami(?:e|ę)ci\s+)?"
            r"(?:informacj\w*|to|że|ze)?\s*[:,-]?\s*",
            "",
            tail,
            flags=re.IGNORECASE,
        ).strip()
        return {"content": content, "kind": "fact"} if content else {}
    if action_id == "recall":
        query = re.sub(
            r"^(?:(?:z|w)\s+)?(?:pami(?:e|ę)ci|zapisk\w*|sobie)"
            r"\s*[:,-]?\s*",
            "",
            tail,
            flags=re.IGNORECASE,
        ).strip()
        return {"query": query} if query else {}
    if action_id == "describe_recent_activity":
        minutes = _extract_minutes(text)
        return {"minutes": minutes}
    if action_id == "rename_under_cursor":
        explicit_name = re.search(
            r"\bna\s+(.+)$",
            tail,
            flags=re.IGNORECASE,
        )
        if explicit_name:
            name = explicit_name.group(1)
        else:
            name = re.sub(
                r"^(?:(?:now\w+\s+)?nazw\w+\s+)?"
                r"(?:(?:ikon|plik|element)\w*\s+)?"
                r"(?:(?:pod\s+kursor\w*|wskazan\w*\s+(?:mysz|kursor)\w*)\s+)?",
                "",
                tail,
                flags=re.IGNORECASE,
            )
        name = name.strip(" .\"'")
        return {"new_name": name} if name else {}
    if action_id == "remember_last_source":
        return {"index": _extract_source_index(normalized_tail), "kind": "web_source"}
    return {}


def _extract_minutes(text: str) -> int:
    normalized = normalize_text(text)
    word_minutes = {
        "czterdziestu pieciu minut": 45,
        "trzydziestu minut": 30,
        "dwudziestu minut": 20,
        "pietnastu minut": 15,
        "dziesieciu minut": 10,
        "pieciu minut": 5,
    }
    for phrase, minutes in word_minutes.items():
        if phrase in normalized:
            return minutes
    minute_match = re.search(r"\b(\d{1,5})\s*(?:minut|minuty|min)\b", normalized)
    hour_match = re.search(r"\b(\d{1,3})\s*(?:godzin|godziny|godzine|h)\b", normalized)
    if minute_match:
        return max(1, min(int(minute_match.group(1)), 20160))
    if hour_match:
        return max(1, min(int(hour_match.group(1)) * 60, 20160))
    if "godzin" in normalized or "godzine" in normalized:
        return 60
    return 30


def _describe_action_for(normalized: str, *, target: str | None) -> str | None:
    if any(
        marker in normalized
        for marker in (
            "screenpipe",
            "screen pipe",
            "co robilem",
            "co robilam",
            "przed chwila",
            "ostatnia aktywn",
            "ostatnie aktywn",
            "ostatnio robil",
            "co sie dzialo",
            "ekran z ostatn",
            "uzywane przez ostatn",
            "przez ostatnia godzin",
            "przez ostatnia godzine",
        )
    ):
        return "describe_recent_activity"
    if any(
        marker in normalized
        for marker in (
            "gdzie trafi",
            "gdzie pis",
            "gdzie teraz pis",
            "wpisywan",
            "pole do pis",
            "czy moge tu pis",
            "pasek adresu",
            "mozna bezpiecznie pis",
            "kursor wskazuje pole",
            "pole adresu",
        )
    ):
        return "describe_text_target"
    if target in {"active_window", "window"} or any(
        marker in normalized
        for marker in (
            "co mam otwarte",
            "co jest otwarte",
            "aktywne okno",
            "aktywnego okna",
            "biezace okno",
            "jakie okno",
            "tytul okna",
            "w jakim programie",
            "co mam na wierzchu",
            "tytul tego co mam",
        )
    ):
        return "describe_active_window"
    return None


def _extract_source_index(normalized: str) -> int:
    digit = re.search(r"\b(\d{1,2})\b", normalized)
    if digit:
        return max(1, min(int(digit.group(1)), 20))
    for marker, value in _ORDINALS.items():
        if marker in normalized:
            return value
    return 1
