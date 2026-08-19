from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import (
    SegmentationDecisionV1,
    SegmentationResultV1,
    SubtaskV1,
    TextSpanV1,
    normalize_transcript_text,
)
from ..router import normalize_text
from .taxonomy import (
    OPEN_COORDINATION_TARGETS,
    TARGET_PATTERNS,
    operation_for_token,
    operation_prompt,
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_AMBIGUOUS_CONNECTORS = {"albo", "lub", "jesli", "jezeli", "gdy", "chyba", "ale"}
_SPLIT_CONNECTORS = {"i", "oraz", "potem", "nastepnie", "pozniej"}
_PUNCTUATION_CONNECTORS = {",", ";", "."}
_FREEFORM_ARGUMENT_OPERATIONS = {
    "create_note",
    "describe",
    "paste",
    "recall",
    "remember",
    "rename",
    "search",
}
_UNSUPPORTED_OPERATIONS = {"do", "send"}
_VOCATIVE_PREFIXES = {
    "asystent",
    "asystencie",
    "assistant",
    "venice",
    "venive",
    "wenice",
    "voiceloop",
}
_FILLER_PREFIXES = {
    "mi",
    "prosze",
    "teraz",
    "szybko",
    "po",
    "prostu",
}

@dataclass(frozen=True, slots=True)
class _Token:
    raw: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Anchor:
    token_index: int
    operation: str


@dataclass(frozen=True, slots=True)
class _RawSegment:
    start: int
    end: int
    operation: str
    confidence: float
    inherited_operation: bool = False


def segment_command(text: str, *, max_subtasks: int = 12) -> SegmentationResultV1:
    raw = text or ""
    if not raw.strip():
        return SegmentationResultV1(
            decision=SegmentationDecisionV1.NON_COMMAND,
            confidence=1.0,
            reason="empty_text",
        )

    tokens = _tokenize(raw)
    ambiguous = _ambiguous_connectors(tokens)
    if ambiguous:
        span = _content_span(raw, 0, len(raw))
        return SegmentationResultV1(
            decision=SegmentationDecisionV1.AMBIGUOUS,
            unrecognized_spans=(span,) if span else (),
            confidence=0.0,
            reason=f"ambiguous_connector:{sorted(ambiguous)[0]}",
        )

    anchors = _operation_anchors(tokens)
    anchors = _collapse_freeform_argument_anchors(tokens, anchors)
    atomic_api_check = _is_atomic_api_check(raw, anchors)
    coordinated_segments = (
        None if atomic_api_check else _split_fully_classified_coordination(raw, tokens)
    )
    if coordinated_segments is not None:
        raw_segments = coordinated_segments
    else:
        unclassified = None if atomic_api_check else _unclassified_coordination(raw, tokens)
        if unclassified is not None:
            known_segments, unknown_spans = unclassified
            subtasks, _ = _build_subtasks(raw, known_segments)
            return SegmentationResultV1(
                decision=SegmentationDecisionV1.AMBIGUOUS,
                subtasks=tuple(subtasks),
                unrecognized_spans=tuple(unknown_spans),
                confidence=min(
                    (subtask.segmentation_confidence for subtask in subtasks),
                    default=0.0,
                ),
                reason="unrecognized_command_fragment",
            )
        if atomic_api_check:
            anchors = anchors[:1]

        if not anchors:
            alias = _infer_alias_operation(raw)
            if alias is None:
                return SegmentationResultV1(
                    decision=SegmentationDecisionV1.NON_COMMAND,
                    confidence=0.8,
                    reason="no_command_operation",
                )
            start, end = _trim_span(raw, 0, len(raw))
            segment = _RawSegment(start, end, alias, 0.98)
            subtasks, unrecognized = _build_subtasks(raw, [segment])
            return SegmentationResultV1(
                decision=SegmentationDecisionV1.SIMPLE,
                subtasks=tuple(subtasks),
                unrecognized_spans=tuple(unrecognized),
                confidence=0.98,
            )

        raw_segments, split_error = _split_at_anchors(raw, tokens, anchors)
        if split_error is not None:
            span = _content_span(raw, 0, len(raw))
            return SegmentationResultV1(
                decision=SegmentationDecisionV1.AMBIGUOUS,
                unrecognized_spans=(span,) if span else (),
                confidence=0.0,
                reason=split_error,
            )

    expanded: list[_RawSegment] = []
    for segment in raw_segments:
        expanded.extend(_expand_open_target_coordination(raw, segment, tokens))

    if len(expanded) > max(1, max_subtasks):
        span = _content_span(raw, 0, len(raw))
        return SegmentationResultV1(
            decision=SegmentationDecisionV1.AMBIGUOUS,
            unrecognized_spans=(span,) if span else (),
            confidence=0.0,
            reason="too_many_subtasks",
        )

    subtasks, unrecognized = _build_subtasks(raw, expanded)
    if unrecognized or any(subtask.has_unrecognized_text for subtask in subtasks):
        return SegmentationResultV1(
            decision=SegmentationDecisionV1.AMBIGUOUS,
            subtasks=tuple(subtasks),
            unrecognized_spans=tuple(unrecognized),
            confidence=min((item.segmentation_confidence for item in subtasks), default=0.0),
            reason="unrecognized_command_fragment",
        )

    decision = (
        SegmentationDecisionV1.COMPOUND if len(subtasks) > 1 else SegmentationDecisionV1.SIMPLE
    )
    confidence = min((item.segmentation_confidence for item in subtasks), default=0.0)
    confidence = max(0.0, confidence - max(0, len(subtasks) - 1) * 0.02)
    return SegmentationResultV1(
        decision=decision,
        subtasks=tuple(subtasks),
        confidence=confidence,
    )


def has_stop_subtask(result: SegmentationResultV1) -> bool:
    return any(subtask.operation == "stop" for subtask in result.subtasks)


def _tokenize(text: str) -> list[_Token]:
    return [
        _Token(
            raw=match.group(0),
            normalized=normalize_text(match.group(0)),
            start=match.start(),
            end=match.end(),
        )
        for match in _TOKEN_PATTERN.finditer(text)
    ]


def _operation_anchors(tokens: list[_Token]) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    for index, token in enumerate(tokens):
        normalized = token.normalized
        if not normalized:
            continue
        if (
            normalized == "wez"
            and index + 1 < len(tokens)
            and tokens[index + 1].normalized == "sobie"
        ):
            continue
        operation = _operation_for_token(normalized)
        if operation is None and normalized == "zmien":
            next_token = tokens[index + 1].normalized if index + 1 < len(tokens) else ""
            if next_token.startswith("nazw"):
                operation = "rename"
        if operation is None and normalized == "daj":
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 5])
            if "nazw" in tail:
                operation = "rename"
        if operation is None and normalized == "wrzuc":
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 5])
            if any(marker in tail for marker in ("notat", "notes")):
                operation = "create_note"
        if operation is None and normalized == "wyciagnij":
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 6])
            if any(marker in tail for marker in ("pamie", "zapisk")):
                operation = "recall"
        if operation is None:
            continue
        if normalized == "pokaz":
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 4])
            if "pulpit" in tail or "desktop" in tail:
                operation = "minimize"
            elif any(marker in tail for marker in ("kalendarz", "terminarz", "harmonogram")):
                operation = "open"
            else:
                operation = "describe"
        if normalized == "uruchom":
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 6])
            if any(marker in tail for marker in ("makr", "automatyzac", "ui vision")):
                operation = "run"
        if normalized in {"wyszukaj", "poszukaj", "przeszukaj", "sprawdz"}:
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 6])
            if "pamie" in tail or "zapisk" in tail:
                operation = "recall"
        if normalized == "sprawdz":
            tail = " ".join(item.normalized for item in tokens[index + 1 : index + 14])
            if any(
                marker in tail
                for marker in (
                    "screenpipe",
                    "screen pipe",
                    "co robilem",
                    "co robilam",
                    "przed chwila",
                    "ostatnia aktywn",
                    "ostatnie aktywn",
                    "gdzie trafi",
                    "gdzie pis",
                    "wpisywan",
                    "pole do pis",
                )
            ) or (
                "czy" in tail
                and any(marker in tail for marker in ("pis", "pole", "tekst", "adres"))
            ):
                operation = "describe"
        anchors.append(_Anchor(index, operation))
    return anchors


def _operation_for_token(token: str) -> str | None:
    return operation_for_token(token)


def _collapse_freeform_argument_anchors(
    tokens: list[_Token],
    anchors: list[_Anchor],
) -> list[_Anchor]:
    if len(anchors) <= 1 or anchors[0].operation not in _FREEFORM_ARGUMENT_OPERATIONS:
        return anchors
    retained = [anchors[0]]
    retained.extend(
        anchor
        for anchor in anchors[1:]
        if _separator_before(tokens, anchor.token_index) is not None
    )
    return retained


def _split_fully_classified_coordination(
    text: str,
    tokens: list[_Token],
) -> list[_RawSegment] | None:
    piece_bounds = _coordination_piece_bounds(text, tokens)
    if not piece_bounds:
        return None

    segments: list[_RawSegment] = []
    anchored: list[bool] = []
    for start, end in piece_bounds:
        local_tokens = [token for token in tokens if token.start >= start and token.end <= end]
        anchors = _operation_anchors(local_tokens)
        if len(anchors) > 1:
            return None
        operation = anchors[0].operation if anchors else _infer_alias_operation(text[start:end])
        if operation is None:
            return None
        anchored.append(bool(anchors))
        segments.append(
            _RawSegment(
                start=start,
                end=end,
                operation=operation,
                confidence=0.90,
                inherited_operation=not anchors,
            )
        )
    if segments[0].operation in _FREEFORM_ARGUMENT_OPERATIONS and any(
        not has_anchor for has_anchor in anchored[1:]
    ):
        return None
    return segments


def _unclassified_coordination(
    text: str,
    tokens: list[_Token],
) -> tuple[list[_RawSegment], list[TextSpanV1]] | None:
    piece_bounds = _coordination_piece_bounds(text, tokens)
    if not piece_bounds:
        return None

    classified: list[_RawSegment | None] = []
    for start, end in piece_bounds:
        local_tokens = [token for token in tokens if token.start >= start and token.end <= end]
        anchors = _operation_anchors(local_tokens)
        if len(anchors) > 1:
            return None
        operation = anchors[0].operation if anchors else _infer_alias_operation(text[start:end])
        classified.append(
            _RawSegment(
                start=start,
                end=end,
                operation=operation,
                confidence=0.70,
                inherited_operation=not anchors,
            )
            if operation is not None
            else None
        )

    known_indices = [index for index, segment in enumerate(classified) if segment is not None]
    if not known_indices or len(known_indices) == len(classified):
        return None
    first_known = classified[known_indices[0]]
    if first_known is None:
        return None
    can_be_freeform_tail = (
        len(known_indices) == 1
        and known_indices[0] == 0
        and first_known.operation in _FREEFORM_ARGUMENT_OPERATIONS
    )
    if can_be_freeform_tail:
        return None

    known_segments = [segment for segment in classified if segment is not None]
    unknown_spans = [
        TextSpanV1(
            start_char=start,
            end_char=end,
            text=text[start:end],
        )
        for (start, end), segment in zip(
            piece_bounds,
            classified,
            strict=True,
        )
        if segment is None
    ]
    return known_segments, unknown_spans


def _coordination_piece_bounds(
    text: str,
    tokens: list[_Token],
) -> list[tuple[int, int]]:
    connectors = [
        token
        for index, token in enumerate(tokens)
        if token.normalized in _SPLIT_CONNECTORS
        or token.raw in {",", ";"}
        or (
            token.raw == "."
            and token.end < len(text)
            and text[token.end].isspace()
            and not _is_spaced_domain_dot(tokens, index)
        )
    ]
    if not connectors:
        return []

    piece_bounds: list[tuple[int, int]] = []
    piece_start = 0
    for connector in connectors:
        start, end = _trim_span(text, piece_start, connector.start)
        if end > start:
            piece_bounds.append((start, end))
        piece_start = connector.end
    start, end = _trim_span(text, piece_start, len(text))
    if end > start:
        piece_bounds.append((start, end))
    return piece_bounds if len(piece_bounds) >= 2 else []


def _split_at_anchors(
    text: str,
    tokens: list[_Token],
    anchors: list[_Anchor],
) -> tuple[list[_RawSegment], str | None]:
    if len(anchors) == 1:
        start, end = _trim_span(text, 0, len(text))
        return [_RawSegment(start, end, anchors[0].operation, 0.98)], None

    boundaries: list[tuple[int, int, float]] = []
    for anchor in anchors[1:]:
        separator = _separator_before(tokens, anchor.token_index)
        if separator is None:
            return [], "missing_separator_before_command"
        separator_start, next_start, confidence = separator
        boundaries.append((separator_start, next_start, confidence))

    segments: list[_RawSegment] = []
    current_start = 0
    confidence = 0.95
    for index, (previous_end, next_start, split_confidence) in enumerate(boundaries):
        start, end = _trim_span(text, current_start, previous_end)
        if end <= start:
            return [], "empty_segment"
        segments.append(
            _RawSegment(start, end, anchors[index].operation, min(confidence, split_confidence))
        )
        current_start = next_start
        confidence = split_confidence
    start, end = _trim_span(text, current_start, len(text))
    if end <= start:
        return [], "empty_segment"
    segments.append(_RawSegment(start, end, anchors[-1].operation, confidence))
    return segments, None


def _separator_before(
    tokens: list[_Token],
    anchor_index: int,
) -> tuple[int, int, float] | None:
    cursor = anchor_index - 1
    if cursor < 0:
        return None
    consumed: list[_Token] = []
    while cursor >= 0:
        token = tokens[cursor]
        if token.normalized in _SPLIT_CONNECTORS or token.raw in _PUNCTUATION_CONNECTORS:
            consumed.append(token)
            cursor -= 1
            continue
        if (
            token.normalized == "a"
            and consumed
            and any(item.normalized in {"potem", "nastepnie", "pozniej"} for item in consumed)
        ):
            consumed.append(token)
            cursor -= 1
        break
    if not consumed:
        return None
    consumed.reverse()
    confidence = 0.95 if any(item.raw in _PUNCTUATION_CONNECTORS for item in consumed) else 0.92
    return consumed[0].start, tokens[anchor_index].start, confidence


def _expand_open_target_coordination(
    text: str,
    segment: _RawSegment,
    tokens: list[_Token],
) -> list[_RawSegment]:
    if segment.operation != "open":
        return [segment]
    local_tokens = [
        token for token in tokens if token.start >= segment.start and token.end <= segment.end
    ]
    local_anchors = _operation_anchors(local_tokens)
    if len(local_anchors) != 1:
        return [segment]
    operation_token = local_tokens[local_anchors[0].token_index]
    candidate_connectors = [
        token
        for token in local_tokens
        if token.start > operation_token.end
        and (token.normalized in {"i", "oraz"} or token.raw in {",", ";"})
    ]
    if not candidate_connectors:
        return [segment]

    piece_bounds: list[tuple[int, int]] = []
    piece_start = segment.start
    for connector in candidate_connectors:
        start, end = _trim_span(text, piece_start, connector.start)
        if end > start:
            piece_bounds.append((start, end))
        piece_start = connector.end
    start, end = _trim_span(text, piece_start, segment.end)
    if end > start:
        piece_bounds.append((start, end))
    if len(piece_bounds) < 2:
        return [segment]

    targets: list[str | None] = []
    for index, (start, end) in enumerate(piece_bounds):
        source = text[start:end]
        if index == 0:
            source = source[operation_token.end - start :]
        targets.append(_detect_target(source))
    if any(target not in OPEN_COORDINATION_TARGETS for target in targets):
        return [segment]

    result: list[_RawSegment] = []
    for index, (start, end) in enumerate(piece_bounds):
        result.append(
            _RawSegment(
                start=start,
                end=end,
                operation="open",
                confidence=min(segment.confidence, 0.87),
                inherited_operation=index > 0,
            )
        )
    return result


def _build_subtasks(
    text: str,
    segments: list[_RawSegment],
) -> tuple[list[SubtaskV1], list[TextSpanV1]]:
    subtasks: list[SubtaskV1] = []
    unrecognized: list[TextSpanV1] = []
    for order, segment in enumerate(segments):
        source_text = text[segment.start : segment.end].strip()
        target = _detect_target(source_text)
        rendered_text = (
            f"{_operation_prompt(segment.operation)} {source_text}".strip()
            if segment.inherited_operation
            else source_text
        )
        raw_arguments = _raw_arguments(rendered_text, segment.operation)
        unknown_operation = segment.operation in _UNSUPPORTED_OPERATIONS
        has_unrecognized = unknown_operation
        subtask = SubtaskV1(
            text=rendered_text,
            source_text=source_text,
            normalized_text=normalize_transcript_text(rendered_text),
            start_char=segment.start,
            end_char=segment.end,
            order=order,
            operation=segment.operation,
            target=target,
            raw_arguments=raw_arguments,
            segmentation_confidence=segment.confidence,
            has_unrecognized_text=has_unrecognized,
        )
        subtasks.append(subtask)
        if has_unrecognized:
            unrecognized.append(
                TextSpanV1(
                    start_char=segment.start,
                    end_char=segment.end,
                    text=source_text,
                )
            )
    return subtasks, unrecognized


def _raw_arguments(text: str, operation: str) -> dict[str, str]:
    tokens = _tokenize(text)
    anchors = _operation_anchors(tokens)
    if not anchors:
        return {"tail": text.strip()} if text.strip() else {}
    anchor = anchors[0]
    verb_end = tokens[anchor.token_index].end
    if operation == "rename" and anchor.token_index + 1 < len(tokens):
        if tokens[anchor.token_index].normalized == "zmien":
            verb_end = tokens[anchor.token_index + 1].end
    tail = text[verb_end:].strip(" \t,:;-")
    return {"tail": tail} if tail else {}


def _detect_target(text: str) -> str | None:
    normalized = normalize_text(text)
    for target, pattern in TARGET_PATTERNS:
        if pattern.search(normalized):
            return target
    return None


def _infer_alias_operation(text: str) -> str | None:
    normalized = normalize_text(text)
    if normalized in {
        "kalendarz",
        "calendar",
        "przegladarka",
        "browser",
        "chrome",
        "chat",
        "czat",
        "chatgpt",
        "gemini",
        "youtube",
        "whatsapp",
        "whatsap",
        "moj komputer",
        "ten komputer",
        "eksplorator",
        "explorer",
    }:
        return "open"
    if normalized in {
        "pulpit",
        "desktop",
        "show desktop",
        "minimize all",
    }:
        return "minimize"
    if normalized in {"co potrafisz", "co umiesz", "help", "list_capabilities"} or (
        any(marker in normalized for marker in ("komend", "akcj", "mozliw"))
        and any(marker in normalized for marker in ("jakie", "pokaz", "wymien"))
    ):
        return "list"
    if normalized in {
        "active_window",
        "active window",
        "aktywne okno",
        "okno aktywne",
        "co jest otwarte",
        "co mam otwarte",
        "historia",
        "aktywnosc",
        "recent_activity",
        "gdzie pisze",
        "gdzie teraz pisze",
    }:
        return "describe"
    if any(
        marker in normalized
        for marker in (
            "co robilem",
            "co robilam",
            "co bylo na ekranie",
            "w jakim programie",
            "podaj tytul",
            "historia ekranu",
            "ostatnia aktywnosc",
            "ostatnie aktywnosci",
            "co ostatnio robilem",
            "co ostatnio robilam",
            "co sie dzialo",
            "ostatnie okna",
            "czy to pasek adresu",
            "czy to dobre pole do pisania",
            "gdzie trafi tekst",
            "w jakie pole pisze",
            "czy moge tu pisac",
        )
    ):
        return "describe"
    if normalized in {
        "copy_selected_text",
        "kopiuj zaznaczenie",
        "skopiuj zaznaczenie",
        "zaznaczenie do schowka",
        "copy_email_under_cursor",
        "email do schowka",
        "copy_number_under_cursor",
        "numer do schowka",
        "copy_sentence_under_cursor",
        "zdanie do schowka",
        "copy_text_under_cursor",
        "tekst pod kursorem do schowka",
    }:
        return "copy"
    if re.search(r"\b(?:pogoda|temperatura|kurs|notowania|gielda)\b", normalized):
        return "search"
    if re.search(
        r"\b(?:co|cos)\s+(?:sie\s+)?(?:waznego\s+)?"
        r"(?:zdarzylo|wydarzylo|dzieje)\b",
        normalized,
    ) and any(
        marker in normalized
        for marker in ("na swiecie", "ostatnich godzin", "dzisiaj", "teraz")
    ):
        return "search"
    return None


def _ambiguous_connectors(tokens: list[_Token]) -> set[str]:
    ambiguous: set[str] = set()
    for index, token in enumerate(tokens):
        if token.normalized not in _AMBIGUOUS_CONNECTORS:
            continue
        if (
            token.normalized in {"jesli", "jezeli"}
            and index + 1 < len(tokens)
            and tokens[index + 1].normalized == "chodzi"
        ):
            continue
        ambiguous.add(token.normalized)
    return ambiguous


def _is_spaced_domain_dot(tokens: list[_Token], index: int) -> bool:
    if index <= 0 or index + 1 >= len(tokens):
        return False
    previous = tokens[index - 1].normalized
    following = tokens[index + 1].normalized
    return bool(
        previous
        and re.fullmatch(r"[a-z0-9-]+", previous)
        and following in {"pl", "com", "org", "net", "io", "ai"}
    )


def _operation_prompt(operation: str) -> str:
    return operation_prompt(operation)


def _is_atomic_api_check(text: str, anchors: list[_Anchor]) -> bool:
    normalized = normalize_text(text)
    return (
        len(anchors) > 1
        and "api" in normalized
        and re.search(r"\bendp\w*\b", normalized) is not None
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in ",;."):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",;."):
        end -= 1
    return start, end


def _content_span(text: str, start: int, end: int) -> TextSpanV1 | None:
    start, end = _trim_span(text, start, end)
    if end <= start:
        return None
    return TextSpanV1(start_char=start, end_char=end, text=text[start:end])
