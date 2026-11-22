"""Incremental graph of one recorded conversation.

A new utterance may add a node, enrich an existing one, link two nodes, or
revise a value. The store appends those operations and folds them into a view.
It never replaces the graph. Positions are chosen once, when a node is added,
and later operations do not move it. Speaker tiles come only from a diarization
id supplied with the utterance.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NodeKind = Literal["speaker", "concept", "intention"]
EdgeKind = Literal["says", "intends", "relates", "tone"]
OpName = Literal["add", "enrich", "link", "revise"]

_MEETING_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SPEAKER_ID = re.compile(r"^[\w.:-]{1,40}$", re.UNICODE)
_RENAME = re.compile(
    r"to nie jest\s+([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+).{0,24}?"
    r"to jest\s+([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"nie jest\s+(.+)",
    re.IGNORECASE,
)
_MODAL_STATE = re.compile(
    r"powinna być\s+([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)",
    re.IGNORECASE,
)
_FOLD = str.maketrans("ąćęłńóśźż", "acelnoszz")
_STOP = frozenset(
    """
    a aby ale albo bez bo byc byl byla bylo chce chcialam chcialbym chcialem
    gdyby jesli jezeli zeby
    chcielismy chcemy czyli czy dla do dobre dobry dzien dziala dzialac
    dzieki dziekuje hej i jak jednak jeszcze jest jestem jestes juz gdy
    ktora ktore ktory ma macie maja mam masz mamy miec mi mnie moj moja
    moje musi musza na nie oraz po powinien powinna powinno powinni
    pozdrawiam prosze sa sie tak takze te ten tego tej temu to tylko trzeba
    tu tutaj tym bardzo tez w was wam we witam wlasciwie z za ze
    mozna mozliwa mozliwe mozliwy naszym naszych nasza nasze
    kolejny kolejna kolejnym bezmyslny bezmyslnym
    zbudowac zamierzam planuje
    """.split()
)
_MODAL = frozenset(
    """
    chce chcialbym chcialem chcialam chcielismy chcemy zamierzam planuje
    trzeba musi musza powinien powinna powinno powinni
    """.split()
)
_CONTRAST = frozenset("ale jednak natomiast wlasciwie".split())


class GraphRejected(Exception):
    """The store refused an operation because it would replace history."""


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    before: str
    after: str
    utterance_id: str


class NodeView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    label: str
    speaker_id: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    history: list[HistoryEntry] = Field(default_factory=list)
    position: Position
    utterance_id: str


class EdgeView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    kind: EdgeKind
    label: str
    emotion: str | None = None
    speaker_id: str | None = None
    utterance_id: str


class GraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    nodes: list[NodeView]
    edges: list[EdgeView]
    focus_node_id: str | None = None
    event_count: int


class GraphOp(BaseModel):
    """One allowed change. A full graph is not an operation."""

    model_config = ConfigDict(extra="forbid")

    op: OpName
    op_id: str = Field(min_length=1, max_length=80)
    node_id: str | None = None
    edge_id: str | None = None
    kind: str | None = None
    label: str | None = None
    speaker_id: str | None = None
    source_id: str | None = None
    target_id: str | None = None
    attributes: dict[str, str] | None = None
    alias: str | None = None
    emotion: str | None = None
    append: bool = False


class Utterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    utterance_id: str
    speaker_id: str
    text: str
    emotion: str | None = None
    speaker_label: str | None = None


class UtteranceRequest(BaseModel):
    """One diarized line. The body is not a graph."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=2000)
    emotion: str | None = Field(default=None, max_length=40)
    speaker_label: str | None = Field(default=None, max_length=80)
    utterance_id: str | None = Field(default=None, max_length=80)


class OperationRequest(BaseModel):
    """External proposal. Only the four operations, never a replacement graph."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str = Field(min_length=1, max_length=40)
    utterance_id: str = Field(min_length=1, max_length=80)
    operations: list[GraphOp] = Field(min_length=1, max_length=24)


class AppliedUtterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance_id: str
    replayed: bool
    operations: list[GraphOp]
    graph: GraphSnapshot


@dataclass
class _Node:
    id: str
    kind: NodeKind
    label: str
    speaker_id: str | None
    attributes: dict[str, str]
    aliases: list[str]
    history: list[HistoryEntry]
    x: float
    y: float
    utterance_id: str


@dataclass
class _Edge:
    id: str
    source: str
    target: str
    kind: EdgeKind
    label: str
    emotion: str | None
    speaker_id: str | None
    utterance_id: str


@dataclass
class _State:
    nodes: dict[str, _Node] = field(default_factory=dict)
    edges: dict[str, _Edge] = field(default_factory=dict)
    focus_node_id: str | None = None
    speaker_count: int = 0
    content_count: int = 0


@dataclass(frozen=True)
class _StoredEvent:
    seq: int
    utterance_id: str
    speaker_id: str
    op: GraphOp
    created_at: str


def fold(text: str) -> str:
    lowered = text.casefold().translate(_FOLD)
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def stem(folded: str) -> str:
    for ending in (
        "acjami",
        "acjach",
        "acja",
        "acje",
        "cji",
        "cja",
        "cje",
        "owie",
        "iami",
        "owie",
        "enie",
        "aniu",
        "ami",
        "ach",
        "owi",
        "iem",
        "ego",
        "emu",
        "owac",
    ):
        if folded.endswith(ending) and len(folded) - len(ending) >= 4:
            return folded[: -len(ending)]
    if len(folded) > 5 and folded.endswith(("a", "e", "i", "u", "y", "o")):
        return folded[:-1]
    return folded


def display_label(token: str) -> str:
    raw = token.strip(".,;:!?\"'")
    folded = fold(raw)
    if folded.endswith("cje"):
        lemma = folded[:-3] + "cja"
    elif folded.endswith("acje"):
        lemma = folded[:-4] + "acja"
    else:
        return raw[:1].upper() + raw[1:] if raw else raw
    return lemma[:1].upper() + lemma[1:] if lemma else raw


def edge_identity(source: str, target: str, kind: str, label: str) -> str:
    slug = fold(label)[:48]
    raw = f"{source}|{target}|{kind}|{slug}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"edge:{digest}"


def speaker_node_id(meeting_id: str, speaker_id: str) -> str:
    return f"spk:{meeting_id}:{speaker_id}"


def _check_meeting_id(meeting_id: str) -> None:
    if not _MEETING_ID.fullmatch(meeting_id):
        raise GraphRejected("Identyfikator nagrania może zawierać litery, cyfry, kropkę i myślnik.")


def _check_speaker_id(speaker_id: str) -> None:
    if not speaker_id or not _SPEAKER_ID.fullmatch(speaker_id):
        raise GraphRejected("Brak identyfikatora mówcy z diaryzacji.")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9-]+", text)


def _content_tokens(text: str, *, drop: Iterable[str] = ()) -> list[str]:
    dropped = {fold(item) for item in drop}
    kept: list[str] = []
    for token in _tokens(text):
        folded = fold(token)
        if len(folded) < 3 or folded in _STOP or folded in dropped:
            continue
        kept.append(token)
    return kept


def _node_stems(node: _Node) -> set[str]:
    stems = {stem(fold(node.label))}
    stems.update(stem(fold(alias)) for alias in node.aliases)
    stems.discard("")
    return stems


def _best_content_match(state: _State, tokens: Iterable[str]) -> _Node | None:
    wanted = [stem(fold(token)) for token in tokens]
    wanted = [item for item in wanted if item]
    best: _Node | None = None
    best_score = 0
    for node in state.nodes.values():
        if node.kind == "speaker":
            continue
        score = sum(1 for item in wanted if item in _node_stems(node))
        if score > best_score:
            best = node
            best_score = score
    return best


def _find_by_word(state: _State, word: str) -> _Node | None:
    wanted = stem(fold(word))
    if not wanted:
        return None
    for node in state.nodes.values():
        if node.kind != "speaker" and wanted in _node_stems(node):
            return node
    return None


def _focus_concept(state: _State) -> _Node | None:
    focus = state.nodes.get(state.focus_node_id or "")
    if focus is None or focus.kind == "speaker":
        return None
    if focus.kind == "concept":
        return focus
    for node in reversed(list(state.nodes.values())):
        if node.kind == "concept":
            return node
    return focus


def _has_edge(state: _State, source: str, target: str, kind: str, label: str) -> bool:
    return edge_identity(source, target, kind, label) in state.edges


def _queued_edge(ops: list[GraphOp], source: str, target: str, kind: str, label: str) -> bool:
    identity = edge_identity(source, target, kind, label)
    return any(op.op == "link" and op.edge_id == identity for op in ops)


def propose_operations(snapshot: GraphSnapshot, utterance: Utterance) -> list[GraphOp]:
    """Return only add, enrich, link, or revise for this one utterance."""

    _check_meeting_id(utterance.meeting_id)
    _check_speaker_id(utterance.speaker_id)
    text = utterance.text.strip()
    if not text:
        raise GraphRejected("Pusta wypowiedź nie zmienia grafu.")
    state = _state_from_snapshot(snapshot)
    return _propose(state, utterance, text)


def _propose(state: _State, utterance: Utterance, text: str) -> list[GraphOp]:
    ops: list[GraphOp] = []
    speaker_id = speaker_node_id(utterance.meeting_id, utterance.speaker_id)
    if speaker_id not in state.nodes:
        label = (utterance.speaker_label or f"Mówca {utterance.speaker_id}").strip()
        ops.append(
            GraphOp(
                op="add",
                op_id=_op_id(utterance, len(ops)),
                node_id=speaker_id,
                kind="speaker",
                label=label[:80],
                speaker_id=utterance.speaker_id,
            )
        )

    rename = _RENAME.search(text)
    if rename:
        source = _find_by_word(state, rename.group(1))
        if source is not None:
            new_label = display_label(rename.group(2))
            if new_label and new_label != source.label:
                ops.append(
                    GraphOp(
                        op="revise",
                        op_id=_op_id(utterance, len(ops)),
                        node_id=source.id,
                        label=new_label,
                        alias=source.label,
                    )
                )
            _link_speaker(ops, state, utterance, text, speaker_id, source.id, "says")
            _emotion_edge(ops, state, utterance, speaker_id, source.id)
            return ops

    folded = fold(text)
    connectivity = _connectivity_value(folded)
    if connectivity is not None:
        target = _best_content_match(state, _content_tokens(text)) or _focus_concept(state)
        if target is None:
            target_id = _add_content(
                ops,
                state,
                utterance,
                label="Łączność",
                kind="concept",
                attributes={"lacznosc": connectivity},
            )
        else:
            target_id = target.id
            _set_attribute(ops, utterance, target, "lacznosc", connectivity)
        _link_speaker(ops, state, utterance, text, speaker_id, target_id, "says")
        _emotion_edge(ops, state, utterance, speaker_id, target_id)
        return ops

    negation = _NEGATION.search(text)
    negation_value = ""
    dropped: list[str] = []
    if negation and not text.casefold().startswith("to nie jest"):
        negation_value = negation.group(1).strip(" .,;:")[:160]
        dropped = _tokens(negation_value)

    modal_state = _MODAL_STATE.search(text)
    content = _content_tokens(text, drop=dropped)
    matched = _best_content_match(state, content)
    matched_stems = _node_stems(matched) if matched is not None else set()
    unmatched = [
        token for token in content if stem(fold(token)) not in matched_stems
    ]
    if matched is not None and not _significant(unmatched):
        if negation_value:
            _set_attribute(ops, utterance, matched, "nie_jest", negation_value)
        elif text and "notatka" not in matched.attributes:
            _set_attribute(ops, utterance, matched, "notatka", text[:160])
        elif text:
            _append_note(ops, utterance, matched, text[:160])
        _link_speaker(ops, state, utterance, text, speaker_id, matched.id, "says")
        _emotion_edge(ops, state, utterance, speaker_id, matched.id)
        return ops

    head = _head_token(unmatched or content)
    if head is None:
        _emotion_edge(ops, state, utterance, speaker_id, state.focus_node_id)
        return ops

    existing = _find_by_word(state, head)
    if existing is not None:
        if negation_value:
            _set_attribute(ops, utterance, existing, "nie_jest", negation_value)
        _link_speaker(
            ops,
            state,
            utterance,
            text,
            speaker_id,
            existing.id,
            "intends" if _is_modal(folded) else "says",
        )
        _emotion_edge(ops, state, utterance, speaker_id, existing.id)
        return ops

    kind: NodeKind = "concept"
    attributes: dict[str, str] = {}
    focus = _focus_concept(state)
    if _is_modal(folded) and _is_contrast(folded) and focus is not None:
        kind = "intention"
    if modal_state:
        attributes["stan"] = modal_state.group(1)
    if negation_value:
        attributes["nie_jest"] = negation_value
    node_id = _add_content(
        ops,
        state,
        utterance,
        label=display_label(head),
        kind=kind,
        attributes=attributes,
        alias=head,
    )
    if kind == "intention" and focus is not None:
        _link_speaker(ops, state, utterance, text, speaker_id, node_id, "says")
        _add_link(
            ops,
            state,
            utterance,
            source=node_id,
            target=focus.id,
            kind="relates",
            label=text[:180],
        )
    else:
        _link_speaker(
            ops,
            state,
            utterance,
            text,
            speaker_id,
            node_id,
            "intends" if _is_modal(folded) else "says",
        )
    _emotion_edge(ops, state, utterance, speaker_id, node_id)
    return ops


def _op_id(utterance: Utterance, index: int) -> str:
    return f"{utterance.utterance_id}:{index}"


def _connectivity_value(folded: str) -> str | None:
    if "bez internetu" in folded or "bez sieci" in folded or re.search(r"\boffline\b", folded):
        return "bez internetu"
    if "z internetem" in folded or "z siecia" in folded or re.search(r"\bonline\b", folded):
        return "z internetem"
    return None


def _is_modal(folded: str) -> bool:
    return any(token in _MODAL for token in folded.split())


def _is_contrast(folded: str) -> bool:
    return any(token in _CONTRAST for token in folded.split())


def _significant(tokens: Iterable[str]) -> bool:
    return any(len(stem(fold(token))) >= 4 for token in tokens)


def _is_verbish(token: str) -> bool:
    folded = fold(token)
    return folded.endswith(("owac", "ic", "yc", "ac", "lo", "lem", "lam"))


def _head_token(tokens: list[str]) -> str | None:
    nouns = [token for token in tokens if not _is_verbish(token)]
    pool = nouns or tokens
    for token in reversed(pool):
        if len(stem(fold(token))) >= 4:
            return token
    return pool[-1] if pool else None


def _add_content(
    ops: list[GraphOp],
    state: _State,
    utterance: Utterance,
    *,
    label: str,
    kind: NodeKind,
    attributes: Mapping[str, str] | None = None,
    alias: str | None = None,
) -> str:
    node_id = f"node:{uuid.uuid4()}"
    ops.append(
        GraphOp(
            op="add",
            op_id=_op_id(utterance, len(ops)),
            node_id=node_id,
            kind=kind,
            label=label[:80],
            attributes=dict(attributes or {}),
            alias=alias,
        )
    )
    state.nodes[node_id] = _Node(
        id=node_id,
        kind=kind,
        label=label[:80],
        speaker_id=None,
        attributes=dict(attributes or {}),
        aliases=[alias] if alias else [],
        history=[],
        x=0,
        y=0,
        utterance_id=utterance.utterance_id,
    )
    return node_id


def _set_attribute(
    ops: list[GraphOp],
    utterance: Utterance,
    node: _Node,
    key: str,
    value: str,
) -> None:
    current = node.attributes.get(key)
    if current == value:
        return
    if current is None:
        ops.append(
            GraphOp(
                op="enrich",
                op_id=_op_id(utterance, len(ops)),
                node_id=node.id,
                attributes={key: value},
            )
        )
        node.attributes[key] = value
        return
    ops.append(
        GraphOp(
            op="revise",
            op_id=_op_id(utterance, len(ops)),
            node_id=node.id,
            attributes={key: value},
        )
    )
    node.attributes[key] = value


def _append_note(ops: list[GraphOp], utterance: Utterance, node: _Node, text: str) -> None:
    current = node.attributes.get("notatka", "")
    if text in current:
        return
    ops.append(
        GraphOp(
            op="enrich",
            op_id=_op_id(utterance, len(ops)),
            node_id=node.id,
            attributes={"notatka": text},
            append=True,
        )
    )
    node.attributes["notatka"] = f"{current} · {text}" if current else text


def _speaker_already_linked(
    state: _State,
    ops: list[GraphOp],
    speaker_node: str,
    target: str,
) -> bool:
    if any(edge.source == speaker_node and edge.target == target for edge in state.edges.values()):
        return True
    return any(
        op.op == "link" and op.source_id == speaker_node and op.target_id == target for op in ops
    )


def _link_speaker(
    ops: list[GraphOp],
    state: _State,
    utterance: Utterance,
    text: str,
    speaker_node: str,
    target: str,
    kind: EdgeKind,
) -> None:
    if _speaker_already_linked(state, ops, speaker_node, target):
        return
    _add_link(
        ops,
        state,
        utterance,
        source=speaker_node,
        target=target,
        kind=kind,
        label=text[:180],
    )


def _emotion_edge(
    ops: list[GraphOp],
    state: _State,
    utterance: Utterance,
    speaker_node: str,
    target: str | None,
) -> None:
    known = bool(target) and (target in state.nodes or _will_add(ops, target))
    if not utterance.emotion or not known:
        return
    label = utterance.emotion
    if _has_edge(state, speaker_node, target, "tone", label) or _queued_edge(
        ops, speaker_node, target, "tone", label
    ):
        return
    _add_link(
        ops,
        state,
        utterance,
        source=speaker_node,
        target=target,
        kind="tone",
        label=label,
        emotion=utterance.emotion,
    )


def _will_add(ops: list[GraphOp], node_id: str) -> bool:
    return any(op.op == "add" and op.node_id == node_id for op in ops)


def _add_link(
    ops: list[GraphOp],
    state: _State,
    utterance: Utterance,
    *,
    source: str,
    target: str,
    kind: EdgeKind,
    label: str,
    emotion: str | None = None,
) -> None:
    identity = edge_identity(source, target, kind, label)
    ops.append(
        GraphOp(
            op="link",
            op_id=_op_id(utterance, len(ops)),
            edge_id=identity,
            source_id=source,
            target_id=target,
            kind=kind,
            label=label,
            emotion=emotion,
            speaker_id=utterance.speaker_id,
        )
    )
    state.edges[identity] = _Edge(
        id=identity,
        source=source,
        target=target,
        kind=kind,
        label=label,
        emotion=emotion,
        speaker_id=utterance.speaker_id,
        utterance_id=utterance.utterance_id,
    )


def _state_from_snapshot(snapshot: GraphSnapshot) -> _State:
    state = _State(focus_node_id=snapshot.focus_node_id)
    for node in snapshot.nodes:
        state.nodes[node.id] = _Node(
            id=node.id,
            kind=node.kind,
            label=node.label,
            speaker_id=node.speaker_id,
            attributes=dict(node.attributes),
            aliases=list(node.aliases),
            history=list(node.history),
            x=node.position.x,
            y=node.position.y,
            utterance_id=node.utterance_id,
        )
        if node.kind == "speaker":
            state.speaker_count += 1
        else:
            state.content_count += 1
    for edge in snapshot.edges:
        state.edges[edge.id] = _Edge(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            kind=edge.kind,
            label=edge.label,
            emotion=edge.emotion,
            speaker_id=edge.speaker_id,
            utterance_id=edge.utterance_id,
        )
    return state


def reduce_events(events: Iterable[_StoredEvent]) -> _State:
    state = _State()
    for event in events:
        _reduce(state, event)
    return state


def _reduce(state: _State, event: _StoredEvent) -> None:
    op = event.op
    if op.op == "add":
        _reduce_add(state, event)
    elif op.op == "enrich":
        _reduce_enrich(state, event)
    elif op.op == "link":
        _reduce_link(state, event)
    elif op.op == "revise":
        _reduce_revise(state, event)
    else:
        raise GraphRejected("Graf przyjmuje tylko add, enrich, link albo revise.")


def _reduce_add(state: _State, event: _StoredEvent) -> None:
    op = event.op
    if op.kind not in {"speaker", "concept", "intention"}:
        raise GraphRejected("Nowy węzeł musi być mówcą, pojęciem albo intencją.")
    if not op.node_id or not op.label or not op.label.strip():
        raise GraphRejected("Nowy węzeł potrzebuje identyfikatora i nazwy.")
    if op.node_id in state.nodes:
        raise GraphRejected("Ta operacja podmieniłaby istniejący węzeł.")
    kind: NodeKind = op.kind  # type: ignore[assignment]
    if kind == "speaker":
        x, y = 36.0, 28.0 + state.speaker_count * 124.0
        state.speaker_count += 1
    else:
        column = state.content_count % 3
        row = state.content_count // 3
        x, y = 280.0 + column * 260.0, 28.0 + row * 132.0
        state.content_count += 1
    attributes = dict(op.attributes or {})
    aliases = [op.alias] if op.alias else []
    state.nodes[op.node_id] = _Node(
        id=op.node_id,
        kind=kind,
        label=op.label.strip()[:80],
        speaker_id=op.speaker_id,
        attributes=attributes,
        aliases=aliases,
        history=[],
        x=x,
        y=y,
        utterance_id=event.utterance_id,
    )
    if kind == "concept":
        state.focus_node_id = op.node_id
    elif kind == "intention" and state.focus_node_id is None:
        state.focus_node_id = op.node_id


def _reduce_enrich(state: _State, event: _StoredEvent) -> None:
    node = _require_node(state, event.op.node_id)
    wrote = False
    for key, value in (event.op.attributes or {}).items():
        _check_attribute(key, value)
        current = node.attributes.get(key)
        if current == value:
            continue
        if current is not None and not event.op.append:
            raise GraphRejected("Uzupełnienie nie może nadpisać atrybutu. Użyj korekty.")
        if current and event.op.append:
            if value in current:
                continue
            node.attributes[key] = f"{current} · {value}"
        else:
            node.attributes[key] = value
        wrote = True
    if event.op.alias and event.op.alias not in node.aliases:
        node.aliases.append(event.op.alias)
        wrote = True
    if not wrote and not event.op.alias:
        return
    if node.kind != "speaker":
        _keep_concept_focus(state, node)


def _reduce_link(state: _State, event: _StoredEvent) -> None:
    op = event.op
    if op.kind not in {"says", "intends", "relates", "tone"}:
        raise GraphRejected("Połączenie ma rodzaj says, intends, relates albo tone.")
    if not op.source_id or not op.target_id:
        raise GraphRejected("Połączenie potrzebuje dwóch istniejących węzłów.")
    if op.source_id not in state.nodes or op.target_id not in state.nodes:
        raise GraphRejected("Połączenie wskazuje węzeł, którego nie ma.")
    label = (op.label or "").strip()[:180]
    identity = op.edge_id or edge_identity(op.source_id, op.target_id, op.kind, label)
    existing = state.edges.get(identity)
    edge = _Edge(
        id=identity,
        source=op.source_id,
        target=op.target_id,
        kind=op.kind,  # type: ignore[arg-type]
        label=label,
        emotion=(op.emotion or None),
        speaker_id=op.speaker_id,
        utterance_id=event.utterance_id,
    )
    if existing is not None:
        if existing == edge:
            return
        raise GraphRejected("Takie połączenie już jest i nie zostanie podmienione.")
    state.edges[identity] = edge


def _reduce_revise(state: _State, event: _StoredEvent) -> None:
    node = _require_node(state, event.op.node_id)
    changed = False
    if event.op.label and event.op.label.strip() and event.op.label.strip() != node.label:
        before = node.label
        node.label = event.op.label.strip()[:80]
        if before not in node.aliases:
            node.aliases.append(before)
        node.history.append(
            HistoryEntry(
                field="label",
                before=before,
                after=node.label,
                utterance_id=event.utterance_id,
            )
        )
        changed = True
    if event.op.alias and event.op.alias not in node.aliases:
        node.aliases.append(event.op.alias)
        changed = True
    for key, value in (event.op.attributes or {}).items():
        _check_attribute(key, value)
        current = node.attributes.get(key)
        if current is None:
            raise GraphRejected("Korekta dotyczy tylko atrybutu, który już jest.")
        if current == value:
            continue
        node.attributes[key] = value
        node.history.append(
            HistoryEntry(
                field=f"attribute:{key}",
                before=current,
                after=value,
                utterance_id=event.utterance_id,
            )
        )
        changed = True
    if not changed:
        return
    if node.kind != "speaker":
        _keep_concept_focus(state, node)


def _keep_concept_focus(state: _State, node: _Node) -> None:
    if node.kind == "concept":
        state.focus_node_id = node.id
    elif state.focus_node_id is None:
        state.focus_node_id = node.id


def _require_node(state: _State, node_id: str | None) -> _Node:
    if not node_id or node_id not in state.nodes:
        raise GraphRejected("Nie ma węzła, który ta operacja miałaby zmienić.")
    return state.nodes[node_id]


def _check_attribute(key: str, value: str) -> None:
    if not key or len(key) > 40 or len(value) > 180:
        raise GraphRejected("Atrybut jest pusty albo za długi.")


def _snapshot(meeting_id: str, state: _State, event_count: int) -> GraphSnapshot:
    nodes = [
        NodeView(
            id=node.id,
            kind=node.kind,
            label=node.label,
            speaker_id=node.speaker_id,
            attributes=dict(node.attributes),
            aliases=list(node.aliases),
            history=list(node.history),
            position=Position(x=node.x, y=node.y),
            utterance_id=node.utterance_id,
        )
        for node in state.nodes.values()
    ]
    edges = [
        EdgeView(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            kind=edge.kind,
            label=edge.label,
            emotion=edge.emotion,
            speaker_id=edge.speaker_id,
            utterance_id=edge.utterance_id,
        )
        for edge in state.edges.values()
    ]
    return GraphSnapshot(
        meeting_id=meeting_id,
        nodes=nodes,
        edges=edges,
        focus_node_id=state.focus_node_id,
        event_count=event_count,
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_events (
    meeting_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    utterance_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    op_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (meeting_id, seq)
);
CREATE TABLE IF NOT EXISTS graph_utterances (
    meeting_id TEXT NOT NULL,
    utterance_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    text TEXT NOT NULL,
    emotion TEXT,
    speaker_label TEXT,
    ops_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (meeting_id, utterance_id)
);
"""


class ConversationGraphStore:
    """Append-only event log. The view is the fold of that log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def snapshot(self, meeting_id: str) -> GraphSnapshot:
        _check_meeting_id(meeting_id)
        with self._lock, self._connect() as conn:
            events = self._load_events(conn, meeting_id)
        return _snapshot(meeting_id, reduce_events(events), len(events))

    def events(self, meeting_id: str) -> list[dict[str, object]]:
        _check_meeting_id(meeting_id)
        with self._lock, self._connect() as conn:
            loaded = self._load_events(conn, meeting_id)
        return [
            {
                "seq": event.seq,
                "utterance_id": event.utterance_id,
                "speaker_id": event.speaker_id,
                "created_at": event.created_at,
                "op": event.op.model_dump(),
            }
            for event in loaded
        ]

    def apply_utterance(self, utterance: Utterance) -> AppliedUtterance:
        _check_meeting_id(utterance.meeting_id)
        _check_speaker_id(utterance.speaker_id)
        text = utterance.text.strip()
        if not text:
            raise GraphRejected("Pusta wypowiedź nie zmienia grafu.")
        with self._lock, self._connect() as conn:
            existing = self._load_utterance(conn, utterance.meeting_id, utterance.utterance_id)
            if existing is not None:
                if existing["text"] != text or existing["speaker_id"] != utterance.speaker_id:
                    raise GraphRejected(
                        "Ten identyfikator wypowiedzi jest już zajęty inną treścią."
                    )
                events = self._load_events(conn, utterance.meeting_id)
                ops = [GraphOp.model_validate(item) for item in json.loads(existing["ops_json"])]
                graph = _snapshot(utterance.meeting_id, reduce_events(events), len(events))
                return AppliedUtterance(
                    utterance_id=utterance.utterance_id,
                    replayed=True,
                    operations=ops,
                    graph=graph,
                )
            events = self._load_events(conn, utterance.meeting_id)
            state = reduce_events(events)
            ops = _propose(state, utterance, text)
            self._commit(
                conn,
                utterance,
                ops,
                emotion=utterance.emotion,
                label=utterance.speaker_label,
            )
            stored = self._load_events(conn, utterance.meeting_id)
            graph = _snapshot(utterance.meeting_id, reduce_events(stored), len(stored))
            return AppliedUtterance(
                utterance_id=utterance.utterance_id,
                replayed=False,
                operations=ops,
                graph=graph,
            )

    def apply_operations(
        self,
        *,
        meeting_id: str,
        utterance_id: str,
        speaker_id: str,
        operations: list[GraphOp],
    ) -> AppliedUtterance:
        """Apply an external proposal. The caller cannot send a replacement graph."""

        _check_meeting_id(meeting_id)
        _check_speaker_id(speaker_id)
        if not operations:
            raise GraphRejected("Brak operacji. Graf bez zmian.")
        utterance = Utterance(
            meeting_id=meeting_id,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            text="[operacje]",
        )
        with self._lock, self._connect() as conn:
            if self._load_utterance(conn, meeting_id, utterance_id) is not None:
                raise GraphRejected("Ten identyfikator wypowiedzi jest już w historii.")
            events = self._load_events(conn, meeting_id)
            trial = reduce_events(events)
            for index, op in enumerate(operations):
                _reduce(
                    trial,
                    _StoredEvent(
                        seq=len(events) + index + 1,
                        utterance_id=utterance_id,
                        speaker_id=speaker_id,
                        op=op,
                        created_at=_now(),
                    ),
                )
            self._commit(conn, utterance, operations, emotion=None, label=None)
            stored = self._load_events(conn, meeting_id)
            graph = _snapshot(meeting_id, reduce_events(stored), len(stored))
            return AppliedUtterance(
                utterance_id=utterance_id,
                replayed=False,
                operations=operations,
                graph=graph,
            )

    def _commit(
        self,
        conn: sqlite3.Connection,
        utterance: Utterance,
        ops: list[GraphOp],
        *,
        emotion: str | None,
        label: str | None,
    ) -> None:
        created_at = _now()
        seq = self._next_seq(conn, utterance.meeting_id)
        for offset, op in enumerate(ops):
            event = _StoredEvent(
                seq=seq + offset,
                utterance_id=utterance.utterance_id,
                speaker_id=utterance.speaker_id,
                op=op,
                created_at=created_at,
            )
            _reduce(reduce_events(self._load_events(conn, utterance.meeting_id)), event)
            conn.execute(
                """
                INSERT INTO graph_events(
                    meeting_id, seq, utterance_id, speaker_id, op_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    utterance.meeting_id,
                    event.seq,
                    event.utterance_id,
                    event.speaker_id,
                    op.model_dump_json(),
                    created_at,
                ),
            )
        conn.execute(
            """
            INSERT INTO graph_utterances(
                meeting_id, utterance_id, speaker_id, text, emotion,
                speaker_label, ops_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utterance.meeting_id,
                utterance.utterance_id,
                utterance.speaker_id,
                utterance.text.strip(),
                emotion,
                label,
                json.dumps([op.model_dump() for op in ops], ensure_ascii=False),
                created_at,
            ),
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _load_events(self, conn: sqlite3.Connection, meeting_id: str) -> list[_StoredEvent]:
        rows = conn.execute(
            """
            SELECT seq, utterance_id, speaker_id, op_json, created_at
            FROM graph_events
            WHERE meeting_id = ?
            ORDER BY seq
            """,
            (meeting_id,),
        ).fetchall()
        return [
            _StoredEvent(
                seq=int(row["seq"]),
                utterance_id=str(row["utterance_id"]),
                speaker_id=str(row["speaker_id"]),
                op=GraphOp.model_validate_json(row["op_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def _load_utterance(
        self,
        conn: sqlite3.Connection,
        meeting_id: str,
        utterance_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT speaker_id, text, ops_json
            FROM graph_utterances
            WHERE meeting_id = ? AND utterance_id = ?
            """,
            (meeting_id, utterance_id),
        ).fetchone()

    def _next_seq(self, conn: sqlite3.Connection, meeting_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM graph_events WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()
        return int(row["next_seq"])


def _now() -> str:
    return datetime.now(UTC).isoformat()
