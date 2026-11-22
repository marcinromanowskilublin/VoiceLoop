import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from voiceloop.conversation_graph import (
    ConversationGraphStore,
    GraphOp,
    GraphRejected,
    Utterance,
    fold,
    stem,
)


def _say(
    store: ConversationGraphStore,
    meeting: str,
    utterance_id: str,
    speaker_id: str,
    text: str,
    emotion: str | None = None,
):
    return store.apply_utterance(
        Utterance(
            meeting_id=meeting,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            text=text,
            emotion=emotion,
        )
    )


def _node(graph, label: str):
    found = [node for node in graph.nodes if node.label == label]
    assert found, [node.label for node in graph.nodes]
    return found[0]


def test_example_enriches_the_same_node_and_keeps_its_place(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    meeting = "przyklad"

    first = _say(store, meeting, "u1", "A", "Chcę zbudować aplikację")
    app_node = _node(first.graph, "Aplikacja")
    speaker_a = next(node for node in first.graph.nodes if node.speaker_id == "A")
    assert app_node.kind == "concept"
    assert speaker_a.kind == "speaker"
    assert {op.op for op in first.operations} <= {"add", "link"}
    assert "replace" not in {op.op for op in first.operations}

    second = _say(store, meeting, "u2", "A", "Ma działać bez internetu")
    same = _node(second.graph, "Aplikacja")
    assert same.id == app_node.id
    assert same.position == app_node.position
    assert same.attributes["lacznosc"] == "bez internetu"
    assert any(op.op == "enrich" for op in second.operations)
    assert not any(op.op == "add" and op.kind != "speaker" for op in second.operations)

    third = _say(
        store,
        meeting,
        "u3",
        "B",
        "Ale synchronizacja powinna być możliwa",
    )
    kept = _node(third.graph, "Aplikacja")
    sync = _node(third.graph, "Synchronizacja")
    speaker_b = next(node for node in third.graph.nodes if node.speaker_id == "B")
    assert kept.id == app_node.id
    assert kept.position == app_node.position
    assert kept.attributes["lacznosc"] == "bez internetu"
    assert speaker_a.position == next(
        node.position for node in third.graph.nodes if node.id == speaker_a.id
    )
    assert sync.kind == "intention"
    assert sync.attributes["stan"] == "możliwa"
    assert any(
        edge.source == sync.id and edge.target == kept.id and edge.speaker_id == "B"
        for edge in third.graph.edges
    )
    assert speaker_b.kind == "speaker"
    assert {node.id for node in first.graph.nodes} <= {node.id for node in third.graph.nodes}
    assert third.graph.event_count > first.graph.event_count


def test_later_words_do_not_drop_earlier_nodes(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    meeting = "film"
    first = _say(
        store,
        meeting,
        "u1",
        "0",
        "Chciałem zaprezentować program, który nie jest kolejnym notatnikiem.",
    )
    program = _node(first.graph, "Program")
    assert "notatnikiem" in program.attributes["nie_jest"]

    second = _say(store, meeting, "u2", "0", "Pomaga wrócić do wątku, jeśli się go zgubiło.")
    assert any(node.id == program.id for node in second.graph.nodes)
    assert _node(second.graph, "Program").position == program.position
    assert any("watk" in fold(node.label) for node in second.graph.nodes)

    third = _say(store, meeting, "u3", "0", "Dzięki. Pozdrawiam.")
    assert {node.id for node in second.graph.nodes} <= {node.id for node in third.graph.nodes}
    assert third.graph.event_count == second.graph.event_count


def test_revision_keeps_id_position_and_history(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    meeting = "korekta"
    _say(store, meeting, "u1", "A", "Chcę zbudować aplikację")
    _say(store, meeting, "u2", "A", "Ma działać bez internetu")
    renamed = _say(store, meeting, "u3", "A", "To nie jest aplikacja, to jest narzędzie")
    node = _node(renamed.graph, "Narzędzie")
    assert node.attributes["lacznosc"] == "bez internetu"
    assert node.history[0].field == "label"
    assert node.history[0].before == "Aplikacja"
    assert node.history[0].after == "Narzędzie"
    assert "Aplikacja" in node.aliases

    revised = _say(store, meeting, "u4", "A", "Jednak ma działać z internetem")
    current = next(item for item in revised.graph.nodes if item.id == node.id)
    assert current.position == node.position
    assert current.attributes["lacznosc"] == "z internetem"
    assert any(
        entry.field == "attribute:lacznosc" and entry.before == "bez internetu"
        for entry in current.history
    )


def test_emotion_is_an_edge_label_not_a_node(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    applied = _say(
        store,
        "emocja",
        "u1",
        "A",
        "Chcę zbudować aplikację",
        emotion="przeciążenie",
    )
    assert all(node.label != "przeciążenie" for node in applied.graph.nodes)
    assert any(edge.emotion == "przeciążenie" for edge in applied.graph.edges)


def test_same_utterance_does_not_duplicate_events(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    first = _say(store, "raz", "u1", "A", "Chcę zbudować aplikację")
    second = _say(store, "raz", "u1", "A", "Chcę zbudować aplikację")
    assert second.replayed is True
    assert second.graph.event_count == first.graph.event_count
    assert len(second.graph.nodes) == len(first.graph.nodes)


def test_store_refuses_to_replace_an_existing_node(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    applied = _say(store, "odrzut", "u1", "A", "Chcę zbudować aplikację")
    existing = _node(applied.graph, "Aplikacja")
    before = applied.graph.event_count
    with pytest.raises(GraphRejected, match="podmieniłaby"):
        store.apply_operations(
            meeting_id="odrzut",
            utterance_id="u2",
            speaker_id="A",
            operations=[
                GraphOp(
                    op="add",
                    op_id="u2:0",
                    node_id=existing.id,
                    kind="concept",
                    label="Coś innego",
                )
            ],
        )
    assert store.snapshot("odrzut").event_count == before
    assert _node(store.snapshot("odrzut"), "Aplikacja").id == existing.id


def test_enrich_cannot_overwrite_an_attribute(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    applied = _say(store, "nadpis", "u1", "A", "Chcę zbudować aplikację")
    _say(store, "nadpis", "u2", "A", "Ma działać bez internetu")
    node = _node(applied.graph, "Aplikacja")
    with pytest.raises(GraphRejected, match="nadpisać"):
        store.apply_operations(
            meeting_id="nadpis",
            utterance_id="u3",
            speaker_id="A",
            operations=[
                GraphOp(
                    op="enrich",
                    op_id="u3:0",
                    node_id=node.id,
                    attributes={"lacznosc": "z internetem"},
                )
            ],
        )
    assert _node(store.snapshot("nadpis"), "Aplikacja").attributes["lacznosc"] == "bez internetu"


def test_meetings_do_not_share_nodes(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    _say(store, "jeden", "u1", "A", "Chcę zbudować aplikację")
    other = store.snapshot("drugi")
    assert other.nodes == []
    assert other.event_count == 0


def test_reopen_folds_the_same_graph(tmp_path) -> None:
    path = tmp_path / "graph.sqlite"
    store = ConversationGraphStore(path)
    _say(store, "trwale", "u1", "A", "Chcę zbudować aplikację")
    _say(store, "trwale", "u2", "A", "Ma działać bez internetu")
    original = store.snapshot("trwale")
    reopened = ConversationGraphStore(path).snapshot("trwale")
    assert reopened == original


def test_missing_speaker_is_rejected(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    with pytest.raises(GraphRejected, match="diaryzacji"):
        _say(store, "brak", "u1", "", "Chcę zbudować aplikację")
    assert store.snapshot("brak").nodes == []


def test_full_graph_payload_is_not_an_operation() -> None:
    with pytest.raises(ValidationError):
        GraphOp.model_validate(
            {
                "op": "replace",
                "op_id": "x",
                "nodes": [],
                "edges": [],
            }
        )


def test_stem_joins_inflected_aplikacja() -> None:
    assert stem(fold("aplikację")) == stem(fold("Aplikacja"))
    assert stem(fold("synchronizacja")) != stem(fold("Aplikacja"))


def test_events_are_the_source_of_the_view(tmp_path) -> None:
    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    _say(store, "zrodlo", "u1", "A", "Chcę zbudować aplikację")
    _say(store, "zrodlo", "u2", "B", "Ale synchronizacja powinna być możliwa")
    names = [event["op"]["op"] for event in store.events("zrodlo")]
    assert names
    assert set(names) <= {"add", "enrich", "link", "revise"}
    raw = json.dumps(store.events("zrodlo"), ensure_ascii=False)
    assert "nodes" not in raw


def test_renderer_uses_stored_positions_without_a_layout_library() -> None:
    page_path = Path(__file__).resolve().parents[1] / "panel" / "graph.html"
    page = page_path.read_text(encoding="utf-8")

    assert "node.position.x" in page
    assert "dagre" not in page
    assert "cytoscape" not in page
    assert "fcose" not in page


@pytest.mark.asyncio
async def test_graph_endpoint_appends_and_refuses_a_replacement(tmp_path) -> None:
    from types import SimpleNamespace

    import httpx

    from voiceloop.app import app

    store = ConversationGraphStore(tmp_path / "graph.sqlite")
    panel = Path(__file__).resolve().parents[1] / "panel"
    missing = object()
    previous = getattr(app.state, "services", missing)
    app.state.services = SimpleNamespace(
        token="expected-local-token",
        settings=SimpleNamespace(panel_dir=panel),
        conversation_graph=store,
    )
    headers = {"X-VoiceLoop-Token": "expected-local-token"}
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            page = await client.get("/graph")
            assert page.status_code == 200
            assert "Graf spotkania" in page.text

            denied = await client.get("/api/v1/conversation-graph/przyklad")
            assert denied.status_code == 401

            created = await client.post(
                "/api/v1/conversation-graph/przyklad/utterances",
                headers=headers,
                json={"speaker_id": "A", "text": "Chcę zbudować aplikację", "utterance_id": "u1"},
            )
            assert created.status_code == 200
            body = created.json()
            app_node = next(node for node in body["graph"]["nodes"] if node["label"] == "Aplikacja")
            enriched = await client.post(
                "/api/v1/conversation-graph/przyklad/utterances",
                headers=headers,
                json={"speaker_id": "A", "text": "Ma działać bez internetu", "utterance_id": "u2"},
            )
            assert enriched.status_code == 200
            again = next(
                node for node in enriched.json()["graph"]["nodes"] if node["id"] == app_node["id"]
            )
            assert again["position"] == app_node["position"]
            assert again["attributes"]["lacznosc"] == "bez internetu"

            replaced = await client.post(
                "/api/v1/conversation-graph/przyklad/operations",
                headers=headers,
                json={"speaker_id": "A", "utterance_id": "u3", "nodes": [], "edges": []},
            )
            assert replaced.status_code == 422
            kept = await client.get("/api/v1/conversation-graph/przyklad", headers=headers)
            assert any(node["id"] == app_node["id"] for node in kept.json()["nodes"])
    finally:
        if previous is missing:
            del app.state.services
        else:
            app.state.services = previous
