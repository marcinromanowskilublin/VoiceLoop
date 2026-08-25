"""Testy segmentacji na fragmenty i łańcucha rodzic–dziecko.

Poziom segmentacji to ziarnistość tego samego znaczenia, a nie osobna
przestrzeń wektorowa. Każdy fragment musi więc być samodzielnym punktem
z poprawnym wskaźnikiem na rodzica, bo na tym opiera się nawigacja
słowo → fraza → zdanie → wypowiedź.
"""

from __future__ import annotations

from typing import Any

from vectorscope.analysis import hierarchy_edges, nearest_shown_ancestor
from vectorscope.fragments import (
    LEVEL_SENTENCE,
    LEVEL_UTTERANCE,
    LEVEL_WORD,
    LEVELS,
    build_fragments,
    merge_identical,
    normalise_transcript,
    select_levels,
)


def _word(text: str, start: float, end: float, speaker: int = 0) -> dict[str, Any]:
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def _normalised_transcript() -> dict[str, Any]:
    words = [
        ("Pacjent", 0.0, 0.4),
        ("zgłasza", 0.4, 0.9),
        ("silny", 0.9, 1.2),
        ("lęk.", 1.2, 1.5),
        ("Boli", 2.1, 2.4),
        ("go", 2.4, 2.6),
        ("głowa", 2.6, 3.1),
        ("od", 3.1, 3.3),
        ("tygodnia.", 3.3, 3.9),
    ]
    return {
        "text": "Pacjent zgłasza silny lęk. Boli go głowa od tygodnia.",
        "words": [_word(text, start, end) for text, start, end in words],
    }


def _raw_deepgram_transcript() -> dict[str, Any]:
    """Kształt, jaki zapisywały wcześniejsze wersje panelu: surowa odpowiedź API."""

    return {
        "metadata": {"model_info": {}},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Kot siedzi na parapecie. Faktura nie zapłacona.",
                            "words": [
                                {
                                    "word": "kot",
                                    "punctuated_word": "Kot",
                                    "start": 0.0,
                                    "end": 0.3,
                                    "confidence": 0.99,
                                    "speaker": 0,
                                },
                                {
                                    "word": "siedzi",
                                    "punctuated_word": "siedzi",
                                    "start": 0.3,
                                    "end": 0.8,
                                    "speaker": 0,
                                },
                                {
                                    "word": "na",
                                    "punctuated_word": "na",
                                    "start": 0.8, "end": 0.9,
                                    "speaker": 0,
                                },
                                {
                                    "word": "parapecie",
                                    "punctuated_word": "parapecie.",
                                    "start": 0.9,
                                    "end": 1.6,
                                    "speaker": 0,
                                },
                                {
                                    "word": "faktura",
                                    "punctuated_word": "Faktura",
                                    "start": 2.2,
                                    "end": 2.7,
                                    "speaker": 0,
                                },
                                {
                                    "word": "nie",
                                    "punctuated_word": "nie",
                                    "start": 2.7,
                                    "end": 2.9,
                                    "speaker": 0,
                                },
                                {
                                    "word": "zapłacona",
                                    "punctuated_word": "zapłacona.",
                                    "start": 2.9,
                                    "end": 3.6,
                                    "speaker": 0,
                                },
                            ],
                            "paragraphs": {
                                "paragraphs": [
                                    {
                                        "sentences": [
                                            {"text": "Kot siedzi na parapecie."},
                                            {"text": "Faktura nie zapłacona."},
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        },
    }


# ------------------------------------------------------- normalizacja formatu


def test_normalised_transcript_is_passed_through_unchanged() -> None:
    transcript = _normalised_transcript()
    assert normalise_transcript(transcript) is transcript


def test_raw_deepgram_response_is_understood() -> None:
    """Regresja: nagrania z wcześniejszej wersji panelu dawały zero fragmentów."""

    normalised = normalise_transcript(_raw_deepgram_transcript())
    assert normalised["text"].startswith("Kot siedzi")
    assert len(normalised["words"]) == 7
    assert normalised["words"][0]["text"] == "Kot", "punctuated_word ma pierwszeństwo"
    assert normalised["sentences"] == [
        "Kot siedzi na parapecie.",
        "Faktura nie zapłacona.",
    ]


def test_raw_response_without_paragraphs_still_yields_words() -> None:
    raw = _raw_deepgram_transcript()
    del raw["results"]["channels"][0]["alternatives"][0]["paragraphs"]
    normalised = normalise_transcript(raw)
    assert len(normalised["words"]) == 7
    assert normalised["sentences"] == []


def test_unrecognised_shape_degrades_to_empty_instead_of_raising() -> None:
    normalised = normalise_transcript({"results": {}})
    assert normalised["words"] == []
    assert normalised["text"] == ""
    assert build_fragments(transcript={"results": {}}, recording_id="r", recording_label="x") == []


def test_raw_response_produces_the_same_fragments_as_the_normalised_one() -> None:
    fragments = build_fragments(
        transcript=_raw_deepgram_transcript(), recording_id="r1", recording_label="stare"
    )
    assert fragments, "stare nagranie musi dać fragmenty"
    levels = {fragment.level for fragment in fragments}
    assert LEVEL_WORD in levels and LEVEL_SENTENCE in levels


# ------------------------------------------------------------------ struktura


def test_every_produced_level_is_declared() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    assert {fragment.level for fragment in fragments} <= set(LEVELS)


def test_fragment_identifiers_are_unique() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    identifiers = [fragment.id for fragment in fragments]
    assert len(identifiers) == len(set(identifiers))


def test_parent_chain_reaches_an_utterance_from_every_word() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    by_id = {fragment.id: fragment for fragment in fragments}
    words = [fragment for fragment in fragments if fragment.level == LEVEL_WORD]
    assert words

    for word in words:
        seen: set[str] = set()
        current = word.parent_id
        levels_on_the_way = []
        while current:
            assert current not in seen, "łańcuch rodziców nie może się zapętlić"
            seen.add(current)
            levels_on_the_way.append(by_id[current].level)
            current = by_id[current].parent_id
        assert levels_on_the_way[-1] == LEVEL_UTTERANCE


def test_only_utterances_have_no_parent() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    for fragment in fragments:
        if fragment.parent_id is None:
            assert fragment.level == LEVEL_UTTERANCE


def test_children_stay_inside_the_time_range_of_their_parent() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    by_id = {fragment.id: fragment for fragment in fragments}
    for fragment in fragments:
        parent = by_id.get(fragment.parent_id or "")
        if parent is None or fragment.start_ms is None or parent.start_ms is None:
            continue
        assert parent.start_ms <= fragment.start_ms
        assert parent.end_ms >= fragment.end_ms


def test_words_carry_millisecond_timing() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    first = next(f for f in fragments if f.level == LEVEL_WORD)
    assert first.start_ms == 0
    assert first.end_ms == 400


# ------------------------------------------------------------------- wybory


def test_select_levels_keeps_only_the_requested_ones() -> None:
    fragments = build_fragments(
        transcript=_normalised_transcript(), recording_id="r1", recording_label="test"
    )
    chosen = select_levels(fragments, [LEVEL_SENTENCE])
    assert chosen
    assert {fragment.level for fragment in chosen} == {LEVEL_SENTENCE}


def test_merge_identical_collapses_repeats_and_reports_members() -> None:
    transcript = {
        "text": "kot kot pies",
        "words": [_word("kot", 0.0, 0.3), _word("kot", 0.4, 0.7), _word("pies", 0.8, 1.2)],
    }
    fragments = build_fragments(
        transcript=transcript, recording_id="r1", recording_label="test"
    )
    words = select_levels(fragments, [LEVEL_WORD])
    merged, members = merge_identical(words)
    texts = [fragment.text.casefold() for fragment in merged]
    assert texts.count("kot") == 1
    assert len(merged) == len(members)
    assert any(len(group) == 2 for group in members)


# ------------------------------------------------- wspinaczka po przodkach


def test_ancestor_walk_skips_levels_that_are_not_displayed() -> None:
    """Słowo → fraza → zdanie: gdy fraz nie ma na wykresie, linia musi sięgnąć zdania."""

    parent_of = {"w1": "p1", "p1": "s1", "s1": None}
    index_by_id = {"w1": 0, "s1": 1}
    assert nearest_shown_ancestor("w1", parent_of, index_by_id) == 1


def test_ancestor_walk_returns_nothing_when_no_ancestor_is_shown() -> None:
    parent_of = {"w1": "p1", "p1": "s1", "s1": None}
    assert nearest_shown_ancestor("w1", parent_of, {"w1": 0}) is None


def test_ancestor_walk_survives_a_corrupted_cycle() -> None:
    parent_of = {"a": "b", "b": "c", "c": "a"}
    assert nearest_shown_ancestor("a", parent_of, {"a": 0}) is None


def test_ancestor_walk_prefers_the_nearest_ancestor() -> None:
    parent_of = {"w1": "p1", "p1": "s1", "s1": "u1", "u1": None}
    index_by_id = {"w1": 0, "p1": 1, "s1": 2, "u1": 3}
    assert nearest_shown_ancestor("w1", parent_of, index_by_id) == 1


def test_hierarchy_edges_never_point_a_node_at_itself() -> None:
    parent_of = {"a": "a", "b": "a"}
    edges = hierarchy_edges(["a", "b"], parent_of, {"a": 0, "b": 1})
    assert edges == [{"source": 1, "target": 0}]


def test_hierarchy_edges_are_empty_without_any_parents() -> None:
    assert hierarchy_edges(["a", "b"], {"a": None, "b": None}, {"a": 0, "b": 1}) == []
