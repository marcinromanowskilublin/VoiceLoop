"""Testy werdyktów o progach.

Próg cosinusa może być martwy na dwa przeciwne sposoby: leżeć poniżej całego
szumu (nie odrzuca niczego) albo powyżej sufitu porównania, które wykonuje
(nie może zadziałać nigdy). Panel musi rozróżniać oba przypadki i milczeć tam,
gdzie niczego nie zmierzył.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from vectorscope.scale import (
    DOMAIN_CORPUS,
    _percentiles,
    _reachability_entry,
    _retrieval_view,
)


@dataclass(frozen=True)
class _Threshold:
    key: str
    label: str
    value: float
    origin: str


def _threshold(value: float) -> _Threshold:
    return _Threshold(key="prog", label="Próg", value=value, origin="settings.py")


# ------------------------------------------------------------- werdykt progu


def test_threshold_above_the_ceiling_can_never_fire() -> None:
    entry = _reachability_entry(
        _threshold(0.92), space="x", ceiling=0.789, floor_min=0.007, floor_p95=0.265
    )
    assert entry["verdict"] == "nieosiagalny"
    assert "0.789" in entry["message"]


def test_threshold_below_the_whole_noise_rejects_nothing() -> None:
    entry = _reachability_entry(
        _threshold(0.15), space="x", ceiling=0.781, floor_min=0.162, floor_p95=0.53
    )
    assert entry["verdict"] == "martwy"


def test_threshold_inside_the_noise_is_reported_as_partial() -> None:
    entry = _reachability_entry(
        _threshold(0.30), space="x", ceiling=0.78, floor_min=0.10, floor_p95=0.53
    )
    assert entry["verdict"] == "wewnatrz_szumu"


def test_threshold_between_noise_and_ceiling_actually_separates() -> None:
    entry = _reachability_entry(
        _threshold(0.60), space="x", ceiling=0.78, floor_min=0.10, floor_p95=0.53
    )
    assert entry["verdict"] == "dziala"


def test_a_threshold_exactly_at_the_ceiling_is_still_reachable() -> None:
    entry = _reachability_entry(
        _threshold(0.78), space="x", ceiling=0.78, floor_min=0.10, floor_p95=0.53
    )
    assert entry["verdict"] != "nieosiagalny"


def test_missing_threshold_does_not_produce_a_verdict() -> None:
    assert _reachability_entry(None, space="x", ceiling=1.0, floor_min=0.0, floor_p95=0.5) == {
        "verdict": "brak_progu"
    }


def test_entry_records_the_space_it_was_measured_in() -> None:
    entry = _reachability_entry(
        _threshold(0.5), space="surowa treść kontra dokument", ceiling=0.9,
        floor_min=0.1, floor_p95=0.4,
    )
    assert entry["space"] == "surowa treść kontra dokument"
    assert entry["ceiling"] == 0.9


# ----------------------------------------------------------------- rozkłady


def test_retrieval_view_separates_same_and_cross_domain_pairs() -> None:
    domains = np.array(["a", "a", "b", "b"])
    # Punkty tej samej dziedziny bliżej siebie niż punkty różnych dziedzin.
    vectors = np.array(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]],
    )
    view = _retrieval_view(queries=vectors, documents=vectors, domains=domains)
    assert view["pairs_related"] == 4
    assert view["pairs_unrelated"] == 8
    assert view["signal"]["p50"] > view["floor"]["p50"]
    assert view["separation"] > 0


def test_retrieval_view_reports_identical_pairs_separately() -> None:
    domains = np.array(["a", "b"])
    vectors = np.eye(2)
    view = _retrieval_view(queries=vectors, documents=vectors, domains=domains)
    assert view["identical_text"]["p50"] == 1.0
    assert view["pairs_related"] == 0


def test_empty_group_never_produces_nan_which_is_invalid_json() -> None:
    """Regresja: NaN przechodził do odpowiedzi i wywracał panel daleko od przyczyny."""

    domains = np.array(["a", "b"])
    vectors = np.eye(2)
    view = _retrieval_view(
        queries=vectors, documents=vectors, domains=domains, threshold=0.5
    )
    numbers = [
        view["separation"],
        view["signal_below_floor_p95"],
        view["noise_above_threshold"],
        view["signal_above_threshold"],
        *view["signal"].values(),
        *view["floor"].values(),
    ]
    assert all(np.isfinite(value) for value in numbers)
    assert json.dumps(view, allow_nan=False)


def test_retrieval_view_counts_threshold_shares_only_when_asked() -> None:
    domains = np.array(["a", "b", "a", "b"])
    vectors = np.eye(4)[:, :4]
    plain = _retrieval_view(queries=vectors, documents=vectors, domains=domains)
    assert "noise_above_threshold" not in plain

    scored = _retrieval_view(
        queries=vectors, documents=vectors, domains=domains, threshold=0.5
    )
    assert 0.0 <= scored["noise_above_threshold"] <= 1.0
    assert 0.0 <= scored["signal_above_threshold"] <= 1.0


def test_percentiles_are_ordered() -> None:
    values = np.linspace(0.0, 1.0, 500)
    result = _percentiles(values)
    assert result["min"] <= result["p05"] <= result["p50"]
    assert result["p50"] <= result["p90"] <= result["p95"] <= result["p99"] <= result["max"]


def test_percentiles_survive_an_empty_sample() -> None:
    assert _percentiles(np.array([]))["p50"] == 0.0


# ------------------------------------------------------------------- korpus


def test_domain_corpus_has_enough_pairs_to_speak_in_percentiles() -> None:
    """Jedna kotwica nie wystarczy do orzekania o dnie skali — stąd ten korpus."""

    total = sum(len(items) for items in DOMAIN_CORPUS.values())
    cross_domain = sum(
        len(left) * len(right)
        for index, left in enumerate(DOMAIN_CORPUS.values())
        for right in list(DOMAIN_CORPUS.values())[index + 1 :]
    )
    assert len(DOMAIN_CORPUS) >= 6, "za mało dziedzin, żeby pary były naprawdę rozłączne"
    assert total >= 40
    assert cross_domain >= 500


def test_domain_corpus_has_no_duplicate_sentences() -> None:
    sentences = [item for items in DOMAIN_CORPUS.values() for item in items]
    assert len(sentences) == len(set(sentences))
