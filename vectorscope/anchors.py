"""Kotwice skali — pary o znanej relacji, żeby liczba cosinusa cokolwiek znaczyła.

Bez punktu odniesienia cosinus 0.62 nie mówi nic: u jednego modelu to blisko,
u innego daleko. Antonimy są tu celowo: w modelach językowych zwykle mają
wysoki cosinus, co jest najczęstszym źródłem błędnych wniosków o embeddingach.
"""

from __future__ import annotations

from dataclasses import dataclass

ANCHOR_LABEL = "kotwica"


@dataclass(frozen=True)
class AnchorPair:
    key: str
    left: str
    right: str
    relation: str
    expectation: str


ANCHOR_PAIRS: tuple[AnchorPair, ...] = (
    AnchorPair(
        key="identyczne",
        left="pacjent czeka w gabinecie",
        right="pacjent czeka w gabinecie",
        relation="identyczny tekst",
        expectation="dokładnie 1.00 — sprawdza determinizm modelu",
    ),
    AnchorPair(
        key="parafraza",
        left="boli mnie głowa",
        right="mam ból głowy",
        relation="parafraza",
        expectation="wysoko — to samo znaczenie, inne słowa",
    ),
    AnchorPair(
        key="synonimy",
        left="lęk",
        right="niepokój",
        relation="synonimy",
        expectation="wysoko, jeśli model rozumie polski",
    ),
    AnchorPair(
        key="ta_sama_dziedzina",
        left="lekarz",
        right="pacjent",
        relation="ta sama dziedzina",
        expectation="średnio — powiązane, ale nie tożsame",
    ),
    AnchorPair(
        key="antonimy",
        left="wysoki",
        right="niski",
        relation="antonimy",
        expectation="zwykle wysoko — embedding łapie wymiar, nie znak",
    ),
    AnchorPair(
        key="bez_zwiazku",
        left="lęk",
        right="silnik wysokoprężny",
        relation="bez związku",
        expectation="nisko — to jest twoje dno skali",
    ),
)


def anchor_texts() -> list[str]:
    """Unikalne teksty kotwic w stabilnej kolejności."""

    seen: dict[str, None] = {}
    for pair in ANCHOR_PAIRS:
        seen.setdefault(pair.left, None)
        seen.setdefault(pair.right, None)
    return list(seen.keys())


def anchor_pair_payload() -> list[dict[str, str]]:
    return [
        {
            "key": pair.key,
            "left": pair.left,
            "right": pair.right,
            "relation": pair.relation,
            "expectation": pair.expectation,
        }
        for pair in ANCHOR_PAIRS
    ]
