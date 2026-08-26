"""Geometria prefiksów zadania: co prefiks robi z przestrzenią.

VoiceLoop miesza trzy konwencje. Nowa ścieżka indeksowania woła `embed_documents`
(`search_document: `), stara `embed_texts` (bez prefiksu), a zapytania idą zawsze
przez `embed_queries` (`search_query: `). Ten moduł mierzy, co ta niespójność
robi z geometrią — i tylko to.

Poprzednia wersja liczyła tu jeszcze trafienie w pierwszym wyniku i MRR na
dwudziestu czterech ręcznie napisanych parach pytanie–dokument. To był mikro-
benchmark retrievalowy, którego wnioski sam moduł musiał opatrywać ostrzeżeniem,
że różnica jednej sondy mieści się w szumie. Pytania o jakość wyszukiwania mają
teraz właściwe miejsce: `live.py` mierzy je na prawdziwej kolekcji, na realnych
wektorach i bez wymyślonych etykiet. Zostaje tu wyłącznie to, co jest
własnością modelu, a nie własnością zbioru testowego.

Cztery pomiary, wszystkie niezależne od etykiet:

1. **Wstrzyknięcie prefiksu** — czy serwer nie dokleja prefiksu po cichu. Gdyby
   doklejał, cała reszta porównywałaby to samo ze sobą, więc to jest warunek
   ważności, nie ciekawostka.
2. **Przesunięcie** — jak daleko prefiks odsuwa wektor tego samego tekstu.
3. **Ściśnięcie zakresu** — ile wspólnego kierunku prefiks dokłada wszystkim
   wektorom, czyli o ile podnosi dno skali.
4. **Zakres użyteczny** — odstęp między parami z tej samej dziedziny i z różnych.
   To jedyna liczba, która mówi, czy w danej konwencji jest jeszcze czym
   rozróżniać.

Nic tutaj nie modyfikuje pamięci VoiceLoopa.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from voiceloop.embeddings import EmbeddingUnavailableError

from . import geometry
from .config import (
    PREFIX_DOCUMENT,
    PREFIX_NONE,
    PREFIX_QUERY,
    build_embedding_client,
    settings,
)
from .embed import embed_texts_with_prefix
from .scale import DOMAIN_CORPUS

INJECTION_LIMIT = 0.99


@dataclass(frozen=True)
class PrefixUse:
    key: str
    label: str
    note: str


PREFIX_USES: tuple[PrefixUse, ...] = (
    PrefixUse(
        key=PREFIX_DOCUMENT,
        label="search_document",
        note="Nowa ścieżka indeksowania: embed_documents.",
    ),
    PrefixUse(
        key=PREFIX_QUERY,
        label="search_query",
        note="Każde zapytanie do pamięci: embed_queries.",
    ),
    PrefixUse(
        key=PREFIX_NONE,
        label="bez prefiksu",
        note="Stara ścieżka Screenpipe: embed_texts, wbrew karcie modelu.",
    ),
)


async def run_prefix_check(
    *,
    corpus: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    active = settings()
    client = build_embedding_client(active)

    domains = corpus or DOMAIN_CORPUS
    texts: list[str] = []
    labels: list[str] = []
    for domain, sentences in domains.items():
        for sentence in sentences:
            texts.append(sentence)
            labels.append(domain)

    try:
        runs = {
            use.key: await embed_texts_with_prefix(client, texts, prefix=use.key)
            for use in PREFIX_USES
        }
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    vectors = {key: geometry.l2_normalize(run.vectors) for key, run in runs.items()}
    reference = runs[PREFIX_DOCUMENT]

    injection = _prefix_injection(vectors)
    clouds = [_cloud(use, vectors[use.key], labels) for use in PREFIX_USES]
    displacement = _displacement(vectors)

    return {
        "ok": True,
        "model": reference.model,
        "dimension": reference.dimension,
        "text_count": len(texts),
        "domain_count": len(domains),
        "prefix_injection": injection,
        "clouds": clouds,
        "displacement": displacement,
        "scope_note": (
            "Ten moduł mierzy geometrię przestrzeni, nie jakość wyszukiwania. "
            "Trafność retrievalu jest mierzona w live.py na prawdziwej kolekcji, "
            "bo tylko tam sygnał nie zależy od tego, jak napisano zbiór testowy."
        ),
        "interpretation": _interpret(injection, clouds, displacement),
    }


def _cloud(use: PrefixUse, matrix: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Rozkład par w obrębie jednej konwencji: dno, sygnał i odstęp między nimi."""

    similarity = matrix @ matrix.T
    same: list[float] = []
    cross: list[float] = []
    for left, right in combinations(range(len(labels)), 2):
        value = float(similarity[left, right])
        if labels[left] == labels[right]:
            same.append(value)
        else:
            cross.append(value)

    same_array = np.asarray(same)
    cross_array = np.asarray(cross)
    separation = float(np.median(same_array) - np.median(cross_array))

    return {
        "prefix": use.key,
        "label": use.label,
        "note": use.note,
        "same_domain": _distribution(same_array),
        "cross_domain": _distribution(cross_array),
        "floor": round(float(np.percentile(cross_array, 95)), 3),
        "usable_range": round(separation, 3),
        "pairs": {"same": len(same), "cross": len(cross)},
    }


def _displacement(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Jak daleko prefiks odsuwa wektor tego samego, niezmienionego tekstu."""

    pairs = {
        f"{left}|{right}": _distribution(np.sum(vectors[left] * vectors[right], axis=1))
        for left, right in combinations(sorted(vectors), 2)
    }
    return {"same_text_between_prefixes": pairs}


def _prefix_injection(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Czy serwer sam dokleja prefiks, kiedy go nie podamy.

    Gdyby doklejał, tekst wysłany bez prefiksu miałby z którymś z wariantów
    cosinus bliski 1.000 i wszystkie pozostałe pomiary porównywałyby to samo
    ze sobą. Dlatego to jest pierwszy pomiar, nie ostatni.
    """

    bare = vectors[PREFIX_NONE]
    scores = {
        prefix: round(float(np.median(np.sum(bare * matrix, axis=1))), 3)
        for prefix, matrix in vectors.items()
        if prefix != PREFIX_NONE
    }
    suspected = [prefix for prefix, value in scores.items() if value > INJECTION_LIMIT]
    nearest = max(scores, key=scores.get) if scores else None

    if suspected:
        verdict = (
            f"Serwer prawdopodobnie sam dokleja prefiks {suspected[0]} — "
            "pomiary konwencji są wtedy bez wartości."
        )
    else:
        verdict = (
            "Serwer nie dokleja prefiksu samodzielnie: tekst bez prefiksu daje "
            "inny wektor niż każdy z wariantów z prefiksem."
        )

    lean: str | None = None
    if nearest and scores[nearest] > 0.95 and nearest not in suspected:
        lean = (
            f"Tekst bez prefiksu leży bardzo blisko wariantu {nearest} "
            f"(cosinus {scores[nearest]:.3f}). Model traktuje goły tekst niemal jak "
            f"{nearest}, więc stara ścieżka Screenpipe de facto zapisuje dokumenty "
            "tak, jakby były zapytaniami."
        )

    return {
        "median_similarity_to_prefixed": scores,
        "suspected_injection": suspected,
        "nearest_prefixed_variant": nearest,
        "verdict": verdict,
        "lean": lean,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    """Mediana i kwartyle. Średnia z czterema miejscami po przecinku sugerowałaby
    precyzję, której na takiej próbie nie ma."""

    if values.size == 0:
        return {key: 0.0 for key in ("median", "q1", "q3", "min", "max")}
    return {
        "median": round(float(np.median(values)), 3),
        "q1": round(float(np.percentile(values, 25)), 3),
        "q3": round(float(np.percentile(values, 75)), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def _interpret(
    injection: dict[str, Any],
    clouds: list[dict[str, Any]],
    displacement: dict[str, Any],
) -> list[str]:
    notes: list[str] = [injection["verdict"]]
    if injection.get("lean"):
        notes.append(injection["lean"])

    by_prefix = {item["prefix"]: item for item in clouds}
    mismatch = displacement["same_text_between_prefixes"].get(
        "|".join(sorted((PREFIX_DOCUMENT, PREFIX_QUERY)))
    )
    if mismatch and mismatch["median"] < 0.99:
        notes.append(
            f"Ten sam tekst jako dokument i jako zapytanie ma cosinus "
            f"{mismatch['median']:.3f}, nie 1.000. To jest cena pomyłki prefiksu i "
            "dokładnie ona unieruchomiła próg deduplikacji Screenpipe."
        )

    widest = max(clouds, key=lambda item: item["usable_range"])
    narrowest = min(clouds, key=lambda item: item["usable_range"])
    notes.append(
        f"Zakres użyteczny, czyli odstęp między parami z tej samej dziedziny a "
        f"parami z różnych: najszerszy w {widest['label']} ({widest['usable_range']:.3f}), "
        f"najwęższy w {narrowest['label']} ({narrowest['usable_range']:.3f})."
    )

    document = by_prefix.get(PREFIX_DOCUMENT)
    bare = by_prefix.get(PREFIX_NONE)
    if document and bare:
        lift = document["cross_domain"]["median"] - bare["cross_domain"]["median"]
        if abs(lift) > 0.02:
            direction = "podnosi" if lift > 0 else "obniża"
            notes.append(
                f"Prefiks dokumentu {direction} dno skali o {abs(lift):.3f} "
                f"({bare['cross_domain']['median']:.3f} → "
                f"{document['cross_domain']['median']:.3f}). Każdy próg absolutny "
                "przeniesiony między konwencjami znaczy więc co innego."
            )

    floors = {item["label"]: item["floor"] for item in clouds}
    notes.append(
        "Dno mierzone jako 95. percentyl par z różnych dziedzin: "
        + ", ".join(f"{label} {value:.3f}" for label, value in floors.items())
        + ". Żaden próg poniżej najniższej z tych liczb nie odrzuci niczego."
    )
    return notes


__all__ = [
    "INJECTION_LIMIT",
    "PREFIX_USES",
    "PrefixUse",
    "run_prefix_check",
]
