"""Rozkłady, na których stoją progi pamięci wektorowej.

Rdzeń pomiaru leży w kodzie produkcyjnym, a nie w panelu diagnostycznym, z jednego
powodu: próg, którego nikt nie mierzy w środowisku docelowym, umiera po cichu.
Trzy takie progi stały martwe miesiącami, mimo że jedna trzecia tego projektu
zajmuje się mierzeniem samej siebie. Pomiar bez konsumenta nie jest sprzężeniem
zwrotnym.

Moduł jest wyłącznie do odczytu — `scroll` i `query_points`, nigdy `upsert`.
Przyrząd, który zmienia mierzony obiekt, jest bezużyteczny.

Kluczowa zasada: **każdy próg trzeba mierzyć w przestrzeni, w której działa.**
Ten sam liczbowy próg 0.92 był nieosiągalny przy porównaniu zapytania z dokumentem
i całkowicie osiągalny przy porównaniu dokumentu z dokumentem. Bez rozdzielenia
przestrzeni wynik pomiaru nic nie znaczy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .memory_vectorization import MEMORY_VECTOR_NAMES, memory_query_documents

STATUS_DISABLED = "disabled"
STATUS_DEAD = "dead"
STATUS_UNREACHABLE = "unreachable"
STATUS_OVER_BROAD = "over_broad"
STATUS_DRIFTED = "drifted"
STATUS_WORKING = "working"
STATUS_UNMEASURED = "unmeasured"

#: Statusy, które znaczą „to nie robi tego, co obiecuje".
BROKEN_STATUSES = frozenset(
    {STATUS_DEAD, STATUS_UNREACHABLE, STATUS_OVER_BROAD, STATUS_DRIFTED}
)

#: Powyżej tego podobieństwa uznajemy, że dokument odtworzono dokładnie.
#: Ten sam tekst przepuszczony przez ten sam szablon i ten sam model wraca
#: na 1.000, więc wszystko poniżej znaczy, że zapisany wektor powstał inaczej.
EXACT_RECONSTRUCTION = 0.999

#: Poniżej tego odsetka dokładnych odtworzeń kolekcja nie zgadza się ze schematem,
#: który deklaruje w metadanych.
RECONSTRUCTION_EXPECTED_SHARE = 0.90

#: Powyżej tego odsetka par *różnych* dokumentów próg deduplikacji zaczyna
#: odrzucać treść nową, nie tylko powtórzoną. Granica z pomiaru na realnej
#: kolekcji, gdzie p99 szumu wynosił 0.874 przy progu 0.97.
DUPLICATE_FALSE_POSITIVE_LIMIT = 0.05

PERCENTILE_KEYS = ("min", "p50", "p90", "p95", "p99", "max")


@dataclass(frozen=True)
class ThresholdVerdict:
    """Jedno orzeczenie o jednym progu w jednej przestrzeni."""

    name: str
    value: float
    space: str
    status: str
    rejected_share: float
    sample: int
    observed: dict[str, float]
    message: str

    @property
    def broken(self) -> bool:
        return self.status in BROKEN_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "space": self.space,
            "status": self.status,
            "rejected_share": self.rejected_share,
            "sample": self.sample,
            "observed": self.observed,
            "message": self.message,
        }


@dataclass
class AxisScores:
    """Surowe wyniki jednej osi, zebrane bez żadnego progu."""

    axis: str
    scores: list[float] = field(default_factory=list)
    self_scores: list[float] = field(default_factory=list)
    self_ranks: list[int] = field(default_factory=list)
    not_found: int = 0
    error: str | None = None


def percentiles(values: Any) -> dict[str, float]:
    """Percentyle rozkładu; pusty rozkład zwraca zera, nie NaN.

    NaN psuje serializację JSON, a strażnik ma raportować przez `/api/v1/health`.
    """

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return dict.fromkeys(PERCENTILE_KEYS, 0.0)
    return {
        "min": round(float(array.min()), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "max": round(float(array.max()), 4),
    }


def unmeasured_verdict(*, name: str, value: float, space: str, reason: str) -> ThresholdVerdict:
    """Próg, którego ten pomiar nie obejmuje — powiedziane wprost, nie przemilczane."""

    return ThresholdVerdict(
        name=name,
        value=value,
        space=space,
        status=STATUS_UNMEASURED,
        rejected_share=0.0,
        sample=0,
        observed=dict.fromkeys(PERCENTILE_KEYS, 0.0),
        message=reason,
    )


def classify_threshold(
    *,
    name: str,
    value: float,
    space: str,
    observed: Any,
) -> ThresholdVerdict:
    """Orzeczenie o progu odsiewającym: martwy, nieosiągalny albo działający.

    Klasyfikacja idzie z odsetka odrzuceń, nie z porównania z minimum i maksimum.
    Wynik jest ten sam, ale odsetek mówi dodatkowo, ile próg naprawdę robi.
    """

    scores = np.asarray(list(observed), dtype=float)
    stats = percentiles(scores)

    if scores.size == 0:
        return unmeasured_verdict(
            name=name,
            value=value,
            space=space,
            reason=f"Brak obserwacji w przestrzeni {space} — nie ma na czym orzekać.",
        )

    if value <= 0.0:
        return ThresholdVerdict(
            name=name,
            value=value,
            space=space,
            status=STATUS_DISABLED,
            rejected_share=0.0,
            sample=int(scores.size),
            observed=stats,
            message=(
                f"Próg {value:g} nie deklaruje bramki, więc nie ma czego mierzyć. "
                f"Rozkład w tej przestrzeni: mediana {stats['p50']:.3f}, "
                f"minimum {stats['min']:.3f}."
            ),
        )

    rejected = float(np.mean(scores < value))
    if rejected <= 0.0:
        status = STATUS_DEAD
        message = (
            f"Próg {value:g} nie odrzuca niczego: najniższy zmierzony wynik to "
            f"{stats['min']:.3f} przy {scores.size} obserwacjach. Bramka istnieje "
            "w kodzie, ale nie w danych."
        )
    elif rejected >= 1.0:
        status = STATUS_UNREACHABLE
        message = (
            f"Próg {value:g} nie przepuszcza niczego: najwyższy zmierzony wynik to "
            f"{stats['max']:.3f}. Wszystko za bramką jest martwym kodem."
        )
    else:
        status = STATUS_WORKING
        message = (
            f"Próg {value:g} odrzuca {rejected:.1%} obserwacji "
            f"(mediana {stats['p50']:.3f}, p99 {stats['p99']:.3f})."
        )

    return ThresholdVerdict(
        name=name,
        value=value,
        space=space,
        status=status,
        rejected_share=round(rejected, 4),
        sample=int(scores.size),
        observed=stats,
        message=message,
    )


def classify_duplicate_threshold(
    *,
    name: str,
    value: float,
    space: str,
    identical: Any,
    distinct: Any,
) -> ThresholdVerdict:
    """Orzeczenie o progu deduplikacji, który ma dwa zadania naraz.

    Próg deduplikacji musi przepuścić powtórzenie i nie odrzucić treści nowej,
    więc jedna dystrybucja nie wystarczy. `identical` to podobieństwo tekstu do
    własnego zapisanego wektora — sufit osiągalności. `distinct` to podobieństwo
    dokumentów faktycznie różnych — podłoga, nad którą próg musi leżeć.

    Tu właśnie ukrył się błąd, którego zwykły pomiar nie widział: próg 0.92
    porównywano w przestrzeni zapytanie–dokument, gdzie identyczny tekst sięgał
    najwyżej 0.90. Deduplikacja nie odrzuciła nigdy niczego i nie mogła.
    """

    ceiling = np.asarray(list(identical), dtype=float)
    floor = np.asarray(list(distinct), dtype=float)
    stats = percentiles(ceiling)

    if ceiling.size == 0:
        return unmeasured_verdict(
            name=name,
            value=value,
            space=space,
            reason=f"Brak obserwacji w przestrzeni {space} — nie ma na czym orzekać.",
        )

    if value <= 0.0:
        return ThresholdVerdict(
            name=name,
            value=value,
            space=space,
            status=STATUS_DISABLED,
            rejected_share=0.0,
            sample=int(ceiling.size),
            observed=stats,
            message=f"Próg {value:g} nie deklaruje bramki, więc deduplikacja nie działa.",
        )

    reached = float(np.mean(ceiling >= value))
    if reached <= 0.0:
        return ThresholdVerdict(
            name=name,
            value=value,
            space=space,
            status=STATUS_UNREACHABLE,
            rejected_share=0.0,
            sample=int(ceiling.size),
            observed=stats,
            message=(
                f"Próg {value:g} jest nieosiągalny: nawet tekst identyczny z zapisanym "
                f"sięga najwyżej {stats['max']:.3f}. Deduplikacja nie odrzuci nigdy "
                "niczego. Sprawdź, czy porównanie nie miesza przestrzeni zapytania "
                "i dokumentu."
            ),
        )

    false_positive = float(np.mean(floor >= value)) if floor.size else 0.0
    if false_positive > DUPLICATE_FALSE_POSITIVE_LIMIT:
        return ThresholdVerdict(
            name=name,
            value=value,
            space=space,
            status=STATUS_OVER_BROAD,
            rejected_share=round(false_positive, 4),
            sample=int(ceiling.size),
            observed=stats,
            message=(
                f"Próg {value:g} przepuszcza powtórzenie w {reached:.0%} przypadków, ale "
                f"{false_positive:.1%} par *różnych* dokumentów też go przekracza "
                f"(p99 szumu {percentiles(floor)['p99']:.3f}). Odrzuci też treść nową."
            ),
        )

    return ThresholdVerdict(
        name=name,
        value=value,
        space=space,
        status=STATUS_WORKING,
        rejected_share=round(false_positive, 4),
        sample=int(ceiling.size),
        observed=stats,
        message=(
            f"Próg {value:g} jest osiągalny: identyczny tekst sięga {stats['max']:.3f}. "
            f"Różne dokumenty przekraczają go w {false_positive:.1%} par, więc rozdziela "
            "powtórzenie od treści nowej."
        ),
    )


def classify_reconstruction(*, identical: Any) -> ThresholdVerdict:
    """Czy zapisane wektory zgadzają się ze schematem, który deklarują.

    Ten pomiar nie dotyczy żadnego progu, ale wychodzi z tych samych danych i bez
    niego werdykt o deduplikacji byłby mylący. Dokument odtworzony z payloadu tym
    samym szablonem i przepuszczony przez ten sam model musi wrócić do własnego
    wektora na 1.000. Jeśli nie wraca, wektor powstał inaczej, niż mówią metadane —
    i wtedy deduplikacja nie rozpozna powtórzenia nie z winy progu, a z winy danych.
    """

    scores = np.asarray(list(identical), dtype=float)
    stats = percentiles(scores)
    name = "memory_document_schema"
    space = "odtworzenie dokumentu z payloadu"

    if scores.size == 0:
        return unmeasured_verdict(
            name=name,
            value=EXACT_RECONSTRUCTION,
            space=space,
            reason="Nie udało się odtworzyć ani jednego dokumentu.",
        )

    exact = float(np.mean(scores >= EXACT_RECONSTRUCTION))
    if exact >= RECONSTRUCTION_EXPECTED_SHARE:
        return ThresholdVerdict(
            name=name,
            value=EXACT_RECONSTRUCTION,
            space=space,
            status=STATUS_WORKING,
            rejected_share=round(1.0 - exact, 4),
            sample=int(scores.size),
            observed=stats,
            message=(
                f"{exact:.0%} punktów odtwarza się dokładnie, więc zapisane wektory "
                "zgadzają się ze schematem z metadanych."
            ),
        )

    return ThresholdVerdict(
        name=name,
        value=EXACT_RECONSTRUCTION,
        space=space,
        status=STATUS_DRIFTED,
        rejected_share=round(1.0 - exact, 4),
        sample=int(scores.size),
        observed=stats,
        message=(
            f"Tylko {exact:.0%} punktów odtwarza się dokładnie; najsłabsze wraca na "
            f"{stats['min']:.3f} (mediana {stats['p50']:.3f}). Pozostałe wektory powstały "
            "inaczej, niż mówi ich pole `schema`, więc deduplikacja ich nie rozpozna. "
            "To defekt danych, nie progu."
        ),
    )


async def scroll_points(
    client: Any,
    collection: str,
    *,
    limit: int,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Odczyt punktów z kolekcji. Tylko `scroll`, żadnego zapisu."""

    collected: list[dict[str, Any]] = []
    offset = None
    while len(collected) < limit:
        batch, offset = await client.scroll(
            collection_name=collection,
            limit=min(256, limit - len(collected)),
            offset=offset,
            with_payload=True,
            with_vectors=["semantic"] if with_vectors else False,
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload or {}
            metadata = payload.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            observations = metadata.get("observations")
            vectors = point.vector if with_vectors else None
            collected.append(
                {
                    "id": str(point.id),
                    "source": str(payload.get("source") or ""),
                    "content": str(payload.get("content") or ""),
                    "title": str(payload.get("title") or ""),
                    "schema": str(metadata.get("schema") or "(brak)"),
                    "profile": str(metadata.get("vector_profile") or "(brak)"),
                    "observations": [
                        str(item) for item in observations or [] if str(item).strip()
                    ],
                    "vector": (
                        vectors.get("semantic") if isinstance(vectors, dict) else None
                    ),
                }
            )
        if offset is None:
            break
    return collected


async def measure_axis_scores(
    *,
    client: Any,
    collection: str,
    embeddings: Any,
    probes: list[dict[str, Any]],
    depth: int,
) -> dict[str, AxisScores]:
    """Rozkład wyników per oś, zebrany bez progu.

    Zapytania buduje produkcyjna `memory_query_documents`, żeby mierzyć tę samą
    geometrię, której używa wyszukiwanie. Odpytujemy z `score_threshold=None`,
    bo próg zastosowany w trakcie pomiaru progu byłby błędem cyrkularnym.

    Zapytanie wywiedzione z treści dokumentu jest łatwiejsze niż prawdziwe
    pytanie użytkownika, więc `self_scores` są górnym oszacowaniem. Rozkład
    `scores` pochodzi z realnych wektorów i jest prawdziwy.
    """

    result = {name: AxisScores(axis=name) for name in MEMORY_VECTOR_NAMES}
    for probe in probes:
        documents = memory_query_documents(str(probe.get("content") or "")[:2000])
        names = [name for name in MEMORY_VECTOR_NAMES if documents.get(name)]
        if not names:
            continue
        vectors = await embeddings.embed_queries([documents[name] for name in names])
        if len(vectors) != len(names):
            continue
        for name, vector in zip(names, vectors, strict=True):
            axis = result[name]
            try:
                response = await client.query_points(
                    collection_name=collection,
                    query=list(vector),
                    using=name,
                    limit=depth,
                    score_threshold=None,
                    with_payload=False,
                    with_vectors=False,
                )
            except Exception as exc:  # noqa: BLE001
                # Oś, która nie istnieje w kolekcji, to informacja, nie awaria —
                # ale musi być zapisana, nie zjedzona.
                axis.error = type(exc).__name__
                continue
            found_self = False
            for rank, point in enumerate(response.points, start=1):
                score = float(point.score)
                axis.scores.append(score)
                if str(point.id) == str(probe.get("id")):
                    axis.self_scores.append(score)
                    axis.self_ranks.append(rank)
                    found_self = True
            if not found_self:
                axis.not_found += 1
    return result


async def measure_document_neighbourhood(
    *,
    client: Any,
    collection: str,
    embeddings: Any,
    documents: list[tuple[str, str]],
    depth: int = 10,
) -> tuple[list[float], list[float]]:
    """Podobieństwo dokumentu do własnego wektora i do najbliższego obcego.

    `documents` to pary (identyfikator punktu, dokument semantyczny odtworzony
    produkcyjnym szablonem). Wektoryzujemy je jako dokumenty, nie zapytania, bo
    tak robi to deduplikacja po naprawie — a próg mierzy się tam, gdzie działa.

    Zwraca (identyczne, różne): pierwszy rozkład to sufit osiągalności progu,
    drugi to podłoga, nad którą próg musi leżeć, żeby nie odrzucać treści nowej.
    """

    identical: list[float] = []
    distinct: list[float] = []
    if not documents:
        return identical, distinct

    vectors = await embeddings.embed_documents([text for _, text in documents])
    if len(vectors) != len(documents):
        return identical, distinct

    for (point_id, _), vector in zip(documents, vectors, strict=True):
        try:
            response = await client.query_points(
                collection_name=collection,
                query=list(vector),
                using="semantic",
                limit=depth,
                score_threshold=None,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:  # noqa: BLE001
            continue
        for point in response.points:
            score = float(point.score)
            if str(point.id) == str(point_id):
                identical.append(score)
            else:
                distinct.append(score)
    return identical, distinct
