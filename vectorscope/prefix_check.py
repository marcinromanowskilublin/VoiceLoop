"""Pomiar skutku niespójnych prefiksów zadania w pamięci Screenpipe.

Nowa ścieżka indeksowania woła `embed_documents` (prefiks `search_document: `),
stara `embed_texts` (bez prefiksu, screenpipe_memory.py:230), a zapytania idą
zawsze przez `embed_queries` (`search_query: `). Ten moduł mierzy, ile realnie
kosztuje to rozjechanie.

Uwaga metodologiczna, bo pierwsza wersja tego modułu myliła się właśnie tutaj:
**średniego cosinusa nie wolno porównywać między konwencjami prefiksów**. Każdy
prefiks przesuwa całą chmurę wektorów w inny rejon przestrzeni, więc konwencja
o niższym średnim cosinusie może mieć lepszy retrieval i odwrotnie. Liczy się
kolejność wyników, nie ich bezwzględna wysokość. Dlatego podstawowymi miarami
są tu trafienie w pierwszym wyniku, MRR i margines nad najlepszym dystraktorem —
wszystkie odporne na przesunięcie skali.

Nic tutaj nie modyfikuje pamięci VoiceLoopa. To wyłącznie pomiar.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Probe:
    query: str
    document: str


@dataclass(frozen=True)
class Convention:
    key: str
    label: str
    query_prefix: str
    document_prefix: str
    note: str


CONVENTIONS: tuple[Convention, ...] = (
    Convention(
        key="official",
        label="Konwencja nomica",
        query_prefix=PREFIX_QUERY,
        document_prefix=PREFIX_DOCUMENT,
        note="To, czego wymaga karta modelu i co robi nowa ścieżka VoiceLoopa.",
    ),
    Convention(
        key="legacy_raw",
        label="Stara ścieżka Screenpipe",
        query_prefix=PREFIX_QUERY,
        document_prefix=PREFIX_NONE,
        note="Dokumenty zapisane przez embed_texts, bez prefiksu.",
    ),
    Convention(
        key="both_query",
        label="Wszystko jako zapytanie",
        query_prefix=PREFIX_QUERY,
        document_prefix=PREFIX_QUERY,
        note="Przypadek kontrolny: co się dzieje, gdy prefiks jest jeden dla wszystkiego.",
    ),
    Convention(
        key="no_prefix",
        label="Bez prefiksów w ogóle",
        query_prefix=PREFIX_NONE,
        document_prefix=PREFIX_NONE,
        note="Przypadek kontrolny: model używany wbrew karcie modelu.",
    ),
)


DEFAULT_PROBES: tuple[Probe, ...] = (
    Probe(
        query="co robiłem w Cursorze",
        document=(
            "Temat zapamiętanej informacji:\n"
            "Praca w edytorze Cursor nad projektem VoiceLoop, "
            "przeglądanie kodu i uruchamianie testów."
        ),
    ),
    Probe(
        query="o czym była rozmowa z pacjentem",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Rozmowa z pacjentem o nasileniu lęku i skuteczności leczenia."
        ),
    ),
    Probe(
        query="jaka była decyzja co do dawki",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Ustalono zwiększenie dawki i kontrolę za dwa tygodnie."
        ),
    ),
    Probe(
        query="co ustaliliśmy na spotkaniu zespołu",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Zespół ustalił termin wdrożenia i podział zadań na sprint."
        ),
    ),
    Probe(
        query="kto dzwonił w sprawie recepty",
        document=(
            "Jawny kontekst osoby lub relacji:\n"
            "Telefon od pacjentki w sprawie przedłużenia recepty."
        ),
    ),
    Probe(
        query="jakie strony przeglądałem w przeglądarce",
        document=(
            "Cel lub intencja zaobserwowanej aktywności:\n"
            "Przeglądanie dokumentacji technicznej w Chrome i szukanie rozwiązania błędu."
        ),
    ),
    Probe(
        query="kiedy mam następną wizytę",
        document=(
            "Temat zapamiętanej informacji:\n"
            "Kolejna wizyta kontrolna zaplanowana na piątek rano."
        ),
    ),
    Probe(
        query="co z fakturą za marzec",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Faktura za marzec nieopłacona, trzeba uregulować rachunek w tym tygodniu."
        ),
    ),
    Probe(
        query="czy pacjent skarżył się na sen",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Pacjent zgłasza bezsenność utrzymującą się od tygodnia."
        ),
    ),
    Probe(
        query="jak poszło wdrożenie na produkcję",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Wdrożenie na produkcję przebiegło bez błędów, monitoring czysty."
        ),
    ),
    Probe(
        query="co trzeba kupić do domu",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Lista zakupów: chleb, mleko i środek do prania."
        ),
    ),
    Probe(
        query="jaki był wynik badania krwi",
        document=(
            "Temat zapamiętanej informacji:\n"
            "Wyniki morfologii w normie, poziom żelaza lekko obniżony."
        ),
    ),
    Probe(
        query="o czym pisał klient w mailu",
        document=(
            "Cel lub intencja zaobserwowanej aktywności:\n"
            "Klient prosi w mailu o wycenę rozszerzenia zakresu prac."
        ),
    ),
    Probe(
        query="kiedy jedziemy na urlop",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Wyjazd zaplanowany na drugi tydzień lipca, nocleg zarezerwowany."
        ),
    ),
    Probe(
        query="co się zepsuło w samochodzie",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Awaria sprzęgła, samochód oddany do warsztatu na trzy dni."
        ),
    ),
    Probe(
        query="jakie leki bierze pacjentka",
        document=(
            "Jawny kontekst osoby lub relacji:\n"
            "Pacjentka przyjmuje lek przeciwdepresyjny i preparat żelaza."
        ),
    ),
    Probe(
        query="co ustaliliśmy z księgową",
        document=(
            "Jawny kontekst osoby lub relacji:\n"
            "Księgowa poprosiła o komplet dokumentów do końca miesiąca."
        ),
    ),
    Probe(
        query="jaki błąd wyrzucał program",
        document=(
            "Cel lub intencja zaobserwowanej aktywności:\n"
            "Aplikacja zwracała błąd połączenia z bazą przy starcie."
        ),
    ),
    Probe(
        query="czy zapłaciłem za prąd",
        document=(
            "Temat zapamiętanej informacji:\n"
            "Rachunek za energię elektryczną opłacony przelewem w poniedziałek."
        ),
    ),
    Probe(
        query="co mówił lekarz o wynikach",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Lekarz ocenił wyniki jako stabilne i nie zmienił leczenia."
        ),
    ),
    Probe(
        query="kiedy oddać sprawozdanie",
        document=(
            "Jawna decyzja, ustalenie lub następny krok:\n"
            "Termin złożenia sprawozdania mija piętnastego przyszłego miesiąca."
        ),
    ),
    Probe(
        query="jak przebiegła rozmowa kwalifikacyjna",
        document=(
            "Znaczenie zdarzenia lub zapamiętanej informacji:\n"
            "Rozmowa kwalifikacyjna wypadła dobrze, decyzja w przyszłym tygodniu."
        ),
    ),
    Probe(
        query="co ćwiczyłem na treningu",
        document=(
            "Cel lub intencja zaobserwowanej aktywności:\n"
            "Trening siłowy: przysiady, martwy ciąg i wiosłowanie."
        ),
    ),
    Probe(
        query="gdzie zaparkowałem samochód",
        document=(
            "Temat zapamiętanej informacji:\n"
            "Samochód zostawiony na parkingu podziemnym, poziom minus dwa."
        ),
    ),
)


async def run_prefix_check(
    *,
    probes: tuple[Probe, ...] = DEFAULT_PROBES,
    min_score: float | None = None,
) -> dict[str, Any]:
    active = settings()
    threshold = min_score if min_score is not None else active.vector_memory_min_score
    client = build_embedding_client(active)

    queries = [probe.query for probe in probes]
    documents = [probe.document for probe in probes]

    needed_query = {convention.query_prefix for convention in CONVENTIONS}
    needed_document = {convention.document_prefix for convention in CONVENTIONS}

    try:
        query_vectors = {
            prefix: geometry.l2_normalize(
                (await embed_texts_with_prefix(client, queries, prefix=prefix)).vectors
            )
            for prefix in sorted(needed_query)
        }
        document_runs = {
            prefix: await embed_texts_with_prefix(client, documents, prefix=prefix)
            for prefix in sorted(needed_document)
        }
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    document_vectors = {
        prefix: geometry.l2_normalize(run.vectors) for prefix, run in document_runs.items()
    }
    reference = document_runs[PREFIX_DOCUMENT]

    results = [
        _score_convention(
            convention,
            query_vectors[convention.query_prefix],
            document_vectors[convention.document_prefix],
            threshold=threshold,
        )
        for convention in CONVENTIONS
    ]
    by_key = {entry["key"]: entry for entry in results}

    displacement = _displacement(document_vectors)
    injection = _prefix_injection(document_vectors)

    return {
        "ok": True,
        "threshold": threshold,
        "model": reference.model,
        "dimension": reference.dimension,
        "probe_count": len(probes),
        "conventions": results,
        "displacement": displacement,
        "prefix_injection": injection,
        "method_note": (
            "Bezwzględne cosinusy są porównywalne tylko wewnątrz jednej konwencji. "
            "Prefiks przesuwa całą chmurę wektorów, więc między konwencjami "
            "porównujemy trafienia i MRR, a nie wysokość liczby."
        ),
        "interpretation": _interpret(
            by_key=by_key,
            displacement=displacement,
            injection=injection,
            probe_count=len(probes),
        ),
    }


def _score_convention(
    convention: Convention,
    queries: np.ndarray,
    documents: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Metryki rankingowe: odporne na to, że prefiks przesuwa całą skalę."""

    similarity = queries @ documents.T
    count = similarity.shape[0]
    correct = np.diag(similarity).copy()

    distractors = similarity.copy()
    np.fill_diagonal(distractors, -np.inf)
    best_distractor = distractors.max(axis=1)
    margin = correct - best_distractor

    ranks = 1 + (similarity > correct[:, None]).sum(axis=1)

    return {
        "key": convention.key,
        "label": convention.label,
        "note": convention.note,
        "query_prefix": convention.query_prefix,
        "document_prefix": convention.document_prefix,
        "hits_at_1": int(np.sum(ranks == 1)),
        "hit_at_1": float(np.mean(ranks == 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "worst_rank": int(ranks.max()),
        "margin": _distribution(margin),
        "correct_pair_cosine": _distribution(correct),
        "above_threshold": int(np.sum(correct >= threshold)),
        "of_total": count,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    """Mediana i kwartyle. Na kilkudziesięciu sondach średnia z czterema
    miejscami po przecinku sugerowałaby precyzję, której nie ma."""

    return {
        "median": round(float(np.median(values)), 3),
        "q1": round(float(np.percentile(values, 25)), 3),
        "q3": round(float(np.percentile(values, 75)), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def _displacement(document_vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Jak bardzo sam prefiks przesuwa wektor tego samego tekstu."""

    prefixed = document_vectors[PREFIX_DOCUMENT]
    bare = document_vectors[PREFIX_NONE]
    same_text = np.sum(prefixed * bare, axis=1)

    return {
        "same_text_across_prefixes": _distribution(same_text),
        "mean_within_document_prefix": _mean_offdiagonal(prefixed @ prefixed.T),
        "mean_within_no_prefix": _mean_offdiagonal(bare @ bare.T),
    }


def _prefix_injection(document_vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Czy serwer sam dokleja prefiks, kiedy go nie podamy?

    Gdyby LM Studio doklejało prefiks po cichu, tekst wysłany bez prefiksu
    miałby z którymś z wariantów cosinus bliski 1.000 i wszystkie pozostałe
    pomiary porównywałyby to samo ze sobą.
    """

    bare = document_vectors[PREFIX_NONE]
    scores = {
        prefix: round(float(np.median(np.sum(bare * vectors, axis=1))), 3)
        for prefix, vectors in document_vectors.items()
        if prefix != PREFIX_NONE
    }
    suspected = [prefix for prefix, value in scores.items() if value > 0.99]
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
            "tak, jakby były zapytaniami. Stąd biorą się pozornie lepsze cosinusy "
            "tej ścieżki: porównuje zapytanie z zapytaniem, a nie z dokumentem."
        )

    return {
        "median_similarity_to_prefixed": scores,
        "suspected_injection": suspected,
        "nearest_prefixed_variant": nearest,
        "verdict": verdict,
        "lean": lean,
    }


def _mean_offdiagonal(matrix: np.ndarray) -> float:
    count = matrix.shape[0]
    if count < 2:
        return 0.0
    upper = np.triu_indices(count, 1)
    return round(float(np.mean(matrix[upper])), 3)


def _interpret(
    *,
    by_key: dict[str, dict[str, Any]],
    displacement: dict[str, Any],
    injection: dict[str, Any],
    probe_count: int,
) -> list[str]:
    notes: list[str] = [injection["verdict"]]
    if injection.get("lean"):
        notes.append(injection["lean"])

    identity = displacement["same_text_across_prefixes"]["median"]
    if identity < 0.99:
        notes.append(
            f"Ten sam tekst pod dwoma prefiksami ma cosinus {identity:.3f}, nie 1.000 — "
            "prefiks realnie przesuwa wektor i obie konwencje nie są wymienne."
        )

    official = by_key["official"]
    legacy = by_key["legacy_raw"]
    difference = official["hits_at_1"] - legacy["hits_at_1"]
    notes.append(
        f"Trafienie w pierwszym wyniku: konwencja nomica {official['hits_at_1']}/{probe_count}, "
        f"stara ścieżka Screenpipe {legacy['hits_at_1']}/{probe_count}."
    )

    # Próg szumu. Przy kilkudziesięciu sondach różnica jednej sondy to nie
    # sygnał, tylko przypadek — a poprzednia wersja tego modułu ogłaszała
    # na takiej różnicy zwycięstwo konwencji.
    if abs(difference) <= max(1, probe_count // 20):
        notes.append(
            f"Różnica {abs(difference)} sondy przy {probe_count} próbach mieści się "
            "w szumie. Na tym zbiorze konwencje działają tak samo dobrze i nic tu "
            "nie uzasadnia migracji w żadną stronę."
        )
    elif difference > 0:
        notes.append(
            "Poprawny prefiks wygrywa na rankingu, mimo że jego bezwzględne cosinusy "
            "bywają niższe. To jest właśnie powód, dla którego nie wolno porównywać "
            "konwencji po średnim cosinusie."
        )
    else:
        notes.append(
            "Stara ścieżka wypada lepiej na rankingu, co przeczy karcie modelu. "
            "Zanim cokolwiek zmienisz w VoiceLoopie, powtórz pomiar na własnym korpusie."
        )

    within_prefixed = displacement["mean_within_document_prefix"]
    within_bare = displacement["mean_within_no_prefix"]
    if within_prefixed - within_bare > 0.05:
        notes.append(
            f"Prefiks podnosi wzajemne podobieństwo niepowiązanych dokumentów z "
            f"{within_bare:.3f} do {within_prefixed:.3f}. Dokłada wspólny kierunek, "
            "czyli ściska realny zakres skali."
        )

    return notes
