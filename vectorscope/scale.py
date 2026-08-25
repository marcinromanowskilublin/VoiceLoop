"""Dno skali modelu wyznaczone z rozkładu, a nie z jednej pary.

Kotwice z `anchors.py` są dobre do pokazania, *co* znaczy dana wysokość
cosinusa. Nie nadają się jednak do orzekania, gdzie leży dno skali, bo
pojedyncza para („lęk" ↔ „silnik wysokoprężny") to próba o liczności jeden.
Werdykt o progu odcięcia oparty na takiej próbie jest efektowny i bezwartościowy.

Ten moduł liczy dno inaczej: bierze korpus zdań z ośmiu wyraźnie rozłącznych
dziedzin i traktuje każdą parę z *różnych* dziedzin jako niepowiązaną. Daje to
ponad tysiąc par i pozwala mówić percentylami zamiast anegdotą.

Wynik zależy od prefiksu, i to mocno — `search_document: ` dokłada wszystkim
wektorom wspólny kierunek, przez co podnosi dno. Dlatego pomiar zawsze jest
raportowany razem z prefiksem, którym go wykonano.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from voiceloop.embeddings import EmbeddingUnavailableError

from . import geometry
from .config import PREFIX_DOCUMENT, PREFIX_QUERY, build_embedding_client, settings
from .embed import embed_texts_with_prefix

DOMAIN_CORPUS: dict[str, tuple[str, ...]] = {
    "zdrowie psychiczne": (
        "Pacjent zgłasza narastający lęk przed wyjściem z domu.",
        "Bezsenność utrzymuje się od kilku tygodni mimo leczenia.",
        "Ustaliliśmy zwiększenie dawki leku przeciwdepresyjnego.",
        "Chory opisuje napady paniki w miejscach publicznych.",
        "Terapia poznawczo-behawioralna przynosi powolną poprawę.",
        "Nastrój obniżony, apetyt zachowany, myśli rezygnacyjne nieobecne.",
    ),
    "programowanie": (
        "Aplikacja zwraca błąd połączenia z bazą danych przy starcie.",
        "Testy jednostkowe przechodzą, ale integracyjne wywalają się na timeout.",
        "Refaktoryzacja modułu autoryzacji zajmie około trzech dni.",
        "Wdrożenie na produkcję przebiegło bez błędów, monitoring czysty.",
        "Trzeba zaktualizować zależności, bo biblioteka ma lukę bezpieczeństwa.",
        "Kod przeglądu wymaga poprawek w obsłudze wyjątków.",
    ),
    "finanse": (
        "Faktura za marzec nie została jeszcze opłacona.",
        "Księgowa poprosiła o komplet dokumentów do końca miesiąca.",
        "Przelew za energię elektryczną poszedł w poniedziałek.",
        "Rozliczenie kwartalne trzeba złożyć do piętnastego.",
        "Koszty utrzymania serwera wzrosły o jedną piątą.",
        "Bank odrzucił wniosek o podwyższenie limitu na karcie.",
    ),
    "kuchnia": (
        "Ciasto drożdżowe musi wyrastać w ciepłym miejscu przez godzinę.",
        "Do rosołu dodaję marchew, pietruszkę i kawałek selera.",
        "Piekarnik rozgrzej do stu osiemdziesięciu stopni.",
        "Mięso trzeba zamarynować dzień wcześniej.",
        "Zupa wyszła za słona, dolałem wody i dodałem ziemniaka.",
        "Chleb na zakwasie piekę w garnku żeliwnym.",
    ),
    "motoryzacja": (
        "Awaria sprzęgła, samochód oddany do warsztatu na trzy dni.",
        "Trzeba wymienić opony na letnie przed dłuższą trasą.",
        "Silnik zaczyna głośno pracować na zimnym rozruchu.",
        "Przegląd techniczny wygasa w przyszłym miesiącu.",
        "Zaparkowałem na poziomie minus dwa parkingu podziemnego.",
        "Zużycie paliwa wzrosło po ostatniej naprawie.",
    ),
    "pogoda i przyroda": (
        "Nad ranem spadł gęsty śnieg i drogi zrobiły się śliskie.",
        "Wiatr od morza przybiera na sile po południu.",
        "Lipiec był najcieplejszym miesiącem od lat.",
        "Na łące zakwitły maki i chabry.",
        "Rzeka wystąpiła z brzegów po tygodniu opadów.",
        "Mgła utrzymywała się w dolinie aż do południa.",
    ),
    "sport": (
        "Trening siłowy obejmował przysiady i martwy ciąg.",
        "Drużyna przegrała mecz różnicą jednego gola.",
        "Przebiegłem dziesięć kilometrów w niecałą godzinę.",
        "Kontuzja kolana wykluczyła go z sezonu.",
        "Rozgrzewka przed biegiem trwa kwadrans.",
        "Zawodnik poprawił swój rekord życiowy.",
    ),
    "urzędy i prawo": (
        "Wniosek o wydanie dowodu osobistego złożyłem w urzędzie miasta.",
        "Termin odwołania od decyzji mija za czternaście dni.",
        "Umowa najmu wymaga formy pisemnej pod rygorem nieważności.",
        "Sprawa spadkowa toczy się przed sądem rejonowym.",
        "Potrzebny jest odpis aktu urodzenia z urzędu stanu cywilnego.",
        "Pełnomocnictwo musi być poświadczone notarialnie.",
    ),
}


async def measure_scale(
    *,
    prefix: str = PREFIX_DOCUMENT,
    corpus: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Rozkład cosinusów par z różnych dziedzin (dno) i z tej samej (sygnał)."""

    active = settings()
    client = build_embedding_client(active)
    domains = corpus if corpus is not None else DOMAIN_CORPUS

    texts: list[str] = []
    labels: list[str] = []
    for domain, sentences in domains.items():
        for sentence in sentences:
            texts.append(sentence)
            labels.append(domain)

    try:
        result = await embed_texts_with_prefix(client, texts, prefix=prefix)
        query_side = await embed_texts_with_prefix(client, texts, prefix=PREFIX_QUERY)
        document_side = await embed_texts_with_prefix(client, texts, prefix=PREFIX_DOCUMENT)
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    similarity = geometry.cosine_matrix(result.vectors)
    upper = np.triu_indices(len(texts), 1)
    values = similarity[upper]

    domain_array = np.array(labels)
    same_domain = domain_array[upper[0]] == domain_array[upper[1]]
    unrelated = values[~same_domain]
    related = values[same_domain]

    floor = _percentiles(unrelated)
    signal = _percentiles(related)
    overlap = float(np.mean(related <= floor["p95"]))

    retrieval = _retrieval_view(
        queries=geometry.l2_normalize(query_side.vectors),
        documents=geometry.l2_normalize(document_side.vectors),
        domains=domain_array,
    )

    # Progi konfrontujemy z rozkładem retrievalowym, bo to ten cosinus realnie
    # trafia na próg w VoiceLoopie — nie cosinus dwóch dokumentów.
    thresholds = [
        {
            "key": item.key,
            "value": item.value,
            "label": item.label,
            "origin": item.origin,
            "position": _position(item.value, retrieval["floor"]),
        }
        for item in _relevant_thresholds(active)
    ]

    return {
        "ok": True,
        "prefix": prefix,
        "model": result.model,
        "dimension": result.dimension,
        "domain_count": len(domains),
        "text_count": len(texts),
        "unrelated_pairs": int(unrelated.size),
        "related_pairs": int(related.size),
        "floor": floor,
        "signal": signal,
        "separation": round(float(signal["p50"] - floor["p50"]), 3),
        "signal_below_floor_p95": round(overlap, 3),
        "retrieval": retrieval,
        "recommended_min_score": retrieval["floor"]["p95"],
        "thresholds": thresholds,
        "interpretation": _interpret(
            floor=floor,
            signal=signal,
            overlap=overlap,
            retrieval=retrieval,
            thresholds=thresholds,
            unrelated_count=int(unrelated.size),
            prefix=prefix,
        ),
    }


def _retrieval_view(
    *,
    queries: np.ndarray,
    documents: np.ndarray,
    domains: np.ndarray,
) -> dict[str, Any]:
    """Rozkład cosinusów tak, jak powstają w retrievalu VoiceLoopa.

    Zapytanie idzie przez `embed_queries`, dokument przez `embed_documents`,
    więc operacyjny cosinus jest międzyprefiksowy. Rozkład par dokument–dokument
    leży zupełnie gdzie indziej i nie wolno na nim ustawiać progu.
    """

    similarity = queries @ documents.T
    count = similarity.shape[0]

    identity = np.diag(similarity).copy()
    off_diagonal = ~np.eye(count, dtype=bool)
    same_domain = (domains[:, None] == domains[None, :]) & off_diagonal
    cross_domain = (domains[:, None] != domains[None, :]) & off_diagonal

    related = similarity[same_domain]
    unrelated = similarity[cross_domain]
    floor = _percentiles(unrelated)

    return {
        "pairs_unrelated": int(unrelated.size),
        "pairs_related": int(related.size),
        "identical_text": _percentiles(identity),
        "floor": floor,
        "signal": _percentiles(related),
        "separation": round(float(np.median(related) - np.median(unrelated)), 3),
        "signal_below_floor_p95": round(float(np.mean(related <= floor["p95"])), 3),
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p05", "p50", "p90", "p95", "p99", "max")}
    return {
        "min": round(float(values.min()), 3),
        "p05": round(float(np.percentile(values, 5)), 3),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "p99": round(float(np.percentile(values, 99)), 3),
        "max": round(float(values.max()), 3),
    }


def _relevant_thresholds(active: Any) -> list[Any]:
    from .config import collect_thresholds

    interesting = {
        "vector_memory_min_score",
        "screenpipe_duplicate_min_score",
        "screenpipe_related_history_min_score",
        "capability_match_min_score",
    }
    return [item for item in collect_thresholds(active) if item.key in interesting]


def _position(value: float, floor: dict[str, float]) -> str:
    if value < floor["min"]:
        return "ponizej_dna"
    if value < floor["p95"]:
        return "wewnatrz_szumu"
    if value < floor["max"]:
        return "na_krawedzi_szumu"
    return "powyzej_szumu"


def _interpret(
    *,
    floor: dict[str, float],
    signal: dict[str, float],
    overlap: float,
    retrieval: dict[str, Any],
    thresholds: list[dict[str, Any]],
    unrelated_count: int,
    prefix: str,
) -> list[str]:
    retrieval_floor = retrieval["floor"]
    retrieval_signal = retrieval["signal"]
    notes = [
        f"Rozkład operacyjny (zapytanie kontra dokument, {retrieval['pairs_unrelated']} par "
        f"z różnych dziedzin): dno ma medianę {retrieval_floor['p50']:.3f}, "
        f"95. percentyl {retrieval_floor['p95']:.3f}, maksimum {retrieval_floor['max']:.3f}. "
        f"Ten sam tekst po obu stronach daje {retrieval['identical_text']['p50']:.3f}.",
        f"Pary z tej samej dziedziny mają medianę {retrieval_signal['p50']:.3f}, czyli o "
        f"{retrieval['separation']:.3f} wyżej niż szum. To jest cały zapas, jakim "
        "dysponuje retrieval na tym modelu.",
        f"Dla porównania rozkład dokument–dokument (prefiks {prefix}) leży zupełnie "
        f"gdzie indziej: dno {floor['p50']:.3f}, sygnał {signal['p50']:.3f}. Progu nie "
        f"wolno kalibrować na tych liczbach, bo retrieval ich nigdy nie ogląda.",
    ]

    dead = [item for item in thresholds if item["position"] == "ponizej_dna"]
    noisy = [item for item in thresholds if item["position"] == "wewnatrz_szumu"]

    for item in dead:
        notes.append(
            f"Próg {item['key']} = {item['value']} leży poniżej najniższej pary "
            f"niepowiązanej ({retrieval_floor['min']:.3f}). Nie odrzuca niczego — "
            "przechodzi przez niego dowolny tekst."
        )
    for item in noisy:
        notes.append(
            f"Próg {item['key']} = {item['value']} wpada w rozkład par niepowiązanych, "
            f"którego 95. percentyl to {retrieval_floor['p95']:.3f}. Przepuszcza więc "
            "znaczną część treści bez związku."
        )

    cost = retrieval["signal_below_floor_p95"]
    notes.append(
        f"Gdyby ustawić odcięcie na 95. percentylu szumu ({retrieval_floor['p95']:.3f}), "
        f"odpadłoby {cost:.0%} par faktycznie powiązanych. To jest realny koszt takiego "
        "progu i trzeba go rozważyć razem z korzyścią."
    )
    if cost > 0.5:
        notes.append(
            "Koszt przekracza połowę trafnych par, więc na tym modelu sam próg cosinusa "
            "nie rozdzieli sygnału od szumu. Sensowniejsze jest ograniczanie liczby "
            "wyników i sortowanie, a nie odcinanie wartością."
        )
    return notes
