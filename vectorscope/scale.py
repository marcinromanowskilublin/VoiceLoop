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
from voiceloop.memory_vectorization import (
    MEMORY_DOCUMENT_SCHEMA_VERSION,
    MEMORY_QUERY_DOCUMENTS_VERSION,
    MEMORY_VECTOR_NAMES,
    MEMORY_VECTOR_WEIGHTS,
    memory_query_documents,
    memory_vector_documents,
)

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

    # Świadomie nie orzekamy tu o progach. Ten rozkład powstał na gołych zdaniach,
    # a każdy próg VoiceLoopa porównuje teksty owinięte własnymi szablonami, więc
    # leży w innym rejonie skali. Werdykty wydaje measure_threshold_reachability,
    # które mierzy każdy próg w jego własnej przestrzeni.
    thresholds = [
        {
            "key": item.key,
            "value": item.value,
            "label": item.label,
            "origin": item.origin,
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
    threshold: float | None = None,
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

    # Puste grupy zdarzają się przy małym albo jednorodnym korpusie. Bez tych
    # zabezpieczeń numpy zwraca NaN, a `NaN` nie jest poprawnym JSON-em i wywraca
    # panel w miejscu odległym od przyczyny.
    view = {
        "pairs_unrelated": int(unrelated.size),
        "pairs_related": int(related.size),
        "identical_text": _percentiles(identity),
        "floor": floor,
        "signal": _percentiles(related),
        "separation": (
            round(float(np.median(related) - np.median(unrelated)), 3)
            if related.size and unrelated.size
            else 0.0
        ),
        "signal_below_floor_p95": (
            round(float(np.mean(related <= floor["p95"])), 3) if related.size else 0.0
        ),
    }
    if threshold is not None:
        view["noise_above_threshold"] = (
            round(float(np.mean(unrelated >= threshold)), 3) if unrelated.size else 0.0
        )
        view["signal_above_threshold"] = (
            round(float(np.mean(related >= threshold)), 3) if related.size else 0.0
        )
    return view


async def measure_axis_floors(
    *,
    corpus: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Osobny rozkład dla każdej z pięciu osi pamięci VoiceLoopa.

    Powód, dla którego to w ogóle trzeba mierzyć: `vector_memory_min_score` jest
    stosowany do surowego cosinusa **każdej osi z osobna, przed fuzją RRF**
    (qdrant_memory.py: `score_threshold=min_score` w zapytaniu i powtórzony
    warunek przy zbieraniu wyników). Tymczasem każda oś ma własny stały nagłówek
    po stronie dokumentu i inny po stronie zapytania. Nagłówek działa jak prefiks:
    dokłada wspólny kierunek i przesuwa cały rozkład tej osi.

    Jeden próg dla pięciu różnie przesuniętych rozkładów nie może znaczyć tego
    samego w każdym z nich. Ten pomiar sprawdza, jak bardzo się rozjeżdżają.

    Dokumenty i zapytania budujemy produkcyjnymi funkcjami VoiceLoopa, żeby
    pomiar nie mógł się rozminąć z tym, co naprawdę trafia do Qdranta.
    """

    active = settings()
    client = build_embedding_client(active)
    domains = corpus if corpus is not None else DOMAIN_CORPUS

    sentences: list[str] = []
    labels: list[str] = []
    for domain, items in domains.items():
        for sentence in items:
            sentences.append(sentence)
            labels.append(domain)
    domain_array = np.array(labels)

    # Ten sam tekst wypełnia każdy aspekt, więc różnice między osiami biorą się
    # wyłącznie z nagłówków, a nie z innej treści.
    document_texts: dict[str, list[str]] = {name: [] for name in MEMORY_VECTOR_NAMES}
    query_texts: dict[str, list[str]] = {name: [] for name in MEMORY_VECTOR_NAMES}
    for sentence in sentences:
        documents = memory_vector_documents(
            summary=sentence,
            topic=sentence,
            intent=sentence,
            decision=sentence,
            person_context=sentence,
            redact=False,
        )
        queries = memory_query_documents(sentence)
        for name in MEMORY_VECTOR_NAMES:
            document_texts[name].append(documents.get(name, sentence))
            query_texts[name].append(queries.get(name, sentence))

    threshold = float(active.vector_memory_min_score)
    axes: list[dict[str, Any]] = []
    try:
        for name in MEMORY_VECTOR_NAMES:
            document_side = await embed_texts_with_prefix(
                client, document_texts[name], prefix=PREFIX_DOCUMENT
            )
            query_side = await embed_texts_with_prefix(
                client, query_texts[name], prefix=PREFIX_QUERY
            )
            view = _retrieval_view(
                queries=geometry.l2_normalize(query_side.vectors),
                documents=geometry.l2_normalize(document_side.vectors),
                domains=domain_array,
                threshold=threshold,
            )
            axes.append(
                {
                    "axis": name,
                    "weight": MEMORY_VECTOR_WEIGHTS[name],
                    "floor": view["floor"],
                    "signal": view["signal"],
                    "identical_text": view["identical_text"],
                    "separation": view["separation"],
                    "noise_share_above_threshold": view["noise_above_threshold"],
                    "signal_share_above_threshold": view["signal_above_threshold"],
                }
            )
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    return {
        "ok": True,
        "threshold": threshold,
        "threshold_key": "vector_memory_min_score",
        "applied": "surowy cosinus każdej osi osobno, przed fuzją RRF",
        "document_format": MEMORY_DOCUMENT_SCHEMA_VERSION,
        "query_format": MEMORY_QUERY_DOCUMENTS_VERSION,
        "sentence_count": len(sentences),
        "axes": axes,
        "interpretation": _interpret_axes(axes, threshold),
    }


def _interpret_axes(axes: list[dict[str, Any]], threshold: float) -> list[str]:
    medians = {item["axis"]: item["floor"]["p50"] for item in axes}
    lowest = min(medians, key=medians.get)
    highest = max(medians, key=medians.get)
    spread = medians[highest] - medians[lowest]

    notes = [
        f"Próg {threshold} jest stosowany do każdej z pięciu osi tak samo, ale osie "
        f"nie leżą w tym samym miejscu skali. Mediana szumu waha się od "
        f"{medians[lowest]:.3f} ({lowest}) do {medians[highest]:.3f} ({highest}), "
        f"czyli o {spread:.3f}."
    ]

    passing = {item["axis"]: item["noise_share_above_threshold"] for item in axes}
    inert = [axis for axis, share in passing.items() if share > 0.95]

    if len(inert) == len(axes):
        notes.append(
            f"We wszystkich pięciu osiach przez próg przechodzi ponad 95% par bez "
            f"związku. Najniższy zmierzony cosinus szumu to "
            f"{min(item['floor']['min'] for item in axes):.3f}, czyli i tak powyżej "
            f"{threshold}. Ten próg nie odrzuca niczego — jest martwy."
        )
    else:
        worst = max(passing, key=passing.get)
        best = min(passing, key=passing.get)
        notes.append(
            f"Przy tym progu przez oś {worst} przechodzi {passing[worst]:.0%} par bez "
            f"związku, a przez oś {best} tylko {passing[best]:.0%}. Ten sam parametr "
            "znaczy więc w każdej osi co innego."
        )
        if inert:
            notes.append(
                f"W osiach {', '.join(inert)} próg przepuszcza praktycznie wszystko, "
                "więc tam jest martwy."
            )

    separations = [item["separation"] for item in axes]
    widest = max(separations)
    if spread > widest:
        notes.append(
            f"Rozstaw między osiami ({spread:.3f}) jest większy niż rozstaw sygnału i "
            f"szumu wewnątrz najlepszej osi ({widest:.3f}). Innymi słowy: pozycja punktu "
            "w skali mówi więcej o tym, którego nagłówka użyto, niż o tym, czy treść "
            "pasuje do pytania."
        )

    if spread > 0.05:
        notes.append(
            "Wniosek architektoniczny: skoro RRF łączy rangi właśnie po to, żeby nie "
            "zależeć od bezwzględnych wartości, to bramka na bezwzględnym cosinusie "
            "przed fuzją działa wbrew tej konstrukcji. Sensowniejszy byłby próg "
            "wyrażony percentylem rozkładu danej osi albo brak progu i sama fuzja."
        )
    return notes


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


async def measure_threshold_reachability(
    *,
    corpus: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Czy próg w ogóle może zadziałać w porównaniu, które naprawdę wykonuje?

    Poprzednia wersja panelu klasyfikowała wszystkie progi względem jednego
    rozkładu. To był błąd: każdy próg porównuje co innego. `vector_memory_min_score`
    zestawia dokument zapytania z dokumentem pamięci, a próg deduplikacji Screenpipe
    zestawia *surową treść* z dokumentem semantycznym, który ma własny nagłówek.
    Te dwa porównania żyją w innych rejonach przestrzeni i wspólna etykieta dla
    obu jest bez wartości.

    Dla każdego progu liczymy dwie rzeczy w jego własnej przestrzeni:

    - sufit: jaki cosinus osiąga treść identyczna po obu stronach,
    - podłogę: jak wysoko leży szum par bez związku.

    Próg powyżej sufitu nigdy nie zadziała. Próg poniżej podłogi nigdy nie
    odrzuci. Oba przypadki to martwy kod, tylko w przeciwne strony.
    """

    active = settings()
    client = build_embedding_client(active)
    domains = corpus if corpus is not None else DOMAIN_CORPUS

    sentences: list[str] = []
    labels: list[str] = []
    for domain, items in domains.items():
        for sentence in items:
            sentences.append(sentence)
            labels.append(domain)
    domain_array = np.array(labels)

    semantic_documents = [
        memory_vector_documents(summary=sentence, redact=False)["semantic"]
        for sentence in sentences
    ]

    try:
        raw_queries = geometry.l2_normalize(
            (await embed_texts_with_prefix(client, sentences, prefix=PREFIX_QUERY)).vectors
        )
        semantic_side = geometry.l2_normalize(
            (
                await embed_texts_with_prefix(
                    client, semantic_documents, prefix=PREFIX_DOCUMENT
                )
            ).vectors
        )
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "message": f"LM Studio niedostępne: {exc}"}

    dedup = _retrieval_view(
        queries=raw_queries,
        documents=semantic_side,
        domains=domain_array,
        threshold=float(getattr(active, "screenpipe_duplicate_min_score", 0.92)),
    )

    axis_report = await measure_axis_floors(corpus=domains)
    if not axis_report.get("ok"):
        return axis_report

    known = {item.key: item for item in _collect(active)}
    entries: list[dict[str, Any]] = []

    memory_axes = axis_report["axes"]
    memory_ceiling = min(item["identical_text"]["p50"] for item in memory_axes)
    memory_floor_min = min(item["floor"]["min"] for item in memory_axes)
    entries.append(
        _reachability_entry(
            known.get("vector_memory_min_score"),
            space=(
                "dokument zapytania kontra dokument pamięci, osobno w każdej z pięciu "
                "osi, przed fuzją RRF"
            ),
            ceiling=memory_ceiling,
            floor_min=memory_floor_min,
            floor_p95=max(item["floor"]["p95"] for item in memory_axes),
        )
    )
    entries.append(
        _reachability_entry(
            known.get("screenpipe_duplicate_min_score"),
            space=(
                "surowa treść przez embed_query kontra wektor semantic zapisanego "
                "dokumentu, który ma własny nagłówek"
            ),
            ceiling=dedup["identical_text"]["max"],
            floor_min=dedup["floor"]["min"],
            floor_p95=dedup["floor"]["p95"],
            ceiling_note=(
                f"mediana dla identycznej treści: {dedup['identical_text']['p50']:.3f}"
            ),
        )
    )

    for key in ("capability_match_min_score", "screenpipe_related_history_min_score"):
        item = known.get(key)
        if item is None:
            continue
        entries.append(
            {
                "key": item.key,
                "label": item.label,
                "value": item.value,
                "origin": item.origin,
                "space": "nieustalone — inny korpus i inne szablony niż zmierzone",
                "verdict": "nie_mierzone",
                "message": (
                    "Ten próg działa na innym zbiorze niż zmierzony, więc panel nie "
                    "orzeka o nim niczego. Zgadywanie byłoby tu gorsze od milczenia."
                ),
            }
        )

    return {
        "ok": True,
        "sentence_count": len(sentences),
        "thresholds": entries,
        "axes": memory_axes,
    }


def _reachability_entry(
    item: Any,
    *,
    space: str,
    ceiling: float,
    floor_min: float,
    floor_p95: float,
    ceiling_note: str | None = None,
) -> dict[str, Any]:
    if item is None:
        return {"verdict": "brak_progu"}

    value = float(item.value)
    if value > ceiling:
        verdict = "nieosiagalny"
        message = (
            f"Nawet identyczna treść po obu stronach osiąga najwyżej {ceiling:.3f}, "
            f"a próg wynosi {value}. Warunek nie może się spełnić nigdy."
        )
    elif value < floor_min:
        verdict = "martwy"
        message = (
            f"Najniższa zmierzona para bez związku ma {floor_min:.3f}, czyli i tak "
            f"powyżej progu {value}. Nie odrzuca niczego."
        )
    elif value < floor_p95:
        verdict = "wewnatrz_szumu"
        message = (
            f"Próg {value} leży wewnątrz rozkładu szumu, którego 95. percentyl to "
            f"{floor_p95:.3f}. Odrzuca część treści bez związku, ale nie większość."
        )
    else:
        verdict = "dziala"
        message = (
            f"Próg {value} leży ponad szumem ({floor_p95:.3f}) i poniżej sufitu "
            f"({ceiling:.3f}), więc realnie rozdziela."
        )

    return {
        "key": item.key,
        "label": item.label,
        "value": value,
        "origin": item.origin,
        "space": space,
        "ceiling": round(float(ceiling), 3),
        "floor_min": round(float(floor_min), 3),
        "floor_p95": round(float(floor_p95), 3),
        "verdict": verdict,
        "message": message + (f" ({ceiling_note})" if ceiling_note else ""),
    }


def _collect(active: Any) -> list[Any]:
    from .config import collect_thresholds

    return list(collect_thresholds(active))


def _relevant_thresholds(active: Any) -> list[Any]:
    interesting = {
        "vector_memory_min_score",
        "screenpipe_duplicate_min_score",
        "screenpipe_related_history_min_score",
        "capability_match_min_score",
    }
    return [item for item in _collect(active) if item.key in interesting]


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

    notes.append(
        "O progach ten pomiar nie orzeka. Powstał na gołych zdaniach, a VoiceLoop "
        "owija obie strony własnymi szablonami, co przesuwa cały rozkład. Werdykty "
        "wydaje osobny pomiar osiągalności, liczony w przestrzeni każdego progu."
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
