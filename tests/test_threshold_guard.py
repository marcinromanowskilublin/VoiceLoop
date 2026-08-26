"""Testy strażnika progów.

Strażnik istnieje, żeby wykryć próg, który nic nie robi. Więc testy muszą pokazać
przede wszystkim, że każdy z trzech sposobów zepsucia progu zostanie nazwany po
imieniu: martwy, nieosiągalny, zbyt szeroki. Reszta to sprawdzenie, że przyrząd
nie umiera, gdy mierzony obiekt milczy.
"""

import asyncio
import re

import pytest

from voiceloop.settings import Settings
from voiceloop.threshold_guard import ThresholdGuard
from voiceloop.threshold_measure import (
    STATUS_DEAD,
    STATUS_DISABLED,
    STATUS_DRIFTED,
    STATUS_OVER_BROAD,
    STATUS_UNMEASURED,
    STATUS_UNREACHABLE,
    STATUS_WORKING,
    classify_duplicate_threshold,
    classify_reconstruction,
    classify_threshold,
    percentiles,
)

DEDUP_DEPTH = 10


class FakePoint:
    def __init__(self, point_id: str, score: float) -> None:
        self.id = point_id
        self.score = score


class FakeResponse:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeScrollPoint:
    def __init__(self, point_id: str, content: str) -> None:
        self.id = point_id
        self.vector = None
        self.payload = {
            "source": "screenpipe_behavior",
            "content": content,
            "title": "Aktywność",
            "metadata": {
                "observations": [f"Obserwacja dla {point_id}"],
                "schema": "memory-documents-v2",
            },
        }


class FakeClient:
    """Kolekcja sterowana funkcją odpowiedzi, żeby dyktować rozkład wyników."""

    def __init__(self, points, responder, *, unavailable_axes=()) -> None:
        self._points = list(points)
        self._responder = responder
        self.unavailable_axes = set(unavailable_axes)
        self.asked = []
        self.thresholds_used = []

    async def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        return self._points, None

    async def query_points(
        self,
        *,
        collection_name,
        query,
        using,
        limit,
        score_threshold,
        with_payload,
        with_vectors,
    ):
        if using in self.unavailable_axes:
            raise RuntimeError(f"Vector {using} does not exist")
        self.asked.append((using, limit))
        self.thresholds_used.append(score_threshold)
        return FakeResponse(self._responder(using, limit, int(query[0])))


class FailingClient:
    async def scroll(self, **_kwargs):
        raise RuntimeError("Qdrant nie odpowiada")

    async def query_points(self, **_kwargs):
        raise RuntimeError("Qdrant nie odpowiada")


class FakeEmbeddings:
    """Wektor niesie numer punktu, żeby atrapa Qdranta wiedziała, o kogo pytamy.

    Bez tego sonda deduplikacji nie umiałaby odróżnić trafienia w siebie od
    trafienia w obcy dokument, a właśnie ta różnica jest tu mierzona.
    """

    enabled = True

    def __init__(self) -> None:
        self.query_batches = 0
        self.document_batches = 0

    async def embed_queries(self, texts):
        self.query_batches += 1
        return [[float(_point_number(text)), 0.0] for text in texts]

    async def embed_documents(self, texts):
        self.document_batches += 1
        return [[float(_point_number(text)), 0.0] for text in texts]


def _point_number(text: str) -> int:
    match = re.search(r"numer (\d+)", text)
    return int(match.group(1)) if match else -1


class FakeStore:
    def __init__(self, client, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.client = client
        self.collection_name = "voiceloop_memories_v2"


def _points(count: int = 6) -> list[FakeScrollPoint]:
    return [
        FakeScrollPoint(f"point-{index}", f"Praca nad pamięcią numer {index}.")
        for index in range(count)
    ]


def _responder(*, search_scores: list[float], self_score: float, distinct_score: float):
    """Rozkład szumu dla osi i osobny dla sondy deduplikacji.

    Sonda deduplikacji pyta z płytszą głębokością, więc da się ją rozpoznać bez
    zgadywania: to jedyne wywołanie z `limit == DEDUP_DEPTH`.
    """

    def respond(using: str, limit: int, asked_for: int) -> list[FakePoint]:
        if limit == DEDUP_DEPTH:
            return [
                FakePoint(f"point-{asked_for}", self_score),
                FakePoint("point-obcy", distinct_score),
            ]
        return [FakePoint(f"noise-{index}", score) for index, score in enumerate(search_scores)]

    return respond


def _guard(tmp_path, *, client, embeddings=None, **overrides) -> ThresholdGuard:
    settings = Settings(voiceloop_data_dir=str(tmp_path), **overrides)
    return ThresholdGuard(
        settings,
        embeddings=embeddings or FakeEmbeddings(),
        qdrant=FakeStore(client),  # type: ignore[arg-type]
    )


def _by_axis(verdicts, axis: str):
    return next(item for item in verdicts if item.name == f"vector_memory_min_score[{axis}]")


# --- Klasyfikacja progu odsiewającego ---------------------------------------


def test_threshold_below_every_observation_is_called_dead() -> None:
    """Próg, który nie odrzuca niczego, ma być nazwany martwym, nie działającym."""

    verdict = classify_threshold(
        name="vector_memory_min_score",
        value=0.10,
        space="zapytanie kontra dokument",
        observed=[0.44, 0.55, 0.61, 0.90],
    )

    assert verdict.status == STATUS_DEAD
    assert verdict.broken is True
    assert verdict.rejected_share == 0.0
    assert "nie odrzuca niczego" in verdict.message
    assert "0.440" in verdict.message


def test_threshold_above_every_observation_is_called_unreachable() -> None:
    verdict = classify_threshold(
        name="vector_memory_min_score",
        value=0.99,
        space="zapytanie kontra dokument",
        observed=[0.44, 0.55, 0.61, 0.90],
    )

    assert verdict.status == STATUS_UNREACHABLE
    assert verdict.broken is True
    assert verdict.rejected_share == 1.0
    assert "nie przepuszcza niczego" in verdict.message


def test_threshold_inside_distribution_reports_the_share_it_rejects() -> None:
    verdict = classify_threshold(
        name="vector_memory_min_score",
        value=0.60,
        space="zapytanie kontra dokument",
        observed=[0.40, 0.50, 0.70, 0.80],
    )

    assert verdict.status == STATUS_WORKING
    assert verdict.broken is False
    assert verdict.rejected_share == 0.5
    assert "50.0%" in verdict.message


def test_zero_threshold_is_disabled_not_dead() -> None:
    """Zero to świadoma rezygnacja z bramki, a nie awaria — i tak ma być nazwane."""

    verdict = classify_threshold(
        name="vector_memory_min_score",
        value=0.0,
        space="zapytanie kontra dokument",
        observed=[0.44, 0.55, 0.90],
    )

    assert verdict.status == STATUS_DISABLED
    assert verdict.broken is False
    assert "nie deklaruje bramki" in verdict.message


def test_threshold_without_observations_is_unmeasured() -> None:
    verdict = classify_threshold(
        name="vector_memory_min_score",
        value=0.5,
        space="zapytanie kontra dokument",
        observed=[],
    )

    assert verdict.status == STATUS_UNMEASURED
    assert verdict.broken is False
    assert verdict.sample == 0


def test_empty_percentiles_are_zeros_so_the_report_stays_valid_json() -> None:
    """NaN przechodzi przez numpy, ale nie przez JSON w health-checku."""

    stats = percentiles([])

    assert stats == {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}


# --- Klasyfikacja progu deduplikacji ---------------------------------------


def test_duplicate_threshold_is_unreachable_when_identical_text_cannot_reach_it() -> None:
    """Dokładnie ten błąd stał w kodzie miesiącami: 0.92 w niewłaściwej przestrzeni."""

    verdict = classify_duplicate_threshold(
        name="ACTIVITY_DUPLICATE_MIN_SCORE",
        value=0.92,
        space="zapytanie kontra dokument",
        identical=[0.86, 0.88, 0.90],
        distinct=[0.40, 0.55],
    )

    assert verdict.status == STATUS_UNREACHABLE
    assert verdict.broken is True
    assert "nawet tekst identyczny" in verdict.message
    assert "miesza przestrzeni" in verdict.message


def test_duplicate_threshold_that_also_rejects_new_content_is_called_over_broad() -> None:
    verdict = classify_duplicate_threshold(
        name="ACTIVITY_DUPLICATE_MIN_SCORE",
        value=0.90,
        space="dokument kontra dokument",
        identical=[1.0, 1.0, 1.0],
        distinct=[0.95, 0.96, 0.40, 0.50],
    )

    assert verdict.status == STATUS_OVER_BROAD
    assert verdict.broken is True
    assert verdict.rejected_share == 0.5
    assert "treść nową" in verdict.message


def test_duplicate_threshold_between_the_two_distributions_works() -> None:
    verdict = classify_duplicate_threshold(
        name="ACTIVITY_DUPLICATE_MIN_SCORE",
        value=0.97,
        space="dokument kontra dokument",
        identical=[1.0, 0.999, 0.998],
        distinct=[0.55, 0.62, 0.87],
    )

    assert verdict.status == STATUS_WORKING
    assert verdict.broken is False
    assert verdict.rejected_share == 0.0


# --- Zgodność danych ze schematem ------------------------------------------


def test_collection_that_cannot_reproduce_its_own_vectors_is_called_drifted() -> None:
    """Zmierzone na żywej kolekcji: 14% punktów odtwarza się dokładnie, reszta nie.

    Werdykt musi wskazać dane, nie próg. Próg deduplikacji jest tu bez winy —
    po prostu nigdy nie zobaczy tych punktów jako powtórzeń.
    """

    verdict = classify_reconstruction(identical=[1.0, 0.83, 0.79, 0.75, 0.88])

    assert verdict.status == STATUS_DRIFTED
    assert verdict.broken is True
    assert verdict.name == "memory_document_schema"
    assert "20%" in verdict.message
    assert "defekt danych, nie progu" in verdict.message


def test_collection_written_by_one_schema_reconstructs_exactly() -> None:
    verdict = classify_reconstruction(identical=[1.0, 1.0, 0.9995, 1.0])

    assert verdict.status == STATUS_WORKING
    assert verdict.broken is False
    assert "100%" in verdict.message


def test_reconstruction_without_documents_is_unmeasured() -> None:
    verdict = classify_reconstruction(identical=[])

    assert verdict.status == STATUS_UNMEASURED
    assert verdict.broken is False


# --- Worker ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_measures_every_axis_without_applying_any_threshold(tmp_path) -> None:
    """Próg zastosowany w trakcie mierzenia progu byłby błędem cyrkularnym."""

    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5, 0.6, 0.7], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client, vector_memory_min_score=0.60)

    verdicts = await guard.measure_once()

    assert all(threshold is None for threshold in client.thresholds_used)
    axes = {"semantic", "topic", "intent", "decision", "person_context"}
    assert {
        item.name for item in verdicts if item.name.startswith("vector_memory_min_score")
    } == {f"vector_memory_min_score[{axis}]" for axis in axes}
    assert _by_axis(verdicts, "semantic").status == STATUS_WORKING


@pytest.mark.asyncio
async def test_guard_flags_a_dead_gate_in_health(tmp_path) -> None:
    client = FakeClient(
        _points(),
        _responder(search_scores=[0.44, 0.60, 0.90], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client, vector_memory_min_score=0.10)

    await guard.measure_once()
    ok, detail = guard.health()

    assert _by_axis(guard._verdicts, "topic").status == STATUS_DEAD
    assert ok is False
    assert "dead" in detail
    assert "vector_memory_min_score" in detail


@pytest.mark.asyncio
async def test_guard_reports_no_reservations_when_every_gate_behaves(tmp_path) -> None:
    client = FakeClient(
        _points(),
        _responder(search_scores=[0.40, 0.50, 0.70, 0.80], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client, vector_memory_min_score=0.60)

    await guard.measure_once()
    ok, detail = guard.health()

    assert ok is True
    assert "bez zastrzeżeń" in detail


@pytest.mark.asyncio
async def test_guard_names_the_axis_qdrant_refuses_instead_of_dropping_it(tmp_path) -> None:
    """Brak osi w kolekcji to informacja. Milczenie o niej byłoby tym samym błędem."""

    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5, 0.6], self_score=1.0, distinct_score=0.5),
        unavailable_axes=("decision",),
    )
    guard = _guard(tmp_path, client=client, vector_memory_min_score=0.55)

    verdicts = await guard.measure_once()

    decision = _by_axis(verdicts, "decision")
    assert decision.status == STATUS_UNMEASURED
    assert "RuntimeError" in decision.message
    assert _by_axis(verdicts, "semantic").status != STATUS_UNMEASURED


@pytest.mark.asyncio
async def test_guard_judges_the_duplicate_gate_in_document_space(tmp_path) -> None:
    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5, 0.6], self_score=1.0, distinct_score=0.55),
    )
    embeddings = FakeEmbeddings()
    guard = _guard(tmp_path, client=client, embeddings=embeddings)

    verdicts = await guard.measure_once()

    duplicate = next(
        item for item in verdicts if item.name == "ACTIVITY_DUPLICATE_MIN_SCORE"
    )
    assert duplicate.status == STATUS_WORKING
    assert duplicate.space == "dokument kontra dokument, oś semantic"
    # Dokumenty wektoryzujemy jako dokumenty, nie zapytania — inaczej pomiar
    # trafiłby w tę samą pułapkę, którą ma wykrywać.
    assert embeddings.document_batches == 1


@pytest.mark.asyncio
async def test_guard_lists_thresholds_it_cannot_measure_instead_of_skipping_them(
    tmp_path,
) -> None:
    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5, 0.6], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client)

    verdicts = await guard.measure_once()
    names = {item.name for item in verdicts}

    assert "capability_match_min_score" in names
    assert "behavior_digest_min_confidence" in names
    out_of_scope = next(item for item in verdicts if item.name == "capability_match_min_score")
    assert out_of_scope.status == STATUS_UNMEASURED
    assert "RRF" in out_of_scope.message


@pytest.mark.asyncio
async def test_guard_says_so_when_the_collection_is_too_small_to_judge(tmp_path) -> None:
    client = FakeClient(
        [],
        _responder(search_scores=[0.5], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client)

    verdicts = await guard.measure_once()
    ok, detail = guard.health()

    assert verdicts == []
    assert ok is False
    assert "za mało dokumentów" in detail


@pytest.mark.asyncio
async def test_guard_survives_an_unavailable_qdrant_and_reports_it(tmp_path) -> None:
    """Worker, który umiera po pierwszej awarii Qdranta, przestaje pilnować progów."""

    guard = _guard(
        tmp_path,
        client=FailingClient(),
        threshold_guard_interval_seconds=600,
    )

    task = asyncio.create_task(guard._run())
    await asyncio.sleep(0.05)

    try:
        assert task.done() is False
        ok, detail = guard.health()
        assert ok is False
        assert "RuntimeError" in detail
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_disabled_guard_starts_no_task_and_admits_it(tmp_path) -> None:
    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client, threshold_guard_enabled=False)

    await guard.start()
    ok, detail = guard.health()

    assert guard._task is None
    assert ok is False
    assert detail == "wyłączony w konfiguracji"


@pytest.mark.asyncio
async def test_guard_without_qdrant_does_not_pretend_to_have_measured(tmp_path) -> None:
    settings = Settings(voiceloop_data_dir=str(tmp_path))
    guard = ThresholdGuard(
        settings,
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        qdrant=FakeStore(FailingClient(), enabled=False),  # type: ignore[arg-type]
    )

    assert await guard.measure_once() == []
    ok, detail = guard.health()
    assert ok is False
    assert detail == "Qdrant jest wyłączony"


@pytest.mark.asyncio
async def test_report_carries_the_full_verdict_list_for_inspection(tmp_path) -> None:
    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5, 0.9], self_score=1.0, distinct_score=0.5),
    )
    guard = _guard(tmp_path, client=client, threshold_guard_sample=12)

    await guard.measure_once()
    report = guard.report()

    assert report["sample"] == 12
    assert report["measured_at"] is not None
    assert report["last_error"] is None
    # Pięć osi, próg deduplikacji, zgodność danych ze schematem, dwa progi poza zasięgiem.
    assert len(report["verdicts"]) == 9
    assert all("status" in item and "message" in item for item in report["verdicts"])


@pytest.mark.asyncio
async def test_guard_separates_a_schema_defect_from_a_threshold_defect(tmp_path) -> None:
    """Gdy dane nie odtwarzają własnych wektorów, wina jest po stronie danych.

    Próg deduplikacji dostaje wtedy werdykt "osiągalny", a osobny werdykt wskazuje
    schemat. Zlepienie tego w jedno orzeczenie kazałoby szukać błędu w progu.
    """

    client = FakeClient(
        _points(),
        _responder(search_scores=[0.5, 0.6], self_score=0.80, distinct_score=0.45),
    )
    guard = _guard(tmp_path, client=client)

    verdicts = await guard.measure_once()
    by_name = {item.name: item for item in verdicts}
    ok, detail = guard.health()

    assert by_name["ACTIVITY_DUPLICATE_MIN_SCORE"].status == STATUS_UNREACHABLE
    assert by_name["memory_document_schema"].status == STATUS_DRIFTED
    assert ok is False
    assert "memory_document_schema" in detail
