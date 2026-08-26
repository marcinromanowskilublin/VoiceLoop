"""Strażnik progów: sprzężenie zwrotne między pomiarem a konfiguracją.

Ten projekt mierzy sam siebie na dziesięć tysięcy linii kodu, a mimo to trzy progi
stały martwe miesiącami i prawie połowa pamięci zapełniła się duplikatami. Powód
był jeden i prosty: pomiar nie miał konsumenta. Raport, którego nikt nie odczytuje,
nie różni się od braku raportu.

Strażnik zamyka tę pętlę. Raz na dobę bierze próbkę realnej kolekcji, mierzy
rozkłady w tych samych przestrzeniach, w których pracują progi, i orzeka o każdym
z nich. Werdykt trafia do `/api/v1/health`, nie tylko do logu — bo log, którego
nikt nie czyta, to dokładnie ten problem, który naprawiamy.

Wszystko jest do odczytu: `scroll` i `query_points`. Strażnik niczego nie zapisuje
do pamięci VoiceLoopa.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import Any

from .embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient
from .memory_vectorization import memory_vector_documents
from .qdrant_memory import QdrantMemoryError, QdrantVectorStore
from .screenpipe_memory import ACTIVITY_DUPLICATE_MIN_SCORE
from .settings import Settings
from .threshold_measure import (
    ThresholdVerdict,
    classify_duplicate_threshold,
    classify_reconstruction,
    classify_threshold,
    measure_axis_scores,
    measure_document_neighbourhood,
    scroll_points,
    unmeasured_verdict,
)

LOGGER = logging.getLogger(__name__)

#: Ile wyników zbierać z każdej osi. Głębiej znaczy lepsze oszacowanie ogona
#: rozkładu, ale też więcej pracy Qdranta na jedną sondę.
SEARCH_DEPTH = 100

#: Ile punktów przeczytać, żeby mieć z czego losować próbkę.
SCROLL_MULTIPLIER = 20
SCROLL_CEILING = 2000

MINIMUM_PROBES = 5
BACKOFF_CEILING_SECONDS = 3600.0


class ThresholdGuard:
    """Worker okresowy orzekający, czy progi robią to, co obiecują."""

    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: OpenAICompatibleEmbeddingClient,
        qdrant: QdrantVectorStore,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.enabled = settings.threshold_guard_enabled
        self.interval_seconds = max(60, int(settings.threshold_guard_interval_seconds))
        self.sample = max(MINIMUM_PROBES, min(int(settings.threshold_guard_sample), 200))
        self._task: asyncio.Task[None] | None = None
        self._verdicts: list[ThresholdVerdict] = []
        self._measured_at: datetime | None = None
        self._last_error: str | None = None
        self._points_seen = 0

    async def start(self) -> None:
        if not self.enabled:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="threshold-guard")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "wyłączony w konfiguracji"
        if not self.embeddings.enabled:
            return False, "lokalne embeddings są wyłączone"
        if not getattr(self.qdrant, "enabled", False):
            return False, "Qdrant jest wyłączony"
        if self._last_error:
            return False, self._last_error
        if not self._verdicts:
            if self._task and not self._task.done():
                return True, "czeka na pierwszy pomiar"
            return False, "nie wystartował"
        return self._summarise()

    def report(self) -> dict[str, Any]:
        """Pełny wynik ostatniego pomiaru, do wglądu poza health-checkiem."""

        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "sample": self.sample,
            "measured_at": self._measured_at.isoformat() if self._measured_at else None,
            "points_seen": self._points_seen,
            "last_error": self._last_error,
            "verdicts": [verdict.as_dict() for verdict in self._verdicts],
        }

    async def _run(self) -> None:
        backoff_seconds = float(self.interval_seconds)
        while True:
            try:
                await self.measure_once()
                backoff_seconds = float(self.interval_seconds)
            except asyncio.CancelledError:
                raise
            except (EmbeddingUnavailableError, QdrantMemoryError) as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                LOGGER.warning("Threshold guard paused: %s", exc)
                backoff_seconds = min(backoff_seconds * 2, BACKOFF_CEILING_SECONDS)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                LOGGER.exception("Threshold guard measurement failed")
                backoff_seconds = min(backoff_seconds * 2, BACKOFF_CEILING_SECONDS)
            await asyncio.sleep(backoff_seconds)

    async def measure_once(self) -> list[ThresholdVerdict]:
        """Jeden przebieg pomiaru. Podnosi wyjątki — pętla je łapie i loguje."""

        if not self.embeddings.enabled or not getattr(self.qdrant, "enabled", False):
            self._last_error = "brak Qdranta albo lokalnych embeddingów"
            return []

        client = self.qdrant.client
        collection = self.qdrant.collection_name
        limit = min(self.sample * SCROLL_MULTIPLIER, SCROLL_CEILING)
        points = await scroll_points(client, collection, limit=limit)
        self._points_seen = len(points)

        usable = [item for item in points if item["content"].strip()]
        if len(usable) < MINIMUM_PROBES:
            self._verdicts = []
            self._measured_at = datetime.now(UTC)
            self._last_error = (
                f"za mało dokumentów z treścią: {len(usable)} z {len(points)}"
            )
            return []

        probes = random.Random().sample(usable, min(self.sample, len(usable)))
        verdicts = [
            *await self._judge_search_gate(client, collection, probes),
            *await self._judge_duplicate_gate(client, collection, probes),
            *self._out_of_scope(),
        ]

        self._verdicts = verdicts
        self._measured_at = datetime.now(UTC)
        self._last_error = None
        for verdict in verdicts:
            if verdict.broken:
                LOGGER.warning("Próg %s: %s", verdict.name, verdict.message)
        return verdicts

    async def _judge_search_gate(
        self,
        client: Any,
        collection: str,
        probes: list[dict[str, Any]],
    ) -> list[ThresholdVerdict]:
        """Bramka przed fuzją działa per oś, więc orzeczenie też musi być per oś.

        Jedna liczba na pięć osi znaczyłaby w każdej co innego — mediany szumu
        różnią się między osiami o kilkanaście setnych.
        """

        threshold = float(self.settings.vector_memory_min_score)
        measured = await measure_axis_scores(
            client=client,
            collection=collection,
            embeddings=self.embeddings,
            probes=probes,
            depth=SEARCH_DEPTH,
        )
        verdicts: list[ThresholdVerdict] = []
        for axis, scores in measured.items():
            name = f"vector_memory_min_score[{axis}]"
            if scores.error:
                verdicts.append(
                    unmeasured_verdict(
                        name=name,
                        value=threshold,
                        space=f"zapytanie kontra dokument, oś {axis}",
                        reason=f"Qdrant odmówił odpowiedzi dla tej osi: {scores.error}.",
                    )
                )
                continue
            verdicts.append(
                classify_threshold(
                    name=name,
                    value=threshold,
                    space=f"zapytanie kontra dokument, oś {axis}",
                    observed=scores.scores,
                )
            )
        return verdicts

    async def _judge_duplicate_gate(
        self,
        client: Any,
        collection: str,
        probes: list[dict[str, Any]],
    ) -> list[ThresholdVerdict]:
        """Próg deduplikacji mierzony tam, gdzie naprawdę działa.

        Dokument odtwarzamy produkcyjnym szablonem z `content` i `observations`,
        więc porównanie jest dokument do dokumentu — dokładnie tak, jak po naprawie
        robi to `_semantic_duplicate_exists`.

        Ten sam pomiar odpowiada na drugie pytanie: czy zapisane wektory w ogóle
        zgadzają się ze schematem, który deklarują. Dwa werdykty, bo to dwie różne
        rzeczy — próg może być bez zarzutu przy danych, które go ominą.
        """

        documents: list[tuple[str, str]] = []
        for probe in probes:
            rebuilt = memory_vector_documents(
                summary=probe["content"],
                observations=probe["observations"],
                redact=False,
            )
            semantic = rebuilt.get("semantic")
            if semantic:
                documents.append((probe["id"], semantic))

        if not documents:
            return [
                unmeasured_verdict(
                    name="ACTIVITY_DUPLICATE_MIN_SCORE",
                    value=ACTIVITY_DUPLICATE_MIN_SCORE,
                    space="dokument kontra dokument, oś semantic",
                    reason="Nie udało się odtworzyć ani jednego dokumentu semantycznego.",
                )
            ]

        identical, distinct = await measure_document_neighbourhood(
            client=client,
            collection=collection,
            embeddings=self.embeddings,
            documents=documents,
        )
        return [
            classify_duplicate_threshold(
                name="ACTIVITY_DUPLICATE_MIN_SCORE",
                value=ACTIVITY_DUPLICATE_MIN_SCORE,
                space="dokument kontra dokument, oś semantic",
                identical=identical,
                distinct=distinct,
            ),
            classify_reconstruction(identical=identical),
        ]

    def _out_of_scope(self) -> list[ThresholdVerdict]:
        """Progi, których ten pomiar nie obejmuje — wymienione, nie przemilczane.

        Milczenie o progu jest tym samym błędem, który strażnik ma wykrywać.
        """

        return [
            unmeasured_verdict(
                name="capability_match_min_score",
                value=float(self.settings.capability_match_min_score),
                space="znormalizowany RRF, nie cosinus",
                reason=(
                    "Próg porównuje znormalizowaną fuzję rang, a nie podobieństwo. "
                    "Pomiar wymaga ujednolicenia dwóch implementacji RRF."
                ),
            ),
            unmeasured_verdict(
                name="behavior_digest_min_confidence",
                value=float(self.settings.behavior_digest_min_confidence),
                space="pewność zwracana przez model",
                reason=(
                    "Próg dotyczy wyjścia modelu językowego, nie geometrii "
                    "embeddingów. Wymaga osobnego pomiaru."
                ),
            ),
        ]

    def _summarise(self) -> tuple[bool, str]:
        stamp = (
            self._measured_at.strftime("%Y-%m-%d %H:%M")
            if self._measured_at
            else "brak daty"
        )
        broken = [verdict for verdict in self._verdicts if verdict.broken]
        judged = [
            verdict for verdict in self._verdicts if verdict.status != "unmeasured"
        ]
        if broken:
            names = ", ".join(f"{item.name} ({item.status})" for item in broken[:3])
            more = f" i {len(broken) - 3} więcej" if len(broken) > 3 else ""
            return False, f"progi wymagające uwagi: {names}{more}; pomiar {stamp}"
        return True, f"{len(judged)} progów bez zastrzeżeń; pomiar {stamp}"
