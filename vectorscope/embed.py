"""Wektoryzacja z jawnym prefiksem i pilnowaniem zgodności indeksów.

Klient VoiceLoopa w `embed_texts` po cichu wyrzuca puste teksty i przycina
wejście do 8000 znaków. Przy takiej liście wyjściowej numer wektora przestaje
odpowiadać numerowi fragmentu, a panel narysowałby graf, w którym podpisy są
poprzesuwane względem punktów. Ten moduł istnieje po to, żeby taki cichy
rozjazd był niemożliwy: prefiks nakładamy sami, puste odrzucamy głośno,
a liczbę zwróconych wektorów sprawdzamy przed użyciem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from voiceloop.embeddings import EmbeddingUnavailableError, OpenAICompatibleEmbeddingClient

from .config import (
    EMBEDDING_CONTEXT_TOKENS,
    PREFIX_DOCUMENT,
    PREFIX_NONE,
    PREFIX_QUERY,
    estimate_tokens,
)

BATCH_SIZE = 64

PREFIX_TEMPLATES = {
    PREFIX_QUERY: "search_query: ",
    PREFIX_DOCUMENT: "search_document: ",
    PREFIX_NONE: "",
}


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model: str
    dimension: int
    prefix: str
    over_context: list[int] = field(default_factory=list)


def apply_prefix(text: str, prefix: str) -> str:
    template = PREFIX_TEMPLATES.get(prefix)
    if template is None:
        raise ValueError(f"Nieznany prefiks: {prefix}")
    if not template:
        return text
    return text if text.casefold().startswith(template.casefold()) else f"{template}{text}"


async def embed_texts_with_prefix(
    client: OpenAICompatibleEmbeddingClient,
    texts: list[str],
    *,
    prefix: str,
) -> EmbeddingResult:
    if prefix not in PREFIX_TEMPLATES:
        raise ValueError(f"Nieznany prefiks: {prefix}")
    if not texts:
        return EmbeddingResult(
            vectors=np.zeros((0, 0), dtype=float),
            model="brak",
            dimension=0,
            prefix=prefix,
            over_context=[],
        )

    blank = [index for index, text in enumerate(texts) if not text.strip()]
    if blank:
        raise ValueError(
            f"Puste teksty na pozycjach {blank[:10]} — klient embeddingów wyrzuciłby je "
            "po cichu i wszystkie kolejne wektory trafiłyby pod złe fragmenty."
        )

    over_context = [
        index
        for index, text in enumerate(texts)
        if estimate_tokens(text) > EMBEDDING_CONTEXT_TOKENS
    ]

    prepared = [apply_prefix(text.strip(), prefix) for text in texts]

    collected: list[list[float]] = []
    for start in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[start : start + BATCH_SIZE]
        vectors = await client.embed_texts(batch)
        if not vectors:
            raise EmbeddingUnavailableError(
                "LM Studio nie zwróciło wektorów — sprawdź, czy model embeddingowy "
                "jest wczytany i czy klient jest włączony w ustawieniach."
            )
        if len(vectors) != len(batch):
            raise EmbeddingUnavailableError(
                f"LM Studio zwróciło {len(vectors)} wektorów dla {len(batch)} tekstów. "
                "Przerywam, bo przypisanie wektorów do fragmentów byłoby zmyślone."
            )
        collected.extend(vectors)

    matrix = np.asarray(collected, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise EmbeddingUnavailableError(
            f"Otrzymano macierz {matrix.shape} dla {len(texts)} tekstów."
        )

    dimensions = {len(vector) for vector in collected}
    if len(dimensions) > 1:
        raise EmbeddingUnavailableError(
            f"Niespójny wymiar wektorów w jednej odpowiedzi: {sorted(dimensions)}."
        )

    return EmbeddingResult(
        vectors=matrix,
        model=await client.resolve_model(),
        dimension=int(matrix.shape[1]),
        prefix=prefix,
        over_context=over_context,
    )


__all__ = [
    "BATCH_SIZE",
    "EmbeddingResult",
    "apply_prefix",
    "embed_texts_with_prefix",
]
