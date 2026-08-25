"""Geometria wektorów w pełnych 768 wymiarach — czysty NumPy.

Podział na prawdę i ilustrację jest tu twardy:

* macierz cosinusów, kNN i dendrogram liczymy w oryginalnej przestrzeni,
  więc są to fakty o wektorach;
* MDS i PCA spłaszczają tę przestrzeń do dwóch wymiarów, więc są to obrazki,
  którym zawsze towarzyszy pomiar zniekształcenia (trustworthiness,
  continuity, stress, diagram Sheparda).

Wszystko jest zaimplementowane ręcznie, bo w środowisku VoiceLoopa nie ma
`scipy` ani `scikit-learn` (polityka kontroli aplikacji blokuje ładowanie ich
bibliotek DLL), a dokładanie ich do produkcyjnego venva tylko dla panelu
diagnostycznego byłoby złą wymianą.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Powyżej tej liczby krawędzi graf przestaje być czytelny i zaczyna być tapetą.
MAX_GRAPH_EDGES = 1200

MDS_SEED = 20260824


@dataclass
class GraphEdges:
    pairs: list[tuple[int, int, float]]
    truncated: bool
    total_candidates: int


@dataclass
class Projection:
    coords: np.ndarray
    method: str
    explained_variance: list[float] | None
    explained_variance_total: float | None
    iterations: int | None


# --------------------------------------------------------------- podobieństwo


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosinusy liczone na kopii znormalizowanej — plik z wektorami zostaje surowy."""

    normalized = l2_normalize(vectors)
    similarity = normalized @ normalized.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def cosine_distance(similarity: np.ndarray) -> np.ndarray:
    distance = 1.0 - np.asarray(similarity, dtype=float)
    np.clip(distance, 0.0, 2.0, out=distance)
    np.fill_diagonal(distance, 0.0)
    return distance


def pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    matrix = np.asarray(coords, dtype=float)
    squared = np.sum(matrix**2, axis=1)
    distance = squared[:, None] + squared[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(distance, 0.0, out=distance)
    distance = np.sqrt(distance)
    np.fill_diagonal(distance, 0.0)
    return distance


# ---------------------------------------------------------------------- graf


def knn_edges(
    similarity: np.ndarray,
    *,
    k: int,
    threshold: float | None,
) -> GraphEdges:
    """Krawędzie z realnych cosinusów: k najbliższych w sumie z progiem.

    Krawędź jest informacją, pozycja węzła nie. Dlatego to jedyne miejsce,
    które decyduje, co panel pokazuje jako powiązanie.
    """

    count = int(similarity.shape[0])
    if count < 2:
        return GraphEdges(pairs=[], truncated=False, total_candidates=0)

    chosen: set[tuple[int, int]] = set()

    if k < 1:
        # k=0 to tryb „tylko próg": bierzemy każdą parę powyżej progu, bez
        # ograniczania do najbliższych sąsiadów. Bez progu byłby to komplet par.
        if threshold is None:
            return GraphEdges(pairs=[], truncated=False, total_candidates=0)
        upper = np.triu_indices(count, 1)
        passing = np.asarray(similarity)[upper] >= threshold
        chosen = {
            (int(left), int(right))
            for left, right in zip(upper[0][passing], upper[1][passing], strict=True)
        }
    else:
        work = np.array(similarity, dtype=float, copy=True)
        np.fill_diagonal(work, -np.inf)
        limit = min(k, count - 1)
        for row in range(count):
            candidates = np.argpartition(-work[row], limit - 1)[:limit]
            for column in candidates:
                value = work[row, int(column)]
                if not np.isfinite(value):
                    continue
                if threshold is not None and value < threshold:
                    continue
                left, right = (row, int(column)) if row < int(column) else (int(column), row)
                chosen.add((left, right))

    pairs = [(left, right, float(similarity[left, right])) for left, right in chosen]
    pairs.sort(key=lambda item: item[2], reverse=True)

    total = len(pairs)
    truncated = total > MAX_GRAPH_EDGES
    if truncated:
        pairs = pairs[:MAX_GRAPH_EDGES]
    return GraphEdges(pairs=pairs, truncated=truncated, total_candidates=total)


# ------------------------------------------------------------------ hierarchia


def average_linkage(distance: np.ndarray) -> np.ndarray:
    """UPGMA. Zwraca macierz w układzie zgodnym ze scipy: [lewy, prawy, odległość, rozmiar]."""

    count = int(distance.shape[0])
    if count < 2:
        return np.zeros((0, 4), dtype=float)

    current = np.array(distance, dtype=float, copy=True)
    np.fill_diagonal(current, np.inf)

    labels = list(range(count))
    sizes = {index: 1 for index in range(count)}
    linkage: list[list[float]] = []
    next_label = count

    while len(labels) > 1:
        size = current.shape[0]
        flat = int(np.argmin(current))
        row, column = divmod(flat, size)
        if row > column:
            row, column = column, row
        merge_distance = float(current[row, column])

        left_label, right_label = labels[row], labels[column]
        left_size, right_size = sizes[left_label], sizes[right_label]
        merged_size = left_size + right_size
        linkage.append([float(left_label), float(right_label), merge_distance, float(merged_size)])
        sizes[next_label] = merged_size

        merged_row = (left_size * current[row] + right_size * current[column]) / merged_size
        keep = [index for index in range(size) if index not in (row, column)]

        updated = np.empty((len(keep) + 1, len(keep) + 1), dtype=float)
        if keep:
            updated[: len(keep), : len(keep)] = current[np.ix_(keep, keep)]
            values = merged_row[keep]
            updated[: len(keep), -1] = values
            updated[-1, : len(keep)] = values
        updated[-1, -1] = np.inf

        current = updated
        labels = [labels[index] for index in keep] + [next_label]
        next_label += 1

    return np.array(linkage, dtype=float)


def leaf_order(linkage: np.ndarray, count: int) -> list[int]:
    """Kolejność liści zgodna z rysunkiem dendrogramu, bez przecinających się gałęzi."""

    if not len(linkage):
        return list(range(count))

    children = {
        count + index: (int(row[0]), int(row[1])) for index, row in enumerate(linkage)
    }
    order: list[int] = []
    stack = [count + len(linkage) - 1]
    while stack:
        node = stack.pop()
        if node < count:
            order.append(node)
            continue
        left, right = children[node]
        stack.append(right)
        stack.append(left)
    return order


def dendrogram_segments(
    linkage: np.ndarray,
    count: int,
    order: list[int],
) -> list[dict[str, list[float]]]:
    if not len(linkage):
        return []
    positions: dict[int, float] = {leaf: float(index) for index, leaf in enumerate(order)}
    heights: dict[int, float] = {leaf: 0.0 for leaf in order}
    segments: list[dict[str, list[float]]] = []
    for index, row in enumerate(linkage):
        left, right = int(row[0]), int(row[1])
        merge_distance = float(row[2])
        node = count + index
        left_x, right_x = positions[left], positions[right]
        left_y, right_y = heights[left], heights[right]
        segments.append(
            {
                "x": [left_x, left_x, right_x, right_x],
                "y": [left_y, merge_distance, merge_distance, right_y],
            }
        )
        positions[node] = (left_x + right_x) / 2.0
        heights[node] = merge_distance
    return segments


# ---------------------------------------------------------------------- rzuty


def pca_2d(vectors: np.ndarray) -> Projection:
    matrix = np.asarray(vectors, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = min(2, right.shape[0])
    coords = centered @ right[:components].T
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
    variance = singular**2
    total = float(variance.sum())
    ratios = (variance[:components] / total).tolist() if total > 0 else [0.0, 0.0]
    while len(ratios) < 2:
        ratios.append(0.0)
    return Projection(
        coords=coords,
        method="pca",
        explained_variance=[float(value) for value in ratios],
        explained_variance_total=float(sum(ratios)),
        iterations=None,
    )


def _classical_mds(distance: np.ndarray) -> np.ndarray:
    """Klasyczne skalowanie — deterministyczny start dla SMACOF."""

    count = int(distance.shape[0])
    squared = distance**2
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ squared @ centering
    gram = (gram + gram.T) / 2.0
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:2]
    scale = np.sqrt(np.maximum(values[top], 0.0))
    return vectors[:, top] * scale


def smacof(
    distance: np.ndarray,
    *,
    max_iterations: int = 300,
    tolerance: float = 1e-7,
) -> Projection:
    """Metryczne MDS metodą majoryzacji. Minimalizuje surowy stress wprost."""

    count = int(distance.shape[0])
    if count < 3:
        coords = np.zeros((count, 2), dtype=float)
        if count == 2:
            coords[1, 0] = float(distance[0, 1])
        return Projection(
            coords=coords,
            method="mds",
            explained_variance=None,
            explained_variance_total=None,
            iterations=0,
        )

    target = np.asarray(distance, dtype=float)
    coords = _classical_mds(target)
    if not np.all(np.isfinite(coords)) or np.allclose(coords, 0.0):
        generator = np.random.default_rng(MDS_SEED)
        coords = generator.normal(scale=0.1, size=(count, 2))

    previous: float | None = None
    iterations = 0
    for step in range(1, max_iterations + 1):
        iterations = step
        current = pairwise_euclidean(coords)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(current > 1e-12, target / current, 0.0)
        np.fill_diagonal(ratio, 0.0)

        guttman = -ratio
        np.fill_diagonal(guttman, ratio.sum(axis=1))
        coords = (guttman @ coords) / count

        stress = float(np.sum((target - pairwise_euclidean(coords)) ** 2) / 2.0)
        if previous is not None and abs(previous - stress) <= tolerance * max(previous, 1e-12):
            previous = stress
            break
        previous = stress

    return Projection(
        coords=coords,
        method="mds",
        explained_variance=None,
        explained_variance_total=None,
        iterations=iterations,
    )


# ------------------------------------------------------------ zniekształcenie


def kruskal_stress(high: np.ndarray, low: np.ndarray) -> float:
    """Stress-1 po dopasowaniu skali — rzut wolno przeskalować, nie wolno przestawić."""

    upper = np.triu_indices(high.shape[0], 1)
    target = high[upper]
    projected = low[upper]
    denominator = float(np.sum(projected**2))
    if denominator > 0:
        projected = projected * float(np.sum(projected * target) / denominator)
    total = float(np.sum(target**2))
    if total <= 0:
        return 0.0
    return float(np.sqrt(np.sum((target - projected) ** 2) / total))


def max_valid_neighbours(count: int) -> int:
    """Powyżej tej liczby sąsiadów normalizacja trustworthiness traci sens."""

    return max(1, int((2 * count - 2) // 3))


def _rank_matrix(distance: np.ndarray) -> np.ndarray:
    """Ranga sąsiedztwa: najbliższy obcy punkt ma 1, sam punkt ma 0."""

    count = int(distance.shape[0])
    work = np.array(distance, dtype=float, copy=True)
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.zeros((count, count), dtype=int)
    positions = np.arange(count)
    for row in range(count):
        ranks[row, order[row]] = positions
    return ranks


def _neighbour_indices(distance: np.ndarray, neighbours: int) -> np.ndarray:
    count = int(distance.shape[0])
    work = np.array(distance, dtype=float, copy=True)
    np.fill_diagonal(work, np.inf)
    limit = max(1, min(neighbours, count - 1))
    partial = np.argpartition(work, limit - 1, axis=1)[:, :limit]
    ordered = np.empty_like(partial)
    for row in range(count):
        ordered[row] = partial[row][np.argsort(work[row, partial[row]], kind="stable")]
    return ordered


def trustworthiness(high: np.ndarray, low: np.ndarray, neighbours: int) -> float | None:
    """Ile z sąsiadów widocznych na rzucie jest sąsiadami także w 768D."""

    count = int(high.shape[0])
    limit = max_valid_neighbours(count)
    if count < 4 or neighbours < 1 or neighbours > limit:
        return None
    high_ranks = _rank_matrix(high)
    low_neighbours = _neighbour_indices(low, neighbours)
    penalty = 0.0
    for row in range(count):
        for column in low_neighbours[row]:
            rank = int(high_ranks[row, column])
            if rank > neighbours:
                penalty += rank - neighbours
    normaliser = count * neighbours * (2 * count - 3 * neighbours - 1)
    if normaliser <= 0:
        return None
    return float(1.0 - (2.0 / normaliser) * penalty)


def continuity(high: np.ndarray, low: np.ndarray, neighbours: int) -> float | None:
    """Ile z prawdziwych sąsiadów przetrwało rzut — trustworthiness na odwrót."""

    return trustworthiness(low, high, neighbours)


def neighbourhood_preservation(
    high: np.ndarray,
    low: np.ndarray,
    neighbours: int,
) -> np.ndarray:
    count = int(high.shape[0])
    if count < 2:
        return np.zeros(count, dtype=float)
    limit = max(1, min(neighbours, count - 1))
    high_neighbours = _neighbour_indices(high, limit)
    low_neighbours = _neighbour_indices(low, limit)
    preserved = np.zeros(count, dtype=float)
    for row in range(count):
        shared = np.intersect1d(high_neighbours[row], low_neighbours[row], assume_unique=False)
        preserved[row] = len(shared) / limit
    return preserved


def shepard_sample(
    high: np.ndarray,
    low: np.ndarray,
    *,
    limit: int = 4000,
    seed: int = MDS_SEED,
) -> tuple[list[float], list[float], bool]:
    upper = np.triu_indices(high.shape[0], 1)
    target = high[upper]
    projected = low[upper]
    denominator = float((projected * projected).sum())
    if denominator > 0:
        projected = projected * float((projected * target).sum() / denominator)
    sampled = False
    if target.size > limit:
        generator = np.random.default_rng(seed)
        picked = generator.choice(target.size, size=limit, replace=False)
        target = target[picked]
        projected = projected[picked]
        sampled = True
    return target.tolist(), projected.tolist(), sampled


def pair_histogram(
    similarity: np.ndarray,
    *,
    bins: int = 60,
) -> tuple[list[float], list[int]]:
    count = int(similarity.shape[0])
    if count < 2:
        return [], []
    upper = np.triu_indices(count, 1)
    values = similarity[upper]
    counts, edges = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    centres = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    return centres, [int(value) for value in counts]
