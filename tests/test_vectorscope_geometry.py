"""Testy geometrii Vectorscope.

Środowisko VoiceLoopa nie ma scipy ani sklearn (kontrola aplikacji blokuje
ładowanie ich DLL), więc cała geometria jest napisana ręcznie. Tym bardziej
musi być sprawdzana wobec wartości wyliczonych niezależnie od implementacji,
a nie wobec samej siebie.
"""

from __future__ import annotations

import numpy as np
import pytest

from vectorscope import geometry as g


def _random_vectors(count: int, dimension: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, dimension))


# ------------------------------------------------------------------ cosinusy


def test_cosine_matrix_is_symmetric_with_unit_diagonal() -> None:
    similarity = g.cosine_matrix(_random_vectors(12, 8, 1))
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    assert similarity.min() >= -1.0 and similarity.max() <= 1.0


def test_cosine_matrix_matches_hand_computation() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, -1.0]])
    similarity = g.cosine_matrix(vectors)
    assert similarity[0, 1] == pytest.approx(1 / np.sqrt(2))
    assert similarity[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert similarity[1, 2] == pytest.approx(-1 / np.sqrt(2))


def test_l2_normalize_survives_a_zero_vector() -> None:
    normalized = g.l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.all(np.isfinite(normalized))
    assert np.linalg.norm(normalized[1]) == pytest.approx(1.0)


def test_cosine_distance_has_zero_diagonal_and_no_negatives() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(10, 6, 2)))
    assert np.allclose(np.diag(distance), 0.0)
    assert distance.min() >= 0.0


def test_pairwise_euclidean_matches_direct_formula() -> None:
    coords = _random_vectors(7, 2, 3)
    distance = g.pairwise_euclidean(coords)
    for i in range(7):
        for j in range(7):
            assert distance[i, j] == pytest.approx(np.linalg.norm(coords[i] - coords[j]))


# ---------------------------------------------------------------------- graf


def test_knn_edges_are_undirected_and_respect_the_threshold() -> None:
    similarity = g.cosine_matrix(_random_vectors(15, 5, 4))
    edges = g.knn_edges(similarity, k=3, threshold=0.2)
    seen = {(left, right) for left, right, _ in edges.pairs}
    assert len(seen) == len(edges.pairs), "krawędzie nie mogą się dublować"
    for left, right, value in edges.pairs:
        assert left < right
        assert value >= 0.2
        assert value == pytest.approx(similarity[left, right])


def test_knn_edges_with_k_zero_take_every_pair_above_threshold() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    similarity = g.cosine_matrix(vectors)
    edges = g.knn_edges(similarity, k=0, threshold=0.9)
    assert {(left, right) for left, right, _ in edges.pairs} == {(0, 1)}


def test_knn_edges_actually_limit_the_neighbour_count() -> None:
    """Regresja: wcześniejsza wersja zwracała pełny graf, więc suwak nic nie robił."""

    similarity = g.cosine_matrix(_random_vectors(20, 5, 5))
    complete = 20 * 19 // 2
    edges = g.knn_edges(similarity, k=2, threshold=None)
    assert len(edges.pairs) < complete
    assert len(edges.pairs) <= 20 * 2


def test_knn_edges_never_link_a_node_to_itself() -> None:
    similarity = g.cosine_matrix(_random_vectors(9, 4, 6))
    edges = g.knn_edges(similarity, k=8, threshold=None)
    assert all(left != right for left, right, _ in edges.pairs)


def test_knn_edges_are_truncated_with_a_flag() -> None:
    similarity = np.ones((70, 70))
    edges = g.knn_edges(similarity, k=0, threshold=0.5)
    assert edges.total_candidates == 70 * 69 // 2
    assert edges.truncated is (edges.total_candidates > g.MAX_GRAPH_EDGES)
    assert len(edges.pairs) <= g.MAX_GRAPH_EDGES


# ----------------------------------------------------------------- hierarchia


def _cophenetic(linkage: np.ndarray, count: int) -> np.ndarray:
    """Odległości kofenetyczne wyliczone niezależnie od implementacji linkage."""

    members: dict[int, list[int]] = {index: [index] for index in range(count)}
    result = np.zeros((count, count))
    for step, row in enumerate(linkage):
        left, right = int(row[0]), int(row[1])
        height = float(row[2])
        for a in members[left]:
            for b in members[right]:
                result[a, b] = height
                result[b, a] = height
        members[count + step] = members[left] + members[right]
    return result


def test_average_linkage_reproduces_upgma_by_hand() -> None:
    distance = np.array(
        [
            [0.0, 2.0, 6.0, 10.0],
            [2.0, 0.0, 5.0, 9.0],
            [6.0, 5.0, 0.0, 4.0],
            [10.0, 9.0, 4.0, 0.0],
        ]
    )
    linkage = g.average_linkage(distance)
    assert linkage.shape == (3, 4)
    # Najbliższa para to (0,1) w odległości 2, potem (2,3) w odległości 4.
    assert sorted(linkage[0, :2].astype(int)) == [0, 1]
    assert linkage[0, 2] == pytest.approx(2.0)
    assert sorted(linkage[1, :2].astype(int)) == [2, 3]
    assert linkage[1, 2] == pytest.approx(4.0)
    # Ostatnie scalenie: średnia z 0-2, 0-3, 1-2, 1-3 = (6+10+5+9)/4 = 7.5
    assert linkage[2, 2] == pytest.approx(7.5)
    assert linkage[2, 3] == pytest.approx(4.0)


def test_average_linkage_is_monotonic_and_merges_every_point() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(18, 7, 7)))
    linkage = g.average_linkage(distance)
    assert linkage.shape == (17, 4)
    heights = linkage[:, 2]
    assert np.all(np.diff(heights) >= -1e-12), "UPGMA nie może tworzyć inwersji"
    assert linkage[-1, 3] == 18


def test_cophenetic_distances_respect_the_original_ordering() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(14, 6, 8)))
    cophenetic = _cophenetic(g.average_linkage(distance), 14)
    upper = np.triu_indices(14, 1)
    correlation = np.corrcoef(distance[upper], cophenetic[upper])[0, 1]
    assert correlation > 0.6


def test_leaf_order_is_a_permutation() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(11, 5, 9)))
    order = g.leaf_order(g.average_linkage(distance), 11)
    assert sorted(order) == list(range(11))


def test_dendrogram_segments_start_at_the_leaves() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(8, 4, 10)))
    linkage = g.average_linkage(distance)
    order = g.leaf_order(linkage, 8)
    segments = g.dendrogram_segments(linkage, 8, order)
    assert len(segments) == 7
    for segment in segments:
        assert len(segment["x"]) == 4 and len(segment["y"]) == 4
        assert segment["y"][1] == segment["y"][2], "poprzeczka musi być pozioma"
        assert segment["y"][1] >= segment["y"][0]


def test_linkage_of_a_single_point_is_empty() -> None:
    assert g.average_linkage(np.zeros((1, 1))).shape == (0, 4)
    assert g.leaf_order(np.zeros((0, 4)), 1) == [0]


# ---------------------------------------------------------------------- PCA


def test_pca_variance_ratio_matches_the_covariance_eigenvalues() -> None:
    vectors = _random_vectors(60, 9, 11)
    projection = g.pca_2d(vectors)
    centered = vectors - vectors.mean(axis=0)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(centered, rowvar=False)))[::-1]
    expected = eigenvalues[:2] / eigenvalues.sum()
    assert projection.explained_variance == pytest.approx(list(expected), abs=1e-9)


def test_pca_recovers_a_plane_exactly() -> None:
    """Dane leżące w 2D muszą zostać odtworzone bez straty odległości."""

    plane = _random_vectors(30, 2, 12)
    embedded = np.hstack([plane, np.zeros((30, 6))])
    rotation = np.linalg.qr(_random_vectors(8, 8, 13))[0]
    projection = g.pca_2d(embedded @ rotation)
    original = g.pairwise_euclidean(plane)
    recovered = g.pairwise_euclidean(projection.coords)
    assert np.allclose(original, recovered, atol=1e-8)


# --------------------------------------------------------------------- MDS


def test_smacof_recovers_a_configuration_that_exists_in_2d() -> None:
    """Najmocniejszy test MDS: jeśli odległości są realizowalne na płaszczyźnie,
    poprawny SMACOF musi je odtworzyć, a nie tylko zbliżyć się do nich."""

    truth = _random_vectors(25, 2, 14)
    target = g.pairwise_euclidean(truth)
    projection = g.smacof(target, max_iterations=1000, tolerance=1e-12)
    recovered = g.pairwise_euclidean(projection.coords)
    assert np.allclose(target, recovered, atol=1e-4)
    assert g.kruskal_stress(target, recovered) < 1e-4


def test_smacof_keeps_the_scale_and_does_not_only_match_shape() -> None:
    truth = _random_vectors(20, 2, 15) * 7.0
    target = g.pairwise_euclidean(truth)
    recovered = g.pairwise_euclidean(g.smacof(target, max_iterations=1000).coords)
    upper = np.triu_indices(20, 1)
    ratio = recovered[upper] / target[upper]
    assert ratio.mean() == pytest.approx(1.0, abs=1e-3)
    assert ratio.std() < 1e-3


def test_smacof_is_invariant_to_rotation_of_the_input() -> None:
    truth = _random_vectors(18, 2, 16)
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    plain = g.smacof(g.pairwise_euclidean(truth), max_iterations=800)
    rotated = g.smacof(g.pairwise_euclidean(truth @ rotation), max_iterations=800)
    assert g.kruskal_stress(
        g.pairwise_euclidean(plain.coords), g.pairwise_euclidean(rotated.coords)
    ) < 1e-3


def test_smacof_never_increases_stress_between_iterations() -> None:
    target = g.cosine_distance(g.cosine_matrix(_random_vectors(30, 12, 17)))
    previous = None
    for iterations in (1, 2, 4, 8, 16, 32):
        coords = g.smacof(target, max_iterations=iterations, tolerance=0.0).coords
        stress = float(np.sum((target - g.pairwise_euclidean(coords)) ** 2))
        if previous is not None:
            assert stress <= previous + 1e-9
        previous = stress


def test_smacof_beats_pca_on_distance_faithfulness() -> None:
    vectors = _random_vectors(40, 16, 18)
    target = g.cosine_distance(g.cosine_matrix(vectors))
    mds = g.kruskal_stress(target, g.pairwise_euclidean(g.smacof(target).coords))
    pca = g.kruskal_stress(target, g.pairwise_euclidean(g.pca_2d(vectors).coords))
    assert mds < pca


def test_smacof_handles_degenerate_sizes() -> None:
    assert g.smacof(np.zeros((1, 1))).coords.shape == (1, 2)
    two = g.smacof(np.array([[0.0, 3.0], [3.0, 0.0]]))
    assert g.pairwise_euclidean(two.coords)[0, 1] == pytest.approx(3.0)


# ------------------------------------------------------------ zniekształcenie


def test_kruskal_stress_is_zero_for_a_faithful_projection() -> None:
    coords = _random_vectors(15, 2, 19)
    distance = g.pairwise_euclidean(coords)
    assert g.kruskal_stress(distance, distance) == pytest.approx(0.0, abs=1e-12)


def test_kruskal_stress_ignores_a_pure_change_of_scale() -> None:
    distance = g.pairwise_euclidean(_random_vectors(15, 2, 20))
    assert g.kruskal_stress(distance, distance * 4.2) == pytest.approx(0.0, abs=1e-9)


def test_trustworthiness_matches_the_published_formula() -> None:
    high = g.cosine_distance(g.cosine_matrix(_random_vectors(30, 10, 21)))
    low = g.pairwise_euclidean(g.pca_2d(_random_vectors(30, 10, 21)).coords)
    neighbours = 5

    count = 30
    ranks = np.zeros((count, count), dtype=int)
    for row in range(count):
        order = np.argsort([np.inf if row == col else high[row, col] for col in range(count)])
        for position, column in enumerate(order):
            ranks[row, column] = position + 1

    penalty = 0.0
    for row in range(count):
        candidates = [col for col in range(count) if col != row]
        nearest = sorted(candidates, key=lambda col: low[row, col])[:neighbours]
        for column in nearest:
            if ranks[row, column] > neighbours:
                penalty += ranks[row, column] - neighbours
    expected = 1 - (2 / (count * neighbours * (2 * count - 3 * neighbours - 1))) * penalty

    assert g.trustworthiness(high, low, neighbours) == pytest.approx(expected)


def test_trustworthiness_is_one_for_an_identical_embedding() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(24, 8, 22)))
    assert g.trustworthiness(distance, distance, 4) == pytest.approx(1.0)
    assert g.continuity(distance, distance, 4) == pytest.approx(1.0)


def test_trustworthiness_refuses_an_oversized_neighbourhood() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(12, 5, 23)))
    limit = g.max_valid_neighbours(12)
    assert g.trustworthiness(distance, distance, limit) is not None
    assert g.trustworthiness(distance, distance, limit + 1) is None


def test_a_deliberately_bad_drawing_scores_worse_than_mds() -> None:
    vectors = _random_vectors(35, 14, 24)
    high = g.cosine_distance(g.cosine_matrix(vectors))
    good = g.pairwise_euclidean(g.smacof(high).coords)
    bad = g.pairwise_euclidean(np.random.default_rng(99).normal(size=(35, 2)))
    assert g.trustworthiness(high, good, 6) > g.trustworthiness(high, bad, 6)


def test_neighbourhood_preservation_is_a_share_per_point() -> None:
    distance = g.cosine_distance(g.cosine_matrix(_random_vectors(20, 6, 25)))
    preserved = g.neighbourhood_preservation(distance, distance, 5)
    assert preserved.shape == (20,)
    assert np.allclose(preserved, 1.0)


def test_shepard_sample_is_complete_when_small_and_capped_when_large() -> None:
    small = g.cosine_distance(g.cosine_matrix(_random_vectors(10, 4, 26)))
    high, low, sampled = g.shepard_sample(small, small)
    assert sampled is False
    assert len(high) == len(low) == 10 * 9 // 2

    large = g.cosine_distance(g.cosine_matrix(_random_vectors(60, 4, 27)))
    high, low, sampled = g.shepard_sample(large, large, limit=100)
    assert sampled is True
    assert len(high) == len(low) == 100


def test_pair_histogram_counts_every_pair_exactly_once() -> None:
    similarity = g.cosine_matrix(_random_vectors(16, 5, 28))
    centres, counts = g.pair_histogram(similarity, bins=20)
    assert len(centres) == len(counts) == 20
    assert sum(counts) == 16 * 15 // 2


def test_pair_histogram_is_empty_for_a_single_point() -> None:
    centres, counts = g.pair_histogram(np.ones((1, 1)))
    assert centres == [] and counts == []
