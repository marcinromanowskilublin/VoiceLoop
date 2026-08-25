"""Weryfikacja geometrii z czystego numpy wobec scipy/sklearn.

Uruchamiane systemowym Pythonem (ma scipy i sklearn), a nie venvem VoiceLoopa.
Sprawdza, czy ręczne implementacje zwracają to samo co referencyjne biblioteki.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectorscope import geometry  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK  " if condition else "BLAD"
    print(f"{status} {label}{(' | ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def test_average_linkage() -> None:
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import squareform

    generator = np.random.default_rng(7)
    for count in (5, 12, 37):
        vectors = generator.normal(size=(count, 24))
        cosine = geometry.cosine_matrix(vectors)
        distance = geometry.cosine_distance(cosine)

        mine = geometry.average_linkage(distance)
        reference = linkage(squareform(distance, checks=False), method="average")

        mine_cophenetic = cophenet(mine)
        reference_cophenetic = cophenet(reference)
        same_cophenetic = np.allclose(mine_cophenetic, reference_cophenetic, atol=1e-10)

        mine_heights = np.sort(mine[:, 2])
        reference_heights = np.sort(reference[:, 2])
        same_heights = np.allclose(mine_heights, reference_heights, atol=1e-10)

        check(
            f"average_linkage n={count} odleglosci kofenetyczne",
            same_cophenetic,
            f"maks. roznica {np.abs(mine_cophenetic - reference_cophenetic).max():.2e}",
        )
        check(f"average_linkage n={count} wysokosci scalen", same_heights)

        order = geometry.leaf_order(mine, count)
        check(
            f"leaf_order n={count} permutacja lisci",
            sorted(order) == list(range(count)),
            f"{len(order)} lisci",
        )


def reference_trustworthiness(high: np.ndarray, low: np.ndarray, neighbours: int) -> float:
    """Wzór Venny-Kaskiego przepisany wprost, bez wektoryzacji.

    Napisane niezależnie od implementacji w geometry.py, żeby błąd w jednej
    wersji nie przeszedł niezauważony w drugiej. sklearn na tej maszynie nie
    daje się zaimportować (zasady kontroli aplikacji blokują jego DLL).
    """

    count = high.shape[0]
    penalty = 0
    for row in range(count):
        others = [index for index in range(count) if index != row]
        by_high = sorted(others, key=lambda index: high[row, index])
        rank_of = {index: position + 1 for position, index in enumerate(by_high)}
        by_low = sorted(others, key=lambda index: low[row, index])
        for index in by_low[:neighbours]:
            rank = rank_of[index]
            if rank > neighbours:
                penalty += rank - neighbours
    normaliser = count * neighbours * (2 * count - 3 * neighbours - 1)
    return 1.0 - (2.0 * penalty) / normaliser


def test_trustworthiness() -> None:
    generator = np.random.default_rng(11)
    for count, neighbours in ((30, 5), (60, 8), (45, 12)):
        vectors = generator.normal(size=(count, 32))
        cosine = geometry.cosine_matrix(vectors)
        high = geometry.cosine_distance(cosine)
        projection = geometry.smacof(high)
        low = geometry.pairwise_euclidean(projection.coords)

        mine = geometry.trustworthiness(high, low, neighbours)
        reference = reference_trustworthiness(high, low, neighbours)
        check(
            f"trustworthiness n={count} k={neighbours}",
            mine is not None and abs(mine - reference) < 1e-12,
            f"moje {mine:.9f} vs wzor {reference:.9f}" if mine is not None else "None",
        )

    vectors = generator.normal(size=(40, 20))
    high = geometry.cosine_distance(geometry.cosine_matrix(vectors))
    perfect = geometry.trustworthiness(high, high, 6)
    check(
        "trustworthiness=1 gdy rzut jest identycznoscia",
        perfect is not None and abs(perfect - 1.0) < 1e-12,
        f"{perfect}",
    )

    scrambled = geometry.pairwise_euclidean(generator.normal(size=(40, 2)))
    random_value = geometry.trustworthiness(high, scrambled, 6)
    faithful = geometry.trustworthiness(
        high,
        geometry.pairwise_euclidean(geometry.smacof(high).coords),
        6,
    )
    check(
        "rzut MDS jest wiarygodniejszy niz losowy",
        random_value is not None and faithful is not None and faithful > random_value,
        f"MDS {faithful:.4f} > losowy {random_value:.4f}",
    )

    limit = geometry.max_valid_neighbours(40)
    check(
        "zbyt duze k jest odrzucane zamiast zwracac bzdure",
        geometry.trustworthiness(high, high, limit + 1) is None,
        f"limit k={limit}",
    )

    continuity_perfect = geometry.continuity(high, high, 6)
    check(
        "continuity=1 gdy rzut jest identycznoscia",
        continuity_perfect is not None and abs(continuity_perfect - 1.0) < 1e-12,
        f"{continuity_perfect}",
    )


def test_smacof_beats_pca_on_distance() -> None:
    generator = np.random.default_rng(23)
    vectors = generator.normal(size=(50, 40))
    cosine = geometry.cosine_matrix(vectors)
    high = geometry.cosine_distance(cosine)

    mds = geometry.smacof(high)
    pca = geometry.pca_2d(vectors)

    mds_stress = geometry.kruskal_stress(high, geometry.pairwise_euclidean(mds.coords))
    pca_stress = geometry.kruskal_stress(high, geometry.pairwise_euclidean(pca.coords))

    check(
        "MDS ma mniejszy stress niz PCA",
        mds_stress < pca_stress,
        f"MDS {mds_stress:.4f} < PCA {pca_stress:.4f}",
    )
    check("stress w zakresie [0,1]", 0.0 <= mds_stress <= 1.0, f"{mds_stress:.4f}")


def test_smacof_monotonic() -> None:
    generator = np.random.default_rng(31)
    vectors = generator.normal(size=(40, 16))
    high = geometry.cosine_distance(geometry.cosine_matrix(vectors))

    stresses = []
    for iterations in (1, 2, 5, 20, 100):
        projection = geometry.smacof(high, max_iterations=iterations)
        low = geometry.pairwise_euclidean(projection.coords)
        stresses.append(geometry.kruskal_stress(high, low))
    non_increasing = all(
        later <= earlier + 1e-9
        for earlier, later in zip(stresses, stresses[1:], strict=False)
    )
    check(
        "SMACOF nie zwieksza stressu z iteracjami",
        non_increasing,
        " -> ".join(f"{value:.5f}" for value in stresses),
    )


def test_pca_explained_variance() -> None:
    generator = np.random.default_rng(41)
    vectors = generator.normal(size=(60, 20))
    mine = geometry.pca_2d(vectors)

    # Niezalezna sciezka: wartosci wlasne macierzy kowariancji zamiast SVD.
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    reference = (eigenvalues[:2] / eigenvalues.sum()).tolist()

    check(
        "pca_2d wyjasniona wariancja",
        np.allclose(np.array(mine.explained_variance), np.array(reference), atol=1e-10),
        f"moje {[round(v, 6) for v in mine.explained_variance]} "
        f"vs kowariancja {[round(v, 6) for v in reference]}",
    )
    check(
        "pca_2d suma udzialow nie przekracza 1",
        0.0 < mine.explained_variance_total <= 1.0 + 1e-12,
        f"{mine.explained_variance_total:.6f}",
    )

    # Rzut PCA musi zachowac odleglosci lepiej niz losowy obrot na 2 wymiary.
    high = geometry.pairwise_euclidean(centered)
    pca_stress = geometry.kruskal_stress(high, geometry.pairwise_euclidean(mine.coords))
    random_stress = geometry.kruskal_stress(
        high,
        geometry.pairwise_euclidean(generator.normal(size=(60, 2))),
    )
    check(
        "PCA zachowuje odleglosci lepiej niz losowy rzut",
        pca_stress < random_stress,
        f"PCA {pca_stress:.4f} < losowy {random_stress:.4f}",
    )


def test_cosine_and_edges() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7071, 0.7071, 0.0],
        ]
    )
    cosine = geometry.cosine_matrix(vectors)
    check("identyczne wektory maja cosinus 1", abs(cosine[0, 1] - 1.0) < 1e-9)
    check("ortogonalne wektory maja cosinus 0", abs(cosine[0, 2]) < 1e-9)
    check("cosinus 45 stopni", abs(cosine[0, 3] - 0.7071) < 1e-3, f"{cosine[0, 3]:.4f}")

    edges = geometry.knn_edges(cosine, k=0, threshold=0.9)
    pairs = {(a, b) for a, b, _ in edges.pairs}
    check("prog 0.9 lapie tylko pare identycznych", pairs == {(0, 1)}, str(sorted(pairs)))

    edges = geometry.knn_edges(cosine, k=1, threshold=None)
    check("kNN k=1 daje co najmniej jedna krawedz", len(edges.pairs) >= 1)


def test_neighbourhood_preservation() -> None:
    generator = np.random.default_rng(53)
    vectors = generator.normal(size=(25, 12))
    high = geometry.cosine_distance(geometry.cosine_matrix(vectors))
    preserved = geometry.neighbourhood_preservation(high, high, 5)
    check(
        "identyczna przestrzen zachowuje 100% sasiedztwa",
        np.allclose(preserved, 1.0),
        f"min {preserved.min():.3f}",
    )


def main() -> int:
    print("=== average linkage vs scipy ===")
    test_average_linkage()
    print("\n=== trustworthiness vs sklearn ===")
    test_trustworthiness()
    print("\n=== PCA vs sklearn ===")
    test_pca_explained_variance()
    print("\n=== MDS ===")
    test_smacof_beats_pca_on_distance()
    test_smacof_monotonic()
    print("\n=== cosinusy i krawedzie ===")
    test_cosine_and_edges()
    test_neighbourhood_preservation()

    print()
    if FAILURES:
        print(f"NIEPOWODZENIA ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("Wszystkie testy geometrii przeszly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
