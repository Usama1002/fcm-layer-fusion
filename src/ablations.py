"""
Ablation study utilities.

Implements alternative grouping strategies for comparison:
1. Fisher Optimal Segmentation (contiguous only, like SGLP)
2. Random grouping (baseline)
3. Uniform grouping (equal-sized contiguous blocks)
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def fisher_segmentation(cka_matrix: np.ndarray, n_segments: int) -> list[list[int]]:
    """
    Fisher Optimal Segmentation on the CKA diagonal.

    This enforces contiguous groups (like SGLP) — layers in the same
    segment must be adjacent. Uses dynamic programming to minimize
    within-segment variance of CKA similarities.

    Simplified version: uses hierarchical clustering with
    ward linkage on the ordered distance matrix.
    """
    L = cka_matrix.shape[0]

    # Extract adjacent-layer similarities as features
    # Use each row of the CKA matrix as a feature vector for that layer
    features = cka_matrix.copy()

    # Constrained clustering: only allow merging adjacent clusters
    # Use ward linkage on ordered features
    dist_matrix = 1.0 - features
    np.fill_diagonal(dist_matrix, 0)

    # For contiguous segmentation, we use a 1D dynamic programming approach
    # Minimize total within-segment dissimilarity
    # Cost of segment [i, j] = sum of pairwise distances within the segment
    def segment_cost(i, j):
        if i == j:
            return 0
        sub = cka_matrix[i:j+1, i:j+1]
        # Cost = total dissimilarity = sum(1 - CKA) for all pairs in segment
        n = j - i + 1
        return n * (n - 1) / 2 - (sub.sum() - np.trace(sub)) / 2

    # DP: dp[k][j] = min cost to partition layers [0..j] into k segments
    INF = float("inf")
    dp = [[INF] * L for _ in range(n_segments + 1)]
    split = [[0] * L for _ in range(n_segments + 1)]

    # Base case: 1 segment
    for j in range(L):
        dp[1][j] = segment_cost(0, j)

    # Fill DP
    for k in range(2, n_segments + 1):
        for j in range(k - 1, L):
            for i in range(k - 2, j):
                cost = dp[k-1][i] + segment_cost(i+1, j)
                if cost < dp[k][j]:
                    dp[k][j] = cost
                    split[k][j] = i

    # Backtrack to find segments
    segments = []
    k = n_segments
    j = L - 1
    while k > 1:
        i = split[k][j]
        segments.append(list(range(i + 1, j + 1)))
        j = i
        k -= 1
    segments.append(list(range(0, j + 1)))
    segments.reverse()

    return segments


def uniform_grouping(L: int, n_groups: int) -> list[list[int]]:
    """
    Uniform contiguous grouping: divide L layers into n_groups equal-sized blocks.
    """
    base_size = L // n_groups
    remainder = L % n_groups
    groups = []
    start = 0
    for i in range(n_groups):
        size = base_size + (1 if i < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def random_grouping(L: int, n_groups: int, seed: int = 42) -> list[list[int]]:
    """
    Random grouping: randomly assign layers to groups.
    """
    rng = np.random.RandomState(seed)
    assignments = rng.randint(0, n_groups, size=L)
    # Ensure each group has at least one member
    for g in range(n_groups):
        if g not in assignments:
            assignments[rng.randint(0, L)] = g

    groups = {}
    for idx, g in enumerate(assignments):
        groups.setdefault(int(g), []).append(idx)

    return [sorted(v) for v in sorted(groups.values(), key=lambda x: x[0])]
