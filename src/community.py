"""
Graph community detection for layer grouping.

Builds a weighted graph from the CKA similarity matrix and runs
community detection to find natural fusion groups.
"""

import numpy as np
import igraph as ig
import leidenalg


def build_similarity_graph(
    cka_matrix: np.ndarray,
    threshold: float = 0.0,
) -> ig.Graph:
    """
    Build a weighted undirected graph from a CKA similarity matrix.

    Args:
        cka_matrix: L x L CKA similarity matrix
        threshold: minimum CKA to include an edge (0 = keep all)

    Returns:
        igraph Graph with edge weights
    """
    L = cka_matrix.shape[0]
    edges = []
    weights = []

    for i in range(L):
        for j in range(i + 1, L):
            if cka_matrix[i, j] >= threshold:
                edges.append((i, j))
                weights.append(float(cka_matrix[i, j]))

    g = ig.Graph(n=L, edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["label"] = [str(i) for i in range(L)]

    return g


def detect_communities(
    cka_matrix: np.ndarray,
    method: str = "leiden",
    resolution: float = 1.0,
    threshold: float = 0.0,
    n_clusters: int | None = None,
) -> list[list[int]]:
    """
    Detect communities of functionally similar layers.

    Args:
        cka_matrix: L x L CKA similarity matrix
        method: 'leiden', 'louvain', or 'spectral'
        resolution: resolution parameter (higher = more communities)
        threshold: minimum CKA to include edge
        n_clusters: number of clusters (only for spectral)

    Returns:
        List of communities, each a list of layer indices
    """
    g = build_similarity_graph(cka_matrix, threshold=threshold)

    if method == "leiden":
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
        )
    elif method == "louvain":
        partition = g.community_multilevel(weights="weight")
    elif method == "spectral":
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage

        # Convert similarity to distance
        dist_matrix = 1.0 - cka_matrix
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="ward")
        if n_clusters is None:
            n_clusters = max(2, cka_matrix.shape[0] // 4)
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        communities = {}
        for idx, label in enumerate(labels):
            communities.setdefault(label, []).append(idx)
        return list(communities.values())
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract communities from partition
    communities = []
    for comm_idx in range(len(partition)):
        members = list(partition[comm_idx])
        communities.append(sorted(members))

    return sorted(communities, key=lambda c: c[0])


def analyze_communities(communities: list[list[int]], L: int) -> dict:
    """
    Analyze detected communities for non-adjacency and compression stats.
    """
    stats = {
        "num_communities": len(communities),
        "community_sizes": [len(c) for c in communities],
        "compression_ratio": L / len(communities),
        "non_adjacent_communities": 0,
        "max_span": 0,
    }

    for comm in communities:
        if len(comm) > 1:
            span = max(comm) - min(comm) + 1
            stats["max_span"] = max(stats["max_span"], span)
            # Check if community contains non-adjacent layers
            sorted_comm = sorted(comm)
            for k in range(len(sorted_comm) - 1):
                if sorted_comm[k + 1] - sorted_comm[k] > 1:
                    stats["non_adjacent_communities"] += 1
                    break

    return stats


def print_communities(communities: list[list[int]], stats: dict):
    """Pretty-print community detection results."""
    print(f"\n{'='*50}")
    print(f"Community Detection Results")
    print(f"{'='*50}")
    print(f"Number of communities: {stats['num_communities']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Communities with non-adjacent layers: {stats['non_adjacent_communities']}")
    print(f"Max community span: {stats['max_span']} layers")
    print(f"\nCommunity breakdown:")
    for i, comm in enumerate(communities):
        size = len(comm)
        is_contiguous = all(comm[j+1] - comm[j] == 1 for j in range(len(comm)-1))
        tag = "contiguous" if is_contiguous else "NON-ADJACENT"
        print(f"  C{i}: layers {comm} (size={size}, {tag})")
    print(f"{'='*50}")
