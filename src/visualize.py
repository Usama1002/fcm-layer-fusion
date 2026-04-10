"""
Visualization utilities for CKA matrices and community detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    title: str = "Layer-wise CKA Similarity Matrix",
    save_path: str = "results/cka_matrix.png",
    communities: list[list[int]] | None = None,
):
    """
    Plot the full L x L CKA similarity heatmap.

    Args:
        cka_matrix: L x L CKA similarity matrix
        title: plot title
        save_path: where to save the figure
        communities: optional list of communities to overlay as boxes
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sns.heatmap(
        cka_matrix,
        ax=ax,
        cmap="magma",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=5,
        yticklabels=5,
        cbar_kws={"label": "CKA Similarity"},
    )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)

    # Overlay community boundaries if provided
    if communities is not None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(communities)))
        for cidx, comm in enumerate(communities):
            if len(comm) <= 1:
                continue
            sorted_layers = sorted(comm)
            # Draw rectangles around community members
            for i in sorted_layers:
                for j in sorted_layers:
                    rect = plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor=colors[cidx],
                        linewidth=0.5,
                        alpha=0.7,
                    )
                    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CKA matrix plot to {save_path}")


def plot_cka_off_diagonal_analysis(
    cka_matrix: np.ndarray,
    save_path: str = "results/cka_off_diagonal.png",
):
    """
    Analyze and plot off-diagonal similarity patterns.
    Shows whether non-adjacent layers have high CKA.
    """
    L = cka_matrix.shape[0]

    # Compute average CKA as a function of distance between layers
    max_dist = L - 1
    avg_cka_by_dist = []
    std_cka_by_dist = []

    for d in range(max_dist + 1):
        values = []
        for i in range(L - d):
            values.append(cka_matrix[i, i + d])
        avg_cka_by_dist.append(np.mean(values))
        std_cka_by_dist.append(np.std(values))

    avg_cka_by_dist = np.array(avg_cka_by_dist)
    std_cka_by_dist = np.array(std_cka_by_dist)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average CKA vs layer distance
    ax = axes[0]
    distances = np.arange(max_dist + 1)
    ax.plot(distances, avg_cka_by_dist, "b-", linewidth=2)
    ax.fill_between(
        distances,
        avg_cka_by_dist - std_cka_by_dist,
        avg_cka_by_dist + std_cka_by_dist,
        alpha=0.2,
    )
    ax.set_xlabel("Layer Distance")
    ax.set_ylabel("Average CKA")
    ax.set_title("CKA Decay with Layer Distance")
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of high-CKA pairs by adjacency
    ax = axes[1]
    threshold = 0.8
    adjacent_high = []
    nonadjacent_high = []
    for i in range(L):
        for j in range(i + 1, L):
            if cka_matrix[i, j] >= threshold:
                if abs(i - j) <= 1:
                    adjacent_high.append(cka_matrix[i, j])
                else:
                    nonadjacent_high.append(cka_matrix[i, j])

    labels = ["Adjacent\n(dist=1)", f"Non-adjacent\n(dist>1)"]
    counts = [len(adjacent_high), len(nonadjacent_high)]
    bars = ax.bar(labels, counts, color=["steelblue", "coral"])
    ax.set_ylabel(f"Number of pairs with CKA >= {threshold}")
    ax.set_title("High-Similarity Pairs: Adjacent vs Non-Adjacent")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved off-diagonal analysis to {save_path}")

    # Print summary statistics
    total_pairs = L * (L - 1) // 2
    high_pairs = len(adjacent_high) + len(nonadjacent_high)
    print(f"\n--- Off-Diagonal Analysis ---")
    print(f"Total layer pairs: {total_pairs}")
    print(f"Pairs with CKA >= {threshold}: {high_pairs} ({100*high_pairs/total_pairs:.1f}%)")
    print(f"  Adjacent (dist=1): {len(adjacent_high)}")
    print(f"  Non-adjacent (dist>1): {len(nonadjacent_high)}")
    if len(nonadjacent_high) > 0:
        print(f"  => NON-ADJACENT HIGH-SIMILARITY EXISTS! Approach A is viable.")
    else:
        print(f"  => No non-adjacent high similarity found. May need to lower threshold or reconsider.")

    return {
        "avg_cka_by_distance": avg_cka_by_dist,
        "std_cka_by_distance": std_cka_by_dist,
        "num_adjacent_high": len(adjacent_high),
        "num_nonadjacent_high": len(nonadjacent_high),
    }
