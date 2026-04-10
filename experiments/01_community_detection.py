"""
Experiment 01: Community Detection on CKA Graph

Loads the CKA matrix from Experiment 00 and runs community detection
to find natural fusion groups. Tests multiple methods and resolution
parameters. Visualizes communities overlaid on the CKA matrix.

Usage:
    python experiments/01_community_detection.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.community import detect_communities, analyze_communities, print_communities
from src.visualize import plot_cka_matrix


def main():
    output_dir = "results/exp01"
    os.makedirs(output_dir, exist_ok=True)

    # Load CKA matrix from Exp 00
    matrix_path = "results/exp00/cka_matrix_Qwen2.5-7B.npy"
    if not os.path.exists(matrix_path):
        print(f"CKA matrix not found at {matrix_path}. Run experiment 00 first.")
        return

    cka_matrix = np.load(matrix_path)
    L = cka_matrix.shape[0]
    print(f"Loaded CKA matrix: {L} x {L}")

    # Test multiple configurations
    configs = [
        # (method, resolution/n_clusters, threshold, label)
        ("leiden", 0.5, 0.0, "leiden_res0.5"),
        ("leiden", 0.8, 0.0, "leiden_res0.8"),
        ("leiden", 1.0, 0.0, "leiden_res1.0"),
        ("leiden", 1.5, 0.0, "leiden_res1.5"),
        ("leiden", 1.0, 0.5, "leiden_res1.0_thresh0.5"),
        ("louvain", None, 0.0, "louvain"),
        ("spectral", 4, 0.0, "spectral_k4"),
        ("spectral", 6, 0.0, "spectral_k6"),
        ("spectral", 8, 0.0, "spectral_k8"),
    ]

    all_results = []

    for method, param, threshold, label in configs:
        print(f"\n--- {label} ---")
        kwargs = {"method": method, "threshold": threshold}
        if method == "spectral":
            kwargs["n_clusters"] = param
        elif method == "leiden":
            kwargs["resolution"] = param

        communities = detect_communities(cka_matrix, **kwargs)
        stats = analyze_communities(communities, L)
        print_communities(communities, stats)

        # Visualize
        plot_cka_matrix(
            cka_matrix,
            title=f"CKA + Communities ({label})",
            save_path=os.path.join(output_dir, f"cka_communities_{label}.png"),
            communities=communities,
        )

        all_results.append({
            "label": label,
            "communities": communities,
            "stats": stats,
        })

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"{'Config':<30} {'#Comm':>6} {'Comp.':>6} {'NonAdj':>7} {'MaxSpan':>8}")
    print(f"{'-'*70}")
    for r in all_results:
        s = r["stats"]
        print(
            f"{r['label']:<30} {s['num_communities']:>6} "
            f"{s['compression_ratio']:>6.2f} {s['non_adjacent_communities']:>7} "
            f"{s['max_span']:>8}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
