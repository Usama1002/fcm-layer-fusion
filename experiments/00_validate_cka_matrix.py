"""
Experiment 0: Validate the core assumption of Approach A.

Computes the full L x L CKA similarity matrix for a 7B LLM and checks
whether non-adjacent layers show high functional similarity.

This is the make-or-break experiment: if the CKA matrix only shows
adjacent similarity (tri-diagonal), non-consecutive fusion won't help.
If it shows block-diagonal or off-diagonal structure, we have a paper.

Usage:
    python experiments/00_validate_cka_matrix.py --model meta-llama/Llama-2-7b-hf
    python experiments/00_validate_cka_matrix.py --model Qwen/Qwen2.5-7B
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.profiler import collect_activations
from src.cka import compute_cka_matrix
from src.visualize import plot_cka_matrix, plot_cka_off_diagonal_analysis


def main():
    parser = argparse.ArgumentParser(description="Validate CKA matrix structure for LLMs")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name",
    )
    parser.add_argument("--num_samples", type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default="results/exp00", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_short = args.model.split("/")[-1]
    print(f"=" * 60)
    print(f"Experiment 0: CKA Matrix Validation")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}, Seq len: {args.max_seq_len}")
    print(f"=" * 60)

    # Step 1: Collect activations
    activations = collect_activations(
        model_name=args.model,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Step 2: Compute full CKA matrix
    print("\nComputing full CKA matrix...")
    cka_matrix = compute_cka_matrix(activations)

    # Save raw matrix
    matrix_path = os.path.join(args.output_dir, f"cka_matrix_{model_short}.npy")
    np.save(matrix_path, cka_matrix)
    print(f"Saved CKA matrix to {matrix_path}")

    # Step 3: Visualize
    plot_cka_matrix(
        cka_matrix,
        title=f"CKA Similarity Matrix — {model_short}",
        save_path=os.path.join(args.output_dir, f"cka_matrix_{model_short}.png"),
    )

    stats = plot_cka_off_diagonal_analysis(
        cka_matrix,
        save_path=os.path.join(args.output_dir, f"cka_off_diagonal_{model_short}.png"),
    )

    # Step 4: Print verdict
    print(f"\n{'=' * 60}")
    print("VERDICT:")
    if stats["num_nonadjacent_high"] > 0:
        ratio = stats["num_nonadjacent_high"] / max(1, stats["num_adjacent_high"])
        print(f"  Non-adjacent high-CKA pairs: {stats['num_nonadjacent_high']}")
        print(f"  Adjacent high-CKA pairs: {stats['num_adjacent_high']}")
        print(f"  Ratio (non-adj / adj): {ratio:.2f}")
        print(f"  => APPROACH A IS VIABLE. Non-consecutive fusion has empirical support.")
    else:
        print(f"  No non-adjacent pairs with CKA >= 0.8 found.")
        print(f"  Checking lower threshold (0.7)...")
        # Re-check with lower threshold
        L = cka_matrix.shape[0]
        count_07 = sum(
            1 for i in range(L) for j in range(i + 2, L) if cka_matrix[i, j] >= 0.7
        )
        print(f"  Non-adjacent pairs with CKA >= 0.7: {count_07}")
        if count_07 > 0:
            print(f"  => Marginal support. Consider adjusting threshold for community detection.")
        else:
            print(f"  => APPROACH A MAY NOT HOLD. Consider pivoting to Approach B or C.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
