#!/usr/bin/env python3
"""
Reproduce the main results table (Table 4) from the paper:

  "Beyond Adjacent Layers: Graph-Guided Layer Fusion for
   Compressing Large Language Models"

Runs FCM and ShortGPT on all 11 models at 2x compression with 3 seeds,
then prints the results table matching the paper.

Usage:
    python reproduce.py                                      # all 11 models
    python reproduce.py --models Qwen/Qwen2.5-7B             # single model
    python reproduce.py --models Qwen/Qwen2.5-7B Qwen/Qwen3-4B  # subset
    python reproduce.py --seeds 42                            # single seed (fast)
    python reproduce.py --device cpu                          # CPU mode (slow)
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fcm import (
    compute_sim_matrix,
    select_layers_fcm,
    build_compressed_model,
    _set_seed,
    _load_model,
    _free_model,
)
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity

# All 11 models from the paper, ordered by family and size
ALL_MODELS = [
    "Qwen/Qwen3-1.7B",
    "google/gemma-2-2b",
    "stabilityai/stablelm-2-1_6b",
    "Qwen/Qwen3-4B",
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.1-8B",
]

# Model family mapping for display
FAMILY = {
    "Qwen/Qwen3-1.7B": "Qwen3",
    "google/gemma-2-2b": "Gemma",
    "stabilityai/stablelm-2-1_6b": "StableLM",
    "Qwen/Qwen3-4B": "Qwen3",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM",
    "meta-llama/Llama-3.2-3B": "LLaMA",
    "Qwen/Qwen2.5-3B": "Qwen2",
    "Qwen/Qwen3-8B": "Qwen3",
    "Qwen/Qwen2.5-7B": "Qwen2",
    "mistralai/Mistral-7B-v0.3": "Mistral",
    "meta-llama/Llama-3.1-8B": "LLaMA",
}

# Default similarity metric per model (cosine works best for most)
METRICS = {
    "Qwen/Qwen3-1.7B": "cka",
    "google/gemma-2-2b": "cosine",
    "stabilityai/stablelm-2-1_6b": "cosine",
    "Qwen/Qwen3-4B": "cka",
    "HuggingFaceTB/SmolLM3-3B": "cka",
    "meta-llama/Llama-3.2-3B": "cosine",
    "Qwen/Qwen2.5-3B": "cosine",
    "Qwen/Qwen3-8B": "cosine",
    "Qwen/Qwen2.5-7B": "cosine",
    "mistralai/Mistral-7B-v0.3": "cosine",
    "meta-llama/Llama-3.1-8B": "cosine",
}

# LoRA recovery hyperparameters (fixed across all experiments)
LORA_KW = dict(num_steps=500, lr=2e-4, lora_r=16, num_samples=2048)


def run_fcm_single(model_name, sim_matrix, bi_scores, target, seed, device):
    """Run FCM compression for a single model and seed."""
    _set_seed(seed)
    keep = select_layers_fcm(sim_matrix, bi_scores, target)
    model, tokenizer = _load_model(model_name, device=device)
    model = build_compressed_model(model, keep)
    _set_seed(seed)
    model, _ = lora_recovery_finetune(model, tokenizer, device=device, **LORA_KW)
    ppl = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=device, max_samples=50)
    _free_model(model)
    return ppl


def run_shortgpt_single(model_name, bi_scores, total_layers, target, seed, device):
    """Run ShortGPT compression for a single model and seed."""
    _set_seed(seed)
    model, tokenizer = _load_model(model_name, device=device)
    model, _ = remove_layers_by_bi(model, bi_scores, total_layers - target)
    _set_seed(seed)
    model, _ = lora_recovery_finetune(model, tokenizer, device=device, **LORA_KW)
    ppl = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=device, max_samples=50)
    _free_model(model)
    return ppl


def run_experiment(models, seeds, device, output_dir):
    """Run the full experiment for the given models and seeds."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "reproduce_results.json")
    all_results = {}

    # Load existing partial results if available
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {results_path}")

    for model_name in models:
        short = model_name.split("/")[-1]
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        total_layers = config.num_hidden_layers
        target = max(total_layers // 2, 4)
        metric = METRICS.get(model_name, "cosine")

        print(f"\n{'#' * 70}")
        print(f"# {short} ({total_layers} -> {target} layers, metric={metric})")
        print(f"{'#' * 70}")

        # Step 1: Compute similarity matrix
        print("Computing similarity matrix...")
        sim_matrix = compute_sim_matrix(
            model_name, metric=metric, device=device,
        )

        # Step 2: Compute BI scores and baseline perplexity
        print("Computing Block Influence scores and baseline perplexity...")
        model, tokenizer = _load_model(model_name, device=device)
        bi_scores = compute_block_influence(model, tokenizer, device=device)
        baseline_ppl = evaluate_perplexity(
            model, tokenizer, max_seq_len=2048, device=device, max_samples=50
        )
        _free_model(model)
        print(f"  Baseline perplexity: {baseline_ppl:.2f}")

        # Step 3: Run FCM and ShortGPT for each seed
        fcm_ppls = []
        sg_ppls = []
        for seed in seeds:
            print(f"\n  Seed {seed}:", end=" ", flush=True)

            sg_ppl = run_shortgpt_single(
                model_name, bi_scores, total_layers, target, seed, device
            )
            sg_ppls.append(sg_ppl)
            print(f"ShortGPT={sg_ppl:.1f}", end=" ", flush=True)

            fcm_ppl = run_fcm_single(
                model_name, sim_matrix, bi_scores, target, seed, device
            )
            fcm_ppls.append(fcm_ppl)
            print(f"FCM={fcm_ppl:.1f}")

        # Aggregate
        sg_mean, sg_std = np.mean(sg_ppls), np.std(sg_ppls)
        fcm_mean, fcm_std = np.mean(fcm_ppls), np.std(fcm_ppls)
        delta = (fcm_mean - sg_mean) / sg_mean * 100

        all_results[model_name] = {
            "short_name": short,
            "family": FAMILY.get(model_name, "Unknown"),
            "total_layers": total_layers,
            "target_layers": target,
            "baseline": float(baseline_ppl),
            "shortgpt": {"mean": float(sg_mean), "std": float(sg_std), "raw": [float(p) for p in sg_ppls]},
            "fcm": {"mean": float(fcm_mean), "std": float(fcm_std), "raw": [float(p) for p in fcm_ppls]},
            "delta_pct": float(delta),
        }

        print(f"\n  Summary: ShortGPT={sg_mean:.2f}+/-{sg_std:.2f}  "
              f"FCM={fcm_mean:.2f}+/-{fcm_std:.2f}  delta={delta:+.1f}%")

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    return all_results


def print_results_table(results):
    """Print the results table matching the paper's Table 4 format."""
    if not results:
        print("No results to display.")
        return

    # Sort by delta (best FCM improvement first)
    fcm_wins = {k: v for k, v in results.items() if v["delta_pct"] <= 0}
    sg_wins = {k: v for k, v in results.items() if v["delta_pct"] > 0}
    sorted_fcm = sorted(fcm_wins.items(), key=lambda x: x[1]["delta_pct"])
    sorted_sg = sorted(sg_wins.items(), key=lambda x: x[1]["delta_pct"])

    print(f"\n{'=' * 100}")
    print("TABLE: Multi-model comparison at 2x compression")
    print(f"       (mean +/- std over {len(next(iter(results.values()))['shortgpt']['raw'])} random seeds, "
          f"LoRA r=16, 500 steps)")
    print(f"{'=' * 100}")
    header = (f"{'Model':<22} {'Family':<10} {'Base':>6} {'ShortGPT':>18} "
              f"{'FCM (ours)':>18} {'Delta':>8}")
    print(header)
    print("-" * 100)

    if sorted_fcm:
        print(f"  FCM wins ({len(sorted_fcm)}/{len(results)}):")
        for name, r in sorted_fcm:
            sg = r["shortgpt"]
            fcm = r["fcm"]
            line = (f"  {r['short_name']:<20} {r['family']:<10} {r['baseline']:>6.2f} "
                    f"{sg['mean']:>7.2f} +/- {sg['std']:>5.2f} "
                    f"{fcm['mean']:>7.2f} +/- {fcm['std']:>5.2f} "
                    f"{r['delta_pct']:>+7.1f}%")
            print(line)

    if sorted_sg:
        print("-" * 100)
        print(f"  ShortGPT wins ({len(sorted_sg)}/{len(results)}):")
        for name, r in sorted_sg:
            sg = r["shortgpt"]
            fcm = r["fcm"]
            line = (f"  {r['short_name']:<20} {r['family']:<10} {r['baseline']:>6.2f} "
                    f"{sg['mean']:>7.2f} +/- {sg['std']:>5.2f} "
                    f"{fcm['mean']:>7.2f} +/- {fcm['std']:>5.2f} "
                    f"{r['delta_pct']:>+7.1f}%")
            print(line)

    print("=" * 100)

    # Summary statistics
    deltas = [r["delta_pct"] for r in results.values()]
    fcm_win_count = sum(1 for d in deltas if d < 0)
    print(f"\nFCM wins: {fcm_win_count}/{len(results)} models")
    if sorted_fcm:
        best = sorted_fcm[0]
        print(f"Best improvement: {best[1]['delta_pct']:+.1f}% ({best[1]['short_name']})")
    print(f"Mean delta: {np.mean(deltas):+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the main results table from the FCM paper."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names to evaluate (default: all 11 models from the paper)."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 7],
        help="Random seeds (default: 42 123 7)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for computation (default: cuda)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/reproduce",
        help="Directory for output files (default: results/reproduce)."
    )
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS
    # Validate model names
    for m in models:
        if m not in ALL_MODELS:
            print(f"Warning: {m} is not in the paper's model list. Proceeding anyway.")

    print(f"Models: {len(models)}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    start = time.time()
    results = run_experiment(models, args.seeds, args.device, args.output_dir)
    elapsed = time.time() - start

    print_results_table(results)
    print(f"\nTotal time: {elapsed / 3600:.1f} hours")
    print(f"Results saved to: {os.path.join(args.output_dir, 'reproduce_results.json')}")


if __name__ == "__main__":
    main()
