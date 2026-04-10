"""
Experiment 04: Compression Ratio Sweep

Compares FCM vs ShortGPT across multiple compression levels:
  - Moderate: 28 -> 20 layers (1.4x, 29% reduction)
  - Medium:   28 -> 14 layers (2.0x, 50% reduction)
  - High:     28 -> 10 layers (2.8x, 64% reduction)
  - Extreme:  28 -> 7  layers (4.0x, 75% reduction)

Also includes ablation: FCM with contiguous-only constraint vs full graph.

Usage:
    python experiments/04_compression_sweep.py --model Qwen/Qwen2.5-7B
"""

import argparse
import copy
import gc
import json
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cka import compute_cka_matrix
from src.community import detect_communities, analyze_communities, print_communities
from src.fusion import distill_community, build_fused_model, get_all_layers
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity


def load_fresh(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    return model, tokenizer


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def prepare_calibration(tokenizer, num_samples=512, max_seq_len=128):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for s in ds:
        if len(s.get("text", "")) > 50:
            texts.append(s["text"])
        if len(texts) >= num_samples:
            break
    return tokenizer(texts, max_length=max_seq_len, truncation=True,
                     padding="max_length", return_tensors="pt")


def run_shortgpt(model, tokenizer, num_remove, device, lora_steps, lora_lr, lora_r, lora_samples, ppl_samples):
    """Run ShortGPT: compute BI, remove layers, evaluate, LoRA, evaluate."""
    bi = compute_block_influence(model, tokenizer, device=device)
    model, kept = remove_layers_by_bi(model, bi, num_remove)
    n = len(get_all_layers(model))
    print(f"  ShortGPT kept layers: {kept}")

    ppl_raw = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=device, max_samples=ppl_samples)
    print(f"  PPL (raw): {ppl_raw:.2f}")

    model, _ = lora_recovery_finetune(
        model, tokenizer, num_steps=lora_steps, lr=lora_lr,
        lora_r=lora_r, num_samples=lora_samples, device=device,
    )
    ppl_lora = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=device, max_samples=ppl_samples)
    print(f"  PPL (LoRA): {ppl_lora:.2f}")

    return {"layers": n, "ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "kept": kept}


def run_fcm(model, tokenizer, cka_matrix, target_layers, cal_enc, device,
            distill_steps, distill_lr, lora_steps, lora_lr, lora_r, lora_samples, ppl_samples,
            method="spectral"):
    """Run FCM: community detection, distill, fuse, evaluate, LoRA, evaluate."""
    L = cka_matrix.shape[0]

    if method == "spectral":
        communities = detect_communities(cka_matrix, method="spectral", n_clusters=target_layers)
    elif method == "leiden":
        # Binary search on resolution to hit target community count
        lo, hi = 0.01, 50.0
        best_comm = None
        best_diff = float("inf")
        for _ in range(30):  # binary search iterations
            mid = (lo + hi) / 2
            c = detect_communities(cka_matrix, method="leiden", resolution=mid)
            n_comm = len(c)
            diff = abs(n_comm - target_layers)
            if diff < best_diff:
                best_diff = diff
                best_comm = c
            if n_comm == target_layers:
                break
            elif n_comm < target_layers:
                lo = mid  # need more communities -> higher resolution
            else:
                hi = mid
        # If still far off, fall back to spectral
        if best_diff > max(2, target_layers // 3):
            print(f"  Leiden couldn't match target {target_layers} (got {len(best_comm)}), falling back to spectral")
            communities = detect_communities(cka_matrix, method="spectral", n_clusters=target_layers)
        else:
            communities = best_comm
        print(f"  Leiden communities: {len(communities)} (target: {target_layers})")
    else:
        raise ValueError(f"Unknown method: {method}")

    stats = analyze_communities(communities, L)
    print_communities(communities, stats)

    # Distill each community
    rep_layers = []
    for i, comm in enumerate(communities):
        if len(comm) <= 1:
            from src.fusion import get_layer_modules
            rep_layers.append(get_layer_modules(model, comm)[0])
            continue
        print(f"  Community {i}: layers {comm} (size={len(comm)})")
        rep, loss = distill_community(
            model, comm, cka_matrix, cal_enc,
            num_steps=distill_steps, lr=distill_lr, device=device,
        )
        rep_layers.append(rep)
        print(f"    Final loss: {loss:.4f}")

    model = build_fused_model(model, communities, rep_layers)
    n = len(get_all_layers(model))

    ppl_raw = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=device, max_samples=ppl_samples)
    print(f"  PPL (raw): {ppl_raw:.2f}")

    model, _ = lora_recovery_finetune(
        model, tokenizer, num_steps=lora_steps, lr=lora_lr,
        lora_r=lora_r, num_samples=lora_samples, device=device,
    )
    ppl_lora = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=device, max_samples=ppl_samples)
    print(f"  PPL (LoRA): {ppl_lora:.2f}")

    return {
        "layers": n, "ppl_raw": ppl_raw, "ppl_lora": ppl_lora,
        "communities": communities, "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--distill_steps", type=int, default=300, help="Base distill steps (scaled by community size)")
    parser.add_argument("--distill_lr", type=float, default=3e-5)
    parser.add_argument("--num_cal_samples", type=int, default=512)
    parser.add_argument("--lora_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=2048)
    parser.add_argument("--ppl_max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp04")
    parser.add_argument("--cka_matrix_path", type=str, default="results/exp00/cka_matrix_Qwen2.5-7B.npy")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]

    cka_matrix = np.load(args.cka_matrix_path)
    L = cka_matrix.shape[0]

    # Define compression levels
    targets = [20, 14, 10, 7]

    print("=" * 80)
    print("Experiment 04: Compression Ratio Sweep")
    print(f"Model: {args.model} ({L} layers)")
    print(f"Target layers: {targets}")
    print("=" * 80)

    # Baseline
    print("\n--- BASELINE ---")
    model, tokenizer = load_fresh(args.model, args.device)
    baseline_ppl = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                       device=args.device, max_samples=args.ppl_max_samples)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    free(model)

    # Prepare calibration data once
    model_tmp, tokenizer = load_fresh(args.model, args.device)
    cal_enc = prepare_calibration(tokenizer, args.num_cal_samples)
    free(model_tmp)

    all_results = {"baseline": {"ppl": baseline_ppl, "layers": L}}

    for target in targets:
        num_remove = L - target
        ratio = L / target
        print(f"\n{'='*80}")
        print(f"TARGET: {L} -> {target} layers ({ratio:.1f}x compression, {100*num_remove/L:.0f}% reduction)")
        print(f"{'='*80}")

        # ShortGPT
        print(f"\n--- ShortGPT (target={target}) ---")
        model, tokenizer = load_fresh(args.model, args.device)
        sg_res = run_shortgpt(model, tokenizer, num_remove, args.device,
                              args.lora_steps, args.lora_lr, args.lora_r, args.lora_samples, args.ppl_max_samples)
        all_results[f"shortgpt_{target}"] = sg_res
        free(model)

        # FCM-spectral
        print(f"\n--- FCM-spectral (target={target}) ---")
        model, tokenizer = load_fresh(args.model, args.device)
        fcm_res = run_fcm(model, tokenizer, cka_matrix, target, cal_enc, args.device,
                          args.distill_steps, args.distill_lr,
                          args.lora_steps, args.lora_lr, args.lora_r, args.lora_samples, args.ppl_max_samples,
                          method="spectral")
        all_results[f"fcm_spectral_{target}"] = fcm_res
        free(model)

        # FCM-leiden (non-adjacent grouping)
        print(f"\n--- FCM-leiden (target={target}) ---")
        model, tokenizer = load_fresh(args.model, args.device)
        fcm_leiden_res = run_fcm(model, tokenizer, cka_matrix, target, cal_enc, args.device,
                                 args.distill_steps, args.distill_lr,
                                 args.lora_steps, args.lora_lr, args.lora_r, args.lora_samples, args.ppl_max_samples,
                                 method="leiden")
        all_results[f"fcm_leiden_{target}"] = fcm_leiden_res
        free(model)

    # Print summary table
    print(f"\n{'='*80}")
    print("FULL RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Layers':>6} {'Ratio':>6} {'PPL(raw)':>10} {'PPL(LoRA)':>10} {'vs Base':>10}")
    print("-" * 80)
    print(f"{'Baseline':<25} {L:>6} {'1.0x':>6} {baseline_ppl:>10.2f} {'N/A':>10} {'---':>10}")
    for target in targets:
        ratio_str = f"{L/target:.1f}x"
        for method_key in [f"shortgpt_{target}", f"fcm_spectral_{target}", f"fcm_leiden_{target}"]:
            if method_key in all_results:
                r = all_results[method_key]
                name = method_key.replace(f"_{target}", "")
                delta = ((r["ppl_lora"] - baseline_ppl) / baseline_ppl) * 100
                print(f"{name:<25} {r['layers']:>6} {ratio_str:>6} {r['ppl_raw']:>10.2f} {r['ppl_lora']:>10.2f} {delta:>+9.1f}%")
        print("-" * 80)
    print(f"{'='*80}")

    # Save
    results_path = os.path.join(args.output_dir, f"sweep_{model_short}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
