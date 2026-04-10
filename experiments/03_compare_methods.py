"""
Experiment 03: Compare FCM vs ShortGPT at matched compression ratios.

Tests at moderate compression (removing ~30% of layers) with LoRA recovery.
Compares:
  1. ShortGPT (layer deletion by BI score)
  2. FCM-spectral (our method, spectral community detection + distillation)
  3. FCM-leiden (our method, Leiden community detection + distillation)

Each method is followed by LoRA recovery fine-tuning for fair comparison.

Usage:
    python experiments/03_compare_methods.py --model Qwen/Qwen2.5-7B
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
from src.profiler import collect_activations
from src.community import detect_communities, analyze_communities, print_communities
from src.fusion import (
    distill_community, build_fused_model, get_all_layers,
)
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity


def load_model_and_tokenizer(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--target_layers", type=int, default=20,
                        help="Target number of layers after compression (28->20 = ~29%% reduction)")
    parser.add_argument("--distill_steps", type=int, default=500)
    parser.add_argument("--distill_lr", type=float, default=5e-5)
    parser.add_argument("--num_cal_samples", type=int, default=512)
    parser.add_argument("--lora_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=2048)
    parser.add_argument("--ppl_max_samples", type=int, default=50)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/exp03")
    parser.add_argument("--cka_matrix_path", type=str, default="results/exp00/cka_matrix_Qwen2.5-7B.npy")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]

    print("=" * 70)
    print("Experiment 03: Method Comparison")
    print(f"Model: {args.model}")
    print(f"Target layers: {args.target_layers}")
    print("=" * 70)

    # Load CKA matrix
    cka_matrix = np.load(args.cka_matrix_path)
    L = cka_matrix.shape[0]
    num_to_remove = L - args.target_layers
    print(f"Original layers: {L}, Target: {args.target_layers}, Removing: {num_to_remove}")

    results = {}

    # ===== BASELINE: Full model =====
    print("\n" + "=" * 70)
    print("BASELINE: Full Model")
    print("=" * 70)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    baseline_ppl = evaluate_perplexity(
        model, tokenizer, max_seq_len=2048,
        device=args.device, max_samples=args.ppl_max_samples,
    )
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    results["baseline"] = {"perplexity": baseline_ppl, "layers": L}
    free_model(model)

    # ===== METHOD 1: ShortGPT (layer deletion by BI) =====
    print("\n" + "=" * 70)
    print("METHOD 1: ShortGPT (layer deletion)")
    print("=" * 70)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    print("Computing Block Influence scores...")
    bi_scores = compute_block_influence(model, tokenizer, device=args.device)
    print(f"BI scores: min={bi_scores.min():.4f}, max={bi_scores.max():.4f}")

    # Remove layers
    model, kept = remove_layers_by_bi(model, bi_scores, num_to_remove)
    n_after = len(get_all_layers(model))
    print(f"Layers after removal: {n_after} (removed {num_to_remove})")
    print(f"Kept layers: {kept}")

    # Evaluate before LoRA
    ppl_before_lora = evaluate_perplexity(
        model, tokenizer, max_seq_len=2048,
        device=args.device, max_samples=args.ppl_max_samples,
    )
    print(f"ShortGPT PPL (no LoRA): {ppl_before_lora:.2f}")

    # LoRA recovery
    print("Applying LoRA recovery...")
    model, lora_loss = lora_recovery_finetune(
        model, tokenizer,
        num_steps=args.lora_steps, lr=args.lora_lr, lora_r=args.lora_r,
        num_samples=args.lora_samples, device=args.device,
    )
    ppl_after_lora = evaluate_perplexity(
        model, tokenizer, max_seq_len=2048,
        device=args.device, max_samples=args.ppl_max_samples,
    )
    print(f"ShortGPT PPL (with LoRA): {ppl_after_lora:.2f}")

    results["shortgpt"] = {
        "layers": n_after,
        "ppl_no_lora": ppl_before_lora,
        "ppl_with_lora": ppl_after_lora,
        "bi_scores": bi_scores.tolist(),
        "kept_layers": kept,
    }
    free_model(model)

    # ===== METHOD 2: FCM-spectral =====
    print("\n" + "=" * 70)
    print("METHOD 2: FCM-spectral (ours)")
    print("=" * 70)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    communities = detect_communities(cka_matrix, method="spectral", n_clusters=args.target_layers)
    stats = analyze_communities(communities, L)
    print_communities(communities, stats)

    # Prepare calibration data
    from datasets import load_dataset as ld
    ds = ld("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for s in ds:
        if len(s.get("text", "")) > 50:
            texts.append(s["text"])
        if len(texts) >= args.num_cal_samples:
            break
    cal_enc = tokenizer(texts, max_length=args.max_seq_len, truncation=True,
                        padding="max_length", return_tensors="pt")

    # Distill each community
    rep_layers = []
    for i, comm in enumerate(communities):
        print(f"\nCommunity {i}: layers {comm} (size={len(comm)})")
        rep, loss = distill_community(
            model, comm, cka_matrix, cal_enc,
            num_steps=args.distill_steps, lr=args.distill_lr, device=args.device,
        )
        rep_layers.append(rep)
        print(f"  Distill loss: {loss:.4f}")

    model = build_fused_model(model, communities, rep_layers)
    n_after = len(get_all_layers(model))
    print(f"\nLayers after fusion: {n_after}")

    ppl_before_lora = evaluate_perplexity(
        model, tokenizer, max_seq_len=2048,
        device=args.device, max_samples=args.ppl_max_samples,
    )
    print(f"FCM-spectral PPL (no LoRA): {ppl_before_lora:.2f}")

    # LoRA recovery
    print("Applying LoRA recovery...")
    model, lora_loss = lora_recovery_finetune(
        model, tokenizer,
        num_steps=args.lora_steps, lr=args.lora_lr, lora_r=args.lora_r,
        num_samples=args.lora_samples, device=args.device,
    )
    ppl_after_lora = evaluate_perplexity(
        model, tokenizer, max_seq_len=2048,
        device=args.device, max_samples=args.ppl_max_samples,
    )
    print(f"FCM-spectral PPL (with LoRA): {ppl_after_lora:.2f}")

    results["fcm_spectral"] = {
        "layers": n_after,
        "communities": communities,
        "ppl_no_lora": ppl_before_lora,
        "ppl_with_lora": ppl_after_lora,
    }
    free_model(model)

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Layers':>6} {'PPL (no LoRA)':>15} {'PPL (+ LoRA)':>15}")
    print("-" * 70)
    print(f"{'Baseline (full model)':<25} {L:>6} {baseline_ppl:>15.2f} {'N/A':>15}")
    r = results["shortgpt"]
    print(f"{'ShortGPT':<25} {r['layers']:>6} {r['ppl_no_lora']:>15.2f} {r['ppl_with_lora']:>15.2f}")
    r = results["fcm_spectral"]
    print(f"{'FCM-spectral (ours)':<25} {r['layers']:>6} {r['ppl_no_lora']:>15.2f} {r['ppl_with_lora']:>15.2f}")
    print("=" * 70)

    # Save results
    results_path = os.path.join(args.output_dir, f"comparison_{model_short}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
