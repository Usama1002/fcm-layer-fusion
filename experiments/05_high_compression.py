"""
Experiment 05: High compression with stability fixes.

Re-runs the 2.8x and 4x compression levels that had NaN issues,
plus adds Fisher segmentation (contiguous-only) as ablation baseline.

Usage:
    python experiments/05_high_compression.py --model Qwen/Qwen2.5-7B
"""

import argparse
import gc
import json
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.community import detect_communities, analyze_communities, print_communities
from src.fusion import distill_community, build_fused_model, get_all_layers, get_layer_modules
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.ablations import fisher_segmentation, uniform_grouping


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


def run_fcm_with_grouping(model, tokenizer, cka_matrix, communities, cal_enc, device,
                          distill_steps, distill_lr, lora_steps, lora_lr, lora_r, lora_samples, ppl_samples):
    """Run FCM with pre-computed communities."""
    L = cka_matrix.shape[0]
    stats = analyze_communities(communities, L)
    print_communities(communities, stats)

    rep_layers = []
    for i, comm in enumerate(communities):
        if len(comm) <= 1:
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

    return {"layers": n, "ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "communities": communities, "stats": stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--distill_steps", type=int, default=500)
    parser.add_argument("--distill_lr", type=float, default=3e-5)
    parser.add_argument("--num_cal_samples", type=int, default=512)
    parser.add_argument("--lora_steps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=4096)
    parser.add_argument("--ppl_max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp05")
    parser.add_argument("--cka_matrix_path", type=str, default="results/exp00/cka_matrix_Qwen2.5-7B.npy")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]

    cka_matrix = np.load(args.cka_matrix_path)
    L = cka_matrix.shape[0]

    # Focus on 2x compression — our sweet spot — with proper ablations
    target = 14
    print("=" * 80)
    print(f"Experiment 05: Ablation Study at 2x Compression ({L}->{target})")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Baseline
    model, tokenizer = load_fresh(args.model, args.device)
    baseline_ppl = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                       device=args.device, max_samples=args.ppl_max_samples)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    free(model)

    cal_enc = prepare_calibration(
        AutoTokenizer.from_pretrained(args.model, trust_remote_code=True),
        args.num_cal_samples
    )

    all_results = {"baseline": {"ppl": baseline_ppl}}

    methods = {
        "shortgpt": None,  # handled separately
        "fcm_spectral": lambda: detect_communities(cka_matrix, method="spectral", n_clusters=target),
        "fcm_fisher": lambda: fisher_segmentation(cka_matrix, n_segments=target),
        "fcm_uniform": lambda: uniform_grouping(L, target),
    }

    # ShortGPT baseline
    print(f"\n--- ShortGPT ---")
    model, tokenizer = load_fresh(args.model, args.device)
    bi = compute_block_influence(model, tokenizer, device=args.device)
    model, kept = remove_layers_by_bi(model, bi, L - target)
    print(f"  Kept: {kept}")
    ppl_raw = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=args.device, max_samples=args.ppl_max_samples)
    print(f"  PPL (raw): {ppl_raw:.2f}")
    model, _ = lora_recovery_finetune(model, tokenizer, num_steps=args.lora_steps, lr=args.lora_lr,
                                       lora_r=args.lora_r, num_samples=args.lora_samples, device=args.device)
    ppl_lora = evaluate_perplexity(model, tokenizer, max_seq_len=2048, device=args.device, max_samples=args.ppl_max_samples)
    print(f"  PPL (LoRA): {ppl_lora:.2f}")
    all_results["shortgpt"] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora}
    free(model)

    # FCM variants
    for name, get_communities in methods.items():
        if get_communities is None:
            continue
        print(f"\n--- {name} ---")
        communities = get_communities()
        model, tokenizer = load_fresh(args.model, args.device)
        res = run_fcm_with_grouping(
            model, tokenizer, cka_matrix, communities, cal_enc, args.device,
            args.distill_steps, args.distill_lr,
            args.lora_steps, args.lora_lr, args.lora_r, args.lora_samples, args.ppl_max_samples,
        )
        all_results[name] = res
        free(model)

    # Summary
    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS @ 2x COMPRESSION ({L}->{target} layers)")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'PPL(raw)':>12} {'PPL(+LoRA)':>12} {'vs Base':>10}")
    print("-" * 65)
    print(f"{'Baseline':<25} {baseline_ppl:>12.2f} {'---':>12} {'---':>10}")
    for name in ["shortgpt", "fcm_spectral", "fcm_fisher", "fcm_uniform"]:
        if name in all_results and "ppl_lora" in all_results[name]:
            r = all_results[name]
            delta = ((r["ppl_lora"] - baseline_ppl) / baseline_ppl) * 100
            print(f"{name:<25} {r['ppl_raw']:>12.2f} {r['ppl_lora']:>12.2f} {delta:>+9.1f}%")
    print(f"{'='*80}")

    results_path = os.path.join(args.output_dir, f"ablation_{model_short}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
