"""
Experiment 13: 3-seed variance on ALL 11 models

Runs FCM (best metric per model) and ShortGPT with 3 seeds each.
Reports mean +/- std for all models.
"""

import argparse
import gc
import json
import os
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.profiler import collect_activations
from src.cka import compute_similarity_matrix
from src.community import detect_communities
from src.merge import merge_community_select, build_compressed_model
from src.fusion import get_all_layers
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity

SEEDS = [42, 123, 7]

# Best metric per model from Exp 12
MODEL_CONFIG = {
    "Qwen/Qwen3-1.7B":              {"metric": "cka"},
    "google/gemma-2-2b":             {"metric": "cosine"},
    "stabilityai/stablelm-2-1_6b":   {"metric": "cosine"},
    "Qwen/Qwen3-4B":                {"metric": "cka"},
    "HuggingFaceTB/SmolLM3-3B":     {"metric": "cka"},
    "meta-llama/Llama-3.2-3B":      {"metric": "cosine"},
    "Qwen/Qwen2.5-3B":              {"metric": "cosine"},
    "Qwen/Qwen3-8B":                {"metric": "cosine"},
    "Qwen/Qwen2.5-7B":              {"metric": "cosine"},
    "mistralai/Mistral-7B-v0.3":    {"metric": "cosine"},
    "meta-llama/Llama-3.1-8B":      {"metric": "cosine"},
}


def load_fresh(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_one_seed(model_name, sim_matrix, L, target, device, lora_kw, ppl_n, seed):
    """Run both ShortGPT and FCM for one seed."""
    results = {}

    # ShortGPT
    set_seed(seed)
    m, t = load_fresh(model_name, device)
    bi = compute_block_influence(m, t, device=device)
    m, _ = remove_layers_by_bi(m, bi, L - target)
    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    sg_ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    free(m)
    results["sg"] = sg_ppl

    # FCM
    set_seed(seed)
    comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)
    m, t = load_fresh(model_name, device)
    merged = [merge_community_select(m, comm, sim_matrix) for comm in comms]
    m = build_compressed_model(m, comms, merged)
    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    fcm_ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    free(m)
    results["fcm"] = fcm_ppl

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIG.keys()))
    parser.add_argument("--lora_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=2048)
    parser.add_argument("--ppl_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp13")
    parser.add_argument("--sim_dir", type=str, default="results/exp12",
                        help="Dir with cached similarity matrices from Exp 12")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_kw = dict(num_steps=args.lora_steps, lr=args.lora_lr,
                   lora_r=args.lora_r, num_samples=args.lora_samples)

    all_results = {}
    results_path = os.path.join(args.output_dir, "full_variance.json")

    for model_name in args.models:
        short = model_name.split("/")[-1]
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        L = cfg.num_hidden_layers
        target = max(L // 2, 4)
        metric = MODEL_CONFIG.get(model_name, {}).get("metric", "cosine")

        print(f"\n{'#'*70}")
        print(f"# {short} ({L}->{target}, metric={metric}, 3 seeds)")
        print(f"{'#'*70}")

        # Load similarity matrix
        sim_path = os.path.join(args.sim_dir, f"{metric}_{short}.npy")
        if not os.path.exists(sim_path):
            print(f"  Computing {metric} similarity matrix...")
            acts = collect_activations(model_name, num_samples=512, max_seq_len=128,
                                       batch_size=4, device=args.device)
            sim = compute_similarity_matrix(acts, metric=metric)
            np.save(sim_path, sim)
            del acts; torch.cuda.empty_cache()
        else:
            sim = np.load(sim_path)

        # Baseline
        m, t = load_fresh(model_name, args.device)
        bl = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=args.ppl_n)
        free(m)
        print(f"  Baseline: {bl:.2f}")

        # Run 3 seeds
        sg_ppls = []
        fcm_ppls = []
        for seed in SEEDS:
            print(f"  Seed {seed}:", end=" ", flush=True)
            res = run_one_seed(model_name, sim, L, target, args.device, lora_kw, args.ppl_n, seed)
            sg_ppls.append(res["sg"])
            fcm_ppls.append(res["fcm"])
            print(f"SG={res['sg']:.2f}, FCM={res['fcm']:.2f}")

        sg_mean, sg_std = np.mean(sg_ppls), np.std(sg_ppls)
        fcm_mean, fcm_std = np.mean(fcm_ppls), np.std(fcm_ppls)
        delta = (fcm_mean - sg_mean) / sg_mean * 100
        winner = "FCM" if fcm_mean < sg_mean else "ShortGPT"

        all_results[model_name] = {
            "L": L, "target": target, "metric": metric, "baseline": bl,
            "sg_mean": sg_mean, "sg_std": sg_std, "sg_runs": sg_ppls,
            "fcm_mean": fcm_mean, "fcm_std": fcm_std, "fcm_runs": fcm_ppls,
            "delta_pct": delta, "winner": winner,
        }
        print(f"  => SG: {sg_mean:.2f}+/-{sg_std:.2f}  FCM: {fcm_mean:.2f}+/-{fcm_std:.2f}  {winner} ({delta:+.1f}%)")

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*100}")
    print("3-SEED VARIANCE RESULTS (all 11 models)")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Base':>6} {'ShortGPT':>18} {'FCM':>18} {'Delta':>8} {'Win':>5}")
    print("-" * 90)
    wins = 0
    for name, r in all_results.items():
        short = name.split("/")[-1]
        w = r["winner"]
        if w == "FCM": wins += 1
        print(f"{short:<30} {r['baseline']:>6.2f} {r['sg_mean']:>7.2f}+/-{r['sg_std']:<7.2f} "
              f"{r['fcm_mean']:>7.2f}+/-{r['fcm_std']:<7.2f} {r['delta_pct']:>+7.1f}% {w:>5}")
    print("-" * 90)
    print(f"FCM wins: {wins}/{len(all_results)}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
