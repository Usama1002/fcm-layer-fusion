"""
Experiment 18: 5-Seed Variance on Key Models

Upgrades from 3-seed to 5-seed on the most important models
for stronger statistical claims. Runs on A100 GPU.

Run: python experiments/18_five_seeds.py --device cuda:1
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cka import compute_similarity_matrix
from src.community import detect_communities
from src.fusion import get_all_layers
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity

SEEDS = [42, 123, 7, 2024, 999]

# Key models: biggest win, biggest loss, and the two most important 7B models
KEY_MODELS = {
    "Qwen/Qwen2.5-7B":          {"metric": "cosine"},  # Our flagship
    "Qwen/Qwen3-4B":            {"metric": "cka"},      # Biggest win (-50%)
    "meta-llama/Llama-3.1-8B":   {"metric": "cosine"},  # Previously catastrophic, now wins
    "mistralai/Mistral-7B-v0.3": {"metric": "cosine"},  # Only loss (+13.5%)
}
DEVICE = "cuda:1"


def load_fresh(name, device):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                              device_map=device, trust_remote_code=True)
    return m, tok

def free(m): del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_one_seed(model_name, sim_matrix, bi, L, target, seed, device, lora_kw):
    results = {}

    # ShortGPT
    set_seed(seed)
    m, t = load_fresh(model_name, device)
    m, _ = remove_layers_by_bi(m, bi, L - target)
    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    results["sg"] = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=50)
    free(m)

    # FCM-Hybrid
    set_seed(seed)
    comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)
    keep = sorted([max(c, key=lambda i: bi[i]) if len(c) > 1 else c[0] for c in comms])

    m, t = load_fresh(model_name, device)
    layers = get_all_layers(m)
    new_layers = nn.ModuleList([layers[i] for i in keep])
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        m.model.layers = new_layers
        m.config.num_hidden_layers = len(new_layers)
    m.config.use_cache = False
    if hasattr(m, "generation_config"):
        m.generation_config.cache_implementation = None

    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    results["fcm"] = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=50)
    free(m)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--output_dir", type=str, default="results/exp18")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_kw = dict(num_steps=500, lr=2e-4, lora_r=16, num_samples=2048)

    all_results = {}
    results_path = os.path.join(args.output_dir, "five_seeds.json")

    for model_name, cfg in KEY_MODELS.items():
        short = model_name.split("/")[-1]
        model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        L = model_cfg.num_hidden_layers
        target = max(L // 2, 4)
        metric = cfg["metric"]

        print(f"\n{'#'*70}")
        print(f"# {short} ({L}->{target}, metric={metric}, 5 seeds)")
        print(f"{'#'*70}")

        # Load similarity matrix (from exp12 or compute fresh)
        sim_path = f"results/exp12/{metric}_{short}.npy"
        if os.path.exists(sim_path):
            sim = np.load(sim_path)
        else:
            print(f"  Computing {metric} similarity matrix...")
            from src.profiler import collect_activations
            acts = collect_activations(model_name, num_samples=512, max_seq_len=128,
                                       batch_size=4, device=args.device)
            sim = compute_similarity_matrix(acts, metric=metric)
            os.makedirs("results/exp12", exist_ok=True)
            np.save(sim_path, sim)
            del acts; torch.cuda.empty_cache()

        # Compute BI once
        m, t = load_fresh(model_name, args.device)
        bi = compute_block_influence(m, t, device=args.device)
        bl = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=50)
        free(m)
        print(f"  Baseline: {bl:.2f}")

        sg_ppls, fcm_ppls = [], []
        for seed in SEEDS:
            print(f"  Seed {seed}:", end=" ", flush=True)
            res = run_one_seed(model_name, sim, bi, L, target, seed, args.device, lora_kw)
            sg_ppls.append(res["sg"])
            fcm_ppls.append(res["fcm"])
            print(f"SG={res['sg']:.2f}, FCM={res['fcm']:.2f}")

        sg_m, sg_s = np.mean(sg_ppls), np.std(sg_ppls)
        fcm_m, fcm_s = np.mean(fcm_ppls), np.std(fcm_ppls)
        delta = (fcm_m - sg_m) / sg_m * 100

        all_results[model_name] = {
            "L": L, "target": target, "metric": metric, "baseline": bl,
            "shortgpt": {"mean": sg_m, "std": sg_s, "runs": sg_ppls},
            "fcm_hybrid": {"mean": fcm_m, "std": fcm_s, "runs": fcm_ppls},
            "delta_pct": delta,
            "winner": "FCM" if fcm_m < sg_m else "ShortGPT",
        }
        print(f"  => SG: {sg_m:.2f}+/-{sg_s:.2f}  FCM: {fcm_m:.2f}+/-{fcm_s:.2f}  ({delta:+.1f}%)")

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*80}")
    print("5-SEED VARIANCE RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'SG (5 seeds)':>20} {'FCM (5 seeds)':>20} {'Delta':>8}")
    print("-" * 78)
    for name, r in all_results.items():
        short = name.split("/")[-1]
        sg = r["shortgpt"]
        fcm = r["fcm_hybrid"]
        print(f"{short:<25} {sg['mean']:>7.2f}+/-{sg['std']:>5.2f}   {fcm['mean']:>7.2f}+/-{fcm['std']:>5.2f}   {r['delta_pct']:>+6.1f}%")


if __name__ == "__main__":
    main()
