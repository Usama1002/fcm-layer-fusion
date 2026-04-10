"""
Experiment 15: Hybrid FCM-BI sweep across all 11 models (3 seeds)

The hybrid approach: use spectral community detection for grouping,
but select the highest-BI layer per community (instead of most central).
This combines FCM's coverage guarantee with ShortGPT's importance scoring.
"""

import gc
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cka import compute_similarity_matrix
from src.community import detect_communities
from src.fusion import get_all_layers, find_central_layer
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity

SEEDS = [42, 123, 7]
MODELS = [
    "Qwen/Qwen3-1.7B", "google/gemma-2-2b", "stabilityai/stablelm-2-1_6b",
    "Qwen/Qwen3-4B", "HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.1-8B",
]
# Best metric per model from Exp 12/13
METRICS = {
    "Qwen/Qwen3-1.7B": "cka", "google/gemma-2-2b": "cosine",
    "stabilityai/stablelm-2-1_6b": "cosine", "Qwen/Qwen3-4B": "cka",
    "HuggingFaceTB/SmolLM3-3B": "cka", "meta-llama/Llama-3.2-3B": "cosine",
    "Qwen/Qwen2.5-3B": "cosine", "Qwen/Qwen3-8B": "cosine",
    "Qwen/Qwen2.5-7B": "cosine", "mistralai/Mistral-7B-v0.3": "cosine",
    "meta-llama/Llama-3.1-8B": "cosine",
}
DEVICE = "cuda"

def load_fresh(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                              device_map=DEVICE, trust_remote_code=True)
    return m, tok

def free(m): del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_hybrid(model_name, sim_matrix, bi_scores, L, target, seed, lora_kw):
    """FCM-BI hybrid: community grouping + BI-guided selection."""
    set_seed(seed)
    comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)

    # Select highest-BI layer per community
    keep = []
    for comm in comms:
        if len(comm) == 1:
            keep.append(comm[0])
        else:
            best = max(comm, key=lambda i: bi_scores[i])
            keep.append(best)
    keep = sorted(keep)

    m, t = load_fresh(model_name)
    layers = get_all_layers(m)
    new_layers = nn.ModuleList([layers[i] for i in keep])
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        m.model.layers = new_layers
        m.config.num_hidden_layers = len(new_layers)
    m.config.use_cache = False
    if hasattr(m, "generation_config"):
        m.generation_config.cache_implementation = None

    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=DEVICE, **lora_kw)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    free(m)
    return ppl


def run_shortgpt(model_name, bi_scores, L, target, seed, lora_kw):
    set_seed(seed)
    m, t = load_fresh(model_name)
    m, _ = remove_layers_by_bi(m, bi_scores, L - target)
    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=DEVICE, **lora_kw)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    free(m)
    return ppl


def run_fcm_central(model_name, sim_matrix, L, target, seed, lora_kw):
    """Original FCM with centrality-based selection."""
    set_seed(seed)
    comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)
    keep = sorted([find_central_layer(sim_matrix, c) for c in comms])

    m, t = load_fresh(model_name)
    layers = get_all_layers(m)
    new_layers = nn.ModuleList([layers[i] for i in keep])
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        m.model.layers = new_layers
        m.config.num_hidden_layers = len(new_layers)
    m.config.use_cache = False
    if hasattr(m, "generation_config"):
        m.generation_config.cache_implementation = None

    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=DEVICE, **lora_kw)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    free(m)
    return ppl


def main():
    output_dir = "results/exp15"
    os.makedirs(output_dir, exist_ok=True)
    lora_kw = dict(num_steps=500, lr=2e-4, lora_r=16, num_samples=2048)

    all_results = {}
    results_path = os.path.join(output_dir, "hybrid_sweep.json")

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        L = cfg.num_hidden_layers
        target = max(L // 2, 4)
        metric = METRICS.get(model_name, "cosine")

        print(f"\n{'#'*70}")
        print(f"# {short} ({L}->{target}, metric={metric})")
        print(f"{'#'*70}")

        # Load similarity matrix
        sim_path = f"results/exp12/{metric}_{short}.npy"
        if not os.path.exists(sim_path):
            print(f"  Missing {sim_path}, skipping")
            continue
        sim = np.load(sim_path)

        # Compute BI scores once
        m, t = load_fresh(model_name)
        bi = compute_block_influence(m, t, device=DEVICE)
        bl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
        free(m)
        print(f"  Baseline: {bl:.2f}")

        sg_ppls, hybrid_ppls, central_ppls = [], [], []
        for seed in SEEDS:
            print(f"  Seed {seed}:", end=" ", flush=True)
            sg = run_shortgpt(model_name, bi, L, target, seed, lora_kw)
            sg_ppls.append(sg)
            hyb = run_hybrid(model_name, sim, bi, L, target, seed, lora_kw)
            hybrid_ppls.append(hyb)
            cen = run_fcm_central(model_name, sim, L, target, seed, lora_kw)
            central_ppls.append(cen)
            print(f"SG={sg:.1f} Hybrid={hyb:.1f} Central={cen:.1f}")

        sg_m, sg_s = np.mean(sg_ppls), np.std(sg_ppls)
        hyb_m, hyb_s = np.mean(hybrid_ppls), np.std(hybrid_ppls)
        cen_m, cen_s = np.mean(central_ppls), np.std(central_ppls)
        d_hyb = (hyb_m - sg_m) / sg_m * 100
        d_cen = (cen_m - sg_m) / sg_m * 100

        all_results[model_name] = {
            "L": L, "target": target, "baseline": bl,
            "shortgpt": {"mean": sg_m, "std": sg_s},
            "hybrid": {"mean": hyb_m, "std": hyb_s, "delta": d_hyb},
            "central": {"mean": cen_m, "std": cen_s, "delta": d_cen},
        }
        print(f"  SG: {sg_m:.2f}+/-{sg_s:.2f}  Hybrid: {hyb_m:.2f}+/-{hyb_s:.2f} ({d_hyb:+.1f}%)  Central: {cen_m:.2f}+/-{cen_s:.2f} ({d_cen:+.1f}%)")

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*100}")
    print("HYBRID vs SHORTGPT vs CENTRAL (3 seeds)")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'SG':>15} {'Hybrid':>15} {'Central':>15} {'Best':>8}")
    print("-" * 85)
    hyb_wins = 0
    for name, r in all_results.items():
        short = name.split("/")[-1]
        sg = r["shortgpt"]["mean"]
        hyb = r["hybrid"]["mean"]
        cen = r["central"]["mean"]
        best = min(sg, hyb, cen)
        w = "Hybrid" if best == hyb else ("SG" if best == sg else "Central")
        if w == "Hybrid": hyb_wins += 1
        print(f"{short:<25} {sg:>7.1f}+/-{r['shortgpt']['std']:>4.1f} {hyb:>7.1f}+/-{r['hybrid']['std']:>4.1f} {cen:>7.1f}+/-{r['central']['std']:>4.1f} {w:>8}")
    print(f"\nHybrid wins: {hyb_wins}/{len(all_results)}")


if __name__ == "__main__":
    main()
