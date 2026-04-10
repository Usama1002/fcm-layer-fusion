"""
Experiment 16: Zero-shot benchmarks on all 11 models

Runs FCM-Hybrid and ShortGPT compressed models through lm-evaluation-harness
on ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA.

For each model: baseline (uncompressed) + ShortGPT + FCM-Hybrid at 2x compression.
Single seed (42) to keep runtime manageable (~30min per model per method).
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

from src.profiler import collect_activations
from src.cka import compute_similarity_matrix
from src.community import detect_communities
from src.fusion import get_all_layers, find_central_layer
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.benchmark import evaluate_zero_shot

MODELS = [
    "Qwen/Qwen3-1.7B", "google/gemma-2-2b", "stabilityai/stablelm-2-1_6b",
    "Qwen/Qwen3-4B", "HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.1-8B",
]
METRICS = {
    "Qwen/Qwen3-1.7B": "cka", "google/gemma-2-2b": "cosine",
    "stabilityai/stablelm-2-1_6b": "cosine", "Qwen/Qwen3-4B": "cka",
    "HuggingFaceTB/SmolLM3-3B": "cka", "meta-llama/Llama-3.2-3B": "cosine",
    "Qwen/Qwen2.5-3B": "cosine", "Qwen/Qwen3-8B": "cosine",
    "Qwen/Qwen2.5-7B": "cosine", "mistralai/Mistral-7B-v0.3": "cosine",
    "meta-llama/Llama-3.1-8B": "cosine",
}
ZS_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]
DEVICE = "cuda"
SEED = 42

def load_fresh(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                              device_map=DEVICE, trust_remote_code=True)
    return m, tok

def free(m): del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s=42):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def eval_zs(model, tokenizer):
    """Run zero-shot evaluation, return dict of task->accuracy."""
    try:
        scores = evaluate_zero_shot(model, tokenizer, tasks=ZS_TASKS, device=DEVICE)
        return scores
    except Exception as e:
        print(f"    ZS eval failed: {e}")
        return {}


def run_baseline(model_name):
    """Evaluate uncompressed model."""
    m, t = load_fresh(model_name)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    zs = eval_zs(m, t)
    free(m)
    return {"ppl": ppl, "zs": zs}


def run_shortgpt(model_name, L, target):
    """Compress with ShortGPT + LoRA, then evaluate."""
    set_seed(SEED)
    m, t = load_fresh(model_name)
    bi = compute_block_influence(m, t, device=DEVICE)
    m, kept = remove_layers_by_bi(m, bi, L - target)
    set_seed(SEED)
    m, _ = lora_recovery_finetune(m, t, num_steps=500, lr=2e-4,
                                   lora_r=16, num_samples=2048, device=DEVICE)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    zs = eval_zs(m, t)
    free(m)
    return {"ppl": ppl, "zs": zs, "kept": kept}


def run_fcm_hybrid(model_name, sim_matrix, L, target):
    """Compress with FCM-Hybrid (community + BI) + LoRA, then evaluate."""
    set_seed(SEED)
    # Community detection
    comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)

    # BI scores
    m, t = load_fresh(model_name)
    bi = compute_block_influence(m, t, device=DEVICE)
    free(m)

    # Select highest-BI layer per community
    keep = []
    for comm in comms:
        if len(comm) == 1:
            keep.append(comm[0])
        else:
            best = max(comm, key=lambda i: bi[i])
            keep.append(best)
    keep = sorted(keep)

    # Build compressed model
    m, t = load_fresh(model_name)
    layers = get_all_layers(m)
    new_layers = nn.ModuleList([layers[i] for i in keep])
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        m.model.layers = new_layers
        m.config.num_hidden_layers = len(new_layers)
    m.config.use_cache = False
    if hasattr(m, "generation_config"):
        m.generation_config.cache_implementation = None

    # LoRA recovery
    set_seed(SEED)
    m, _ = lora_recovery_finetune(m, t, num_steps=500, lr=2e-4,
                                   lora_r=16, num_samples=2048, device=DEVICE)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    zs = eval_zs(m, t)
    free(m)
    return {"ppl": ppl, "zs": zs, "kept": keep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output_dir", type=str, default="results/exp16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    results_path = os.path.join(args.output_dir, "zero_shot.json")

    # Load existing results to allow resuming
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")

    for model_name in args.models:
        if model_name in all_results:
            print(f"Skipping {model_name} (already done)")
            continue

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

        t0 = time.time()
        result = {"model": model_name, "L": L, "target": target}

        # Baseline
        print(f"  [Baseline]")
        bl = run_baseline(model_name)
        result["baseline"] = bl
        print(f"    PPL: {bl['ppl']:.2f}, ZS avg: {bl['zs'].get('avg', 0):.4f}")

        # ShortGPT
        print(f"  [ShortGPT]")
        sg = run_shortgpt(model_name, L, target)
        result["shortgpt"] = sg
        print(f"    PPL: {sg['ppl']:.2f}, ZS avg: {sg['zs'].get('avg', 0):.4f}")

        # FCM-Hybrid
        print(f"  [FCM-Hybrid]")
        fcm = run_fcm_hybrid(model_name, sim, L, target)
        result["fcm"] = fcm
        print(f"    PPL: {fcm['ppl']:.2f}, ZS avg: {fcm['zs'].get('avg', 0):.4f}")

        elapsed = time.time() - t0
        result["time_min"] = elapsed / 60
        all_results[model_name] = result

        # Save after each model
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Done in {elapsed/60:.1f} min, saved.")

    # Final summary
    print(f"\n{'='*110}")
    print("ZERO-SHOT BENCHMARK RESULTS (seed=42, 2x compression)")
    print(f"{'='*110}")
    print(f"{'Model':<25} {'Method':<10} {'PPL':>7} {'ARC-e':>7} {'ARC-c':>7} {'Hella':>7} {'Wino':>7} {'PIQA':>7} {'Avg':>7}")
    print("-" * 95)

    fcm_zs_wins = 0
    for name, res in all_results.items():
        short = name.split("/")[-1][:22]
        for method, key in [("Base", "baseline"), ("SG", "shortgpt"), ("FCM", "fcm")]:
            r = res[key]
            zs = r.get("zs", {})
            print(f"{short if method=='Base' else '':<25} {method:<10} {r['ppl']:>7.2f} "
                  f"{zs.get('arc_easy',0):>7.4f} {zs.get('arc_challenge',0):>7.4f} "
                  f"{zs.get('hellaswag',0):>7.4f} {zs.get('winogrande',0):>7.4f} "
                  f"{zs.get('piqa',0):>7.4f} {zs.get('avg',0):>7.4f}")
        # Check if FCM ZS avg > SG ZS avg
        sg_avg = res["shortgpt"].get("zs", {}).get("avg", 0)
        fcm_avg = res["fcm"].get("zs", {}).get("avg", 0)
        if fcm_avg > sg_avg:
            fcm_zs_wins += 1
        print("-" * 95)

    print(f"\nFCM zero-shot avg beats ShortGPT: {fcm_zs_wins}/{len(all_results)}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
