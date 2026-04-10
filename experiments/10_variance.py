"""
Experiment 10: Variance Reporting (3 seeds)

Re-runs FCM and ShortGPT on Qwen2.5-7B and Qwen2.5-3B with 3 different
random seeds to report mean +/- std. Also measures wall-clock time.

This addresses the #1 fatal flaw from the self-review.
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


def load_fresh(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                              device_map=device, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_shortgpt_seeded(model_name, L, target, device, lora_kw, ppl_n, seed):
    set_seed(seed)
    m, t = load_fresh(model_name, device)
    t0 = time.time()
    bi = compute_block_influence(m, t, device=device)
    m, kept = remove_layers_by_bi(m, bi, L - target)
    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    set_seed(seed)  # reset seed before LoRA
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    ppl_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    elapsed = time.time() - t0
    free(m)
    return {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "time": elapsed, "seed": seed}


def run_fcm_seeded(model_name, L, target, cos_matrix, device, lora_kw, ppl_n, seed):
    set_seed(seed)
    comms = detect_communities(cos_matrix, method="spectral", n_clusters=target)
    m, t = load_fresh(model_name, device)
    t0 = time.time()
    merged = [merge_community_select(m, comm, cos_matrix) for comm in comms]
    m = build_compressed_model(m, comms, merged)
    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    ppl_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    elapsed = time.time() - t0
    free(m)
    return {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "time": elapsed, "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-3B"])
    parser.add_argument("--ppl_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp10")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    results_path = os.path.join(args.output_dir, "variance.json")

    # Two LoRA budgets
    lora_configs = {
        "low": dict(num_steps=500, lr=2e-4, lora_r=16, num_samples=2048),
        "high": dict(num_steps=1000, lr=2e-4, lora_r=32, num_samples=4096),
    }

    for model_name in args.models:
        short = model_name.split("/")[-1]
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        L = cfg.num_hidden_layers
        target = max(L // 2, 4)

        print(f"\n{'#'*70}")
        print(f"# {model_name} ({L} -> {target}) x 3 seeds x 2 budgets")
        print(f"{'#'*70}")

        # Load cosine similarity matrix
        cos_path = None
        for p in [f"results/exp09/cosine_{short}.npy", f"results/exp09_lora32/cosine_{short}.npy",
                   f"results/exp08/cosine_{short}.npy"]:
            if os.path.exists(p):
                cos_path = p; break
        if cos_path is None:
            print(f"  Computing cosine similarity matrix...")
            acts = collect_activations(model_name, num_samples=512, max_seq_len=128, batch_size=4, device=args.device)
            cos_matrix = compute_similarity_matrix(acts, metric="cosine")
            cos_path = os.path.join(args.output_dir, f"cosine_{short}.npy")
            np.save(cos_path, cos_matrix)
            del acts; torch.cuda.empty_cache()
        else:
            cos_matrix = np.load(cos_path)
            print(f"  Loaded cosine matrix from {cos_path}")

        # Baseline (deterministic, run once)
        m, t = load_fresh(model_name, args.device)
        bl_ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=args.ppl_n)
        print(f"  Baseline PPL: {bl_ppl:.2f}")
        free(m)

        model_res = {"model": model_name, "L": L, "target": target, "baseline_ppl": bl_ppl}

        for budget_name, lora_kw in lora_configs.items():
            print(f"\n  === {budget_name} LoRA budget ===")

            sg_results = []
            fcm_results = []

            for seed in SEEDS:
                print(f"\n  Seed {seed}:")

                print(f"    ShortGPT...")
                sg = run_shortgpt_seeded(model_name, L, target, args.device, lora_kw, args.ppl_n, seed)
                sg_results.append(sg)
                print(f"      PPL: raw={sg['ppl_raw']:.2f}, +LoRA={sg['ppl_lora']:.2f} ({sg['time']:.0f}s)")

                print(f"    FCM...")
                fcm = run_fcm_seeded(model_name, L, target, cos_matrix, args.device, lora_kw, args.ppl_n, seed)
                fcm_results.append(fcm)
                print(f"      PPL: raw={fcm['ppl_raw']:.2f}, +LoRA={fcm['ppl_lora']:.2f} ({fcm['time']:.0f}s)")

            # Compute statistics
            sg_ppls = [r["ppl_lora"] for r in sg_results]
            fcm_ppls = [r["ppl_lora"] for r in fcm_results]
            sg_times = [r["time"] for r in sg_results]
            fcm_times = [r["time"] for r in fcm_results]

            model_res[budget_name] = {
                "shortgpt": {
                    "mean": np.mean(sg_ppls), "std": np.std(sg_ppls),
                    "runs": sg_results,
                    "time_mean": np.mean(sg_times), "time_std": np.std(sg_times),
                },
                "fcm": {
                    "mean": np.mean(fcm_ppls), "std": np.std(fcm_ppls),
                    "runs": fcm_results,
                    "time_mean": np.mean(fcm_times), "time_std": np.std(fcm_times),
                },
            }

            print(f"\n  {budget_name} Summary:")
            print(f"    ShortGPT: {np.mean(sg_ppls):.2f} +/- {np.std(sg_ppls):.2f} ({np.mean(sg_times):.0f}s)")
            print(f"    FCM:      {np.mean(fcm_ppls):.2f} +/- {np.std(fcm_ppls):.2f} ({np.mean(fcm_times):.0f}s)")
            delta = (np.mean(fcm_ppls) - np.mean(sg_ppls)) / np.mean(sg_ppls) * 100
            print(f"    Delta:    {delta:+.1f}%")

        all_results[model_name] = model_res
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*90}")
    print("VARIANCE RESULTS (mean +/- std across 3 seeds)")
    print(f"{'='*90}")
    print(f"{'Model':<22} {'Budget':<8} {'ShortGPT':>20} {'FCM':>20} {'Delta%':>8}")
    print("-" * 80)
    for name, res in all_results.items():
        short = name.split("/")[-1]
        for budget in ["low", "high"]:
            if budget in res:
                sg = res[budget]["shortgpt"]
                fcm = res[budget]["fcm"]
                delta = (fcm["mean"] - sg["mean"]) / sg["mean"] * 100
                print(f"{short:<22} {budget:<8} {sg['mean']:>8.2f} +/- {sg['std']:<8.2f} {fcm['mean']:>8.2f} +/- {fcm['std']:<8.2f} {delta:>+7.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
