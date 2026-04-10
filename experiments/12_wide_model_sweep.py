"""
Experiment 12: Wide Model Sweep — 11 models, 6 families, 3 scales

For each model:
  1. CKA analysis (non-adjacent ratio)
  2. FCM (cosine + spectral + select) at 2x compression
  3. ShortGPT baseline at 2x compression
  4. Both with LoRA recovery (r=16, 500 steps)
  5. Perplexity evaluation

Models are grouped by tier for efficient GPU memory management.
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
from src.community import detect_communities, analyze_communities
from src.merge import merge_community_select, build_compressed_model
from src.fusion import get_all_layers
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.visualize import plot_cka_matrix, plot_cka_off_diagonal_analysis

ALL_MODELS = [
    # Tier 1: 7-8B
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.1-8B",
    # Tier 2: 3-4B
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B",
    "HuggingFaceTB/SmolLM3-3B",
    # Tier 3: 1-2B
    "Qwen/Qwen3-1.7B",
    "google/gemma-2-2b",
    "stabilityai/stablelm-2-1_6b",
]


def load_fresh(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s=42):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def process_model(model_name, output_dir, device, lora_kw, ppl_n):
    short = model_name.split("/")[-1]
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = max(L // 2, 4)

    print(f"\n{'#'*70}")
    print(f"# {model_name} ({L} layers -> {target}, {L/target:.1f}x)")
    print(f"{'#'*70}")

    result = {"model": model_name, "L": L, "target": target}

    # 1. CKA / Cosine profiling
    cos_path = os.path.join(output_dir, f"cosine_{short}.npy")
    cka_path = os.path.join(output_dir, f"cka_{short}.npy")

    if os.path.exists(cos_path):
        cos = np.load(cos_path)
        cka = np.load(cka_path) if os.path.exists(cka_path) else cos
        print(f"  Loaded cached similarity matrices")
    else:
        print(f"  Computing similarity matrices...")
        acts = collect_activations(model_name, num_samples=512, max_seq_len=128,
                                   batch_size=4, device=device)
        cos = compute_similarity_matrix(acts, metric="cosine")
        cka = compute_similarity_matrix(acts, metric="cka")
        np.save(cos_path, cos)
        np.save(cka_path, cka)
        del acts; torch.cuda.empty_cache()

    # CKA analysis
    plot_cka_matrix(cka, title=f"CKA - {short}",
                    save_path=os.path.join(output_dir, f"cka_{short}.png"))
    offdiag = plot_cka_off_diagonal_analysis(
        cka, save_path=os.path.join(output_dir, f"offdiag_{short}.png"))
    adj = int(offdiag["num_adjacent_high"])
    nonadj = int(offdiag["num_nonadjacent_high"])
    ratio = nonadj / max(1, adj)
    result["cka"] = {"adj": adj, "nonadj": nonadj, "ratio": ratio}
    print(f"  CKA: adj={adj}, nonadj={nonadj}, ratio={ratio:.2f}x")

    # 2. Baseline perplexity
    print(f"  [Baseline]")
    m, t = load_fresh(model_name, device)
    bl_ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    result["baseline_ppl"] = bl_ppl
    print(f"    PPL: {bl_ppl:.2f}")
    free(m)

    # 3. ShortGPT
    print(f"  [ShortGPT]")
    set_seed(42)
    m, t = load_fresh(model_name, device)
    t0 = time.time()
    bi = compute_block_influence(m, t, device=device)
    m, kept = remove_layers_by_bi(m, bi, L - target)
    sg_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    set_seed(42)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    sg_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    sg_time = time.time() - t0
    result["shortgpt"] = {"ppl_raw": sg_raw, "ppl_lora": sg_lora, "time": sg_time, "kept": kept}
    print(f"    PPL: raw={sg_raw:.2f}, +LoRA={sg_lora:.2f} ({sg_time:.0f}s)")
    free(m)

    # 4. FCM (cosine + spectral + select)
    print(f"  [FCM]")
    set_seed(42)
    comms = detect_communities(cos, method="spectral", n_clusters=target)
    stats = analyze_communities(comms, L)
    m, t = load_fresh(model_name, device)
    t0 = time.time()
    merged = [merge_community_select(m, comm, cos) for comm in comms]
    m = build_compressed_model(m, comms, merged)
    fcm_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    set_seed(42)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    fcm_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    fcm_time = time.time() - t0
    result["fcm"] = {"ppl_raw": fcm_raw, "ppl_lora": fcm_lora, "time": fcm_time,
                     "communities": [list(c) for c in comms], "stats": stats}
    print(f"    PPL: raw={fcm_raw:.2f}, +LoRA={fcm_lora:.2f} ({fcm_time:.0f}s)")
    free(m)

    # 5. FCM with CKA metric (ablation)
    print(f"  [FCM-CKA]")
    set_seed(42)
    comms_cka = detect_communities(cka, method="spectral", n_clusters=target)
    m, t = load_fresh(model_name, device)
    merged_cka = [merge_community_select(m, comm, cka) for comm in comms_cka]
    m = build_compressed_model(m, comms_cka, merged_cka)
    cka_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    set_seed(42)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    cka_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=ppl_n)
    result["fcm_cka"] = {"ppl_raw": cka_raw, "ppl_lora": cka_lora}
    print(f"    PPL: raw={cka_raw:.2f}, +LoRA={cka_lora:.2f}")
    free(m)

    # Summary
    delta_sg = (result["fcm"]["ppl_lora"] - result["shortgpt"]["ppl_lora"]) / result["shortgpt"]["ppl_lora"] * 100
    winner = "FCM" if delta_sg < 0 else "ShortGPT"
    result["delta_pct"] = delta_sg
    result["winner"] = winner
    print(f"  => {winner} wins ({delta_sg:+.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--lora_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=2048)
    parser.add_argument("--ppl_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp12")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_kw = dict(num_steps=args.lora_steps, lr=args.lora_lr,
                   lora_r=args.lora_r, num_samples=args.lora_samples)

    all_results = {}
    results_path = os.path.join(args.output_dir, "wide_sweep.json")

    for model_name in args.models:
        t0 = time.time()
        try:
            res = process_model(model_name, args.output_dir, args.device, lora_kw, args.ppl_n)
            all_results[model_name] = res
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

        elapsed = time.time() - t0
        print(f"  Time: {elapsed/60:.1f} min")

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*110}")
    print("WIDE MODEL SWEEP — 11 MODELS @ 2x COMPRESSION")
    print(f"{'='*110}")
    print(f"{'Model':<35} {'L':>3} {'Base':>6} {'SG':>8} {'FCM':>8} {'FCM-CKA':>8} {'CKA ratio':>10} {'Winner':>8}")
    print("-" * 110)

    fcm_wins = 0
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name.split('/')[-1]:<35} ERROR")
            continue
        short = name.split("/")[-1]
        bl = res["baseline_ppl"]
        sg = res["shortgpt"]["ppl_lora"]
        fcm = res["fcm"]["ppl_lora"]
        fcm_cka = res.get("fcm_cka", {}).get("ppl_lora", float("nan"))
        cka_r = res["cka"]["ratio"]
        w = res["winner"]
        if w == "FCM": fcm_wins += 1
        print(f"{short:<35} {res['L']:>3} {bl:>6.2f} {sg:>8.2f} {fcm:>8.2f} {fcm_cka:>8.2f} {cka_r:>9.1f}x {w:>8}")

    print("-" * 110)
    print(f"FCM wins: {fcm_wins}/{len([r for r in all_results.values() if 'error' not in r])}")
    print(f"{'='*110}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
