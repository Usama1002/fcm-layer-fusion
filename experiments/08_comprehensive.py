"""
Experiment 08: Comprehensive Multi-Model, Multi-Ablation Study

Covers:
  A. CKA universality analysis across 5 models
  B. Multi-model compression: FCM-select, FCM-average, ShortGPT
  C. Grouping ablation: spectral vs fisher vs uniform
  D. Similarity metric ablation: CKA vs cosine
  E. Compression ratio sweep per model
  F. Zero-shot benchmarks + perplexity

All methods get identical LoRA recovery. Results saved incrementally.
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.profiler import collect_activations
from src.cka import compute_similarity_matrix, compute_cka_matrix
from src.community import detect_communities, analyze_communities, print_communities
from src.select import select_representatives
from src.merge import merge_community_select, merge_community_average, build_compressed_model
from src.fusion import get_all_layers, get_layer_modules, distill_community
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.benchmark import evaluate_zero_shot
from src.ablations import fisher_segmentation, uniform_grouping
from src.visualize import plot_cka_matrix, plot_cka_off_diagonal_analysis

MODELS = [
    "Qwen/Qwen2.5-7B",        # 28 layers
    "mistralai/Mistral-7B-v0.3",  # 32 layers
    "meta-llama/Llama-3.1-8B",    # 32 layers
    "meta-llama/Llama-3.2-3B",    # 28 layers
    "Qwen/Qwen2.5-3B",            # 36 layers
]

ZS_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]


def load_fresh(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                              device_map=device, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

def eval_ppl(m, t, dev, n=50):
    return evaluate_perplexity(m, t, max_seq_len=2048, device=dev, max_samples=n)

def eval_zs(m, t, dev):
    try:
        return evaluate_zero_shot(m, t, tasks=ZS_TASKS, device=dev)
    except Exception as e:
        print(f"    ZS eval failed: {e}")
        return {}

def prep_cal(tok, n=1024):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = [s["text"] for s in ds if len(s.get("text", "")) > 50][:n]
    return tok(texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")


def run_shortgpt(model_name, L, target, device, lora_kw, ppl_n):
    """ShortGPT baseline."""
    m, t = load_fresh(model_name, device)
    bi = compute_block_influence(m, t, device=device)
    m, kept = remove_layers_by_bi(m, bi, L - target)
    ppl_raw = eval_ppl(m, t, device, ppl_n)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    ppl_lora = eval_ppl(m, t, device, ppl_n)
    zs = eval_zs(m, t, device)
    res = {"method": "shortgpt", "layers": len(get_all_layers(m)),
           "ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "zs": zs, "kept": kept}
    free(m)
    return res


def run_fcm(model_name, L, target, cka, device, lora_kw, ppl_n,
            grouping="spectral", merge="select", cal_enc=None, distill_kw=None):
    """FCM with configurable grouping and merge strategy."""
    # Grouping
    if grouping == "spectral":
        comms = detect_communities(cka, method="spectral", n_clusters=target)
    elif grouping == "fisher":
        comms = fisher_segmentation(cka, n_segments=target)
    elif grouping == "uniform":
        comms = uniform_grouping(L, target)
    else:
        raise ValueError(grouping)

    stats = analyze_communities(comms, L)
    print(f"    {grouping}: {stats['num_communities']} comms, non-adj={stats['non_adjacent_communities']}")

    m, t = load_fresh(model_name, device)

    # Merge
    merged = []
    for comm in comms:
        if merge == "select":
            merged.append(merge_community_select(m, comm, cka))
        elif merge == "average":
            merged.append(merge_community_average(m, comm, cka))
        elif merge == "distill":
            if len(comm) <= 1:
                merged.append(get_layer_modules(m, comm)[0])
            else:
                rep, loss = distill_community(m, comm, cka, cal_enc, device=device, **(distill_kw or {}))
                merged.append(rep)
        else:
            raise ValueError(merge)

    m = build_compressed_model(m, comms, merged)
    n_layers = len(get_all_layers(m))
    ppl_raw = eval_ppl(m, t, device, ppl_n)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    ppl_lora = eval_ppl(m, t, device, ppl_n)
    zs = eval_zs(m, t, device)
    res = {"method": f"fcm_{grouping}_{merge}", "layers": n_layers,
           "ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "zs": zs,
           "communities": [list(c) for c in comms], "stats": stats}
    free(m)
    return res


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--lora_steps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=4096)
    parser.add_argument("--distill_steps", type=int, default=500)
    parser.add_argument("--distill_lr", type=float, default=3e-5)
    parser.add_argument("--cal_samples", type=int, default=1024)
    parser.add_argument("--cka_samples", type=int, default=512)
    parser.add_argument("--ppl_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp08")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_distill", action="store_true", help="Skip distillation experiments (slow)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_kw = dict(num_steps=args.lora_steps, lr=args.lora_lr, lora_r=args.lora_r, num_samples=args.lora_samples)
    distill_kw = dict(num_steps=args.distill_steps, lr=args.distill_lr)

    all_results = {}
    results_path = os.path.join(args.output_dir, "comprehensive.json")

    for model_name in args.models:
        short = model_name.split("/")[-1]
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        L = cfg.num_hidden_layers
        target = max(L // 2, 4)  # 2x compression

        print(f"\n{'#'*80}")
        print(f"# {model_name} ({L} layers -> {target})")
        print(f"{'#'*80}")

        model_res = {"model": model_name, "L": L, "target": target, "experiments": {}}

        # ============================================================
        # PART A: CKA Analysis
        # ============================================================
        print(f"\n  === PART A: CKA Analysis ===")
        cka_path = os.path.join(args.output_dir, f"cka_{short}.npy")
        cos_path = os.path.join(args.output_dir, f"cosine_{short}.npy")

        if os.path.exists(cka_path):
            cka = np.load(cka_path)
            print(f"  Loaded CKA matrix from cache")
        else:
            acts = collect_activations(model_name, num_samples=args.cka_samples,
                                       max_seq_len=128, batch_size=4, device=args.device)
            cka = compute_similarity_matrix(acts, metric="cka")
            np.save(cka_path, cka)
            # Also compute cosine similarity matrix for ablation
            cos = compute_similarity_matrix(acts, metric="cosine")
            np.save(cos_path, cos)
            del acts; torch.cuda.empty_cache()

        if os.path.exists(cos_path):
            cos_matrix = np.load(cos_path)
        else:
            cos_matrix = None

        # CKA analysis stats
        plot_cka_matrix(cka, title=f"CKA - {short}",
                        save_path=os.path.join(args.output_dir, f"cka_{short}.png"))
        offdiag = plot_cka_off_diagonal_analysis(
            cka, save_path=os.path.join(args.output_dir, f"offdiag_{short}.png"))
        model_res["cka_analysis"] = {
            "adj_high_08": int(offdiag["num_adjacent_high"]),
            "nonadj_high_08": int(offdiag["num_nonadjacent_high"]),
            "ratio": offdiag["num_nonadjacent_high"] / max(1, offdiag["num_adjacent_high"]),
        }
        print(f"  CKA: adj={offdiag['num_adjacent_high']}, nonadj={offdiag['num_nonadjacent_high']}, "
              f"ratio={model_res['cka_analysis']['ratio']:.2f}x")

        # ============================================================
        # PART B: Baseline
        # ============================================================
        print(f"\n  === PART B: Baseline ===")
        m, t = load_fresh(model_name, args.device)
        bl_ppl = eval_ppl(m, t, args.device, args.ppl_n)
        bl_zs = eval_zs(m, t, args.device)
        model_res["baseline"] = {"ppl": bl_ppl, "zs": bl_zs}
        print(f"  Baseline PPL: {bl_ppl:.2f}, ZS avg: {bl_zs.get('avg', 0):.4f}")
        free(m)

        # Prepare calibration data
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        cal_enc = prep_cal(tok, args.cal_samples)

        # ============================================================
        # PART C: ShortGPT
        # ============================================================
        print(f"\n  === PART C: ShortGPT ===")
        try:
            sg = run_shortgpt(model_name, L, target, args.device, lora_kw, args.ppl_n)
            model_res["experiments"]["shortgpt"] = sg
            print(f"  ShortGPT: PPL raw={sg['ppl_raw']:.2f}, +LoRA={sg['ppl_lora']:.2f}, "
                  f"ZS={sg['zs'].get('avg', 0):.4f}")
        except Exception as e:
            print(f"  ShortGPT FAILED: {e}"); traceback.print_exc()

        # ============================================================
        # PART D: FCM variants (grouping x merge ablation)
        # ============================================================
        groupings = ["spectral", "fisher", "uniform"]
        merges = ["select", "average"]
        if not args.skip_distill:
            merges.append("distill")

        for grouping in groupings:
            for merge in merges:
                key = f"fcm_{grouping}_{merge}"
                print(f"\n  === {key} ===")
                try:
                    if merge == "distill":
                        res = run_fcm(model_name, L, target, cka, args.device, lora_kw, args.ppl_n,
                                      grouping=grouping, merge="distill",
                                      cal_enc=cal_enc, distill_kw=distill_kw)
                    else:
                        res = run_fcm(model_name, L, target, cka, args.device, lora_kw, args.ppl_n,
                                      grouping=grouping, merge=merge)
                    model_res["experiments"][key] = res
                    print(f"  {key}: PPL raw={res['ppl_raw']:.2f}, +LoRA={res['ppl_lora']:.2f}, "
                          f"ZS={res['zs'].get('avg', 0):.4f}")
                except Exception as e:
                    print(f"  {key} FAILED: {e}"); traceback.print_exc()
                    model_res["experiments"][key] = {"error": str(e)}

        # ============================================================
        # PART E: Similarity metric ablation (cosine instead of CKA)
        # ============================================================
        if cos_matrix is not None:
            print(f"\n  === Similarity Metric Ablation: Cosine ===")
            for merge in ["select", "average"]:
                key = f"fcm_spectral_{merge}_cosine"
                try:
                    comms_cos = detect_communities(cos_matrix, method="spectral", n_clusters=target)
                    m, t = load_fresh(model_name, args.device)
                    merged_layers = []
                    for comm in comms_cos:
                        if merge == "select":
                            merged_layers.append(merge_community_select(m, comm, cos_matrix))
                        else:
                            merged_layers.append(merge_community_average(m, comm, cos_matrix))
                    m = build_compressed_model(m, comms_cos, merged_layers)
                    ppl_raw = eval_ppl(m, t, args.device, args.ppl_n)
                    m, _ = lora_recovery_finetune(m, t, device=args.device, **lora_kw)
                    ppl_lora = eval_ppl(m, t, args.device, args.ppl_n)
                    zs = eval_zs(m, t, args.device)
                    model_res["experiments"][key] = {
                        "method": key, "ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "zs": zs
                    }
                    print(f"  {key}: PPL +LoRA={ppl_lora:.2f}")
                    free(m)
                except Exception as e:
                    print(f"  {key} FAILED: {e}"); traceback.print_exc()

        # Save incrementally
        all_results[model_name] = model_res
        save_results(all_results, results_path)
        print(f"\n  Saved results for {short}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*120}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*120}")

    # CKA universality
    print(f"\n--- CKA Universality ---")
    print(f"{'Model':<25} {'Layers':>6} {'Adj>=0.8':>10} {'NonAdj>=0.8':>12} {'Ratio':>8}")
    print("-" * 65)
    for name, res in all_results.items():
        if "cka_analysis" not in res: continue
        c = res["cka_analysis"]
        print(f"{name.split('/')[-1]:<25} {res['L']:>6} {c['adj_high_08']:>10} {c['nonadj_high_08']:>12} {c['ratio']:>7.2f}x")

    # Compression comparison
    print(f"\n--- Compression @ 2x ---")
    print(f"{'Model':<20} {'Method':<30} {'PPL raw':>10} {'PPL+LoRA':>10} {'ZS Avg':>8}")
    print("-" * 85)
    for name, res in all_results.items():
        short = name.split('/')[-1][:18]
        bl = res.get("baseline", {})
        print(f"{short:<20} {'Baseline':<30} {bl.get('ppl',0):>10.2f} {'--':>10} {bl.get('zs',{}).get('avg',0):>8.4f}")
        for key in sorted(res.get("experiments", {}).keys()):
            r = res["experiments"][key]
            if "error" in r: continue
            ppl_r = r.get("ppl_raw", float("nan"))
            ppl_l = r.get("ppl_lora", float("nan"))
            zs_a = r.get("zs", {}).get("avg", 0)
            print(f"{'':20} {key:<30} {ppl_r:>10.2f} {ppl_l:>10.2f} {zs_a:>8.4f}")
        print("-" * 85)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
