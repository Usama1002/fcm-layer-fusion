"""
Experiment 07: Graph-Guided Selection vs Distillation vs ShortGPT

Compares at 2x compression:
  1. ShortGPT: remove layers with lowest BI scores
  2. FCM-select-spectral: graph communities + keep central layer (our main method)
  3. FCM-select-fisher: contiguous groups + keep central (ablation)
  4. FCM-distill-spectral: graph communities + distill (original FCM)

All methods followed by identical LoRA recovery for fair comparison.

Usage:
    python experiments/07_select_vs_distill.py --model Qwen/Qwen2.5-7B
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
from src.cka import compute_cka_matrix
from src.community import detect_communities, analyze_communities, print_communities
from src.select import select_representatives
from src.fusion import distill_community, build_fused_model, get_all_layers, get_layer_modules
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.benchmark import evaluate_zero_shot
from src.ablations import fisher_segmentation
from src.visualize import plot_cka_matrix

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]


def load_fresh(model_name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    return model, tok


def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()


def eval_model(model, tok, device, ppl_samples):
    ppl = evaluate_perplexity(model, tok, max_seq_len=2048, device=device, max_samples=ppl_samples)
    zs = evaluate_zero_shot(model, tok, tasks=TASKS, device=device)
    return ppl, zs


def prepare_cal(tok, n=1024, seq_len=128):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = [s["text"] for s in ds if len(s.get("text",""))>50][:n]
    return tok(texts, max_length=seq_len, truncation=True, padding="max_length", return_tensors="pt")


def run_on_model(model_name, args):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = max(L // 2, 4)
    short = model_name.split("/")[-1]

    print(f"\n{'#'*80}")
    print(f"# {model_name}: {L} -> {target} layers (2x)")
    print(f"{'#'*80}")

    results = {"model": model_name, "L": L, "target": target}

    # CKA matrix
    cka_path = os.path.join(args.output_dir, f"cka_{short}.npy")
    if os.path.exists(cka_path):
        cka = np.load(cka_path)
    else:
        acts = collect_activations(model_name, num_samples=512, max_seq_len=128, batch_size=4, device=args.device)
        cka = compute_cka_matrix(acts); del acts; torch.cuda.empty_cache()
        np.save(cka_path, cka)
        plot_cka_matrix(cka, title=f"CKA — {short}", save_path=os.path.join(args.output_dir, f"cka_{short}.png"))

    # Baseline
    print(f"\n  [Baseline]")
    m, t = load_fresh(model_name, args.device)
    bl_ppl, bl_zs = eval_model(m, t, args.device, args.ppl_samples)
    print(f"  PPL: {bl_ppl:.2f}, ZS avg: {bl_zs.get('avg',0):.4f}")
    results["baseline"] = {"ppl": bl_ppl, "zs": bl_zs}
    free(m)

    # Method 1: ShortGPT
    print(f"\n  [ShortGPT]")
    m, t = load_fresh(model_name, args.device)
    bi = compute_block_influence(m, t, device=args.device)
    m, kept = remove_layers_by_bi(m, bi, L - target)
    print(f"  Kept: {kept}")
    ppl_raw, _ = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=args.ppl_samples), None
    print(f"  PPL raw: {ppl_raw:.2f}")
    m, _ = lora_recovery_finetune(m, t, num_steps=args.lora_steps, lr=args.lora_lr,
                                   lora_r=args.lora_r, num_samples=args.lora_samples, device=args.device)
    sg_ppl, sg_zs = eval_model(m, t, args.device, args.ppl_samples)
    print(f"  PPL +LoRA: {sg_ppl:.2f}, ZS avg: {sg_zs.get('avg',0):.4f}")
    results["shortgpt"] = {"ppl_raw": ppl_raw, "ppl_lora": sg_ppl, "zs": sg_zs, "kept": kept}
    free(m)

    # Method 2: FCM-select-spectral (our main method)
    print(f"\n  [FCM-select-spectral]")
    comms_spectral = detect_communities(cka, method="spectral", n_clusters=target)
    stats = analyze_communities(comms_spectral, L)
    print_communities(comms_spectral, stats)
    m, t = load_fresh(model_name, args.device)
    m, sel_kept = select_representatives(m, comms_spectral, cka)
    print(f"  Selected layers: {sel_kept}")
    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=args.ppl_samples)
    print(f"  PPL raw: {ppl_raw:.2f}")
    m, _ = lora_recovery_finetune(m, t, num_steps=args.lora_steps, lr=args.lora_lr,
                                   lora_r=args.lora_r, num_samples=args.lora_samples, device=args.device)
    sel_ppl, sel_zs = eval_model(m, t, args.device, args.ppl_samples)
    print(f"  PPL +LoRA: {sel_ppl:.2f}, ZS avg: {sel_zs.get('avg',0):.4f}")
    results["fcm_select_spectral"] = {"ppl_raw": ppl_raw, "ppl_lora": sel_ppl, "zs": sel_zs,
                                       "kept": sel_kept, "communities": comms_spectral}
    free(m)

    # Method 3: FCM-select-fisher (contiguous ablation)
    print(f"\n  [FCM-select-fisher]")
    comms_fisher = fisher_segmentation(cka, n_segments=target)
    print_communities(comms_fisher, analyze_communities(comms_fisher, L))
    m, t = load_fresh(model_name, args.device)
    m, fish_kept = select_representatives(m, comms_fisher, cka)
    print(f"  Selected layers: {fish_kept}")
    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=args.ppl_samples)
    print(f"  PPL raw: {ppl_raw:.2f}")
    m, _ = lora_recovery_finetune(m, t, num_steps=args.lora_steps, lr=args.lora_lr,
                                   lora_r=args.lora_r, num_samples=args.lora_samples, device=args.device)
    fish_ppl, fish_zs = eval_model(m, t, args.device, args.ppl_samples)
    print(f"  PPL +LoRA: {fish_ppl:.2f}, ZS avg: {fish_zs.get('avg',0):.4f}")
    results["fcm_select_fisher"] = {"ppl_raw": ppl_raw, "ppl_lora": fish_ppl, "zs": fish_zs,
                                     "kept": fish_kept, "communities": comms_fisher}
    free(m)

    # Method 4: FCM-distill-spectral (original distillation approach)
    print(f"\n  [FCM-distill-spectral]")
    m, t = load_fresh(model_name, args.device)
    cal = prepare_cal(t, args.cal_samples)
    rep_layers = []
    for i, comm in enumerate(comms_spectral):
        if len(comm) <= 1:
            rep_layers.append(get_layer_modules(m, comm)[0])
            continue
        rep, loss = distill_community(m, comm, cka, cal,
                                       num_steps=args.distill_steps, lr=args.distill_lr, device=args.device)
        rep_layers.append(rep)
    m = build_fused_model(m, comms_spectral, rep_layers)
    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=args.device, max_samples=args.ppl_samples)
    print(f"  PPL raw: {ppl_raw:.2f}")
    m, _ = lora_recovery_finetune(m, t, num_steps=args.lora_steps, lr=args.lora_lr,
                                   lora_r=args.lora_r, num_samples=args.lora_samples, device=args.device)
    dist_ppl, dist_zs = eval_model(m, t, args.device, args.ppl_samples)
    print(f"  PPL +LoRA: {dist_ppl:.2f}, ZS avg: {dist_zs.get('avg',0):.4f}")
    results["fcm_distill_spectral"] = {"ppl_raw": ppl_raw, "ppl_lora": dist_ppl, "zs": dist_zs}
    free(m)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.2-3B"])
    parser.add_argument("--distill_steps", type=int, default=500)
    parser.add_argument("--distill_lr", type=float, default=3e-5)
    parser.add_argument("--cal_samples", type=int, default=1024)
    parser.add_argument("--lora_steps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=4096)
    parser.add_argument("--ppl_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp07")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for model_name in args.models:
        t0 = time.time()
        try:
            res = run_on_model(model_name, args)
            all_results[model_name] = res
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
        print(f"\n  Time: {(time.time()-t0)/60:.1f} min")

    # Summary table
    print(f"\n{'='*110}")
    print(f"{'Model':<22} {'Method':<22} {'Layers':>6} {'PPL':>8} {'ARC-e':>7} {'ARC-c':>7} {'HellaS':>7} {'WinoG':>7} {'PIQA':>7} {'ZS Avg':>7}")
    print("-" * 110)
    for model_name, res in all_results.items():
        if "error" in res: continue
        short = model_name.split("/")[-1][:18]
        bl = res["baseline"]
        zs = bl.get("zs", {})
        print(f"{short:<22} {'Baseline':<22} {res['L']:>6} {bl['ppl']:>8.2f} "
              f"{zs.get('arc_easy',0):>7.3f} {zs.get('arc_challenge',0):>7.3f} "
              f"{zs.get('hellaswag',0):>7.3f} {zs.get('winogrande',0):>7.3f} "
              f"{zs.get('piqa',0):>7.3f} {zs.get('avg',0):>7.3f}")
        for meth in ["shortgpt", "fcm_select_spectral", "fcm_select_fisher", "fcm_distill_spectral"]:
            if meth not in res: continue
            r = res[meth]
            zs = r.get("zs", {})
            n = r.get("layers", len(r.get("kept", [])))
            if not n: n = res["target"]
            ppl = r.get("ppl_lora", r.get("ppl_raw", float("nan")))
            print(f"{'':22} {meth:<22} {n:>6} {ppl:>8.2f} "
                  f"{zs.get('arc_easy',0):>7.3f} {zs.get('arc_challenge',0):>7.3f} "
                  f"{zs.get('hellaswag',0):>7.3f} {zs.get('winogrande',0):>7.3f} "
                  f"{zs.get('piqa',0):>7.3f} {zs.get('avg',0):>7.3f}")
        print("-" * 110)
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
