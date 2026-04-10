"""
Experiment 06: Multi-Model Evaluation

Comprehensive evaluation across multiple model families and sizes.
For each model:
  1. Compute CKA matrix
  2. Run community detection
  3. Compress with FCM-spectral, FCM-fisher, ShortGPT
  4. LoRA recovery
  5. Evaluate perplexity + zero-shot benchmarks

Usage:
    python experiments/06_multi_model.py
    python experiments/06_multi_model.py --models Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.3
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
from src.fusion import distill_community, build_fused_model, get_all_layers, get_layer_modules
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.benchmark import evaluate_zero_shot
from src.ablations import fisher_segmentation
from src.visualize import plot_cka_matrix, plot_cka_off_diagonal_analysis


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
]

ZERO_SHOT_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]


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


def prepare_calibration(tokenizer, num_samples=1024, max_seq_len=128):
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


def run_compression(model, tokenizer, cka_matrix, communities, cal_enc, device, args):
    """Distill communities, build fused model, LoRA recovery, evaluate."""
    rep_layers = []
    for i, comm in enumerate(communities):
        if len(comm) <= 1:
            rep_layers.append(get_layer_modules(model, comm)[0])
            continue
        print(f"    C{i}: {comm} (size={len(comm)})")
        rep, loss = distill_community(
            model, comm, cka_matrix, cal_enc,
            num_steps=args.distill_steps, lr=args.distill_lr, device=device,
        )
        rep_layers.append(rep)

    model = build_fused_model(model, communities, rep_layers)
    n = len(get_all_layers(model))
    print(f"    Fused: {n} layers")

    # Perplexity
    ppl_raw = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                   device=device, max_samples=args.ppl_samples)
    print(f"    PPL (raw): {ppl_raw:.2f}")

    # LoRA recovery
    model, _ = lora_recovery_finetune(
        model, tokenizer, num_steps=args.lora_steps, lr=args.lora_lr,
        lora_r=args.lora_r, num_samples=args.lora_samples, device=device,
    )
    ppl_lora = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                    device=device, max_samples=args.ppl_samples)
    print(f"    PPL (LoRA): {ppl_lora:.2f}")

    # Zero-shot (quick subset)
    zs = evaluate_zero_shot(model, tokenizer, tasks=ZERO_SHOT_TASKS, device=device)
    print(f"    Zero-shot avg: {zs.get('avg', 0):.4f}")

    return {"layers": n, "ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "zero_shot": zs}


def process_model(model_name, args):
    """Full pipeline for one model."""
    model_short = model_name.split("/")[-1]
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    L = config.num_hidden_layers
    target = max(L // 2, 4)  # 2x compression

    print(f"\n{'#'*80}")
    print(f"# MODEL: {model_name} ({L} layers -> {target})")
    print(f"{'#'*80}")

    model_results = {"model": model_name, "original_layers": L, "target_layers": target}

    # Step 1: CKA Matrix
    cka_path = os.path.join(args.output_dir, f"cka_{model_short}.npy")
    if os.path.exists(cka_path):
        print(f"\n  Loading cached CKA matrix from {cka_path}")
        cka_matrix = np.load(cka_path)
    else:
        print(f"\n  Computing CKA matrix...")
        activations = collect_activations(
            model_name=model_name, num_samples=args.cka_samples,
            max_seq_len=128, batch_size=4, device=args.device,
        )
        cka_matrix = compute_cka_matrix(activations)
        np.save(cka_path, cka_matrix)
        del activations
        torch.cuda.empty_cache()

    # Save CKA visualization
    plot_cka_matrix(cka_matrix, title=f"CKA — {model_short}",
                    save_path=os.path.join(args.output_dir, f"cka_{model_short}.png"))
    stats = plot_cka_off_diagonal_analysis(
        cka_matrix, save_path=os.path.join(args.output_dir, f"cka_offdiag_{model_short}.png"))
    model_results["cka_stats"] = {
        "nonadj_high": int(stats["num_nonadjacent_high"]),
        "adj_high": int(stats["num_adjacent_high"]),
    }

    # Step 2: Baseline perplexity + zero-shot
    print(f"\n  --- Baseline ---")
    model, tokenizer = load_fresh(model_name, args.device)
    baseline_ppl = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                       device=args.device, max_samples=args.ppl_samples)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    baseline_zs = evaluate_zero_shot(model, tokenizer, tasks=ZERO_SHOT_TASKS, device=args.device)
    print(f"  Baseline zero-shot avg: {baseline_zs.get('avg', 0):.4f}")
    model_results["baseline"] = {"ppl": baseline_ppl, "zero_shot": baseline_zs}
    free(model)

    # Prepare calibration
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cal_enc = prepare_calibration(tokenizer, args.cal_samples)

    # Step 3: ShortGPT
    print(f"\n  --- ShortGPT ---")
    model, tokenizer = load_fresh(model_name, args.device)
    bi = compute_block_influence(model, tokenizer, device=args.device)
    model, kept = remove_layers_by_bi(model, bi, L - target)
    print(f"  Kept: {kept}")
    ppl_raw = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                   device=args.device, max_samples=args.ppl_samples)
    print(f"  PPL (raw): {ppl_raw:.2f}")
    model, _ = lora_recovery_finetune(
        model, tokenizer, num_steps=args.lora_steps, lr=args.lora_lr,
        lora_r=args.lora_r, num_samples=args.lora_samples, device=args.device)
    ppl_lora = evaluate_perplexity(model, tokenizer, max_seq_len=2048,
                                    device=args.device, max_samples=args.ppl_samples)
    print(f"  PPL (LoRA): {ppl_lora:.2f}")
    sg_zs = evaluate_zero_shot(model, tokenizer, tasks=ZERO_SHOT_TASKS, device=args.device)
    print(f"  Zero-shot avg: {sg_zs.get('avg', 0):.4f}")
    model_results["shortgpt"] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora,
                                  "zero_shot": sg_zs, "kept": kept}
    free(model)

    # Step 4: FCM-spectral
    print(f"\n  --- FCM-spectral ---")
    communities = detect_communities(cka_matrix, method="spectral", n_clusters=target)
    print_communities(communities, analyze_communities(communities, L))
    model, tokenizer = load_fresh(model_name, args.device)
    fcm_res = run_compression(model, tokenizer, cka_matrix, communities, cal_enc, args.device, args)
    fcm_res["communities"] = communities
    model_results["fcm_spectral"] = fcm_res
    free(model)

    # Step 5: FCM-fisher (contiguous ablation)
    print(f"\n  --- FCM-fisher (contiguous) ---")
    fisher_comms = fisher_segmentation(cka_matrix, n_segments=target)
    print_communities(fisher_comms, analyze_communities(fisher_comms, L))
    model, tokenizer = load_fresh(model_name, args.device)
    fisher_res = run_compression(model, tokenizer, cka_matrix, fisher_comms, cal_enc, args.device, args)
    fisher_res["communities"] = fisher_comms
    model_results["fcm_fisher"] = fisher_res
    free(model)

    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cka_samples", type=int, default=512)
    parser.add_argument("--cal_samples", type=int, default=1024)
    parser.add_argument("--distill_steps", type=int, default=500)
    parser.add_argument("--distill_lr", type=float, default=3e-5)
    parser.add_argument("--lora_steps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=4096)
    parser.add_argument("--ppl_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp06")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Experiment 06: Multi-Model Comprehensive Evaluation")
    print(f"Models: {args.models}")
    print("=" * 80)

    all_results = {}
    for model_name in args.models:
        t0 = time.time()
        try:
            res = process_model(model_name, args)
            all_results[model_name] = res
            # Save incrementally
            results_path = os.path.join(args.output_dir, "all_results.json")
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            print(f"\n  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
        elapsed = time.time() - t0
        print(f"\n  Time for {model_name}: {elapsed/60:.1f} min")

    # Final summary table
    print(f"\n{'='*100}")
    print("COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'Method':<18} {'Layers':>6} {'PPL':>8} {'ARC-e':>7} {'ARC-c':>7} {'HellaS':>7} {'WinoG':>7} {'PIQA':>7} {'Avg':>7}")
    print("-" * 100)

    for model_name, res in all_results.items():
        if "error" in res:
            continue
        mshort = model_name.split("/")[-1][:20]
        L = res["original_layers"]
        bl = res["baseline"]
        zs = bl.get("zero_shot", {})
        print(f"{mshort:<25} {'Baseline':<18} {L:>6} {bl['ppl']:>8.2f} "
              f"{zs.get('arc_easy',0):>7.4f} {zs.get('arc_challenge',0):>7.4f} "
              f"{zs.get('hellaswag',0):>7.4f} {zs.get('winogrande',0):>7.4f} "
              f"{zs.get('piqa',0):>7.4f} {zs.get('avg',0):>7.4f}")

        for method in ["shortgpt", "fcm_spectral", "fcm_fisher"]:
            if method in res:
                r = res[method]
                zs = r.get("zero_shot", {})
                n = r.get("layers", "?")
                ppl = r.get("ppl_lora", r.get("ppl_raw", float("nan")))
                print(f"{'':25} {method:<18} {n:>6} {ppl:>8.2f} "
                      f"{zs.get('arc_easy',0):>7.4f} {zs.get('arc_challenge',0):>7.4f} "
                      f"{zs.get('hellaswag',0):>7.4f} {zs.get('winogrande',0):>7.4f} "
                      f"{zs.get('piqa',0):>7.4f} {zs.get('avg',0):>7.4f}")
        print("-" * 100)

    print(f"{'='*100}")
    print(f"Results saved to {os.path.join(args.output_dir, 'all_results.json')}")


if __name__ == "__main__":
    main()
