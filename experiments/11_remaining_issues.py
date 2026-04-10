"""
Experiment 11: Address remaining review issues

A. Wall-clock timing comparison (FCM pipeline vs ShortGPT)
B. Calibration size ablation (N=64, 128, 256, 512, 1024)
C. Compression ratio sweep on Qwen2.5-7B (1.2x to 3.5x)
D. Additional baselines: Contiguous Block Removal (LaCo-style)

All on Qwen2.5-7B with 3-seed variance where feasible.
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

MODEL = "Qwen/Qwen2.5-7B"
DEVICE = "cuda"


def load_fresh(device="cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                              device_map=device, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def contiguous_block_removal(model, bi_scores, num_remove):
    """
    LaCo-style: find the contiguous block of `num_remove` layers
    with the lowest total BI score and remove it.
    """
    import torch.nn as nn
    L = len(bi_scores)
    best_start = 0
    best_score = float("inf")
    for start in range(L - num_remove + 1):
        score = sum(bi_scores[start:start + num_remove])
        if score < best_score:
            best_score = score
            best_start = start

    keep = [i for i in range(L) if i < best_start or i >= best_start + num_remove]

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        new_layers = nn.ModuleList([layers[i] for i in keep])
        model.model.layers = new_layers
        model.config.num_hidden_layers = len(new_layers)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    return model, keep


def part_a_timing(output_dir):
    """Wall-clock timing of each FCM pipeline stage."""
    print("\n" + "=" * 70)
    print("PART A: Wall-Clock Timing")
    print("=" * 70)

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = L // 2

    timings = {}

    # Stage 1: CKA/Cosine profiling
    t0 = time.time()
    acts = collect_activations(MODEL, num_samples=512, max_seq_len=128, batch_size=4, device=DEVICE)
    t_profile = time.time() - t0
    timings["profiling"] = t_profile
    print(f"  Profiling: {t_profile:.1f}s")

    t0 = time.time()
    cos = compute_similarity_matrix(acts, metric="cosine")
    t_sim = time.time() - t0
    timings["similarity"] = t_sim
    print(f"  Similarity matrix: {t_sim:.1f}s")
    del acts; torch.cuda.empty_cache()

    # Stage 2: Community detection
    t0 = time.time()
    comms = detect_communities(cos, method="spectral", n_clusters=target)
    t_community = time.time() - t0
    timings["community_detection"] = t_community
    print(f"  Community detection: {t_community:.3f}s")

    # Stage 3: Layer selection + model assembly
    m, tok = load_fresh()
    t0 = time.time()
    merged = [merge_community_select(m, comm, cos) for comm in comms]
    m = build_compressed_model(m, comms, merged)
    t_select = time.time() - t0
    timings["selection_assembly"] = t_select
    print(f"  Selection + assembly: {t_select:.3f}s")

    # Stage 4: LoRA recovery
    t0 = time.time()
    m, _ = lora_recovery_finetune(m, tok, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
    t_lora = time.time() - t0
    timings["lora_recovery"] = t_lora
    print(f"  LoRA recovery: {t_lora:.1f}s")
    free(m)

    timings["total_fcm"] = t_profile + t_sim + t_community + t_select + t_lora

    # ShortGPT timing
    m, tok = load_fresh()
    t0 = time.time()
    bi = compute_block_influence(m, tok, device=DEVICE)
    m, _ = remove_layers_by_bi(m, bi, L - target)
    t_shortgpt_compress = time.time() - t0
    timings["shortgpt_compress"] = t_shortgpt_compress

    t0 = time.time()
    m, _ = lora_recovery_finetune(m, tok, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
    t_shortgpt_lora = time.time() - t0
    timings["shortgpt_lora"] = t_shortgpt_lora
    timings["total_shortgpt"] = t_shortgpt_compress + t_shortgpt_lora
    free(m)

    print(f"\n  FCM total: {timings['total_fcm']:.1f}s ({timings['total_fcm']/60:.1f} min)")
    print(f"  ShortGPT total: {timings['total_shortgpt']:.1f}s ({timings['total_shortgpt']/60:.1f} min)")

    with open(os.path.join(output_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    return timings


def part_b_calibration_ablation(output_dir):
    """Calibration set size ablation."""
    print("\n" + "=" * 70)
    print("PART B: Calibration Size Ablation")
    print("=" * 70)

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = L // 2
    sizes = [64, 128, 256, 512, 1024]
    results = {}

    for N in sizes:
        print(f"\n  N={N}:")
        acts = collect_activations(MODEL, num_samples=N, max_seq_len=128, batch_size=4, device=DEVICE)
        cos = compute_similarity_matrix(acts, metric="cosine")
        del acts; torch.cuda.empty_cache()

        comms = detect_communities(cos, method="spectral", n_clusters=target)
        set_seed(42)
        m, tok = load_fresh()
        merged = [merge_community_select(m, comm, cos) for comm in comms]
        m = build_compressed_model(m, comms, merged)
        ppl_raw = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        m, _ = lora_recovery_finetune(m, tok, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
        ppl_lora = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        free(m)

        results[N] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora}
        print(f"    PPL raw={ppl_raw:.2f}, +LoRA={ppl_lora:.2f}")

    with open(os.path.join(output_dir, "calibration_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Calibration Size Summary:")
    for N, r in sorted(results.items(), key=lambda x: int(x[0])):
        print(f"    N={N:>5}: PPL raw={r['ppl_raw']:>10.2f}, +LoRA={r['ppl_lora']:>8.2f}")
    return results


def part_c_compression_sweep(output_dir):
    """Compression ratio sweep on Qwen2.5-7B."""
    print("\n" + "=" * 70)
    print("PART C: Compression Ratio Sweep")
    print("=" * 70)

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    L = cfg.num_hidden_layers  # 28

    # Load cosine matrix
    cos_path = "results/exp09/cosine_Qwen2.5-7B.npy"
    if os.path.exists(cos_path):
        cos = np.load(cos_path)
    else:
        acts = collect_activations(MODEL, num_samples=512, max_seq_len=128, batch_size=4, device=DEVICE)
        cos = compute_similarity_matrix(acts, metric="cosine")
        del acts; torch.cuda.empty_cache()

    targets = [24, 22, 20, 18, 16, 14, 12, 10, 8]  # various compression levels
    results = {}

    for target in targets:
        ratio = L / target
        print(f"\n  Target={target} ({ratio:.1f}x):")

        # FCM
        comms = detect_communities(cos, method="spectral", n_clusters=target)
        set_seed(42)
        m, tok = load_fresh()
        merged = [merge_community_select(m, comm, cos) for comm in comms]
        m = build_compressed_model(m, comms, merged)
        fcm_raw = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        m, _ = lora_recovery_finetune(m, tok, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
        fcm_lora = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        free(m)

        # ShortGPT
        set_seed(42)
        m, tok = load_fresh()
        bi = compute_block_influence(m, tok, device=DEVICE)
        m, _ = remove_layers_by_bi(m, bi, L - target)
        sg_raw = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        m, _ = lora_recovery_finetune(m, tok, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
        sg_lora = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        free(m)

        results[target] = {
            "ratio": ratio,
            "fcm_raw": fcm_raw, "fcm_lora": fcm_lora,
            "sg_raw": sg_raw, "sg_lora": sg_lora,
        }
        print(f"    FCM: raw={fcm_raw:.2f}, +LoRA={fcm_lora:.2f}")
        print(f"    SG:  raw={sg_raw:.2f}, +LoRA={sg_lora:.2f}")

    with open(os.path.join(output_dir, "compression_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Compression Sweep Summary:")
    print(f"  {'Target':>6} {'Ratio':>6} {'FCM':>10} {'ShortGPT':>10} {'Winner':>8}")
    for t in targets:
        r = results[t]
        w = "FCM" if r["fcm_lora"] < r["sg_lora"] else "SG"
        print(f"  {t:>6} {r['ratio']:>5.1f}x {r['fcm_lora']:>10.2f} {r['sg_lora']:>10.2f} {w:>8}")
    return results


def part_d_laco_baseline(output_dir):
    """LaCo-style contiguous block removal baseline."""
    print("\n" + "=" * 70)
    print("PART D: LaCo-Style Contiguous Block Removal")
    print("=" * 70)

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = L // 2
    num_remove = L - target

    results = {}
    for seed in [42, 123, 7]:
        set_seed(seed)
        m, tok = load_fresh()
        bi = compute_block_influence(m, tok, device=DEVICE)
        m, kept = contiguous_block_removal(m, bi, num_remove)
        ppl_raw = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        set_seed(seed)
        m, _ = lora_recovery_finetune(m, tok, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
        ppl_lora = evaluate_perplexity(m, tok, max_seq_len=2048, device=DEVICE, max_samples=50)
        free(m)
        results[seed] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora, "kept": kept}
        print(f"  Seed {seed}: raw={ppl_raw:.2f}, +LoRA={ppl_lora:.2f}, kept={kept}")

    ppls = [r["ppl_lora"] for r in results.values()]
    results["mean"] = float(np.mean(ppls))
    results["std"] = float(np.std(ppls))
    print(f"\n  LaCo-style: {results['mean']:.2f} +/- {results['std']:.2f}")

    with open(os.path.join(output_dir, "laco_baseline.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/exp11")
    parser.add_argument("--parts", type=str, default="abcd", help="Which parts to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if "a" in args.parts:
        part_a_timing(args.output_dir)
    if "b" in args.parts:
        part_b_calibration_ablation(args.output_dir)
    if "c" in args.parts:
        part_c_compression_sweep(args.output_dir)
    if "d" in args.parts:
        part_d_laco_baseline(args.output_dir)

    print("\n" + "=" * 70)
    print("ALL PARTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
