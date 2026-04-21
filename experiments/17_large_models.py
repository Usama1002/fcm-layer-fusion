"""
Experiment 17: Large Models (14B) on A100 GPUs

Tests FCM-Hybrid and ShortGPT on 14B-scale models that don't fit on
smaller GPUs. Proves FCM scales to larger models.

Run on A100-80GB:
    python experiments/17_large_models.py --device cuda:0

Each 14B model needs ~28GB fp16 + LoRA overhead. Fits on A100 with room.
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
from src.fusion import get_all_layers
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.visualize import plot_cka_matrix, plot_cka_off_diagonal_analysis

SEEDS = [42, 123, 7]
LARGE_MODELS = [
    "Qwen/Qwen2.5-14B",    # 40 layers, d=5120
    "Qwen/Qwen3-14B",      # 40 layers, d=5120
]
DEVICE = "cuda:0"


def load_fresh(name, device):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_shortgpt(model_name, bi, L, target, seed, device, lora_kw):
    set_seed(seed)
    m, t = load_fresh(model_name, device)
    m, kept = remove_layers_by_bi(m, bi, L - target)
    set_seed(seed)
    m, _ = lora_recovery_finetune(m, t, device=device, **lora_kw)
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=50)
    free(m)
    return ppl


def run_fcm_hybrid(model_name, sim_matrix, bi, L, target, seed, device, lora_kw):
    set_seed(seed)
    comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)
    keep = []
    for comm in comms:
        if len(comm) == 1:
            keep.append(comm[0])
        else:
            keep.append(max(comm, key=lambda i: bi[i]))
    keep = sorted(keep)

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
    ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=50)
    free(m)
    return ppl


def process_model(model_name, device, output_dir, lora_kw):
    short = model_name.split("/")[-1]
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = L // 2

    print(f"\n{'#'*70}")
    print(f"# {model_name} ({L} layers -> {target}, 2x compression)")
    print(f"{'#'*70}")

    result = {"model": model_name, "L": L, "target": target}

    # Step 1: CKA/Cosine profiling
    cos_path = os.path.join(output_dir, f"cosine_{short}.npy")
    cka_path = os.path.join(output_dir, f"cka_{short}.npy")

    if os.path.exists(cos_path):
        cos = np.load(cos_path)
        cka = np.load(cka_path) if os.path.exists(cka_path) else cos
        print(f"  Loaded cached similarity matrices")
    else:
        print(f"  Computing similarity matrices...")
        acts = collect_activations(model_name, num_samples=512, max_seq_len=128,
                                   batch_size=2, device=device)
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
    result["cka"] = {
        "adj": int(offdiag["num_adjacent_high"]),
        "nonadj": int(offdiag["num_nonadjacent_high"]),
        "ratio": offdiag["num_nonadjacent_high"] / max(1, offdiag["num_adjacent_high"]),
    }
    print(f"  CKA ratio: {result['cka']['ratio']:.2f}x")

    # Step 2: Baseline PPL
    m, t = load_fresh(model_name, device)
    bl_ppl = evaluate_perplexity(m, t, max_seq_len=2048, device=device, max_samples=50)
    result["baseline_ppl"] = bl_ppl
    print(f"  Baseline PPL: {bl_ppl:.2f}")

    # Step 3: BI scores (compute once, reuse)
    bi = compute_block_influence(m, t, device=device)
    result["bi_scores"] = bi.tolist()
    free(m)

    # Step 4: 3-seed comparison at 2x
    sg_ppls, fcm_ppls = [], []
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        sg = run_shortgpt(model_name, bi, L, target, seed, device, lora_kw)
        sg_ppls.append(sg)
        print(f"    ShortGPT: {sg:.2f}")

        fcm = run_fcm_hybrid(model_name, cos, bi, L, target, seed, device, lora_kw)
        fcm_ppls.append(fcm)
        print(f"    FCM-Hybrid: {fcm:.2f}")

    sg_m, sg_s = np.mean(sg_ppls), np.std(sg_ppls)
    fcm_m, fcm_s = np.mean(fcm_ppls), np.std(fcm_ppls)
    delta = (fcm_m - sg_m) / sg_m * 100

    result["shortgpt"] = {"mean": sg_m, "std": sg_s, "runs": sg_ppls}
    result["fcm_hybrid"] = {"mean": fcm_m, "std": fcm_s, "runs": fcm_ppls}
    result["delta_pct"] = delta
    result["winner"] = "FCM" if fcm_m < sg_m else "ShortGPT"

    print(f"\n  RESULT: SG={sg_m:.2f}+/-{sg_s:.2f}  FCM={fcm_m:.2f}+/-{fcm_s:.2f}  ({delta:+.1f}%)  {result['winner']}")

    # Step 5: Compression ratio sweep (1.5x, 2x, 2.5x, 3x)
    print(f"\n  Compression sweep:")
    sweep = {}
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        tgt = max(int(L / ratio), 4)
        set_seed(42)

        # ShortGPT
        sg = run_shortgpt(model_name, bi, L, tgt, 42, device, lora_kw)

        # FCM-Hybrid
        fcm = run_fcm_hybrid(model_name, cos, bi, L, tgt, 42, device, lora_kw)

        d = (fcm - sg) / sg * 100
        sweep[f"{ratio:.1f}x"] = {"target": tgt, "shortgpt": sg, "fcm": fcm, "delta": d}
        print(f"    {ratio:.1f}x ({L}->{tgt}): SG={sg:.2f}  FCM={fcm:.2f}  ({d:+.1f}%)")

    result["compression_sweep"] = sweep
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=LARGE_MODELS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--lora_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/exp17")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_kw = dict(num_steps=args.lora_steps, lr=2e-4, lora_r=args.lora_r, num_samples=2048)

    all_results = {}
    results_path = os.path.join(args.output_dir, "large_models.json")

    for model_name in args.models:
        t0 = time.time()
        try:
            res = process_model(model_name, args.device, args.output_dir, lora_kw)
            all_results[model_name] = res
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

        elapsed = time.time() - t0
        print(f"\n  Time: {elapsed/60:.1f} min")

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*80}")
    print("LARGE MODEL RESULTS")
    print(f"{'='*80}")
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name}: ERROR")
            continue
        short = name.split("/")[-1]
        print(f"\n  {short} ({res['L']} layers):")
        print(f"    CKA ratio: {res['cka']['ratio']:.1f}x")
        print(f"    Baseline: {res['baseline_ppl']:.2f}")
        print(f"    ShortGPT: {res['shortgpt']['mean']:.2f} +/- {res['shortgpt']['std']:.2f}")
        print(f"    FCM:      {res['fcm_hybrid']['mean']:.2f} +/- {res['fcm_hybrid']['std']:.2f}")
        print(f"    Delta:    {res['delta_pct']:+.1f}%  ({res['winner']})")


if __name__ == "__main__":
    main()
