"""
Experiment 14: Diagnose why FCM fails catastrophically on LLaMA

Investigates:
1. What communities does spectral clustering form on LLaMA vs Qwen?
2. Which layers are selected vs removed?
3. Does a different K (not L/2) help?
4. Does Leiden (non-contiguous) help vs spectral?
5. What if we use BI-weighted selection instead of centrality?
"""

import gc
import json
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cka import compute_similarity_matrix
from src.community import detect_communities, analyze_communities
from src.merge import merge_community_select, build_compressed_model
from src.fusion import get_all_layers, get_layer_modules
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity

DEVICE = "cuda"

def load_fresh(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                              device_map=DEVICE, trust_remote_code=True)
    return m, tok

def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

def set_seed(s=42):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def diagnose_model(model_name, cos_matrix, cka_matrix, output_dir):
    """Deep diagnosis of compression behavior."""
    short = model_name.split("/")[-1]
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    L = cfg.num_hidden_layers
    target = L // 2

    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {short} ({L} layers)")
    print(f"{'='*70}")

    results = {"model": model_name, "L": L, "target": target}

    # Baseline
    m, t = load_fresh(model_name)
    bl = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    print(f"Baseline PPL: {bl:.2f}")
    results["baseline"] = bl

    # Get BI scores
    bi = compute_block_influence(m, t, device=DEVICE)
    results["bi_scores"] = bi.tolist()
    free(m)

    # ShortGPT: which layers does it keep?
    sg_keep = sorted(np.argsort(bi)[::-1][:target].tolist())
    sg_remove = [i for i in range(L) if i not in sg_keep]
    print(f"\nShortGPT keeps:   {sg_keep}")
    print(f"ShortGPT removes: {sg_remove}")
    results["shortgpt_keep"] = sg_keep
    results["shortgpt_remove"] = sg_remove

    # FCM-cosine-spectral: which layers does it keep?
    comms_cos = detect_communities(cos_matrix, method="spectral", n_clusters=target)
    from src.fusion import find_central_layer
    fcm_cos_keep = sorted([find_central_layer(cos_matrix, c) for c in comms_cos])
    fcm_cos_remove = [i for i in range(L) if i not in fcm_cos_keep]
    print(f"\nFCM-cos keeps:    {fcm_cos_keep}")
    print(f"FCM-cos removes:  {fcm_cos_remove}")
    results["fcm_cos_keep"] = fcm_cos_keep
    results["fcm_cos_remove"] = fcm_cos_remove

    # FCM-CKA-spectral
    comms_cka = detect_communities(cka_matrix, method="spectral", n_clusters=target)
    fcm_cka_keep = sorted([find_central_layer(cka_matrix, c) for c in comms_cka])
    print(f"FCM-cka keeps:    {fcm_cka_keep}")
    results["fcm_cka_keep"] = fcm_cka_keep

    # Overlap analysis
    sg_set = set(sg_keep)
    cos_set = set(fcm_cos_keep)
    cka_set = set(fcm_cka_keep)
    overlap_sg_cos = len(sg_set & cos_set)
    overlap_sg_cka = len(sg_set & cka_set)
    print(f"\nOverlap SG & FCM-cos: {overlap_sg_cos}/{target} ({100*overlap_sg_cos/target:.0f}%)")
    print(f"Overlap SG & FCM-cka: {overlap_sg_cka}/{target} ({100*overlap_sg_cka/target:.0f}%)")
    results["overlap_sg_cos"] = overlap_sg_cos
    results["overlap_sg_cka"] = overlap_sg_cka

    # Critical question: does FCM remove critical early/late layers?
    print(f"\nFCM-cos removes from first 3: {[i for i in fcm_cos_remove if i < 3]}")
    print(f"FCM-cos removes from last 3:  {[i for i in fcm_cos_remove if i >= L-3]}")
    print(f"ShortGPT removes from first 3: {[i for i in sg_remove if i < 3]}")
    print(f"ShortGPT removes from last 3:  {[i for i in sg_remove if i >= L-3]}")

    # Strategy: FCM-select but use BI to choose which layer to keep per community
    # (instead of centrality, use the highest-BI layer in each community)
    import torch.nn as nn
    print(f"\n--- FCM with BI-guided selection ---")
    set_seed(42)
    bi_keep = []
    for comm in comms_cos:
        if len(comm) == 1:
            bi_keep.append(comm[0])
        else:
            best_bi_idx = max(comm, key=lambda i: bi[i])
            bi_keep.append(best_bi_idx)
    bi_keep = sorted(bi_keep)
    print(f"FCM-BI keeps: {bi_keep}")
    results["fcm_bi_keep"] = bi_keep

    m, t = load_fresh(model_name)
    layers = get_all_layers(m)
    new_layers = nn.ModuleList([layers[i] for i in bi_keep])
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        m.model.layers = new_layers
        m.config.num_hidden_layers = len(new_layers)
    m.config.use_cache = False
    if hasattr(m, "generation_config"):
        m.generation_config.cache_implementation = None

    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    set_seed(42)
    m, _ = lora_recovery_finetune(m, t, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
    ppl_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    print(f"FCM-BI: raw={ppl_raw:.2f}, +LoRA={ppl_lora:.2f}")
    results["fcm_bi"] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora}
    free(m)

    # Strategy: Use ShortGPT's layer selection but with FCM's community constraint
    # i.e., ensure at least one layer per community is kept
    print(f"\n--- Hybrid: community-constrained BI selection ---")
    set_seed(42)
    hybrid_keep = []
    for comm in comms_cos:
        if len(comm) == 1:
            hybrid_keep.append(comm[0])
        else:
            # Keep the highest-BI layer from this community
            best = max(comm, key=lambda i: bi[i])
            hybrid_keep.append(best)
    hybrid_keep = sorted(hybrid_keep)
    print(f"Hybrid keeps: {hybrid_keep}")

    m, t = load_fresh(model_name)
    layers = get_all_layers(m)
    new_layers = nn.ModuleList([layers[i] for i in hybrid_keep])
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        m.model.layers = new_layers
        m.config.num_hidden_layers = len(new_layers)
    m.config.use_cache = False
    if hasattr(m, "generation_config"):
        m.generation_config.cache_implementation = None

    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    set_seed(42)
    m, _ = lora_recovery_finetune(m, t, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
    ppl_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    print(f"Hybrid: raw={ppl_raw:.2f}, +LoRA={ppl_lora:.2f}")
    results["hybrid"] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora}
    free(m)

    # ShortGPT reference
    print(f"\n--- ShortGPT reference ---")
    set_seed(42)
    m, t = load_fresh(model_name)
    m, _ = remove_layers_by_bi(m, bi, L - target)
    ppl_raw = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    set_seed(42)
    m, _ = lora_recovery_finetune(m, t, num_steps=500, lr=2e-4, lora_r=16, num_samples=2048, device=DEVICE)
    ppl_lora = evaluate_perplexity(m, t, max_seq_len=2048, device=DEVICE, max_samples=50)
    print(f"ShortGPT: raw={ppl_raw:.2f}, +LoRA={ppl_lora:.2f}")
    results["shortgpt"] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora}
    free(m)

    return results


def main():
    output_dir = "results/exp14"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model_name in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B",
                        "Qwen/Qwen2.5-7B"]:  # Qwen as control
        short = model_name.split("/")[-1]

        # Load similarity matrices
        cos_path = f"results/exp12/cosine_{short}.npy"
        cka_path = f"results/exp12/cka_{short}.npy"
        if not os.path.exists(cos_path):
            print(f"Missing {cos_path}, skipping")
            continue
        cos = np.load(cos_path)
        cka = np.load(cka_path)

        res = diagnose_model(model_name, cos, cka, output_dir)
        all_results[model_name] = res

        with open(os.path.join(output_dir, "diagnosis.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    for name, res in all_results.items():
        short = name.split("/")[-1]
        print(f"\n{short}:")
        print(f"  Baseline:  {res['baseline']:.2f}")
        print(f"  ShortGPT:  {res['shortgpt']['ppl_lora']:.2f}")
        print(f"  FCM-cos:   overlap={res['overlap_sg_cos']}/{res['target']}")
        if "fcm_bi" in res:
            print(f"  FCM-BI:    {res['fcm_bi']['ppl_lora']:.2f}")
        if "hybrid" in res:
            print(f"  Hybrid:    {res['hybrid']['ppl_lora']:.2f}")


if __name__ == "__main__":
    main()
