"""
Experiment 09: Fast PPL-Only Sweep Across All Models and Methods

Skips zero-shot eval (the bottleneck) to rapidly collect perplexity
results across the full experimental matrix. Zero-shot will be run
separately on the winning configurations.

Matrix: 5 models x (ShortGPT + 6 FCM variants) = 35 experiments
Each takes ~5-10 min = ~3-6 hours total
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
from src.cka import compute_similarity_matrix
from src.community import detect_communities, analyze_communities
from src.merge import merge_community_select, merge_community_average, build_compressed_model
from src.fusion import get_all_layers, get_layer_modules
from src.baselines import compute_block_influence, remove_layers_by_bi
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.ablations import fisher_segmentation, uniform_grouping
from src.visualize import plot_cka_matrix, plot_cka_off_diagonal_analysis

MODELS = [
    "Qwen/Qwen2.5-7B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--lora_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--lora_samples", type=int, default=2048)
    parser.add_argument("--cka_samples", type=int, default=512)
    parser.add_argument("--ppl_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/exp09")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_kw = dict(num_steps=args.lora_steps, lr=args.lora_lr,
                   lora_r=args.lora_r, num_samples=args.lora_samples)

    all_results = {}
    results_path = os.path.join(args.output_dir, "fast_sweep.json")

    for model_name in args.models:
        short = model_name.split("/")[-1]
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        L = cfg.num_hidden_layers
        target = max(L // 2, 4)
        t0_model = time.time()

        print(f"\n{'#'*70}")
        print(f"# {model_name} ({L} -> {target} layers, 2x compression)")
        print(f"{'#'*70}")

        model_res = {"model": model_name, "L": L, "target": target}

        # CKA matrix
        cka_path = os.path.join(args.output_dir, f"cka_{short}.npy")
        cos_path = os.path.join(args.output_dir, f"cosine_{short}.npy")
        if os.path.exists(cka_path):
            cka = np.load(cka_path)
            cos = np.load(cos_path) if os.path.exists(cos_path) else None
            print(f"  Loaded cached similarity matrices")
        else:
            print(f"  Computing similarity matrices...")
            acts = collect_activations(model_name, num_samples=args.cka_samples,
                                       max_seq_len=128, batch_size=4, device=args.device)
            cka = compute_similarity_matrix(acts, metric="cka")
            cos = compute_similarity_matrix(acts, metric="cosine")
            np.save(cka_path, cka); np.save(cos_path, cos)
            del acts; torch.cuda.empty_cache()

        # CKA analysis
        offdiag = plot_cka_off_diagonal_analysis(
            cka, save_path=os.path.join(args.output_dir, f"offdiag_{short}.png"))
        plot_cka_matrix(cka, title=f"CKA - {short}",
                        save_path=os.path.join(args.output_dir, f"cka_{short}.png"))
        adj = offdiag["num_adjacent_high"]
        nonadj = offdiag["num_nonadjacent_high"]
        model_res["cka"] = {"adj": int(adj), "nonadj": int(nonadj),
                            "ratio": nonadj / max(1, adj)}

        # Baseline PPL
        print(f"  [Baseline]")
        m, t = load_fresh(model_name, args.device)
        bl_ppl = eval_ppl(m, t, args.device, args.ppl_n)
        model_res["baseline_ppl"] = bl_ppl
        print(f"    PPL: {bl_ppl:.2f}")
        free(m)

        # All methods to test
        methods = {}

        # ShortGPT
        methods["shortgpt"] = {"type": "shortgpt"}

        # FCM: grouping x merge
        for grp in ["spectral", "fisher", "uniform"]:
            for mrg in ["select", "average"]:
                methods[f"fcm_{grp}_{mrg}"] = {"type": "fcm", "grouping": grp,
                                                "merge": mrg, "sim": "cka"}

        # Cosine similarity ablation
        if cos is not None:
            for mrg in ["select", "average"]:
                methods[f"fcm_spectral_{mrg}_cosine"] = {"type": "fcm", "grouping": "spectral",
                                                          "merge": mrg, "sim": "cosine"}

        results = {}
        for key, cfg_m in methods.items():
            print(f"  [{key}]")
            t0 = time.time()
            try:
                m, t = load_fresh(model_name, args.device)

                if cfg_m["type"] == "shortgpt":
                    bi = compute_block_influence(m, t, device=args.device)
                    m, kept = remove_layers_by_bi(m, bi, L - target)
                    n_layers = len(get_all_layers(m))
                else:
                    sim_matrix = cka if cfg_m["sim"] == "cka" else cos
                    grp = cfg_m["grouping"]
                    mrg = cfg_m["merge"]

                    if grp == "spectral":
                        comms = detect_communities(sim_matrix, method="spectral", n_clusters=target)
                    elif grp == "fisher":
                        comms = fisher_segmentation(sim_matrix, n_segments=target)
                    elif grp == "uniform":
                        comms = uniform_grouping(L, target)

                    merged = []
                    for comm in comms:
                        if mrg == "select":
                            merged.append(merge_community_select(m, comm, sim_matrix))
                        elif mrg == "average":
                            merged.append(merge_community_average(m, comm, sim_matrix))

                    m = build_compressed_model(m, comms, merged)
                    n_layers = len(get_all_layers(m))

                ppl_raw = eval_ppl(m, t, args.device, args.ppl_n)

                # LoRA recovery
                m, _ = lora_recovery_finetune(m, t, device=args.device, **lora_kw)
                ppl_lora = eval_ppl(m, t, args.device, args.ppl_n)

                elapsed = time.time() - t0
                results[key] = {"ppl_raw": ppl_raw, "ppl_lora": ppl_lora,
                               "layers": n_layers, "time": elapsed}
                print(f"    PPL: raw={ppl_raw:.2f}, +LoRA={ppl_lora:.2f} ({elapsed:.0f}s)")
                free(m)

            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                results[key] = {"error": str(e)}
                try: free(m)
                except: pass

        model_res["methods"] = results
        model_res["total_time"] = time.time() - t0_model
        all_results[model_name] = model_res

        # Save after each model
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Model done in {model_res['total_time']/60:.1f} min, saved.")

    # Final summary
    print(f"\n{'='*100}")
    print("FAST SWEEP RESULTS (PPL only, 2x compression)")
    print(f"{'='*100}")
    print(f"{'Model':<22} {'Method':<28} {'Layers':>6} {'PPL raw':>10} {'PPL+LoRA':>10}")
    print("-" * 80)
    for name, res in all_results.items():
        short = name.split('/')[-1][:20]
        print(f"{short:<22} {'Baseline':<28} {res['L']:>6} {res['baseline_ppl']:>10.2f} {'--':>10}")
        for key in sorted(res.get("methods", {}).keys()):
            r = res["methods"][key]
            if "error" in r:
                print(f"{'':22} {key:<28} {'':>6} {'ERROR':>10} {'':>10}")
            else:
                print(f"{'':22} {key:<28} {r['layers']:>6} {r['ppl_raw']:>10.2f} {r['ppl_lora']:>10.2f}")
        print("-" * 80)
    print(f"{'='*100}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
