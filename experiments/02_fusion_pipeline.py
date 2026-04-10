"""
Experiment 02: Full Fusion Pipeline

End-to-end test of the FCM (Fusion via Community Mining) pipeline:
1. Load model and compute CKA matrix (or load from Exp 00)
2. Run community detection with spectral clustering (k communities)
3. Distill each community into a single representative layer
4. Evaluate perplexity before and after fusion

Usage:
    python experiments/02_fusion_pipeline.py --model Qwen/Qwen2.5-7B --n_clusters 8
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cka import compute_cka_matrix
from src.profiler import collect_activations
from src.community import detect_communities, analyze_communities, print_communities
from src.fusion import distill_community, build_fused_model, get_all_layers
from src.evaluate import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser(description="FCM Fusion Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of fusion groups")
    parser.add_argument("--method", type=str, default="spectral", choices=["spectral", "leiden", "louvain"])
    parser.add_argument("--resolution", type=float, default=1.0, help="Resolution for Leiden")
    parser.add_argument("--distill_steps", type=int, default=200, help="Distillation steps per community")
    parser.add_argument("--distill_lr", type=float, default=1e-4)
    parser.add_argument("--num_cal_samples", type=int, default=256, help="Calibration samples")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Max seq len for calibration")
    parser.add_argument("--ppl_max_samples", type=int, default=50, help="Max chunks for perplexity eval")
    parser.add_argument("--output_dir", type=str, default="results/exp02")
    parser.add_argument("--cka_matrix_path", type=str, default="results/exp00/cka_matrix_Qwen2.5-7B.npy")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]

    print(f"{'='*60}")
    print(f"Experiment 02: FCM Fusion Pipeline")
    print(f"Model: {args.model}")
    print(f"Clusters: {args.n_clusters}, Method: {args.method}")
    print(f"{'='*60}")

    # Step 1: Load or compute CKA matrix
    if os.path.exists(args.cka_matrix_path):
        print(f"\nLoading CKA matrix from {args.cka_matrix_path}")
        cka_matrix = np.load(args.cka_matrix_path)
    else:
        print(f"\nComputing CKA matrix...")
        activations = collect_activations(
            model_name=args.model,
            num_samples=args.num_cal_samples,
            max_seq_len=args.max_seq_len,
            batch_size=4,
            device=args.device,
        )
        cka_matrix = compute_cka_matrix(activations)
        np.save(args.cka_matrix_path, cka_matrix)
        del activations

    L = cka_matrix.shape[0]

    # Step 2: Community detection
    print(f"\nRunning community detection ({args.method})...")
    kwargs = {"method": args.method}
    if args.method == "spectral":
        kwargs["n_clusters"] = args.n_clusters
    elif args.method == "leiden":
        kwargs["resolution"] = args.resolution

    communities = detect_communities(cka_matrix, **kwargs)
    stats = analyze_communities(communities, L)
    print_communities(communities, stats)

    # Step 3: Load model for distillation and evaluation
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )

    # Step 4: Evaluate baseline perplexity
    print(f"\nEvaluating baseline perplexity...")
    baseline_ppl = evaluate_perplexity(
        model, tokenizer,
        max_seq_len=2048,
        device=args.device,
        max_samples=args.ppl_max_samples,
    )
    print(f"Baseline perplexity: {baseline_ppl:.2f}")

    # Step 5: Prepare calibration data for distillation
    print(f"\nPreparing calibration data...")
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for sample in ds:
        text = sample.get("text", "")
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= args.num_cal_samples:
            break

    cal_encodings = tokenizer(
        texts,
        max_length=args.max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Step 6: Distill each community
    print(f"\nDistilling {len(communities)} communities...")
    representative_layers = []
    distill_losses = []

    for i, comm in enumerate(communities):
        print(f"\nCommunity {i}: layers {comm} (size={len(comm)})")
        rep_layer, loss = distill_community(
            model=model,
            community=comm,
            cka_matrix=cka_matrix,
            calibration_inputs=cal_encodings,
            num_steps=args.distill_steps,
            lr=args.distill_lr,
            device=args.device,
        )
        representative_layers.append(rep_layer)
        distill_losses.append(loss)
        print(f"  Final distillation loss: {loss:.6f}")

    # Step 7: Build fused model
    print(f"\nBuilding fused model...")
    original_num_layers = L
    model = build_fused_model(model, communities, representative_layers)
    new_num_layers = len(get_all_layers(model))
    print(f"Layers: {original_num_layers} -> {new_num_layers} ({original_num_layers/new_num_layers:.2f}x compression)")

    # Step 8: Evaluate fused model perplexity
    print(f"\nEvaluating fused model perplexity...")
    fused_ppl = evaluate_perplexity(
        model, tokenizer,
        max_seq_len=2048,
        device=args.device,
        max_samples=args.ppl_max_samples,
    )
    print(f"Fused perplexity: {fused_ppl:.2f}")

    # Results summary
    ppl_increase = ((fused_ppl - baseline_ppl) / baseline_ppl) * 100

    results = {
        "model": args.model,
        "method": args.method,
        "n_clusters": args.n_clusters,
        "original_layers": original_num_layers,
        "fused_layers": new_num_layers,
        "compression_ratio": original_num_layers / new_num_layers,
        "baseline_perplexity": baseline_ppl,
        "fused_perplexity": fused_ppl,
        "perplexity_increase_pct": ppl_increase,
        "distill_losses": distill_losses,
        "communities": communities,
    }

    results_path = os.path.join(args.output_dir, f"results_{model_short}_k{args.n_clusters}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Layers: {original_num_layers} -> {new_num_layers} ({results['compression_ratio']:.2f}x)")
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Fused PPL: {fused_ppl:.2f}")
    print(f"PPL increase: {ppl_increase:+.1f}%")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
