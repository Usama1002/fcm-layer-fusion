# A100 Experiment Plan — Complete Briefing

## Context: What This Project Is

This is the codebase for a NeurIPS 2026 paper: "Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models."

**The method (FCM)** compresses LLMs by:
1. Computing pairwise cosine similarity between all layer activations on a small calibration set
2. Running spectral clustering on the similarity matrix to group functionally similar layers (including non-adjacent ones)
3. Within each community, keeping only the layer with the highest Block Influence (BI) score
4. Applying LoRA recovery fine-tuning to heal the damage

**Current results**: FCM outperforms ShortGPT (the main baseline) on 10/11 models at 2x compression, with improvements up to 50%. Tested on 11 models from 6 architecture families (1.6B to 8B parameters), all with 3-seed variance.

**What these A100 experiments add**: The paper's main weakness is that all models tested are ≤8B. Testing on 14B models proves the method scales. The 5-seed variance experiments strengthen statistical claims on key models.

---

## Server Specs

- 2x NVIDIA A100 80GB PCIe
- ~40GB free VRAM per GPU (the other 40GB is in use by other processes)
- CUDA 13.0, Driver 580.105.08

**IMPORTANT**: Since only ~40GB is free per GPU, 14B models in fp16 (~28GB) should fit, but may be tight with LoRA training overhead. If OOM occurs, reduce `--lora_samples` from 2048 to 1024 or use `batch_size=1` in the profiler.

---

## Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/Usama1002/fcm-layer-fusion.git
cd fcm-layer-fusion

# 2. Create conda environment
conda create -n fcm python=3.11 -y
conda activate fcm

# 3. Install PyTorch with CUDA support
# Check CUDA version first:
nvidia-smi | grep "CUDA Version"
# For CUDA 13.0 / 12.x compatibility:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install all dependencies
pip install -r requirements.txt
# requirements.txt contains: torch, transformers, datasets, accelerate, peft,
# lm-eval, matplotlib, seaborn, numpy, scipy, leidenalg, igraph, tqdm

# 5. Verify everything works
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free = torch.cuda.mem_get_info(i)[0] / 1e9
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {props.total_memory/1e9:.0f}GB total, {free:.0f}GB free')
"

# 6. Verify the codebase imports work
python -c "from src.fcm import fcm_compress; print('FCM imports OK')"
python -c "from src.baselines import compute_block_influence; print('Baselines OK')"
python -c "from src.profiler import collect_activations; print('Profiler OK')"
```

### Troubleshooting Setup

- **If `leidenalg` fails to install**: `pip install leidenalg` requires `igraph`. Try `pip install python-igraph leidenalg`.
- **If `transformers` can't download models**: You may need `huggingface-cli login` or to set `HF_TOKEN`.
- **If CUDA version mismatch**: Try `pip install torch --index-url https://download.pytorch.org/whl/cu121` instead.
- **If peft version conflict**: `pip install peft>=0.11` should work.

---

## Experiment A: Large Models (14B) — HIGHEST PRIORITY

### What and Why
Tests FCM on 14B-parameter models (Qwen2.5-14B, Qwen3-14B) that have 40 layers. These are the largest open-weight dense models that fit on a single A100 in fp16. The paper currently only tests up to 8B — showing FCM works at 14B is a strong scaling result.

### How to Run

```bash
conda activate fcm

# Run on GPU 0 (one model at a time to avoid OOM)
python experiments/17_large_models.py --device cuda:0
```

### What It Does (step by step)
For each 14B model:
1. Profiles activations (512 calibration samples from C4, seq_len=128, batch_size=2)
2. Computes cosine and CKA similarity matrices
3. Saves CKA heatmap visualizations
4. Computes Block Influence scores
5. Runs 3-seed comparison: ShortGPT vs FCM-Hybrid at 2x compression
6. Runs compression ratio sweep: 1.5x, 2x, 2.5x, 3x
7. Saves everything to `results/exp17/large_models.json`

### Expected Runtime
- ~90 min per model (profiling ~10 min, 3 seeds × 2 methods × ~8 min each, sweep ~20 min)
- Total: ~3 hours for both models

### What to Expect
- Qwen2.5-14B: Likely FCM wins (Qwen family has been consistently strong for FCM)
- Qwen3-14B: Likely FCM wins (Qwen3 family all won in our 8B and below tests)
- CKA ratio should be 3-6x (consistent with smaller Qwen models)

### If Something Goes Wrong

**OOM during profiling**: Edit line in `17_large_models.py`:
```python
acts = collect_activations(model_name, num_samples=512, max_seq_len=128,
                           batch_size=2, device=device)
```
Change `batch_size=2` to `batch_size=1`, or reduce `num_samples=256`.

**OOM during LoRA recovery**: The `lora_kw` dict uses `num_samples=2048`. Reduce to 1024:
```python
lora_kw = dict(num_steps=args.lora_steps, lr=2e-4, lora_r=args.lora_r, num_samples=1024)
```

**Model download fails**: Qwen2.5-14B and Qwen3-14B are open-weight and should download automatically. If HF gating, run `huggingface-cli login` first.

**The script saves incrementally**: If it crashes after the first model, the results are already saved. Just re-run and it will skip completed models (check `results/exp17/large_models.json`).

### Output Files to Bring Back
```
results/exp17/large_models.json    # Main results (CRITICAL)
results/exp17/cka_Qwen2.5-14B.png # CKA heatmap visualization
results/exp17/cka_Qwen3-14B.png   # CKA heatmap visualization
results/exp17/offdiag_Qwen2.5-14B.png
results/exp17/offdiag_Qwen3-14B.png
results/exp17/cosine_Qwen2.5-14B.npy  # Similarity matrix (optional, large)
results/exp17/cka_Qwen2.5-14B.npy     # CKA matrix (optional, large)
```

---

## Experiment B: 5-Seed Variance — HIGH PRIORITY

### What and Why
The paper currently reports 3-seed variance (seeds 42, 123, 7). Upgrading to 5 seeds (adding 2024 and 999) on the 4 most important models strengthens statistical claims. This is especially important for:
- Qwen2.5-7B: our flagship model
- Qwen3-4B: our biggest win (-50%)
- LLaMA-3.1-8B: previously catastrophic failure, now a win (-18%)
- Mistral-7B: our only loss (+13.5%)

### How to Run

```bash
conda activate fcm

# Run on GPU 1 (can run in parallel with Exp A on GPU 0)
python experiments/18_five_seeds.py --device cuda:1
```

### What It Does
For each model:
1. Loads the pre-computed similarity matrix from `results/exp12/` (if not available, computes fresh)
2. Computes BI scores once
3. Runs 5 seeds × 2 methods (ShortGPT + FCM-Hybrid) = 10 runs per model
4. Reports mean ± std with 5 seeds
5. Saves to `results/exp18/five_seeds.json`

### Expected Runtime
- ~60 min per model (5 seeds × 2 methods × ~6 min each)
- Total: ~4 hours for 4 models

### IMPORTANT: Similarity matrices
The script looks for pre-computed similarity matrices in `results/exp12/`. If this directory doesn't exist (because you cloned a fresh repo), the script will compute them from scratch. This adds ~10 min per 7B model.

If you want to avoid this, you can pre-compute them:
```python
# Quick way to compute similarity matrices for the 4 models
from src.profiler import collect_activations
from src.cka import compute_similarity_matrix
import numpy as np, os

os.makedirs("results/exp12", exist_ok=True)
for model, metric in [("Qwen/Qwen2.5-7B", "cosine"), ("Qwen/Qwen3-4B", "cka"),
                       ("meta-llama/Llama-3.1-8B", "cosine"), ("mistralai/Mistral-7B-v0.3", "cosine")]:
    short = model.split("/")[-1]
    acts = collect_activations(model, num_samples=512, max_seq_len=128, batch_size=4, device="cuda:1")
    sim = compute_similarity_matrix(acts, metric=metric)
    np.save(f"results/exp12/{metric}_{short}.npy", sim)
    del acts
    import torch; torch.cuda.empty_cache()
```

### What to Expect
- Qwen2.5-7B: FCM should still win (~-35 to -45%), std should decrease with more seeds
- Qwen3-4B: FCM should still win (~-45 to -55%)
- LLaMA-3.1-8B: FCM should still win (~-15 to -20%)
- Mistral-7B: ShortGPT should still win (~+10 to +15%)

### Output Files to Bring Back
```
results/exp18/five_seeds.json   # Main results (CRITICAL)
```

---

## Running Both Experiments in Parallel

Open two terminal windows/tmux panes:

**Terminal 1 (GPU 0):**
```bash
cd fcm-layer-fusion
conda activate fcm
CUDA_VISIBLE_DEVICES=0 python experiments/17_large_models.py --device cuda:0
```

**Terminal 2 (GPU 1):**
```bash
cd fcm-layer-fusion
conda activate fcm
CUDA_VISIBLE_DEVICES=1 python experiments/18_five_seeds.py --device cuda:1
```

Or use tmux:
```bash
tmux new-session -d -s exp17 'cd fcm-layer-fusion && conda activate fcm && python experiments/17_large_models.py --device cuda:0'
tmux new-session -d -s exp18 'cd fcm-layer-fusion && conda activate fcm && python experiments/18_five_seeds.py --device cuda:1'
tmux ls  # verify both are running
```

---

## How the Code Works (for debugging)

### Key Source Files
```
src/fcm.py       — Clean API: fcm_compress() does the full pipeline
src/profiler.py  — Collects activations from model layers via hooks
src/cka.py       — Computes CKA and cosine similarity matrices
src/community.py — Spectral clustering and Leiden community detection
src/baselines.py — ShortGPT (Block Influence computation + layer removal)
src/recovery.py  — LoRA fine-tuning with peft
src/evaluate.py  — WikiText-2 perplexity evaluation
src/fusion.py    — Layer merging utilities
```

### The FCM-Hybrid Method (what the experiments implement)
```python
# 1. Profile activations
acts = collect_activations(model_name, num_samples=512, ...)
sim = compute_similarity_matrix(acts, metric="cosine")

# 2. Community detection
comms = detect_communities(sim, method="spectral", n_clusters=target)

# 3. BI-guided selection (highest-BI layer per community)
bi = compute_block_influence(model, tokenizer, ...)
keep = [max(comm, key=lambda i: bi[i]) for comm in comms]

# 4. Build compressed model
layers = model.model.layers
model.model.layers = nn.ModuleList([layers[i] for i in sorted(keep)])
model.config.num_hidden_layers = len(keep)
model.config.use_cache = False

# 5. LoRA recovery
model = lora_recovery_finetune(model, tokenizer, num_steps=500, lr=2e-4, lora_r=16, ...)

# 6. Evaluate
ppl = evaluate_perplexity(model, tokenizer, max_seq_len=2048, max_samples=50)
```

### The ShortGPT Baseline
```python
bi = compute_block_influence(model, tokenizer, ...)
# Remove L-K layers with lowest BI scores
model, kept = remove_layers_by_bi(model, bi, num_to_remove=L-target)
# Same LoRA recovery as FCM
model = lora_recovery_finetune(model, tokenizer, ...)
ppl = evaluate_perplexity(model, tokenizer, ...)
```

### Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| KV cache mismatch | `IndexError: list index out of range` in `cache_utils.py` | Already handled: `model.config.use_cache = False` |
| RoPE position embeddings | `TypeError: cannot unpack non-iterable NoneType` | Already handled in `src/fusion.py` |
| Model needs auth | `401 Unauthorized` on download | Run `huggingface-cli login` |
| OOM during profiling | `CUDA out of memory` during activation collection | Reduce `batch_size` or `num_samples` |
| OOM during LoRA | `CUDA out of memory` during fine-tuning | Reduce `num_samples` in `lora_kw` from 2048 to 1024 |
| NaN perplexity | PPL shows `nan` or `inf` | Likely OOM caused silent corruption; reduce batch size |
| New model architecture | `ValueError: Unsupported architecture` | Check if model has `model.model.layers`; if not, add to `get_layer_hook_points()` in `src/profiler.py` |

---

## What Results We Need (Summary)

### From Experiment A (large_models.json):
```json
{
  "Qwen/Qwen2.5-14B": {
    "L": 40, "target": 20, "baseline_ppl": ...,
    "cka": {"adj": ..., "nonadj": ..., "ratio": ...},
    "shortgpt": {"mean": ..., "std": ..., "runs": [...]},
    "fcm_hybrid": {"mean": ..., "std": ..., "runs": [...]},
    "delta_pct": ..., "winner": "FCM" or "ShortGPT",
    "compression_sweep": { "1.5x": {...}, "2.0x": {...}, ... }
  }
}
```

### From Experiment B (five_seeds.json):
```json
{
  "Qwen/Qwen2.5-7B": {
    "baseline": ...,
    "shortgpt": {"mean": ..., "std": ..., "runs": [5 values]},
    "fcm_hybrid": {"mean": ..., "std": ..., "runs": [5 values]},
    "delta_pct": ..., "winner": ...
  }
}
```

These JSON files are the critical deliverables. The PNG files (CKA heatmaps) are nice-to-have for the paper figures.
