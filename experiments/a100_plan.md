# A100 Experiment Plan

## Server: 2x NVIDIA A100 80GB PCIe (~40GB free per GPU)

## Goal
Run experiments that address the top reviewer objections and significantly strengthen the paper.

## Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/Usama1002/fcm-layer-fusion.git
cd fcm-layer-fusion

# 2. Create conda environment
conda create -n fcm python=3.11 -y
conda activate fcm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 3. Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.0f}GB') for i in range(torch.cuda.device_count())]"
```

## Experiments (in priority order)

### Experiment A: Large Models (14B-27B) — HIGHEST PRIORITY
Tests FCM on models too large for our current GPU, proving the method scales.

**Models:**
- `Qwen/Qwen2.5-14B` (40 layers, ~28GB fp16) — GPU 0
- `Qwen/Qwen3-14B` (40 layers, ~28GB fp16) — GPU 1
- `meta-llama/Llama-3.1-70B` — too large even for 2x A100 in fp16, skip

**Run:** `python experiments/17_large_models.py --device cuda:0` (one model at a time)

### Experiment B: 5-Seed Variance on Key Models
Upgrades from 3-seed to 5-seed on the 4 most important models.

**Models:** Qwen2.5-7B, Qwen3-4B, LLaMA-3.1-8B, Mistral-7B
**Run:** `python experiments/18_five_seeds.py --device cuda:1`

### Experiment C: Zero-Shot with Hybrid Method
Re-runs zero-shot benchmarks using the final FCM-Hybrid (BI-guided) method on all 11 models.

**Run:** `python experiments/16_zero_shot.py --device cuda:0` (already exists, just run it on A100)

### Experiment D: Additional Compression Ratios on Large Models
Tests 1.5x, 2x, 2.5x, 3x compression on the 14B models.

**Run:** Included in `experiments/17_large_models.py`

## Time Estimates

| Experiment | Models | Seeds | Methods | Est. Time | GPU |
|-----------|--------|-------|---------|-----------|-----|
| A: Large models | 2 | 3 | 2 (SG+FCM) | ~3 hours | cuda:0 |
| B: 5-seed variance | 4 | 5 | 2 | ~4 hours | cuda:1 |
| C: Zero-shot (11 models) | 11 | 1 | 3 (base+SG+FCM) | ~8 hours | cuda:0 (after A) |

Experiments A and B can run in parallel on the two GPUs.

## Output Files
All results save to `results/` as JSON. Bring back:
- `results/exp17/large_models.json`
- `results/exp18/five_seeds.json`
- `results/exp16/zero_shot.json` (if re-run)
