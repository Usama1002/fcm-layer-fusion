# Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models

**NeurIPS 2026 (Main Track, Poster)**

Muhammad Usama, Yeoreum Jung (Braindeck Inc, Seoul, Republic of Korea)

This repository is the official code release for our NeurIPS 2026 paper.

**Fusion via Community Mining (FCM)** compresses large language models by removing redundant transformer layers. Unlike prior methods that treat layers independently or only merge adjacent or contiguous layers, FCM builds a full pairwise activation similarity graph over all layers, applies spectral community detection to discover clusters of functionally equivalent layers (including non-adjacent ones), and retains the single highest-Block-Influence layer per cluster. A lightweight LoRA recovery step restores performance.

Across thirteen models from six architecture families (1.6B to 14B parameters), non-adjacent high-similarity layer pairs outnumber adjacent ones by 3.6x to 15.9x. At 2x depth compression, FCM lowers mean perplexity below ShortGPT on 12 of 13 models (up to 50% on Qwen3-4B), significant on 10 under a two-sided Welch's t-test with Holm-Bonferroni correction; 2 are statistical ties and Mistral-7B is the one loss. Against ShortGPT, FlattenGPT, SWM, and DeltaLLM on all 13 models, FCM has the best perplexity on 10 of 13 (suite-wide mean rank 1.15). The gains carry over to zero-shot reasoning, MMLU, LAMBADA, HumanEval, and free-form generation.

## Key Results

All numbers below match the NeurIPS 2026 camera-ready paper. 2x compression (layers halved), LoRA r=16, 500 recovery steps, WikiText-2 perplexity (lower is better).

### FCM vs ShortGPT on 13 models

| Model | Family | Base PPL | ShortGPT | FCM (ours) | Delta | Sig. |
|---|---|---:|---:|---:|---:|:---:|
| **FCM significant wins (10/13)** | | | | | | |
| Qwen3-4B† | Qwen3 | 13.17 | 116.20 +/- 8.12 | **58.15 +/- 9.45** | -50.0% | ** |
| Qwen2.5-7B† | Qwen2 | 6.67 | 44.92 +/- 2.85 | **26.65 +/- 0.94** | -40.7% | ** |
| Qwen3-1.7B† | Qwen3 | 11.82 | 74.20 +/- 3.10 | **46.45 +/- 1.80** | -37.4% | * |
| StableLM-2-1.6B | StableLM | 12.45 | 52.60 +/- 2.40 | **33.24 +/- 1.10** | -36.8% | * |
| Qwen3-8B† | Qwen3 | 6.12 | 39.80 +/- 1.90 | **28.45 +/- 1.10** | -28.5% | * |
| Qwen2.5-3B | Qwen2 | 8.92 | 48.50 +/- 2.10 | **35.40 +/- 1.20** | -27.0% | * |
| Qwen2.5-14B | Qwen2 | 5.12 | 32.45 +/- 1.95 | **24.18 +/- 0.82** | -25.5% | * |
| Qwen3-14B | Qwen3 | 7.28 | 48.31 +/- 2.41 | **36.85 +/- 1.15** | -23.7% | * |
| Gemma-2-2B† | Gemma | 10.14 | 68.40 +/- 2.90 | **54.58 +/- 1.65** | -20.2% | * |
| LLaMA-3.1-8B† | LLaMA | 6.40 | 42.75 +/- 1.70 | **35.25 +/- 1.15** | -17.5% | * |
| **Statistical ties (2/13)** | | | | | | |
| SmolLM3-3B† | SmolLM | 9.45 | 41.80 +/- 1.65 | 39.85 +/- 1.30 | -4.7% | ns |
| LLaMA-3.2-3B† | LLaMA | 7.92 | 55.36 +/- 2.12 | 52.60 +/- 1.85 | -5.0% | ns |
| **ShortGPT win (1/13)** | | | | | | |
| Mistral-7B† | Mistral | 5.36 | **28.36 +/- 0.54** | 32.16 +/- 1.20 | +13.4% | -- |

`†` 5 seeds (42, 123, 7, 2024, 999); other rows 3 seeds (42, 123, 7). Significance: `**` p_corr < 0.001, `*` p_corr < 0.05, `ns` p_corr >= 0.05 (two-sided Welch's t-test, Holm-Bonferroni correction across the 12 rows with mean reduction). Within each row, all perplexities use one evaluation protocol. On Mistral-7B, full-token CKA narrows the gap to +5.3% (29.86 +/- 1.87).

### Head-to-head against recent baselines on 13 models

| Model | ShortGPT | FlattenGPT | SWM | DeltaLLM | FCM (ours) |
|---|---:|---:|---:|---:|---:|
| StableLM-2-1.6B | 52.60 | 48.90 | 51.20 | 44.10 | **33.24** |
| Qwen3-1.7B | 74.20 | 64.50 | 71.10 | 58.60 | **46.45** |
| Gemma-2-2B | 68.40 | 63.10 | 66.80 | 61.20 | **54.58** |
| Qwen2.5-3B | 48.50 | 44.20 | 46.80 | 42.10 | **35.40** |
| LLaMA-3.2-3B | 55.36 | 58.20 | 61.10 | 54.80 | **52.60** (tie) |
| SmolLM3-3B | 41.80 | 43.20 | 46.10 | 40.50 | **39.85** (tie) |
| Qwen3-4B | 116.20 | 92.30 | 105.40 | 84.50 | **58.15** |
| Qwen2.5-7B | 44.92 | 38.15 | 41.20 | 35.60 | **26.65** |
| Qwen3-8B | 39.80 | 35.40 | 38.10 | 33.20 | **28.45** |
| LLaMA-3.1-8B | 42.75 | 40.12 | 44.55 | 38.25 | **35.25** |
| Qwen2.5-14B | 32.45 | 29.80 | 31.40 | 27.95 | **24.18** |
| Qwen3-14B | 48.31 | 43.50 | 45.80 | 41.20 | **36.85** |
| Mistral-7B | **28.36** | 4962 | 471.5 | 32.76 | 32.16 |

Mean over 3 seeds, identical calibration (512 C4 samples, length 128) and identical 500-step LoRA recovery for every method. FCM's win over the strongest baseline is significant (p_corr < 0.05) on all 10 rows it wins. Suite-wide mean rank: FCM 1.15, DeltaLLM 2.15, FlattenGPT 3.38, SWM 4.08, ShortGPT 4.23.

### Compression sweep on Qwen2.5-14B

| Ratio | Layers | ShortGPT | FlattenGPT | DeltaLLM | FCM (ours) |
|---|---|---:|---:|---:|---:|
| 1.5x | 48 -> 32 | **8.42** | 8.95 | 8.55 | 8.68 |
| 2.0x | 48 -> 24 | 32.45 | 29.80 | 27.95 | **24.18** |
| 2.5x | 48 -> 19 | 118.50 | 94.20 | 82.30 | **54.60** |
| 3.0x | 48 -> 16 | 395.14 | 284.10 | 210.60 | **118.20** |

Below 2x a baseline can win; from 2x onward FCM leads and its margin grows with the ratio. The same crossover holds on Qwen2.5-7B and LLaMA-3.1-8B.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Compress a model in 5 lines

```python
from src.fcm import fcm_compress

result = fcm_compress("Qwen/Qwen2.5-7B", target_layers=14)
model = result["model"]          # compressed model (14 of 28 layers)
ppl = result["perplexity"]       # WikiText-2 perplexity
print(f"Compressed perplexity: {ppl:.2f}")
```

### Compress without recovery (fast, for inspection)

```python
from src.fcm import fcm_compress

result = fcm_compress(
    "Qwen/Qwen2.5-7B",
    target_layers=14,
    skip_recovery=True,
    skip_eval=True,
)
print(f"Kept layers: {result['keep_indices']}")
```

## Full Reproduction

Reproduce the paper's main results table (all 13 models, 3-5 seeds each):

```bash
python reproduce.py
```

Run a single model (faster):

```bash
python reproduce.py --models Qwen/Qwen2.5-7B
```

Run with a single seed (fastest):

```bash
python reproduce.py --models Qwen/Qwen2.5-7B --seeds 42
```

Results are saved incrementally to `results/reproduce/reproduce_results.json`.

## How FCM Works

1. **Profile**: Run calibration data through the model and record each layer's output activations.
2. **Similarity matrix**: Compute pairwise cosine similarity (or CKA) between all L layers, producing an L x L matrix. Cosine is the default; FCM falls back to CKA when the cosine partition is degenerate (normalized size-balance entropy below 0.70 or largest community covering more than 40% of layers), a rule computed from calibration data only.
3. **Community detection**: Apply spectral clustering on the similarity matrix to group layers into `target_layers` communities of functionally similar layers (which may be non-adjacent).
4. **BI-guided selection**: Within each community, keep the layer with the highest Block Influence score (the layer that changes the hidden state the most).
5. **Recovery**: Apply LoRA fine-tuning on C4 data to recover from the compression.

The key insight is that non-adjacent layers can be functionally similar (non-adjacent high-similarity pairs outnumber adjacent ones by 3.6x to 15.9x across all tested models), and community detection on the full similarity graph captures this structure.

## Repository Structure

```
fcm-layer-fusion/
  src/
    fcm.py          # Main FCM method
    cka.py          # CKA and cosine similarity computation
    community.py    # Graph community detection (spectral, Leiden)
    profiler.py     # Activation profiling with forward hooks
    baselines.py    # ShortGPT (Block Influence scores, layer removal)
    recovery.py     # LoRA recovery fine-tuning
    evaluate.py     # WikiText-2 perplexity evaluation
    benchmark.py    # Downstream benchmark evaluation
    ablations.py    # Grouping and selection ablations
    fusion.py       # Layer fusion utilities
    merge.py        # Weight-merging variants
    select.py       # Centrality-based selection (ablation)
    visualize.py    # Plotting utilities
  experiments/      # Experiment scripts (research log)
  results/          # Experiment outputs (JSON, numpy arrays)
  reproduce.py      # One-command reproduction script
  requirements.txt  # Python dependencies
```

## Models

FCM has been tested on thirteen models spanning six architecture families and four parameter scales:

**1-2B scale**: Qwen3-1.7B, Gemma-2-2B, StableLM-2-1.6B

**3-4B scale**: Qwen3-4B, SmolLM3-3B, LLaMA-3.2-3B, Qwen2.5-3B

**7-8B scale**: Qwen3-8B, Qwen2.5-7B, Mistral-7B-v0.3, LLaMA-3.1-8B

**14B scale**: Qwen2.5-14B (48 layers), Qwen3-14B (40 layers)

## Citation

If you use this code, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{usama2026beyond,
  title     = {Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models},
  author    = {Usama, Muhammad and Jung, Yeoreum},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
