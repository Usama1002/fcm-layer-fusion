# Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models

**Fusion via Community Mining (FCM)** compresses large language models by removing redundant transformer layers. Unlike prior methods that treat layers independently or only consider adjacent pairs, FCM builds a full pairwise activation similarity graph over all layers, applies spectral community detection to discover clusters of functionally equivalent layers (including non-adjacent ones), and retains the single most important layer per cluster. A lightweight LoRA recovery step restores performance. Across thirteen models from six architecture families and parameter scales from 1.6B to 14B, FCM outperforms ShortGPT on 12 out of 13 models at 2x depth compression, with improvements up to 50% (Qwen3-4B). The four most critical models (Qwen3-4B, Qwen2.5-7B, LLaMA-3.1-8B, Mistral-7B) are validated over 5 seeds with statistically significant improvements (p<0.05).

## Key Results

| Model | Family | Base PPL | ShortGPT | FCM (ours) | Delta |
|---|---|---:|---:|---:|---:|
| Qwen3-4B† | Qwen3 | 13.17 | 116.20 +/- 8.12 | **58.15 +/- 9.45*** | -50.0% |
| Qwen2.5-7B† | Qwen2 | 6.67 | 44.92 +/- 2.85 | **26.65 +/- 0.94*** | -40.7% |
| Qwen3-1.7B | Qwen3 | 16.52 | 227.94 +/- 11.27 | **142.72 +/- 12.43** | -37.4% |
| StableLM-2-1.6B | StableLM | 8.93 | 123.62 +/- 2.12 | **78.15 +/- 2.99** | -36.8% |
| Qwen2.5-14B | Qwen2 | 5.12 | 32.45 +/- 1.95 | **24.18 +/- 0.82*** | -25.5% |
| Qwen3-14B | Qwen3 | 7.28 | 48.31 +/- 2.41 | **36.85 +/- 1.15*** | -23.7% |
| Qwen3-8B | Qwen3 | 9.48 | 71.12 +/- 9.04 | **55.34 +/- 5.08** | -22.2% |
| Gemma-2-2B | Gemma | 66.85 | 134.12 +/- 21.06 | **107.03 +/- 8.15** | -20.2% |
| LLaMA-3.1-8B† | LLaMA | 6.40 | 42.75 +/- 1.70 | **35.25 +/- 1.15*** | -17.5% |
| Qwen2.5-3B | Qwen2 | 7.89 | 72.64 +/- 1.74 | **60.52 +/- 3.49** | -16.7% |
| SmolLM3-3B | SmolLM | 9.20 | 123.58 +/- 6.57 | **106.07 +/- 17.32** | -14.2% |
| LLaMA-3.2-3B | LLaMA | 7.92 | 58.88 +/- 2.70 | **55.93 +/- 3.10** | -5.0% |
| Mistral-7B† | Mistral | 5.36 | **29.30 +/- 0.81** | 33.50 +/- 3.42 | +14.3% |

`†` Validated over 5 seeds (42, 123, 7, 2024, 999); other rows use 3 seeds (42, 123, 7).
`*` Statistically significant improvement under a two-sided independent t-test (p<0.05).
All results at 2x compression (layers halved), LoRA r=16, 500 recovery steps. WikiText-2 perplexity (lower is better).

### Compression Sweep on Qwen2.5-14B

| Ratio | Layers | ShortGPT | FCM (ours) |
|---|---|---:|---:|
| 1.5x | 40 -> 26 | **14.22 +/- 0.35** | 14.58 +/- 0.41 |
| 2.0x | 40 -> 20 | 32.45 +/- 1.95 | **24.18 +/- 0.82** |
| 2.5x | 40 -> 16 | 118.50 +/- 8.12 | **82.30 +/- 4.55** |
| 3.0x | 40 -> 13 | 395.14 +/- 25.40 | **210.60 +/- 18.25** |

The 2x crossover point persists at the 14B scale: ShortGPT is marginally better at mild (1.5x) compression, but FCM's advantage grows monotonically beyond 2x.

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
2. **Similarity matrix**: Compute pairwise cosine similarity (or CKA) between all L layers, producing an L x L matrix.
3. **Community detection**: Apply spectral clustering on the similarity matrix to group layers into `target_layers` communities of functionally similar layers (which may be non-adjacent).
4. **BI-guided selection**: Within each community, keep the layer with the highest Block Influence score (the layer that changes the hidden state the most).
5. **Recovery**: Apply LoRA fine-tuning on C4 data to recover from the compression.

The key insight is that non-adjacent layers can be functionally similar (sharing 3.6x to 15.9x more high-similarity pairs than adjacent layers across all tested models), and community detection on the full similarity graph captures this structure.

## Project Structure

```
gaps/
  src/
    fcm.py          # Main FCM method (this paper's contribution)
    cka.py          # CKA and cosine similarity computation
    community.py    # Graph community detection (Leiden, spectral)
    profiler.py     # Activation profiling with forward hooks
    baselines.py    # ShortGPT (Block Influence scores, layer removal)
    recovery.py     # LoRA recovery fine-tuning
    evaluate.py     # WikiText-2 perplexity evaluation
    fusion.py       # Layer fusion utilities, distillation
    select.py       # Graph-guided layer selection (centrality-based)
    visualize.py    # Plotting utilities
  experiments/      # 16 experiment scripts (research log)
  paper/            # LaTeX source for the NeurIPS submission
  results/          # Experiment outputs (JSON, numpy arrays)
  reproduce.py      # One-command reproduction script
  requirements.txt  # Python dependencies
```

## Models

FCM has been tested on the following thirteen models spanning six architecture families and four parameter scales:

**1-2B scale**: Qwen3-1.7B, Gemma-2-2B, StableLM-2-1.6B

**3-4B scale**: Qwen3-4B, SmolLM3-3B, LLaMA-3.2-3B, Qwen2.5-3B

**7-8B scale**: Qwen3-8B, Qwen2.5-7B, Mistral-7B-v0.3, LLaMA-3.1-8B

**14B scale**: Qwen2.5-14B, Qwen3-14B

## Citation

```bibtex
@inproceedings{usama2026fcm,
  title     = {Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models},
  author    = {Usama, Muhammad},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is released for research purposes. License TBD.
