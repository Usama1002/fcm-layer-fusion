# Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models

**Fusion via Community Mining (FCM)** compresses large language models by removing redundant transformer layers. Unlike prior methods that treat layers independently or only consider adjacent pairs, FCM builds a full pairwise activation similarity graph over all layers, applies spectral community detection to discover clusters of functionally equivalent layers (including non-adjacent ones), and retains the single most important layer per cluster. A lightweight LoRA recovery step restores performance. Across eleven models from six architecture families at 2x depth compression, FCM outperforms ShortGPT on 10 out of 11 models, with improvements up to 50% (Qwen3-4B) and a median improvement of 20%.

## Key Results

| Model | Family | Base PPL | ShortGPT | FCM (ours) | Delta |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | Qwen3 | 13.17 | 115.99 +/- 8.59 | **57.98 +/- 11.23** | -50.0% |
| Qwen2.5-7B | Qwen2 | 6.67 | 44.81 +/- 3.07 | **26.72 +/- 1.08** | -40.4% |
| Qwen3-1.7B | Qwen3 | 16.52 | 227.94 +/- 11.27 | **142.72 +/- 12.43** | -37.4% |
| StableLM-2-1.6B | StableLM | 8.93 | 123.62 +/- 2.12 | **78.15 +/- 2.99** | -36.8% |
| Qwen3-8B | Qwen3 | 9.48 | 71.12 +/- 9.04 | **55.34 +/- 5.08** | -22.2% |
| Gemma-2-2B | Gemma | 66.85 | 134.12 +/- 21.06 | **107.03 +/- 8.15** | -20.2% |
| LLaMA-3.1-8B | LLaMA | 6.40 | 42.88 +/- 1.64 | **35.18 +/- 1.29** | -18.0% |
| Qwen2.5-3B | Qwen2 | 7.89 | 72.64 +/- 1.74 | **60.52 +/- 3.49** | -16.7% |
| SmolLM3-3B | SmolLM | 9.20 | 123.58 +/- 6.57 | **106.07 +/- 17.32** | -14.2% |
| LLaMA-3.2-3B | LLaMA | 7.92 | 58.88 +/- 2.70 | **55.93 +/- 3.10** | -5.0% |
| Mistral-7B | Mistral | 5.36 | **29.22 +/- 0.73** | 33.16 +/- 3.93 | +13.5% |

All results at 2x compression (layers halved), mean +/- std over 3 random seeds, LoRA r=16, 500 recovery steps. WikiText-2 perplexity (lower is better).

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

Reproduce the paper's main results table (all 11 models, 3 seeds each):

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

FCM has been tested on the following eleven models spanning six architecture families and three parameter scales:

**1-2B scale**: Qwen3-1.7B, Gemma-2-2B, StableLM-2-1.6B

**3-4B scale**: Qwen3-4B, SmolLM3-3B, LLaMA-3.2-3B, Qwen2.5-3B

**7-8B scale**: Qwen3-8B, Qwen2.5-7B, Mistral-7B-v0.3, LLaMA-3.1-8B

## Citation

```bibtex
@inproceedings{anonymous2026fcm,
  title     = {Beyond Adjacent Layers: Graph-Guided Layer Fusion for Compressing Large Language Models},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is released for research purposes. License TBD.
