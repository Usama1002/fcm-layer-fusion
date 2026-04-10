"""
Baseline compression methods for comparison.

Implements ShortGPT-style layer removal (drop least important layers)
using Block Influence (BI) scores.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def compute_block_influence(
    model,
    tokenizer,
    num_samples: int = 128,
    max_seq_len: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute Block Influence (BI) scores as in ShortGPT.
    BI measures how much each layer changes the hidden state,
    using cosine similarity between input and output.

    BI(layer_i) = 1 - cos_sim(input_i, output_i)
    Low BI = layer barely changes anything = redundant.
    """
    from datasets import load_dataset

    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unsupported architecture")

    num_layers = len(layers)

    # Load calibration data
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for sample in ds:
        if len(sample.get("text", "")) > 50:
            texts.append(sample["text"])
        if len(texts) >= num_samples:
            break

    encodings = tokenizer(
        texts, max_length=max_seq_len, truncation=True,
        padding="max_length", return_tensors="pt",
    )

    # Collect BI scores using hooks
    bi_scores = np.zeros(num_layers)
    layer_input_cache = {}
    layer_output_cache = {}

    def make_input_hook(idx):
        def hook(module, inp, out):
            if isinstance(inp, tuple):
                layer_input_cache[idx] = inp[0].detach()
            else:
                layer_input_cache[idx] = inp.detach()
        return hook

    def make_output_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                layer_output_cache[idx] = out[0].detach()
            else:
                layer_output_cache[idx] = out.detach()
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_input_hook(i)))
        hooks.append(layer.register_forward_hook(make_output_hook(i)))

    model.eval()
    batch_size = 4
    cos_sim = nn.CosineSimilarity(dim=-1)

    with torch.no_grad():
        for start in tqdm(range(0, len(encodings["input_ids"]), batch_size), desc="Computing BI"):
            end = min(start + batch_size, len(encodings["input_ids"]))
            input_ids = encodings["input_ids"][start:end].to(device)
            attention_mask = encodings["attention_mask"][start:end].to(device)

            layer_input_cache.clear()
            layer_output_cache.clear()
            model(input_ids=input_ids, attention_mask=attention_mask)

            for i in range(num_layers):
                inp = layer_input_cache[i].float()
                out = layer_output_cache[i].float()
                # Mean over batch and sequence
                sim = cos_sim(inp, out).mean().item()
                bi_scores[i] += (1.0 - sim)

    for h in hooks:
        h.remove()

    # Average over batches
    num_batches = len(encodings["input_ids"]) // batch_size
    bi_scores /= max(1, num_batches)

    return bi_scores


def remove_layers_by_bi(model, bi_scores: np.ndarray, num_to_remove: int):
    """
    Remove the least important layers (lowest BI scores).
    ShortGPT baseline: simply delete layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unsupported architecture")

    # Find indices to keep (highest BI)
    keep_indices = np.argsort(bi_scores)[::-1][:len(layers) - num_to_remove]
    keep_indices = sorted(keep_indices)

    new_layers = nn.ModuleList([layers[i] for i in keep_indices])

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(new_layers)

    # Clear any cached generation config that references old layer count
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    # Force no cache during eval to avoid layer index mismatch
    model.config.use_cache = False

    return model, keep_indices
