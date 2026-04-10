"""
Evaluation utilities for compressed models.

Measures perplexity on WikiText-2 as the standard benchmark
for LLM compression papers.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_seq_len: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
    max_samples: int = None,
) -> float:
    """
    Evaluate perplexity on WikiText-2 test set.
    """
    model.eval()
    ds = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all text and tokenize
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    # Split into chunks of max_seq_len
    total_len = input_ids.shape[0]
    num_chunks = total_len // max_seq_len
    if max_samples is not None:
        num_chunks = min(num_chunks, max_samples)

    nlls = []
    with torch.no_grad():
        for i in tqdm(range(num_chunks), desc="Evaluating perplexity"):
            start = i * max_seq_len
            end = start + max_seq_len
            chunk = input_ids[start:end].unsqueeze(0).to(device)

            outputs = model(input_ids=chunk, labels=chunk)
            nll = outputs.loss.float().item()
            nlls.append(nll)

    avg_nll = sum(nlls) / len(nlls)
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return perplexity
