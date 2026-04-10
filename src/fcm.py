"""
Fusion via Community Mining (FCM) — the paper's final compression method.

Compresses a HuggingFace causal LM by:
  1. Profiling layer activations to build a pairwise similarity matrix
  2. Running spectral clustering to group layers into communities
  3. Selecting the highest Block-Influence layer per community
  4. Reassembling the model with only the selected layers
  5. Applying LoRA recovery fine-tuning to restore performance

This is the single entry point for compressing any supported model.
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.profiler import collect_activations
from src.cka import compute_similarity_matrix
from src.community import detect_communities
from src.baselines import compute_block_influence
from src.recovery import lora_recovery_finetune
from src.evaluate import evaluate_perplexity
from src.fusion import get_all_layers


def _set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(model_name: str, device: str = "cuda"):
    """Load a HuggingFace causal LM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer


def _free_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


def compute_sim_matrix(
    model_name: str,
    metric: str = "cosine",
    num_samples: int = 1024,
    max_seq_len: int = 128,
    batch_size: int = 8,
    device: str = "cuda",
) -> np.ndarray:
    """
    Profile activations and compute the L x L pairwise similarity matrix.

    Args:
        model_name: HuggingFace model identifier.
        metric: Similarity metric ("cosine" or "cka").
        num_samples: Number of calibration samples from C4.
        max_seq_len: Maximum sequence length for calibration.
        batch_size: Batch size for profiling.
        device: Device for computation.

    Returns:
        L x L numpy similarity matrix.
    """
    activations = collect_activations(
        model_name,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )
    sim_matrix = compute_similarity_matrix(activations, metric=metric)
    return sim_matrix


def select_layers_fcm(
    sim_matrix: np.ndarray,
    bi_scores: np.ndarray,
    target_layers: int,
) -> list[int]:
    """
    FCM layer selection: spectral community detection + BI-guided picking.

    Groups all layers into ``target_layers`` communities via spectral
    clustering on the similarity matrix, then keeps the highest Block-
    Influence layer from each community.

    Args:
        sim_matrix: L x L pairwise similarity matrix.
        bi_scores: Length-L array of Block Influence scores.
        target_layers: Number of layers to retain.

    Returns:
        Sorted list of layer indices to keep.
    """
    communities = detect_communities(
        sim_matrix, method="spectral", n_clusters=target_layers
    )
    keep = []
    for comm in communities:
        if len(comm) == 1:
            keep.append(comm[0])
        else:
            best = max(comm, key=lambda i: bi_scores[i])
            keep.append(best)
    return sorted(keep)


def build_compressed_model(model, keep_indices: list[int]):
    """
    Build the compressed model by retaining only the selected layers.

    Args:
        model: A HuggingFace causal LM.
        keep_indices: Sorted list of layer indices to keep.

    Returns:
        The model with its layer list replaced.
    """
    layers = get_all_layers(model)
    new_layers = nn.ModuleList([layers[i] for i in keep_indices])

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
        model.config.num_hidden_layers = len(new_layers)
    else:
        raise ValueError("Unsupported architecture — expected model.model.layers")

    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    return model


def fcm_compress(
    model_name: str,
    target_layers: int | None = None,
    metric: str = "cosine",
    seed: int = 42,
    device: str = "cuda",
    sim_matrix: np.ndarray | None = None,
    bi_scores: np.ndarray | None = None,
    lora_r: int = 16,
    lora_lr: float = 2e-4,
    lora_steps: int = 500,
    lora_samples: int = 2048,
    eval_max_samples: int = 50,
    eval_max_seq_len: int = 2048,
    profile_samples: int = 1024,
    profile_seq_len: int = 128,
    profile_batch_size: int = 8,
    skip_recovery: bool = False,
    skip_eval: bool = False,
):
    """
    End-to-end FCM compression of any HuggingFace causal LM.

    This is the ONE function you need to compress a model using the
    paper's final method: spectral community detection on a pairwise
    activation similarity graph, with Block-Influence-guided layer
    selection within each community, followed by LoRA recovery.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen2.5-7B").
        target_layers: Number of layers to keep. Defaults to L // 2 (2x
            compression). Must be >= 4.
        metric: Similarity metric for profiling ("cosine" or "cka").
        seed: Random seed for reproducibility.
        device: Device for computation ("cuda" or "cpu").
        sim_matrix: Pre-computed L x L similarity matrix. If None, it will
            be profiled from scratch.
        bi_scores: Pre-computed Block Influence scores. If None, they will
            be computed from scratch.
        lora_r: LoRA rank for recovery fine-tuning.
        lora_lr: Learning rate for LoRA recovery.
        lora_steps: Number of LoRA recovery training steps.
        lora_samples: Number of training samples for LoRA recovery.
        eval_max_samples: Maximum WikiText-2 chunks for perplexity eval.
        eval_max_seq_len: Sequence length for perplexity evaluation.
        profile_samples: Number of calibration samples for activation profiling.
        profile_seq_len: Sequence length for activation profiling.
        profile_batch_size: Batch size for activation profiling.
        skip_recovery: If True, skip LoRA recovery (return raw compressed model).
        skip_eval: If True, skip perplexity evaluation.

    Returns:
        dict with keys:
            - "model": the compressed (and optionally recovered) model
            - "tokenizer": the tokenizer
            - "perplexity": WikiText-2 perplexity (None if skip_eval=True)
            - "keep_indices": list of retained layer indices
            - "sim_matrix": the L x L similarity matrix used
            - "bi_scores": the Block Influence scores used
            - "communities": list of communities from spectral clustering
    """
    _set_seed(seed)

    # Resolve target_layers
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    total_layers = config.num_hidden_layers
    if target_layers is None:
        target_layers = max(total_layers // 2, 4)
    if target_layers < 4:
        raise ValueError(f"target_layers must be >= 4, got {target_layers}")
    if target_layers >= total_layers:
        raise ValueError(
            f"target_layers ({target_layers}) must be < total layers ({total_layers})"
        )

    short_name = model_name.split("/")[-1]
    print(f"FCM compress: {short_name} ({total_layers} -> {target_layers} layers, metric={metric})")

    # Step 1: Compute similarity matrix
    if sim_matrix is None:
        print("Step 1/5: Profiling activations and computing similarity matrix...")
        sim_matrix = compute_sim_matrix(
            model_name,
            metric=metric,
            num_samples=profile_samples,
            max_seq_len=profile_seq_len,
            batch_size=profile_batch_size,
            device=device,
        )
    else:
        print("Step 1/5: Using pre-computed similarity matrix.")

    # Step 2: Compute Block Influence scores
    if bi_scores is None:
        print("Step 2/5: Computing Block Influence scores...")
        model, tokenizer = _load_model(model_name, device=device)
        bi_scores = compute_block_influence(model, tokenizer, device=device)
        _free_model(model)
    else:
        print("Step 2/5: Using pre-computed BI scores.")

    # Step 3: Select layers via FCM
    print("Step 3/5: Running spectral community detection + BI-guided selection...")
    communities = detect_communities(
        sim_matrix, method="spectral", n_clusters=target_layers
    )
    keep_indices = select_layers_fcm(sim_matrix, bi_scores, target_layers)
    print(f"  Keeping layers: {keep_indices}")

    # Step 4: Build compressed model
    print("Step 4/5: Building compressed model...")
    model, tokenizer = _load_model(model_name, device=device)
    model = build_compressed_model(model, keep_indices)

    # Step 5: LoRA recovery
    if not skip_recovery:
        print("Step 5/5: Running LoRA recovery fine-tuning...")
        _set_seed(seed)
        model, _ = lora_recovery_finetune(
            model, tokenizer,
            device=device,
            num_steps=lora_steps,
            lr=lora_lr,
            lora_r=lora_r,
            num_samples=lora_samples,
        )
    else:
        print("Step 5/5: Skipping LoRA recovery (skip_recovery=True).")

    # Evaluate
    perplexity = None
    if not skip_eval:
        print("Evaluating perplexity on WikiText-2...")
        perplexity = evaluate_perplexity(
            model, tokenizer,
            max_seq_len=eval_max_seq_len,
            device=device,
            max_samples=eval_max_samples,
        )
        print(f"Perplexity: {perplexity:.2f}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "perplexity": perplexity,
        "keep_indices": keep_indices,
        "sim_matrix": sim_matrix,
        "bi_scores": bi_scores,
        "communities": communities,
    }
