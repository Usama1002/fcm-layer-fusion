"""
CKA (Centered Kernel Alignment) computation for layer similarity analysis.

Based on Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019).
Uses linear CKA for efficiency: CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
where X and Y are centered activation matrices of shape (N, D).
"""

import torch
import numpy as np


def center_matrix(X: torch.Tensor) -> torch.Tensor:
    """Center columns of X to have zero mean."""
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute linear CKA between two activation matrices.

    Args:
        X: (N, D1) activation matrix from layer i
        Y: (N, D2) activation matrix from layer j

    Returns:
        CKA similarity score in [0, 1]
    """
    X = center_matrix(X.float())
    Y = center_matrix(Y.float())

    # Compute HSIC terms
    YtX = Y.T @ X  # (D2, D1)
    XtX = X.T @ X  # (D1, D1)
    YtY = Y.T @ Y  # (D2, D2)

    hsic_xy = (YtX * YtX).sum()  # ||Y^T X||_F^2
    hsic_xx = (XtX * XtX).sum()  # ||X^T X||_F^2
    hsic_yy = (YtY * YtY).sum()  # ||Y^T Y||_F^2

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return (hsic_xy / denom).item()


def cosine_similarity_matrix(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute mean cosine similarity between two activation matrices.
    This is the metric ShortGPT uses (Block Influence is 1 - cosine_sim).
    """
    X = X.float()
    Y = Y.float()
    # Mean-pool to get per-sample vectors, then compute cosine similarity
    cos = torch.nn.functional.cosine_similarity(X, Y, dim=-1)
    return cos.mean().item()


def weight_cosine_similarity(model, i: int, j: int) -> float:
    """
    Compute cosine similarity between flattened weight matrices of two layers.
    This is a weight-space metric (no data needed).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unsupported architecture")

    params_i = torch.cat([p.flatten().float() for p in layers[i].parameters()])
    params_j = torch.cat([p.flatten().float() for p in layers[j].parameters()])
    return torch.nn.functional.cosine_similarity(params_i.unsqueeze(0), params_j.unsqueeze(0)).item()


def compute_similarity_matrix(activations: list[torch.Tensor], metric: str = "cka",
                               model=None) -> np.ndarray:
    """
    Compute full L x L pairwise similarity matrix with specified metric.

    Args:
        activations: list of L tensors, each of shape (N, D)
        metric: "cka", "cosine", or "weight_cosine"
        model: required for "weight_cosine" metric

    Returns:
        L x L numpy array of similarities
    """
    L = len(activations)
    sim_matrix = np.zeros((L, L))

    for i in range(L):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, L):
            if metric == "cka":
                score = linear_cka(activations[i], activations[j])
            elif metric == "cosine":
                score = cosine_similarity_matrix(activations[i], activations[j])
            elif metric == "weight_cosine":
                score = weight_cosine_similarity(model, i, j)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    return sim_matrix


def compute_cka_matrix(activations: list[torch.Tensor]) -> np.ndarray:
    """Convenience wrapper for CKA-based similarity matrix."""
    return compute_similarity_matrix(activations, metric="cka")
