"""
Graph-guided layer selection (no distillation).

Instead of distilling communities into new layers, simply keep
the most representative (central) layer from each community.
This combines our CKA-guided grouping advantage with ShortGPT's
intact weight preservation.

Key insight: ShortGPT removes N least important layers by individual BI score.
We identify communities of functionally similar layers via graph analysis,
then keep the most central member of each — a globally-informed selection
strategy vs ShortGPT's locally-greedy one.
"""

import numpy as np
import torch.nn as nn

from src.fusion import get_all_layers, find_central_layer


def select_representatives(
    model,
    communities: list[list[int]],
    cka_matrix: np.ndarray,
):
    """
    For each community, keep only the most central layer (highest avg CKA
    to other community members). Remove all others.

    Returns modified model with len(communities) layers.
    """
    all_layers = get_all_layers(model)

    # For each community, find the central layer
    keep_indices = []
    for comm in communities:
        if len(comm) == 1:
            keep_indices.append(comm[0])
        else:
            central = find_central_layer(cka_matrix, comm)
            keep_indices.append(central)

    # Sort by original layer order
    keep_indices = sorted(keep_indices)

    # Build new layer list
    new_layers = nn.ModuleList([all_layers[i] for i in keep_indices])

    # Replace
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(new_layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = new_layers

    # Disable KV cache
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    return model, keep_indices
