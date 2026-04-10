"""
Multiple merge strategies for layer communities.

Provides three ways to consolidate a community of layers into one:
1. select: keep the most central layer (no modification)
2. average: average the weights of all layers in the community
3. distill: train a representative via knowledge distillation
"""

import copy
import torch
import torch.nn as nn
import numpy as np

from src.fusion import (
    get_layer_modules, get_all_layers, find_central_layer,
    compute_position_embeddings, CommunityTeacher,
)


def merge_community_select(model, community, cka_matrix):
    """Keep the most central layer from the community."""
    if len(community) == 1:
        return get_layer_modules(model, community)[0]
    central = find_central_layer(cka_matrix, community)
    return get_layer_modules(model, [central])[0]


def merge_community_average(model, community, cka_matrix):
    """Average the weights of all layers in the community."""
    if len(community) == 1:
        return get_layer_modules(model, community)[0]

    layers = get_layer_modules(model, community)
    # Use central layer as base, then average in the others
    central_idx = find_central_layer(cka_matrix, community)
    base_layer = copy.deepcopy(get_layer_modules(model, [central_idx])[0])

    # Weighted average: central layer gets more weight
    state_dicts = [layer.state_dict() for layer in layers]
    base_sd = base_layer.state_dict()

    for key in base_sd:
        if base_sd[key].is_floating_point():
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            base_sd[key] = stacked.mean(dim=0).to(base_sd[key].dtype)

    base_layer.load_state_dict(base_sd)
    return base_layer


def merge_community_slerp(model, community, cka_matrix, t=0.5):
    """
    Spherical linear interpolation between community members.
    Merges pairs iteratively using SLERP, weighted by CKA distance to central.
    """
    if len(community) == 1:
        return get_layer_modules(model, community)[0]
    if len(community) == 2:
        layers = get_layer_modules(model, community)
        return _slerp_layers(layers[0], layers[1], t)

    # For >2 layers, iteratively merge closest pairs
    central = find_central_layer(cka_matrix, community)
    result = copy.deepcopy(get_layer_modules(model, [central])[0])
    other_indices = [i for i in community if i != central]

    for idx in other_indices:
        other = get_layer_modules(model, [idx])[0]
        # Weight by similarity to central
        sim = cka_matrix[central, idx]
        merge_t = 1.0 / (len(community))  # give less weight to each addition
        result = _slerp_layers(result, other, merge_t)

    return result


def _slerp_layers(layer_a, layer_b, t):
    """SLERP between two layers' parameters."""
    result = copy.deepcopy(layer_a)
    sd_a = layer_a.state_dict()
    sd_b = layer_b.state_dict()
    sd_r = result.state_dict()

    for key in sd_r:
        if sd_r[key].is_floating_point():
            a = sd_a[key].float().flatten()
            b = sd_b[key].float().flatten()
            # Normalize
            a_norm = a / (a.norm() + 1e-8)
            b_norm = b / (b.norm() + 1e-8)
            omega = torch.acos(torch.clamp(torch.dot(a_norm, b_norm), -1, 1))
            if omega.abs() < 1e-6:
                # Nearly parallel, use linear interpolation
                merged = (1 - t) * a + t * b
            else:
                merged = (torch.sin((1 - t) * omega) / torch.sin(omega)) * a + \
                         (torch.sin(t * omega) / torch.sin(omega)) * b
            sd_r[key] = merged.reshape(sd_r[key].shape).to(sd_r[key].dtype)

    result.load_state_dict(sd_r)
    return result


def build_compressed_model(model, communities, merge_fn_results):
    """
    Replace each community's layers with a single merged result.
    merge_fn_results is a list of nn.Module, one per community.
    """
    indexed = sorted(enumerate(communities), key=lambda x: min(x[1]))
    new_layers = nn.ModuleList()
    for orig_idx, comm in indexed:
        layer = merge_fn_results[orig_idx].half()
        new_layers.append(layer)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(new_layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = new_layers

    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    return model
