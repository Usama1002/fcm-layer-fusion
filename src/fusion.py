"""
Distillation-based layer fusion.

Given a community of layers, trains a single representative layer
to approximate the composed transformation of the full community.
"""

import copy
import torch
import torch.nn as nn
from tqdm import tqdm


def get_rotary_emb(model):
    """Get the rotary embedding module if it exists."""
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    return None


def compute_position_embeddings(model, position_ids, hidden_states):
    """Compute RoPE position embeddings (cos, sin) if the model uses them."""
    rotary_emb = get_rotary_emb(model)
    if rotary_emb is not None:
        return rotary_emb(hidden_states, position_ids)
    return None


def get_layer_modules(model, layer_indices: list[int]) -> list[nn.Module]:
    """Get the actual nn.Module objects for given layer indices."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [model.model.layers[i] for i in layer_indices]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [model.transformer.h[i] for i in layer_indices]
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def get_all_layers(model) -> nn.ModuleList:
    """Get the full layer list from the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def find_central_layer(cka_matrix, community: list[int]) -> int:
    """
    Find the most central layer in a community (highest avg CKA to others).
    """
    best_idx = community[0]
    best_avg = -1
    for i in community:
        avg_sim = sum(cka_matrix[i, j] for j in community if j != i) / max(1, len(community) - 1)
        if avg_sim > best_avg:
            best_avg = avg_sim
            best_idx = i
    return best_idx


class CommunityTeacher(nn.Module):
    """
    Wraps a sequence of transformer layers (the community) as a teacher.
    Given hidden states input, applies all community layers in order.
    """
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None):
        for layer in self.layers:
            kwargs = {"attention_mask": attention_mask}
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            else:
                kwargs["position_ids"] = position_ids
            outputs = layer(hidden_states, **kwargs)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
        return hidden_states


def distill_community(
    model,
    community: list[int],
    cka_matrix,
    calibration_inputs: dict,
    num_steps: int = 200,
    lr: float = 1e-4,
    device: str = "cuda",
) -> tuple[nn.Module, float]:
    """
    Distill a community of layers into a single representative layer.

    Args:
        model: the full model
        community: list of layer indices in this community
        cka_matrix: CKA matrix for finding the central layer
        calibration_inputs: dict with 'input_ids' and 'attention_mask'
        num_steps: distillation training steps
        lr: learning rate
        device: device

    Returns:
        (representative_layer, final_loss)
    """
    if len(community) <= 1:
        layers = get_layer_modules(model, community)
        return layers[0], 0.0

    # Sort community by layer index (maintain execution order)
    community = sorted(community)

    # Build teacher from the community layers
    teacher_layers = get_layer_modules(model, community)
    teacher = CommunityTeacher(teacher_layers).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialize student from the central layer
    central_idx = find_central_layer(cka_matrix, community)
    student_layer = copy.deepcopy(get_layer_modules(model, [central_idx])[0]).to(device)
    # Convert to float32 for training and ensure gradients are enabled
    student_layer = student_layer.float()
    for p in student_layer.parameters():
        p.requires_grad = True
    student_layer.train()

    # Collect hidden states at the input of the first community layer
    # We need to run the model up to the first community layer
    first_layer_idx = community[0]
    all_layers = get_all_layers(model)

    print(f"  Collecting teacher inputs at layer {first_layer_idx}...")
    hidden_states_cache = []
    pos_emb_cache = []

    model.eval()
    batch_size = 4
    with torch.no_grad():
        input_ids = calibration_inputs["input_ids"].to(device)
        attention_mask = calibration_inputs["attention_mask"].to(device)
        for start in range(0, len(input_ids), batch_size):
            end = min(start + batch_size, len(input_ids))
            batch_ids = input_ids[start:end]

            # Get embeddings
            if hasattr(model, "model"):
                embed_out = model.model.embed_tokens(batch_ids)
            else:
                embed_out = model.transformer.wte(batch_ids)

            hidden = embed_out
            seq_len = batch_ids.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(end - start, -1)

            # Compute position embeddings (RoPE cos/sin) if available
            pos_emb = compute_position_embeddings(model, pos_ids, hidden)

            # Build layer kwargs
            def layer_kwargs():
                kw = {"attention_mask": None}
                if pos_emb is not None:
                    kw["position_embeddings"] = pos_emb
                else:
                    kw["position_ids"] = pos_ids
                return kw

            # Run through layers before the community
            for idx in range(first_layer_idx):
                layer_out = all_layers[idx](hidden, **layer_kwargs())
                if isinstance(layer_out, tuple):
                    hidden = layer_out[0]
                else:
                    hidden = layer_out

            hidden_states_cache.append(hidden.detach())
            if pos_emb is not None:
                pos_emb_cache.append((pos_emb[0].detach(), pos_emb[1].detach()))
            else:
                pos_emb_cache.append(None)

    hidden_states_all = torch.cat(hidden_states_cache, dim=0)
    # Build uniform pos_emb from first batch (same seq_len for all due to padding)
    has_pos_emb = pos_emb_cache[0] is not None
    if has_pos_emb:
        pos_emb_cos = torch.cat([p[0] for p in pos_emb_cache], dim=0)
        pos_emb_sin = torch.cat([p[1] for p in pos_emb_cache], dim=0)

    # Compute teacher outputs
    print(f"  Computing teacher outputs for {len(community)} layers...")
    teacher_outputs_cache = []
    with torch.no_grad():
        for start in range(0, len(hidden_states_all), batch_size):
            end = min(start + batch_size, len(hidden_states_all))
            h = hidden_states_all[start:end]
            if has_pos_emb:
                pe = (pos_emb_cos[start:end], pos_emb_sin[start:end])
                teacher_out = teacher(h, position_embeddings=pe)
            else:
                teacher_out = teacher(h)
            teacher_outputs_cache.append(teacher_out.detach())

    teacher_outputs_all = torch.cat(teacher_outputs_cache, dim=0)

    # Scale steps by community size, capped to avoid excessive training
    effective_steps = min(num_steps * len(community), num_steps * 5)
    # Scale LR: slight reduction for large communities, boost for small ones
    if len(community) <= 2:
        effective_lr = lr * 1.5
    elif len(community) <= 4:
        effective_lr = lr
    else:
        effective_lr = lr / (len(community) ** 0.3)
    print(f"  Distilling {len(community)} layers -> 1 (central: layer {central_idx}, {effective_steps} steps, lr={effective_lr:.2e})...")

    optimizer = torch.optim.AdamW(student_layer.parameters(), lr=effective_lr, weight_decay=0.01)
    warmup_steps = min(100, effective_steps // 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, effective_steps - warmup_steps)
    )
    # Use SmoothL1Loss (Huber) for robustness against large-magnitude targets
    loss_fn = nn.SmoothL1Loss()

    num_samples = len(hidden_states_all)
    losses = []
    best_loss = float("inf")
    best_state = None
    nan_count = 0

    for step in range(effective_steps):
        # Linear warmup
        if step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = effective_lr * warmup_factor

        # Random batch
        idx = torch.randint(0, num_samples, (min(batch_size, num_samples),))
        h_in = hidden_states_all[idx].to(device)
        target = teacher_outputs_all[idx].to(device)

        # Student forward with position embeddings
        student_kwargs = {"attention_mask": None}
        if has_pos_emb:
            student_kwargs["position_embeddings"] = (pos_emb_cos[idx], pos_emb_sin[idx])
        student_out = student_layer(h_in, **student_kwargs)
        if isinstance(student_out, tuple):
            student_out = student_out[0]

        loss = loss_fn(student_out.float(), target.float())

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count > 10:
                print(f"    WARNING: Too many NaN losses, stopping early at step {step}")
                break
            continue
        nan_count = 0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_layer.parameters(), 0.5)
        optimizer.step()
        if step >= warmup_steps:
            scheduler.step()

        cur_loss = loss.item()
        losses.append(cur_loss)

        # Save best checkpoint
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_state = {k: v.clone() for k, v in student_layer.state_dict().items()}

        if (step + 1) % max(1, effective_steps // 4) == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            print(f"    Step {step+1}/{effective_steps}, Loss: {avg_loss:.6f}")

    # Restore best checkpoint
    if best_state is not None:
        student_layer.load_state_dict(best_state)

    final_loss = best_loss if best_loss < float("inf") else (sum(losses[-20:]) / max(1, len(losses[-20:])))
    student_layer.eval()

    return student_layer, final_loss


def build_fused_model(
    model,
    communities: list[list[int]],
    representative_layers: list[nn.Module],
):
    """
    Replace each community's layers with a single representative in the model.

    Communities must cover all layers exactly once.
    Layers are reordered so that communities appear in order of their
    minimum layer index.

    Returns the modified model with fewer layers.
    """
    all_layers = get_all_layers(model)

    # Sort communities by their minimum layer index
    indexed = sorted(enumerate(communities), key=lambda x: min(x[1]))

    new_layers = nn.ModuleList()
    for orig_idx, comm in indexed:
        # Convert back to model dtype (fp16) for inference
        layer = representative_layers[orig_idx].half()
        new_layers.append(layer)

    # Replace the layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(new_layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = new_layers

    # Disable KV cache to avoid layer index mismatch after fusion
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    return model
