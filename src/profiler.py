"""
Activation profiler for transformer models.

Runs calibration data through a model and captures hidden state outputs
at each transformer block boundary.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def get_layer_hook_points(model) -> list[str]:
    """
    Identify the module names for each transformer block's output.
    Supports LLaMA, Mistral, Qwen, Phi architectures.
    """
    # Try common attribute names for the transformer block list
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA, Mistral, Qwen
        return [f"model.layers.{i}" for i in range(len(model.model.layers))]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style
        return [f"transformer.h.{i}" for i in range(len(model.transformer.h))]
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        # OPT style
        layers = model.model.decoder.layers
        return [f"model.decoder.layers.{i}" for i in range(len(layers))]
    else:
        raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def get_module_by_name(model, name: str):
    """Retrieve a submodule by dot-separated name."""
    parts = name.split(".")
    module = model
    for p in parts:
        module = getattr(module, p)
    return module


def collect_activations(
    model_name: str,
    num_samples: int = 1024,
    max_seq_len: int = 128,
    batch_size: int = 8,
    dataset_name: str = "allenai/c4",
    dataset_split: str = "train",
    dataset_config: str = "en",
    device: str = "cuda",
    dtype=torch.float16,
) -> list[torch.Tensor]:
    """
    Run calibration data through model and collect per-layer activations.

    Returns list of L tensors, each of shape (num_samples, hidden_dim).
    We take the mean hidden state across sequence positions per sample
    to get a fixed-size representation per layer per sample.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    hook_points = get_layer_hook_points(model)
    num_layers = len(hook_points)
    print(f"Found {num_layers} transformer blocks")

    # Load calibration data
    print(f"Loading calibration data: {dataset_name}")
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    texts = []
    for sample in ds:
        text = sample.get("text", "")
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= num_samples:
            break

    print(f"Collected {len(texts)} calibration samples")

    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Storage for mean-pooled activations per layer
    all_activations = [[] for _ in range(num_layers)]

    # Register hooks
    hooks = []
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Mean-pool across sequence length, move to CPU
            # hidden shape: (batch, seq_len, hidden_dim)
            pooled = hidden.float().mean(dim=1).detach().cpu()
            layer_outputs[layer_idx] = pooled
        return hook_fn

    for idx, name in enumerate(hook_points):
        module = get_module_by_name(model, name)
        h = module.register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Run inference in batches
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Profiling activations...")
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(loader, desc="Profiling"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            layer_outputs.clear()
            model(input_ids=input_ids, attention_mask=attention_mask)
            for idx in range(num_layers):
                all_activations[idx].append(layer_outputs[idx])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate batches: each tensor is (num_samples, hidden_dim)
    activations = [torch.cat(acts, dim=0) for acts in all_activations]
    print(f"Activation shapes: {activations[0].shape} x {num_layers} layers")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    return activations
