"""
LoRA recovery fine-tuning for compressed models.

After layer fusion or pruning, applies LoRA adapters and fine-tunes
on a small dataset to recover performance.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def lora_recovery_finetune(
    model,
    tokenizer,
    num_steps: int = 500,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_len: int = 512,
    num_samples: int = 2048,
    device: str = "cuda",
) -> float:
    """
    Apply LoRA adapters and fine-tune the compressed model.

    Returns final training loss.
    """
    # Apply LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load training data
    print(f"  Loading training data ({num_samples} samples)...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for sample in ds:
        text = sample.get("text", "")
        if len(text) > 100:
            texts.append(text)
        if len(texts) >= num_samples:
            break

    encodings = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    dataset = TextDataset(encodings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    model.train()
    step = 0
    losses = []
    pbar = tqdm(total=num_steps, desc="  LoRA recovery")

    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            step += 1
            pbar.update(1)

            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                pbar.set_postfix({"loss": f"{avg:.4f}"})

    pbar.close()

    # Merge LoRA weights back into base model
    model = model.merge_and_unload()
    model.eval()

    final_loss = sum(losses[-50:]) / max(1, len(losses[-50:]))
    print(f"  LoRA recovery complete. Final loss: {final_loss:.4f}")

    return model, final_loss
