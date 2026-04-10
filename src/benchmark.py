"""
Zero-shot benchmark evaluation using lm-evaluation-harness.
"""

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM


def evaluate_zero_shot(
    model,
    tokenizer,
    tasks: list[str] = None,
    batch_size: int = 4,
    device: str = "cuda",
    num_fewshot: int = 0,
) -> dict:
    """
    Evaluate model on zero-shot benchmarks.

    Returns dict mapping task name -> accuracy.
    """
    if tasks is None:
        tasks = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]

    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=str(device),
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    # Extract accuracies
    scores = {}
    for task_name in tasks:
        if task_name in results["results"]:
            task_res = results["results"][task_name]
            # Try different metric keys
            for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if key in task_res:
                    scores[task_name] = task_res[key]
                    break

    # Compute average
    if scores:
        scores["avg"] = sum(scores.values()) / len(scores)

    return scores
