import torch
from typing import List, Dict, Any


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Simple collate function: stacks tensors, no augmentation.
    All preprocessing (including mass augment + norm) is done in Dataset.
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "mass": torch.stack([item["mass"] for item in batch]),
        "dish_ids": [item["dish_id"] for item in batch],
    }