import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import List, Dict, Any

def multimodal_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, torch.Tensor]:
    """Collates a batch of samples into tensors for multimodal training.

    Tokenizes text, stacks images, and converts labels to tensors.

    Args:
        batch (List[Dict[str, Any]]): List of samples from MultimodalDataset.
        tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer.
        max_length (Optional[int]): Max sequence length for tokenization.
            If None, tokenizer's default is used.

    Returns:
        Dict[str, torch.Tensor]: Batch dictionary with keys:
            - 'label': (B,)
            - 'image': (B, C, H, W)
            - 'input_ids': (B, L)
            - 'attention_mask': (B, L)
    """
    texts: List[str] = [item["text"] for item in batch]
    images: List[torch.Tensor] = [item["image"] for item in batch]
    labels: List[int] = [item["label"] for item in batch]

    # Stack images (assumes all have same shape after transforms)
    image_tensor = torch.stack(images, dim=0)  # (B, C, H, W)
    label_tensor = torch.LongTensor(labels)    # (B,)

    # Tokenize texts
    tokenized: BatchEncoding = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    return {
        "label": label_tensor,
        "image": image_tensor,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    } # type:ignore