import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer
from torchmetrics import F1Score, Metric

# Наши модули для мультимодальной нейросети
from DeepPlate import (
    MultimodalDataset,
    multimodal_collate_fn, 
    get_multimodal_transforms,
    CrossAttentionModel,
    set_requires_grad,
    Config
)

def train_multimodal_model(config: Config) -> None:
    """
    End-to-end training pipeline for a multimodal (text + image) classification model.

    This function:
    - Initializes model, tokenizer, and data loaders from a single config,
    - Applies layer freezing/unfreezing as specified,
    - Runs training with per-epoch validation,
    - Saves the best checkpoint based on validation accuracy.

    Args:
        config (Any): Configuration object with the following expected attributes:
            - TEXT_MODEL_NAME, IMAGE_MODEL_NAME (str)
            - TEXT_MODEL_UNFREEZE, IMAGE_MODEL_UNFREEZE (str)
            - BATCH_SIZE, EPOCHS, HIDDEN_DIM, NUM_CLASSES (int)
            - TEXT_LR, IMAGE_LR, CLASSIFIER_LR (float)
            - TRAIN_DF_PATH, VAL_DF_PATH, SAVE_PATH (str)
    """
    # === Устройство ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # === Модель и токенизатор ===
    model = CrossAttentionModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # === Разморозка слоёв ===
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE)

    # === Оптимизатор с раздельными LR ===
    optimizer = AdamW([
        {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
        {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
        {"params": model.classifier.parameters(), "lr": config.CLASSIFIER_LR},
    ])

    # === Loss ===
    criterion = torch.nn.CrossEntropyLoss()

    # === DataLoader'ы ===
    train_transforms = get_multimodal_transforms(config, ds_type="train")
    val_transforms = get_multimodal_transforms(config, ds_type="val")

    train_dataset = MultimodalDataset(config, transforms=train_transforms, dataset_type="train")
    val_dataset = MultimodalDataset(config, transforms=val_transforms, dataset_type="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(multimodal_collate_fn, tokenizer=tokenizer),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(multimodal_collate_fn, tokenizer=tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    # === Цикл обучения ===
    f1_metric = F1Score(
        task="binary" if config.NUM_CLASSES == 2 else "multiclass", 
        num_classes=config.NUM_CLASSES).to(device)
    f1_metric_train = F1Score(
        task="binary" if config.NUM_CLASSES == 2 else "multiclass", 
        num_classes=config.NUM_CLASSES).to(device)
    best_f1 = 0.0
    print("🚀 Starting training...")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        with tqdm(
            total=num_batches,
            desc=f"Epoch {epoch}/{config.EPOCHS}",
            unit="batch",
            leave=True
        ) as pbar:
            for batch in train_loader:
                # Перенос на устройство
                inputs = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                    "image": batch["image"].to(device, non_blocking=True),
                }
                labels = batch["label"].to(device, non_blocking=True)

                # Forward
                optimizer.zero_grad()
                logits = model(**inputs)
                loss = criterion(logits, labels)

                # Backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # f1 score
                _ = f1_metric_train(
                    preds=logits.softmax(),
                    labels=labels
                )
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)
        result_epoch_f1_train_score = f1_metric_train.compute().cpu().numpy()
        f1_metric_train.reset()

        val_f1 = validate(
            model=model, 
            val_loader=val_loader,
            device=device,
            fone=f1_metric)
        f1_metric.reset()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} \n| avg_Loss: {total_loss/len(train_loader):.4f}\n| Train F1: {result_epoch_f1_train_score} \n| Val F1: {val_f1 :.4f}")

        # === Сохранение лучшей модели ===
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"🏆 New best model saved: {config.SAVE_PATH} (val_f1 = {val_f1:.4f})\n")
        else:
            print("   ❌ No improvement.\n")


@torch.no_grad()
def validate(
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    device: torch.device,
    fone: Metric) -> float:
    """Compute validation accuracy."""
    model.eval()

    for batch in val_loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "image": batch["image"].to(device),
        }
        labels = batch["label"].to(device)

        logits = model(**inputs)
        _, preds = logits.argmax(dim=1)
        _ = fone(preds=preds, target=labels)
        # correct += (preds == labels).sum().item()
        # total += labels.size(0)

    # return correct / total
    return fone.compute().cpu().numpy()