from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import MeanAbsoluteError, R2Score
import os
from pathlib import Path
import random
import numpy as np
import torch

from .multimodal_dataset import MultimodalDataset
from .multimodal_collate_fn import multimodal_collate_fn
from .get_multimodal_transforms import get_multimodal_transforms
from ..experiment_config import Config


def seed_everywhere(seed: int) -> None:
    """
    Set seed for all possible sources of randomness to ensure reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def train_multimodal_model(model, config: Config) -> None:
    """
    End-to-end training pipeline for multimodal calorie regression.

    Args:
        config (Config): Training configuration with all hyperparameters and paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    seed_everywhere(config.SEED)

    # === Transforms ===
    train_transforms = get_multimodal_transforms(config, ds_type="train")
    val_transforms = get_multimodal_transforms(config, ds_type="test")

    # === Datasets ===
    train_dataset = MultimodalDataset(config, split="train", transform=train_transforms)
    val_dataset = MultimodalDataset(config, split="test", transform=val_transforms)

    # === DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # === Model ===
    model = model.to(device)

    # === Optimizer ===
    optimizer = AdamW([
        {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
        {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
        {"params": model.mass_mlp.parameters(), "lr": config.HEAD_LR},
        {"params": model.cross_attn.parameters(), "lr": config.FUSION_LR},
        {"params": model.regressor.parameters(), "lr": config.HEAD_LR},
    ])

    # === Loss & Metrics ===
    criterion = torch.nn.HuberLoss(delta=1.0)
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    # === Training Loop ===
    best_val_mae = float("inf")
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Starting training...")

    for epoch in range(1, config.EPOCHS + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        mae_metric.reset()
        r2_metric.reset()

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch}/{config.EPOCHS} [Train]",
            unit="batch",
            leave=False,
        ) as pbar:
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                mass = batch["mass"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad()
                preds = model(image=image, input_ids=input_ids, attention_mask=attention_mask, mass=mass)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                mae_metric.update(preds, labels)
                r2_metric.update(preds, labels)

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        train_mae = mae_metric.compute().item()
        train_r2 = r2_metric.compute().item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        mae_metric.reset()
        r2_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                mass = batch["mass"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                preds = model(image=image, input_ids=input_ids, attention_mask=attention_mask, mass=mass)
                loss = criterion(preds, labels)

                val_loss += loss.item()
                mae_metric.update(preds, labels)
                r2_metric.update(preds, labels)

        avg_val_loss = val_loss / len(val_loader)
        val_mae = mae_metric.compute().item()
        val_r2 = r2_metric.compute().item()

        # --- Logging ---
        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}"
        )

        # --- Save best model (by val MAE) ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved: {config.model_save_path} (Val MAE: {val_mae:.2f})")