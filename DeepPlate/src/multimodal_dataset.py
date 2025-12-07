from typing import Dict, Any, Union, List
import random
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np

from DeepPlate import Config

class MultimodalDataset(Dataset):
    """
    Multimodal dataset for calorie regression from food images and ingredient lists.

    Args:
        config (Config): Configuration object with dataset paths and tokenizer info.
        split (str): Either 'train' or 'test'. Used to filter dish.csv.
        transform (callable, optional): Transform to apply to images.
    """

    def __init__(
        self,
        config: Config,
        split: str,
        transform=None,
    ):
        self.config: Config = config
        self.split = split
        self.transform = transform
        self.augment_mass = (split == "train") 

        # Load CSVs
        self.dish_df = pd.read_csv(config.dish_csv_path)
        self.ingredients_df = pd.read_csv(config.ingredients_csv_path)
        # Outliers handling
        mask_outliers = self.dish_df["total_mass"] > 1200
        self.dish_df.loc[mask_outliers, "total_mass"] = self.dish_df.loc[mask_outliers, "total_mass"] / 10.0

        # Build ingredient ID → name map
        self.ingredients_df["id_str"] = "ingr_" + self.ingredients_df["id"].astype(str).str.zfill(10)
        self.ingr_id_to_name = dict(zip(self.ingredients_df["id_str"], self.ingredients_df["ingr"]))

        self.mass_mean = self.dish_df["total_mass"].mean()
        self.mass_std = self.dish_df["total_mass"].std()

        # Filter by split
        self.dish_df = self.dish_df[self.dish_df["split"] == split].reset_index(drop=True)

        # Initialize tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    def _parse_ingredients_text(self, ingr_ids_str: str) -> str:
        if not isinstance(ingr_ids_str, str) or ingr_ids_str == "":
            return ""
        ingr_ids = ingr_ids_str.split(";")
        names = [self.ingr_id_to_name.get(ingr_id, "") for ingr_id in ingr_ids]
        names = [name for name in names if name]
        return "; ".join(names)
    
    # аугментация массы:
    def _augment_mass_raw(self, mass: float) -> float:
        """Apply noise in grams, based on config logic."""
        if not self.augment_mass:
            return mass
        if random.random() > 0.8:  # 80% chance to augment
            return mass
        if mass <= 200:
            noise = random.uniform(-0.05, 0.05) * mass
        else:
            noise = random.uniform(-0.10, 0.10) * mass
        return mass + noise

    def __len__(self) -> int:
        return len(self.dish_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]: # type:ignore
        row = self.dish_df.iloc[idx]
        dish_id = row["dish_id"]

        # --- Image ---
        image_path = self.config.images_dir / str(dish_id) / "rgb.png"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.asarray(image, dtype=np.float32))
            image = image['image']

        # --- Text ---
        ingredients_raw = row["ingredients"]
        ingredients_text = self._parse_ingredients_text(ingredients_raw)
        encoded = self.tokenizer(
            ingredients_text,
            padding="max_length",
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # --- Labels (regression targets) ---
        calories = float(row["total_calories"])
        # Сразу нормализуем массу
        mass_aug = self._augment_mass_raw(float(row["total_mass"]))
        mass_norm = (float(row["total_mass"]) - self.mass_mean) / self.mass_std

        label = torch.tensor(calories, dtype=torch.float32)
        mass_tensor = torch.tensor(mass_norm, dtype=torch.float32)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,          # калории
            "mass": mass_tensor,     # масса (грамм)
            "dish_id": dish_id,
        }
    
    
    # =============== EDA / Visualization Methods ===============

    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Return raw (untransformed) data for EDA."""
        row = self.dish_df.iloc[idx]
        dish_id = row["dish_id"]

        image_path = self.config.images_dir / str(dish_id) / "rgb.png"
        image = Image.open(image_path).convert("RGB")
        ingredients_text = self._parse_ingredients_text(row["ingredients"])
        calories = float(row["total_calories"])
        mass = float(row["total_mass"])

        return {
            "image": image,
            "ingredients_text": ingredients_text,
            "calories": calories,
            "mass": mass,            # граммы
            "dish_id": dish_id,
        }

        
    def get_raw_sample_for_vis(self, idx: int) -> Dict[str, Any]:
        """For clarity in visualization context."""
        return self.get_raw_item(idx)

    def show_samples(
        self,
        idx_or_indices: Union[int, List[int]],
        figsize_per_sample: tuple = (12, 5),
    ) -> None:
        """Display samples with image, ingredients, calories, and mass."""
        if isinstance(idx_or_indices, int):
            idx_or_indices = [idx_or_indices]

        n = len(idx_or_indices)
        fig, axes = plt.subplots(n, 1, figsize=(figsize_per_sample[0], n * figsize_per_sample[1]))
        if n == 1:
            axes = [axes]

        for ax, idx in zip(axes, idx_or_indices):
            sample = self.get_raw_sample_for_vis(idx)
            image = sample["image"]
            text = sample["ingredients_text"] or "<no ingredients>"
            calories = sample["calories"]
            mass = sample["mass"]
            dish_id = sample["dish_id"]

            # Display image
            ax.imshow(image)
            ax.axis("off")

            # Add border
            from matplotlib.patches import Rectangle
            h, w = image.height, image.width
            rect = Rectangle((0, 0), w, h, linewidth=2, edgecolor='white', facecolor='none')
            ax.add_patch(rect)

            # Legend: ID, calories, mass
            legend_text = f"ID: {dish_id}\nCal: {calories:.1f} kcal\nMass: {mass:.0f} g"
            ax.text(
                10, 10, legend_text,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                transform=ax.transData
            )

            # Ingredients below image
            ax.text(
                0.5, 0, "Ingredients:",
                transform=ax.transAxes,
                fontsize=15,
                ha='center',
                va='top',
                wrap=True,
                linespacing=1.4
            )
            # Ingredients below image
            ax.text(
                0.5, -0.05, text,
                transform=ax.transAxes,
                fontsize=10,
                ha='center',
                va='top',
                wrap=True,
                linespacing=1.4
            )

        plt.tight_layout()
        plt.show()