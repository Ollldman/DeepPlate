from typing import Any, Callable, Dict, Union
import os

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import timm
from timm.models import PretrainedCfg
import albumentations as A
from torchvision.transforms import Compose

from DeepPlate import Config

class MultimodalDataset(Dataset):
    """A PyTorch Dataset for multimodal (text + image) classification tasks.

    This dataset loads image-text pairs from a pandas DataFrame, applies
    image transformations, and prepares raw text for tokenization
    (tokenization is deferred to the collate function).

    Expected DataFrame columns:
        - 'text' (str): Input text.
        - 'label' (int): Class label.
        - 'image_path' (str): Relative path to image file (relative to `dataset_root_path/images/`).

    Args:
        df (pd.DataFrame): Input DataFrame containing text, label, and image paths.
        text_model (str): Hugging Face model name for tokenizer (e.g., 'bert-base-uncased').
        image_model (str): TIMM model name to infer preprocessing config (e.g., 'resnet50').
        transforms (Union[A.Compose, Compose, Callable]): Image transformation pipeline.
            Supports both Albumentations and Torchvision transforms.
        dataset_root_path (str): Root directory of the dataset. Images are expected under
            `{dataset_root_path}/images/`.
    """

    def __init__(
        self,
        config: Config,
        transforms: Union[A.Compose, Compose, Callable[[np.ndarray], torch.Tensor]],
        dataset_type="train"
    ) -> None:
        if dataset_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
            self.root_path: str = config.TRAIN_DF_PATH
        else:
            self.df = pd.read_csv(config.VAL_DF_PATH)
            self.root_path: str = config.VAL_DF_PATH

        self.image_cfg: PretrainedCfg =\
              timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME) # type:ignore
        
        self.tokenizer: PreTrainedTokenizerBase =\
              AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and preprocesses a single image-text pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - 'text' (str): Raw input text.
                - 'label' (int): Ground truth class label.
                - 'image' (np.ndarray or torch.Tensor): Transformed image.
        """
        row = self.df.iloc[idx]
        text: str = row["text"]
        label: int = row["label"]
        

        # Load image
        img_path: str = row["image_path"]
        full_img_path = os.path.join(self.root_path, "images", img_path)
        image_pil = Image.open(full_img_path).convert("RGB")
        image_np = np.array(image_pil)

        # Apply transforms
        if isinstance(self.transforms, A.Compose):
            # Albumentations expects and returns numpy arrays
            transformed = self.transforms(image=image_np)
            image = transformed["image"]
        else:
            # Torchvision expects PIL or tensor; we pass numpy → it auto-converts
            image = self.transforms(image_np)

        return {
            "text": text,
            "label": label,
            "image": image,
        }

