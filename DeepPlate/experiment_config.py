from pydantic import BaseModel, Field
from pathlib import Path
from typing import Tuple, List

class Config(BaseModel):
    # --- Dataset paths ---
    data_root: Path = Field(default=Path("data"))

    @property
    def dish_csv_path(self) -> Path:
        return self.data_root / "dish.csv"

    @property
    def ingredients_csv_path(self) -> Path:
        return self.data_root / "ingredients.csv"

    @property
    def images_dir(self) -> Path:
        return self.data_root / "images"

    # --- Model backbones ---
    TEXT_MODEL_NAME: str = "bert-base-uncased"
    IMAGE_MODEL_NAME: str = "efficientnet_b3"
    
    IMAGE_SIZE: Tuple[int, int] = (224, 224)  # стандарт для ResNet/EfficientNet
    IMAGE_MEAN: List[float] = [0.485, 0.456, 0.406]  # ImageNet stats
    IMAGE_STD: List[float] = [0.229, 0.224, 0.225]

    # --- Tokenization ---
    MAX_LENGTH: int = 96

    # --- Fine-tuning strategy ---
    TEXT_MODEL_UNFREEZE: str = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE: str = "blocks.5|blocks.6"
    EMB_DIM: int = 256
    
    # --- MLP head ---
    MLP_HIDDEN_DIM: int = 128
    MLP_OUT_DIM: int = 256
    MLP_DROPOUT: float = 0.1
    
    # --- Regression head  ---
    HIDDEN_DIM: int = 512  # размер скрытого слоя перед выходом
    DROPOUT: float = 0.3  # для регуляризации регрессии

    # --- Training hyperparameters ---
    BATCH_SIZE: int = Field(default=16, ge=1) # можно 16 если мало памяти
    EPOCHS: int = Field(default=2, ge=1)
    SEED: int = 2025
    NUM_WORKERS: int = 4
    

    
    # --- Validation ---
    VAL_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 10  # для регрессии

    # --- Per-component learning rates ---
    TEXT_LR: float = 1e-5
    IMAGE_LR: float = 5e-5
    FUSION_LR: float = 1e-4  # for cross-attention / fusion head
    HEAD_LR: float = 5e-4    # for final regression head (if separate)

    # --- Output ---
    @property
    def model_save_path(self) -> Path:
        return Path("models") / "best_model.pth"

    class Config:
        arbitrary_types_allowed = True