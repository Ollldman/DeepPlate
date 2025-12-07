from pydantic import BaseModel, Field

class Config(BaseModel):
    # Модели
    TEXT_MODEL_NAME: str = "bert-base-uncased"
    IMAGE_MODEL_NAME: str = "resnet50"
    
    # Какие слои размораживаем
    TEXT_MODEL_UNFREEZE: str = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE: str = "layer.3|layer.4"
    
    # Гиперпараметры
    BATCH_SIZE: int = Field(default=32, ge=1)
    TEXT_LR: float = 3e-5
    IMAGE_LR: float = 1e-4
    CLASSIFIER_LR: float = 5e-4
    EPOCHS: int = Field(default=10, ge=1)
    DROPOUT: float = Field(default=0.3, ge=0.0, le=1.0)
    HIDDEN_DIM: int = 256
    NUM_CLASSES: int = Field(default=5, ge=1)
    
    # Пути
    TRAIN_DF_PATH: str = "path/train.csv"
    VAL_DF_PATH: str = "path/val.csv"
    SAVE_PATH: str = "best_model.pth"

    class Config:
        frozen = True  # делает объект иммутабельным