from typing import Tuple
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding
import timm
from DeepPlate import Config


class BaseMultimodalModel(nn.Module):
    """Base multimodal encoder that extracts and projects text and image features.

    This model wraps a pretrained text transformer (e.g., BERT) and a pretrained
    vision model (e.g., ResNet, EfficientNet, ViT) from TIMM. Both modalities are
    projected to a common embedding dimension for downstream fusion.

    Args:
        text_model_name (str): Name of the pretrained text model from Hugging Face.
        image_model_name (str): Name of the pretrained image model from TIMM.
        emb_dim (int): Target embedding dimension for both modalities after projection.
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "resnet50",
        emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        # Text encoder
        self.text_model: PreTrainedModel = AutoModel.from_pretrained(text_model_name)
        text_config: PretrainedConfig = self.text_model.config
        assert hasattr(text_config, "hidden_size"), (
            f"Text model config must have 'hidden_size'. Got: {text_config}"
        )

        # Image encoder
        self.image_model = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0,  # Remove classifier head
            global_pool="avg",  # Ensure global pooling is applied
        )
        assert hasattr(self.image_model, "num_features"), (
            f"TIMM model must expose 'num_features'. Got: {type(self.image_model)}"
        )
        self.hidden_size = text_config.hidden_size
        self.num_features = self.image_model.num_features
        if isinstance(self.hidden_size, int) and isinstance(self.num_features, int): 
            # Projection layers
            self.text_proj = nn.Linear(self.hidden_size, emb_dim)
            self.image_proj = nn.Linear( self.num_features, emb_dim)
        else:
            raise TypeError(f"Hidden_size from text_model and Num_fetures from image_model should be int type, got: {type(self.hidden_size)}, and {type(self.num_features)}")

    def forward(
        self,
        text_input: BatchEncoding,
        image_input: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Extracts and projects text and image embeddings.

        Args:
            text_input (BatchEncoding): Tokenized text batch from Hugging Face tokenizer.
                Must contain 'input_ids', 'attention_mask', etc.
            image_input (Tensor): Batch of images with shape (B, C, H, W).

        Returns:
            Tuple[Tensor, Tensor]:
                - text_emb: Projected text embeddings of shape (B, emb_dim).
                - image_emb: Projected image embeddings of shape (B, emb_dim).
        """
        # Extract [CLS] token for text (common in BERT-style models)
        text_features: Tensor = self.text_model(**text_input).last_hidden_state[:, 0, :]  # (B, H_text)

        # Extract global image features
        image_features: Tensor = self.image_model(image_input)  # (B, H_img)

        # Project to common embedding space
        text_emb: Tensor = self.text_proj(text_features)  # (B, emb_dim)
        image_emb: Tensor = self.image_proj(image_features)  # (B, emb_dim)

        return text_emb, image_emb


class CrossAttentionModel(nn.Module):
    """Multimodal classifier using cross-attention between text and image embeddings.

    Uses a base multimodal encoder to extract features, then applies cross-attention
    (text as query, image as key/value), followed by a classification head.

    Args:
        text_model_name (str): Pretrained text model name (Hugging Face).
        image_model_name (str): Pretrained image model name (TIMM).
        num_classes (int): Number of output classes for classification.
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "resnet50",
        num_classes: int = 2,
        num_heads: int = 4
    ) -> None:
        super().__init__()
        self.base_model = BaseMultimodalModel(text_model_name, image_model_name)
        emb_dim: int = self.base_model.emb_dim

        # Cross-attention layer (text queries image)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=False,  # PyTorch MHA expects (L, B, E) by default
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(emb_dim // 2, num_classes),
        )

    def forward(
        self,
        text_input: BatchEncoding,
        image_input: Tensor,
    ) -> Tensor:
        """Forward pass for multimodal classification.

        Args:
            text_input (BatchEncoding): Tokenized text inputs.
            image_input (Tensor): Image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Classification logits of shape (B, num_classes).
        """
        text_emb, image_emb = self.base_model(text_input, image_input)

        # Reshape for MultiheadAttention: (L=1, B, E)
        text_emb = text_emb.unsqueeze(0)    # (1, B, emb_dim)
        image_emb = image_emb.unsqueeze(0)  # (1, B, emb_dim)

        # Cross-attention: text queries image context
        attended_emb, _ = self.cross_attn(
            query=text_emb,
            key=image_emb,
            value=image_emb,
            need_weights=False,
        )  # (1, B, emb_dim)

        # Squeeze sequence dimension and classify
        attended_emb = attended_emb.squeeze(0)  # (B, emb_dim)
        logits: Tensor = self.classifier(attended_emb)  # (B, num_classes)

        return logits