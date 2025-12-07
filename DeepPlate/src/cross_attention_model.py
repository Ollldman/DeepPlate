from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, PreTrainedModel
import timm

from DeepPlate import Config


class MassMLP(nn.Module):
    """Simple MLP to embed normalized mass into a vector space."""
    def __init__(self, hidden_dim: int = 128, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, mass: Tensor) -> Tensor:
        # mass: (B,) → (B, 1)
        return self.mlp(mass.unsqueeze(1))  # (B, out_dim)


class CalorieRegressionModel(nn.Module):
    """
    Multimodal calorie regression model with:
      - EfficientNet-like for images
      - BERT for ingredient text
      - MLP for normalized mass
      - Cross-attention (text queries image)
      - Final regressor head

    Outputs a single scalar: predicted calories.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.emb_dim = self.config.EMB_DIM 

        # ===== 1. Image backbone =====
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,  # remove classifier
            global_pool="avg"
        )
        img_feat_dim = self.image_model.num_features 

        # ===== 2. Text backbone =====
        self.text_model: PreTrainedModel = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        text_feat_dim = self.text_model.config.hidden_size  # 768 for BERT

        # ===== 3. Mass MLP =====
        self.mass_mlp = MassMLP(hidden_dim=config.MLP_HIDDEN_DIM, out_dim=self.emb_dim, dropout=config.MLP_DROPOUT)

        # ===== 4. Projection layers to common space =====
        self.image_proj = nn.Linear(img_feat_dim, self.emb_dim)
        self.text_proj = nn.Linear(text_feat_dim, self.emb_dim)

        # ===== 5. Cross-attention (text queries image) =====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=4,
            batch_first=False,  # expects (L, B, D)
            dropout=config.DROPOUT
        )

        # ===== 6. Regression head =====
        # Input: [attended_text_emb (256) + mass_emb (256)] → 512
        total_fusion_dim = self.emb_dim + self.emb_dim  # text+image attn + mass
        self.regressor = nn.Sequential(
            nn.Linear(total_fusion_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1)  # scalar output
        )

    def forward(
        self,
        image: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        mass: Tensor  # normalized mass (B,)
    ) -> Tensor:
        """
        Forward pass.

        Args:
            image: (B, C, H, W)
            input_ids: (B, L)
            attention_mask: (B, L)
            mass: (B,) — normalized

        Returns:
            calories: (B,) — predicted total calories
        """
        B = image.shape[0]

        # --- Image features ---
        img_feat = self.image_model(image)  # (B, img_feat_dim)
        img_emb = self.image_proj(img_feat)  # (B, emb_dim)

        # --- Text features ([CLS]) ---
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state[:, 0, :]  # (B, text_feat_dim)
        text_emb = self.text_proj(text_feat)  # (B, emb_dim)

        # --- Mass embedding ---
        mass_emb = self.mass_mlp(mass)  # (B, emb_dim)

        # --- Cross-attention (text queries image) ---
        # Reshape for MHA: (L=1, B, D)
        query = text_emb.unsqueeze(0)      # (1, B, emb_dim)
        key = value = img_emb.unsqueeze(0) # (1, B, emb_dim)

        attn_out, _ = self.cross_attn(query, key, value)  # (1, B, emb_dim)
        attn_out = attn_out.squeeze(0)    # (B, emb_dim)

        # --- Fusion + regression ---
        fused = torch.cat([attn_out, mass_emb], dim=1)  # (B, 2 * emb_dim)
        calories = self.regressor(fused).squeeze(-1)    # (B,)

        return calories