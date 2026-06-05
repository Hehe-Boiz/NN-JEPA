"""Simple image + sensor policy for RC car driving."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from data import settings


DEFAULT_SENSOR_NAMES = tuple(settings.STATE_COLUMNS)


def build_sensor_indices(sensor_names: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for name in sensor_names:
        if name not in settings.STATE_COLUMNS:
            raise ValueError(f"Unknown sensor name: {name}")
        indices.append(settings.STATE_COLUMNS.index(name))
    return indices


def select_sensor_features(state: torch.Tensor, sensor_indices: Sequence[int]) -> torch.Tensor:
    index_tensor = torch.tensor(sensor_indices, device=state.device, dtype=torch.long)
    return state.index_select(dim=1, index=index_tensor)


class SmallImageEncoder(nn.Module):
    """Lightweight CNN baseline for single-frame driving."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(256, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = x.flatten(1)
        return self.projection(x)


class VJepa2ImageEncoder(nn.Module):
    """Thin wrapper around the local V-JEPA 2.1 ViT encoder."""

    def __init__(
        self,
        variant: str = "vit_base_384",
        checkpoint_path: str | Path | None = None,
        freeze_encoder: bool = False,
        output_dim: int = 512,
        image_size: int = 384,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.encoder = self._build_encoder(variant)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.projection = nn.Linear(self.encoder.embed_dim, output_dim)

    def _build_encoder(self, variant: str) -> nn.Module:
        vjepa_root = Path(__file__).resolve().parents[2] / "vjepa2"
        if not vjepa_root.exists():
            raise FileNotFoundError(f"Missing local vjepa2 repo: {vjepa_root}")

        if str(vjepa_root) not in sys.path:
            sys.path.insert(0, str(vjepa_root))

        from app.vjepa_2_1.models import vision_transformer as vjepa_vit

        builders = {
            "vit_base_384": vjepa_vit.vit_base,
            "vit_large_384": vjepa_vit.vit_large,
        }
        if variant not in builders:
            raise ValueError(f"Unsupported V-JEPA encoder variant: {variant}")

        return builders[variant](
            img_size=(self.image_size, self.image_size),
            num_frames=1,
            tubelet_size=1,
            use_rope=True,
            img_temporal_dim_size=1,
        )

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "ema_encoder" in checkpoint:
                state_dict = checkpoint["ema_encoder"]
            elif "target_encoder" in checkpoint:
                state_dict = checkpoint["target_encoder"]
            elif "encoder" in checkpoint:
                state_dict = checkpoint["encoder"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            key = key.replace("backbone.", "")
            cleaned_state_dict[key] = value

        self.encoder.load_state_dict(cleaned_state_dict, strict=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        tokens = self.encoder(image)
        pooled = tokens.mean(dim=1)
        return self.projection(pooled)


class RCDrivingModel(nn.Module):
    """Predict steering/throttle from one image and a small sensor vector."""

    def __init__(
        self,
        sensor_names: Sequence[str] = DEFAULT_SENSOR_NAMES,
        image_backbone: str = "small_cnn",
        image_feature_dim: int = 256,
        sensor_feature_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        vjepa_checkpoint_path: str | Path | None = None,
        freeze_image_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.sensor_names = tuple(sensor_names)
        self.sensor_indices = build_sensor_indices(self.sensor_names)

        if image_backbone == "small_cnn":
            self.image_encoder = SmallImageEncoder(output_dim=image_feature_dim)
        elif image_backbone == "vjepa2_1_vitb":
            self.image_encoder = VJepa2ImageEncoder(
                variant="vit_base_384",
                checkpoint_path=vjepa_checkpoint_path,
                freeze_encoder=freeze_image_encoder,
                output_dim=image_feature_dim,
                image_size=384,
            )
        elif image_backbone == "vjepa2_1_vitl":
            self.image_encoder = VJepa2ImageEncoder(
                variant="vit_large_384",
                checkpoint_path=vjepa_checkpoint_path,
                freeze_encoder=freeze_image_encoder,
                output_dim=image_feature_dim,
                image_size=384,
            )
        else:
            raise ValueError(f"Unsupported image backbone: {image_backbone}")

        self.sensor_encoder = nn.Sequential(
            nn.Linear(len(self.sensor_indices), sensor_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sensor_feature_dim, sensor_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(image_feature_dim + sensor_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(settings.ACTION_COLUMNS)),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_feature = self.image_encoder(image)
        sensor_input = select_sensor_features(state, self.sensor_indices)
        sensor_feature = self.sensor_encoder(sensor_input)
        fused = torch.cat([image_feature, sensor_feature], dim=1)
        return self.policy_head(fused)
