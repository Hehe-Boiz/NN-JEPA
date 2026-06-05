"""Simple Dataset/DataLoader helpers for JEPA driving data."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset

from . import settings

HORIZONTAL_FLIP_SIGN_COLUMNS = {
    "yaw_rate_t": -1.0,
    "accel_y_t": -1.0,
    "steering_last_t": -1.0,
    "steering_cmd_t": -1.0,
}


class TrainAugmentor:
    """Photometric augmentations that do not destroy control semantics."""

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        image: Image.Image,
        state: dict[str, float],
        action: dict[str, float],
    ) -> tuple[Image.Image, dict[str, float], dict[str, float]]:
        augmented = image
        next_state = dict(state)
        next_action = dict(action)

        if settings.HORIZONTAL_FLIP_PROB > 0.0 and random.random() < settings.HORIZONTAL_FLIP_PROB:
            augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for column, sign in HORIZONTAL_FLIP_SIGN_COLUMNS.items():
                if column in next_state:
                    next_state[column] *= sign
                if column in next_action:
                    next_action[column] *= sign

        if settings.BRIGHTNESS_JITTER > 0:
            factor = random.uniform(1.0 - settings.BRIGHTNESS_JITTER, 1.0 + settings.BRIGHTNESS_JITTER)
            augmented = ImageEnhance.Brightness(augmented).enhance(factor)

        if settings.CONTRAST_JITTER > 0:
            factor = random.uniform(1.0 - settings.CONTRAST_JITTER, 1.0 + settings.CONTRAST_JITTER)
            augmented = ImageEnhance.Contrast(augmented).enhance(factor)

        if settings.SATURATION_JITTER > 0:
            factor = random.uniform(1.0 - settings.SATURATION_JITTER, 1.0 + settings.SATURATION_JITTER)
            augmented = ImageEnhance.Color(augmented).enhance(factor)

        if settings.GAUSSIAN_BLUR_PROB > 0.0 and random.random() < settings.GAUSSIAN_BLUR_PROB:
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=settings.GAUSSIAN_BLUR_RADIUS))

        return augmented, next_state, next_action


class DrivingJEPADataset(Dataset):
    """Single-step dataset yielding image, state, and action."""

    def __init__(
        self,
        split: str,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.split = split
        self.manifest_path = Path(manifest_path or settings.MANIFEST_DIR / f"{split}.jsonl")
        self.samples = load_manifest(self.manifest_path)
        self.augmentor = TrainAugmentor() if split == "train" else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample["frame_path"]) as image:
            rgb = image.convert("RGB")

        state = dict(sample["state"])
        action = dict(sample["action"])
        if self.augmentor is not None:
            rgb, state, action = self.augmentor(rgb, state, action)

        image_tensor = image_to_tensor(rgb)
        image_tensor = normalize_tensor(
            image_tensor,
            mean=list(settings.NORMALIZE_MEAN),
            std=list(settings.NORMALIZE_STD),
        )
        state_tensor = torch.tensor(
            [state[column] for column in settings.STATE_COLUMNS],
            dtype=torch.float32,
        )
        action_tensor = torch.tensor(
            [action[column] for column in settings.ACTION_COLUMNS],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "state": state_tensor,
            "action": action_tensor,
            "sample_id": sample["sample_id"],
            "session_id": sample["session_id"],
            "frame_index": sample["frame_index"],
            "timestamp_sec": sample["timestamp_sec"],
        }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    samples: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    np = _require_numpy()
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def normalize_tensor(
    tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def create_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, DataLoader]:
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = settings.NUM_WORKERS if num_workers is None else num_workers

    datasets = {
        split: DrivingJEPADataset(
            split=split,
            manifest_path=settings.MANIFEST_DIR / f"{split}.jsonl",
        )
        for split in ("train", "val", "test")
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=settings.SHUFFLE_TRAIN,
            num_workers=num_workers,
            pin_memory=settings.PIN_MEMORY,
            persistent_workers=settings.PERSISTENT_WORKERS and num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=settings.PIN_MEMORY,
            persistent_workers=settings.PERSISTENT_WORKERS and num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=settings.PIN_MEMORY,
            persistent_workers=settings.PERSISTENT_WORKERS and num_workers > 0,
        ),
    }
    return dataloaders


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Image tensor conversion requires numpy. Install dependencies with `pip install -e .`."
        ) from exc
    return np
