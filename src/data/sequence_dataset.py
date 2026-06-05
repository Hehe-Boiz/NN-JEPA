"""Sequence Dataset/DataLoader helpers for RC JEPA-AC world model training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Sequence

from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset

from . import settings
from .dataset import HORIZONTAL_FLIP_SIGN_COLUMNS, image_to_tensor, load_manifest, normalize_tensor


DEFAULT_AC_STATE_COLUMNS = tuple(settings.AC_STATE_COLUMNS)
DEFAULT_AC_ACTION_COLUMNS = tuple(settings.AC_ACTION_COLUMNS)


class SequenceAugmentor:
    """Shared photometric augmentation for all frames in a sequence."""

    def __call__(
        self,
        images: list[Image.Image],
        states: list[dict[str, float]],
        actions: list[dict[str, float]],
    ) -> tuple[list[Image.Image], list[dict[str, float]], list[dict[str, float]]]:
        next_images = list(images)
        next_states = [dict(state) for state in states]
        next_actions = [dict(action) for action in actions]

        if settings.HORIZONTAL_FLIP_PROB > 0.0 and random.random() < settings.HORIZONTAL_FLIP_PROB:
            next_images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in next_images]
            for column, sign in HORIZONTAL_FLIP_SIGN_COLUMNS.items():
                for state in next_states:
                    if column in state:
                        state[column] *= sign
                for action in next_actions:
                    if column in action:
                        action[column] *= sign

        if settings.BRIGHTNESS_JITTER > 0:
            factor = random.uniform(1.0 - settings.BRIGHTNESS_JITTER, 1.0 + settings.BRIGHTNESS_JITTER)
            next_images = [ImageEnhance.Brightness(image).enhance(factor) for image in next_images]

        if settings.CONTRAST_JITTER > 0:
            factor = random.uniform(1.0 - settings.CONTRAST_JITTER, 1.0 + settings.CONTRAST_JITTER)
            next_images = [ImageEnhance.Contrast(image).enhance(factor) for image in next_images]

        if settings.SATURATION_JITTER > 0:
            factor = random.uniform(1.0 - settings.SATURATION_JITTER, 1.0 + settings.SATURATION_JITTER)
            next_images = [ImageEnhance.Color(image).enhance(factor) for image in next_images]

        if settings.GAUSSIAN_BLUR_PROB > 0.0 and random.random() < settings.GAUSSIAN_BLUR_PROB:
            next_images = [
                image.filter(ImageFilter.GaussianBlur(radius=settings.GAUSSIAN_BLUR_RADIUS))
                for image in next_images
            ]

        return next_images, next_states, next_actions


class RCJepaACSequenceDataset(Dataset):
    """Sequence dataset for frozen-encoder action-conditioned JEPA training."""

    def __init__(
        self,
        split: str,
        manifest_path: str | Path | None = None,
        raw_frames_per_sample: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
        sequence_stride: int = settings.AC_SEQUENCE_STRIDE,
        state_columns: Sequence[str] = DEFAULT_AC_STATE_COLUMNS,
        action_columns: Sequence[str] = DEFAULT_AC_ACTION_COLUMNS,
        augment: bool | None = None,
    ) -> None:
        if raw_frames_per_sample < 2:
            raise ValueError("raw_frames_per_sample must be >= 2")
        if sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1")

        self.split = split
        self.manifest_path = Path(manifest_path or settings.MANIFEST_DIR / f"{split}.jsonl")
        self.raw_frames_per_sample = raw_frames_per_sample
        self.sequence_stride = sequence_stride
        self.state_columns = tuple(state_columns)
        self.action_columns = tuple(action_columns)
        self.samples = load_manifest(self.manifest_path)
        self.windows = build_sequence_windows(
            self.samples,
            raw_frames_per_sample=self.raw_frames_per_sample,
            sequence_stride=self.sequence_stride,
            state_columns=self.state_columns,
            action_columns=self.action_columns,
        )
        use_augment = split == "train" if augment is None else augment
        self.augmentor = SequenceAugmentor() if use_augment else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_indices = self.windows[index]
        sequence = [self.samples[sample_index] for sample_index in sample_indices]

        images: list[Image.Image] = []
        for sample in sequence:
            with Image.open(sample["frame_path"]) as image:
                images.append(image.convert("RGB"))

        states = [dict(sample["state"]) for sample in sequence]
        actions = [dict(sample["action"]) for sample in sequence[:-1]]
        if self.augmentor is not None:
            images, states, actions = self.augmentor(images, states, actions)

        image_tensors = [
            normalize_tensor(
                image_to_tensor(image),
                mean=list(settings.NORMALIZE_MEAN),
                std=list(settings.NORMALIZE_STD),
            )
            for image in images
        ]
        images_tensor = torch.stack(image_tensors, dim=1).contiguous()
        states_tensor = torch.tensor(
            [[state[column] for column in self.state_columns] for state in states],
            dtype=torch.float32,
        )
        actions_tensor = torch.tensor(
            [[action[column] for column in self.action_columns] for action in actions],
            dtype=torch.float32,
        )

        first_sample = sequence[0]
        last_sample = sequence[-1]
        return {
            "images": images_tensor,
            "states": states_tensor,
            "actions": actions_tensor,
            "sample_id": f"{first_sample['sample_id']}__to__{last_sample['sample_id']}",
            "session_id": first_sample["session_id"],
            "frame_indices": torch.tensor([sample["frame_index"] for sample in sequence], dtype=torch.long),
            "timestamps_sec": torch.tensor(
                [timestamp_to_float(sample.get("timestamp_sec")) for sample in sequence],
                dtype=torch.float32,
            ),
        }


def build_sequence_windows(
    samples: list[dict[str, Any]],
    raw_frames_per_sample: int,
    sequence_stride: int,
    state_columns: Sequence[str],
    action_columns: Sequence[str],
) -> list[list[int]]:
    session_to_indices: dict[str, list[int]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        if has_required_columns(sample, state_columns=state_columns, action_columns=action_columns):
            session_to_indices[str(sample["session_id"])].append(sample_index)

    windows: list[list[int]] = []
    for indices in session_to_indices.values():
        indices.sort(key=lambda sample_index: sample_sort_key(samples[sample_index]))
        if len(indices) < raw_frames_per_sample:
            continue
        last_start = len(indices) - raw_frames_per_sample
        for start in range(0, last_start + 1, sequence_stride):
            windows.append(indices[start : start + raw_frames_per_sample])
    return windows


def has_required_columns(
    sample: dict[str, Any],
    state_columns: Sequence[str],
    action_columns: Sequence[str],
) -> bool:
    state = sample.get("state")
    action = sample.get("action")
    if not isinstance(state, dict) or not isinstance(action, dict):
        return False
    return all(column in state for column in state_columns) and all(column in action for column in action_columns)


def sample_sort_key(sample: dict[str, Any]) -> tuple[float, int]:
    timestamp = timestamp_to_float(sample.get("timestamp_sec"))
    if timestamp != timestamp:
        timestamp = float("inf")
    return timestamp, int(sample["frame_index"])


def timestamp_to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def create_ac_sequence_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
    manifest_dir: str | Path | None = None,
    raw_frames_per_sample: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
    sequence_stride: int = settings.AC_SEQUENCE_STRIDE,
    state_columns: Sequence[str] = DEFAULT_AC_STATE_COLUMNS,
    action_columns: Sequence[str] = DEFAULT_AC_ACTION_COLUMNS,
) -> dict[str, DataLoader]:
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = settings.NUM_WORKERS if num_workers is None else num_workers
    manifest_root = Path(manifest_dir or settings.MANIFEST_DIR)

    datasets = {
        split: RCJepaACSequenceDataset(
            split=split,
            manifest_path=manifest_root / f"{split}.jsonl",
            raw_frames_per_sample=raw_frames_per_sample,
            sequence_stride=sequence_stride,
            state_columns=state_columns,
            action_columns=action_columns,
        )
        for split in ("train", "val", "test")
    }

    return {
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
