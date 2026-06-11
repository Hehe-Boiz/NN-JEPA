"""Sequence Dataset/DataLoader helpers for RC JEPA-AC world model training."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import random
from typing import Any, Sequence
import warnings

from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset

from . import settings
from .dataset import HORIZONTAL_FLIP_SIGN_COLUMNS, image_to_tensor, load_manifest, normalize_tensor
from .normalization import FeatureNormalizer, build_feature_normalizer


DEFAULT_AC_STATE_COLUMNS = tuple(settings.AC_STATE_COLUMNS)
DEFAULT_AC_ACTION_COLUMNS = tuple(settings.AC_ACTION_COLUMNS)
DOMAIN_ID_ACTION_COLUMN = "domain_id"
DEFAULT_DOMAIN_ID = 1.0
DOMAIN_ID_BY_DATA_DOMAIN = {
    "old": 0.0,
    "old_servo": 0.0,
    "kds": 0.0,
    "kds_680hv": 0.0,
    "data servo cũ kds 680hv": 0.0,
    "current": 1.0,
    "current_servo": 1.0,
    "new_servo": 1.0,
    "towerpro": 1.0,
    "tower_pro": 1.0,
}
_WARNED_DOMAIN_KEYS: set[str] = set()


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
        frame_stride: int = settings.AC_FRAME_STRIDE,
        target_fps: float = settings.AC_TARGET_FPS,
        state_columns: Sequence[str] = DEFAULT_AC_STATE_COLUMNS,
        action_columns: Sequence[str] = DEFAULT_AC_ACTION_COLUMNS,
        augment: bool | None = None,
        state_normalizer: FeatureNormalizer | None = None,
        action_normalizer: FeatureNormalizer | None = None,
        max_frame_index_gap: int = settings.AC_MAX_FRAME_INDEX_GAP,
        max_time_gap_sec: float = settings.AC_MAX_TIME_GAP_SEC,
    ) -> None:
        if raw_frames_per_sample < 2:
            raise ValueError("raw_frames_per_sample must be >= 2")
        if sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if target_fps < 0:
            raise ValueError("target_fps must be >= 0")

        self.split = split
        self.manifest_path = Path(manifest_path or settings.MANIFEST_DIR / f"{split}.jsonl")
        self.raw_frames_per_sample = raw_frames_per_sample
        self.sequence_stride = sequence_stride
        self.frame_stride = frame_stride
        self.target_fps = target_fps
        self.state_columns = tuple(state_columns)
        self.action_columns = tuple(action_columns)
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.max_frame_index_gap = max_frame_index_gap
        self.max_time_gap_sec = max_time_gap_sec
        self.samples = load_manifest(self.manifest_path)
        self.windows = build_sequence_windows(
            self.samples,
            raw_frames_per_sample=self.raw_frames_per_sample,
            sequence_stride=self.sequence_stride,
            frame_stride=self.frame_stride,
            target_fps=self.target_fps,
            state_columns=self.state_columns,
            action_columns=self.action_columns,
            max_frame_index_gap=self.max_frame_index_gap,
            max_time_gap_sec=self.max_time_gap_sec,
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
        actions = [build_action_row(sample, self.action_columns) for sample in sequence[:-1]]
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
        if self.state_normalizer is None:
            state_values = [[state[column] for column in self.state_columns] for state in states]
        else:
            state_values = [self.state_normalizer.normalize_row(state, self.state_columns) for state in states]

        if self.action_normalizer is None:
            action_values = [[action[column] for column in self.action_columns] for action in actions]
        else:
            action_values = [self.action_normalizer.normalize_row(action, self.action_columns) for action in actions]

        states_tensor = torch.tensor(state_values, dtype=torch.float32)
        actions_tensor = torch.tensor(action_values, dtype=torch.float32)

        first_sample = sequence[0]
        last_sample = sequence[-1]
        return {
            "images": images_tensor,
            "states": states_tensor,
            "actions": actions_tensor,
            "sample_id": f"{first_sample['sample_id']}__to__{last_sample['sample_id']}",
            "session_id": first_sample["session_id"],
            "data_domain": first_sample.get("data_domain", "unknown"),
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
    frame_stride: int = settings.AC_FRAME_STRIDE,
    target_fps: float = settings.AC_TARGET_FPS,
    max_frame_index_gap: int = settings.AC_MAX_FRAME_INDEX_GAP,
    max_time_gap_sec: float = settings.AC_MAX_TIME_GAP_SEC,
) -> list[list[int]]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if target_fps < 0:
        raise ValueError("target_fps must be >= 0")

    session_to_indices: dict[str, list[int]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        if has_required_columns(sample, state_columns=state_columns, action_columns=action_columns):
            session_to_indices[str(sample["session_id"])].append(sample_index)

    windows: list[list[int]] = []
    for indices in session_to_indices.values():
        indices.sort(key=lambda sample_index: sample_sort_key(samples[sample_index]))
        effective_frame_stride = resolve_effective_frame_stride(
            samples=samples,
            indices=indices,
            frame_stride=frame_stride,
            target_fps=target_fps,
        )
        required_span = ((raw_frames_per_sample - 1) * effective_frame_stride) + 1
        if len(indices) < required_span:
            continue
        last_start = len(indices) - required_span
        for start in range(0, last_start + 1, sequence_stride):
            window = [
                indices[start + (offset * effective_frame_stride)]
                for offset in range(raw_frames_per_sample)
            ]
            if is_contiguous_window(
                samples,
                window,
                expected_frame_stride=effective_frame_stride,
                max_frame_index_gap=max_frame_index_gap,
                max_time_gap_sec=max_time_gap_sec,
            ):
                windows.append(window)
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
    return all(column in state for column in state_columns) and all(
        column == DOMAIN_ID_ACTION_COLUMN or column in action
        for column in action_columns
    )


def build_action_row(sample: dict[str, Any], action_columns: Sequence[str]) -> dict[str, float]:
    """Return an action row, including synthetic domain_id when requested."""
    action = dict(sample.get("action", {}))
    if DOMAIN_ID_ACTION_COLUMN in action_columns:
        action[DOMAIN_ID_ACTION_COLUMN] = sample_domain_id(sample)
    return action


def sample_domain_id(sample: dict[str, Any]) -> float:
    """Map manifest data_domain to the JEPA-style domain action token.

    The mixed JEPA data uses 0 for old/KDS servo and 1 for current/TowerPro
    servo. Older single-domain NN-JEPA manifests have no data_domain; those are
    treated as current-servo data for backward compatibility.
    """
    raw_value = sample.get("data_domain")
    raw_domain = str(raw_value or "").strip().lower()
    if not raw_domain:
        if "missing" not in _WARNED_DOMAIN_KEYS:
            warnings.warn(
                "Manifest sample is missing data_domain; using domain_id=1.0 for backward compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_DOMAIN_KEYS.add("missing")
        return DEFAULT_DOMAIN_ID
    if raw_domain not in DOMAIN_ID_BY_DATA_DOMAIN:
        if raw_domain not in _WARNED_DOMAIN_KEYS:
            warnings.warn(
                f"Unknown data_domain={raw_value!r}; using domain_id=1.0 fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_DOMAIN_KEYS.add(raw_domain)
        return DEFAULT_DOMAIN_ID
    return float(DOMAIN_ID_BY_DATA_DOMAIN[raw_domain])


def sample_sort_key(sample: dict[str, Any]) -> tuple[float, int]:
    timestamp = timestamp_to_float(sample.get("timestamp_sec"))
    if timestamp != timestamp:
        timestamp = float("inf")
    return timestamp, int(sample["frame_index"])


def is_contiguous_window(
    samples: list[dict[str, Any]],
    window: Sequence[int],
    expected_frame_stride: int,
    max_frame_index_gap: int,
    max_time_gap_sec: float,
) -> bool:
    expected_frame_stride = max(int(expected_frame_stride), 1)
    max_allowed_frame_gap = (
        expected_frame_stride + max(max_frame_index_gap - 1, 0)
        if max_frame_index_gap > 0
        else 0
    )
    max_allowed_time_gap = max_time_gap_sec * expected_frame_stride
    for left_index, right_index in zip(window, window[1:]):
        left = samples[left_index]
        right = samples[right_index]
        frame_gap = int(right["frame_index"]) - int(left["frame_index"])
        if max_frame_index_gap > 0 and frame_gap < expected_frame_stride:
            return False
        if max_frame_index_gap > 0 and frame_gap > max_allowed_frame_gap:
            return False

        left_t = timestamp_to_float(left.get("timestamp_sec"))
        right_t = timestamp_to_float(right.get("timestamp_sec"))
        if max_time_gap_sec > 0 and left_t == left_t and right_t == right_t:
            time_gap = right_t - left_t
            if time_gap <= 0 or time_gap > max_allowed_time_gap:
                return False
    return True


def resolve_effective_frame_stride(
    samples: list[dict[str, Any]],
    indices: Sequence[int],
    frame_stride: int,
    target_fps: float,
) -> int:
    if target_fps <= 0:
        return max(int(frame_stride), 1)

    source_fps = estimate_source_fps(samples, indices)
    if source_fps is None or source_fps <= 0:
        return max(int(frame_stride), 1)
    return max(1, int(math.ceil(source_fps / target_fps)))


def estimate_source_fps(samples: list[dict[str, Any]], indices: Sequence[int]) -> float | None:
    frame_periods: list[float] = []
    for left_index, right_index in zip(indices, indices[1:]):
        left = samples[left_index]
        right = samples[right_index]
        left_t = timestamp_to_float(left.get("timestamp_sec"))
        right_t = timestamp_to_float(right.get("timestamp_sec"))
        if left_t != left_t or right_t != right_t:
            continue
        frame_gap = int(right["frame_index"]) - int(left["frame_index"])
        time_gap = right_t - left_t
        if frame_gap <= 0 or time_gap <= 0:
            continue
        frame_periods.append(time_gap / frame_gap)
    if not frame_periods:
        return None
    period = median_float(frame_periods)
    if period <= 0:
        return None
    return 1.0 / period


def median_float(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def timestamp_to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def create_ac_sequence_dataloaders(
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    manifest_dir: str | Path | None = None,
    raw_frames_per_sample: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
    sequence_stride: int = settings.AC_SEQUENCE_STRIDE,
    frame_stride: int = settings.AC_FRAME_STRIDE,
    target_fps: float = settings.AC_TARGET_FPS,
    state_columns: Sequence[str] = DEFAULT_AC_STATE_COLUMNS,
    action_columns: Sequence[str] = DEFAULT_AC_ACTION_COLUMNS,
) -> dict[str, DataLoader]:
    batch_size = batch_size or settings.BATCH_SIZE
    eval_batch_size = eval_batch_size or settings.AC_EVAL_BATCH_SIZE
    num_workers = settings.NUM_WORKERS if num_workers is None else num_workers
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": settings.PIN_MEMORY,
        "persistent_workers": settings.PERSISTENT_WORKERS and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = settings.PREFETCH_FACTOR
    manifest_root = Path(manifest_dir or settings.MANIFEST_DIR)
    train_samples = load_manifest(manifest_root / "train.jsonl")
    state_normalizer = (
        build_feature_normalizer(train_samples, state_columns, source_key="state")
        if settings.NORMALIZE_STATE_INPUTS
        else None
    )
    action_normalizer = (
        build_ac_action_normalizer(train_samples, action_columns, state_normalizer)
        if settings.NORMALIZE_AC_ACTION_INPUTS
        else None
    )

    datasets = {
        split: RCJepaACSequenceDataset(
            split=split,
            manifest_path=manifest_root / f"{split}.jsonl",
            raw_frames_per_sample=raw_frames_per_sample,
            sequence_stride=sequence_stride,
            frame_stride=frame_stride,
            target_fps=target_fps,
            state_columns=state_columns,
            action_columns=action_columns,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        )
        for split in ("train", "val", "test")
    }

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=settings.SHUFFLE_TRAIN,
            **loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }


def build_ac_action_normalizer(
    train_samples: list[dict[str, Any]],
    action_columns: Sequence[str],
    state_normalizer: FeatureNormalizer | None,
) -> FeatureNormalizer:
    action_normalizer = build_feature_normalizer(train_samples, action_columns, source_key="action")
    if state_normalizer is None:
        return action_normalizer

    stats = dict(action_normalizer.stats)
    control_state_map = {
        "steering_cmd_t": "steering_last_t",
        "throttle_cmd_t": "throttle_last_t",
    }
    for action_column, state_column in control_state_map.items():
        if action_column in stats and state_column in state_normalizer.stats:
            stats[action_column] = state_normalizer.stats[state_column]
    return FeatureNormalizer(stats, clip_value=action_normalizer.clip_value)
