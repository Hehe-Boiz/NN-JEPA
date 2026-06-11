"""Feature-cache Dataset/DataLoader helpers for RC JEPA-AC training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from . import settings
from .dataset import load_manifest
from .normalization import FeatureNormalizer, build_feature_normalizer
from .sequence_dataset import (
    DEFAULT_AC_ACTION_COLUMNS,
    DEFAULT_AC_STATE_COLUMNS,
    build_ac_action_normalizer,
    build_action_row,
    build_sequence_windows,
    timestamp_to_float,
)


FEATURE_METADATA_NAME = "metadata.json"
FEATURE_SESSIONS_DIR_NAME = "sessions"
EXPECTED_IMAGE_PATH_KEY = "source_frame_path"
EXPECTED_IMAGE_PATH_FALLBACK = True
SUPPORTED_FEATURE_SAMPLERS = ("global", "session")


class SessionBatchSampler(Sampler[list[int]]):
    """Yield batches whose windows all come from one session.

    This keeps the per-session feature memmap/cache hot while preserving the
    causal order inside each sampled window. Training can still randomize by
    shuffling both session order and window order within each session.
    """

    def __init__(
        self,
        window_session_ids: Sequence[str],
        batch_size: int,
        shuffle_sessions: bool,
        shuffle_windows: bool,
        drop_last: bool = False,
        seed: int = settings.RANDOM_SEED,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self.shuffle_sessions = bool(shuffle_sessions)
        self.shuffle_windows = bool(shuffle_windows)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.by_session: dict[str, list[int]] = {}
        for index, session_id in enumerate(window_session_ids):
            self.by_session.setdefault(str(session_id), []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        sessions = list(self.by_session)
        if self.shuffle_sessions:
            rng.shuffle(sessions)
        for session_id in sessions:
            rows = np.array(self.by_session[session_id], dtype=np.int64)
            if self.shuffle_windows:
                rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start : start + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for rows in self.by_session.values():
            if self.drop_last:
                total += len(rows) // self.batch_size
            else:
                total += (len(rows) + self.batch_size - 1) // self.batch_size
        return total


class RCJepaACFeatureSequenceDataset(Dataset):
    """Sequence dataset backed by precomputed V-JEPA frame features."""

    def __init__(
        self,
        split: str,
        features_dir: str | Path,
        manifest_path: str | Path | None = None,
        raw_frames_per_sample: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
        sequence_stride: int = settings.AC_SEQUENCE_STRIDE,
        frame_stride: int = settings.AC_FRAME_STRIDE,
        target_fps: float = settings.AC_TARGET_FPS,
        state_columns: Sequence[str] = DEFAULT_AC_STATE_COLUMNS,
        action_columns: Sequence[str] = DEFAULT_AC_ACTION_COLUMNS,
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
        self.features_dir = Path(features_dir)
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

        self.feature_metadata = load_feature_metadata(self.features_dir)
        self.tokens_per_frame = int(self.feature_metadata["tokens_per_frame"])
        self.embed_dim = int(self.feature_metadata["embed_dim"])

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
        self.window_session_ids = [
            self.window_session_id(window)
            for window in self.windows
        ]

        session_ids = {str(self.samples[index]["session_id"]) for window in self.windows for index in window}
        self.session_features = {
            session_id: load_session_feature_index(self.features_dir, session_id)
            for session_id in sorted(session_ids)
        }

    def __len__(self) -> int:
        return len(self.windows)

    def window_session_id(self, window: Sequence[int]) -> str:
        session_ids = {str(self.samples[sample_index]["session_id"]) for sample_index in window}
        if len(session_ids) != 1:
            raise ValueError(f"Sequence window crosses sessions: {sorted(session_ids)}")
        return next(iter(session_ids))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_indices = self.windows[index]
        sequence = [self.samples[sample_index] for sample_index in sample_indices]
        session_id = str(sequence[0]["session_id"])
        session_features = self.session_features[session_id]

        latent_frames = [
            session_features.get_frame(int(sample["frame_index"]))
            for sample in sequence
        ]
        latents = torch.stack(latent_frames, dim=0).reshape(
            self.raw_frames_per_sample * self.tokens_per_frame,
            self.embed_dim,
        )

        states = [dict(sample["state"]) for sample in sequence]
        actions = [build_action_row(sample, self.action_columns) for sample in sequence[:-1]]
        if self.state_normalizer is None:
            state_values = [[state[column] for column in self.state_columns] for state in states]
        else:
            state_values = [self.state_normalizer.normalize_row(state, self.state_columns) for state in states]

        if self.action_normalizer is None:
            action_values = [[action[column] for column in self.action_columns] for action in actions]
        else:
            action_values = [self.action_normalizer.normalize_row(action, self.action_columns) for action in actions]

        first_sample = sequence[0]
        last_sample = sequence[-1]
        return {
            "latents": latents,
            "states": torch.tensor(state_values, dtype=torch.float32),
            "actions": torch.tensor(action_values, dtype=torch.float32),
            "sample_id": f"{first_sample['sample_id']}__to__{last_sample['sample_id']}",
            "session_id": session_id,
            "data_domain": first_sample.get("data_domain", "unknown"),
            "frame_indices": torch.tensor([sample["frame_index"] for sample in sequence], dtype=torch.long),
            "timestamps_sec": torch.tensor(
                [timestamp_to_float(sample.get("timestamp_sec")) for sample in sequence],
                dtype=torch.float32,
            ),
        }


class SessionFeatureIndex:
    def __init__(self, feature_array: np.ndarray, frame_to_row: dict[int, int]) -> None:
        self.feature_array = feature_array
        self.frame_to_row = frame_to_row

    def get_frame(self, frame_index: int) -> torch.Tensor:
        if frame_index not in self.frame_to_row:
            raise KeyError(f"Missing cached feature for frame_index={frame_index}")
        row = self.frame_to_row[frame_index]
        frame = np.array(self.feature_array[row], copy=True)
        return torch.from_numpy(frame).to(dtype=torch.float32)


def load_feature_metadata(features_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(features_dir) / FEATURE_METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ("tokens_per_frame", "embed_dim", "dtype")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Feature metadata missing keys: {missing}")
    validate_feature_image_source(metadata, metadata_path)
    return metadata


def validate_feature_image_source(metadata: dict[str, Any], metadata_path: Path) -> None:
    """Prevent silent training from stale processed-image feature caches.

    Current V-JEPA feature extraction must read the original raw frame path and
    resize directly to the encoder size. Older caches did not record this field
    and were usually built from 224px processed images, so they are rejected.
    """
    missing = [
        key
        for key in ("image_path_key", "image_path_fallback")
        if key not in metadata
    ]
    if missing:
        raise ValueError(
            f"Feature metadata {metadata_path} is missing {missing}. "
            "This cache was likely created by an older extractor and may use "
            "processed 224px frames. Re-run `tools.extract_vjepa_features` with "
            "the current code, or choose a feature directory whose metadata has "
            f"image_path_key={EXPECTED_IMAGE_PATH_KEY!r}."
        )

    image_path_key = metadata.get("image_path_key")
    image_path_fallback = bool(metadata.get("image_path_fallback"))
    if image_path_key != EXPECTED_IMAGE_PATH_KEY or image_path_fallback != EXPECTED_IMAGE_PATH_FALLBACK:
        raise ValueError(
            f"Feature metadata {metadata_path} was built with "
            f"image_path_key={image_path_key!r}, image_path_fallback={image_path_fallback!r}. "
            f"Expected image_path_key={EXPECTED_IMAGE_PATH_KEY!r}, "
            f"image_path_fallback={EXPECTED_IMAGE_PATH_FALLBACK!r}. "
            "Use a matching feature cache or re-extract features."
        )


def load_session_feature_index(features_dir: str | Path, session_id: str) -> SessionFeatureIndex:
    sessions_dir = Path(features_dir) / FEATURE_SESSIONS_DIR_NAME
    npy_path = sessions_dir / f"{session_id}.npy"
    json_path = sessions_dir / f"{session_id}.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"Feature array not found for {session_id}: {npy_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Feature index not found for {session_id}: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    frame_to_row = {int(frame["frame_index"]): int(frame["row"]) for frame in frames}
    feature_array = np.load(npy_path, mmap_mode="r")
    if feature_array.ndim != 3:
        raise ValueError(f"Expected feature array [N, K, D], got {feature_array.shape} for {session_id}")
    return SessionFeatureIndex(feature_array=feature_array, frame_to_row=frame_to_row)


def create_ac_feature_sequence_dataloaders(
    features_dir: str | Path,
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
    include_test: bool = True,
    train_sampler: str = "global",
    eval_sampler: str = "global",
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
    validate_feature_sampler(train_sampler, name="train_sampler")
    validate_feature_sampler(eval_sampler, name="eval_sampler")
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

    splits = ("train", "val", "test") if include_test else ("train", "val")
    datasets = {
        split: RCJepaACFeatureSequenceDataset(
            split=split,
            features_dir=features_dir,
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
        for split in splits
    }

    dataloaders = {
        "train": build_feature_dataloader(
            dataset=datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle_sessions=settings.SHUFFLE_TRAIN,
            shuffle_windows=settings.SHUFFLE_TRAIN,
            loader_kwargs=loader_kwargs,
        ),
        "val": build_feature_dataloader(
            dataset=datasets["val"],
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            shuffle_sessions=False,
            shuffle_windows=False,
            loader_kwargs=loader_kwargs,
        ),
    }
    if include_test:
        dataloaders["test"] = build_feature_dataloader(
            dataset=datasets["test"],
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            shuffle_sessions=False,
            shuffle_windows=False,
            loader_kwargs=loader_kwargs,
        )
    return dataloaders


def validate_feature_sampler(value: str, name: str) -> None:
    if value not in SUPPORTED_FEATURE_SAMPLERS:
        raise ValueError(f"{name} must be one of {SUPPORTED_FEATURE_SAMPLERS}, got {value!r}")


def build_feature_dataloader(
    dataset: RCJepaACFeatureSequenceDataset,
    batch_size: int,
    sampler: str,
    shuffle_sessions: bool,
    shuffle_windows: bool,
    loader_kwargs: dict[str, Any],
) -> DataLoader:
    if sampler == "global":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_windows,
            **loader_kwargs,
        )
    batch_sampler = SessionBatchSampler(
        window_session_ids=dataset.window_session_ids,
        batch_size=batch_size,
        shuffle_sessions=shuffle_sessions,
        shuffle_windows=shuffle_windows,
        drop_last=False,
        seed=settings.RANDOM_SEED,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        **loader_kwargs,
    )
