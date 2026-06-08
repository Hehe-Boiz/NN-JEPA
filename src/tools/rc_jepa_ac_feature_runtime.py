"""Runtime helpers for RC JEPA-AC feature-cache evaluation and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from data import settings
from data.sequence_dataset import DEFAULT_AC_ACTION_COLUMNS, DEFAULT_AC_STATE_COLUMNS
from models.rc_jepa_ac import (
    DEFAULT_PREDICTOR_TYPE,
    DEFAULT_PREDICTOR_DEPTH,
    DEFAULT_PREDICTOR_DIM,
    DEFAULT_PREDICTOR_HEADS,
    build_ac_predictor,
)


DEFAULT_FEATURES_DIR = settings.PROCESSED_DATA_DIR / "features" / "vjepa2_1_vitb_384_ema_fp32"


@dataclass(frozen=True)
class FeaturePredictorConfig:
    state_columns: tuple[str, ...]
    action_columns: tuple[str, ...]
    raw_frames_per_sample: int
    sequence_stride: int
    auto_steps: int
    predictor_type: str
    predictor_dim: int
    predictor_depth: int
    predictor_heads: int
    dropout: float
    tokens_per_frame: int
    embed_dim: int
    feature_metadata: dict[str, Any]

    def to_jsonable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_columns"] = list(self.state_columns)
        payload["action_columns"] = list(self.action_columns)
        return payload


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        best_path = checkpoint_path / "best.pt"
        if best_path.exists():
            return best_path
        last_path = checkpoint_path / "last.pt"
        if last_path.exists():
            return last_path
        raise FileNotFoundError(f"No best.pt or last.pt found in checkpoint dir: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_feature_checkpoint(path: str | Path, device: torch.device) -> tuple[dict[str, Any], Path]:
    checkpoint_path = resolve_checkpoint_path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "predictor_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain predictor_state_dict: {checkpoint_path}")
    return checkpoint, checkpoint_path


def config_from_checkpoint(checkpoint: dict[str, Any]) -> FeaturePredictorConfig:
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_metadata = dict(checkpoint.get("feature_metadata", {}))
    if not feature_metadata:
        raise ValueError("Checkpoint is missing feature_metadata")

    state_columns = tuple(checkpoint.get("state_columns") or checkpoint_args.get("state_columns") or DEFAULT_AC_STATE_COLUMNS)
    action_columns = tuple(
        checkpoint.get("action_columns") or checkpoint_args.get("action_columns") or DEFAULT_AC_ACTION_COLUMNS
    )
    tokens_per_frame = int(feature_metadata.get("tokens_per_frame", 0))
    embed_dim = int(feature_metadata.get("embed_dim", 0))
    if tokens_per_frame <= 0 or embed_dim <= 0:
        raise ValueError(f"Invalid feature metadata tokens/embed dims: {feature_metadata}")

    return FeaturePredictorConfig(
        state_columns=state_columns,
        action_columns=action_columns,
        raw_frames_per_sample=int(checkpoint_args.get("raw_frames_per_sample", settings.AC_RAW_FRAMES_PER_SAMPLE)),
        sequence_stride=int(checkpoint_args.get("sequence_stride", settings.AC_SEQUENCE_STRIDE)),
        auto_steps=int(checkpoint_args.get("auto_steps", settings.AC_AUTO_STEPS)),
        predictor_type=str(checkpoint_args.get("predictor_type", DEFAULT_PREDICTOR_TYPE)),
        predictor_dim=int(checkpoint_args.get("predictor_dim", DEFAULT_PREDICTOR_DIM)),
        predictor_depth=int(checkpoint_args.get("predictor_depth", DEFAULT_PREDICTOR_DEPTH)),
        predictor_heads=int(checkpoint_args.get("predictor_heads", DEFAULT_PREDICTOR_HEADS)),
        dropout=float(checkpoint_args.get("dropout", 0.0)),
        tokens_per_frame=tokens_per_frame,
        embed_dim=embed_dim,
        feature_metadata=feature_metadata,
    )


def build_predictor_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, FeaturePredictorConfig]:
    config = config_from_checkpoint(checkpoint)
    predictor = build_ac_predictor(
        predictor_type=config.predictor_type,
        latent_dim=config.embed_dim,
        state_dim=len(config.state_columns),
        action_dim=len(config.action_columns),
        tokens_per_frame=config.tokens_per_frame,
        max_frames=config.raw_frames_per_sample,
        predictor_dim=config.predictor_dim,
        depth=config.predictor_depth,
        num_heads=config.predictor_heads,
        dropout=config.dropout,
    ).to(device)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    predictor.eval()
    return predictor, config


def checkpoint_default_path(checkpoint: dict[str, Any], key: str, fallback: Path) -> Path:
    checkpoint_args = checkpoint.get("args", {})
    value = checkpoint_args.get(key) if isinstance(checkpoint_args, dict) else None
    return Path(value) if value else fallback


def validate_feature_metadata(
    dataset_metadata: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> None:
    for key in ("tokens_per_frame", "embed_dim"):
        if int(dataset_metadata.get(key, -1)) != int(checkpoint_metadata.get(key, -2)):
            raise ValueError(
                f"Feature metadata mismatch for {key}: "
                f"dataset={dataset_metadata.get(key)} checkpoint={checkpoint_metadata.get(key)}"
            )
