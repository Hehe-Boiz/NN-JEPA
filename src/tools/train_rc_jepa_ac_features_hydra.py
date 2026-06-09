"""Hydra entrypoint for training RC JEPA-AC predictor from feature cache."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tools import train_rc_jepa_ac_features as train_cli


CONFIG_PATH = "../../configs/hydra"
CONFIG_NAME = "config"


def build_train_args(cfg: Any) -> Namespace:
    """Convert a Hydra config into the Namespace expected by the CLI trainer."""
    data = to_plain_dict(cfg)
    args = train_cli.parse_args([])

    data_cfg = require_mapping(data, "data")
    model_cfg = require_mapping(data, "model")
    train_cfg = require_mapping(data, "train")
    wandb_cfg = require_mapping(data, "wandb")

    args.features_dir = path_value(data_cfg.get("features_dir", args.features_dir))
    args.manifest_dir = path_value(data_cfg.get("manifest_dir", args.manifest_dir))
    args.state_columns = list(data_cfg.get("state_columns", args.state_columns))
    args.action_columns = list(data_cfg.get("action_columns", args.action_columns))
    args.raw_frames_per_sample = int(data_cfg.get("raw_frames_per_sample", args.raw_frames_per_sample))
    args.sequence_stride = int(data_cfg.get("sequence_stride", args.sequence_stride))
    args.frame_stride = int(data_cfg.get("frame_stride", args.frame_stride))
    args.target_fps = float(data_cfg.get("target_fps", args.target_fps))
    args.auto_steps = int(data_cfg.get("auto_steps", args.auto_steps))

    args.predictor_type = str(model_cfg.get("type", args.predictor_type))
    args.model_size = str(model_cfg.get("size", args.model_size))
    args.predictor_dim = optional_int(model_cfg.get("predictor_dim", args.predictor_dim))
    args.predictor_depth = optional_int(model_cfg.get("predictor_depth", args.predictor_depth))
    args.predictor_heads = optional_int(model_cfg.get("predictor_heads", args.predictor_heads))
    args.dropout = float(model_cfg.get("dropout", args.dropout))

    args.epochs = int(train_cfg.get("epochs", args.epochs))
    args.batch_size = int(train_cfg.get("batch_size", args.batch_size))
    args.eval_batch_size = int(train_cfg.get("eval_batch_size", args.eval_batch_size))
    args.num_workers = int(train_cfg.get("num_workers", args.num_workers))
    args.lr = float(train_cfg.get("lr", args.lr))
    args.weight_decay = float(train_cfg.get("weight_decay", args.weight_decay))
    args.grad_clip = float(train_cfg.get("grad_clip", args.grad_clip))
    args.warmup_epochs = int(train_cfg.get("warmup_epochs", args.warmup_epochs))
    args.warmup_start_factor = float(train_cfg.get("warmup_start_factor", args.warmup_start_factor))
    args.min_lr_ratio = float(train_cfg.get("min_lr_ratio", args.min_lr_ratio))
    args.early_stopping_patience = int(train_cfg.get("early_stopping_patience", args.early_stopping_patience))
    args.resume_from = optional_path(train_cfg.get("resume_from", args.resume_from))
    args.seed = int(train_cfg.get("seed", args.seed))
    device_value = str(train_cfg.get("device", args.device))
    args.device = train_cli.default_device() if device_value == "auto" else device_value
    args.no_progress = bool(train_cfg.get("no_progress", args.no_progress))

    output_dir = data.get("output_dir", args.output_dir)
    args.output_dir = path_value(output_dir)
    args._output_dir_was_provided = output_dir is not None

    args.no_wandb = bool(wandb_cfg.get("disabled", args.no_wandb))
    args.wandb_project = optional_str(wandb_cfg.get("project", args.wandb_project))
    args.wandb_entity = optional_str(wandb_cfg.get("entity", args.wandb_entity))
    args.wandb_run_name = optional_str(wandb_cfg.get("run_name", args.wandb_run_name))
    args.wandb_run_id = optional_str(wandb_cfg.get("run_id", args.wandb_run_id))
    args.wandb_continue_run = bool(wandb_cfg.get("continue_run", args.wandb_continue_run))
    args.wandb_resume = str(wandb_cfg.get("resume", args.wandb_resume))
    args.wandb_mode = str(wandb_cfg.get("mode", args.wandb_mode))
    args.wandb_tags = list(wandb_cfg.get("tags", args.wandb_tags))
    args.wandb_log_every = int(wandb_cfg.get("log_every", args.wandb_log_every))
    args.wandb_watch_log = str(wandb_cfg.get("watch_log", args.wandb_watch_log))
    args.wandb_watch_freq = int(wandb_cfg.get("watch_freq", args.wandb_watch_freq))
    args.wandb_grad_stats_every = int(wandb_cfg.get("grad_stats_every", args.wandb_grad_stats_every))
    args.wandb_param_stats_every = int(wandb_cfg.get("param_stats_every", args.wandb_param_stats_every))
    return args


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        if isinstance(cfg, Mapping):
            return {str(key): to_plain_value(value) for key, value in cfg.items()}
        raise RuntimeError("Hydra/OmegaConf is required. Install with `pip install hydra-core`.") from None

    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

    if isinstance(cfg, Mapping):
        return {str(key): to_plain_value(value) for key, value in cfg.items()}
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


def to_plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_plain_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain_value(child) for child in value]
    return value


def require_mapping(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Hydra config must contain a `{key}` mapping")
    return dict(value)


def path_value(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return path_value(value)


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def hydra_entrypoint(cfg: Any) -> None:
    data = to_plain_dict(cfg)
    args = build_train_args(cfg)
    runtime_cfg = require_mapping(data, "runtime")
    if bool(runtime_cfg.get("dry_run", False)):
        train_cli.apply_predictor_size_preset(args)
        print(json.dumps(train_cli.args_to_jsonable_dict(args), indent=2), flush=True)
        return
    if bool(runtime_cfg.get("require_cuda", False)) and args.device != "cuda":
        raise RuntimeError(
            "Hydra config requires CUDA, but CUDA is not available. "
            "Check NVIDIA driver with `nvidia-smi`, or override "
            "`runtime.require_cuda=false train.device=cpu` for CPU-only debugging."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`train.device=cuda` was requested, but torch.cuda.is_available() is False.")
    train_cli.main(args)


def run() -> None:
    try:
        import hydra
    except ImportError:
        raise RuntimeError(
            "hydra-core is not installed. Run `pip install -e .` or `pip install hydra-core` first."
        ) from None

    hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)(hydra_entrypoint)()


if __name__ == "__main__":
    run()
