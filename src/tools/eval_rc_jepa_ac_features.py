"""Evaluate a trained RC JEPA-AC predictor from cached V-JEPA features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data import settings
from data.feature_sequence_dataset import create_ac_feature_sequence_dataloaders
from tools.rc_jepa_ac_feature_runtime import (
    DEFAULT_FEATURES_DIR,
    build_predictor_from_checkpoint,
    checkpoint_default_path,
    default_device,
    load_feature_checkpoint,
    validate_feature_metadata,
)
from tools.train_rc_jepa_ac_features import run_epoch, write_json
from tools.wandb_utils import add_wandb_args, finish_wandb, flatten_metrics, init_wandb, log_metrics, update_summary


DEFAULT_CHECKPOINT = Path("checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RC JEPA-AC predictor from cached V-JEPA features.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--eval-batch-size", type=int, default=settings.AC_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    add_wandb_args(parser)
    return parser.parse_args()


def maybe_cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint, checkpoint_path = load_feature_checkpoint(args.checkpoint, device)
    predictor, config = build_predictor_from_checkpoint(checkpoint, device)

    features_dir = args.features_dir or checkpoint_default_path(checkpoint, "features_dir", DEFAULT_FEATURES_DIR)
    manifest_dir = args.manifest_dir or checkpoint_default_path(checkpoint, "manifest_dir", settings.MANIFEST_DIR)
    output_dir = args.output_dir or checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = create_ac_feature_sequence_dataloaders(
        features_dir=features_dir,
        batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        raw_frames_per_sample=config.raw_frames_per_sample,
        sequence_stride=config.sequence_stride,
        state_columns=config.state_columns,
        action_columns=config.action_columns,
    )
    validate_feature_metadata(dataloaders["train"].dataset.feature_metadata, config.feature_metadata)

    selected_splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    run_config: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "features_dir": str(features_dir),
        "manifest_dir": str(manifest_dir),
        "split": args.split,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "model": config.to_jsonable_dict(),
        "sequence_counts": {split: len(dataloaders[split].dataset) for split in selected_splits},
    }
    print(json.dumps(run_config, indent=2), flush=True)

    wandb_run = init_wandb(args, config=run_config, job_type="eval-rc-jepa-ac-features")
    try:
        metrics_by_split: dict[str, dict[str, float]] = {}
        with torch.inference_mode():
            for split in selected_splits:
                maybe_cleanup_cuda()
                metrics, _ = run_epoch(
                    predictor=predictor,
                    dataloader=dataloaders[split],
                    device=device,
                    optimizer=None,
                    lr_scheduler=None,
                    grad_clip=0.0,
                    tokens_per_frame=config.tokens_per_frame,
                    auto_steps=config.auto_steps,
                    state_columns=config.state_columns,
                    action_columns=config.action_columns,
                    label=f"eval {split}",
                    show_progress=not args.no_progress,
                )
                metrics_by_split[split] = metrics
                log_metrics(wandb_run, flatten_metrics(split, metrics), step=1)

        result = {
            "checkpoint": str(checkpoint_path),
            "features_dir": str(features_dir),
            "manifest_dir": str(manifest_dir),
            "eval_batch_size": args.eval_batch_size,
            "metrics": metrics_by_split,
        }
        output_path = output_dir / f"eval_{args.split}.json"
        write_json(output_path, result)
        update_summary(
            wandb_run,
            {f"{split}/{key}": value for split, metrics in metrics_by_split.items() for key, value in metrics.items()},
        )
        print(json.dumps({"output_path": str(output_path), **result}, indent=2), flush=True)
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
