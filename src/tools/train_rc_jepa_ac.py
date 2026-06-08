"""Train RC JEPA-AC world model with a frozen V-JEPA 2.1 encoder."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import settings
from data.sequence_dataset import DEFAULT_AC_ACTION_COLUMNS, DEFAULT_AC_STATE_COLUMNS, create_ac_sequence_dataloaders
from data.normalization import normalizer_to_dict
from models.rc_jepa_ac import (
    DEFAULT_CHECKPOINT_KEY,
    DEFAULT_ENCODER_NAME,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PREDICTOR_TYPE,
    PREDICTOR_SIZE_PRESETS,
    RCJepaACWorldModel,
    SUPPORTED_PREDICTOR_TYPES,
    apply_predictor_size_preset,
    count_trainable_parameters,
)
from models.vjepa21_presets import SUPPORTED_VJEPA21_ENCODER_NAMES
from tools.wandb_utils import (
    add_wandb_args,
    collect_gradient_metrics,
    collect_parameter_metrics,
    finish_wandb,
    flatten_metrics,
    init_wandb,
    log_metrics,
    update_summary,
    watch_model,
)


DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 10
DEFAULT_EVAL_BATCH_SIZE = settings.AC_EVAL_BATCH_SIZE
DEFAULT_NUM_WORKERS = settings.NUM_WORKERS
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_EPOCHS = 4
DEFAULT_WARMUP_START_FACTOR = 0.1
DEFAULT_MIN_LR_RATIO = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_OUTPUT_DIR = Path("checkpoints/rc_jepa_ac")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Train frozen-encoder RC JEPA-AC world model.")
    parser.add_argument("--manifest-dir", type=Path, default=settings.MANIFEST_DIR)
    parser.add_argument("--vjepa-root", type=Path, default=settings.REPO_ROOT / "vjepa2")
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default=DEFAULT_CHECKPOINT_KEY)
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER_NAME, choices=list(SUPPORTED_VJEPA21_ENCODER_NAMES))
    parser.add_argument("--state-columns", nargs="+", default=list(DEFAULT_AC_STATE_COLUMNS))
    parser.add_argument("--action-columns", nargs="+", default=list(DEFAULT_AC_ACTION_COLUMNS))
    parser.add_argument("--raw-frames-per-sample", type=int, default=settings.AC_RAW_FRAMES_PER_SAMPLE)
    parser.add_argument("--sequence-stride", type=int, default=settings.AC_SEQUENCE_STRIDE)
    parser.add_argument("--image-size", type=int, default=settings.AC_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--tubelet-size", type=int, default=settings.AC_TUBELET_SIZE)
    parser.add_argument("--auto-steps", type=int, default=settings.AC_AUTO_STEPS)
    parser.add_argument("--predictor-type", choices=SUPPORTED_PREDICTOR_TYPES, default=DEFAULT_PREDICTOR_TYPE)
    parser.add_argument("--model-size", choices=tuple(PREDICTOR_SIZE_PRESETS), default="base")
    parser.add_argument("--predictor-dim", type=int, default=None)
    parser.add_argument("--predictor-depth", type=int, default=None)
    parser.add_argument("--predictor-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--warmup-start-factor", type=float, default=DEFAULT_WARMUP_START_FACTOR)
    parser.add_argument("--min-lr-ratio", type=float, default=DEFAULT_MIN_LR_RATIO)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args(argv)
    args._output_dir_was_provided = "--output-dir" in argv or any(arg.startswith("--output-dir=") for arg in argv)
    return args


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> RCJepaACWorldModel:
    return RCJepaACWorldModel(
        vjepa_root=args.vjepa_root,
        checkpoint_path=args.vjepa_checkpoint,
        encoder_name=args.encoder,
        checkpoint_key=args.checkpoint_key,
        image_size=args.image_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        raw_frames_per_sample=args.raw_frames_per_sample,
        state_dim=len(args.state_columns),
        action_dim=len(args.action_columns),
        state_columns=tuple(args.state_columns),
        action_columns=tuple(args.action_columns),
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_type=args.predictor_type,
        dropout=args.dropout,
        auto_steps=args.auto_steps,
        strict_checkpoint=not args.allow_partial_checkpoint,
    )


def compute_steps_per_epoch(dataloader: torch.utils.data.DataLoader) -> int:
    return max(len(dataloader), 1)


def compute_warmup_steps(warmup_epochs: int, steps_per_epoch: int, total_train_steps: int) -> int:
    if warmup_epochs <= 0:
        return 0
    return min(warmup_epochs * steps_per_epoch, max(total_train_steps - 1, 0))


def should_apply_early_stopping(epoch: int, warmup_epochs: int) -> bool:
    return epoch > max(warmup_epochs, 0)


def compute_lr_scale(
    step: int,
    total_train_steps: int,
    warmup_steps: int,
    warmup_start_factor: float,
    min_lr_ratio: float,
) -> float:
    if total_train_steps <= 0:
        return 1.0

    step = min(max(step, 0), total_train_steps)
    if warmup_steps > 0 and step < warmup_steps:
        warmup_progress = step / max(warmup_steps, 1)
        return warmup_start_factor + (1.0 - warmup_start_factor) * warmup_progress

    if total_train_steps <= warmup_steps:
        return 1.0

    cosine_progress = (step - warmup_steps) / max(total_train_steps - warmup_steps, 1)
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_train_steps: int,
    warmup_steps: int,
    warmup_start_factor: float,
    min_lr_ratio: float,
    base_lr: float | None = None,
) -> LambdaLR:
    if base_lr is not None:
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = base_lr
            param_group["lr"] = base_lr
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: compute_lr_scale(
            step=step,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            warmup_start_factor=warmup_start_factor,
            min_lr_ratio=min_lr_ratio,
        ),
    )


def sync_lr_scheduler(lr_scheduler: LambdaLR, global_step: int) -> None:
    """Move LambdaLR to the exact step stored in a checkpoint."""
    global_step = max(int(global_step), 0)
    lr_scheduler.last_epoch = global_step
    lrs = [
        base_lr * lr_lambda(global_step)
        for base_lr, lr_lambda in zip(lr_scheduler.base_lrs, lr_scheduler.lr_lambdas)
    ]
    for param_group, lr in zip(lr_scheduler.optimizer.param_groups, lrs):
        param_group["lr"] = lr
    lr_scheduler._last_lr = lrs


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: LambdaLR | None,
    grad_clip: float,
    label: str,
    show_progress: bool,
    wandb_run: Any | None = None,
    wandb_prefix: str | None = None,
    wandb_log_every: int = 0,
    epoch: int | None = None,
    global_step_start: int = 0,
    wandb_grad_stats_every: int = 0,
    wandb_param_stats_every: int = 0,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "teacher_forcing_loss": 0.0,
        "rollout_loss": 0.0,
    }
    total_samples = 0
    progress = tqdm(
        dataloader,
        desc=label,
        leave=False,
        disable=not show_progress,
    )

    global_step = global_step_start
    for step, batch in enumerate(progress, start=1):
        should_log_batch = (
            training
            and wandb_run is not None
            and wandb_prefix
            and wandb_log_every > 0
            and step % wandb_log_every == 0
        )
        should_log_grad_stats = should_log_batch and wandb_grad_stats_every > 0 and step % wandb_grad_stats_every == 0
        should_log_param_stats = should_log_batch and wandb_param_stats_every > 0 and step % wandb_param_stats_every == 0
        extra_batch_metrics: dict[str, float] = {}

        images = batch["images"].to(device, non_blocking=True)
        states = batch["states"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)

        if training:
            if optimizer is None or lr_scheduler is None:
                raise RuntimeError("Training epoch requires optimizer and lr_scheduler")
            current_lr = float(lr_scheduler.get_last_lr()[0])
            optimizer.zero_grad(set_to_none=True)
        else:
            current_lr = None

        with torch.set_grad_enabled(training):
            outputs = model(images=images, states=states, actions=actions)
            loss = outputs["loss"]

            if training:
                loss.backward()
                if should_log_grad_stats:
                    extra_batch_metrics.update(collect_gradient_metrics(model, prefix="grad_pre_clip"))
                if grad_clip > 0:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if should_log_grad_stats:
                        extra_batch_metrics["grad_clip/pre_clip_global_l2"] = float(pre_clip_norm)
                        extra_batch_metrics["grad_clip/max_norm"] = float(grad_clip)
                if should_log_grad_stats:
                    extra_batch_metrics.update(collect_gradient_metrics(model, prefix="grad_post_clip"))
                optimizer.step()
                lr_scheduler.step()
                if should_log_param_stats:
                    extra_batch_metrics.update(collect_parameter_metrics(model, prefix="param"))

        batch_size = images.size(0)
        for key in totals:
            totals[key] += float(outputs[key].detach().item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(average_metrics(totals, total_samples))

        if training:
            global_step += 1
            if should_log_batch:
                batch_metrics = {
                    key: float(value.detach().item())
                    for key, value in outputs.items()
                }
                batch_metrics["epoch"] = float(epoch or 0)
                if current_lr is not None:
                    batch_metrics["lr"] = current_lr
                batch_metrics.update(extra_batch_metrics)
                log_metrics(
                    wandb_run,
                    flatten_metrics(wandb_prefix, batch_metrics),
                    step=global_step,
                )

    return average_metrics(totals, total_samples), global_step


def average_metrics(totals: dict[str, float], total_samples: int) -> dict[str, float]:
    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: RCJepaACWorldModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR | None,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    normalization: dict[str, Any],
    best_val_loss: float,
    best_epoch: int,
    global_step: int,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
) -> None:
    payload = {
        "epoch": epoch,
        "predictor_state_dict": model.predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "args": args_to_jsonable_dict(args),
        "metrics": metrics,
        "state_columns": list(args.state_columns),
        "action_columns": list(args.action_columns),
        "normalization": normalization,
        "encoder_checkpoint_path": str(args.vjepa_checkpoint),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "note": "Frozen V-JEPA 2.1 encoder weights are not saved in this checkpoint.",
    }
    torch.save(payload, path)


def args_to_jsonable_dict(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_normalization_metadata(dataset: Any) -> dict[str, Any]:
    return {
        "state": normalizer_to_dict(getattr(dataset, "state_normalizer", None)),
        "action": normalizer_to_dict(getattr(dataset, "action_normalizer", None)),
    }


def epoch_wandb_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    lr: float,
) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "lr": lr,
        "best/val_loss": best_val_loss,
        "best/epoch": float(best_epoch),
        "early_stop/patience": float(epochs_without_improvement),
        **flatten_metrics("train", train_metrics),
        **flatten_metrics("val", val_metrics),
    }


def load_resume_checkpoint(
    resume_path: Path,
    model: RCJepaACWorldModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint = torch.load(resume_path, map_location=device)
    validate_resume_predictor_config(checkpoint, args)
    model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def validate_resume_predictor_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    checkpoint_args = dict(checkpoint.get("args", {}))
    checked_fields = (
        "predictor_type",
        "predictor_dim",
        "predictor_depth",
        "predictor_heads",
        "dropout",
    )
    mismatches = []
    for field in checked_fields:
        if field not in checkpoint_args:
            continue
        current_value = getattr(args, field)
        checkpoint_value = checkpoint_args[field]
        if current_value != checkpoint_value:
            mismatches.append(
                {
                    "field": field,
                    "current": current_value,
                    "checkpoint": checkpoint_value,
                }
            )
    if mismatches:
        raise ValueError(
            "Resume checkpoint predictor config does not match current args. "
            f"Mismatches: {mismatches}"
        )


def main() -> None:
    args = parse_args()
    apply_predictor_size_preset(args)
    if not args._output_dir_was_provided:
        suffix_parts = []
        if args.predictor_type != DEFAULT_PREDICTOR_TYPE:
            suffix_parts.append(args.predictor_type)
        if args.model_size != "base":
            suffix_parts.append(args.model_size)
        if suffix_parts:
            args.output_dir = DEFAULT_OUTPUT_DIR.with_name(f"{DEFAULT_OUTPUT_DIR.name}_{'_'.join(suffix_parts)}")

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataloaders = create_ac_sequence_dataloaders(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        manifest_dir=args.manifest_dir,
        raw_frames_per_sample=args.raw_frames_per_sample,
        sequence_stride=args.sequence_stride,
        state_columns=args.state_columns,
        action_columns=args.action_columns,
    )

    model = build_model(args).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = compute_steps_per_epoch(dataloaders["train"])
    total_train_steps = args.epochs * steps_per_epoch
    warmup_steps = compute_warmup_steps(args.warmup_epochs, steps_per_epoch, total_train_steps)
    epochs_dir = args.output_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)

    normalization_metadata = collect_normalization_metadata(dataloaders["train"].dataset)
    run_config = {
        "device": str(device),
        "train_sequences": len(dataloaders["train"].dataset),
        "val_sequences": len(dataloaders["val"].dataset),
        "test_sequences": len(dataloaders["test"].dataset),
        "trainable_parameters": count_trainable_parameters(model),
        "steps_per_epoch": steps_per_epoch,
        "total_train_steps": total_train_steps,
        "warmup_steps": warmup_steps,
        "normalization": normalization_metadata,
        "args": args_to_jsonable_dict(args),
    }
    print(json.dumps(run_config, indent=2), flush=True)
    write_json(args.output_dir / "run_config.json", run_config)

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    start_epoch = 1
    global_step = 0
    resumed_from = None
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_from is not None:
        resumed_from = args.resume_from
        resume_checkpoint = load_resume_checkpoint(
            resume_path=args.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            args=args,
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val_loss = float(resume_checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))
        history = list(resume_checkpoint.get("history", []))
        print(
            json.dumps(
                {
                    "resume_from": str(args.resume_from),
                    "start_epoch": start_epoch,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                indent=2,
            ),
            flush=True,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint already finished epoch {resume_checkpoint['epoch']}, "
                f"but --epochs={args.epochs}. Increase --epochs to continue."
            )
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_train_steps=total_train_steps,
        warmup_steps=warmup_steps,
        warmup_start_factor=args.warmup_start_factor,
        min_lr_ratio=args.min_lr_ratio,
        base_lr=args.lr,
    )
    if resume_checkpoint is not None and resume_checkpoint.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(resume_checkpoint["lr_scheduler_state_dict"])
    sync_lr_scheduler(lr_scheduler, global_step)
    wandb_run = init_wandb(args, config=run_config, job_type="train-rc-jepa-ac")
    watch_model(wandb_run, model.predictor, args)

    try:
        final_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs + 1):
            final_epoch = epoch
            train_metrics, global_step = run_epoch(
                model=model,
                dataloader=dataloaders["train"],
                device=device,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_clip=args.grad_clip,
                label=f"epoch {epoch:03d}/{args.epochs:03d} train",
                show_progress=not args.no_progress,
                wandb_run=wandb_run,
                wandb_prefix="train_batch",
                wandb_log_every=args.wandb_log_every,
                epoch=epoch,
                global_step_start=global_step,
                wandb_grad_stats_every=args.wandb_grad_stats_every,
                wandb_param_stats_every=args.wandb_param_stats_every,
            )

            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    model=model,
                    dataloader=dataloaders["val"],
                    device=device,
                    optimizer=None,
                    lr_scheduler=None,
                    grad_clip=args.grad_clip,
                    label=f"epoch {epoch:03d}/{args.epochs:03d} val",
                    show_progress=not args.no_progress,
                )

            epoch_metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_metrics)
            write_json(args.output_dir / "history.json", history)

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"train_tf={train_metrics['teacher_forcing_loss']:.5f} "
                f"val_tf={val_metrics['teacher_forcing_loss']:.5f} "
                f"train_rollout={train_metrics['rollout_loss']:.5f} "
                f"val_rollout={val_metrics['rollout_loss']:.5f}",
                flush=True,
            )

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
            elif should_apply_early_stopping(epoch=epoch, warmup_epochs=args.warmup_epochs):
                epochs_without_improvement += 1

            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                lr_scheduler,
                epoch,
                args,
                epoch_metrics,
                normalization_metadata,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                global_step=global_step,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )
            save_checkpoint(
                epochs_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                lr_scheduler,
                epoch,
                args,
                epoch_metrics,
                normalization_metadata,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                global_step=global_step,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    epoch_metrics,
                    normalization_metadata,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    global_step=global_step,
                    epochs_without_improvement=epochs_without_improvement,
                    history=history,
                )

            log_metrics(
                wandb_run,
                epoch_wandb_metrics(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    lr=float(optimizer.param_groups[0]["lr"]),
                ),
                step=global_step,
            )

            if (
                args.early_stopping_patience > 0
                and should_apply_early_stopping(epoch=epoch, warmup_epochs=args.warmup_epochs)
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "stopped_epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "patience": args.early_stopping_patience,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                break

        best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
        model.predictor.load_state_dict(best_checkpoint["predictor_state_dict"])

        with torch.no_grad():
            test_metrics, _ = run_epoch(
                model=model,
                dataloader=dataloaders["test"],
                device=device,
                optimizer=None,
                lr_scheduler=None,
                grad_clip=args.grad_clip,
                label="test",
                show_progress=not args.no_progress,
            )

        result = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "resume_from": None if resumed_from is None else str(resumed_from),
            "test": test_metrics,
        }
        write_json(args.output_dir / "test_metrics.json", result)
        log_metrics(
            wandb_run,
            {"best/val_loss": best_val_loss, **flatten_metrics("test", test_metrics)},
            step=max(global_step, 1),
        )
        update_summary(wandb_run, {"best/val_loss": best_val_loss, **flatten_metrics("test", test_metrics)})
        print(json.dumps(result, indent=2), flush=True)
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
