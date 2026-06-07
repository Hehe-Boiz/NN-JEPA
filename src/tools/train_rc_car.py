"""Train a simple RC driving policy with image + sensor inputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from data import create_dataloaders, settings
from data.normalization import normalizer_to_dict
from models.rc_car_model import DEFAULT_SENSOR_NAMES, RCDrivingModel
from tools.wandb_utils import add_wandb_args, finish_wandb, flatten_metrics, init_wandb, log_metrics, update_summary


DEFAULT_BACKBONE = "small_cnn"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = settings.NUM_WORKERS
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_STEERING_WEIGHT = 1.0
DEFAULT_THROTTLE_WEIGHT = 1.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_OUTPUT_DIR = Path("checkpoints/rc_car_bc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple RC driving policy.")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=["small_cnn", "vjepa2_1_vitb", "vjepa2_1_vitl"])
    parser.add_argument("--vjepa-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--sensor-names", nargs="+", default=list(DEFAULT_SENSOR_NAMES))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--steering-weight", type=float, default=DEFAULT_STEERING_WEIGHT)
    parser.add_argument("--throttle-weight", type=float, default=DEFAULT_THROTTLE_WEIGHT)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    add_wandb_args(parser)
    return parser.parse_args()


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


def build_model(args: argparse.Namespace) -> RCDrivingModel:
    return RCDrivingModel(
        sensor_names=args.sensor_names,
        image_backbone=args.backbone,
        vjepa_checkpoint_path=args.vjepa_checkpoint,
        freeze_image_encoder=args.freeze_image_encoder,
    )


def action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    steering_weight: float,
    throttle_weight: float,
) -> torch.Tensor:
    weights = prediction.new_tensor([steering_weight, throttle_weight])
    per_value_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (per_value_loss * weights).mean()


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    steering_weight: float,
    throttle_weight: float,
    grad_clip: float,
    label: str,
    show_progress: bool,
    wandb_run: Any | None = None,
    wandb_prefix: str | None = None,
    wandb_log_every: int = 0,
    epoch: int | None = None,
    global_step_start: int = 0,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_steering_mae = 0.0
    total_throttle_mae = 0.0
    total_samples = 0
    progress = tqdm(
        dataloader,
        desc=label,
        leave=False,
        disable=not show_progress,
    )

    global_step = global_step_start
    for step, batch in enumerate(progress, start=1):
        image = batch["image"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        prediction = model(image, state)
        loss = action_loss(
            prediction,
            target,
            steering_weight=steering_weight,
            throttle_weight=throttle_weight,
        )

        if training:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = image.size(0)
        mae = (prediction - target).abs().detach()
        total_loss += loss.item() * batch_size
        total_steering_mae += mae[:, 0].sum().item()
        total_throttle_mae += mae[:, 1].sum().item()
        total_samples += batch_size
        progress.set_postfix(
            average_metrics(
                total_loss=total_loss,
                total_steering_mae=total_steering_mae,
                total_throttle_mae=total_throttle_mae,
                total_samples=total_samples,
            )
        )

        if training:
            global_step += 1
            if wandb_run is not None and wandb_prefix and wandb_log_every > 0 and step % wandb_log_every == 0:
                mae_metrics = (prediction - target).abs().detach()
                batch_metrics = {
                    "loss": float(loss.detach().item()),
                    "steering_mae": float(mae_metrics[:, 0].mean().item()),
                    "throttle_mae": float(mae_metrics[:, 1].mean().item()),
                    "epoch": float(epoch or 0),
                }
                log_metrics(
                    wandb_run,
                    flatten_metrics(wandb_prefix, batch_metrics),
                    step=global_step,
                )

    return average_metrics(
        total_loss=total_loss,
        total_steering_mae=total_steering_mae,
        total_throttle_mae=total_throttle_mae,
        total_samples=total_samples,
    ), global_step


def average_metrics(
    total_loss: float,
    total_steering_mae: float,
    total_throttle_mae: float,
    total_samples: int,
) -> dict[str, float]:
    return {
        "loss": total_loss / max(total_samples, 1),
        "steering_mae": total_steering_mae / max(total_samples, 1),
        "throttle_mae": total_throttle_mae / max(total_samples, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    normalization: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "state_columns": list(settings.STATE_COLUMNS),
        "action_columns": list(settings.ACTION_COLUMNS),
        "normalization": normalization,
    }
    torch.save(payload, path)


def collect_normalization_metadata(dataset: Any) -> dict[str, Any]:
    return {
        "state": normalizer_to_dict(getattr(dataset, "state_normalizer", None)),
        "action": None,
    }


def args_to_jsonable_dict(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def epoch_wandb_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    lr: float,
) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "lr": lr,
        "best/val_loss": best_val_loss,
        **flatten_metrics("train", train_metrics),
        **flatten_metrics("val", val_metrics),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(args).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    normalization_metadata = collect_normalization_metadata(dataloaders["train"].dataset)

    run_config = {
        "device": str(device),
        "backbone": args.backbone,
        "sensor_names": args.sensor_names,
        "train_samples": len(dataloaders["train"].dataset),
        "val_samples": len(dataloaders["val"].dataset),
        "test_samples": len(dataloaders["test"].dataset),
        "normalization": normalization_metadata,
        "args": args_to_jsonable_dict(args),
    }
    print(json.dumps(run_config, indent=2))

    wandb_run = init_wandb(args, config=run_config, job_type="train-rc-car-bc")

    try:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            train_metrics, global_step = run_epoch(
                model=model,
                dataloader=dataloaders["train"],
                device=device,
                optimizer=optimizer,
                steering_weight=args.steering_weight,
                throttle_weight=args.throttle_weight,
                grad_clip=args.grad_clip,
                label=f"epoch {epoch:03d}/{args.epochs:03d} train",
                show_progress=not args.no_progress,
                wandb_run=wandb_run,
                wandb_prefix="train_batch",
                wandb_log_every=args.wandb_log_every,
                epoch=epoch,
                global_step_start=global_step,
            )

            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    model=model,
                    dataloader=dataloaders["val"],
                    device=device,
                    optimizer=None,
                    steering_weight=args.steering_weight,
                    throttle_weight=args.throttle_weight,
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

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"train_steer_mae={train_metrics['steering_mae']:.4f} "
                f"val_steer_mae={val_metrics['steering_mae']:.4f} "
                f"train_throt_mae={train_metrics['throttle_mae']:.4f} "
                f"val_throt_mae={val_metrics['throttle_mae']:.4f}",
                flush=True,
            )

            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args, epoch_metrics, normalization_metadata)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, epoch_metrics, normalization_metadata)

            log_metrics(
                wandb_run,
                epoch_wandb_metrics(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_loss=best_val_loss,
                    lr=float(optimizer.param_groups[0]["lr"]),
                ),
                step=epoch,
            )

        history_path = output_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

        with torch.no_grad():
            test_metrics, _ = run_epoch(
                model=model,
                dataloader=dataloaders["test"],
                device=device,
                optimizer=None,
                steering_weight=args.steering_weight,
                throttle_weight=args.throttle_weight,
                grad_clip=args.grad_clip,
                label="test",
                show_progress=not args.no_progress,
            )

        result = {"best_val_loss": best_val_loss, "test": test_metrics}
        test_metrics_path = output_dir / "test_metrics.json"
        test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        log_metrics(
            wandb_run,
            {"best/val_loss": best_val_loss, **flatten_metrics("test", test_metrics)},
            step=args.epochs,
        )
        update_summary(wandb_run, {"best/val_loss": best_val_loss, **flatten_metrics("test", test_metrics)})
        print(json.dumps(result, indent=2))
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
