"""Train RC JEPA-AC world model with a frozen V-JEPA 2.1 encoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from data import settings
from data.sequence_dataset import DEFAULT_AC_ACTION_COLUMNS, DEFAULT_AC_STATE_COLUMNS, create_ac_sequence_dataloaders
from models.rc_jepa_ac import (
    DEFAULT_CHECKPOINT_KEY,
    DEFAULT_ENCODER_NAME,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PREDICTOR_DEPTH,
    DEFAULT_PREDICTOR_DIM,
    DEFAULT_PREDICTOR_HEADS,
    RCJepaACWorldModel,
    count_trainable_parameters,
)


DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 0
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_OUTPUT_DIR = Path("checkpoints/rc_jepa_ac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-encoder RC JEPA-AC world model.")
    parser.add_argument("--manifest-dir", type=Path, default=settings.MANIFEST_DIR)
    parser.add_argument("--vjepa-root", type=Path, default=settings.REPO_ROOT / "vjepa2")
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default=DEFAULT_CHECKPOINT_KEY)
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER_NAME, choices=["vit_small_384", "vit_base_384", "vit_large_384"])
    parser.add_argument("--state-columns", nargs="+", default=list(DEFAULT_AC_STATE_COLUMNS))
    parser.add_argument("--action-columns", nargs="+", default=list(DEFAULT_AC_ACTION_COLUMNS))
    parser.add_argument("--raw-frames-per-sample", type=int, default=settings.AC_RAW_FRAMES_PER_SAMPLE)
    parser.add_argument("--sequence-stride", type=int, default=settings.AC_SEQUENCE_STRIDE)
    parser.add_argument("--image-size", type=int, default=settings.AC_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--tubelet-size", type=int, default=settings.AC_TUBELET_SIZE)
    parser.add_argument("--auto-steps", type=int, default=settings.AC_AUTO_STEPS)
    parser.add_argument("--predictor-dim", type=int, default=DEFAULT_PREDICTOR_DIM)
    parser.add_argument("--predictor-depth", type=int, default=DEFAULT_PREDICTOR_DEPTH)
    parser.add_argument("--predictor-heads", type=int, default=DEFAULT_PREDICTOR_HEADS)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        dropout=args.dropout,
        auto_steps=args.auto_steps,
        strict_checkpoint=not args.allow_partial_checkpoint,
    )


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "teacher_forcing_loss": 0.0,
        "rollout_loss": 0.0,
    }
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        states = batch["states"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images=images, states=states, actions=actions)
            loss = outputs["loss"]

            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = images.size(0)
        for key in totals:
            totals[key] += float(outputs[key].detach().item()) * batch_size
        total_samples += batch_size

    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: RCJepaACWorldModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "predictor_state_dict": model.predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args_to_jsonable_dict(args),
        "metrics": metrics,
        "state_columns": list(args.state_columns),
        "action_columns": list(args.action_columns),
        "encoder_checkpoint_path": str(args.vjepa_checkpoint),
        "note": "Frozen V-JEPA 2.1 encoder weights are not saved in this checkpoint.",
    }
    torch.save(payload, path)


def args_to_jsonable_dict(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataloaders = create_ac_sequence_dataloaders(
        batch_size=args.batch_size,
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

    run_config = {
        "device": str(device),
        "train_sequences": len(dataloaders["train"].dataset),
        "val_sequences": len(dataloaders["val"].dataset),
        "test_sequences": len(dataloaders["test"].dataset),
        "trainable_parameters": count_trainable_parameters(model),
        "args": args_to_jsonable_dict(args),
    }
    print(json.dumps(run_config, indent=2), flush=True)
    write_json(args.output_dir / "run_config.json", run_config)

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=dataloaders["val"],
                device=device,
                optimizer=None,
                grad_clip=args.grad_clip,
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

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, args, epoch_metrics)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, args, epoch_metrics)

    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.predictor.load_state_dict(best_checkpoint["predictor_state_dict"])

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["test"],
            device=device,
            optimizer=None,
            grad_clip=args.grad_clip,
        )

    result = {
        "best_val_loss": best_val_loss,
        "test": test_metrics,
    }
    write_json(args.output_dir / "test_metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
