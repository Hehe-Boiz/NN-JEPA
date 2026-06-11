"""Run RC JEPA-AC latent inference from cached V-JEPA features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import settings
from data.feature_sequence_dataset import create_ac_feature_sequence_dataloaders
from models.rc_jepa_ac import build_rollout_state_context
from tools.rc_jepa_ac_feature_runtime import (
    DEFAULT_FEATURES_DIR,
    FeaturePredictorConfig,
    build_predictor_from_checkpoint,
    checkpoint_default_path,
    default_device,
    load_feature_checkpoint,
    validate_feature_metadata,
)


DEFAULT_CHECKPOINT = Path("checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt")
INFER_ROLLOUT_STATE_MODE_FALLBACK = "fallback"
INFER_ROLLOUT_STATE_MODE_MEASURED = "measured"
SUPPORTED_INFER_ROLLOUT_STATE_MODES = (
    INFER_ROLLOUT_STATE_MODE_FALLBACK,
    INFER_ROLLOUT_STATE_MODE_MEASURED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC JEPA-AC latent inference from cached V-JEPA features.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=settings.AC_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--rollout-state-mode",
        choices=SUPPORTED_INFER_ROLLOUT_STATE_MODES,
        default=INFER_ROLLOUT_STATE_MODE_FALLBACK,
        help=(
            "State conditioning for autoregressive rollout. fallback matches inference/planning; "
            "measured is only valid for offline dataset eval with future states available."
        ),
    )
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def maybe_cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_batch(
    predictor: nn.Module,
    latents: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: FeaturePredictorConfig,
    rollout_state_mode: str = INFER_ROLLOUT_STATE_MODE_FALLBACK,
) -> dict[str, torch.Tensor]:
    tokens_per_frame = config.tokens_per_frame
    num_frames = states.size(1)
    input_latents = latents[:, :-tokens_per_frame]
    teacher_target = latents[:, tokens_per_frame:]
    teacher_pred = predictor(
        latent_tokens=input_latents,
        actions=actions,
        states=states[:, :-1],
        tokens_per_frame=tokens_per_frame,
    )

    rollout_steps = min(config.auto_steps, num_frames - 1)
    rollout_tokens = latents[:, :tokens_per_frame]
    if rollout_state_mode == INFER_ROLLOUT_STATE_MODE_MEASURED:
        if states.size(1) < rollout_steps:
            raise ValueError(
                "Measured rollout state mode requires future measured states in the batch; "
                f"need {rollout_steps}, got {states.size(1)}"
            )
        rollout_states = states[:, :rollout_steps]
    elif rollout_state_mode == INFER_ROLLOUT_STATE_MODE_FALLBACK:
        rollout_states = build_rollout_state_context(
            initial_state=states[:, :1],
            actions=actions,
            rollout_steps=rollout_steps,
            state_columns=config.state_columns,
            action_columns=config.action_columns,
        )
    else:
        raise ValueError(f"Unsupported inference rollout_state_mode={rollout_state_mode!r}")
    rollout_predictions = []
    for step in range(rollout_steps):
        pred_tokens = predictor(
            latent_tokens=rollout_tokens,
            actions=actions[:, : step + 1],
            states=rollout_states[:, : step + 1],
            tokens_per_frame=tokens_per_frame,
        )
        next_tokens = pred_tokens[:, -tokens_per_frame:]
        rollout_predictions.append(next_tokens)
        rollout_tokens = torch.cat([rollout_tokens, next_tokens], dim=1)

    rollout_pred = torch.cat(rollout_predictions, dim=1)
    rollout_target = latents[:, tokens_per_frame : tokens_per_frame * (rollout_steps + 1)]
    return {
        "teacher_pred": teacher_pred,
        "teacher_target": teacher_target,
        "rollout_pred": rollout_pred,
        "rollout_target": rollout_target,
    }


def per_sample_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target, reduction="none").mean(dim=(1, 2))


def tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1")

    device = torch.device(args.device)
    checkpoint, checkpoint_path = load_feature_checkpoint(args.checkpoint, device)
    predictor, config = build_predictor_from_checkpoint(checkpoint, device)

    features_dir = args.features_dir or checkpoint_default_path(checkpoint, "features_dir", DEFAULT_FEATURES_DIR)
    manifest_dir = args.manifest_dir or checkpoint_default_path(checkpoint, "manifest_dir", settings.MANIFEST_DIR)
    output_dir = args.output_dir or checkpoint_path.parent / "inference"
    tensor_dir = output_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_tensors:
        tensor_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = create_ac_feature_sequence_dataloaders(
        features_dir=features_dir,
        batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        raw_frames_per_sample=config.raw_frames_per_sample,
        sequence_stride=config.sequence_stride,
        frame_stride=config.frame_stride,
        target_fps=config.target_fps,
        state_columns=config.state_columns,
        action_columns=config.action_columns,
    )
    validate_feature_metadata(dataloaders["train"].dataset.feature_metadata, config.feature_metadata)
    if args.rollout_state_mode == INFER_ROLLOUT_STATE_MODE_FALLBACK:
        print(
            "Inference uses fallback rollout states, not measured future states. "
            "Dynamic IMU states are stale/approximated in this mode.",
            flush=True,
        )
    else:
        print(
            "Offline inference uses measured future states from the dataset. "
            "Do not use this mode for live closed-loop planning.",
            flush=True,
        )

    records: list[dict[str, Any]] = []
    maybe_cleanup_cuda()
    progress = tqdm(dataloaders[args.split], desc=f"infer {args.split}", leave=False, disable=args.no_progress)
    with torch.inference_mode():
        for batch in progress:
            latents = batch["latents"].to(device, non_blocking=True)
            states = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            outputs = predict_batch(
                predictor=predictor,
                latents=latents,
                states=states,
                actions=actions,
                config=config,
                rollout_state_mode=args.rollout_state_mode,
            )
            teacher_l1 = per_sample_l1(outputs["teacher_pred"], outputs["teacher_target"]).detach().cpu()
            rollout_l1 = per_sample_l1(outputs["rollout_pred"], outputs["rollout_target"]).detach().cpu()

            batch_size = latents.size(0)
            for row in range(batch_size):
                record_index = len(records)
                if record_index >= args.max_samples:
                    break
                tensor_path = None
                if args.save_tensors:
                    tensor_path = tensor_dir / f"{args.split}_{record_index:06d}.pt"
                    torch.save(
                        {
                            "teacher_pred": outputs["teacher_pred"][row].detach().cpu(),
                            "teacher_target": outputs["teacher_target"][row].detach().cpu(),
                            "rollout_pred": outputs["rollout_pred"][row].detach().cpu(),
                            "rollout_target": outputs["rollout_target"][row].detach().cpu(),
                        },
                        tensor_path,
                    )

                records.append(
                    {
                        "record_index": record_index,
                        "sample_id": batch["sample_id"][row],
                        "session_id": batch["session_id"][row],
                        "frame_indices": tensor_to_list(batch["frame_indices"][row]),
                        "timestamps_sec": tensor_to_list(batch["timestamps_sec"][row]),
                        "teacher_forcing_l1": float(teacher_l1[row]),
                        "rollout_l1": float(rollout_l1[row]),
                        "rollout_state_mode": args.rollout_state_mode,
                        "tensor_path": None if tensor_path is None else str(tensor_path),
                    }
                )
            progress.set_postfix({"samples": len(records)})
            if len(records) >= args.max_samples:
                break

    output_path = output_dir / f"inference_{args.split}.jsonl"
    write_jsonl(output_path, records)
    summary = {
        "checkpoint": str(checkpoint_path),
        "features_dir": str(features_dir),
        "manifest_dir": str(manifest_dir),
        "split": args.split,
        "max_samples": args.max_samples,
        "written_samples": len(records),
        "eval_batch_size": args.eval_batch_size,
        "rollout_state_mode": args.rollout_state_mode,
        "save_tensors": args.save_tensors,
        "output_path": str(output_path),
    }
    summary_path = output_dir / f"inference_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
