"""Plan RC actions with a trained NN-JEPA feature-cache world model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import settings
from data.feature_sequence_dataset import create_ac_feature_sequence_dataloaders
from tools.rc_jepa_ac_cem_planner import (
    RCJepaACFeatureCEMPlanner,
    denormalize_action_tensor,
)
from tools.rc_jepa_ac_feature_runtime import (
    DEFAULT_FEATURES_DIR,
    build_predictor_from_checkpoint,
    checkpoint_default_path,
    default_device,
    load_feature_checkpoint,
    validate_feature_metadata,
)


DEFAULT_CHECKPOINT = Path("checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline CEM planner for NN-JEPA feature-cache checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=0, help="CEM horizon. 0 means checkpoint auto_steps.")
    parser.add_argument("--goal-offset", type=int, default=0, help="Goal frame offset inside each sample. 0 means horizon.")
    parser.add_argument("--cem-samples", type=int, default=128)
    parser.add_argument("--cem-elites", type=int, default=16)
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--action-penalty", type=float, default=0.0)
    parser.add_argument("--smooth-penalty", type=float, default=0.0)
    parser.add_argument("--action-low", type=float, nargs="+", default=None)
    parser.add_argument("--action-high", type=float, nargs="+", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_horizon(value: int, auto_steps: int, raw_frames_per_sample: int) -> int:
    horizon = auto_steps if value == 0 else value
    if horizon < 1:
        raise ValueError("Planner horizon must be >= 1")
    if horizon > raw_frames_per_sample - 1:
        raise ValueError(
            f"Planner horizon={horizon} exceeds available future frames={raw_frames_per_sample - 1}"
        )
    return int(horizon)


def resolve_goal_offset(value: int, horizon: int, raw_frames_per_sample: int) -> int:
    goal_offset = horizon if value == 0 else value
    if goal_offset < 1:
        raise ValueError("Goal offset must be >= 1")
    if goal_offset > raw_frames_per_sample - 1:
        raise ValueError(
            f"Goal offset={goal_offset} exceeds available future frames={raw_frames_per_sample - 1}"
        )
    return int(goal_offset)


def resolve_action_bounds(
    values: list[float] | None,
    fallback: list[float],
    action_dim: int,
    name: str,
) -> list[float]:
    resolved = fallback if values is None else list(values)
    if len(resolved) != action_dim:
        raise ValueError(f"{name} must contain {action_dim} values, got {len(resolved)}")
    return [float(value) for value in resolved]


def tensor_to_list(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]], action_columns: tuple[str, ...]) -> None:
    fieldnames = [
        "record_index",
        "sample_id",
        "session_id",
        "start_frame",
        "goal_frame",
        "planned_score",
        "planned_final_l1",
        "groundtruth_final_l1",
        "zero_action_final_l1",
    ]
    for column in action_columns:
        fieldnames.append(f"planned_first_{column}")
        fieldnames.append(f"groundtruth_first_{column}")
        fieldnames.append(f"abs_error_first_{column}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fieldnames}
            writer.writerow(row)


def score_final_l1(predictions: torch.Tensor, goal_tokens: torch.Tensor) -> float:
    return float(F.l1_loss(predictions[:, -1], goal_tokens, reduction="none").mean().detach().cpu())


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1")
    if args.eval_batch_size != 1:
        raise ValueError("Planner currently expects --eval-batch-size 1 so each sample has one goal/context pair")

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint, checkpoint_path = load_feature_checkpoint(args.checkpoint, device)
    predictor, config = build_predictor_from_checkpoint(checkpoint, device)

    horizon = resolve_horizon(args.horizon, config.auto_steps, config.raw_frames_per_sample)
    goal_offset = resolve_goal_offset(args.goal_offset, horizon, config.raw_frames_per_sample)
    action_dim = len(config.action_columns)
    action_low = resolve_action_bounds(
        args.action_low,
        [settings.STEERING_MIN, settings.THROTTLE_MIN],
        action_dim,
        "--action-low",
    )
    action_high = resolve_action_bounds(
        args.action_high,
        [settings.STEERING_MAX, settings.THROTTLE_MAX],
        action_dim,
        "--action-high",
    )

    features_dir = args.features_dir or checkpoint_default_path(checkpoint, "features_dir", DEFAULT_FEATURES_DIR)
    manifest_dir = args.manifest_dir or checkpoint_default_path(checkpoint, "manifest_dir", settings.MANIFEST_DIR)
    output_dir = args.output_dir or checkpoint_path.parent / "planning"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = create_ac_feature_sequence_dataloaders(
        features_dir=features_dir,
        batch_size=1,
        eval_batch_size=1,
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
    split_dataset = dataloaders[args.split].dataset
    action_normalizer = getattr(split_dataset, "action_normalizer", None)

    planner = RCJepaACFeatureCEMPlanner(
        predictor=predictor,
        tokens_per_frame=config.tokens_per_frame,
        state_columns=config.state_columns,
        action_columns=config.action_columns,
        action_normalizer=action_normalizer,
        horizon=horizon,
        n_samples=args.cem_samples,
        n_elite=args.cem_elites,
        n_iter=args.cem_iters,
        action_low=action_low,
        action_high=action_high,
        init_std=args.init_std,
        min_std=args.min_std,
        action_penalty=args.action_penalty,
        smooth_penalty=args.smooth_penalty,
        device=device,
    )

    run_config: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "features_dir": str(features_dir),
        "manifest_dir": str(manifest_dir),
        "split": args.split,
        "max_samples": args.max_samples,
        "horizon": horizon,
        "goal_offset": goal_offset,
        "cem_samples": args.cem_samples,
        "cem_elites": args.cem_elites,
        "cem_iters": args.cem_iters,
        "action_low": action_low,
        "action_high": action_high,
        "action_penalty": args.action_penalty,
        "smooth_penalty": args.smooth_penalty,
        "device": str(device),
        "model": config.to_jsonable_dict(),
    }
    print(json.dumps(run_config, indent=2), flush=True)

    records: list[dict[str, Any]] = []
    progress = tqdm(
        dataloaders[args.split],
        desc=f"plan {args.split}",
        leave=False,
        disable=args.no_progress,
    )
    with torch.inference_mode():
        for batch in progress:
            if len(records) >= args.max_samples:
                break

            latents = batch["latents"].to(device, non_blocking=True)
            states = batch["states"].to(device, non_blocking=True)
            model_actions = batch["actions"].to(device, non_blocking=True)
            context_tokens = latents[0, : config.tokens_per_frame]
            goal_start = goal_offset * config.tokens_per_frame
            goal_end = goal_start + config.tokens_per_frame
            goal_tokens = latents[0, goal_start:goal_end]
            initial_state = states[0, 0]

            plan = planner.plan(
                context_tokens=context_tokens,
                initial_state=initial_state,
                goal_tokens=goal_tokens,
            )
            planned_actions = plan.action_sequence.to(device).unsqueeze(0)
            groundtruth_actions = denormalize_action_tensor(
                model_actions[:, :horizon],
                action_columns=config.action_columns,
                action_normalizer=action_normalizer,
            )
            zero_actions = torch.zeros_like(planned_actions)

            planned_predictions = planner.rollout(context_tokens, initial_state, planned_actions)
            groundtruth_predictions = planner.rollout(context_tokens, initial_state, groundtruth_actions)
            zero_predictions = planner.rollout(context_tokens, initial_state, zero_actions)
            goal_batched = goal_tokens.unsqueeze(0)

            planned_final_l1 = score_final_l1(planned_predictions, goal_batched)
            groundtruth_final_l1 = score_final_l1(groundtruth_predictions, goal_batched)
            zero_action_final_l1 = score_final_l1(zero_predictions, goal_batched)
            planned_first = plan.first_action
            groundtruth_first = groundtruth_actions[0, 0].detach().cpu()
            first_abs_error = (planned_first - groundtruth_first).abs()

            record: dict[str, Any] = {
                "record_index": len(records),
                "sample_id": batch["sample_id"][0],
                "session_id": batch["session_id"][0],
                "frame_indices": tensor_to_list(batch["frame_indices"][0]),
                "timestamps_sec": tensor_to_list(batch["timestamps_sec"][0]),
                "start_frame": int(batch["frame_indices"][0, 0]),
                "goal_frame": int(batch["frame_indices"][0, goal_offset]),
                "planned_score": plan.score,
                "planned_final_l1": planned_final_l1,
                "groundtruth_final_l1": groundtruth_final_l1,
                "zero_action_final_l1": zero_action_final_l1,
                "planned_action_sequence": tensor_to_list(plan.action_sequence),
                "groundtruth_action_sequence": tensor_to_list(groundtruth_actions[0].detach().cpu()),
                "first_action_abs_error": tensor_to_list(first_abs_error),
            }
            for index, column in enumerate(config.action_columns):
                record[f"planned_first_{column}"] = float(planned_first[index])
                record[f"groundtruth_first_{column}"] = float(groundtruth_first[index])
                record[f"abs_error_first_{column}"] = float(first_abs_error[index])

            records.append(record)
            progress.set_postfix(
                {
                    "samples": len(records),
                    "planned_l1": planned_final_l1,
                    "gt_l1": groundtruth_final_l1,
                }
            )

    jsonl_path = output_dir / f"planning_{args.split}.jsonl"
    csv_path = output_dir / f"planning_{args.split}.csv"
    summary_path = output_dir / f"planning_{args.split}_summary.json"
    write_jsonl(jsonl_path, records)
    write_csv(csv_path, records, config.action_columns)

    summary = {
        **run_config,
        "written_samples": len(records),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "mean_planned_final_l1": mean_or_none([float(row["planned_final_l1"]) for row in records]),
        "mean_groundtruth_final_l1": mean_or_none([float(row["groundtruth_final_l1"]) for row in records]),
        "mean_zero_action_final_l1": mean_or_none([float(row["zero_action_final_l1"]) for row in records]),
    }
    for column in config.action_columns:
        summary[f"mean_abs_error_first_{column}"] = mean_or_none(
            [float(row[f"abs_error_first_{column}"]) for row in records]
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
