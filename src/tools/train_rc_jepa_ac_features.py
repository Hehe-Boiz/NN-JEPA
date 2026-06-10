"""Train RC JEPA-AC predictor from precomputed V-JEPA 2.1 features."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import settings
from data.feature_sequence_dataset import (
    SUPPORTED_FEATURE_SAMPLERS,
    create_ac_feature_sequence_dataloaders,
)
from data.normalization import normalizer_to_dict
from data.sequence_dataset import DEFAULT_AC_ACTION_COLUMNS, DEFAULT_AC_STATE_COLUMNS
from models.rc_jepa_ac import (
    DEFAULT_PREDICTOR_TYPE,
    PREDICTOR_SIZE_PRESETS,
    SUPPORTED_PREDICTOR_TYPES,
    apply_predictor_size_preset,
    build_ac_predictor,
    build_rollout_state_context,
    compute_world_model_losses,
    count_trainable_parameters,
)
from tools.train_rc_jepa_ac import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR,
    DEFAULT_MIN_LR_RATIO,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WARMUP_START_FACTOR,
    DEFAULT_WEIGHT_DECAY,
    build_lr_scheduler,
    compute_steps_per_epoch,
    compute_warmup_steps,
    should_apply_early_stopping,
    sync_lr_scheduler,
)
from tools.rc_jepa_ac_cem_planner import (
    RCJepaACFeatureCEMPlanner,
    denormalize_action_tensor,
)
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


DEFAULT_FEATURES_DIR = settings.PROCESSED_DATA_DIR / "features" / "vjepa2_1_vitb_384_ema_fp16"
DEFAULT_OUTPUT_DIR = Path("checkpoints/rc_jepa_ac_vitb_features_20260607")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Train RC JEPA-AC predictor from cached V-JEPA features.")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=settings.MANIFEST_DIR)
    parser.add_argument("--state-columns", nargs="+", default=list(DEFAULT_AC_STATE_COLUMNS))
    parser.add_argument("--action-columns", nargs="+", default=list(DEFAULT_AC_ACTION_COLUMNS))
    parser.add_argument("--raw-frames-per-sample", type=int, default=settings.AC_RAW_FRAMES_PER_SAMPLE)
    parser.add_argument("--sequence-stride", type=int, default=settings.AC_SEQUENCE_STRIDE)
    parser.add_argument("--frame-stride", type=int, default=settings.AC_FRAME_STRIDE)
    parser.add_argument("--target-fps", type=float, default=settings.AC_TARGET_FPS)
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
    parser.add_argument(
        "--train-sampler",
        choices=SUPPORTED_FEATURE_SAMPLERS,
        default="global",
        help="Train sampler. global preserves old behavior; session keeps each batch within one session.",
    )
    parser.add_argument(
        "--eval-sampler",
        choices=SUPPORTED_FEATURE_SAMPLERS,
        default="global",
        help="Val/test sampler. session keeps batch boundaries within one session.",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--warmup-start-factor", type=float, default=DEFAULT_WARMUP_START_FACTOR)
    parser.add_argument("--min-lr-ratio", type=float, default=DEFAULT_MIN_LR_RATIO)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--amp-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Forward/loss autocast dtype. bf16 matches V-JEPA AC style on CUDA; fp32 disables autocast.",
    )
    parser.add_argument(
        "--final-eval-horizon",
        type=int,
        default=3,
        help="Final val rollout-vs-identity horizon after training. 0 disables final eval.",
    )
    parser.add_argument(
        "--val-rollout-eval-horizon",
        type=int,
        default=0,
        help="Optional rollout-vs-identity horizon to log on val after each epoch. 0 disables per-epoch rollout eval.",
    )
    parser.add_argument(
        "--val-rollout-eval-max-batches",
        type=int,
        default=256,
        help="Maximum val batches for per-epoch rollout eval. 0 means full val split.",
    )
    parser.add_argument(
        "--final-planning-eval-samples",
        type=int,
        default=0,
        help="Number of val samples for final offline CEM planning eval. 0 disables planning eval.",
    )
    parser.add_argument(
        "--final-planning-horizon",
        type=int,
        default=0,
        help="CEM planning horizon for final planning eval. 0 means auto_steps.",
    )
    parser.add_argument(
        "--final-planning-goal-offset",
        type=int,
        default=0,
        help="Goal frame offset for final planning eval. 0 means final_planning_horizon.",
    )
    parser.add_argument("--final-planning-cem-samples", type=int, default=64)
    parser.add_argument("--final-planning-cem-elites", type=int, default=8)
    parser.add_argument("--final-planning-cem-iters", type=int, default=3)
    parser.add_argument("--final-planning-init-std", type=float, default=0.5)
    parser.add_argument("--final-planning-min-std", type=float, default=0.05)
    parser.add_argument("--final-planning-action-penalty", type=float, default=0.0)
    parser.add_argument("--final-planning-smooth-penalty", type=float, default=0.0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--skip-test",
        dest="skip_test",
        action="store_true",
        default=True,
        help="Skip final test evaluation after training. Validation still runs every epoch.",
    )
    test_group.add_argument(
        "--run-test",
        dest="skip_test",
        action="store_false",
        help="Run final test evaluation from best.pt after training.",
    )
    add_wandb_args(parser)
    argv_list = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv_list)
    args._output_dir_was_provided = "--output-dir" in argv_list or any(
        arg.startswith("--output-dir=") for arg in argv_list
    )
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


def autocast_context(device: torch.device, amp_dtype: str):
    """Return an autocast context for predictor forward/loss only."""
    if amp_dtype == "fp32" or device.type != "cuda":
        return nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported amp_dtype={amp_dtype!r}")


def build_predictor(args: argparse.Namespace, tokens_per_frame: int, embed_dim: int) -> nn.Module:
    return build_ac_predictor(
        predictor_type=args.predictor_type,
        latent_dim=embed_dim,
        state_dim=len(args.state_columns),
        action_dim=len(args.action_columns),
        tokens_per_frame=tokens_per_frame,
        max_frames=args.raw_frames_per_sample,
        predictor_dim=args.predictor_dim,
        depth=args.predictor_depth,
        num_heads=args.predictor_heads,
        dropout=args.dropout,
    )


def set_dataloader_epoch(dataloader: torch.utils.data.DataLoader, epoch: int) -> None:
    """Let custom samplers reshuffle deterministically for a new epoch."""
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
        return
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def run_epoch(
    predictor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: LambdaLR | None,
    grad_clip: float,
    tokens_per_frame: int,
    auto_steps: int,
    state_columns: tuple[str, ...],
    action_columns: tuple[str, ...],
    label: str,
    show_progress: bool,
    wandb_run: Any | None = None,
    wandb_prefix: str | None = None,
    wandb_log_every: int = 0,
    epoch: int | None = None,
    global_step_start: int = 0,
    wandb_grad_stats_every: int = 0,
    wandb_param_stats_every: int = 0,
    amp_dtype: str = "fp32",
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    predictor.train(training)
    totals = {
        "loss": 0.0,
        "teacher_forcing_loss": 0.0,
        "rollout_loss": 0.0,
    }
    domain_totals: dict[str, dict[str, float]] = {}
    domain_counts: dict[str, int] = {}
    total_samples = 0
    global_step = global_step_start
    progress = tqdm(dataloader, desc=label, leave=False, disable=not show_progress)

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

        latents = batch["latents"].to(device, non_blocking=True)
        states = batch["states"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)

        if training:
            if optimizer is None or lr_scheduler is None:
                raise RuntimeError("Training epoch requires optimizer and lr_scheduler")
            current_lr = float(lr_scheduler.get_last_lr()[0])
            optimizer.zero_grad(set_to_none=True)
        else:
            current_lr = None

        with torch.set_grad_enabled(training), autocast_context(device, amp_dtype):
            outputs = compute_world_model_losses(
                predictor=predictor,
                latents=latents,
                states=states,
                actions=actions,
                tokens_per_frame=tokens_per_frame,
                auto_steps=auto_steps,
                state_columns=state_columns,
                action_columns=action_columns,
            )
            loss = outputs["loss"]
            if not training and "data_domain" in batch:
                domains = [str(value) for value in batch["data_domain"]]
                for domain in sorted(set(domains)):
                    row_indices = [row for row, value in enumerate(domains) if value == domain]
                    if not row_indices:
                        continue
                    if len(row_indices) == latents.size(0):
                        domain_outputs = outputs
                    else:
                        index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
                        domain_outputs = compute_world_model_losses(
                            predictor=predictor,
                            latents=latents.index_select(0, index_tensor),
                            states=states.index_select(0, index_tensor),
                            actions=actions.index_select(0, index_tensor),
                            tokens_per_frame=tokens_per_frame,
                            auto_steps=auto_steps,
                            state_columns=state_columns,
                            action_columns=action_columns,
                        )
                    domain_totals.setdefault(
                        domain,
                        {
                            "loss": 0.0,
                            "teacher_forcing_loss": 0.0,
                            "rollout_loss": 0.0,
                        },
                    )
                    domain_counts[domain] = domain_counts.get(domain, 0) + len(row_indices)
                    for key in totals:
                        domain_totals[domain][key] += (
                            float(domain_outputs[key].detach().item()) * len(row_indices)
                        )
            if training:
                loss.backward()
                if should_log_grad_stats:
                    extra_batch_metrics.update(collect_gradient_metrics(predictor, prefix="grad_pre_clip"))
                if grad_clip > 0:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)
                    if should_log_grad_stats:
                        extra_batch_metrics["grad_clip/pre_clip_global_l2"] = float(pre_clip_norm)
                        extra_batch_metrics["grad_clip/max_norm"] = float(grad_clip)
                if should_log_grad_stats:
                    extra_batch_metrics.update(collect_gradient_metrics(predictor, prefix="grad_post_clip"))
                optimizer.step()
                lr_scheduler.step()
                if should_log_param_stats:
                    extra_batch_metrics.update(collect_parameter_metrics(predictor, prefix="param"))

        batch_size = latents.size(0)
        for key in totals:
            totals[key] += float(outputs[key].detach().item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(average_metrics(totals, total_samples))

        if training:
            global_step += 1
            if should_log_batch:
                batch_metrics = {key: float(value.detach().item()) for key, value in outputs.items()}
                batch_metrics["epoch"] = float(epoch or 0)
                if current_lr is not None:
                    batch_metrics["lr"] = current_lr
                batch_metrics.update(extra_batch_metrics)
                log_metrics(wandb_run, flatten_metrics(wandb_prefix, batch_metrics), step=global_step)

    metrics = average_metrics(totals, total_samples)
    for domain, totals_by_key in domain_totals.items():
        domain_metrics = average_metrics(totals_by_key, domain_counts.get(domain, 0))
        for key, value in domain_metrics.items():
            metrics[f"domain/{domain}/{key}"] = value
    return metrics, global_step


def average_metrics(totals: dict[str, float], total_samples: int) -> dict[str, float]:
    return {key: value / max(total_samples, 1) for key, value in totals.items()}


@torch.no_grad()
def final_rollout_identity_eval(
    predictor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tokens_per_frame: int,
    horizon: int,
    state_columns: tuple[str, ...],
    action_columns: tuple[str, ...],
    show_progress: bool,
    amp_dtype: str = "fp32",
    max_batches: int = 0,
    label: str = "final val rollout",
) -> dict[str, float]:
    """Compare autoregressive rollout against an identity latent baseline on val."""
    if horizon < 1:
        return {}

    predictor.eval()
    model_totals: dict[int, float] = {}
    identity_totals: dict[int, float] = {}
    count = 0
    progress = tqdm(dataloader, desc=label, leave=False, disable=not show_progress)
    for batch_index, batch in enumerate(progress, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        latents = batch["latents"].to(device, non_blocking=True)
        states = batch["states"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        batch_size = latents.size(0)
        num_frames = states.size(1)
        max_horizon = min(int(horizon), num_frames - 1)
        if max_horizon < 1:
            continue

        first_tokens = latents[:, :tokens_per_frame]
        rollout_tokens = first_tokens
        rollout_states = build_rollout_state_context(
            initial_state=states[:, :1],
            actions=actions,
            rollout_steps=max_horizon,
            state_columns=state_columns,
            action_columns=action_columns,
        )

        predictions: list[torch.Tensor] = []
        for step in range(max_horizon):
            with autocast_context(device, amp_dtype):
                pred_tokens = predictor(
                    latent_tokens=rollout_tokens,
                    actions=actions[:, : step + 1],
                    states=rollout_states[:, : step + 1],
                    tokens_per_frame=tokens_per_frame,
                )
            next_tokens = pred_tokens[:, -tokens_per_frame:]
            predictions.append(next_tokens)
            rollout_tokens = torch.cat([rollout_tokens, next_tokens], dim=1)

        for step, predicted in enumerate(predictions, start=1):
            target_start = step * tokens_per_frame
            target = latents[:, target_start : target_start + tokens_per_frame]
            model_l1 = torch.nn.functional.l1_loss(predicted, target, reduction="none").mean(dim=(1, 2))
            identity_l1 = torch.nn.functional.l1_loss(first_tokens, target, reduction="none").mean(dim=(1, 2))
            model_totals[step] = model_totals.get(step, 0.0) + float(model_l1.sum().detach().cpu())
            identity_totals[step] = identity_totals.get(step, 0.0) + float(identity_l1.sum().detach().cpu())
        count += batch_size

    metrics: dict[str, float] = {}
    for step in sorted(model_totals):
        model_l1 = model_totals[step] / max(count, 1)
        identity_l1 = identity_totals[step] / max(count, 1)
        metrics[f"rollout_l1_h{step}"] = model_l1
        metrics[f"identity_l1_h{step}"] = identity_l1
        metrics[f"ratio_h{step}"] = model_l1 / max(identity_l1, 1e-12)
    if max_batches > 0:
        metrics["sampled_batches"] = float(min(max_batches, len(dataloader)))
    metrics["sampled_samples"] = float(count)
    return metrics


def resolve_positive_horizon(value: int, fallback: int, raw_frames_per_sample: int, name: str) -> int:
    horizon = fallback if value == 0 else value
    if horizon < 1:
        raise ValueError(f"{name} must be >= 1")
    if horizon > raw_frames_per_sample - 1:
        raise ValueError(
            f"{name}={horizon} exceeds available future frames={raw_frames_per_sample - 1}"
        )
    return int(horizon)


def action_bounds_for_columns(action_columns: tuple[str, ...]) -> tuple[list[float], list[float]]:
    low_by_column = {
        "steering_cmd_t": settings.STEERING_MIN,
        "throttle_cmd_t": settings.THROTTLE_MIN,
    }
    high_by_column = {
        "steering_cmd_t": settings.STEERING_MAX,
        "throttle_cmd_t": settings.THROTTLE_MAX,
    }
    return (
        [float(low_by_column.get(column, -1.0)) for column in action_columns],
        [float(high_by_column.get(column, 1.0)) for column in action_columns],
    )


def final_prediction_l1(predictions: torch.Tensor, goal_tokens: torch.Tensor) -> float:
    goal = goal_tokens
    if goal.ndim == 2:
        goal = goal.unsqueeze(0)
    return float(F.l1_loss(predictions[:, -1], goal, reduction="none").mean().detach().cpu())


@torch.no_grad()
def final_planning_eval(
    predictor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tokens_per_frame: int,
    horizon: int,
    goal_offset: int,
    max_samples: int,
    state_columns: tuple[str, ...],
    action_columns: tuple[str, ...],
    cem_samples: int,
    cem_elites: int,
    cem_iters: int,
    init_std: float,
    min_std: float,
    action_penalty: float,
    smooth_penalty: float,
    show_progress: bool,
) -> dict[str, float]:
    """Run a small offline CEM planning eval on val using the trained world model."""
    if max_samples < 1:
        return {}
    if horizon < 1:
        raise ValueError("final planning horizon must be >= 1")
    if goal_offset < 1:
        raise ValueError("final planning goal offset must be >= 1")
    if cem_samples < 1:
        raise ValueError("final planning CEM samples must be >= 1")
    if cem_elites < 1 or cem_elites > cem_samples:
        raise ValueError("final planning CEM elites must be in [1, cem_samples]")
    if cem_iters < 1:
        raise ValueError("final planning CEM iters must be >= 1")

    predictor.eval()
    action_low, action_high = action_bounds_for_columns(action_columns)
    split_dataset = dataloader.dataset
    action_normalizer = getattr(split_dataset, "action_normalizer", None)
    planner = RCJepaACFeatureCEMPlanner(
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        state_columns=state_columns,
        action_columns=action_columns,
        action_normalizer=action_normalizer,
        horizon=horizon,
        n_samples=cem_samples,
        n_elite=cem_elites,
        n_iter=cem_iters,
        action_low=action_low,
        action_high=action_high,
        init_std=init_std,
        min_std=min_std,
        action_penalty=action_penalty,
        smooth_penalty=smooth_penalty,
        device=device,
    )

    total_planned_l1 = 0.0
    total_groundtruth_l1 = 0.0
    total_zero_l1 = 0.0
    total_planned_score = 0.0
    total_first_action_mae = 0.0
    total_first_action_mae_by_column = [0.0 for _ in action_columns]
    count = 0

    progress = tqdm(dataloader, desc="final val planning", leave=False, disable=not show_progress)
    for batch in progress:
        if count >= max_samples:
            break
        latents = batch["latents"].to(device, non_blocking=True)
        states = batch["states"].to(device, non_blocking=True)
        model_actions = batch["actions"].to(device, non_blocking=True)
        batch_size = latents.size(0)
        for row in range(batch_size):
            if count >= max_samples:
                break

            context_tokens = latents[row, :tokens_per_frame]
            goal_start = goal_offset * tokens_per_frame
            goal_end = goal_start + tokens_per_frame
            goal_tokens = latents[row, goal_start:goal_end]
            initial_state = states[row, 0]
            plan = planner.plan(
                context_tokens=context_tokens,
                initial_state=initial_state,
                goal_tokens=goal_tokens,
            )

            planned_actions = plan.action_sequence.to(device).unsqueeze(0)
            groundtruth_actions = denormalize_action_tensor(
                model_actions[row : row + 1, :horizon],
                action_columns=action_columns,
                action_normalizer=action_normalizer,
            )
            zero_actions = torch.zeros_like(planned_actions)
            goal_batched = goal_tokens.unsqueeze(0)

            planned_predictions = planner.rollout(context_tokens, initial_state, planned_actions)
            groundtruth_predictions = planner.rollout(context_tokens, initial_state, groundtruth_actions)
            zero_predictions = planner.rollout(context_tokens, initial_state, zero_actions)

            planned_l1 = final_prediction_l1(planned_predictions, goal_batched)
            groundtruth_l1 = final_prediction_l1(groundtruth_predictions, goal_batched)
            zero_l1 = final_prediction_l1(zero_predictions, goal_batched)
            planned_first = plan.first_action
            groundtruth_first = groundtruth_actions[0, 0].detach().cpu()
            first_abs_error = (planned_first - groundtruth_first).abs()

            total_planned_l1 += planned_l1
            total_groundtruth_l1 += groundtruth_l1
            total_zero_l1 += zero_l1
            total_planned_score += float(plan.score)
            total_first_action_mae += float(first_abs_error.mean())
            for index in range(len(action_columns)):
                total_first_action_mae_by_column[index] += float(first_abs_error[index])
            count += 1
            progress.set_postfix(
                {
                    "samples": count,
                    "planned_l1": planned_l1,
                    "gt_l1": groundtruth_l1,
                }
            )

    metrics: dict[str, float] = {"sampled_samples": float(count)}
    if count < 1:
        return metrics

    mean_planned_l1 = total_planned_l1 / count
    mean_groundtruth_l1 = total_groundtruth_l1 / count
    mean_zero_l1 = total_zero_l1 / count
    metrics.update(
        {
            "mean_planned_final_l1": mean_planned_l1,
            "mean_groundtruth_final_l1": mean_groundtruth_l1,
            "mean_zero_action_final_l1": mean_zero_l1,
            "planned_zero_ratio": mean_planned_l1 / max(mean_zero_l1, 1e-12),
            "planned_groundtruth_ratio": mean_planned_l1 / max(mean_groundtruth_l1, 1e-12),
            "mean_planned_score": total_planned_score / count,
            "mean_first_action_mae": total_first_action_mae / count,
            "horizon": float(horizon),
            "goal_offset": float(goal_offset),
            "cem_samples": float(cem_samples),
            "cem_elites": float(cem_elites),
            "cem_iters": float(cem_iters),
        }
    )
    for column, total in zip(action_columns, total_first_action_mae_by_column, strict=True):
        metrics[f"mean_first_action_mae/{column}"] = total / count
    return metrics


def collect_normalization_metadata(dataset: Any) -> dict[str, Any]:
    return {
        "state": normalizer_to_dict(getattr(dataset, "state_normalizer", None)),
        "action": normalizer_to_dict(getattr(dataset, "action_normalizer", None)),
    }


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


def save_checkpoint(
    path: Path,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR | None,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    normalization: dict[str, Any],
    feature_metadata: dict[str, Any],
    best_val_loss: float,
    best_epoch: int,
    global_step: int,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    phase: str = "epoch_complete",
) -> None:
    payload = {
        "epoch": epoch,
        "phase": phase,
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "args": args_to_jsonable_dict(args),
        "metrics": metrics,
        "state_columns": list(args.state_columns),
        "action_columns": list(args.action_columns),
        "normalization": normalization,
        "feature_metadata": feature_metadata,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "note": "Trained from precomputed V-JEPA features. Encoder weights are not saved.",
    }
    torch.save(payload, path)


def load_resume_checkpoint(
    resume_path: Path,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint = torch.load(resume_path, map_location=device)
    validate_resume_predictor_config(checkpoint, args)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def validate_resume_predictor_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    checkpoint_args = dict(checkpoint.get("args", {}))
    legacy_defaults = {
        "raw_frames_per_sample": settings.AC_RAW_FRAMES_PER_SAMPLE,
        "frame_stride": settings.AC_FRAME_STRIDE,
        "target_fps": settings.AC_TARGET_FPS,
        "auto_steps": settings.AC_AUTO_STEPS,
    }
    checked_fields = (
        "predictor_type",
        "raw_frames_per_sample",
        "frame_stride",
        "target_fps",
        "auto_steps",
        "predictor_dim",
        "predictor_depth",
        "predictor_heads",
        "dropout",
        "amp_dtype",
    )
    mismatches = []
    for field in checked_fields:
        if field in checkpoint_args:
            checkpoint_value = checkpoint_args[field]
        elif field in legacy_defaults:
            checkpoint_value = legacy_defaults[field]
        else:
            continue
        current_value = getattr(args, field)
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


def maybe_cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    apply_predictor_size_preset(args)
    if not getattr(args, "_output_dir_was_provided", False):
        suffix_parts = []
        if args.predictor_type != DEFAULT_PREDICTOR_TYPE:
            suffix_parts.append(args.predictor_type)
        if args.model_size != "base":
            suffix_parts.append(args.model_size)
        if suffix_parts:
            args.output_dir = DEFAULT_OUTPUT_DIR.with_name(
                f"{DEFAULT_OUTPUT_DIR.name}_{'_'.join(suffix_parts)}"
            )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir = args.output_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataloaders = create_ac_feature_sequence_dataloaders(
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        manifest_dir=args.manifest_dir,
        raw_frames_per_sample=args.raw_frames_per_sample,
        sequence_stride=args.sequence_stride,
        frame_stride=args.frame_stride,
        target_fps=args.target_fps,
        state_columns=args.state_columns,
        action_columns=args.action_columns,
        include_test=not args.skip_test,
        train_sampler=args.train_sampler,
        eval_sampler=args.eval_sampler,
    )
    train_dataset = dataloaders["train"].dataset
    tokens_per_frame = int(train_dataset.tokens_per_frame)
    embed_dim = int(train_dataset.embed_dim)
    feature_metadata = dict(train_dataset.feature_metadata)

    predictor = build_predictor(args, tokens_per_frame=tokens_per_frame, embed_dim=embed_dim).to(device)
    optimizer = AdamW(
        [parameter for parameter in predictor.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = compute_steps_per_epoch(dataloaders["train"])
    total_train_steps = args.epochs * steps_per_epoch
    warmup_steps = compute_warmup_steps(args.warmup_epochs, steps_per_epoch, total_train_steps)
    normalization_metadata = collect_normalization_metadata(train_dataset)
    run_config = {
        "device": str(device),
        "train_sequences": len(dataloaders["train"].dataset),
        "val_sequences": len(dataloaders["val"].dataset),
        "test_sequences": 0 if args.skip_test else len(dataloaders["test"].dataset),
        "skip_test": bool(args.skip_test),
        "trainable_parameters": count_trainable_parameters(predictor),
        "tokens_per_frame": tokens_per_frame,
        "embed_dim": embed_dim,
        "steps_per_epoch": steps_per_epoch,
        "total_train_steps": total_train_steps,
        "warmup_steps": warmup_steps,
        "train_sampler": args.train_sampler,
        "eval_sampler": args.eval_sampler,
        "normalization": normalization_metadata,
        "feature_metadata": feature_metadata,
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
    resume_phase = "train"
    resumed_train_metrics: dict[str, float] | None = None
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_from is not None:
        resumed_from = args.resume_from
        resume_checkpoint = load_resume_checkpoint(args.resume_from, predictor, optimizer, device, args)
        resume_phase = str(resume_checkpoint.get("phase", "epoch_complete"))
        if resume_phase == "train_complete_waiting_val":
            start_epoch = int(resume_checkpoint["epoch"])
            metrics_payload = resume_checkpoint.get("metrics", {})
            resumed_train_metrics = dict(metrics_payload.get("train", {}))
        else:
            start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val_loss = float(resume_checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))
        history = list(resume_checkpoint.get("history", []))
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

    wandb_run = init_wandb(args, config=run_config, job_type="train-rc-jepa-ac-features")
    watch_model(wandb_run, predictor, args)
    try:
        final_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs + 1):
            final_epoch = epoch
            set_dataloader_epoch(dataloaders["train"], epoch)
            if epoch == start_epoch and resume_phase == "train_complete_waiting_val" and resumed_train_metrics is not None:
                train_metrics = resumed_train_metrics
            else:
                train_metrics, global_step = run_epoch(
                    predictor=predictor,
                    dataloader=dataloaders["train"],
                    device=device,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    grad_clip=args.grad_clip,
                    tokens_per_frame=tokens_per_frame,
                    auto_steps=args.auto_steps,
                    state_columns=tuple(args.state_columns),
                    action_columns=tuple(args.action_columns),
                    label=f"epoch {epoch:03d}/{args.epochs:03d} train",
                    show_progress=not args.no_progress,
                    wandb_run=wandb_run,
                    wandb_prefix="train_batch",
                    wandb_log_every=args.wandb_log_every,
                    epoch=epoch,
                    global_step_start=global_step,
                    wandb_grad_stats_every=args.wandb_grad_stats_every,
                    wandb_param_stats_every=args.wandb_param_stats_every,
                    amp_dtype=args.amp_dtype,
                )
                train_only_metrics = {"epoch": epoch, "train": train_metrics}
                save_checkpoint(
                    args.output_dir / "last_train.pt",
                    predictor,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    train_only_metrics,
                    normalization_metadata,
                    feature_metadata,
                    best_val_loss,
                    best_epoch,
                    global_step,
                    epochs_without_improvement,
                    history,
                    phase="train_complete_waiting_val",
                )
            optimizer.zero_grad(set_to_none=True)
            maybe_cleanup_cuda()
            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    predictor=predictor,
                    dataloader=dataloaders["val"],
                    device=device,
                    optimizer=None,
                    lr_scheduler=None,
                    grad_clip=args.grad_clip,
                    tokens_per_frame=tokens_per_frame,
                    auto_steps=args.auto_steps,
                    state_columns=tuple(args.state_columns),
                    action_columns=tuple(args.action_columns),
                    label=f"epoch {epoch:03d}/{args.epochs:03d} val",
                    show_progress=not args.no_progress,
                    amp_dtype=args.amp_dtype,
                )
            if args.val_rollout_eval_horizon > 0:
                maybe_cleanup_cuda()
                val_rollout_metrics = final_rollout_identity_eval(
                    predictor=predictor,
                    dataloader=dataloaders["val"],
                    device=device,
                    tokens_per_frame=tokens_per_frame,
                    horizon=args.val_rollout_eval_horizon,
                    state_columns=tuple(args.state_columns),
                    action_columns=tuple(args.action_columns),
                    show_progress=not args.no_progress,
                    amp_dtype=args.amp_dtype,
                    max_batches=args.val_rollout_eval_max_batches,
                    label=f"epoch {epoch:03d}/{args.epochs:03d} val rollout",
                )
                val_metrics.update(val_rollout_metrics)

            epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(epoch_metrics)
            write_json(args.output_dir / "history.json", history)
            rollout_suffix = ""
            rollout_steps = sorted(
                int(key.removeprefix("ratio_h"))
                for key in val_metrics
                if key.startswith("ratio_h")
            )
            if rollout_steps:
                rollout_parts: list[str] = []
                for step in rollout_steps:
                    for prefix, precision in (
                        ("rollout_l1", 5),
                        ("identity_l1", 5),
                        ("ratio", 3),
                    ):
                        key = f"{prefix}_h{step}"
                        if key in val_metrics:
                            rollout_parts.append(f"val_{key}={val_metrics[key]:.{precision}f}")
                rollout_suffix = " " + " ".join(rollout_parts)
            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"train_tf={train_metrics['teacher_forcing_loss']:.5f} "
                f"val_tf={val_metrics['teacher_forcing_loss']:.5f} "
                f"train_rollout={train_metrics['rollout_loss']:.5f} "
                f"val_rollout={val_metrics['rollout_loss']:.5f}"
                f"{rollout_suffix}",
                flush=True,
            )

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
            elif should_apply_early_stopping(epoch=epoch, warmup_epochs=args.warmup_epochs):
                epochs_without_improvement += 1

            for checkpoint_path in (args.output_dir / "last.pt", epochs_dir / f"epoch_{epoch:03d}.pt"):
                save_checkpoint(
                    checkpoint_path,
                    predictor,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    epoch_metrics,
                    normalization_metadata,
                    feature_metadata,
                    best_val_loss,
                    best_epoch,
                    global_step,
                    epochs_without_improvement,
                    history,
                    phase="epoch_complete",
                )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt",
                    predictor,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    epoch_metrics,
                    normalization_metadata,
                    feature_metadata,
                    best_val_loss,
                    best_epoch,
                    global_step,
                    epochs_without_improvement,
                    history,
                    phase="epoch_complete",
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

        result = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
            "resume_from": None if resumed_from is None else str(resumed_from),
            "test_skipped": bool(args.skip_test),
        }
        final_eval_metrics: dict[str, float] = {}
        final_planning_metrics: dict[str, float] = {}
        if args.final_eval_horizon > 0 or args.final_planning_eval_samples > 0:
            best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
            predictor.load_state_dict(best_checkpoint["predictor_state_dict"])
            optimizer.zero_grad(set_to_none=True)
            maybe_cleanup_cuda()

        if args.final_eval_horizon > 0:
            final_eval_metrics = final_rollout_identity_eval(
                predictor=predictor,
                dataloader=dataloaders["val"],
                device=device,
                tokens_per_frame=tokens_per_frame,
                horizon=args.final_eval_horizon,
                state_columns=tuple(args.state_columns),
                action_columns=tuple(args.action_columns),
                show_progress=not args.no_progress,
                amp_dtype=args.amp_dtype,
            )
            final_eval_payload = {
                "split": "val",
                "horizon": args.final_eval_horizon,
                "metrics": final_eval_metrics,
            }
            write_json(args.output_dir / "final_eval_val.json", final_eval_payload)
            result["final_eval_val"] = final_eval_metrics
            log_metrics(
                wandb_run,
                flatten_metrics("final_eval_val", final_eval_metrics),
                step=max(global_step, 1),
            )
            update_summary(
                wandb_run,
                {f"final_eval_val/{key}": value for key, value in final_eval_metrics.items()},
            )
        if args.final_planning_eval_samples > 0:
            maybe_cleanup_cuda()
            planning_horizon = resolve_positive_horizon(
                value=args.final_planning_horizon,
                fallback=args.auto_steps,
                raw_frames_per_sample=args.raw_frames_per_sample,
                name="final_planning_horizon",
            )
            planning_goal_offset = resolve_positive_horizon(
                value=args.final_planning_goal_offset,
                fallback=planning_horizon,
                raw_frames_per_sample=args.raw_frames_per_sample,
                name="final_planning_goal_offset",
            )
            final_planning_metrics = final_planning_eval(
                predictor=predictor,
                dataloader=dataloaders["val"],
                device=device,
                tokens_per_frame=tokens_per_frame,
                horizon=planning_horizon,
                goal_offset=planning_goal_offset,
                max_samples=args.final_planning_eval_samples,
                state_columns=tuple(args.state_columns),
                action_columns=tuple(args.action_columns),
                cem_samples=args.final_planning_cem_samples,
                cem_elites=args.final_planning_cem_elites,
                cem_iters=args.final_planning_cem_iters,
                init_std=args.final_planning_init_std,
                min_std=args.final_planning_min_std,
                action_penalty=args.final_planning_action_penalty,
                smooth_penalty=args.final_planning_smooth_penalty,
                show_progress=not args.no_progress,
            )
            final_planning_payload = {
                "split": "val",
                "metrics": final_planning_metrics,
            }
            write_json(args.output_dir / "final_planning_val.json", final_planning_payload)
            result["final_planning_val"] = final_planning_metrics
            log_metrics(
                wandb_run,
                flatten_metrics("final_planning_val", final_planning_metrics),
                step=max(global_step, 1),
            )
            update_summary(
                wandb_run,
                {f"final_planning_val/{key}": value for key, value in final_planning_metrics.items()},
            )
        if args.skip_test:
            write_json(args.output_dir / "final_metrics.json", result)
            log_metrics(wandb_run, {"best/val_loss": best_val_loss}, step=max(global_step, 1))
            update_summary(wandb_run, {"best/val_loss": best_val_loss})
            print(json.dumps(result, indent=2), flush=True)
            return

        best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
        predictor.load_state_dict(best_checkpoint["predictor_state_dict"])
        optimizer.zero_grad(set_to_none=True)
        maybe_cleanup_cuda()
        with torch.no_grad():
            test_metrics, _ = run_epoch(
                predictor=predictor,
                dataloader=dataloaders["test"],
                device=device,
                optimizer=None,
                lr_scheduler=None,
                grad_clip=args.grad_clip,
                tokens_per_frame=tokens_per_frame,
                auto_steps=args.auto_steps,
                state_columns=tuple(args.state_columns),
                action_columns=tuple(args.action_columns),
                label="test",
                show_progress=not args.no_progress,
                amp_dtype=args.amp_dtype,
            )

        result["test"] = test_metrics
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
