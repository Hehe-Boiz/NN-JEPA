"""Evaluate whether an RC JEPA-AC predictor reacts to action tokens.

This is an eval-only diagnostic. It keeps latent/state/target context fixed and
reruns the predictor with ablated action sequences.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import settings
from data.feature_sequence_dataset import create_ac_feature_sequence_dataloaders
from data.normalization import FeatureNormalizer
from tools.rc_jepa_ac_feature_runtime import (
    DEFAULT_FEATURES_DIR,
    build_predictor_from_checkpoint,
    checkpoint_default_path,
    default_device,
    load_feature_checkpoint,
    validate_feature_metadata,
)
from models.rc_jepa_ac import normalize_rollout_feedback
from tools.train_rc_jepa_ac_features import (
    ROLLOUT_EVAL_STATE_MODE_FALLBACK,
    ROLLOUT_EVAL_STATE_MODE_MEASURED,
    autocast_context,
    build_rollout_eval_state_context,
    write_json,
)
from tools.rc_jepa_ac_cem_planner import controllable_action_indices, make_zero_control_actions
from tools.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics, update_summary


DEFAULT_CHECKPOINT = Path("checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt")
SUPPORTED_ROLLOUT_STATE_MODES = (ROLLOUT_EVAL_STATE_MODE_MEASURED, ROLLOUT_EVAL_STATE_MODE_FALLBACK)
ACTION_VARIANT_ZERO = "zero"
ACTION_VARIANT_SHUFFLE = "shuffle"
ACTION_VARIANT_OPPOSITE = "opposite"
ACTION_VARIANT_CANONICAL = "canonical"
SUPPORTED_ACTION_VARIANTS = (
    ACTION_VARIANT_ZERO,
    ACTION_VARIANT_SHUFFLE,
    ACTION_VARIANT_OPPOSITE,
    ACTION_VARIANT_CANONICAL,
)
DEFAULT_ACTION_VARIANTS = (ACTION_VARIANT_ZERO, ACTION_VARIANT_SHUFFLE, ACTION_VARIANT_OPPOSITE)
CANONICAL_RAW_ACTIONS = {
    "canonical_left": {"steering_cmd_t": -0.7, "throttle_cmd_t": 0.3},
    "canonical_right": {"steering_cmd_t": 0.7, "throttle_cmd_t": 0.3},
    "canonical_straight": {"steering_cmd_t": 0.0, "throttle_cmd_t": 0.3},
}
EPS = 1e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only action sensitivity diagnostic for RC JEPA-AC feature checkpoints."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--eval-batch-size", type=int, default=settings.AC_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--max-batches", type=int, default=256, help="0 means full split.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--rollout-state-mode",
        choices=SUPPORTED_ROLLOUT_STATE_MODES,
        default=ROLLOUT_EVAL_STATE_MODE_MEASURED,
        help="measured uses measured future states; fallback uses fixed fallback states built from normal actions.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=SUPPORTED_ACTION_VARIANTS,
        default=list(DEFAULT_ACTION_VARIANTS),
        help="Action ablations to compare against normal actions. `canonical` expands to left/right/straight.",
    )
    parser.add_argument("--amp-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    add_wandb_args(parser)
    return parser.parse_args(argv)


def normalize_horizons(horizons: Iterable[int], raw_frames_per_sample: int) -> list[int]:
    normalized = sorted(set(int(horizon) for horizon in horizons))
    if not normalized:
        raise ValueError("At least one horizon is required")
    max_available = int(raw_frames_per_sample) - 1
    invalid = [horizon for horizon in normalized if horizon < 1 or horizon > max_available]
    if invalid:
        raise ValueError(f"horizons must be in [1, {max_available}], got {invalid}")
    return normalized


def compute_action_gain(variant_loss: float, normal_loss: float) -> float:
    return float(variant_loss) / max(float(normal_loss), EPS)


def build_action_variants(
    actions: torch.Tensor,
    action_columns: Sequence[str],
    variants: Sequence[str],
    action_normalizer: FeatureNormalizer | None = None,
    shuffle_seed: int = 0,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Create ablated action tensors without mutating the input actions."""
    action_columns = tuple(action_columns)
    requested = tuple(dict.fromkeys(variants))
    built: dict[str, torch.Tensor] = {}
    warnings_out: list[str] = []

    if ACTION_VARIANT_ZERO in requested:
        built[ACTION_VARIANT_ZERO] = make_zero_control_actions(actions, action_columns)

    if ACTION_VARIANT_SHUFFLE in requested:
        built[ACTION_VARIANT_SHUFFLE] = shuffle_actions_by_batch(
            actions,
            action_columns=action_columns,
            seed=shuffle_seed,
        )

    if ACTION_VARIANT_OPPOSITE in requested:
        if "steering_cmd_t" not in action_columns:
            warnings_out.append("Skipping opposite variant because action_columns has no steering_cmd_t")
        else:
            steering_index = action_columns.index("steering_cmd_t")
            opposite = actions.clone()
            opposite[..., steering_index] = -opposite[..., steering_index]
            built[ACTION_VARIANT_OPPOSITE] = opposite

    if ACTION_VARIANT_CANONICAL in requested:
        built_canonical, canonical_warnings = build_canonical_action_variants(
            actions=actions,
            action_columns=action_columns,
            action_normalizer=action_normalizer,
        )
        built.update(built_canonical)
        warnings_out.extend(canonical_warnings)

    return built, warnings_out


def shuffle_actions_by_batch(
    actions: torch.Tensor,
    action_columns: Sequence[str],
    seed: int,
) -> torch.Tensor:
    if actions.size(0) < 2:
        return actions.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(actions.size(0), generator=generator).to(actions.device)
    if torch.equal(permutation, torch.arange(actions.size(0), device=actions.device)):
        permutation = torch.roll(permutation, shifts=1)
    shuffled = actions.clone()
    for index in controllable_action_indices(action_columns):
        shuffled[..., index] = actions.index_select(0, permutation)[..., index]
    return shuffled


def build_canonical_action_variants(
    actions: torch.Tensor,
    action_columns: Sequence[str],
    action_normalizer: FeatureNormalizer | None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    action_columns = tuple(action_columns)
    required = ("steering_cmd_t", "throttle_cmd_t")
    missing = [column for column in required if column not in action_columns]
    if missing:
        return {}, [f"Skipping canonical variants because action_columns is missing {missing}"]

    built: dict[str, torch.Tensor] = {}
    for name, raw_values in CANONICAL_RAW_ACTIONS.items():
        canonical = actions.clone()
        for column, raw_value in raw_values.items():
            column_index = action_columns.index(column)
            value = (
                action_normalizer.normalize_value(column, raw_value)
                if action_normalizer is not None
                else float(raw_value)
            )
            canonical[..., column_index] = float(value)
        built[name] = canonical
    return built, []


@torch.no_grad()
def rollout_predictions(
    predictor: torch.nn.Module,
    latents: torch.Tensor,
    fixed_state_context: torch.Tensor,
    actions: torch.Tensor,
    tokens_per_frame: int,
    max_horizon: int,
    amp_dtype: str,
    rollout_feedback_norm: bool = False,
) -> list[torch.Tensor]:
    """Autoregressively predict z_1..z_H with fixed state context."""
    rollout_tokens = latents[:, :tokens_per_frame]
    predictions: list[torch.Tensor] = []
    for step in range(max_horizon):
        with autocast_context(latents.device, amp_dtype):
            pred_tokens = predictor(
                latent_tokens=rollout_tokens,
                actions=actions[:, : step + 1],
                states=fixed_state_context[:, : step + 1],
                tokens_per_frame=tokens_per_frame,
        )
        next_tokens = pred_tokens[:, -tokens_per_frame:]
        predictions.append(next_tokens)
        feedback_tokens = normalize_rollout_feedback(next_tokens, enabled=rollout_feedback_norm)
        rollout_tokens = torch.cat([rollout_tokens, feedback_tokens], dim=1)
    return predictions


@torch.no_grad()
def evaluate_action_sensitivity_batch(
    predictor: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    tokens_per_frame: int,
    horizons: Sequence[int],
    action_columns: Sequence[str],
    state_columns: Sequence[str],
    rollout_state_mode: str,
    variants: Sequence[str],
    action_normalizer: FeatureNormalizer | None = None,
    amp_dtype: str = "fp32",
    shuffle_seed: int = 0,
    rollout_feedback_norm: bool = False,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[int, torch.Tensor], list[str]]:
    latents = batch["latents"].to(device, non_blocking=True)
    states = batch["states"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    max_horizon = min(max(int(horizon) for horizon in horizons), states.size(1) - 1)
    if max_horizon < 1:
        return {}, {}, ["Skipping batch because it has no future frame"]

    fixed_state_context = build_rollout_eval_state_context(
        mode=rollout_state_mode,
        states=states,
        actions=actions,
        rollout_steps=max_horizon,
        state_columns=tuple(state_columns),
        action_columns=tuple(action_columns),
    )
    action_variants, warnings_out = build_action_variants(
        actions=actions,
        action_columns=action_columns,
        variants=variants,
        action_normalizer=action_normalizer,
        shuffle_seed=shuffle_seed,
    )
    all_actions = {"normal": actions, **action_variants}

    predictions_by_variant: dict[str, dict[int, torch.Tensor]] = {}
    for variant_name, variant_actions in all_actions.items():
        predictions = rollout_predictions(
            predictor=predictor,
            latents=latents,
            fixed_state_context=fixed_state_context,
            actions=variant_actions,
            tokens_per_frame=tokens_per_frame,
            max_horizon=max_horizon,
            amp_dtype=amp_dtype,
            rollout_feedback_norm=rollout_feedback_norm,
        )
        predictions_by_variant[variant_name] = {
            horizon: predictions[horizon - 1]
            for horizon in horizons
            if horizon <= max_horizon
        }

    targets = {
        horizon: latents[:, horizon * tokens_per_frame : (horizon + 1) * tokens_per_frame]
        for horizon in horizons
        if horizon <= max_horizon
    }
    return predictions_by_variant, targets, warnings_out


@torch.no_grad()
def evaluate_action_sensitivity(
    predictor: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tokens_per_frame: int,
    horizons: Sequence[int],
    action_columns: Sequence[str],
    state_columns: Sequence[str],
    rollout_state_mode: str,
    variants: Sequence[str],
    max_batches: int,
    show_progress: bool,
    amp_dtype: str = "fp32",
    rollout_feedback_norm: bool = False,
) -> tuple[dict[str, float], list[str]]:
    predictor.eval()
    action_normalizer = getattr(dataloader.dataset, "action_normalizer", None)
    totals: dict[str, dict[int, float]] = {}
    counts: dict[int, int] = {int(horizon): 0 for horizon in horizons}
    warnings_out: list[str] = []

    progress = tqdm(dataloader, desc="action sensitivity", leave=False, disable=not show_progress)
    for batch_index, batch in enumerate(progress, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        predictions_by_variant, targets, batch_warnings = evaluate_action_sensitivity_batch(
            predictor=predictor,
            batch=batch,
            device=device,
            tokens_per_frame=tokens_per_frame,
            horizons=horizons,
            action_columns=action_columns,
            state_columns=state_columns,
            rollout_state_mode=rollout_state_mode,
            variants=variants,
            action_normalizer=action_normalizer,
            amp_dtype=amp_dtype,
            shuffle_seed=batch_index,
            rollout_feedback_norm=rollout_feedback_norm,
        )
        warnings_out.extend(batch_warnings)
        if not predictions_by_variant:
            continue

        latents = batch["latents"].to(device, non_blocking=True)
        first_tokens = latents[:, :tokens_per_frame]
        batch_size = latents.size(0)
        normal_predictions = predictions_by_variant["normal"]
        for horizon, target in targets.items():
            counts[horizon] += batch_size
            normal_loss = per_sample_l1(normal_predictions[horizon], target)
            identity_loss = per_sample_l1(first_tokens, target)
            add_total(totals, "loss_normal", horizon, normal_loss)
            add_total(totals, "identity_loss", horizon, identity_loss)
            for variant_name, predictions_by_horizon in predictions_by_variant.items():
                if variant_name == "normal" or horizon not in predictions_by_horizon:
                    continue
                variant_prediction = predictions_by_horizon[horizon]
                add_total(totals, f"loss_{variant_name}", horizon, per_sample_l1(variant_prediction, target))
                add_total(
                    totals,
                    f"diff_normal_{variant_name}",
                    horizon,
                    per_sample_l1(normal_predictions[horizon], variant_prediction),
                )

    metrics: dict[str, float] = {}
    for horizon in sorted(counts):
        count = max(counts[horizon], 1)
        normal_loss = totals.get("loss_normal", {}).get(horizon, 0.0) / count
        metrics[f"action_sens/loss_normal_h{horizon}"] = normal_loss
        identity_loss = totals.get("identity_loss", {}).get(horizon, 0.0) / count
        metrics[f"action_sens/identity_loss_h{horizon}"] = identity_loss
        metrics[f"action_sens/model_vs_identity_ratio_h{horizon}"] = compute_action_gain(
            normal_loss,
            identity_loss,
        )
        for total_name, values_by_horizon in sorted(totals.items()):
            if not total_name.startswith("loss_") or total_name == "loss_normal":
                continue
            if horizon not in values_by_horizon:
                continue
            variant_name = total_name.removeprefix("loss_")
            variant_loss = values_by_horizon[horizon] / count
            metrics[f"action_sens/loss_{variant_name}_h{horizon}"] = variant_loss
            metrics[f"action_sens/gain_{variant_name}_h{horizon}"] = compute_action_gain(
                variant_loss,
                normal_loss,
            )
        for total_name, values_by_horizon in sorted(totals.items()):
            if not total_name.startswith("diff_normal_") or horizon not in values_by_horizon:
                continue
            metric_name = total_name.removeprefix("diff_")
            metrics[f"action_sens/diff_{metric_name}_h{horizon}"] = values_by_horizon[horizon] / count
        metrics[f"action_sens/sampled_samples_h{horizon}"] = float(counts[horizon])

    return metrics, dedupe_preserve_order(warnings_out)


def per_sample_l1(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(left, right, reduction="none").mean(dim=(1, 2))


def add_total(totals: dict[str, dict[int, float]], name: str, horizon: int, values: torch.Tensor) -> None:
    values_by_horizon = totals.setdefault(name, {})
    values_by_horizon[horizon] = values_by_horizon.get(horizon, 0.0) + float(values.sum().detach().cpu())


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def interpret_action_sensitivity(metrics: dict[str, float], horizons: Sequence[int]) -> list[str]:
    lines: list[str] = []
    for horizon in horizons:
        gains = [
            value
            for key, value in metrics.items()
            if key.startswith("action_sens/gain_")
            and key.endswith(f"_h{horizon}")
            and "identity" not in key
        ]
        diffs = [
            value
            for key, value in metrics.items()
            if key.startswith("action_sens/diff_normal_") and key.endswith(f"_h{horizon}")
        ]
        normal_loss = metrics.get(f"action_sens/loss_normal_h{horizon}", 0.0)
        if not gains:
            continue
        max_gain = max(gains)
        max_diff = max(diffs) if diffs else 0.0
        relative_diff = max_diff / max(normal_loss, EPS)
        if max_gain <= 1.05 and relative_diff <= 0.05:
            lines.append(
                f"h{horizon}: Model likely ignores action or action has weak effect at this horizon."
            )
        elif max_gain >= 1.10:
            lines.append(
                f"h{horizon}: Normal actions improve prediction compared to ablated actions."
            )
        elif relative_diff > 0.10:
            lines.append(
                f"h{horizon}: Model reacts to action, but loss gain is weak; reaction may not be causally correct."
            )
        else:
            lines.append(f"h{horizon}: Action effect is present but modest.")

    if len(horizons) >= 2:
        first_horizon = min(horizons)
        last_horizon = max(horizons)
        first_gains = [
            value
            for key, value in metrics.items()
            if key.startswith("action_sens/gain_") and key.endswith(f"_h{first_horizon}")
        ]
        last_gains = [
            value
            for key, value in metrics.items()
            if key.startswith("action_sens/gain_") and key.endswith(f"_h{last_horizon}")
        ]
        if first_gains and last_gains and max(first_gains) <= 1.05 and max(last_gains) > 1.05:
            lines.append("Action effect may be delayed: short horizon is weak, longer horizon is stronger.")
    return lines


def maybe_cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        print("[warning] Running action sensitivity on CPU; this may be slow.", flush=True)

    checkpoint, checkpoint_path = load_feature_checkpoint(args.checkpoint, device)
    checkpoint_phase = str(checkpoint.get("phase", "epoch_complete"))
    if checkpoint_phase == "train_in_progress":
        print("[warning] Checkpoint phase is train_in_progress; prefer best.pt or last.pt for eval.", flush=True)

    predictor, config = build_predictor_from_checkpoint(checkpoint, device)
    horizons = normalize_horizons(args.horizons, config.raw_frames_per_sample)
    features_dir = args.features_dir or checkpoint_default_path(checkpoint, "features_dir", DEFAULT_FEATURES_DIR)
    manifest_dir = args.manifest_dir or checkpoint_default_path(checkpoint, "manifest_dir", settings.MANIFEST_DIR)
    include_test = args.split == "test"
    output_json = args.output_json or checkpoint_path.parent / f"action_sensitivity_{args.split}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

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
        include_test=include_test,
        eval_sampler="session",
    )
    validate_feature_metadata(dataloaders["train"].dataset.feature_metadata, config.feature_metadata)

    run_config: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_phase": checkpoint_phase,
        "features_dir": str(features_dir),
        "manifest_dir": str(manifest_dir),
        "split": args.split,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "horizons": horizons,
        "rollout_state_mode": args.rollout_state_mode,
        "variants": list(args.variants),
        "amp_dtype": args.amp_dtype,
        "rollout_feedback_norm": config.rollout_feedback_norm,
        "device": str(device),
        "fixed_state_context": True,
        "zero_variant_note": "zero is in model action space; with normalized actions this means train-set mean command.",
        "model": config.to_jsonable_dict(),
        "sequence_count": len(dataloaders[args.split].dataset),
    }
    print(json.dumps(run_config, indent=2), flush=True)

    wandb_run = init_wandb(args, config=run_config, job_type="eval-rc-jepa-ac-action-sensitivity")
    try:
        maybe_cleanup_cuda()
        with torch.inference_mode():
            metrics, warnings_out = evaluate_action_sensitivity(
                predictor=predictor,
                dataloader=dataloaders[args.split],
                device=device,
                tokens_per_frame=config.tokens_per_frame,
                horizons=horizons,
                action_columns=config.action_columns,
                state_columns=config.state_columns,
                rollout_state_mode=args.rollout_state_mode,
                variants=args.variants,
                max_batches=args.max_batches,
                show_progress=not args.no_progress,
                amp_dtype=args.amp_dtype,
                rollout_feedback_norm=config.rollout_feedback_norm,
            )

        interpretation = interpret_action_sensitivity(metrics, horizons)
        for warning_text in warnings_out:
            warnings.warn(warning_text, RuntimeWarning, stacklevel=1)
        result = {
            **run_config,
            "metrics": metrics,
            "warnings": warnings_out,
            "interpretation": interpretation,
        }
        write_json(output_json, result)
        log_metrics(wandb_run, metrics, step=1)
        update_summary(wandb_run, metrics)
        print(json.dumps({"output_json": str(output_json), **result}, indent=2), flush=True)
        if interpretation:
            print("INTERPRETATION:", flush=True)
            for line in interpretation:
                print(f"- {line}", flush=True)
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
