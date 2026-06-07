"""Weights & Biases helpers for training scripts."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


DEFAULT_WANDB_PROJECT = "nn-jepa-rc"
DEFAULT_WANDB_WATCH_LOG = "gradients"
DEFAULT_WANDB_WATCH_FREQ = 200
DEFAULT_WANDB_GRAD_STATS_EVERY = 20
DEFAULT_WANDB_PARAM_STATS_EVERY = 200


def add_wandb_args(parser: Any) -> None:
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-log-every", type=int, default=20)
    parser.add_argument("--wandb-watch-log", choices=["none", "gradients", "parameters", "all"], default=DEFAULT_WANDB_WATCH_LOG)
    parser.add_argument("--wandb-watch-freq", type=int, default=DEFAULT_WANDB_WATCH_FREQ)
    parser.add_argument("--wandb-grad-stats-every", type=int, default=DEFAULT_WANDB_GRAD_STATS_EVERY)
    parser.add_argument("--wandb-param-stats-every", type=int, default=DEFAULT_WANDB_PARAM_STATS_EVERY)


def init_wandb(args: Any, config: dict[str, Any], job_type: str) -> Any | None:
    if args.no_wandb or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb`, "
            "or run training with `--no-wandb`."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        job_type=job_type,
        tags=args.wandb_tags,
        config=config,
    )


def log_metrics(run: Any | None, metrics: dict[str, float], step: int | None = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def update_summary(run: Any | None, values: dict[str, float]) -> None:
    if run is None:
        return
    for key, value in values.items():
        run.summary[key] = value


def finish_wandb(run: Any | None) -> None:
    if run is not None:
        run.finish()


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def watch_model(run: Any | None, model: nn.Module, args: Any) -> None:
    if run is None or getattr(args, "wandb_watch_log", "none") == "none":
        return
    run.watch(
        model,
        log=args.wandb_watch_log,
        log_freq=max(int(args.wandb_watch_freq), 1),
        log_graph=False,
    )


def collect_gradient_metrics(model: nn.Module, prefix: str = "grad") -> dict[str, float]:
    return collect_tensor_metrics(
        model=model,
        prefix=prefix,
        tensor_getter=lambda parameter: parameter.grad,
        trainable_only=True,
    )


def collect_parameter_metrics(model: nn.Module, prefix: str = "param") -> dict[str, float]:
    return collect_tensor_metrics(
        model=model,
        prefix=prefix,
        tensor_getter=lambda parameter: parameter,
        trainable_only=True,
    )


def collect_tensor_metrics(
    model: nn.Module,
    prefix: str,
    tensor_getter: Any,
    trainable_only: bool,
) -> dict[str, float]:
    total_sq = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    value_count = 0
    tensor_count = 0
    missing_count = 0
    zero_count = 0
    nonfinite_count = 0
    group_sq: dict[str, float] = {}

    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue

        tensor_count += 1
        tensor = tensor_getter(parameter)
        if tensor is None:
            missing_count += 1
            continue

        values = tensor.detach()
        if values.numel() == 0:
            continue

        finite_mask = torch.isfinite(values)
        if not bool(finite_mask.all()):
            nonfinite_count += int((~finite_mask).sum().item())
            values = values[finite_mask]
            if values.numel() == 0:
                continue

        values = values.float()
        abs_values = values.abs()
        sq_sum = float(torch.sum(values * values).item())
        group = metric_group_name(name)

        total_sq += sq_sum
        group_sq[group] = group_sq.get(group, 0.0) + sq_sum
        abs_sum += float(torch.sum(abs_values).item())
        max_abs = max(max_abs, float(torch.max(abs_values).item()))
        zero_count += int((values == 0).sum().item())
        value_count += int(values.numel())

    metrics = {
        f"{prefix}/global_l2": math.sqrt(total_sq),
        f"{prefix}/mean_abs": abs_sum / max(value_count, 1),
        f"{prefix}/max_abs": max_abs,
        f"{prefix}/value_count": float(value_count),
        f"{prefix}/tensor_count": float(tensor_count),
        f"{prefix}/missing_tensor_count": float(missing_count),
        f"{prefix}/zero_value_count": float(zero_count),
        f"{prefix}/nonfinite_count": float(nonfinite_count),
    }
    for group, sq_sum in sorted(group_sq.items()):
        metrics[f"{prefix}_norm/{group}"] = math.sqrt(sq_sum)
    return metrics


def metric_group_name(name: str) -> str:
    if name.startswith("predictor."):
        name = name[len("predictor.") :]
    first = name.split(".", 1)[0]
    if first == "blocks":
        return "blocks"
    return first
