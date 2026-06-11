"""Colab experiment: domain-adapt V-JEPA encoder, then train online AC predictor.

This is intentionally separate from the stable feature-cache training path.  It is
for large-VRAM experiments where the encoder is run online instead of reading
precomputed .npy features.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data import settings
from data.dataset import image_to_tensor, load_manifest, normalize_tensor
from data.normalization import FeatureNormalizer, build_feature_normalizer, normalizer_to_dict
from data.sequence_dataset import (
    DEFAULT_AC_ACTION_COLUMNS,
    DEFAULT_AC_STATE_COLUMNS,
    build_sequence_windows,
    timestamp_to_float,
)
from models.rc_jepa_ac import (
    DEFAULT_CHECKPOINT_KEY,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PREDICTOR_TYPE,
    DEFAULT_ROLLOUT_STATE_MODE,
    PREDICTOR_SIZE_PRESETS,
    SUPPORTED_ROLLOUT_STATE_MODES,
    SUPPORTED_PREDICTOR_TYPES,
    build_ac_predictor,
    clean_state_dict_keys,
    compute_world_model_losses,
    count_trainable_parameters,
    extract_checkpoint_state,
)
from models.vjepa21_presets import DEFAULT_VJEPA21_ENCODER_NAME, SUPPORTED_VJEPA21_ENCODER_NAMES
from tools.train_rc_jepa_ac import (
    build_lr_scheduler,
    compute_steps_per_epoch,
    compute_warmup_steps,
    should_apply_early_stopping,
    sync_lr_scheduler,
)


class SourceFrameACSequenceDataset(Dataset):
    """Sequence dataset that reads raw `source_frame_path` before falling back to processed frames."""

    def __init__(
        self,
        split: str,
        manifest_dir: str | Path,
        image_size: int,
        raw_frames_per_sample: int,
        sequence_stride: int,
        frame_stride: int,
        target_fps: float,
        state_columns: Sequence[str],
        action_columns: Sequence[str],
        augment: bool,
        state_normalizer: FeatureNormalizer | None,
        action_normalizer: FeatureNormalizer | None,
        path_rewrite_from: str,
        path_rewrite_to: str,
        max_items: int = 0,
    ) -> None:
        self.split = split
        self.manifest_path = Path(manifest_dir) / f"{split}.jsonl"
        self.image_size = int(image_size)
        self.raw_frames_per_sample = int(raw_frames_per_sample)
        self.state_columns = tuple(state_columns)
        self.action_columns = tuple(action_columns)
        self.augment = bool(augment)
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.path_rewrite_from = str(path_rewrite_from or "")
        self.path_rewrite_to = str(path_rewrite_to or "")
        self.samples = load_manifest(self.manifest_path)
        self.windows = build_sequence_windows(
            self.samples,
            raw_frames_per_sample=raw_frames_per_sample,
            sequence_stride=sequence_stride,
            frame_stride=frame_stride,
            target_fps=target_fps,
            state_columns=state_columns,
            action_columns=action_columns,
            max_frame_index_gap=settings.AC_MAX_FRAME_INDEX_GAP,
            max_time_gap_sec=settings.AC_MAX_TIME_GAP_SEC,
        )
        if max_items > 0:
            self.windows = self.windows[: int(max_items)]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_indices = self.windows[index]
        sequence = [self.samples[sample_index] for sample_index in sample_indices]
        images = [self.load_image(sample) for sample in sequence]
        if self.augment:
            images = augment_sequence_images(images)

        image_tensors = [
            normalize_tensor(
                image_to_tensor(image),
                mean=list(settings.NORMALIZE_MEAN),
                std=list(settings.NORMALIZE_STD),
            )
            for image in images
        ]
        states = [dict(sample["state"]) for sample in sequence]
        actions = [dict(sample["action"]) for sample in sequence[:-1]]
        if self.state_normalizer is None:
            state_values = [[state[column] for column in self.state_columns] for state in states]
        else:
            state_values = [self.state_normalizer.normalize_row(state, self.state_columns) for state in states]
        if self.action_normalizer is None:
            action_values = [[action[column] for column in self.action_columns] for action in actions]
        else:
            action_values = [self.action_normalizer.normalize_row(action, self.action_columns) for action in actions]

        return {
            "images": torch.stack(image_tensors, dim=1).contiguous(),
            "states": torch.tensor(state_values, dtype=torch.float32),
            "actions": torch.tensor(action_values, dtype=torch.float32),
            "session_id": str(sequence[0]["session_id"]),
            "frame_indices": torch.tensor([sample["frame_index"] for sample in sequence], dtype=torch.long),
            "timestamps_sec": torch.tensor(
                [timestamp_to_float(sample.get("timestamp_sec")) for sample in sequence],
                dtype=torch.float32,
            ),
        }

    def load_image(self, sample: dict[str, Any]) -> Image.Image:
        raw_value = str(sample.get("source_frame_path") or sample.get("frame_path"))
        if self.path_rewrite_from and raw_value.startswith(self.path_rewrite_from):
            raw_value = self.path_rewrite_to + raw_value[len(self.path_rewrite_from) :]
        path = Path(raw_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing image for {sample.get('sample_id')}: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        if rgb.size != (self.image_size, self.image_size):
            rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return rgb


def augment_sequence_images(images: list[Image.Image]) -> list[Image.Image]:
    """Photometric-only sequence augmentation; no horizontal flip by default."""
    out = list(images)
    if settings.BRIGHTNESS_JITTER > 0:
        factor = random.uniform(1.0 - settings.BRIGHTNESS_JITTER, 1.0 + settings.BRIGHTNESS_JITTER)
        out = [ImageEnhance.Brightness(image).enhance(factor) for image in out]
    if settings.CONTRAST_JITTER > 0:
        factor = random.uniform(1.0 - settings.CONTRAST_JITTER, 1.0 + settings.CONTRAST_JITTER)
        out = [ImageEnhance.Contrast(image).enhance(factor) for image in out]
    if settings.SATURATION_JITTER > 0:
        factor = random.uniform(1.0 - settings.SATURATION_JITTER, 1.0 + settings.SATURATION_JITTER)
        out = [ImageEnhance.Color(image).enhance(factor) for image in out]
    if settings.GAUSSIAN_BLUR_PROB > 0.0 and random.random() < settings.GAUSSIAN_BLUR_PROB:
        out = [image.filter(ImageFilter.GaussianBlur(radius=settings.GAUSSIAN_BLUR_RADIUS)) for image in out]
    return out


def build_dataloaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    train_samples = load_manifest(Path(args.manifest_dir) / "train.jsonl")
    state_normalizer = (
        build_feature_normalizer(train_samples, args.state_columns, source_key="state")
        if args.normalize_state
        else None
    )
    action_normalizer = (
        build_feature_normalizer(train_samples, args.action_columns, source_key="action")
        if args.normalize_action
        else None
    )
    common = dict(
        manifest_dir=args.manifest_dir,
        image_size=args.image_size,
        raw_frames_per_sample=args.raw_frames_per_sample,
        sequence_stride=args.sequence_stride,
        frame_stride=args.frame_stride,
        target_fps=args.target_fps,
        state_columns=args.state_columns,
        action_columns=args.action_columns,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        path_rewrite_from=args.path_rewrite_from,
        path_rewrite_to=args.path_rewrite_to,
    )
    train_dataset = SourceFrameACSequenceDataset(
        split="train",
        augment=args.augment,
        max_items=args.max_train_windows,
        **common,
    )
    val_dataset = SourceFrameACSequenceDataset(
        split="val",
        augment=False,
        max_items=args.max_val_windows,
        **common,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        ),
    }
    normalization = {
        "state": normalizer_to_dict(state_normalizer),
        "action": normalizer_to_dict(action_normalizer),
    }
    return loaders, normalization


def build_vjepa_encoder(args: argparse.Namespace) -> nn.Module:
    vjepa_root = Path(args.vjepa_root)
    if str(vjepa_root) not in sys.path:
        sys.path.insert(0, str(vjepa_root))
    from app.vjepa_2_1.models import vision_transformer as vjepa_vit

    builder_name = {
        "vit_small_384": "vit_small",
        "vit_base_384": "vit_base",
        "vit_large_384": "vit_large",
        "vit_giant_384": "vit_giant_xformers",
        "vit_gigantic_384": "vit_gigantic_xformers",
    }[args.encoder]
    builder = getattr(vjepa_vit, builder_name)
    encoder = builder(
        img_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
        num_frames=args.tubelet_size,
        tubelet_size=args.tubelet_size,
        use_sdpa=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    checkpoint = torch.load(args.vjepa_checkpoint, map_location="cpu")
    state = extract_checkpoint_state(checkpoint, checkpoint_key=args.checkpoint_key)
    msg = encoder.load_state_dict(clean_state_dict_keys(state), strict=not args.allow_partial_checkpoint)
    if args.allow_partial_checkpoint:
        print(
            {
                "missing_count": len(getattr(msg, "missing_keys", [])),
                "unexpected_count": len(getattr(msg, "unexpected_keys", [])),
            },
            flush=True,
        )
    return encoder


def freeze_encoder_for_phase1(encoder: nn.Module, train_last_n_blocks: int) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    blocks = list(getattr(encoder, "blocks", []))
    if train_last_n_blocks <= 0 or not blocks:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        return
    for block in blocks[-train_last_n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for name in ("norm", "fc_norm"):
        module = getattr(encoder, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


class TokenPredictionHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def encode_images(
    encoder: nn.Module,
    images: torch.Tensor,
    tubelet_size: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode [B,C,T,H,W] into [B,T,N,D]."""
    if images.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(images.shape)}")
    batch_size, channels, frames_count, height, width = images.shape
    frames = images.permute(0, 2, 1, 3, 4).reshape(batch_size * frames_count, channels, height, width)
    pseudo_clips = frames.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)
    tokens = encoder(pseudo_clips)
    if normalize:
        tokens = F.layer_norm(tokens, (tokens.size(-1),))
    return tokens.view(batch_size, frames_count, tokens.size(1), tokens.size(-1))


def flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    bsz, frames_count, tokens_per_frame, dim = tokens.shape
    return tokens.reshape(bsz, frames_count * tokens_per_frame, dim)


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "fp32":
        return nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)


def average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def train_encoder_epoch(
    student: nn.Module,
    teacher: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    label: str,
) -> dict[str, float]:
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    head.train(training)
    totals = {"loss": 0.0, "temporal_loss": 0.0, "anchor_loss": 0.0}
    count = 0
    progress = tqdm(dataloader, desc=label, leave=False)
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast_context(device, args.amp_dtype):
                student_tokens = encode_images(student, images, args.tubelet_size)
                with torch.no_grad():
                    teacher_tokens = encode_images(teacher, images, args.tubelet_size)
                pred_next = head(student_tokens[:, :-1])
                target_next = teacher_tokens[:, 1:].detach()
                temporal_loss = F.l1_loss(pred_next, target_next)
                anchor_loss = F.l1_loss(student_tokens, teacher_tokens.detach())
                loss = temporal_loss + args.phase1_anchor_weight * anchor_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in list(student.parameters()) + list(head.parameters()) if parameter.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            if args.phase1_ema_decay > 0:
                update_ema(student, teacher, args.phase1_ema_decay)
        bsz = images.size(0)
        totals["loss"] += float(loss.detach().item()) * bsz
        totals["temporal_loss"] += float(temporal_loss.detach().item()) * bsz
        totals["anchor_loss"] += float(anchor_loss.detach().item()) * bsz
        count += bsz
        progress.set_postfix(average_metrics(totals, count))
    return average_metrics(totals, count)


def save_encoder_checkpoint(
    path: Path,
    student: nn.Module,
    teacher: nn.Module,
    head: nn.Module,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    normalization: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder": student.state_dict(),
            "ema_encoder": teacher.state_dict(),
            "phase1_head": head.state_dict(),
            "metrics": metrics,
            "normalization": normalization,
            "args": vars(args),
            "note": "Use checkpoint_key='ema_encoder' for stable online predictor training.",
        },
        path,
    )


def run_phase1(args: argparse.Namespace, loaders: dict[str, DataLoader], normalization: dict[str, Any]) -> Path:
    device = torch.device(args.device)
    student = build_vjepa_encoder(args).to(device)
    freeze_encoder_for_phase1(student, args.phase1_train_last_n_blocks)
    teacher = copy.deepcopy(student).to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    embed_dim = int(getattr(student, "embed_dim"))
    head = TokenPredictionHead(embed_dim, args.phase1_head_hidden).to(device)
    optimizer = AdamW(
        [
            {"params": [parameter for parameter in student.parameters() if parameter.requires_grad], "lr": args.phase1_encoder_lr},
            {"params": head.parameters(), "lr": args.phase1_head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    best_loss = float("inf")
    best_path = output_dir / "encoder_finetune_best.pt"
    for epoch in range(1, args.phase1_epochs + 1):
        start = time.time()
        train_metrics = train_encoder_epoch(student, teacher, head, loaders["train"], optimizer, device, args, f"phase1 train {epoch}")
        with torch.no_grad():
            val_metrics = train_encoder_epoch(student, teacher, head, loaders["val"], None, device, args, f"phase1 val {epoch}")
        metrics = {"train": train_metrics, "val": val_metrics}
        save_encoder_checkpoint(output_dir / "encoder_finetune_last.pt", student, teacher, head, epoch, metrics, args, normalization)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_encoder_checkpoint(best_path, student, teacher, head, epoch, metrics, args, normalization)
        print(json.dumps({"phase": "phase1", "epoch": epoch, "seconds": time.time() - start, **metrics}, indent=2), flush=True)
    return best_path


def load_finetuned_encoder(args: argparse.Namespace, checkpoint_path: Path) -> nn.Module:
    encoder = build_vjepa_encoder(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint[args.phase2_encoder_key]
    encoder.load_state_dict(clean_state_dict_keys(state), strict=True)
    return encoder


def build_predictor_for_encoder(args: argparse.Namespace, encoder: nn.Module) -> nn.Module:
    preset = PREDICTOR_SIZE_PRESETS[args.model_size]
    predictor_dim = args.predictor_dim or preset["predictor_dim"]
    predictor_depth = args.predictor_depth or preset["predictor_depth"]
    predictor_heads = args.predictor_heads or preset["predictor_heads"]
    tokens_per_frame = (args.image_size // args.patch_size) ** 2
    return build_ac_predictor(
        predictor_type=args.predictor_type,
        latent_dim=int(getattr(encoder, "embed_dim")),
        state_dim=len(args.state_columns),
        action_dim=len(args.action_columns),
        tokens_per_frame=tokens_per_frame,
        max_frames=args.raw_frames_per_sample,
        predictor_dim=predictor_dim,
        depth=predictor_depth,
        num_heads=predictor_heads,
        dropout=args.dropout,
    )


def train_predictor_epoch(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    device: torch.device,
    args: argparse.Namespace,
    label: str,
) -> dict[str, float]:
    training = optimizer is not None
    encoder.train(training and args.phase2_train_encoder)
    predictor.train(training)
    if not args.phase2_train_encoder:
        encoder.eval()
    totals = {"loss": 0.0, "teacher_forcing_loss": 0.0, "rollout_loss": 0.0}
    count = 0
    tokens_per_frame = (args.image_size // args.patch_size) ** 2
    progress = tqdm(dataloader, desc=label, leave=False)
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        states = batch["states"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast_context(device, args.amp_dtype):
                if args.phase2_train_encoder:
                    tokens = encode_images(encoder, images, args.tubelet_size)
                else:
                    with torch.no_grad():
                        tokens = encode_images(encoder, images, args.tubelet_size)
                latents = flatten_tokens(tokens)
                outputs = compute_world_model_losses(
                    predictor=predictor,
                    latents=latents,
                    states=states,
                    actions=actions,
                    tokens_per_frame=tokens_per_frame,
                    auto_steps=args.auto_steps,
                    state_columns=tuple(args.state_columns),
                    action_columns=tuple(args.action_columns),
                    rollout_state_mode=args.rollout_state_mode,
                )
                loss = outputs["loss"]
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in list(encoder.parameters()) + list(predictor.parameters()) if parameter.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        bsz = images.size(0)
        for key in totals:
            totals[key] += float(outputs[key].detach().item()) * bsz
        count += bsz
        progress.set_postfix(average_metrics(totals, count))
    return average_metrics(totals, count)


def save_online_checkpoint(
    path: Path,
    encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    normalization: dict[str, Any],
    best_loss: float,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": best_loss,
            "global_step": global_step,
            "normalization": normalization,
            "args": vars(args),
            "tokens_per_frame": (args.image_size // args.patch_size) ** 2,
            "embed_dim": int(getattr(encoder, "embed_dim")),
        },
        path,
    )


def run_phase2(args: argparse.Namespace, loaders: dict[str, DataLoader], normalization: dict[str, Any], encoder_path: Path) -> None:
    device = torch.device(args.device)
    encoder = load_finetuned_encoder(args, encoder_path).to(device)
    if args.phase2_train_encoder:
        freeze_encoder_for_phase1(encoder, args.phase2_train_last_n_blocks)
    else:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    predictor = build_predictor_for_encoder(args, encoder).to(device)
    params = [{"params": predictor.parameters(), "lr": args.phase2_predictor_lr}]
    if args.phase2_train_encoder:
        params.append(
            {
                "params": [parameter for parameter in encoder.parameters() if parameter.requires_grad],
                "lr": args.phase2_encoder_lr,
            }
        )
    optimizer = AdamW(params, weight_decay=args.weight_decay)
    steps_per_epoch = compute_steps_per_epoch(loaders["train"])
    total_steps = args.phase2_epochs * steps_per_epoch
    scheduler = build_lr_scheduler(
        optimizer,
        total_train_steps=total_steps,
        warmup_steps=compute_warmup_steps(args.warmup_epochs, steps_per_epoch, total_steps),
        warmup_start_factor=args.warmup_start_factor,
        min_lr_ratio=args.min_lr_ratio,
    )
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    if args.phase2_resume_from is not None:
        checkpoint = torch.load(args.phase2_resume_from, map_location=device)
        encoder.load_state_dict(clean_state_dict_keys(checkpoint["encoder"]), strict=True)
        predictor.load_state_dict(checkpoint["predictor_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", (start_epoch - 1) * steps_per_epoch))
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("metrics", {}).get("val", {}).get("loss", float("inf"))))
        sync_lr_scheduler(scheduler, global_step)
        print(
            json.dumps(
                {
                    "phase2_resume_from": str(args.phase2_resume_from),
                    "start_epoch": start_epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                },
                indent=2,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "phase": "phase2",
                "predictor_params": count_trainable_parameters(predictor),
                "encoder_trainable_params": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                "steps_per_epoch": steps_per_epoch,
            },
            indent=2,
        ),
        flush=True,
    )
    output_dir = Path(args.output_dir)
    bad_epochs = 0
    for epoch in range(start_epoch, args.phase2_epochs + 1):
        start = time.time()
        train_metrics = train_predictor_epoch(encoder, predictor, loaders["train"], optimizer, scheduler, device, args, f"phase2 train {epoch}")
        global_step += len(loaders["train"])
        with torch.no_grad():
            val_metrics = train_predictor_epoch(encoder, predictor, loaders["val"], None, None, device, args, f"phase2 val {epoch}")
        metrics = {"train": train_metrics, "val": val_metrics}
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss = val_metrics["loss"]
            bad_epochs = 0
        elif should_apply_early_stopping(epoch, args.warmup_epochs):
            bad_epochs += 1
        save_online_checkpoint(
            output_dir / "online_ac_last.pt",
            encoder,
            predictor,
            optimizer,
            epoch,
            metrics,
            args,
            normalization,
            best_loss,
            global_step,
        )
        if improved:
            save_online_checkpoint(
                output_dir / "online_ac_best.pt",
                encoder,
                predictor,
                optimizer,
                epoch,
                metrics,
                args,
                normalization,
                best_loss,
                global_step,
            )
        print(json.dumps({"phase": "phase2", "epoch": epoch, "seconds": time.time() - start, "best_val": best_loss, **metrics}, indent=2), flush=True)
        if bad_epochs >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best_val={best_loss:.6f}", flush=True)
            break


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "finetune_encoder", "train_predictor"], default="all")
    parser.add_argument("--manifest-dir", type=Path, default=settings.MANIFEST_DIR)
    parser.add_argument(
        "--path-rewrite-from",
        default="",
        help="Optional prefix in manifest image paths to replace, useful when moving data to Colab/Drive.",
    )
    parser.add_argument(
        "--path-rewrite-to",
        default="",
        help="Replacement prefix for --path-rewrite-from.",
    )
    parser.add_argument("--vjepa-root", type=Path, default=settings.REPO_ROOT / "vjepa2")
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default=DEFAULT_CHECKPOINT_KEY)
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--encoder", choices=list(SUPPORTED_VJEPA21_ENCODER_NAMES), default=DEFAULT_VJEPA21_ENCODER_NAME)
    parser.add_argument("--image-size", type=int, default=settings.AC_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--tubelet-size", type=int, default=settings.AC_TUBELET_SIZE)
    parser.add_argument("--raw-frames-per-sample", type=int, default=settings.AC_RAW_FRAMES_PER_SAMPLE)
    parser.add_argument("--sequence-stride", type=int, default=settings.AC_SEQUENCE_STRIDE)
    parser.add_argument("--frame-stride", type=int, default=settings.AC_FRAME_STRIDE)
    parser.add_argument("--target-fps", type=float, default=settings.AC_TARGET_FPS)
    parser.add_argument("--state-columns", nargs="+", default=list(DEFAULT_AC_STATE_COLUMNS))
    parser.add_argument("--action-columns", nargs="+", default=list(DEFAULT_AC_ACTION_COLUMNS))
    parser.add_argument("--normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase1-encoder-lr", type=float, default=1e-6)
    parser.add_argument("--phase1-head-lr", type=float, default=1e-4)
    parser.add_argument("--phase1-train-last-n-blocks", type=int, default=4)
    parser.add_argument("--phase1-head-hidden", type=int, default=2048)
    parser.add_argument("--phase1-anchor-weight", type=float, default=0.25)
    parser.add_argument("--phase1-ema-decay", type=float, default=0.996)
    parser.add_argument("--phase2-epochs", type=int, default=100)
    parser.add_argument("--phase2-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--phase2-encoder-key", choices=["encoder", "ema_encoder"], default="ema_encoder")
    parser.add_argument("--phase2-resume-from", type=Path, default=None)
    parser.add_argument("--phase2-train-encoder", action="store_true")
    parser.add_argument("--phase2-train-last-n-blocks", type=int, default=2)
    parser.add_argument("--phase2-encoder-lr", type=float, default=5e-7)
    parser.add_argument("--phase2-predictor-lr", type=float, default=5e-5)
    parser.add_argument("--predictor-type", choices=SUPPORTED_PREDICTOR_TYPES, default=DEFAULT_PREDICTOR_TYPE)
    parser.add_argument("--model-size", choices=tuple(PREDICTOR_SIZE_PRESETS), default="base")
    parser.add_argument("--predictor-dim", type=int, default=None)
    parser.add_argument("--predictor-depth", type=int, default=None)
    parser.add_argument("--predictor-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--auto-steps", type=int, default=settings.AC_AUTO_STEPS)
    parser.add_argument(
        "--rollout-state-mode",
        choices=SUPPORTED_ROLLOUT_STATE_MODES,
        default=DEFAULT_ROLLOUT_STATE_MODE,
        help=(
            "State conditioning used during training rollout. measured_train uses measured "
            "states[:, :k+1]; legacy_repeat repeats state_0 and only copies previous action."
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/colab_encoder_online_ac"))
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2), flush=True)
    loaders, normalization = build_dataloaders(args)
    print(
        json.dumps(
            {
                "train_windows": len(loaders["train"].dataset),
                "val_windows": len(loaders["val"].dataset),
                "normalization": normalization,
            },
            indent=2,
        ),
        flush=True,
    )
    encoder_path = args.phase2_encoder_checkpoint
    if args.phase in ("all", "finetune_encoder"):
        encoder_path = run_phase1(args, loaders, normalization)
    if args.phase in ("all", "train_predictor"):
        if encoder_path is None:
            encoder_path = args.output_dir / "encoder_finetune_best.pt"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing phase2 encoder checkpoint: {encoder_path}")
        run_phase2(args, loaders, normalization, encoder_path)


if __name__ == "__main__":
    main()
