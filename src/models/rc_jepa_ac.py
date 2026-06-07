"""Frozen V-JEPA 2.1 encoder + action-conditioned world model for RC car data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from data import settings


DEFAULT_ENCODER_NAME = "vit_base_384"
DEFAULT_CHECKPOINT_KEY = "ema_encoder"
DEFAULT_PREDICTOR_DIM = 512
DEFAULT_PREDICTOR_DEPTH = 6
DEFAULT_PREDICTOR_HEADS = 8
DEFAULT_PATCH_SIZE = 16


class FrozenVJepa21Encoder(nn.Module):
    """V-JEPA 2.1 encoder used only as a frozen target feature extractor."""

    def __init__(
        self,
        vjepa_root: str | Path,
        checkpoint_path: str | Path | None,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
        image_size: int = settings.AC_IMAGE_SIZE,
        patch_size: int = DEFAULT_PATCH_SIZE,
        tubelet_size: int = settings.AC_TUBELET_SIZE,
        strict_checkpoint: bool = True,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.vjepa_root = Path(vjepa_root)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.encoder_name = encoder_name
        self.checkpoint_key = checkpoint_key
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.normalize_output = normalize_output
        self.encoder = self._build_encoder()

        if self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path, strict=strict_checkpoint)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    @property
    def embed_dim(self) -> int:
        return int(self.encoder.embed_dim)

    @property
    def tokens_per_frame(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def train(self, mode: bool = True) -> "FrozenVJepa21Encoder":
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Encode [B, C, T, H, W] images into [B, T*K, D] frozen tokens."""
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, C, T, H, W], got {tuple(images.shape)}")

        batch_size, channels, num_frames, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {channels}")

        frames = images.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        if height != self.image_size or width != self.image_size:
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        pseudo_clips = frames.unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)
        with torch.no_grad():
            tokens = self.encoder(pseudo_clips)
            if self.normalize_output:
                tokens = F.layer_norm(tokens, (tokens.size(-1),))

        tokens_per_frame = tokens.size(1)
        tokens = tokens.view(batch_size, num_frames, tokens_per_frame, tokens.size(-1))
        return tokens.flatten(1, 2).detach(), tokens_per_frame

    def _build_encoder(self) -> nn.Module:
        if not self.vjepa_root.exists():
            raise FileNotFoundError(f"Missing local vjepa2 repo: {self.vjepa_root}")

        if str(self.vjepa_root) not in sys.path:
            sys.path.insert(0, str(self.vjepa_root))

        from app.vjepa_2_1.models import vision_transformer as vjepa_vit

        builders = {
            "vit_small_384": vjepa_vit.vit_small,
            "vit_base_384": vjepa_vit.vit_base,
            "vit_large_384": vjepa_vit.vit_large,
        }
        if self.encoder_name not in builders:
            raise ValueError(f"Unsupported V-JEPA 2.1 encoder: {self.encoder_name}")

        return builders[self.encoder_name](
            img_size=(self.image_size, self.image_size),
            patch_size=self.patch_size,
            num_frames=self.tubelet_size,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

    def _load_checkpoint(self, checkpoint_path: Path, strict: bool) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_checkpoint_state(checkpoint, checkpoint_key=self.checkpoint_key)
        msg = self.encoder.load_state_dict(clean_state_dict_keys(state_dict), strict=strict)
        if not strict:
            missing = list(getattr(msg, "missing_keys", []))
            unexpected = list(getattr(msg, "unexpected_keys", []))
            if missing or unexpected:
                print(
                    {
                        "checkpoint_load": "partial",
                        "missing_keys": missing[:20],
                        "unexpected_keys": unexpected[:20],
                        "missing_count": len(missing),
                        "unexpected_count": len(unexpected),
                    }
                )


class SimpleACPredictor(nn.Module):
    """Small causal transformer that predicts next-frame latent tokens."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        tokens_per_frame: int,
        max_frames: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
        predictor_dim: int = DEFAULT_PREDICTOR_DIM,
        depth: int = DEFAULT_PREDICTOR_DEPTH,
        num_heads: int = DEFAULT_PREDICTOR_HEADS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tokens_per_frame = tokens_per_frame
        self.max_frames = max_frames
        self.predictor_dim = predictor_dim
        self.cond_tokens = 2

        self.latent_proj = nn.Linear(latent_dim, predictor_dim)
        self.state_proj = nn.Linear(state_dim, predictor_dim)
        self.action_proj = nn.Linear(action_dim, predictor_dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, max_frames, 1, predictor_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, predictor_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, 1, predictor_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, 1, predictor_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, latent_dim)
        self._init_parameters()

    def forward(
        self,
        latent_tokens: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        tokens_per_frame: int | None = None,
    ) -> torch.Tensor:
        tokens_per_frame = tokens_per_frame or self.tokens_per_frame
        if tokens_per_frame != self.tokens_per_frame:
            raise ValueError(
                f"Predictor was built for {self.tokens_per_frame} tokens/frame, got {tokens_per_frame}"
            )

        batch_size, total_tokens, latent_dim = latent_tokens.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {latent_dim}")
        if total_tokens % tokens_per_frame != 0:
            raise ValueError("latent_tokens length must be divisible by tokens_per_frame")

        num_frames = total_tokens // tokens_per_frame
        if num_frames > self.max_frames:
            raise ValueError(f"num_frames={num_frames} exceeds max_frames={self.max_frames}")
        if actions.shape[:2] != (batch_size, num_frames):
            raise ValueError(f"Expected actions [B, {num_frames}, A], got {tuple(actions.shape)}")
        if states.shape[:2] != (batch_size, num_frames):
            raise ValueError(f"Expected states [B, {num_frames}, S], got {tuple(states.shape)}")

        frame_pos = self.frame_pos[:, :num_frames]
        latent = latent_tokens.view(batch_size, num_frames, tokens_per_frame, latent_dim)
        latent = self.latent_proj(latent) + frame_pos + self.patch_pos
        action = self.action_proj(actions).unsqueeze(2) + frame_pos + self.action_type
        state = self.state_proj(states).unsqueeze(2) + frame_pos + self.state_type

        sequence = torch.cat([action, state, latent], dim=2).flatten(1, 2)
        mask = build_time_causal_mask(
            num_frames=num_frames,
            tokens_per_step=tokens_per_frame + self.cond_tokens,
            device=sequence.device,
        )
        sequence = self.blocks(sequence, mask=mask)
        sequence = sequence.view(batch_size, num_frames, tokens_per_frame + self.cond_tokens, self.predictor_dim)
        predicted = sequence[:, :, self.cond_tokens :, :].flatten(1, 2)
        predicted = self.output_proj(self.norm(predicted))
        return predicted

    def _init_parameters(self) -> None:
        for parameter in (self.frame_pos, self.patch_pos, self.action_type, self.state_type):
            nn.init.trunc_normal_(parameter, std=0.02)


class RCJepaACWorldModel(nn.Module):
    """Trainable action-conditioned predictor over frozen V-JEPA 2.1 tokens."""

    def __init__(
        self,
        vjepa_root: str | Path,
        checkpoint_path: str | Path | None,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
        image_size: int = settings.AC_IMAGE_SIZE,
        patch_size: int = DEFAULT_PATCH_SIZE,
        tubelet_size: int = settings.AC_TUBELET_SIZE,
        raw_frames_per_sample: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
        state_dim: int = len(settings.AC_STATE_COLUMNS),
        action_dim: int = len(settings.AC_ACTION_COLUMNS),
        state_columns: tuple[str, ...] = tuple(settings.AC_STATE_COLUMNS),
        action_columns: tuple[str, ...] = tuple(settings.AC_ACTION_COLUMNS),
        predictor_dim: int = DEFAULT_PREDICTOR_DIM,
        predictor_depth: int = DEFAULT_PREDICTOR_DEPTH,
        predictor_heads: int = DEFAULT_PREDICTOR_HEADS,
        dropout: float = 0.0,
        auto_steps: int = settings.AC_AUTO_STEPS,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.raw_frames_per_sample = raw_frames_per_sample
        self.auto_steps = auto_steps
        self.state_columns = tuple(state_columns)
        self.action_columns = tuple(action_columns)
        self.target_encoder = FrozenVJepa21Encoder(
            vjepa_root=vjepa_root,
            checkpoint_path=checkpoint_path,
            encoder_name=encoder_name,
            checkpoint_key=checkpoint_key,
            image_size=image_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            strict_checkpoint=strict_checkpoint,
        )
        self.predictor = SimpleACPredictor(
            latent_dim=self.target_encoder.embed_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            tokens_per_frame=self.target_encoder.tokens_per_frame,
            max_frames=raw_frames_per_sample,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dropout=dropout,
        )

    def train(self, mode: bool = True) -> "RCJepaACWorldModel":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latents, tokens_per_frame = self.target_encoder(images)
        return compute_world_model_losses(
            predictor=self.predictor,
            latents=latents,
            states=states,
            actions=actions,
            tokens_per_frame=tokens_per_frame,
            auto_steps=self.auto_steps,
            state_columns=self.state_columns,
            action_columns=self.action_columns,
        )


def compute_world_model_losses(
    predictor: SimpleACPredictor,
    latents: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    tokens_per_frame: int,
    auto_steps: int,
    state_columns: tuple[str, ...] | None = None,
    action_columns: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if auto_steps < 1:
        raise ValueError("auto_steps must be >= 1")

    num_frames = states.size(1)
    if num_frames < 2:
        raise ValueError("Need at least 2 frames to train next-frame dynamics")
    if actions.size(1) != num_frames - 1:
        raise ValueError(f"Expected actions length {num_frames - 1}, got {actions.size(1)}")

    input_latents = latents[:, :-tokens_per_frame]
    target_latents = latents[:, tokens_per_frame:]
    teacher_pred = predictor(
        latent_tokens=input_latents,
        actions=actions,
        states=states[:, :-1],
        tokens_per_frame=tokens_per_frame,
    )
    teacher_forcing_loss = F.l1_loss(teacher_pred, target_latents)

    rollout_steps = min(auto_steps, num_frames - 1)
    rollout_tokens = latents[:, :tokens_per_frame]
    rollout_states = build_rollout_state_context(
        initial_state=states[:, :1],
        actions=actions,
        rollout_steps=rollout_steps,
        state_columns=state_columns,
        action_columns=action_columns,
    )
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
    rollout_loss = F.l1_loss(rollout_pred, rollout_target)
    loss = teacher_forcing_loss + rollout_loss

    return {
        "loss": loss,
        "teacher_forcing_loss": teacher_forcing_loss,
        "rollout_loss": rollout_loss,
    }


def build_rollout_state_context(
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    rollout_steps: int,
    state_columns: tuple[str, ...] | None = None,
    action_columns: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Build rollout state inputs without using future measured state."""
    rollout_states = initial_state.repeat(1, rollout_steps, 1)
    if rollout_steps <= 1 or state_columns is None or action_columns is None:
        return rollout_states

    copy_previous_action_to_state(
        rollout_states,
        actions,
        state_columns=state_columns,
        action_columns=action_columns,
        state_name="steering_last_t",
        action_name="steering_cmd_t",
    )
    copy_previous_action_to_state(
        rollout_states,
        actions,
        state_columns=state_columns,
        action_columns=action_columns,
        state_name="throttle_last_t",
        action_name="throttle_cmd_t",
    )
    return rollout_states


def copy_previous_action_to_state(
    rollout_states: torch.Tensor,
    actions: torch.Tensor,
    state_columns: tuple[str, ...],
    action_columns: tuple[str, ...],
    state_name: str,
    action_name: str,
) -> None:
    if state_name not in state_columns or action_name not in action_columns:
        return
    state_index = state_columns.index(state_name)
    action_index = action_columns.index(action_name)
    for step in range(1, rollout_states.size(1)):
        rollout_states[:, step, state_index] = actions[:, step - 1, action_index]


def build_time_causal_mask(num_frames: int, tokens_per_step: int, device: torch.device) -> torch.Tensor:
    time_ids = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_step)
    return time_ids.unsqueeze(0) > time_ids.unsqueeze(1)


def extract_checkpoint_state(checkpoint: Any, checkpoint_key: str) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if checkpoint_key not in checkpoint:
            if all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
            available = ", ".join(sorted(str(key) for key in checkpoint.keys()))
            raise KeyError(f"Checkpoint key '{checkpoint_key}' not found. Available keys: {available}")
        state_dict = checkpoint[checkpoint_key]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint value for '{checkpoint_key}' must be a state dict")
    return state_dict


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
