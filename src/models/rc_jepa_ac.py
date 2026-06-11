"""Frozen V-JEPA 2.1 encoder + action-conditioned world model for RC car data."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import math
from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from data import settings
from models.vjepa21_presets import (
    DEFAULT_VJEPA21_ENCODER_NAME,
    VJEPA21_ENCODER_SPECS,
)


DEFAULT_ENCODER_NAME = DEFAULT_VJEPA21_ENCODER_NAME
DEFAULT_CHECKPOINT_KEY = "ema_encoder"
DEFAULT_PREDICTOR_DIM = 512
DEFAULT_PREDICTOR_DEPTH = 6
DEFAULT_PREDICTOR_HEADS = 8
DEFAULT_PATCH_SIZE = 16
DEFAULT_PREDICTOR_TYPE = "simple"
SUPPORTED_PREDICTOR_TYPES = ("simple", "official_lite")
ROLLOUT_STATE_MODE_LEGACY_REPEAT = "legacy_repeat"
ROLLOUT_STATE_MODE_MEASURED_TRAIN = "measured_train"
DEFAULT_ROLLOUT_STATE_MODE = ROLLOUT_STATE_MODE_MEASURED_TRAIN
DEFAULT_ROLLOUT_FEEDBACK_NORM = False
SUPPORTED_ROLLOUT_STATE_MODES = (
    ROLLOUT_STATE_MODE_LEGACY_REPEAT,
    ROLLOUT_STATE_MODE_MEASURED_TRAIN,
)
DYNAMIC_STATE_COLUMNS = ("v_t", "yaw_rate_t", "accel_x_t", "accel_y_t")
PREDICTOR_SIZE_PRESETS = {
    "tiny": {
        "predictor_dim": 128,
        "predictor_depth": 2,
        "predictor_heads": 4,
    },
    "small": {
        "predictor_dim": 256,
        "predictor_depth": 4,
        "predictor_heads": 4,
    },
    "base": {
        "predictor_dim": DEFAULT_PREDICTOR_DIM,
        "predictor_depth": DEFAULT_PREDICTOR_DEPTH,
        "predictor_heads": DEFAULT_PREDICTOR_HEADS,
    },
}


@contextmanager
def torch_transformer_eval_fastpath_disabled(disable: bool):
    """Avoid eval-only native Transformer fastpath memory spikes on long token sequences."""
    if not disable:
        yield
        return

    mha_backend = getattr(torch.backends, "mha", None)
    get_fastpath_enabled = getattr(mha_backend, "get_fastpath_enabled", None)
    set_fastpath_enabled = getattr(mha_backend, "set_fastpath_enabled", None)
    if get_fastpath_enabled is None or set_fastpath_enabled is None:
        yield
        return

    previous = bool(get_fastpath_enabled())
    set_fastpath_enabled(False)
    try:
        yield
    finally:
        set_fastpath_enabled(previous)


def apply_predictor_size_preset(args: Any) -> None:
    """Fill missing predictor dimensions from a named size preset."""
    model_size = getattr(args, "model_size", "base")
    if model_size not in PREDICTOR_SIZE_PRESETS:
        raise ValueError(f"Unknown predictor model_size={model_size!r}")

    preset = PREDICTOR_SIZE_PRESETS[model_size]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if int(args.predictor_dim) % int(args.predictor_heads) != 0:
        raise ValueError(
            f"predictor_dim={args.predictor_dim} must be divisible by predictor_heads={args.predictor_heads}"
        )


def build_ac_predictor(
    predictor_type: str,
    latent_dim: int,
    state_dim: int,
    action_dim: int,
    tokens_per_frame: int,
    max_frames: int = settings.AC_RAW_FRAMES_PER_SAMPLE,
    predictor_dim: int = DEFAULT_PREDICTOR_DIM,
    depth: int = DEFAULT_PREDICTOR_DEPTH,
    num_heads: int = DEFAULT_PREDICTOR_HEADS,
    dropout: float = 0.0,
) -> nn.Module:
    """Build one of the supported action-conditioned latent predictors."""
    if predictor_type == "simple":
        return SimpleACPredictor(
            latent_dim=latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            tokens_per_frame=tokens_per_frame,
            max_frames=max_frames,
            predictor_dim=predictor_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
    if predictor_type == "official_lite":
        return VJepaStyleACPredictor(
            latent_dim=latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            tokens_per_frame=tokens_per_frame,
            max_frames=max_frames,
            predictor_dim=predictor_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )

    available = ", ".join(SUPPORTED_PREDICTOR_TYPES)
    raise ValueError(f"Unknown predictor_type={predictor_type!r}. Available: {available}")


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

        spec = VJEPA21_ENCODER_SPECS.get(self.encoder_name)
        if spec is None:
            available = ", ".join(VJEPA21_ENCODER_SPECS)
            raise ValueError(f"Unsupported V-JEPA 2.1 encoder: {self.encoder_name}. Available: {available}")
        builder = getattr(vjepa_vit, spec.builder_name)

        return builder(
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
        with torch_transformer_eval_fastpath_disabled(disable=not self.training):
            sequence = self.blocks(sequence, mask=mask)
        sequence = sequence.view(batch_size, num_frames, tokens_per_frame + self.cond_tokens, self.predictor_dim)
        predicted = sequence[:, :, self.cond_tokens :, :].flatten(1, 2)
        predicted = self.output_proj(self.norm(predicted))
        return predicted

    def _init_parameters(self) -> None:
        for parameter in (self.frame_pos, self.patch_pos, self.action_type, self.state_type):
            nn.init.trunc_normal_(parameter, std=0.02)


class VJepaStyleACPredictor(nn.Module):
    """Official-lite AC predictor adapted from V-JEPA AC token layout and mask logic.

    The public V-JEPA AC predictor inserts action/state tokens before each
    frame's patch tokens, applies an action-block causal attention mask, and
    returns only patch-token predictions. This version keeps that contract but
    uses configurable small depths/dims for local RC-car experiments.
    """

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
        if predictor_dim % num_heads != 0:
            raise ValueError(f"predictor_dim={predictor_dim} must be divisible by num_heads={num_heads}")

        grid_size = int(math.isqrt(tokens_per_frame))
        if grid_size * grid_size != tokens_per_frame:
            raise ValueError(
                "VJepaStyleACPredictor requires square frame-token grids. "
                f"Got tokens_per_frame={tokens_per_frame}."
            )

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tokens_per_frame = tokens_per_frame
        self.max_frames = max_frames
        self.predictor_dim = predictor_dim
        self.num_heads = num_heads
        self.cond_tokens = 2
        self.grid_height = grid_size
        self.grid_width = grid_size

        self.predictor_embed = nn.Linear(latent_dim, predictor_dim, bias=True)
        self.action_encoder = nn.Linear(action_dim, predictor_dim, bias=True)
        self.state_encoder = nn.Linear(state_dim, predictor_dim, bias=True)
        self.blocks = nn.ModuleList(
            [
                VJepaStyleACBlock(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    grid_size=grid_size,
                    use_rope=True,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_dim, latent_dim, bias=True)
        self.register_buffer(
            "attn_mask",
            build_action_block_causal_attention_mask(
                num_frames=max_frames,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                add_tokens=self.cond_tokens,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)
        self._rescale_blocks()

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

        latent = self.predictor_embed(latent_tokens)
        latent = latent.view(batch_size, num_frames, tokens_per_frame, self.predictor_dim)
        action = self.action_encoder(actions).unsqueeze(2)
        state = self.state_encoder(states).unsqueeze(2)
        sequence = torch.cat([action, state, latent], dim=2).flatten(1, 2)

        sequence_length = sequence.size(1)
        attn_mask = self.attn_mask[:sequence_length, :sequence_length].to(
            device=sequence.device,
            non_blocking=True,
        )
        for block in self.blocks:
            sequence = block(
                sequence,
                attn_mask=attn_mask,
                num_frames=num_frames,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                action_tokens=self.cond_tokens,
            )

        sequence = sequence.view(
            batch_size,
            num_frames,
            self.cond_tokens + tokens_per_frame,
            self.predictor_dim,
        )
        predicted = sequence[:, :, self.cond_tokens :, :].flatten(1, 2)
        predicted = self.predictor_proj(self.predictor_norm(predicted))
        return predicted

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _rescale_blocks(self) -> None:
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))


class VJepaStyleACBlock(nn.Module):
    """Transformer block matching the public V-JEPA AC predictor structure."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        grid_size: int,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VJepaStyleACAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            grid_size=grid_size,
            use_rope=use_rope,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = VJepaStyleMLP(
            in_features=dim,
            hidden_features=dim * 4,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        num_frames: int,
        grid_height: int,
        grid_width: int,
        action_tokens: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            num_frames=num_frames,
            grid_height=grid_height,
            grid_width=grid_width,
            action_tokens=action_tokens,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class VJepaStyleACAttention(nn.Module):
    """RoPE attention adapted from `vjepa2/src/models/utils/modules.py`."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        grid_size: int,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = grid_size
        self.use_rope = use_rope
        self.attn_dropout = dropout
        self.proj_dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = int(2 * ((self.head_dim // 3) // 2))
        self.w_dim = int(2 * ((self.head_dim // 3) // 2))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        num_frames: int,
        grid_height: int,
        grid_width: int,
        action_tokens: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        if sequence_length != num_frames * (action_tokens + grid_height * grid_width):
            raise ValueError(
                "Attention sequence length does not match frame/action-token layout: "
                f"sequence_length={sequence_length}, num_frames={num_frames}, "
                f"action_tokens={action_tokens}, grid={grid_height}x{grid_width}"
            )

        if action_tokens > 0:
            x = x.view(batch_size, num_frames, action_tokens + grid_height * grid_width, channels)
            action_q, action_k, action_v = self._encode_action_tokens(x, action_tokens)
            patch_tokens = x[:, :, action_tokens:, :].flatten(1, 2)
        else:
            action_q = action_k = action_v = None
            patch_tokens = x

        q, k, v = self._qkv(patch_tokens)
        if self.use_rope:
            q, k = self._apply_patch_rope(q, k, num_frames, grid_height, grid_width)

        if action_tokens > 0:
            q = self._merge_action_and_patch_heads(q, action_q, batch_size, num_frames, grid_height, grid_width)
            k = self._merge_action_and_patch_heads(k, action_k, batch_size, num_frames, grid_height, grid_width)
            v = self._merge_action_and_patch_heads(v, action_v, batch_size, num_frames, grid_height, grid_width)

        dropout_p = self.attn_dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _encode_action_tokens(
        self,
        x: torch.Tensor,
        action_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, _, _ = x.shape
        action_q: list[torch.Tensor] = []
        action_k: list[torch.Tensor] = []
        action_v: list[torch.Tensor] = []
        frame_pos = torch.arange(num_frames, device=x.device)

        for index in range(action_tokens):
            token = x[:, :, index : index + 1, :].flatten(1, 2)
            q, k, v = self._qkv(token)
            if self.use_rope and self.d_dim > 0:
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=frame_pos)
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=frame_pos)
                q = torch.cat([qd, q[..., self.d_dim :]], dim=-1)
                k = torch.cat([kd, k[..., self.d_dim :]], dim=-1)
            action_q.append(q.view(batch_size, self.num_heads, num_frames, 1, self.head_dim))
            action_k.append(k.view(batch_size, self.num_heads, num_frames, 1, self.head_dim))
            action_v.append(v.view(batch_size, self.num_heads, num_frames, 1, self.head_dim))

        return (
            torch.cat(action_q, dim=3).flatten(2, 3),
            torch.cat(action_k, dim=3).flatten(2, 3),
            torch.cat(action_v, dim=3).flatten(2, 3),
        )

    def _apply_patch_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        num_frames: int,
        grid_height: int,
        grid_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.arange(num_frames * grid_height * grid_width, device=q.device)
        frame_pos, height_pos, width_pos = separate_patch_positions(token_ids, grid_height, grid_width)
        height_pos = height_pos * (self.grid_size / grid_height)
        width_pos = width_pos * (self.grid_size / grid_width)

        parts_q: list[torch.Tensor] = []
        parts_k: list[torch.Tensor] = []
        cursor = 0
        for dim_size, pos in (
            (self.d_dim, frame_pos),
            (self.h_dim, height_pos),
            (self.w_dim, width_pos),
        ):
            if dim_size <= 0:
                continue
            parts_q.append(rotate_queries_or_keys(q[..., cursor : cursor + dim_size], pos=pos))
            parts_k.append(rotate_queries_or_keys(k[..., cursor : cursor + dim_size], pos=pos))
            cursor += dim_size

        if cursor < self.head_dim:
            parts_q.append(q[..., cursor:])
            parts_k.append(k[..., cursor:])
        return torch.cat(parts_q, dim=-1), torch.cat(parts_k, dim=-1)

    def _merge_action_and_patch_heads(
        self,
        patch_heads: torch.Tensor,
        action_heads: torch.Tensor,
        batch_size: int,
        num_frames: int,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        patch_heads = patch_heads.view(
            batch_size,
            self.num_heads,
            num_frames,
            grid_height * grid_width,
            self.head_dim,
        )
        action_heads = action_heads.view(
            batch_size,
            self.num_heads,
            num_frames,
            -1,
            self.head_dim,
        )
        return torch.cat([action_heads, patch_heads], dim=3).flatten(2, 3)


class VJepaStyleMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
        predictor_type: str = DEFAULT_PREDICTOR_TYPE,
        dropout: float = 0.0,
        auto_steps: int = settings.AC_AUTO_STEPS,
        rollout_state_mode: str = DEFAULT_ROLLOUT_STATE_MODE,
        rollout_feedback_norm: bool = DEFAULT_ROLLOUT_FEEDBACK_NORM,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        validate_rollout_state_mode(rollout_state_mode)
        self.raw_frames_per_sample = raw_frames_per_sample
        self.auto_steps = auto_steps
        self.state_columns = tuple(state_columns)
        self.action_columns = tuple(action_columns)
        self.predictor_type = predictor_type
        self.rollout_state_mode = rollout_state_mode
        self.rollout_feedback_norm = bool(rollout_feedback_norm)
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
        self.predictor = build_ac_predictor(
            predictor_type=predictor_type,
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
            rollout_state_mode=self.rollout_state_mode,
            rollout_feedback_norm=self.rollout_feedback_norm,
        )


def compute_world_model_losses(
    predictor: nn.Module,
    latents: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    tokens_per_frame: int,
    auto_steps: int,
    state_columns: tuple[str, ...] | None = None,
    action_columns: tuple[str, ...] | None = None,
    rollout_state_mode: str = DEFAULT_ROLLOUT_STATE_MODE,
    rollout_feedback_norm: bool = DEFAULT_ROLLOUT_FEEDBACK_NORM,
) -> dict[str, torch.Tensor]:
    if auto_steps < 1:
        raise ValueError("auto_steps must be >= 1")
    validate_rollout_state_mode(rollout_state_mode)

    batch_size = states.size(0)
    num_frames = states.size(1)
    latent_dim = latents.size(-1)
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
    if rollout_state_mode == ROLLOUT_STATE_MODE_MEASURED_TRAIN:
        rollout_states = states[:, :rollout_steps]
    else:
        rollout_states = build_rollout_state_context(
            initial_state=states[:, :1],
            actions=actions,
            rollout_steps=rollout_steps,
            state_columns=state_columns,
            action_columns=action_columns,
        )
    rollout_predictions = []
    for step in range(rollout_steps):
        expected_context_frames = step + 1
        action_context = actions[:, :expected_context_frames]
        state_context = rollout_states[:, :expected_context_frames]
        assert_rollout_context_shapes(
            latent_context=rollout_tokens,
            state_context=state_context,
            action_context=action_context,
            batch_size=batch_size,
            context_frames=expected_context_frames,
            tokens_per_frame=tokens_per_frame,
            latent_dim=latent_dim,
        )
        pred_tokens = predictor(
            latent_tokens=rollout_tokens,
            actions=action_context,
            states=state_context,
            tokens_per_frame=tokens_per_frame,
        )
        next_tokens = pred_tokens[:, -tokens_per_frame:]
        rollout_predictions.append(next_tokens)
        feedback_tokens = normalize_rollout_feedback(next_tokens, enabled=rollout_feedback_norm)
        rollout_tokens = torch.cat([rollout_tokens, feedback_tokens], dim=1)

    rollout_pred = torch.cat(rollout_predictions, dim=1)
    rollout_target = latents[:, tokens_per_frame : tokens_per_frame * (rollout_steps + 1)]
    rollout_loss = F.l1_loss(rollout_pred, rollout_target)
    loss = teacher_forcing_loss + rollout_loss

    return {
        "loss": loss,
        "teacher_forcing_loss": teacher_forcing_loss,
        "rollout_loss": rollout_loss,
    }


def normalize_rollout_feedback(tokens: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Optionally re-normalize predicted latent tokens before rollout feedback.

    V-JEPA/JEPA-style AC rollout feeds predicted latent frames back into the
    predictor after a per-token LayerNorm. The target feature cache is already
    LayerNorm-normalized at extraction time, so this keeps autoregressive inputs
    in the same latent scale.
    """
    if not enabled:
        return tokens
    return F.layer_norm(tokens, (tokens.size(-1),))


def build_rollout_state_context(
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    rollout_steps: int,
    state_columns: tuple[str, ...] | None = None,
    action_columns: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Fallback rollout state approximation for inference/planning.

    This helper repeats the initial state and only copies previous action into
    steering_last_t/throttle_last_t. It is intended for inference/planning when
    measured future states are unavailable, or for explicit legacy training
    compatibility. Training rollout should use measured future states via
    rollout_state_mode="measured_train" when they exist.
    """
    warn_if_dynamic_states_are_stale(state_columns)
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


def validate_rollout_state_mode(mode: str) -> None:
    if mode not in SUPPORTED_ROLLOUT_STATE_MODES:
        available = ", ".join(SUPPORTED_ROLLOUT_STATE_MODES)
        raise ValueError(f"Unknown rollout_state_mode={mode!r}. Available: {available}")


def assert_rollout_context_shapes(
    latent_context: torch.Tensor,
    state_context: torch.Tensor,
    action_context: torch.Tensor,
    batch_size: int,
    context_frames: int,
    tokens_per_frame: int,
    latent_dim: int,
) -> None:
    expected_latent_shape = (batch_size, context_frames * tokens_per_frame, latent_dim)
    if tuple(latent_context.shape) != expected_latent_shape:
        raise RuntimeError(
            "Invalid autoregressive latent context shape: "
            f"expected {expected_latent_shape}, got {tuple(latent_context.shape)}"
        )
    if state_context.shape[:2] != (batch_size, context_frames):
        raise RuntimeError(
            "Invalid rollout state context shape: "
            f"expected first dims {(batch_size, context_frames)}, got {tuple(state_context.shape)}"
        )
    if action_context.shape[:2] != (batch_size, context_frames):
        raise RuntimeError(
            "Invalid rollout action context shape: "
            f"expected first dims {(batch_size, context_frames)}, got {tuple(action_context.shape)}"
        )


def warn_if_dynamic_states_are_stale(state_columns: tuple[str, ...] | None) -> None:
    if not state_columns:
        return
    stale_columns = [column for column in DYNAMIC_STATE_COLUMNS if column in state_columns]
    if not stale_columns:
        return
    warnings.warn(
        "build_rollout_state_context is using stale/approximated dynamic state columns "
        f"{stale_columns}. This is only a fallback for inference/planning or explicit "
        "legacy_repeat training; measured_train rollout should use measured states.",
        RuntimeWarning,
        stacklevel=2,
    )


def build_time_causal_mask(num_frames: int, tokens_per_step: int, device: torch.device) -> torch.Tensor:
    time_ids = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_step)
    return time_ids.unsqueeze(0) > time_ids.unsqueeze(1)


def build_action_block_causal_attention_mask(
    num_frames: int,
    grid_height: int,
    grid_width: int,
    add_tokens: int = 2,
) -> torch.Tensor:
    """Return the V-JEPA AC allowed-attention mask.

    This mirrors `vjepa2/src/models/utils/modules.py`: each frame is a block of
    condition tokens plus patch tokens, and a query block can attend only to
    blocks at the same or earlier time index. The returned bool mask uses SDPA
    semantics where True means "allowed to attend".
    """
    tokens_per_frame = add_tokens + (grid_height * grid_width)
    total_tokens = num_frames * tokens_per_frame
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
    mask_block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)
    local_window_time = num_frames

    for target_time in range(num_frames):
        start_time = max(0, target_time - local_window_time + 1)
        for source_time in range(start_time, target_time + 1):
            target_slice = slice(
                target_time * tokens_per_frame,
                (target_time + 1) * tokens_per_frame,
            )
            source_slice = slice(
                source_time * tokens_per_frame,
                (source_time + 1) * tokens_per_frame,
            )
            mask[target_slice, source_slice] = mask_block
    return mask


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply the RoPE rotation used by the public V-JEPA AC attention code."""
    if x.size(-1) == 0:
        return x
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE input dimension must be even")

    omega = torch.arange(x.size(-1) // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / max(x.size(-1) / 2.0, 1.0)))
    freq = torch.einsum("..., f -> ... f", pos.to(dtype=x.dtype, device=x.device), omega)
    emb_sin = expand_source_rope_frequencies(freq.sin(), target_ndim=x.ndim)
    emb_cos = expand_source_rope_frequencies(freq.cos(), target_ndim=x.ndim)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def expand_source_rope_frequencies(values: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Broadcast RoPE frequencies while preserving the public V-JEPA repeat pattern."""
    while values.ndim < target_ndim:
        values = values.unsqueeze(0)
    repeat_shape = [1] * values.ndim
    repeat_shape[-1] = 2
    return values.repeat(*repeat_shape)


def separate_patch_positions(
    token_ids: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens_per_frame = grid_height * grid_width
    frame_ids = token_ids // tokens_per_frame
    token_ids = token_ids - (tokens_per_frame * frame_ids)
    height_ids = token_ids // grid_width
    width_ids = token_ids - (grid_width * height_ids)
    return frame_ids, height_ids, width_ids


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
