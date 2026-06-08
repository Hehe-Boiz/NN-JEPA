"""Shared V-JEPA 2.1 encoder and feature-extraction presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data import settings


DEFAULT_VJEPA21_ENCODER_NAME = "vit_base_384"
DEFAULT_VJEPA21_FEATURE_PRESET = "vitb_384"


@dataclass(frozen=True)
class VJepa21EncoderSpec:
    encoder_name: str
    builder_name: str
    embed_dim: int
    depth: int
    num_heads: int


@dataclass(frozen=True)
class VJepa21FeaturePreset:
    name: str
    label: str
    encoder_name: str
    checkpoint_path: Path
    checkpoint_key: str
    output_dir_stem: str
    note: str


VJEPA21_ENCODER_SPECS: dict[str, VJepa21EncoderSpec] = {
    "vit_small_384": VJepa21EncoderSpec(
        encoder_name="vit_small_384",
        builder_name="vit_small",
        embed_dim=384,
        depth=12,
        num_heads=6,
    ),
    "vit_base_384": VJepa21EncoderSpec(
        encoder_name="vit_base_384",
        builder_name="vit_base",
        embed_dim=768,
        depth=12,
        num_heads=12,
    ),
    "vit_large_384": VJepa21EncoderSpec(
        encoder_name="vit_large_384",
        builder_name="vit_large",
        embed_dim=1024,
        depth=24,
        num_heads=16,
    ),
    "vit_giant_384": VJepa21EncoderSpec(
        encoder_name="vit_giant_384",
        builder_name="vit_giant_xformers",
        embed_dim=1408,
        depth=40,
        num_heads=22,
    ),
    "vit_gigantic_384": VJepa21EncoderSpec(
        encoder_name="vit_gigantic_384",
        builder_name="vit_gigantic_xformers",
        embed_dim=1664,
        depth=48,
        num_heads=26,
    ),
}

SUPPORTED_VJEPA21_ENCODER_NAMES = tuple(VJEPA21_ENCODER_SPECS.keys())


VJEPA21_FEATURE_PRESETS: dict[str, VJepa21FeaturePreset] = {
    "vitb_384": VJepa21FeaturePreset(
        name="vitb_384",
        label="V-JEPA 2.1 ViT-B/16 384",
        encoder_name="vit_base_384",
        checkpoint_path=Path("checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt"),
        checkpoint_key="ema_encoder",
        output_dir_stem="vjepa2_1_vitb_384_ema",
        note="Smallest practical public V-JEPA 2.1 checkpoint for this project.",
    ),
    "vitl_384": VJepa21FeaturePreset(
        name="vitl_384",
        label="V-JEPA 2.1 ViT-L/16 384",
        encoder_name="vit_large_384",
        checkpoint_path=Path("checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"),
        checkpoint_key="ema_encoder",
        output_dir_stem="vjepa2_1_vitl_384_ema",
        note="Larger distilled checkpoint; requires its own extracted feature cache.",
    ),
    "vitg_384": VJepa21FeaturePreset(
        name="vitg_384",
        label="V-JEPA 2.1 ViT-g/16 384",
        encoder_name="vit_giant_384",
        checkpoint_path=Path("checkpoints/vjepa2_1/vjepa2_1_vitg_384.pt"),
        checkpoint_key="target_encoder",
        output_dir_stem="vjepa2_1_vitg_384_target",
        note="Very large checkpoint; likely needs much more VRAM than ViT-B.",
    ),
    "vitG_384": VJepa21FeaturePreset(
        name="vitG_384",
        label="V-JEPA 2.1 ViT-G/16 384",
        encoder_name="vit_gigantic_384",
        checkpoint_path=Path("checkpoints/vjepa2_1/vjepa2_1_vitG_384.pt"),
        checkpoint_key="target_encoder",
        output_dir_stem="vjepa2_1_vitG_384_target",
        note="Largest public preset in the local vjepa2 configs; expect very high VRAM use.",
    ),
}


def get_vjepa21_feature_preset(name: str) -> VJepa21FeaturePreset:
    try:
        return VJEPA21_FEATURE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(VJEPA21_FEATURE_PRESETS)
        raise ValueError(f"Unknown V-JEPA 2.1 feature preset {name!r}. Available: {available}") from exc


def vjepa21_feature_output_dir(preset_name: str, dtype: str = "fp32") -> Path:
    preset = get_vjepa21_feature_preset(preset_name)
    return settings.PROCESSED_DATA_DIR / "features" / f"{preset.output_dir_stem}_{dtype}"


def vjepa21_feature_preset_options() -> list[dict[str, str]]:
    return [
        {
            "name": preset.name,
            "label": preset.label,
            "encoder_name": preset.encoder_name,
            "checkpoint_path": str(preset.checkpoint_path),
            "checkpoint_key": preset.checkpoint_key,
            "default_output_dir": str(vjepa21_feature_output_dir(preset.name, "fp32")),
            "note": preset.note,
        }
        for preset in VJEPA21_FEATURE_PRESETS.values()
    ]
