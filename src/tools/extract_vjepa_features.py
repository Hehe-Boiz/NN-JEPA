"""Precompute frozen V-JEPA 2.1 frame features for faster RC JEPA-AC training."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data import settings
from data.dataset import image_to_tensor, load_manifest, normalize_tensor
from data.feature_sequence_dataset import FEATURE_METADATA_NAME, FEATURE_SESSIONS_DIR_NAME
from data.sequence_dataset import sample_sort_key
from models.rc_jepa_ac import (
    DEFAULT_CHECKPOINT_KEY,
    DEFAULT_PATCH_SIZE,
    FrozenVJepa21Encoder,
)
from models.vjepa21_presets import (
    DEFAULT_VJEPA21_FEATURE_PRESET,
    SUPPORTED_VJEPA21_ENCODER_NAMES,
    VJEPA21_FEATURE_PRESETS,
    get_vjepa21_feature_preset,
    vjepa21_feature_output_dir,
)


DEFAULT_OUTPUT_DIR = vjepa21_feature_output_dir(DEFAULT_VJEPA21_FEATURE_PRESET, "fp32")
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = settings.NUM_WORKERS
PROGRESS_PREFIX = "__JOB_PROGRESS__ "


class FrameFeatureDataset(Dataset):
    """Frame-level image dataset used only during feature extraction."""

    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample["frame_path"]) as image:
            rgb = image.convert("RGB")
        image_tensor = normalize_tensor(
            image_to_tensor(rgb),
            mean=list(settings.NORMALIZE_MEAN),
            std=list(settings.NORMALIZE_STD),
        )
        return {
            "image": image_tensor,
            "row": index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen V-JEPA 2.1 frame features.")
    parser.add_argument("--manifest-dir", type=Path, default=settings.MANIFEST_DIR)
    parser.add_argument(
        "--encoder-preset",
        default=DEFAULT_VJEPA21_FEATURE_PRESET,
        choices=list(VJEPA21_FEATURE_PRESETS),
        help="Safe preset that maps encoder/checkpoint key/checkpoint path/default feature dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seed-from-features-dir",
        type=Path,
        default=None,
        help="Optional existing feature cache to symlink compatible session .npy/.json files from.",
    )
    parser.add_argument("--vjepa-root", type=Path, default=settings.REPO_ROOT / "vjepa2")
    parser.add_argument("--vjepa-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--encoder", default=None, choices=list(SUPPORTED_VJEPA21_ENCODER_NAMES))
    parser.add_argument("--image-size", type=int, default=settings.AC_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--tubelet-size", type=int, default=settings.AC_TUBELET_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_feature_extraction_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = get_vjepa21_feature_preset(args.encoder_preset)
    encoder_was_overridden = args.encoder is not None and args.encoder != preset.encoder_name

    if args.encoder is None:
        args.encoder = preset.encoder_name
    if args.vjepa_checkpoint is None:
        if encoder_was_overridden:
            raise ValueError(
                "When --encoder overrides --encoder-preset, pass --vjepa-checkpoint explicitly "
                "to avoid loading weights from the wrong architecture."
            )
        args.vjepa_checkpoint = preset.checkpoint_path
    if args.checkpoint_key is None:
        args.checkpoint_key = DEFAULT_CHECKPOINT_KEY if encoder_was_overridden else preset.checkpoint_key
    if args.output_dir is None:
        if encoder_was_overridden:
            args.output_dir = settings.PROCESSED_DATA_DIR / "features" / f"{args.encoder}_{args.dtype}"
        else:
            args.output_dir = vjepa21_feature_output_dir(args.encoder_preset, args.dtype)

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


def print_progress(
    percent: float,
    label: str,
    *,
    current: int | None = None,
    total: int | None = None,
    indeterminate: bool = False,
) -> None:
    print(
        PROGRESS_PREFIX
        + json.dumps(
            {
                "percent": max(0.0, min(float(percent), 100.0)),
                "label": label,
                "current": current,
                "total": total,
                "indeterminate": indeterminate,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def load_unique_samples(manifest_dir: Path, splits: Sequence[str]) -> list[dict[str, Any]]:
    samples_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for split in splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        for sample in load_manifest(manifest_path):
            session_id = str(sample["session_id"])
            frame_index = int(sample["frame_index"])
            key = (session_id, frame_index)
            if key not in samples_by_key:
                next_sample = dict(sample)
                next_sample["feature_splits"] = [split]
                samples_by_key[key] = next_sample
            else:
                samples_by_key[key]["feature_splits"].append(split)
    return list(samples_by_key.values())


def group_samples_by_session(samples: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["session_id"])].append(sample)
    for session_samples in grouped.values():
        session_samples.sort(key=sample_sort_key)
    return dict(sorted(grouped.items()))


def numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "fp16":
        return np.dtype(np.float16)
    if dtype_name == "fp32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_encoder(args: argparse.Namespace) -> FrozenVJepa21Encoder:
    return FrozenVJepa21Encoder(
        vjepa_root=args.vjepa_root,
        checkpoint_path=args.vjepa_checkpoint,
        encoder_name=args.encoder,
        checkpoint_key=args.checkpoint_key,
        image_size=args.image_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        strict_checkpoint=not args.allow_partial_checkpoint,
    )


def extract_session_features(
    session_id: str,
    session_samples: Sequence[dict[str, Any]],
    encoder: FrozenVJepa21Encoder,
    sessions_dir: Path,
    batch_size: int,
    num_workers: int,
    dtype: np.dtype,
    device: torch.device,
    overwrite: bool,
) -> str:
    npy_path = sessions_dir / f"{session_id}.npy"
    json_path = sessions_dir / f"{session_id}.json"
    tokens_per_frame = encoder.tokens_per_frame
    embed_dim = encoder.embed_dim
    expected_shape = (len(session_samples), tokens_per_frame, embed_dim)
    if npy_path.exists() and json_path.exists() and not overwrite:
        existing = np.load(npy_path, mmap_mode="r")
        if tuple(existing.shape) != expected_shape or np.dtype(existing.dtype) != dtype:
            raise ValueError(
                f"Existing feature cache for {session_id} is incompatible: "
                f"shape={existing.shape}, dtype={existing.dtype}. "
                f"Expected shape={expected_shape}, dtype={dtype}. "
                "Use --overwrite or a different --output-dir."
            )
        return "skipped_compatible"

    feature_array = np.lib.format.open_memmap(
        npy_path,
        mode="w+",
        dtype=dtype,
        shape=expected_shape,
    )
    dataset = FrameFeatureDataset(session_samples)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": settings.PIN_MEMORY,
        "persistent_workers": settings.PERSISTENT_WORKERS and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = settings.PREFETCH_FACTOR
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    for batch in tqdm(dataloader, desc=session_id, leave=False):
        images = batch["image"].to(device, non_blocking=True).unsqueeze(2)
        rows = batch["row"].numpy()
        latents, batch_tokens_per_frame = encoder(images)
        if batch_tokens_per_frame != tokens_per_frame:
            raise ValueError(f"tokens_per_frame mismatch: {batch_tokens_per_frame} != {tokens_per_frame}")
        latents = latents.view(images.size(0), tokens_per_frame, embed_dim)
        feature_array[rows] = latents.detach().cpu().numpy().astype(dtype, copy=False)
    feature_array.flush()

    frames = [
        {
            "row": row,
            "sample_id": sample["sample_id"],
            "session_id": sample["session_id"],
            "frame_index": int(sample["frame_index"]),
            "timestamp_sec": sample.get("timestamp_sec"),
            "frame_path": sample["frame_path"],
            "splits": sample.get("feature_splits", []),
        }
        for row, sample in enumerate(session_samples)
    ]
    write_json(
        json_path,
        {
            "session_id": session_id,
            "num_frames": len(session_samples),
            "tokens_per_frame": tokens_per_frame,
            "embed_dim": embed_dim,
            "dtype": str(dtype),
            "feature_path": str(npy_path),
            "frames": frames,
        },
    )
    return "extracted"


def read_existing_metadata(output_dir: Path) -> dict[str, Any] | None:
    metadata_path = output_dir / FEATURE_METADATA_NAME
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def metadata_matches_request(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    grouped: dict[str, list[dict[str, Any]]],
    frame_count: int,
) -> bool:
    expected = {
        "format_version": 1,
        "feature_layout": "frame_tokens",
        "encoder_name": args.encoder,
        "checkpoint_path": str(args.vjepa_checkpoint),
        "checkpoint_key": args.checkpoint_key,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "tubelet_size": args.tubelet_size,
        "dtype": args.dtype,
        "normalization_mean": list(settings.NORMALIZE_MEAN),
        "normalization_std": list(settings.NORMALIZE_STD),
        "manifest_dir": str(args.manifest_dir),
        "splits": list(args.splits),
        "session_count": len(grouped),
        "frame_count": frame_count,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def cache_status_from_metadata(
    session_id: str,
    session_samples: Sequence[dict[str, Any]],
    sessions_dir: Path,
    metadata: dict[str, Any],
    dtype: np.dtype,
) -> str:
    npy_path = sessions_dir / f"{session_id}.npy"
    json_path = sessions_dir / f"{session_id}.json"
    if not npy_path.exists() or not json_path.exists():
        return "missing"
    try:
        tokens_per_frame = int(metadata["tokens_per_frame"])
        embed_dim = int(metadata["embed_dim"])
        existing = np.load(npy_path, mmap_mode="r")
    except (KeyError, OSError, ValueError):
        return "incompatible"
    expected_shape = (len(session_samples), tokens_per_frame, embed_dim)
    if tuple(existing.shape) != expected_shape or np.dtype(existing.dtype) != dtype:
        return "incompatible"
    return "compatible"


def cache_status_summary(
    grouped: dict[str, list[dict[str, Any]]],
    sessions_dir: Path,
    metadata: dict[str, Any],
    dtype: np.dtype,
) -> dict[str, int]:
    summary = {"compatible": 0, "missing": 0, "incompatible": 0}
    for session_id, session_samples in grouped.items():
        status = cache_status_from_metadata(session_id, session_samples, sessions_dir, metadata, dtype)
        summary[status] += 1
    return summary


def seed_feature_cache_from_dir(
    seed_dir: Path | None,
    output_dir: Path,
    grouped: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    tokens_per_frame: int,
    embed_dim: int,
    dtype: np.dtype,
    overwrite: bool,
) -> dict[str, int]:
    summary = {"seeded": 0, "already_present": 0, "missing": 0, "incompatible": 0, "skipped": 0}
    if seed_dir is None:
        return summary
    if overwrite:
        summary["skipped"] = len(grouped)
        return summary
    if seed_dir.resolve() == output_dir.resolve():
        summary["skipped"] = len(grouped)
        return summary

    source_sessions_dir = seed_dir / FEATURE_SESSIONS_DIR_NAME
    target_sessions_dir = output_dir / FEATURE_SESSIONS_DIR_NAME
    if not source_sessions_dir.exists():
        summary["missing"] = len(grouped)
        return summary

    seed_metadata = read_existing_metadata(seed_dir)
    if seed_metadata is None or not seed_metadata_is_compatible_for_seed(
        seed_metadata=seed_metadata,
        args=args,
        tokens_per_frame=tokens_per_frame,
        embed_dim=embed_dim,
        dtype=dtype,
    ):
        summary["incompatible"] = len(grouped)
        return summary

    for session_id, session_samples in grouped.items():
        target_npy = target_sessions_dir / f"{session_id}.npy"
        target_json = target_sessions_dir / f"{session_id}.json"
        if target_npy.exists() and target_json.exists():
            summary["already_present"] += 1
            continue

        source_npy = source_sessions_dir / f"{session_id}.npy"
        source_json = source_sessions_dir / f"{session_id}.json"
        if not source_npy.exists() or not source_json.exists():
            summary["missing"] += 1
            continue

        try:
            existing = np.load(source_npy, mmap_mode="r")
        except (OSError, ValueError):
            summary["incompatible"] += 1
            continue
        expected_shape = (len(session_samples), tokens_per_frame, embed_dim)
        if tuple(existing.shape) != expected_shape or np.dtype(existing.dtype) != dtype:
            summary["incompatible"] += 1
            continue

        for source_path, target_path in ((source_npy, target_npy), (source_json, target_json)):
            if target_path.exists():
                continue
            os.symlink(source_path.resolve(), target_path)
        summary["seeded"] += 1
    return summary


def seed_metadata_is_compatible_for_seed(
    seed_metadata: dict[str, Any],
    args: argparse.Namespace,
    tokens_per_frame: int,
    embed_dim: int,
    dtype: np.dtype,
) -> bool:
    """Check encoder-level metadata before reusing per-session feature files.

    Manifest path, split names, and session count may differ because a mixed
    experiment intentionally reuses only overlapping current-servo sessions.
    Encoder identity and tensor layout must still match exactly.
    """
    expected_values = {
        "format_version": 1,
        "feature_layout": "frame_tokens",
        "tokens_per_frame": tokens_per_frame,
        "embed_dim": embed_dim,
        "dtype": "fp16" if dtype == np.dtype(np.float16) else "fp32",
        "encoder_name": args.encoder,
        "checkpoint_key": args.checkpoint_key,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "tubelet_size": args.tubelet_size,
        "normalization_mean": list(settings.NORMALIZE_MEAN),
        "normalization_std": list(settings.NORMALIZE_STD),
    }
    for key, expected_value in expected_values.items():
        if seed_metadata.get(key) != expected_value:
            return False
    if not checkpoint_paths_match(seed_metadata.get("checkpoint_path"), args.vjepa_checkpoint):
        return False
    return True


def checkpoint_paths_match(seed_value: Any, request_value: Path) -> bool:
    if seed_value in (None, ""):
        return False
    seed_path = Path(str(seed_value))
    request_path = Path(request_value)
    try:
        return seed_path.resolve() == request_path.resolve()
    except OSError:
        return str(seed_path) == str(request_path)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args = resolve_feature_extraction_args(args)
    set_seed(args.seed)
    print_progress(0, "Preparing feature extraction", indeterminate=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = args.output_dir / FEATURE_SESSIONS_DIR_NAME
    sessions_dir.mkdir(parents=True, exist_ok=True)

    samples = load_unique_samples(args.manifest_dir, args.splits)
    grouped = group_samples_by_session(samples)
    dtype = numpy_dtype(args.dtype)

    if not args.overwrite:
        existing_metadata = read_existing_metadata(args.output_dir)
        if existing_metadata is not None and metadata_matches_request(
            existing_metadata,
            args=args,
            grouped=grouped,
            frame_count=len(samples),
        ):
            status_summary = cache_status_summary(grouped, sessions_dir, existing_metadata, dtype)
            if status_summary["compatible"] == len(grouped):
                print_progress(100, "Feature cache already complete", current=len(grouped), total=len(grouped))
                print(
                    json.dumps(
                        {
                            "status": "feature_cache_already_complete",
                            "feature_summary": {
                                "skipped_compatible": status_summary["compatible"],
                                "extracted": 0,
                                "missing": 0,
                                "incompatible": 0,
                            },
                            "metadata": existing_metadata,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                return

    print_progress(5, "Loading frozen V-JEPA encoder", indeterminate=True)
    device = torch.device(args.device)
    encoder = build_encoder(args).to(device)
    encoder.eval()
    print_progress(10, "Encoder loaded")

    metadata = {
        "format_version": 1,
        "feature_layout": "frame_tokens",
        "encoder_preset": args.encoder_preset,
        "encoder_name": args.encoder,
        "checkpoint_path": str(args.vjepa_checkpoint),
        "checkpoint_key": args.checkpoint_key,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "tubelet_size": args.tubelet_size,
        "tokens_per_frame": encoder.tokens_per_frame,
        "embed_dim": encoder.embed_dim,
        "dtype": args.dtype,
        "normalization_mean": list(settings.NORMALIZE_MEAN),
        "normalization_std": list(settings.NORMALIZE_STD),
        "manifest_dir": str(args.manifest_dir),
        "splits": list(args.splits),
        "session_count": len(grouped),
        "frame_count": len(samples),
    }
    write_json(args.output_dir / FEATURE_METADATA_NAME, metadata)
    print(json.dumps(metadata, indent=2), flush=True)

    seed_summary = seed_feature_cache_from_dir(
        seed_dir=args.seed_from_features_dir,
        output_dir=args.output_dir,
        grouped=grouped,
        args=args,
        tokens_per_frame=encoder.tokens_per_frame,
        embed_dim=encoder.embed_dim,
        dtype=dtype,
        overwrite=args.overwrite,
    )
    if args.seed_from_features_dir is not None:
        print(json.dumps({"feature_seed_summary": seed_summary}, indent=2), flush=True)

    feature_summary = {"extracted": 0, "skipped_compatible": 0, "seeded": seed_summary["seeded"]}
    total_sessions = len(grouped)
    with torch.no_grad():
        for index, (session_id, session_samples) in enumerate(tqdm(grouped.items(), desc="sessions"), start=1):
            print_progress(
                10 + 90 * ((index - 1) / max(total_sessions, 1)),
                f"Extracting features: {session_id}",
                current=index - 1,
                total=total_sessions,
            )
            status = extract_session_features(
                session_id=session_id,
                session_samples=session_samples,
                encoder=encoder,
                sessions_dir=sessions_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dtype=dtype,
                device=device,
                overwrite=args.overwrite,
            )
            feature_summary[status] = feature_summary.get(status, 0) + 1
    print_progress(100, "Feature extraction complete", current=total_sessions, total=total_sessions)
    print(json.dumps({"status": "feature_extraction_complete", "feature_summary": feature_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
