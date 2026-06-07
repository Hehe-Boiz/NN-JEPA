"""Precompute frozen V-JEPA 2.1 frame features for faster RC JEPA-AC training."""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_ENCODER_NAME,
    DEFAULT_PATCH_SIZE,
    FrozenVJepa21Encoder,
)


DEFAULT_OUTPUT_DIR = settings.PROCESSED_DATA_DIR / "features" / "vjepa2_1_vitb_384_ema_fp32"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = settings.NUM_WORKERS


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vjepa-root", type=Path, default=settings.REPO_ROOT / "vjepa2")
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default=DEFAULT_CHECKPOINT_KEY)
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER_NAME, choices=["vit_small_384", "vit_base_384", "vit_large_384"])
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    device = torch.device(args.device)
    encoder = build_encoder(args).to(device)
    encoder.eval()

    metadata = {
        "format_version": 1,
        "feature_layout": "frame_tokens",
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

    feature_summary = {"extracted": 0, "skipped_compatible": 0}
    with torch.no_grad():
        for session_id, session_samples in tqdm(grouped.items(), desc="sessions"):
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
    print(json.dumps({"status": "feature_extraction_complete", "feature_summary": feature_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
