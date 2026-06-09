"""Convert manifests to train/val-only and make test an alias of val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from data import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge test.jsonl into val.jsonl and rewrite test.jsonl as val alias.")
    parser.add_argument("--manifest-dir", type=Path, default=settings.MANIFEST_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(path: Path, samples: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def infer_report_path(manifest_dir: Path) -> Path | None:
    processed_root = manifest_dir.parent
    candidate = processed_root / "reports" / "preprocess_report.json"
    return candidate if candidate.exists() else None


def session_list(samples: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(sample["session_id"]) for sample in samples})


def domain_session_counts(split_samples: dict[str, Sequence[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name, samples in split_samples.items():
        session_domain: dict[str, str] = {}
        for sample in samples:
            session_id = str(sample["session_id"])
            domain = str(sample.get("data_domain", "unknown"))
            session_domain.setdefault(session_id, domain)
        for domain in session_domain.values():
            counts.setdefault(domain, {"train": 0, "val": 0, "test": 0})
            counts[domain][split_name] += 1
    return counts


def sample_key(sample: dict[str, Any]) -> tuple[str, str, str, str]:
    """Stable identity for avoiding duplicate val samples when this tool is rerun."""
    return (
        str(sample.get("session_id", "")),
        str(sample.get("sample_id", "")),
        str(sample.get("frame_index", "")),
        str(sample.get("frame_path", "")),
    )


def update_report(report_path: Path | None, split_samples: dict[str, list[dict[str, Any]]], merged_count: int) -> None:
    if report_path is None:
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["use_test_split"] = False
    report["include_test"] = False
    report["test_is_val_alias"] = True
    report["counts"] = {split: len(samples) for split, samples in split_samples.items()}
    report["sessions"] = {split: session_list(samples) for split, samples in split_samples.items()}
    report["domain_session_counts"] = domain_session_counts(split_samples)
    report["test_merged_into_val_count"] = merged_count
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir
    report_path = args.report_path if args.report_path is not None else infer_report_path(manifest_dir)

    train_samples = read_jsonl(manifest_dir / "train.jsonl")
    val_samples = read_jsonl(manifest_dir / "val.jsonl")
    test_samples = read_jsonl(manifest_dir / "test.jsonl")

    existing_val_keys = {sample_key(sample) for sample in val_samples}
    converted_test_samples = []
    for sample in test_samples:
        if sample_key(sample) in existing_val_keys:
            continue
        next_sample = dict(sample)
        next_sample["split"] = "val"
        converted_test_samples.append(next_sample)
        existing_val_keys.add(sample_key(next_sample))

    split_samples = {
        "train": train_samples,
        "val": val_samples + converted_test_samples,
        "test": [
            {**sample, "split": "test"}
            for sample in val_samples + converted_test_samples
        ],
    }
    for split_name, samples in split_samples.items():
        write_jsonl(manifest_dir / f"{split_name}.jsonl", samples)
    update_report(report_path, split_samples, merged_count=len(converted_test_samples))

    print(
        json.dumps(
            {
                "manifest_dir": str(manifest_dir),
                "report_path": None if report_path is None else str(report_path),
                "merged_test_samples_into_val": len(converted_test_samples),
                "test_is_val_alias": True,
                "counts": {split: len(samples) for split, samples in split_samples.items()},
                "sessions": {split: session_list(samples) for split, samples in split_samples.items()},
                "domain_session_counts": domain_session_counts(split_samples),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
