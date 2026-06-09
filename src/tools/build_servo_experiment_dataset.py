"""Build isolated manifests for current/old-servo data experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from data import settings
from data.preprocess import compute_feature_stats, preprocess_one_session


DEFAULT_EXPERIMENT_ROOT = Path("data/experiments/servo_old_mix_v1")
DEFAULT_OLD_SERVO_ROOT = Path("JEPA/data/drive_extra_nonzip") / "data servo cũ KDS 680HV"


@dataclass(frozen=True)
class SessionSource:
    path: Path
    domain: str
    raw_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated processed manifest set for current/old-servo experiments."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--current-raw-root", type=Path, default=settings.RAW_DATA_DIR)
    parser.add_argument("--old-servo-root", type=Path, default=DEFAULT_OLD_SERVO_ROOT)
    parser.add_argument("--mode", choices=["mixed", "old-only", "current-only"], default="mixed")
    parser.add_argument("--current-domain", default="current_servo")
    parser.add_argument("--old-domain", default="old_servo")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--no-test-split",
        dest="no_test_split",
        action="store_true",
        default=not settings.USE_TEST_SPLIT,
        help="Build train/val sessions only. test.jsonl aliases val.jsonl for compatibility.",
    )
    split_group.add_argument(
        "--with-test-split",
        dest="no_test_split",
        action="store_false",
        help="Build an independent test split.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_session_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")
    return sorted(path for path in root.glob(settings.SESSION_GLOB) if path.is_dir())


def collect_session_sources(args: argparse.Namespace) -> list[SessionSource]:
    sources: list[SessionSource] = []
    if args.mode in ("mixed", "current-only"):
        sources.extend(
            SessionSource(path=path, domain=args.current_domain, raw_root=args.current_raw_root)
            for path in find_session_dirs(args.current_raw_root)
        )
    if args.mode in ("mixed", "old-only"):
        sources.extend(
            SessionSource(path=path, domain=args.old_domain, raw_root=args.old_servo_root)
            for path in find_session_dirs(args.old_servo_root)
        )
    if not sources:
        raise RuntimeError(f"No sessions selected for mode={args.mode}")
    check_duplicate_session_ids(sources)
    return sources


def check_duplicate_session_ids(sources: Sequence[SessionSource]) -> None:
    by_id: dict[str, list[str]] = {}
    for source in sources:
        by_id.setdefault(source.path.name, []).append(str(source.path))
    duplicates = {session_id: paths for session_id, paths in by_id.items() if len(paths) > 1}
    if duplicates:
        raise ValueError(
            "Duplicate session ids across selected roots. Refusing to build ambiguous manifests: "
            f"{duplicates}"
        )


def prepare_output_dirs(experiment_root: Path) -> dict[str, Path]:
    processed_root = experiment_root / "processed"
    paths = {
        "processed_root": processed_root,
        "image_root": processed_root / "images",
        "manifest_root": processed_root / "manifests",
        "report_root": processed_root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_domain_split(session_domains: dict[str, str], seed: int, include_test: bool) -> dict[str, str]:
    split_map: dict[str, str] = {}
    domain_to_sessions: dict[str, list[str]] = {}
    for session_id, domain in session_domains.items():
        domain_to_sessions.setdefault(domain, []).append(session_id)

    for domain, session_ids in sorted(domain_to_sessions.items()):
        domain_split = split_sessions(
            session_ids,
            seed=seed + stable_domain_offset(domain),
            include_test=include_test,
        )
        split_map.update(domain_split)
    return split_map


def stable_domain_offset(domain: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(domain))


def split_sessions(session_ids: Sequence[str], seed: int, include_test: bool = True) -> dict[str, str]:
    shuffled = list(session_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if not include_test:
        if total == 1:
            counts = {"train": 1, "val": 0, "test": 0}
        else:
            train_count = max(1, int(total * settings.TRAIN_RATIO))
            train_count = min(train_count, total - 1)
            counts = {"train": train_count, "val": total - train_count, "test": 0}
    elif total == 1:
        counts = {"train": 1, "val": 0, "test": 0}
    elif total == 2:
        counts = {"train": 1, "val": 0, "test": 1}
    else:
        counts = {
            "train": max(1, int(total * settings.TRAIN_RATIO)),
            "val": max(1, int(total * settings.VAL_RATIO)),
            "test": max(1, int(total * settings.TEST_RATIO)),
        }
        while sum(counts.values()) > total:
            for split_name in ("train", "val", "test"):
                if counts[split_name] > 1 and sum(counts.values()) > total:
                    counts[split_name] -= 1
        while sum(counts.values()) < total:
            for split_name in ("train", "val", "test"):
                if sum(counts.values()) < total:
                    counts[split_name] += 1

    split_map: dict[str, str] = {}
    for index, session_id in enumerate(shuffled):
        if index < counts["train"]:
            split_map[session_id] = "train"
        elif index < counts["train"] + counts["val"]:
            split_map[session_id] = "val"
        else:
            split_map[session_id] = "test"
    return split_map


def write_jsonl(path: Path, samples: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def domain_counts(session_domains: dict[str, str], split_map: dict[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for session_id, domain in session_domains.items():
        split = split_map[session_id]
        counts.setdefault(domain, {"train": 0, "val": 0, "test": 0})
        counts[domain][split] += 1
    return counts


def temporarily_set_preprocess_outputs(paths: dict[str, Path]) -> dict[str, Path]:
    previous = {
        "PROCESSED_DATA_DIR": settings.PROCESSED_DATA_DIR,
        "PROCESSED_IMAGE_DIR": settings.PROCESSED_IMAGE_DIR,
        "MANIFEST_DIR": settings.MANIFEST_DIR,
        "REPORT_DIR": settings.REPORT_DIR,
    }
    settings.PROCESSED_DATA_DIR = paths["processed_root"]
    settings.PROCESSED_IMAGE_DIR = paths["image_root"]
    settings.MANIFEST_DIR = paths["manifest_root"]
    settings.REPORT_DIR = paths["report_root"]
    return previous


def restore_preprocess_outputs(previous: dict[str, Path]) -> None:
    settings.PROCESSED_DATA_DIR = previous["PROCESSED_DATA_DIR"]
    settings.PROCESSED_IMAGE_DIR = previous["PROCESSED_IMAGE_DIR"]
    settings.MANIFEST_DIR = previous["MANIFEST_DIR"]
    settings.REPORT_DIR = previous["REPORT_DIR"]


def main() -> None:
    args = parse_args()
    sources = collect_session_sources(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "experiment_root": str(args.experiment_root),
                    "selected_sessions": len(sources),
                    "include_test": not args.no_test_split,
                    "test_is_val_alias": bool(args.no_test_split and settings.ALIAS_TEST_TO_VAL),
                    "domains": {
                        domain: sum(1 for source in sources if source.domain == domain)
                        for domain in sorted({source.domain for source in sources})
                    },
                },
                indent=2,
            ),
            flush=True,
        )
        return

    output_paths = prepare_output_dirs(args.experiment_root)
    previous_outputs = temporarily_set_preprocess_outputs(output_paths)
    try:
        session_samples: dict[str, list[dict[str, Any]]] = {}
        session_domains: dict[str, str] = {}
        session_reports: list[dict[str, Any]] = []

        for index, source in enumerate(sources, start=1):
            print(f"[{index:03d}/{len(sources):03d}] preprocessing {source.domain}:{source.path.name}", flush=True)
            samples, report = preprocess_one_session(source.path)
            report["data_domain"] = source.domain
            report["raw_root"] = str(source.raw_root)
            report["source_session_path"] = str(source.path)
            session_reports.append(report)
            if not samples:
                continue
            for sample in samples:
                sample["data_domain"] = source.domain
                sample["source_raw_root"] = str(source.raw_root)
                sample["servo_experiment_mode"] = args.mode
            session_samples[source.path.name] = samples
            session_domains[source.path.name] = source.domain

        if not session_samples:
            raise RuntimeError("No usable samples found for servo experiment")

        split_map = build_domain_split(session_domains, seed=args.seed, include_test=not args.no_test_split)
        manifest_counts = {"train": 0, "val": 0, "test": 0}
        manifest_sessions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        manifest_samples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for session_id, samples in session_samples.items():
            split_name = split_map[session_id]
            manifest_sessions[split_name].append(session_id)
            for sample in samples:
                next_sample = dict(sample)
                next_sample["split"] = split_name
                manifest_samples[split_name].append(next_sample)

        test_is_val_alias = bool(args.no_test_split and settings.ALIAS_TEST_TO_VAL)
        if test_is_val_alias:
            manifest_sessions["test"] = list(manifest_sessions["val"])
            manifest_samples["test"] = [
                {**sample, "split": "test"}
                for sample in manifest_samples["val"]
            ]

        for split_name, split_samples in manifest_samples.items():
            manifest_counts[split_name] = len(split_samples)
            write_jsonl(output_paths["manifest_root"] / f"{split_name}.jsonl", split_samples)

        domain_session_counts = domain_counts(session_domains, split_map)
        if test_is_val_alias:
            for counts in domain_session_counts.values():
                counts["test"] = counts["val"]

        summary = {
            "experiment_root": str(args.experiment_root),
            "mode": args.mode,
            "current_raw_root": str(args.current_raw_root),
            "old_servo_root": str(args.old_servo_root),
            "processed_data_dir": str(output_paths["processed_root"]),
            "include_test": not args.no_test_split,
            "test_is_val_alias": test_is_val_alias,
            "counts": manifest_counts,
            "domain_session_counts": domain_session_counts,
            "sessions": {key: sorted(value) for key, value in manifest_sessions.items()},
            "feature_stats": compute_feature_stats(session_samples),
            "session_reports": session_reports,
        }
        report_path = output_paths["report_root"] / "preprocess_report.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        restore_preprocess_outputs(previous_outputs)


if __name__ == "__main__":
    main()
