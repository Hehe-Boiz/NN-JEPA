"""Incrementally sync RC session zips from Google Drive and prepare raw data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


DRIVE_REMOTE = "gdrive:JEPA"
ZIP_STAGING_DIR = Path("JEPA/data/drive_zips")
EXTRA_NONZIP_STAGING_DIR = Path("JEPA/data/drive_extra_nonzip")
RAW_DIR = Path("data/raw")
JEPA_SRC_DIR = Path("JEPA/src")
SESSION_ZIP_PATTERN = "session_*.zip"
SOURCE_SIGNATURE_NAME = ".source_zip_signature.json"
DEFAULT_TRANSFERS = 8
DEFAULT_CHECKERS = 16
PROGRESS_PREFIX = "__JOB_PROGRESS__ "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync session zip files from Drive, extract new top-level sessions into data/raw, "
            "and run JEPA sensor sync for sessions that need synced CSV files."
        )
    )
    parser.add_argument("--remote", default=DRIVE_REMOTE)
    parser.add_argument("--zip-staging-dir", type=Path, default=ZIP_STAGING_DIR)
    parser.add_argument("--extra-nonzip-staging-dir", type=Path, default=EXTRA_NONZIP_STAGING_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--jepa-src-dir", type=Path, default=JEPA_SRC_DIR)
    parser.add_argument("--rclone-bin", default="rclone")
    parser.add_argument("--transfers", type=int, default=DEFAULT_TRANSFERS)
    parser.add_argument("--checkers", type=int, default=DEFAULT_CHECKERS)
    parser.add_argument("--no-fast-list", action="store_true")
    parser.add_argument("--skip-rclone-copy", action="store_true")
    parser.add_argument("--check-zips", action="store_true")
    parser.add_argument(
        "--sync-extra-nonzip",
        action="store_true",
        help="Also copy non-zip Drive files into a separate staging dir. This is usually old data.",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument(
        "--overwrite-changed",
        action="store_true",
        help="If a local session exists but the source zip changed, replace that session dir.",
    )
    parser.add_argument("--skip-sensor-sync", action="store_true")
    parser.add_argument("--resync-all", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


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


def rclone_common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--transfers",
        str(args.transfers),
        "--checkers",
        str(args.checkers),
        "-P",
    ]
    if not args.no_fast_list:
        common.append("--fast-list")
    return common


def sync_zip_staging(args: argparse.Namespace) -> None:
    args.zip_staging_dir.mkdir(parents=True, exist_ok=True)
    print_progress(5, "Downloading zip files from Drive", indeterminate=True)
    command = [
        args.rclone_bin,
        "copy",
        args.remote,
        str(args.zip_staging_dir),
        "--include",
        "*.zip",
        *rclone_common_args(args),
    ]
    run_command(command, dry_run=args.dry_run)
    print_progress(25, "Drive zip download step finished")


def sync_extra_nonzip_staging(args: argparse.Namespace) -> None:
    args.extra_nonzip_staging_dir.mkdir(parents=True, exist_ok=True)
    print_progress(25, "Downloading non-zip Drive files", indeterminate=True)
    command = [
        args.rclone_bin,
        "copy",
        args.remote,
        str(args.extra_nonzip_staging_dir),
        "--exclude",
        "*.zip",
        *rclone_common_args(args),
    ]
    run_command(command, dry_run=args.dry_run)
    print_progress(35, "Non-zip Drive download step finished")


def check_zip_staging(args: argparse.Namespace) -> None:
    print_progress(35, "Checking local zip staging against Drive", indeterminate=True)
    command = [
        args.rclone_bin,
        "check",
        args.remote,
        str(args.zip_staging_dir),
        "--include",
        "*.zip",
        "--one-way",
    ]
    if not args.no_fast_list:
        command.append("--fast-list")
    run_command(command, dry_run=args.dry_run)
    print_progress(45, "Drive zip check finished")


def zip_signature(zip_path: Path) -> dict[str, Any]:
    stat = zip_path.stat()
    return {
        "zip_path": str(zip_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_signature(session_dir: Path) -> dict[str, Any] | None:
    signature_path = session_dir / SOURCE_SIGNATURE_NAME
    if not signature_path.exists():
        return None
    try:
        return json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_signature(session_dir: Path, signature: dict[str, Any]) -> None:
    (session_dir / SOURCE_SIGNATURE_NAME).write_text(
        json.dumps(signature, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def move_extracted_session(temp_dir: Path, session_dir: Path, session_name: str) -> None:
    children = [child for child in temp_dir.iterdir()]
    nested_session_dir = temp_dir / session_name
    if nested_session_dir.is_dir():
        shutil.move(str(nested_session_dir), str(session_dir))
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    if len(children) == 1 and children[0].is_dir() and children[0].name.startswith("session_"):
        shutil.move(str(children[0]), str(session_dir))
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    shutil.move(str(temp_dir), str(session_dir))


def extract_session_zip(zip_path: Path, raw_dir: Path, *, overwrite_changed: bool, dry_run: bool) -> str:
    session_name = zip_path.stem
    session_dir = raw_dir / session_name
    signature = zip_signature(zip_path)
    old_signature = read_signature(session_dir) if session_dir.exists() else None

    if session_dir.exists() and old_signature == signature:
        return "unchanged"
    if session_dir.exists() and old_signature is None:
        print(f"adopt existing {session_dir}", flush=True)
        if not dry_run:
            write_signature(session_dir, signature)
        return "adopted"
    if session_dir.exists() and old_signature != signature and not overwrite_changed:
        return "changed_skipped"

    temp_dir = raw_dir / f".{session_name}.extracting"
    print(f"extract {zip_path} -> {session_dir}", flush=True)
    if dry_run:
        return "dry_run"
    shutil.rmtree(temp_dir, ignore_errors=True)
    if session_dir.exists():
        shutil.rmtree(session_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_dir)
    move_extracted_session(temp_dir, session_dir, session_name)
    write_signature(session_dir, signature)
    return "extracted"


def extract_top_level_zips(args: argparse.Namespace) -> dict[str, list[str]]:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[str]] = {
        "extracted": [],
        "unchanged": [],
        "adopted": [],
        "changed_skipped": [],
        "dry_run": [],
    }
    zip_paths = sorted(args.zip_staging_dir.glob(SESSION_ZIP_PATTERN))
    total = len(zip_paths)
    if total == 0:
        print_progress(60, "No top-level session zip found", current=0, total=0)
        return summary
    for index, zip_path in enumerate(zip_paths, start=1):
        print_progress(
            45 + 25 * ((index - 1) / max(total, 1)),
            f"Extracting/adopting sessions: {zip_path.stem}",
            current=index - 1,
            total=total,
        )
        status = extract_session_zip(
            zip_path,
            args.raw_dir,
            overwrite_changed=args.overwrite_changed,
            dry_run=args.dry_run,
        )
        summary.setdefault(status, []).append(zip_path.stem)
    print_progress(70, "Session zip extraction step finished", current=total, total=total)
    return summary


def session_needs_sensor_sync(session_dir: Path) -> bool:
    return not (session_dir / "actions_synced.csv").exists() or not (session_dir / "imu_synced.csv").exists()


def run_sensor_sync(args: argparse.Namespace, extract_summary: dict[str, list[str]]) -> list[str]:
    if not args.jepa_src_dir.exists():
        raise FileNotFoundError(f"JEPA source dir not found: {args.jepa_src_dir}")

    if args.resync_all:
        targets = sorted(path for path in args.raw_dir.glob("session_*") if path.is_dir())
    else:
        extracted_names = set(extract_summary.get("extracted", []))
        targets = [
            path
            for path in sorted(args.raw_dir.glob("session_*"))
            if path.is_dir() and (path.name in extracted_names or session_needs_sensor_sync(path))
        ]

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(args.jepa_src_dir)
        if not existing_pythonpath
        else str(args.jepa_src_dir) + os.pathsep + existing_pythonpath
    )

    synced: list[str] = []
    total = len(targets)
    if total == 0:
        print_progress(90, "No session needs sensor sync", current=0, total=0)
        return synced
    for index, session_dir in enumerate(targets, start=1):
        print_progress(
            70 + 20 * ((index - 1) / max(total, 1)),
            f"Running sensor sync: {session_dir.name}",
            current=index - 1,
            total=total,
        )
        command = [sys.executable, "-m", "jepa_wm.data.sync", str(session_dir)]
        run_command(command, env=env, dry_run=args.dry_run)
        synced.append(session_dir.name)
    print_progress(90, "Sensor sync step finished", current=total, total=total)
    return synced


def run_preprocess(dry_run: bool) -> dict[str, Any] | None:
    print_progress(90, "Running preprocessing after sync", indeterminate=True)
    print("+ preprocess_all_sessions()", flush=True)
    if dry_run:
        return None
    from data.preprocess import preprocess_all_sessions

    summary = preprocess_all_sessions()
    print_progress(98, "Preprocessing after sync finished")
    return summary


def main() -> None:
    args = parse_args()
    print_progress(0, "Starting Drive sync", indeterminate=True)
    if not args.skip_rclone_copy:
        sync_zip_staging(args)
    if args.sync_extra_nonzip:
        sync_extra_nonzip_staging(args)
    if args.check_zips:
        check_zip_staging(args)

    extract_summary: dict[str, list[str]] = {}
    if not args.skip_extract:
        extract_summary = extract_top_level_zips(args)

    sensor_synced: list[str] = []
    if not args.skip_sensor_sync:
        sensor_synced = run_sensor_sync(args, extract_summary)

    preprocess_summary = run_preprocess(args.dry_run) if args.preprocess else None
    final_summary = {
        "zip_staging_dir": str(args.zip_staging_dir),
        "raw_dir": str(args.raw_dir),
        "extract": {key: len(value) for key, value in extract_summary.items()},
        "changed_skipped": extract_summary.get("changed_skipped", []),
        "sensor_synced_count": len(sensor_synced),
        "sensor_synced_sessions": sensor_synced,
        "preprocess": preprocess_summary,
    }
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print_progress(100, "Drive sync complete")


if __name__ == "__main__":
    main()
