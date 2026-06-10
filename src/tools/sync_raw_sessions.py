"""Synchronize raw RC sessions into actions_synced.csv and imu_synced.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import settings
from data.sync import sync_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync raw RC sessions before preprocessing.")
    parser.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help="Optional session dirs. If omitted, sync all sessions in data/raw.",
    )
    parser.add_argument("--raw-dir", type=Path, default=settings.RAW_DATA_DIR)
    parser.add_argument("--dcam-ms", type=float, default=settings.SYNC_CAMERA_DELAY_MS)
    parser.add_argument("--tol-ms", type=float, default=settings.SYNC_TELEMETRY_GAP_TOL_MS)
    parser.add_argument("--keep-all-modes", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    return parser.parse_args()


def find_sessions(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob(settings.SESSION_GLOB) if path.is_dir())


def main() -> None:
    args = parse_args()
    session_dirs = args.sessions or find_sessions(args.raw_dir)
    if not session_dirs:
        raise FileNotFoundError(f"No sessions found under {args.raw_dir}")

    reports = []
    total_kept = 0
    total_dropped = 0
    for session_dir in session_dirs:
        report = sync_session(
            session_dir,
            camera_delay_ms=args.dcam_ms,
            telemetry_gap_tol_ms=args.tol_ms,
            keep_all_modes=args.keep_all_modes,
            write_imu=not args.no_imu,
        )
        report = {"session_id": session_dir.name, "session_dir": str(session_dir), **report}
        reports.append(report)
        total_kept += int(report.get("kept", 0))
        total_dropped += int(report.get("dropped", 0))
        print(
            f"{session_dir.name:>28} kept={report.get('kept', 0):>6} "
            f"dropped={report.get('dropped', 0):>6} status={report.get('status')}",
            flush=True,
        )

    summary = {
        "raw_dir": str(args.raw_dir),
        "session_count": len(session_dirs),
        "total_kept": total_kept,
        "total_dropped": total_dropped,
        "reports": reports,
    }
    settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = settings.REPORT_DIR / "sync_report.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
