"""JEPA-style synchronization for raw RC driving sessions.

This module rebuilds per-frame action and IMU rows before preprocessing:

* actions_synced.csv: frame_idx,t_scene_ms,steering,throttle,mode
* imu_synced.csv: frame_idx,t_scene_ms,gx,gy,gz,ax,ay,az,rx,ry,rz

The logic mirrors the JEPA hardware repo:
old logs without a dcam_ms column use t_ms - camera_delay_ms, while newer logs
with dcam_ms are treated as already using scene/exposure time.
"""

from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Any

from . import settings


IMU_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (settings.GYRO_CSV_NAME, ("gx", "gy", "gz")),
    (settings.ACCEL_CSV_NAME, ("ax", "ay", "az")),
    (settings.ROTVEC_CSV_NAME, ("rx", "ry", "rz")),
)


def sync_session(
    session_dir: str | Path,
    *,
    camera_delay_ms: float = settings.SYNC_CAMERA_DELAY_MS,
    telemetry_gap_tol_ms: float = settings.SYNC_TELEMETRY_GAP_TOL_MS,
    keep_all_modes: bool = settings.SYNC_KEEP_ALL_MODES,
    write_imu: bool = settings.SYNC_WRITE_IMU,
) -> dict[str, Any]:
    """Synchronize one raw session and return a JSON-safe report.

    Missing telemetry is not fatal because old or minimal sessions may only have
    actions.csv. In that case preprocessing can still fall back to raw actions.
    """
    session_path = Path(session_dir)
    actions_path = session_path / settings.ACTIONS_CSV_NAME
    telemetry_path = session_path / settings.TELEMETRY_CSV_NAME
    if not actions_path.exists() or not telemetry_path.exists():
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_actions_or_telemetry",
            "kept": 0,
            "dropped": 0,
        }

    telemetry = load_telemetry(telemetry_path)
    if len(telemetry["t_ms"]) < 2:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "empty_telemetry",
            "kept": 0,
            "dropped": 0,
        }

    offset_ms = 0.0 if has_dcam_column(actions_path) else float(camera_delay_ms)
    kept: list[tuple[int, float, float, float, int]] = []
    dropped = 0

    with actions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame_index = int(float(row["frame_idx"]))
                scene_time_ms = float(row["t_ms"]) - offset_ms
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue

            bracket = bisect.bisect_left(telemetry["t_ms"], scene_time_ms)
            if bracket == 0 or bracket >= len(telemetry["t_ms"]):
                dropped += 1
                continue
            if telemetry["t_ms"][bracket] - telemetry["t_ms"][bracket - 1] > telemetry_gap_tol_ms:
                dropped += 1
                continue

            nearest_mode = nearest_value(
                telemetry["t_ms"],
                telemetry["mode"],
                bracket,
                scene_time_ms,
            )
            if not keep_all_modes and int(nearest_mode) != 1:
                dropped += 1
                continue

            steering = interpolate_bracket(
                telemetry["t_ms"],
                telemetry["steering"],
                bracket,
                scene_time_ms,
            )
            throttle = interpolate_bracket(
                telemetry["t_ms"],
                telemetry["throttle"],
                bracket,
                scene_time_ms,
            )
            kept.append((frame_index, scene_time_ms, steering, throttle, int(nearest_mode)))

    if not kept:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_valid_synced_rows",
            "kept": 0,
            "dropped": dropped,
            "camera_delay_applied_ms": offset_ms,
        }

    write_actions_synced(session_path / settings.SYNCED_ACTIONS_CSV_NAME, kept)
    imu_written = write_imu_synced(session_path, kept) if write_imu else False
    return {
        "enabled": True,
        "status": "ok",
        "reason": "",
        "kept": len(kept),
        "dropped": dropped,
        "camera_delay_applied_ms": offset_ms,
        "telemetry_gap_tol_ms": float(telemetry_gap_tol_ms),
        "keep_all_modes": bool(keep_all_modes),
        "imu_written": bool(imu_written),
    }


def sync_sessions(
    session_dirs: list[Path],
    *,
    camera_delay_ms: float = settings.SYNC_CAMERA_DELAY_MS,
    telemetry_gap_tol_ms: float = settings.SYNC_TELEMETRY_GAP_TOL_MS,
    keep_all_modes: bool = settings.SYNC_KEEP_ALL_MODES,
    write_imu: bool = settings.SYNC_WRITE_IMU,
) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session_dir.name,
            **sync_session(
                session_dir,
                camera_delay_ms=camera_delay_ms,
                telemetry_gap_tol_ms=telemetry_gap_tol_ms,
                keep_all_modes=keep_all_modes,
                write_imu=write_imu,
            ),
        }
        for session_dir in session_dirs
    ]


def load_telemetry(path: Path) -> dict[str, list[float]]:
    telemetry = {
        "t_ms": [],
        "steering": [],
        "throttle": [],
        "mode": [],
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                telemetry["t_ms"].append(float(row["t_ms"]))
                telemetry["steering"].append(float(row["steering"]))
                telemetry["throttle"].append(float(row["throttle"]))
                telemetry["mode"].append(float(row["mode"]))
            except (KeyError, TypeError, ValueError):
                continue
    return telemetry


def has_dcam_column(actions_path: Path) -> bool:
    with actions_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(",")
    return "dcam_ms" in header


def nearest_value(times: list[float], values: list[float], bracket: int, target_time: float) -> float:
    left = bracket - 1
    right = bracket
    if times[right] - target_time < target_time - times[left]:
        return values[right]
    return values[left]


def interpolate_bracket(
    times: list[float],
    values: list[float],
    bracket: int,
    target_time: float,
) -> float:
    left = bracket - 1
    right = bracket
    t0 = times[left]
    t1 = times[right]
    if t1 == t0:
        return values[right]
    weight = (target_time - t0) / (t1 - t0)
    return values[left] + weight * (values[right] - values[left])


def load_sensor_series(path: Path) -> tuple[list[float], list[list[float]]]:
    if not path.exists():
        return [], []
    times: list[float] = []
    columns: list[list[float]] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            try:
                timestamp = float(row[0])
                values = [float(value) for value in row[1:]]
            except (IndexError, ValueError):
                continue
            if columns is None:
                columns = [[] for _ in values]
            times.append(timestamp)
            for index, value in enumerate(values):
                if index < len(columns):
                    columns[index].append(value)
    return times, columns or []


def interpolate_at(times: list[float], values: list[float], target_time: float) -> float:
    if not times:
        return 0.0
    if target_time <= times[0]:
        return values[0]
    if target_time >= times[-1]:
        return values[-1]
    bracket = bisect.bisect_left(times, target_time)
    return interpolate_bracket(times, values, bracket, target_time)


def write_actions_synced(path: Path, kept: list[tuple[int, float, float, float, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "t_scene_ms", "steering", "throttle", "mode"])
        for frame_index, scene_time_ms, steering, throttle, mode in kept:
            writer.writerow(
                [
                    frame_index,
                    int(round(scene_time_ms)),
                    f"{steering:.4f}",
                    f"{throttle:.4f}",
                    mode,
                ]
            )


def write_imu_synced(session_dir: Path, kept: list[tuple[int, float, float, float, int]]) -> bool:
    streams: list[tuple[list[float], list[list[float]], tuple[str, ...]]] = []
    header = ["frame_idx", "t_scene_ms"]
    for filename, names in IMU_FILES:
        times, columns = load_sensor_series(session_dir / filename)
        if times and len(columns) >= len(names):
            streams.append((times, columns, names))
            header.extend(names)
    if not streams:
        return False

    with (session_dir / settings.SYNCED_IMU_CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame_index, scene_time_ms, *_ in kept:
            row: list[Any] = [frame_index, int(round(scene_time_ms))]
            for times, columns, names in streams:
                for index in range(len(names)):
                    row.append(f"{interpolate_at(times, columns[index], scene_time_ms):.5f}")
            writer.writerow(row)
    return True
