"""Simple offline preprocessing for JEPA raw sessions."""

from __future__ import annotations

from bisect import bisect_left
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import settings


ProgressCallback = Callable[[int, int, str], None]


def preprocess_all_sessions(progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    settings.make_output_dirs()

    session_dirs = find_session_dirs()
    if not session_dirs:
        raise FileNotFoundError(f"No session found in {settings.RAW_DATA_DIR}")

    session_samples: dict[str, list[dict[str, Any]]] = {}
    session_reports: list[dict[str, Any]] = []

    total_sessions = len(session_dirs)
    for index, session_dir in enumerate(session_dirs, start=1):
        if progress_callback is not None:
            progress_callback(index - 1, total_sessions, f"Preprocessing {session_dir.name}")
        samples, report = preprocess_one_session(session_dir)
        session_reports.append(report)
        if samples:
            session_samples[session_dir.name] = samples
    if progress_callback is not None:
        progress_callback(total_sessions, total_sessions, "Writing manifests and report")

    if not session_samples:
        raise RuntimeError("No usable sample found after preprocessing")

    split_map = build_session_split(list(session_samples.keys()), include_test=settings.USE_TEST_SPLIT)
    manifest_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    manifest_sessions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_samples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for session_id, samples in session_samples.items():
        split_name = split_map[session_id]
        manifest_sessions[split_name].append(session_id)
        for sample in samples:
            next_sample = dict(sample)
            next_sample["split"] = split_name
            split_samples[split_name].append(next_sample)

    test_is_val_alias = not settings.USE_TEST_SPLIT and settings.ALIAS_TEST_TO_VAL
    if test_is_val_alias:
        manifest_sessions["test"] = list(manifest_sessions["val"])
        split_samples["test"] = [
            {**sample, "split": "test"}
            for sample in split_samples["val"]
        ]

    for split_name, samples in split_samples.items():
        write_jsonl(settings.MANIFEST_DIR / f"{split_name}.jsonl", samples)
        manifest_counts[split_name] = len(samples)

    summary = {
        "raw_data_dir": str(settings.RAW_DATA_DIR),
        "processed_data_dir": str(settings.PROCESSED_DATA_DIR),
        "use_test_split": settings.USE_TEST_SPLIT,
        "test_is_val_alias": test_is_val_alias,
        "counts": manifest_counts,
        "preferred_training_files": {
            "actions": settings.SYNCED_ACTIONS_CSV_NAME,
            "imu": settings.SYNCED_IMU_CSV_NAME,
        },
        "sync_fallback_sessions": sorted(
            report["session_id"]
            for report in session_reports
            if report.get("action_source") != settings.SYNCED_ACTIONS_CSV_NAME
        ),
        "sessions": {
            key: sorted(set(value))
            for key, value in manifest_sessions.items()
        },
        "feature_stats": compute_feature_stats(session_samples),
        "session_reports": session_reports,
    }

    report_path = settings.REPORT_DIR / "preprocess_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def find_session_dirs() -> list[Path]:
    return sorted(
        [path for path in settings.RAW_DATA_DIR.glob(settings.SESSION_GLOB) if path.is_dir()]
    )


def preprocess_one_session(session_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = find_actions_csv_file(session_dir)
    frame_map = build_frame_map(session_dir / settings.FRAME_DIR_NAME)
    aux_streams = load_aux_streams(session_dir)
    rows = read_csv_rows(csv_path)
    rows, synced_imu_report = merge_synced_imu_rows(rows, csv_path=csv_path, session_dir=session_dir)

    report = {
        "session_id": session_dir.name,
        "csv_file": csv_path.name,
        "action_source": csv_path.name,
        "raw_rows": len(rows),
        "kept_rows": 0,
        "dropped_missing_frame": 0,
        "dropped_missing_action": 0,
        "dropped_duplicate_frame": 0,
        "dropped_out_of_range": 0,
        "dropped_by_stride": 0,
        "filled_state_rows": 0,
        "missing_state_columns": [],
        "matched_telemetry_rows": 0,
        "matched_accel_rows": 0,
        "matched_gyro_rows": 0,
        "matched_gps_rows": 0,
    }
    report.update(synced_imu_report)

    used_missing_state_columns: set[str] = set()
    seen_frame_indices: set[int] = set()
    previous_action = {
        "steering_cmd_t": settings.DEFAULT_STEERING_LAST,
        "throttle_cmd_t": settings.DEFAULT_THROTTLE_LAST,
    }
    session_samples: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows):
        if settings.USE_EVERY_NTH_FRAME > 1 and row_number % settings.USE_EVERY_NTH_FRAME != 0:
            report["dropped_by_stride"] += 1
            continue

        frame_index = get_frame_index(row, row_number)
        if frame_index is None:
            report["dropped_missing_frame"] += 1
            continue

        if settings.DROP_DUPLICATE_FRAME_INDEX and frame_index in seen_frame_indices:
            report["dropped_duplicate_frame"] += 1
            continue

        frame_path = frame_map.get(frame_index)
        if settings.DROP_ROWS_WITH_MISSING_FRAME and frame_path is None:
            report["dropped_missing_frame"] += 1
            continue

        row_t_ms = read_timestamp_ms(row)
        sensor_rows = match_sensor_rows(aux_streams, row_t_ms)
        if sensor_rows.get("telemetry") is not None:
            report["matched_telemetry_rows"] += 1
        if sensor_rows.get("accel") is not None:
            report["matched_accel_rows"] += 1
        if sensor_rows.get("gyro") is not None:
            report["matched_gyro_rows"] += 1
        if sensor_rows.get("gps") is not None:
            report["matched_gps_rows"] += 1

        action = read_action(row, telemetry_row=sensor_rows.get("telemetry"))
        if action is None:
            report["dropped_missing_action"] += 1
            continue

        action["steering_cmd_t"] *= settings.STEERING_SCALE
        action["throttle_cmd_t"] *= settings.THROTTLE_SCALE

        if not action_in_valid_range(action):
            report["dropped_out_of_range"] += 1
            continue

        state, missing_state_columns = read_state(
            row,
            previous_action,
            sensor_rows=sensor_rows,
        )
        if missing_state_columns:
            report["filled_state_rows"] += 1
            used_missing_state_columns.update(missing_state_columns)

        if frame_path is None:
            continue

        sample_id = f"{session_dir.name}_{frame_index:06d}"
        output_path = settings.PROCESSED_IMAGE_DIR / session_dir.name / f"{frame_index:06d}.{settings.IMAGE_FORMAT}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_image(frame_path, output_path)

        sample = {
            "sample_id": sample_id,
            "session_id": session_dir.name,
            "frame_index": frame_index,
            "timestamp_sec": read_timestamp(row),
            "frame_path": str(output_path),
            "source_frame_path": str(frame_path),
            "state": state,
            "action": action,
            "meta": keep_meta_fields(row, sensor_rows=sensor_rows, target_t_ms=row_t_ms),
        }
        session_samples.append(sample)
        previous_action = dict(action)
        seen_frame_indices.add(frame_index)

    if settings.REMOVE_SIMPLE_OUTLIERS:
        session_samples, dropped_outliers = remove_simple_outliers(session_samples)
        report["dropped_simple_outliers"] = dropped_outliers
    else:
        report["dropped_simple_outliers"] = 0

    report["kept_rows"] = len(session_samples)
    report["missing_state_columns"] = sorted(used_missing_state_columns)
    report["status"] = "ok" if len(session_samples) >= settings.MIN_SESSION_SAMPLES else "too_short"

    if len(session_samples) < settings.MIN_SESSION_SAMPLES:
        return [], report
    return session_samples, report


def find_actions_csv_file(session_dir: Path) -> Path:
    for filename in (settings.SYNCED_ACTIONS_CSV_NAME, settings.ACTIONS_CSV_NAME):
        path = session_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {settings.SYNCED_ACTIONS_CSV_NAME} or {settings.ACTIONS_CSV_NAME} in {session_dir}"
    )


def merge_synced_imu_rows(
    rows: list[dict[str, str]],
    csv_path: Path,
    session_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    if csv_path.name != settings.SYNCED_ACTIONS_CSV_NAME:
        return rows, {
            "merged_synced_imu_rows": 0,
            "missing_synced_imu_rows": 0,
        }

    imu_path = session_dir / settings.SYNCED_IMU_CSV_NAME
    if not imu_path.exists():
        return rows, {
            "merged_synced_imu_rows": 0,
            "missing_synced_imu_rows": len(rows),
        }

    imu_rows = read_csv_rows(imu_path)
    imu_by_frame_index: dict[int, dict[str, str]] = {}
    for imu_row in imu_rows:
        frame_index = read_frame_index_from_row(imu_row)
        if frame_index is None:
            continue
        imu_by_frame_index[frame_index] = imu_row

    merged_rows: list[dict[str, str]] = []
    merged_count = 0
    missing_count = 0
    for row in rows:
        frame_index = read_frame_index_from_row(row)
        merged_row = dict(row)
        imu_row = None if frame_index is None else imu_by_frame_index.get(frame_index)
        if imu_row is None:
            missing_count += 1
        else:
            for key, value in imu_row.items():
                if key == "frame_idx":
                    continue
                if merged_row.get(key) in (None, "") and value not in (None, ""):
                    merged_row[key] = value
            merged_count += 1
        merged_rows.append(merged_row)

    return merged_rows, {
        "merged_synced_imu_rows": merged_count,
        "missing_synced_imu_rows": missing_count,
    }


def load_aux_streams(session_dir: Path) -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    stream_files = {
        "telemetry": settings.TELEMETRY_CSV_NAME,
        "accel": settings.ACCEL_CSV_NAME,
        "gyro": settings.GYRO_CSV_NAME,
        "rotvec": settings.ROTVEC_CSV_NAME,
        "gps": settings.GPS_CSV_NAME,
    }
    for stream_name, filename in stream_files.items():
        path = session_dir / filename
        if not path.exists():
            continue
        indexed_stream = build_time_index(read_csv_rows(path))
        if indexed_stream is not None:
            streams[stream_name] = indexed_stream
    return streams


def build_frame_map(frames_dir: Path) -> dict[int, Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frame directory: {frames_dir}")

    frame_paths = [
        path
        for path in sorted(frames_dir.iterdir(), key=natural_sort_key)
        if path.suffix.lower() in settings.FRAME_EXTENSIONS
    ]
    frame_map: dict[int, Path] = {}
    for order, path in enumerate(frame_paths, start=1):
        stem_digits = extract_digits(path.stem)
        frame_index = stem_digits if stem_digits is not None else order
        frame_map[frame_index] = path
    return frame_map


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def get_frame_index(row: dict[str, str], row_number: int) -> int | None:
    frame_index = read_frame_index_from_row(row)
    if frame_index is not None:
        return frame_index
    return row_number + 1


def read_frame_index_from_row(row: dict[str, str]) -> int | None:
    for key in settings.FRAME_INDEX_KEYS:
        if key in row and row[key] not in (None, ""):
            try:
                return int(float(row[key]))
            except ValueError:
                return None
    return None


def read_timestamp(row: dict[str, str]) -> float | None:
    for key in ("t_ms", "t_scene_ms"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value) / 1000.0
            except ValueError:
                return None

    for key in ("timestamp_sec", "t_scene", "t_pc"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return None
    return None


def read_timestamp_ms(row: dict[str, str]) -> float | None:
    for key in ("t_ms", "t_scene_ms"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return None

    value = row.get("timestamp_sec")
    if value not in (None, ""):
        try:
            return float(value) * 1000.0
        except ValueError:
            return None

    for key in ("t_scene", "t_pc"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value) * 1000.0
            except ValueError:
                return None
    return None


def read_action(
    row: dict[str, str],
    telemetry_row: dict[str, str] | None = None,
) -> dict[str, float] | None:
    action: dict[str, float] = {}
    prefer_row_values = uses_synced_action_row(row)
    for target_key, source_keys in settings.ACTION_SOURCE_KEYS.items():
        value = None
        if prefer_row_values:
            value = read_first_float(row, source_keys)
        if value is None and settings.USE_RAW_TELEMETRY_FOR_ACTION and telemetry_row is not None:
            value = read_first_float(telemetry_row, source_keys)
        if value is None:
            value = read_first_float(row, source_keys)
        if value is None and settings.DROP_ROWS_WITH_MISSING_ACTION:
            return None
        action[target_key] = value if value is not None else 0.0
    return action


def read_state(
    row: dict[str, str],
    previous_action: dict[str, float],
    sensor_rows: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, float], list[str]]:
    sensor_rows = sensor_rows or {}
    state: dict[str, float] = {}
    missing_columns: list[str] = []

    for key, preferred_streams in (
        ("v_t", ("gps",)),
        ("yaw_rate_t", ("gyro",)),
        ("accel_x_t", ("accel",)),
        ("accel_y_t", ("accel",)),
    ):
        value = read_state_value(
            key,
            row,
            sensor_rows=sensor_rows,
            preferred_streams=preferred_streams,
        )
        if value is None:
            if not settings.ALLOW_ACTIONS_ONLY_SESSIONS:
                raise ValueError(f"Missing state value for {key}")
            value = settings.MISSING_STATE_VALUE
            missing_columns.append(key)
        state[key] = value

    steering_last = read_state_value("steering_last_t", row, sensor_rows=sensor_rows)
    throttle_last = read_state_value("throttle_last_t", row, sensor_rows=sensor_rows)

    if steering_last is None:
        steering_last = (
            previous_action["steering_cmd_t"]
            if settings.USE_PREVIOUS_ACTION_AS_LAST_CONTROL
            else settings.DEFAULT_STEERING_LAST
        )
        missing_columns.append("steering_last_t")

    if throttle_last is None:
        throttle_last = (
            previous_action["throttle_cmd_t"]
            if settings.USE_PREVIOUS_ACTION_AS_LAST_CONTROL
            else settings.DEFAULT_THROTTLE_LAST
        )
        missing_columns.append("throttle_last_t")

    state["steering_last_t"] = steering_last
    state["throttle_last_t"] = throttle_last
    return state, missing_columns


def keep_meta_fields(
    row: dict[str, str],
    sensor_rows: dict[str, dict[str, str]] | None = None,
    target_t_ms: float | None = None,
) -> dict[str, Any]:
    sensor_rows = sensor_rows or {}
    meta: dict[str, Any] = {}
    for key in ("t_ms", "t_scene_ms", "t_pc", "t_scene", "latency", "seq", "esp_ms", "mode", "dcam_ms"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            meta[key] = float(value) if "." in value else int(value)
        except ValueError:
            meta[key] = value

    for stream_name, stream_row in sensor_rows.items():
        if stream_row is None:
            continue
        t_ms = read_timestamp_ms(stream_row)
        if t_ms is None:
            continue
        meta[f"{stream_name}_t_ms"] = round(t_ms, 1)
        if target_t_ms is not None:
            meta[f"{stream_name}_dt_ms"] = round(t_ms - target_t_ms, 1)
    return meta


def action_in_valid_range(action: dict[str, float]) -> bool:
    if not settings.DROP_ROWS_OUTSIDE_ACTION_RANGE:
        return True
    steer_ok = settings.STEERING_MIN <= action["steering_cmd_t"] <= settings.STEERING_MAX
    throttle_ok = settings.THROTTLE_MIN <= action["throttle_cmd_t"] <= settings.THROTTLE_MAX
    return steer_ok and throttle_ok


def prepare_image(source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        if settings.RESIZE_IMAGES:
            rgb_image = rgb_image.resize((settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT), Image.Resampling.BILINEAR)
        rgb_image.save(output_path)


def remove_simple_outliers(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not samples:
        return samples, 0

    limits: dict[str, tuple[float, float]] = {}
    for column in settings.OUTLIER_COLUMNS:
        values = [sample["state"][column] for sample in samples]
        if not values:
            continue
        center = median(values)
        deviations = [abs(value - center) for value in values]
        robust_std = 1.4826 * median(deviations)
        if math.isclose(robust_std, 0.0):
            robust_std = compute_std(values)
        if math.isclose(robust_std, 0.0):
            continue
        limits[column] = (
            center - settings.OUTLIER_STD_FACTOR * robust_std,
            center + settings.OUTLIER_STD_FACTOR * robust_std,
        )

    kept: list[dict[str, Any]] = []
    dropped = 0
    for sample in samples:
        keep = True
        for column, (low, high) in limits.items():
            value = sample["state"][column]
            if value < low or value > high:
                keep = False
                break
        if keep:
            kept.append(sample)
        else:
            dropped += 1
    return kept, dropped


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def build_session_split(session_ids: list[str], include_test: bool | None = None) -> dict[str, str]:
    import random

    include_test = settings.USE_TEST_SPLIT if include_test is None else include_test
    shuffled = session_ids[:]
    random.Random(settings.RANDOM_SEED).shuffle(shuffled)
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


def compute_feature_stats(session_samples: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    flat_samples = [sample for samples in session_samples.values() for sample in samples]
    if not flat_samples:
        return {}

    all_columns = list(settings.STATE_COLUMNS) + list(settings.ACTION_COLUMNS)
    stats: dict[str, dict[str, float]] = {}
    for column in all_columns:
        values: list[float] = []
        for sample in flat_samples:
            if column in sample["state"]:
                values.append(sample["state"][column])
            elif column in sample["action"]:
                values.append(sample["action"][column])
        if not values:
            continue
        stats[column] = {
            "mean": float(sum(values) / len(values)),
            "std": float(compute_std(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return stats


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    normalized: list[Any] = []
    for part in parts:
        if part.isdigit():
            normalized.append(int(part))
        else:
            normalized.append(part.lower())
    return tuple(normalized)


def compute_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def extract_digits(text: str) -> int | None:
    digits = "".join(char for char in text if char.isdigit())
    return int(digits) if digits else None


def read_first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    if row is None:
        return None
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def read_state_value(
    key: str,
    row: dict[str, str],
    sensor_rows: dict[str, dict[str, str]] | None = None,
    preferred_streams: tuple[str, ...] = (),
) -> float | None:
    sensor_rows = sensor_rows or {}
    source_keys = settings.STATE_SOURCE_KEYS[key]

    value = read_first_float(row, source_keys)
    if value is not None:
        return value

    for stream_name in preferred_streams:
        stream_row = sensor_rows.get(stream_name)
        value = read_first_float(stream_row, source_keys)
        if value is not None:
            return value

    return None


def uses_synced_action_row(row: dict[str, str]) -> bool:
    value = row.get("t_scene_ms")
    return value not in (None, "")


def build_time_index(rows: list[dict[str, str]]) -> dict[str, list[Any]] | None:
    indexed_rows: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        t_ms = read_timestamp_ms(row)
        if t_ms is None:
            continue
        indexed_rows.append((t_ms, row))

    if not indexed_rows:
        return None

    indexed_rows.sort(key=lambda item: item[0])
    return {
        "times": [item[0] for item in indexed_rows],
        "rows": [item[1] for item in indexed_rows],
    }


def match_sensor_rows(
    aux_streams: dict[str, dict[str, Any]],
    target_t_ms: float | None,
) -> dict[str, dict[str, str] | None]:
    if target_t_ms is None:
        return {
            "telemetry": None,
            "accel": None,
            "gyro": None,
            "rotvec": None,
            "gps": None,
        }

    return {
        "telemetry": nearest_row(aux_streams.get("telemetry"), target_t_ms, settings.TELEMETRY_MATCH_TOL_MS),
        "accel": nearest_row(aux_streams.get("accel"), target_t_ms, settings.ACCEL_MATCH_TOL_MS),
        "gyro": nearest_row(aux_streams.get("gyro"), target_t_ms, settings.GYRO_MATCH_TOL_MS),
        "rotvec": nearest_row(aux_streams.get("rotvec"), target_t_ms, settings.ROTVEC_MATCH_TOL_MS),
        "gps": nearest_row(aux_streams.get("gps"), target_t_ms, settings.GPS_MATCH_TOL_MS),
    }


def nearest_row(
    stream: dict[str, list[Any]] | None,
    target_t_ms: float,
    tolerance_ms: float,
) -> dict[str, str] | None:
    if stream is None:
        return None

    times = stream["times"]
    rows = stream["rows"]
    if not times:
        return None

    pos = bisect_left(times, target_t_ms)
    candidates: list[int] = []
    if pos < len(times):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return None

    best_index = min(candidates, key=lambda index: abs(times[index] - target_t_ms))
    if abs(times[best_index] - target_t_ms) > tolerance_ms:
        return None
    return rows[best_index]
