"""Global settings for the JEPA data pipeline.

Edit this file directly when you want to change dataset paths, frame size,
split ratio, or how missing state values are handled.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
JEPA_ROOT = REPO_ROOT / "JEPA"

RAW_DATA_DIR = JEPA_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = JEPA_ROOT / "data" / "processed"
PROCESSED_IMAGE_DIR = PROCESSED_DATA_DIR / "images"
MANIFEST_DIR = PROCESSED_DATA_DIR / "manifests"
REPORT_DIR = PROCESSED_DATA_DIR / "reports"


# ---------------------------------------------------------------------------
# Session structure
# Current PRIMARY layout after JEPA onboard Android update:
#   JEPA/data/raw/session_xxx/
#     frames/
#     actions.csv
#     telemetry.csv
#     accel.csv
#     gyro.csv
#     rotvec.csv
#     gps.csv
#     meta.json
# Old PC recorder sessions still work as long as they contain frames/ + actions.csv.
# ---------------------------------------------------------------------------
SESSION_GLOB = "session_*"
FRAME_DIR_NAME = "frames"
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")
ACTIONS_CSV_NAME = "actions.csv"
TELEMETRY_CSV_NAME = "telemetry.csv"
ACCEL_CSV_NAME = "accel.csv"
GYRO_CSV_NAME = "gyro.csv"
ROTVEC_CSV_NAME = "rotvec.csv"
GPS_CSV_NAME = "gps.csv"
META_JSON_NAME = "meta.json"


# ---------------------------------------------------------------------------
# Columns used by the model
# s_t = [v_t, yaw_rate_t, accel_x_t, accel_y_t, steering_last_t, throttle_last_t]
# a_t = [steering_cmd_t, throttle_cmd_t]
# ---------------------------------------------------------------------------
STATE_COLUMNS = (
    "v_t",
    "yaw_rate_t",
    "accel_x_t",
    "accel_y_t",
    "steering_last_t",
    "throttle_last_t",
)

ACTION_COLUMNS = (
    "steering_cmd_t",
    "throttle_cmd_t",
)


# ---------------------------------------------------------------------------
# Action-conditioned JEPA world model defaults
# Keep these separate from STATE_COLUMNS so the old single-step BC baseline can
# still use the full state vector while the AC world model can drop weak inputs.
# ---------------------------------------------------------------------------
AC_STATE_COLUMNS = (
    "yaw_rate_t",
    "accel_x_t",
    "accel_y_t",
    "steering_last_t",
    "throttle_last_t",
)

AC_ACTION_COLUMNS = ACTION_COLUMNS
AC_RAW_FRAMES_PER_SAMPLE = 8
AC_SEQUENCE_STRIDE = 1
AC_IMAGE_SIZE = 384
AC_TUBELET_SIZE = 2
AC_AUTO_STEPS = 2


# ---------------------------------------------------------------------------
# How to read the current JEPA recorder CSV
# Android onboard recorder:
#   actions.csv   -> frame_idx,t_ms,steering,throttle,seq,esp_ms,mode
#   telemetry.csv -> t_ms,seq,esp_ms,steering,throttle,mode
#   accel.csv     -> t_ms,ax,ay,az
#   gyro.csv      -> t_ms,gx,gy,gz
#   rotvec.csv    -> t_ms,rx,ry,rz
#   gps.csv       -> t_ms,lat,lon,alt,speed,bearing,acc
# Old PC recorder:
#   actions.csv   -> frame_idx,t_pc,t_scene,steering,throttle,latency,seq,esp_ms,mode
# ---------------------------------------------------------------------------
FRAME_INDEX_KEYS = ("frame_idx",)
TIMESTAMP_KEYS = ("t_ms", "timestamp_sec", "t_scene", "t_pc")

ACTION_SOURCE_KEYS = {
    "steering_cmd_t": ("steering_cmd_t", "steering"),
    "throttle_cmd_t": ("throttle_cmd_t", "throttle"),
}

STATE_SOURCE_KEYS = {
    "v_t": ("v_t", "speed", "velocity"),
    "yaw_rate_t": ("yaw_rate_t", "yaw_rate", "gyro_z", "gz"),
    "accel_x_t": ("accel_x_t", "accel_x", "ax"),
    "accel_y_t": ("accel_y_t", "accel_y", "ay"),
    "steering_last_t": ("steering_last_t", "steering_last"),
    "throttle_last_t": ("throttle_last_t", "throttle_last"),
}


# ---------------------------------------------------------------------------
# Auxiliary sensor streams
# All Android logs share the same t_ms clock, so we can match them by nearest time.
# ---------------------------------------------------------------------------
USE_RAW_TELEMETRY_FOR_ACTION = True
TELEMETRY_MATCH_TOL_MS = 60.0
ACCEL_MATCH_TOL_MS = 80.0
GYRO_MATCH_TOL_MS = 80.0
ROTVEC_MATCH_TOL_MS = 80.0
GPS_MATCH_TOL_MS = 500.0


# ---------------------------------------------------------------------------
# Missing state policy
# Android recorder now logs accel/gyro/gps, but not every session will have every file.
# If a state cannot be matched from row/sensor data, it falls back to these defaults.
# ---------------------------------------------------------------------------
ALLOW_ACTIONS_ONLY_SESSIONS = True
MISSING_STATE_VALUE = 0.0
USE_PREVIOUS_ACTION_AS_LAST_CONTROL = True
DEFAULT_STEERING_LAST = 0.0
DEFAULT_THROTTLE_LAST = 0.0


# ---------------------------------------------------------------------------
# Simple filtering / cleaning
# Keep these global so you can tune quickly while collecting data.
# ---------------------------------------------------------------------------
USE_EVERY_NTH_FRAME = 1
MIN_SESSION_SAMPLES = 8
DROP_DUPLICATE_FRAME_INDEX = True
DROP_ROWS_WITH_MISSING_FRAME = True
DROP_ROWS_WITH_MISSING_ACTION = True
DROP_ROWS_OUTSIDE_ACTION_RANGE = True
REMOVE_SIMPLE_OUTLIERS = False
OUTLIER_STD_FACTOR = 4.0
OUTLIER_COLUMNS = ("v_t", "yaw_rate_t", "accel_x_t", "accel_y_t")

STEERING_MIN = -1.0
STEERING_MAX = 1.0
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0


# ---------------------------------------------------------------------------
# Model-side scaling
# PLAN.md notes throttle may be smaller than steering in practice.
# These multipliers let you rebalance inputs without changing raw logs.
# ---------------------------------------------------------------------------
STEERING_SCALE = 1.0
THROTTLE_SCALE = 1.0


# ---------------------------------------------------------------------------
# Image preprocessing
# Recorder already saves frames, so this pipeline keeps things simple:
# resize once offline, normalize later in the Dataset.
# ---------------------------------------------------------------------------
RESIZE_IMAGES = True
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_FORMAT = "jpg"
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Train / val / test split
# Split by session to avoid trajectory leakage.
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SHUFFLE_TRAIN = True


# ---------------------------------------------------------------------------
# Train augmentation
# Keep horizontal flip off by default. In control tasks it can be dangerous
# unless the sign convention is verified carefully.
# ---------------------------------------------------------------------------
BRIGHTNESS_JITTER = 0.15
CONTRAST_JITTER = 0.15
SATURATION_JITTER = 0.05
GAUSSIAN_BLUR_PROB = 0.1
GAUSSIAN_BLUR_RADIUS = 1.0
HORIZONTAL_FLIP_PROB = 0.0


def make_output_dirs() -> None:
    """Create output folders used by the pipeline."""
    for path in (PROCESSED_DATA_DIR, PROCESSED_IMAGE_DIR, MANIFEST_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
