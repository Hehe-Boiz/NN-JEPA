from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import data.settings as settings
from data.preprocess import (
    build_session_split,
    build_time_index,
    find_actions_csv_file,
    match_sensor_rows,
    merge_synced_imu_rows,
    read_action,
    read_csv_rows,
    read_state,
    read_timestamp,
    read_timestamp_ms,
    remove_simple_outliers,
)
from data.normalization import build_feature_normalizer

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from data.dataset import TrainAugmentor


class SimplePipelineTests(unittest.TestCase):
    def test_global_settings_keep_expected_schema(self) -> None:
        self.assertEqual(
            list(settings.STATE_COLUMNS),
            ["v_t", "yaw_rate_t", "accel_x_t", "accel_y_t", "steering_last_t", "throttle_last_t"],
        )
        self.assertEqual(
            list(settings.ACTION_COLUMNS),
            ["steering_cmd_t", "throttle_cmd_t"],
        )

    def test_build_session_split_keeps_small_dataset_valid(self) -> None:
        split_map = build_session_split(["s1", "s2", "s3"])
        values = list(split_map.values())
        self.assertEqual(len(values), 3)
        self.assertIn("train", values)
        self.assertIn("val", values)
        self.assertIn("test", values)

    def test_actions_csv_can_be_mapped_to_model_action_and_state(self) -> None:
        row = {
            "frame_idx": "12",
            "t_ms": "1250",
            "steering": "0.35",
            "throttle": "-0.20",
        }
        previous_action = {
            "steering_cmd_t": 0.1,
            "throttle_cmd_t": 0.2,
        }

        action = read_action(row)
        state, missing_columns = read_state(row, previous_action)

        self.assertIsNotNone(action)
        self.assertAlmostEqual(action["steering_cmd_t"], 0.35)
        self.assertAlmostEqual(action["throttle_cmd_t"], -0.20)
        self.assertEqual(state["v_t"], settings.MISSING_STATE_VALUE)
        self.assertEqual(state["yaw_rate_t"], settings.MISSING_STATE_VALUE)
        self.assertAlmostEqual(state["steering_last_t"], 0.1)
        self.assertAlmostEqual(state["throttle_last_t"], 0.2)
        self.assertIn("v_t", missing_columns)
        self.assertIn("steering_last_t", missing_columns)

    def test_android_sensor_streams_can_fill_state(self) -> None:
        row = {
            "frame_idx": "7",
            "t_ms": "1000",
            "steering": "0.10",
            "throttle": "0.20",
        }
        telemetry_rows = [
            {"t_ms": "980", "steering": "0.25", "throttle": "0.35", "seq": "1", "esp_ms": "100", "mode": "1"},
            {"t_ms": "1500", "steering": "0.30", "throttle": "0.40", "seq": "2", "esp_ms": "120", "mode": "1"},
        ]
        accel_rows = [{"t_ms": "1010", "ax": "1.5", "ay": "-0.4", "az": "9.8"}]
        gyro_rows = [{"t_ms": "990", "gx": "0.1", "gy": "0.2", "gz": "0.7"}]
        gps_rows = [{"t_ms": "1100", "speed": "0.55", "lat": "0", "lon": "0", "alt": "0"}]

        aux_streams = {
            "telemetry": build_time_index(telemetry_rows),
            "accel": build_time_index(accel_rows),
            "gyro": build_time_index(gyro_rows),
            "gps": build_time_index(gps_rows),
        }
        sensor_rows = match_sensor_rows(aux_streams, 1000.0)
        previous_action = {
            "steering_cmd_t": 0.05,
            "throttle_cmd_t": 0.15,
        }

        action = read_action(row, telemetry_row=sensor_rows["telemetry"])
        state, missing_columns = read_state(row, previous_action, sensor_rows=sensor_rows)

        self.assertAlmostEqual(action["steering_cmd_t"], 0.25)
        self.assertAlmostEqual(action["throttle_cmd_t"], 0.35)
        self.assertAlmostEqual(state["v_t"], 0.55)
        self.assertAlmostEqual(state["yaw_rate_t"], 0.7)
        self.assertAlmostEqual(state["accel_x_t"], 1.5)
        self.assertAlmostEqual(state["accel_y_t"], -0.4)
        self.assertAlmostEqual(state["steering_last_t"], 0.05)
        self.assertAlmostEqual(state["throttle_last_t"], 0.15)
        self.assertIn("steering_last_t", missing_columns)

    def test_synced_rows_take_precedence_over_raw_streams(self) -> None:
        row = {
            "frame_idx": "7",
            "t_scene_ms": "1000",
            "steering": "0.11",
            "throttle": "0.22",
            "gz": "0.75",
            "ax": "1.5",
            "ay": "-0.4",
        }
        telemetry_rows = [
            {"t_ms": "1000", "steering": "0.90", "throttle": "0.95", "seq": "1", "esp_ms": "100", "mode": "1"},
        ]
        accel_rows = [{"t_ms": "1000", "ax": "9.0", "ay": "8.0", "az": "7.0"}]
        gyro_rows = [{"t_ms": "1000", "gx": "0.1", "gy": "0.2", "gz": "9.9"}]

        aux_streams = {
            "telemetry": build_time_index(telemetry_rows),
            "accel": build_time_index(accel_rows),
            "gyro": build_time_index(gyro_rows),
        }
        sensor_rows = match_sensor_rows(aux_streams, 1000.0)
        previous_action = {
            "steering_cmd_t": 0.05,
            "throttle_cmd_t": 0.15,
        }

        action = read_action(row, telemetry_row=sensor_rows["telemetry"])
        state, _ = read_state(row, previous_action, sensor_rows=sensor_rows)

        self.assertAlmostEqual(action["steering_cmd_t"], 0.11)
        self.assertAlmostEqual(action["throttle_cmd_t"], 0.22)
        self.assertAlmostEqual(state["yaw_rate_t"], 0.75)
        self.assertAlmostEqual(state["accel_x_t"], 1.5)
        self.assertAlmostEqual(state["accel_y_t"], -0.4)

    def test_synced_timestamp_columns_are_supported(self) -> None:
        row = {"t_scene_ms": "1234"}

        self.assertAlmostEqual(read_timestamp_ms(row), 1234.0)
        self.assertAlmostEqual(read_timestamp(row), 1.234)

    def test_synced_csv_is_preferred_and_imu_rows_are_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            (session_dir / settings.ACTIONS_CSV_NAME).write_text(
                "frame_idx,t_ms,steering,throttle\n1,1000,0.90,0.95\n",
                encoding="utf-8",
            )
            (session_dir / settings.SYNCED_ACTIONS_CSV_NAME).write_text(
                "frame_idx,t_scene_ms,steering,throttle,mode\n1,1000,0.11,0.22,1\n",
                encoding="utf-8",
            )
            (session_dir / settings.SYNCED_IMU_CSV_NAME).write_text(
                "frame_idx,t_scene_ms,gx,gy,gz,ax,ay,az,rx,ry,rz\n"
                "1,1000,0.10,0.20,0.75,1.50,-0.40,9.80,0.0,0.0,0.0\n",
                encoding="utf-8",
            )

            csv_path = find_actions_csv_file(session_dir)
            rows = read_csv_rows(csv_path)
            merged_rows, report = merge_synced_imu_rows(rows, csv_path=csv_path, session_dir=session_dir)

            self.assertEqual(csv_path.name, settings.SYNCED_ACTIONS_CSV_NAME)
            self.assertEqual(report["merged_synced_imu_rows"], 1)
            self.assertEqual(report["missing_synced_imu_rows"], 0)
            self.assertEqual(merged_rows[0]["gz"], "0.75")
            self.assertEqual(merged_rows[0]["ax"], "1.50")

    def test_robust_outlier_filter_drops_sensor_spikes(self) -> None:
        samples = []
        for index, accel_y in enumerate([0.0, 0.1, -0.1, 0.2, 50.0]):
            samples.append(
                {
                    "state": {
                        "v_t": 0.0,
                        "yaw_rate_t": 0.0,
                        "accel_x_t": 9.8,
                        "accel_y_t": accel_y,
                    }
                }
            )

        kept, dropped = remove_simple_outliers(samples)

        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 4)
        self.assertNotIn(50.0, [sample["state"]["accel_y_t"] for sample in kept])

    def test_feature_normalizer_uses_train_stats(self) -> None:
        samples = [
            {"state": {"yaw_rate_t": 1.0}},
            {"state": {"yaw_rate_t": 3.0}},
            {"state": {"yaw_rate_t": 5.0}},
        ]

        normalizer = build_feature_normalizer(samples, ["yaw_rate_t"], source_key="state")
        row = normalizer.normalize_row({"yaw_rate_t": 3.0}, ["yaw_rate_t"])

        self.assertAlmostEqual(row[0], 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_horizontal_flip_inverts_control_sensitive_columns(self) -> None:
        old_flip_prob = settings.HORIZONTAL_FLIP_PROB
        old_brightness = settings.BRIGHTNESS_JITTER
        old_contrast = settings.CONTRAST_JITTER
        old_saturation = settings.SATURATION_JITTER
        old_blur_prob = settings.GAUSSIAN_BLUR_PROB
        try:
            settings.HORIZONTAL_FLIP_PROB = 1.0
            settings.BRIGHTNESS_JITTER = 0.0
            settings.CONTRAST_JITTER = 0.0
            settings.SATURATION_JITTER = 0.0
            settings.GAUSSIAN_BLUR_PROB = 0.0

            augmentor = TrainAugmentor()
            image = Image.new("RGB", (4, 2), color=(255, 0, 0))
            state = {
                "v_t": 1.0,
                "yaw_rate_t": 0.5,
                "accel_x_t": 0.2,
                "accel_y_t": 0.3,
                "steering_last_t": 0.4,
                "throttle_last_t": 0.6,
            }
            action = {
                "steering_cmd_t": 0.25,
                "throttle_cmd_t": 0.75,
            }

            _, flipped_state, flipped_action = augmentor(image, state, action)
            self.assertEqual(flipped_state["v_t"], 1.0)
            self.assertEqual(flipped_state["yaw_rate_t"], -0.5)
            self.assertEqual(flipped_state["accel_x_t"], 0.2)
            self.assertEqual(flipped_state["accel_y_t"], -0.3)
            self.assertEqual(flipped_state["steering_last_t"], -0.4)
            self.assertEqual(flipped_state["throttle_last_t"], 0.6)
            self.assertEqual(flipped_action["steering_cmd_t"], -0.25)
            self.assertEqual(flipped_action["throttle_cmd_t"], 0.75)
        finally:
            settings.HORIZONTAL_FLIP_PROB = old_flip_prob
            settings.BRIGHTNESS_JITTER = old_brightness
            settings.CONTRAST_JITTER = old_contrast
            settings.SATURATION_JITTER = old_saturation
            settings.GAUSSIAN_BLUR_PROB = old_blur_prob


if __name__ == "__main__":
    unittest.main()
