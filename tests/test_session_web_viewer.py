from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from data import settings
from tools.session_web_viewer import (
    PROGRESS_PREFIX,
    build_extract_feature_command,
    build_job_command,
    build_preprocess_command,
    build_session_payload,
    build_sync_command,
    completed_progress,
    collect_sessions,
    get_frames_dir,
    initial_progress,
    list_frame_paths,
    parse_progress_line,
)
from models.vjepa21_presets import vjepa21_feature_preset_options


class SessionWebViewerTests(unittest.TestCase):
    def test_collect_sessions_reads_raw_and_processed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            processed_root = root / "processed_images"
            (raw_root / "session_1" / "frames").mkdir(parents=True)
            (processed_root / "session_1").mkdir(parents=True)
            (raw_root / "session_1" / "frames" / "000001.jpg").write_bytes(b"raw")
            (processed_root / "session_1" / "000001.jpg").write_bytes(b"processed")
            (raw_root / "session_1" / settings.SYNCED_ACTIONS_CSV_NAME).write_text("frame_idx\n1\n", encoding="utf-8")

            with mock.patch.object(settings, "RAW_DATA_DIR", raw_root), mock.patch.object(
                settings, "PROCESSED_IMAGE_DIR", processed_root
            ):
                raw_sessions = collect_sessions("raw")
                processed_sessions = collect_sessions("processed")

        self.assertEqual(raw_sessions[0]["session_id"], "session_1")
        self.assertEqual(raw_sessions[0]["frame_count"], 1)
        self.assertIn(settings.SYNCED_ACTIONS_CSV_NAME, raw_sessions[0]["files"])
        self.assertEqual(processed_sessions[0]["files"], ["processed_images"])

    def test_build_session_payload_returns_sorted_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            frames_dir = raw_root / "session_1" / "frames"
            frames_dir.mkdir(parents=True)
            (frames_dir / "000002.jpg").write_bytes(b"2")
            (frames_dir / "000001.jpg").write_bytes(b"1")

            with mock.patch.object(settings, "RAW_DATA_DIR", raw_root):
                payload = build_session_payload(source="raw", session_id="session_1")

        self.assertEqual(payload["frames"], ["000001.jpg", "000002.jpg"])
        self.assertEqual(payload["frame_count"], 2)

    def test_list_frame_paths_filters_non_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            (frames_dir / "000001.jpg").write_bytes(b"1")
            (frames_dir / "note.txt").write_text("x", encoding="utf-8")

            frame_paths = list_frame_paths(frames_dir)

        self.assertEqual([path.name for path in frame_paths], ["000001.jpg"])

    def test_get_frames_dir_for_raw_and_processed(self) -> None:
        self.assertEqual(
            get_frames_dir("raw", "session_x"),
            settings.RAW_DATA_DIR / "session_x" / settings.FRAME_DIR_NAME,
        )
        self.assertEqual(
            get_frames_dir("processed", "session_x"),
            settings.PROCESSED_IMAGE_DIR / "session_x",
        )

    def test_web_job_commands_use_expected_tools(self) -> None:
        sync_command = build_sync_command()
        preprocess_command = build_preprocess_command()
        feature_command = build_extract_feature_command(batch_size=16, num_workers=2)
        vitl_feature_command = build_extract_feature_command(
            batch_size=8,
            num_workers=1,
            encoder_preset="vitl_384",
        )

        self.assertIn("tools.sync_drive_data", sync_command)
        self.assertIn("--check-zips", sync_command)
        self.assertNotIn("--preprocess", sync_command)
        self.assertIn("tools.preprocess_data", preprocess_command)
        self.assertIn("tools.extract_vjepa_features", feature_command)
        self.assertIn("--encoder-preset", feature_command)
        self.assertIn("vitb_384", feature_command)
        self.assertIn("--batch-size", feature_command)
        self.assertIn("16", feature_command)
        self.assertIn("--dtype", feature_command)
        self.assertIn("fp16", feature_command)
        self.assertIn("--num-workers", feature_command)
        self.assertIn("2", feature_command)
        self.assertIn("vitl_384", vitl_feature_command)
        self.assertIn("8", vitl_feature_command)

    def test_build_job_command_rejects_bad_feature_args(self) -> None:
        with self.assertRaises(ValueError):
            build_job_command("extract_features", {"batch_size": 0})

        with self.assertRaises(ValueError):
            build_job_command("extract_features", {"num_workers": -1})

        with self.assertRaises(ValueError):
            build_job_command("unknown")

        with self.assertRaises(ValueError):
            build_job_command("extract_features", {"feature_preset": "bad_preset"})

        self.assertIn("tools.preprocess_data", build_job_command("preprocess"))

    def test_feature_preset_options_include_official_vjepa21_configs(self) -> None:
        presets = {preset["name"]: preset for preset in vjepa21_feature_preset_options()}

        self.assertEqual(presets["vitb_384"]["encoder_name"], "vit_base_384")
        self.assertEqual(presets["vitb_384"]["checkpoint_key"], "ema_encoder")
        self.assertTrue(presets["vitb_384"]["default_output_dir"].endswith("_fp16"))
        self.assertEqual(presets["vitl_384"]["encoder_name"], "vit_large_384")
        self.assertEqual(presets["vitg_384"]["checkpoint_key"], "target_encoder")
        self.assertEqual(presets["vitG_384"]["encoder_name"], "vit_gigantic_384")

    def test_progress_payload_helpers(self) -> None:
        initial = initial_progress("sync")
        completed = completed_progress("completed")
        failed = completed_progress("failed")

        self.assertTrue(initial["indeterminate"])
        self.assertEqual(initial["percent"], 0.0)
        self.assertEqual(completed["percent"], 100.0)
        self.assertEqual(failed["label"], "Failed")

    def test_parse_progress_line_accepts_only_marker_json(self) -> None:
        payload = parse_progress_line(PROGRESS_PREFIX + '{"percent": 120, "label": "Extracting"}')

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["percent"], 100.0)
        self.assertEqual(payload["label"], "Extracting")
        self.assertIsNone(parse_progress_line('{"percent": 50}'))
        self.assertIsNone(parse_progress_line(PROGRESS_PREFIX + "not json"))


if __name__ == "__main__":
    unittest.main()
