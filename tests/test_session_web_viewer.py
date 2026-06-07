from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from data import settings
from tools.session_web_viewer import build_session_payload, collect_sessions, get_frames_dir, list_frame_paths


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


if __name__ == "__main__":
    unittest.main()
