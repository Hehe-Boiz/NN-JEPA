from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import data.settings as settings

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from data.sequence_dataset import RCJepaACSequenceDataset
    from models.rc_jepa_ac import SimpleACPredictor, compute_world_model_losses


class RCJepaACTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_sequence_dataset_builds_windows_inside_one_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "train.jsonl"
            samples = []
            for session_id in ("session_a", "session_b"):
                for frame_index in range(5):
                    image_path = root / f"{session_id}_{frame_index}.jpg"
                    Image.new("RGB", (8, 8), color=(frame_index * 20, 0, 0)).save(image_path)
                    samples.append(make_manifest_sample(session_id, frame_index, image_path))

            manifest_path.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )

            dataset = RCJepaACSequenceDataset(
                split="train",
                manifest_path=manifest_path,
                raw_frames_per_sample=4,
                sequence_stride=1,
                augment=False,
            )

            self.assertEqual(len(dataset), 4)
            item = dataset[0]
            self.assertEqual(tuple(item["images"].shape), (3, 4, 8, 8))
            self.assertEqual(tuple(item["states"].shape), (4, len(settings.AC_STATE_COLUMNS)))
            self.assertEqual(tuple(item["actions"].shape), (3, len(settings.AC_ACTION_COLUMNS)))
            self.assertEqual(item["session_id"], "session_a")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_ac_predictor_keeps_latent_shape(self) -> None:
        predictor = SimpleACPredictor(
            latent_dim=8,
            state_dim=5,
            action_dim=2,
            tokens_per_frame=4,
            max_frames=4,
            predictor_dim=16,
            depth=2,
            num_heads=4,
        )
        latents = torch.randn(2, 3 * 4, 8)
        actions = torch.randn(2, 3, 2)
        states = torch.randn(2, 3, 5)

        prediction = predictor(latents, actions, states)

        self.assertEqual(tuple(prediction.shape), (2, 3 * 4, 8))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_world_model_losses_are_scalar(self) -> None:
        predictor = SimpleACPredictor(
            latent_dim=8,
            state_dim=5,
            action_dim=2,
            tokens_per_frame=4,
            max_frames=4,
            predictor_dim=16,
            depth=1,
            num_heads=4,
        )
        latents = torch.randn(2, 4 * 4, 8)
        states = torch.randn(2, 4, 5)
        actions = torch.randn(2, 3, 2)

        outputs = compute_world_model_losses(
            predictor=predictor,
            latents=latents,
            states=states,
            actions=actions,
            tokens_per_frame=4,
            auto_steps=2,
        )

        self.assertEqual(outputs["loss"].ndim, 0)
        self.assertEqual(outputs["teacher_forcing_loss"].ndim, 0)
        self.assertEqual(outputs["rollout_loss"].ndim, 0)


def make_manifest_sample(session_id: str, frame_index: int, image_path: Path) -> dict[str, object]:
    return {
        "sample_id": f"{session_id}_{frame_index:06d}",
        "session_id": session_id,
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index),
        "frame_path": str(image_path),
        "state": {
            "v_t": 0.0,
            "yaw_rate_t": 0.1,
            "accel_x_t": 0.2,
            "accel_y_t": 0.3,
            "steering_last_t": 0.4,
            "throttle_last_t": 0.5,
        },
        "action": {
            "steering_cmd_t": 0.1,
            "throttle_cmd_t": 0.2,
        },
    }


if __name__ == "__main__":
    unittest.main()
