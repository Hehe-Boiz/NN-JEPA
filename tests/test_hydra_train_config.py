from __future__ import annotations

from pathlib import Path
import unittest

from tools.train_rc_jepa_ac_features_hydra import build_train_args


class HydraTrainConfigTests(unittest.TestCase):
    def test_tiny_hydra_config_maps_to_train_args(self) -> None:
        args = build_train_args(
            {
                "output_dir": "checkpoints/tiny",
                "data": {
                    "features_dir": "data/processed/features/vjepa2_1_vitb_384_ema_fp16",
                    "manifest_dir": "data/processed/manifests",
                    "state_columns": ["yaw_rate_t", "accel_x_t"],
                    "action_columns": ["steering_cmd_t", "throttle_cmd_t"],
                    "raw_frames_per_sample": 8,
                    "sequence_stride": 1,
                    "auto_steps": 2,
                },
                "model": {
                    "type": "official_lite",
                    "size": "tiny",
                    "predictor_dim": None,
                    "predictor_depth": None,
                    "predictor_heads": None,
                    "dropout": 0.0,
                },
                "train": {
                    "epochs": 3,
                    "batch_size": 4,
                    "eval_batch_size": 2,
                    "num_workers": 0,
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                    "grad_clip": 1.0,
                    "warmup_epochs": 1,
                    "warmup_start_factor": 0.1,
                    "min_lr_ratio": 0.1,
                    "early_stopping_patience": 2,
                    "resume_from": None,
                    "seed": 123,
                    "device": "cpu",
                    "no_progress": True,
                    "skip_test": False,
                },
                "wandb": {
                    "disabled": True,
                    "project": "nn-jepa-rc",
                    "entity": None,
                    "run_name": None,
                    "run_id": None,
                    "continue_run": False,
                    "resume": "allow",
                    "mode": "disabled",
                    "tags": ["tiny", "test"],
                    "log_every": 10,
                    "watch_log": "none",
                    "watch_freq": 50,
                    "grad_stats_every": 0,
                    "param_stats_every": 0,
                },
            }
        )

        self.assertEqual(args.predictor_type, "official_lite")
        self.assertEqual(args.model_size, "tiny")
        self.assertIsNone(args.predictor_dim)
        self.assertEqual(args.output_dir, Path("checkpoints/tiny"))
        self.assertTrue(args._output_dir_was_provided)
        self.assertEqual(args.state_columns, ["yaw_rate_t", "accel_x_t"])
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.skip_test)
        self.assertTrue(args.no_wandb)
        self.assertFalse(args.wandb_continue_run)
        self.assertEqual(args.wandb_tags, ["tiny", "test"])

    def test_hydra_config_requires_expected_sections(self) -> None:
        with self.assertRaises(ValueError):
            build_train_args({"data": {}, "model": {}, "train": {}})


if __name__ == "__main__":
    unittest.main()
