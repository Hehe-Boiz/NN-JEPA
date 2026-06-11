from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

import data.settings as settings

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    import models.rc_jepa_ac as rc_jepa_ac
    from data.normalization import FeatureNormalizer, FeatureStats
    from data.feature_sequence_dataset import RCJepaACFeatureSequenceDataset
    from data.sequence_dataset import RCJepaACSequenceDataset, build_sequence_windows
    from models.rc_jepa_ac import (
        ROLLOUT_STATE_MODE_LEGACY_REPEAT,
        ROLLOUT_STATE_MODE_MEASURED_TRAIN,
        SimpleACPredictor,
        VJepaStyleACPredictor,
        apply_predictor_size_preset,
        build_ac_predictor,
        build_action_block_causal_attention_mask,
        build_rollout_state_context,
        compute_world_model_losses,
    )
    from tools.extract_vjepa_features import resolve_feature_extraction_args
    from tools.rc_jepa_ac_cem_planner import (
        RCJepaACFeatureCEMPlanner,
        denormalize_action_tensor,
        normalize_action_tensor,
    )
    from tools.rc_jepa_ac_feature_runtime import config_from_checkpoint
    from tools.train_rc_jepa_ac import (
        build_lr_scheduler,
        compute_lr_scale,
        compute_warmup_steps,
        should_apply_early_stopping,
        sync_lr_scheduler,
    )
    from tools.wandb_utils import persist_wandb_run_id, read_saved_wandb_run_id, resolve_wandb_run_id
    from tools.wandb_utils import init_wandb


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
    def test_feature_sequence_dataset_reads_cached_frame_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "train.jsonl"
            features_dir = root / "features"
            sessions_dir = features_dir / "sessions"
            sessions_dir.mkdir(parents=True)

            samples = []
            for session_id in ("session_a", "session_b"):
                feature_array = np.zeros((5, 4, 8), dtype=np.float16)
                frames = []
                for frame_index in range(5):
                    image_path = root / f"{session_id}_{frame_index}.jpg"
                    Image.new("RGB", (8, 8), color=(frame_index * 20, 0, 0)).save(image_path)
                    samples.append(make_manifest_sample(session_id, frame_index, image_path))
                    feature_array[frame_index] = float(frame_index)
                    frames.append(
                        {
                            "row": frame_index,
                            "sample_id": f"{session_id}_{frame_index:06d}",
                            "session_id": session_id,
                            "frame_index": frame_index,
                            "timestamp_sec": float(frame_index) * 0.1,
                        }
                    )

                np.save(sessions_dir / f"{session_id}.npy", feature_array)
                (sessions_dir / f"{session_id}.json").write_text(
                    json.dumps({"session_id": session_id, "frames": frames}),
                    encoding="utf-8",
                )

            (features_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "tokens_per_frame": 4,
                        "embed_dim": 8,
                        "dtype": "fp16",
                        "image_path_key": "source_frame_path",
                        "image_path_fallback": True,
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )

            dataset = RCJepaACFeatureSequenceDataset(
                split="train",
                features_dir=features_dir,
                manifest_path=manifest_path,
                raw_frames_per_sample=4,
                sequence_stride=1,
            )

            self.assertEqual(len(dataset), 4)
            item = dataset[0]
            self.assertEqual(tuple(item["latents"].shape), (16, 8))
            self.assertEqual(item["latents"].dtype, torch.float32)
            self.assertEqual(tuple(item["states"].shape), (4, len(settings.AC_STATE_COLUMNS)))
            self.assertEqual(tuple(item["actions"].shape), (3, len(settings.AC_ACTION_COLUMNS)))
            self.assertEqual(item["session_id"], "session_a")
            self.assertAlmostEqual(float(item["latents"][0, 0]), 0.0)
            self.assertAlmostEqual(float(item["latents"][-1, 0]), 3.0)

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
    def test_official_lite_predictor_keeps_latent_shape(self) -> None:
        predictor = VJepaStyleACPredictor(
            latent_dim=8,
            state_dim=5,
            action_dim=2,
            tokens_per_frame=4,
            max_frames=4,
            predictor_dim=64,
            depth=1,
            num_heads=4,
        )
        latents = torch.randn(2, 3 * 4, 8)
        actions = torch.randn(2, 3, 2)
        states = torch.randn(2, 3, 5)

        prediction = predictor(latents, actions, states)

        self.assertEqual(tuple(prediction.shape), (2, 3 * 4, 8))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_official_lite_mask_allows_only_current_and_past_frames(self) -> None:
        mask = build_action_block_causal_attention_mask(
            num_frames=3,
            grid_height=2,
            grid_width=2,
            add_tokens=2,
        )
        tokens_per_step = 6

        self.assertTrue(bool(mask[0, 0]))
        self.assertFalse(bool(mask[0, tokens_per_step]))
        self.assertTrue(bool(mask[tokens_per_step, 0]))
        self.assertTrue(bool(mask[tokens_per_step, tokens_per_step]))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_predictor_factory_selects_simple_or_official_lite(self) -> None:
        simple = build_ac_predictor(
            predictor_type="simple",
            latent_dim=8,
            state_dim=5,
            action_dim=2,
            tokens_per_frame=4,
            max_frames=4,
            predictor_dim=16,
            depth=1,
            num_heads=4,
        )
        official_lite = build_ac_predictor(
            predictor_type="official_lite",
            latent_dim=8,
            state_dim=5,
            action_dim=2,
            tokens_per_frame=4,
            max_frames=4,
            predictor_dim=64,
            depth=1,
            num_heads=4,
        )

        self.assertIsInstance(simple, SimpleACPredictor)
        self.assertIsInstance(official_lite, VJepaStyleACPredictor)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_predictor_size_preset_fills_missing_values(self) -> None:
        args = SimpleNamespace(
            model_size="tiny",
            predictor_dim=None,
            predictor_depth=None,
            predictor_heads=None,
        )

        apply_predictor_size_preset(args)

        self.assertEqual(args.predictor_dim, 128)
        self.assertEqual(args.predictor_depth, 2)
        self.assertEqual(args.predictor_heads, 4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_predictor_size_preset_keeps_manual_override(self) -> None:
        args = SimpleNamespace(
            model_size="tiny",
            predictor_dim=256,
            predictor_depth=None,
            predictor_heads=None,
        )

        apply_predictor_size_preset(args)

        self.assertEqual(args.predictor_dim, 256)
        self.assertEqual(args.predictor_depth, 2)
        self.assertEqual(args.predictor_heads, 4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_feature_extraction_preset_resolves_official_vitl_config(self) -> None:
        args = SimpleNamespace(
            encoder_preset="vitl_384",
            encoder=None,
            vjepa_checkpoint=None,
            checkpoint_key=None,
            output_dir=None,
            manifest_dir=settings.MANIFEST_DIR,
            dtype="fp32",
        )

        resolved = resolve_feature_extraction_args(args)

        self.assertEqual(resolved.encoder, "vit_large_384")
        self.assertEqual(resolved.checkpoint_key, "ema_encoder")
        self.assertEqual(resolved.vjepa_checkpoint, Path("checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"))
        self.assertEqual(
            resolved.output_dir,
            settings.PROCESSED_DATA_DIR / "features" / "vjepa2_1_vitl_384_ema_fp32",
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_feature_extraction_manual_encoder_requires_checkpoint(self) -> None:
        args = SimpleNamespace(
            encoder_preset="vitb_384",
            encoder="vit_large_384",
            vjepa_checkpoint=None,
            checkpoint_key=None,
            output_dir=None,
            dtype="fp32",
        )

        with self.assertRaises(ValueError):
            resolve_feature_extraction_args(args)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_sequence_windows_do_not_cross_frame_gap(self) -> None:
        samples = [
            make_manifest_sample("session_a", 0, Path("0.jpg")),
            make_manifest_sample("session_a", 1, Path("1.jpg")),
            make_manifest_sample("session_a", 2, Path("2.jpg")),
            make_manifest_sample("session_a", 10, Path("10.jpg")),
            make_manifest_sample("session_a", 11, Path("11.jpg")),
        ]
        for sample, timestamp in zip(samples, [0.0, 0.1, 0.2, 1.0, 1.1]):
            sample["timestamp_sec"] = timestamp

        windows = build_sequence_windows(
            samples,
            raw_frames_per_sample=3,
            sequence_stride=1,
            state_columns=settings.AC_STATE_COLUMNS,
            action_columns=settings.AC_ACTION_COLUMNS,
            max_frame_index_gap=1,
            max_time_gap_sec=0.25,
        )

        self.assertEqual(windows, [[0, 1, 2]])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_rollout_state_context_does_not_use_future_measured_state(self) -> None:
        initial_state = torch.tensor([[[10.0, 20.0, 30.0, 0.1, 0.2]]])
        actions = torch.tensor([[[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]])

        rollout_states = build_rollout_state_context(
            initial_state=initial_state,
            actions=actions,
            rollout_steps=3,
            state_columns=tuple(settings.AC_STATE_COLUMNS),
            action_columns=tuple(settings.AC_ACTION_COLUMNS),
        )

        self.assertEqual(tuple(rollout_states.shape), (1, 3, 5))
        self.assertAlmostEqual(float(rollout_states[0, 1, 0]), 10.0)
        self.assertAlmostEqual(float(rollout_states[0, 1, 1]), 20.0)
        self.assertAlmostEqual(float(rollout_states[0, 1, 2]), 30.0)
        self.assertAlmostEqual(float(rollout_states[0, 1, 3]), 0.3)
        self.assertAlmostEqual(float(rollout_states[0, 1, 4]), 0.4)

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

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_measured_train_rollout_uses_measured_states_without_helper(self) -> None:
        predictor = RecordingRolloutPredictor()
        latents = torch.tensor([[[0.0], [10.0], [20.0], [30.0]]])
        states = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        actions = torch.tensor([[[10.0], [11.0], [12.0]]])

        original_helper = rc_jepa_ac.build_rollout_state_context

        def fail_if_called(*args, **kwargs):
            raise AssertionError("measured_train rollout must not call build_rollout_state_context")

        rc_jepa_ac.build_rollout_state_context = fail_if_called
        try:
            outputs = compute_world_model_losses(
                predictor=predictor,
                latents=latents,
                states=states,
                actions=actions,
                tokens_per_frame=1,
                auto_steps=2,
                rollout_state_mode=ROLLOUT_STATE_MODE_MEASURED_TRAIN,
            )
        finally:
            rc_jepa_ac.build_rollout_state_context = original_helper

        self.assertEqual(outputs["loss"].ndim, 0)
        self.assertEqual(len(predictor.calls), 3)
        _, step0_state, step0_action = predictor.calls[1]
        step1_latent, step1_state, step1_action = predictor.calls[2]
        self.assertEqual(step0_state.flatten().tolist(), [0.0])
        self.assertEqual(step0_action.flatten().tolist(), [10.0])
        self.assertEqual(step1_state.flatten().tolist(), [0.0, 1.0])
        self.assertEqual(step1_action.flatten().tolist(), [10.0, 11.0])
        self.assertEqual(step1_latent.flatten().tolist(), [0.0, 102.0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_legacy_repeat_rollout_keeps_old_state_context_behavior(self) -> None:
        predictor = RecordingRolloutPredictor()
        latents = torch.zeros(1, 3, 1)
        states = torch.tensor([[[10.0, 20.0, 30.0, 0.1, 0.2], [11.0, 21.0, 31.0, 0.3, 0.4], [12.0, 22.0, 32.0, 0.5, 0.6]]])
        actions = torch.tensor([[[0.7, 0.8], [0.9, 1.0]]])

        compute_world_model_losses(
            predictor=predictor,
            latents=latents,
            states=states,
            actions=actions,
            tokens_per_frame=1,
            auto_steps=2,
            state_columns=tuple(settings.AC_STATE_COLUMNS),
            action_columns=tuple(settings.AC_ACTION_COLUMNS),
            rollout_state_mode=ROLLOUT_STATE_MODE_LEGACY_REPEAT,
        )

        _, step1_state, _ = predictor.calls[2]
        self.assertEqual(step1_state[0, 1, :3].tolist(), [10.0, 20.0, 30.0])
        self.assertTrue(torch.allclose(step1_state[0, 1, 3:], torch.tensor([0.7, 0.8])))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_world_model_losses_backward_for_rollout_modes_and_steps(self) -> None:
        for rollout_state_mode in (ROLLOUT_STATE_MODE_LEGACY_REPEAT, ROLLOUT_STATE_MODE_MEASURED_TRAIN):
            for auto_steps in (1, 2, 4):
                with self.subTest(rollout_state_mode=rollout_state_mode, auto_steps=auto_steps):
                    predictor = SimpleACPredictor(
                        latent_dim=8,
                        state_dim=5,
                        action_dim=2,
                        tokens_per_frame=4,
                        max_frames=5,
                        predictor_dim=16,
                        depth=1,
                        num_heads=4,
                    )
                    latents = torch.randn(2, 5 * 4, 8)
                    states = torch.randn(2, 5, 5)
                    actions = torch.randn(2, 4, 2)

                    outputs = compute_world_model_losses(
                        predictor=predictor,
                        latents=latents,
                        states=states,
                        actions=actions,
                        tokens_per_frame=4,
                        auto_steps=auto_steps,
                        state_columns=tuple(settings.AC_STATE_COLUMNS),
                        action_columns=tuple(settings.AC_ACTION_COLUMNS),
                        rollout_state_mode=rollout_state_mode,
                    )
                    outputs["loss"].backward()

                    first_grad = next(parameter.grad for parameter in predictor.parameters() if parameter.grad is not None)
                    self.assertTrue(torch.isfinite(first_grad).all())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_action_normalization_round_trip_for_planner(self) -> None:
        normalizer = FeatureNormalizer(
            {
                "steering_cmd_t": FeatureStats(mean=0.1, std=0.2),
                "throttle_cmd_t": FeatureStats(mean=-0.2, std=0.5),
            },
            clip_value=8.0,
        )
        raw_actions = torch.tensor([[[0.3, 0.8], [0.1, -0.2]]])

        model_actions = normalize_action_tensor(
            raw_actions,
            action_columns=("steering_cmd_t", "throttle_cmd_t"),
            action_normalizer=normalizer,
        )
        recovered = denormalize_action_tensor(
            model_actions,
            action_columns=("steering_cmd_t", "throttle_cmd_t"),
            action_normalizer=normalizer,
        )

        self.assertTrue(torch.allclose(recovered, raw_actions, atol=1e-6))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_feature_cem_planner_rollout_shape_and_action_bounds(self) -> None:
        predictor = ToyPlannerPredictor()
        planner = RCJepaACFeatureCEMPlanner(
            predictor=predictor,
            tokens_per_frame=4,
            state_columns=tuple(settings.AC_STATE_COLUMNS),
            action_columns=tuple(settings.AC_ACTION_COLUMNS),
            action_normalizer=None,
            horizon=2,
            n_samples=8,
            n_elite=2,
            n_iter=2,
            action_low=(-0.25, -0.1),
            action_high=(0.25, 0.1),
            device="cpu",
        )
        context = torch.zeros(4, 8)
        initial_state = torch.zeros(len(settings.AC_STATE_COLUMNS))
        raw_actions = torch.zeros(3, 2, len(settings.AC_ACTION_COLUMNS))
        goal = torch.ones(4, 8) * 0.1

        rollout = planner.rollout(context, initial_state, raw_actions)
        plan = planner.plan(context, initial_state, goal)

        self.assertEqual(tuple(rollout.shape), (3, 2, 4, 8))
        self.assertEqual(tuple(plan.first_action.shape), (len(settings.AC_ACTION_COLUMNS),))
        self.assertGreaterEqual(float(plan.first_action[0]), -0.250001)
        self.assertLessEqual(float(plan.first_action[0]), 0.250001)
        self.assertGreaterEqual(float(plan.first_action[1]), -0.100001)
        self.assertLessEqual(float(plan.first_action[1]), 0.100001)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_lr_schedule_uses_warmup_then_cosine_decay(self) -> None:
        warmup_steps = compute_warmup_steps(warmup_epochs=2, steps_per_epoch=3, total_train_steps=12)
        self.assertEqual(warmup_steps, 6)

        start_scale = compute_lr_scale(
            step=0,
            total_train_steps=12,
            warmup_steps=warmup_steps,
            warmup_start_factor=0.1,
            min_lr_ratio=0.1,
        )
        warmup_end_scale = compute_lr_scale(
            step=warmup_steps,
            total_train_steps=12,
            warmup_steps=warmup_steps,
            warmup_start_factor=0.1,
            min_lr_ratio=0.1,
        )
        final_scale = compute_lr_scale(
            step=12,
            total_train_steps=12,
            warmup_steps=warmup_steps,
            warmup_start_factor=0.1,
            min_lr_ratio=0.1,
        )

        self.assertAlmostEqual(start_scale, 0.1)
        self.assertAlmostEqual(warmup_end_scale, 1.0)
        self.assertAlmostEqual(final_scale, 0.1)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_pytorch_lr_scheduler_matches_scale_and_resume_step(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            total_train_steps=12,
            warmup_steps=6,
            warmup_start_factor=0.1,
            min_lr_ratio=0.1,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2.5e-5)

        sync_lr_scheduler(scheduler, global_step=6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_pytorch_lr_scheduler_resume_uses_base_lr_not_loaded_optimizer_lr(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            total_train_steps=12,
            warmup_steps=6,
            warmup_start_factor=0.1,
            min_lr_ratio=0.1,
            base_lr=1e-4,
        )

        sync_lr_scheduler(scheduler, global_step=6)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_early_stopping_starts_after_warmup(self) -> None:
        self.assertFalse(should_apply_early_stopping(epoch=1, warmup_epochs=1))
        self.assertTrue(should_apply_early_stopping(epoch=2, warmup_epochs=1))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_feature_checkpoint_config_reads_model_shape(self) -> None:
        checkpoint = {
            "args": {
                "state_columns": ["yaw_rate_t", "accel_x_t"],
                "action_columns": ["steering_cmd_t", "throttle_cmd_t"],
                "raw_frames_per_sample": 6,
                "sequence_stride": 2,
                "auto_steps": 3,
                "rollout_state_mode": "measured_train",
                "predictor_type": "official_lite",
                "predictor_dim": 32,
                "predictor_depth": 2,
                "predictor_heads": 4,
                "dropout": 0.1,
            },
            "feature_metadata": {
                "tokens_per_frame": 4,
                "embed_dim": 8,
                "dtype": "fp32",
            },
        }

        config = config_from_checkpoint(checkpoint)

        self.assertEqual(config.state_columns, ("yaw_rate_t", "accel_x_t"))
        self.assertEqual(config.action_columns, ("steering_cmd_t", "throttle_cmd_t"))
        self.assertEqual(config.raw_frames_per_sample, 6)
        self.assertEqual(config.sequence_stride, 2)
        self.assertEqual(config.auto_steps, 3)
        self.assertEqual(config.rollout_state_mode, "measured_train")
        self.assertEqual(config.predictor_type, "official_lite")
        self.assertEqual(config.predictor_dim, 32)
        self.assertEqual(config.predictor_depth, 2)
        self.assertEqual(config.predictor_heads, 4)
        self.assertEqual(config.tokens_per_frame, 4)
        self.assertEqual(config.embed_dim, 8)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_wandb_run_id_is_saved_and_reused_only_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            args = SimpleNamespace(
                output_dir=output_dir,
                resume_from=None,
                wandb_run_id=None,
                wandb_resume="allow",
            )
            run = SimpleNamespace(id="abc123")

            persist_wandb_run_id(args, run, job_type="train-rc-jepa-ac-features")

            self.assertEqual(read_saved_wandb_run_id(args), "abc123")
            self.assertIsNone(resolve_wandb_run_id(args))

            resume_args = SimpleNamespace(
                output_dir=output_dir,
                resume_from=output_dir / "last.pt",
                wandb_run_id=None,
                wandb_resume="allow",
            )
            self.assertEqual(resolve_wandb_run_id(resume_args), "abc123")

            no_resume_args = SimpleNamespace(
                output_dir=output_dir,
                resume_from=output_dir / "last.pt",
                wandb_run_id=None,
                wandb_resume="never",
            )
            self.assertIsNone(resolve_wandb_run_id(no_resume_args))

            new_wandb_run_args = SimpleNamespace(
                output_dir=output_dir,
                resume_from=output_dir / "last.pt",
                wandb_run_id=None,
                wandb_resume="allow",
                wandb_continue_run=False,
            )
            self.assertIsNone(resolve_wandb_run_id(new_wandb_run_args))

            explicit_args = SimpleNamespace(
                output_dir=output_dir,
                resume_from=None,
                wandb_run_id="manual456",
                wandb_resume="allow",
                wandb_continue_run=True,
            )
            self.assertEqual(resolve_wandb_run_id(explicit_args), "manual456")

            explicit_but_disabled_args = SimpleNamespace(
                output_dir=output_dir,
                resume_from=None,
                wandb_run_id="manual456",
                wandb_resume="allow",
                wandb_continue_run=False,
            )
            with self.assertRaises(ValueError):
                resolve_wandb_run_id(explicit_but_disabled_args)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_init_wandb_passes_saved_run_id_to_wandb_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "wandb_run_id.txt").write_text("abc123\n", encoding="utf-8")
            captured_kwargs = {}

            def fake_init(**kwargs: object) -> SimpleNamespace:
                captured_kwargs.update(kwargs)
                return SimpleNamespace(id=kwargs.get("id", "new-run"))

            fake_wandb = SimpleNamespace(init=fake_init)
            missing = object()
            old_wandb = sys.modules.get("wandb", missing)
            sys.modules["wandb"] = fake_wandb
            try:
                args = SimpleNamespace(
                    no_wandb=False,
                    wandb_mode="online",
                    wandb_project="nn-jepa-rc",
                    wandb_entity=None,
                    wandb_run_name=None,
                    wandb_run_id=None,
                    wandb_continue_run=True,
                    wandb_resume="allow",
                    wandb_tags=[],
                    output_dir=output_dir,
                    resume_from=output_dir / "last.pt",
                )

                init_wandb(args, config={"a": 1}, job_type="train-rc-jepa-ac-features")
            finally:
                if old_wandb is missing:
                    del sys.modules["wandb"]
                else:
                    sys.modules["wandb"] = old_wandb

            self.assertEqual(captured_kwargs["id"], "abc123")
            self.assertEqual(captured_kwargs["resume"], "allow")
            self.assertEqual(captured_kwargs["project"], "nn-jepa-rc")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
    def test_init_wandb_can_start_new_run_when_resuming_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "wandb_run_id.txt").write_text("abc123\n", encoding="utf-8")
            captured_kwargs = {}

            def fake_init(**kwargs: object) -> SimpleNamespace:
                captured_kwargs.update(kwargs)
                return SimpleNamespace(id="new-run")

            fake_wandb = SimpleNamespace(init=fake_init)
            missing = object()
            old_wandb = sys.modules.get("wandb", missing)
            sys.modules["wandb"] = fake_wandb
            try:
                args = SimpleNamespace(
                    no_wandb=False,
                    wandb_mode="online",
                    wandb_project="nn-jepa-rc",
                    wandb_entity=None,
                    wandb_run_name=None,
                    wandb_run_id=None,
                    wandb_continue_run=False,
                    wandb_resume="allow",
                    wandb_tags=[],
                    output_dir=output_dir,
                    resume_from=output_dir / "last.pt",
                )

                init_wandb(args, config={"a": 1}, job_type="train-rc-jepa-ac-features")
            finally:
                if old_wandb is missing:
                    del sys.modules["wandb"]
                else:
                    sys.modules["wandb"] = old_wandb

            self.assertNotIn("id", captured_kwargs)
            self.assertNotIn("resume", captured_kwargs)
            self.assertEqual(captured_kwargs["project"], "nn-jepa-rc")
            self.assertEqual(read_saved_wandb_run_id(args), "new-run")


def make_manifest_sample(session_id: str, frame_index: int, image_path: Path) -> dict[str, object]:
    return {
        "sample_id": f"{session_id}_{frame_index:06d}",
        "session_id": session_id,
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index) * 0.1,
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


class ToyPlannerPredictor(torch.nn.Module if TORCH_AVAILABLE else object):
    def forward(
        self,
        latent_tokens: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        tokens_per_frame: int | None = None,
    ) -> torch.Tensor:
        if tokens_per_frame is None:
            raise ValueError("tokens_per_frame is required")
        batch_size, total_tokens, latent_dim = latent_tokens.shape
        num_frames = total_tokens // tokens_per_frame
        latent = latent_tokens.view(batch_size, num_frames, tokens_per_frame, latent_dim)
        action_delta = actions.sum(dim=-1).view(batch_size, num_frames, 1, 1)
        return (latent + action_delta).reshape(batch_size, total_tokens, latent_dim)


class RecordingRolloutPredictor(torch.nn.Module if TORCH_AVAILABLE else object):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        latent_tokens: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        tokens_per_frame: int | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            (
                latent_tokens.detach().cpu().clone(),
                states.detach().cpu().clone(),
                actions.detach().cpu().clone(),
            )
        )
        value = 100.0 + float(len(self.calls))
        return torch.ones_like(latent_tokens) * self.scale * value


if __name__ == "__main__":
    unittest.main()
