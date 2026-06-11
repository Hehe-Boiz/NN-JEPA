"""CEM planner for NN-JEPA feature-cache world models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from data.normalization import FeatureNormalizer
from models.rc_jepa_ac import build_rollout_state_context, normalize_rollout_feedback

DEFAULT_CONTROLLABLE_ACTION_COLUMNS = ("steering_cmd_t", "throttle_cmd_t")


@dataclass(frozen=True)
class CEMPlanResult:
    """Result returned by the feature-cache CEM planner."""

    first_action: torch.Tensor
    action_sequence: torch.Tensor
    score: float
    iterations: int


def controllable_action_indices(
    action_columns: Sequence[str],
    controllable_action_columns: Sequence[str] = DEFAULT_CONTROLLABLE_ACTION_COLUMNS,
) -> tuple[int, ...]:
    """Return action indices that CEM is allowed to optimize."""
    action_columns = tuple(action_columns)
    return tuple(index for index, column in enumerate(action_columns) if column in controllable_action_columns)


def fixed_action_indices(
    action_columns: Sequence[str],
    controllable_action_columns: Sequence[str] = DEFAULT_CONTROLLABLE_ACTION_COLUMNS,
) -> tuple[int, ...]:
    """Return conditioning action indices that must be copied, not optimized."""
    controllable = set(controllable_action_columns)
    return tuple(index for index, column in enumerate(action_columns) if column not in controllable)


def make_zero_control_actions(
    actions: torch.Tensor,
    action_columns: Sequence[str],
    controllable_action_columns: Sequence[str] = DEFAULT_CONTROLLABLE_ACTION_COLUMNS,
) -> torch.Tensor:
    """Zero only controllable commands while preserving domain/context columns."""
    zero_actions = actions.clone()
    for index in controllable_action_indices(action_columns, controllable_action_columns):
        zero_actions[..., index] = 0.0
    return zero_actions


def normalizer_stats_tensors(
    normalizer: FeatureNormalizer | None,
    columns: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return mean/std tensors for the requested columns."""
    means: list[float] = []
    stds: list[float] = []
    for column in columns:
        stats = None if normalizer is None else normalizer.stats.get(column)
        means.append(0.0 if stats is None else float(stats.mean))
        stds.append(1.0 if stats is None else float(max(stats.std, 1e-6)))
    clip_value = 0.0 if normalizer is None else float(normalizer.clip_value)
    return (
        torch.tensor(means, dtype=torch.float32, device=device),
        torch.tensor(stds, dtype=torch.float32, device=device),
        clip_value,
    )


def normalize_action_tensor(
    raw_actions: torch.Tensor,
    action_columns: Sequence[str],
    action_normalizer: FeatureNormalizer | None,
) -> torch.Tensor:
    """Convert raw action tensor to the normalized units used by training."""
    means, stds, clip_value = normalizer_stats_tensors(
        action_normalizer,
        action_columns,
        raw_actions.device,
    )
    normalized = (raw_actions.float() - means.view(*([1] * (raw_actions.ndim - 1)), -1)) / stds.view(
        *([1] * (raw_actions.ndim - 1)),
        -1,
    )
    if clip_value > 0:
        normalized = normalized.clamp(-clip_value, clip_value)
    return normalized


def denormalize_action_tensor(
    model_actions: torch.Tensor,
    action_columns: Sequence[str],
    action_normalizer: FeatureNormalizer | None,
) -> torch.Tensor:
    """Convert normalized model action tensor back to raw action units."""
    means, stds, _ = normalizer_stats_tensors(
        action_normalizer,
        action_columns,
        model_actions.device,
    )
    return model_actions.float() * stds.view(*([1] * (model_actions.ndim - 1)), -1) + means.view(
        *([1] * (model_actions.ndim - 1)),
        -1,
    )


class RCJepaACFeatureCEMPlanner:
    """Plan raw actions by rolling them through a trained NN-JEPA predictor.

    The planner samples raw action sequences, normalizes them with the same
    train-split statistics as the DataLoader, rolls out latent tokens with the
    predictor, and scores the final predicted frame against a goal latent frame.
    """

    def __init__(
        self,
        predictor: torch.nn.Module,
        tokens_per_frame: int,
        state_columns: Sequence[str],
        action_columns: Sequence[str],
        action_normalizer: FeatureNormalizer | None,
        horizon: int,
        n_samples: int = 128,
        n_elite: int = 16,
        n_iter: int = 4,
        action_low: Sequence[float] | torch.Tensor | float = -1.0,
        action_high: Sequence[float] | torch.Tensor | float = 1.0,
        init_std: float = 0.5,
        min_std: float = 0.05,
        action_penalty: float = 0.0,
        smooth_penalty: float = 0.0,
        device: torch.device | str = "cuda",
        rollout_feedback_norm: bool = False,
        controllable_action_columns: Sequence[str] = DEFAULT_CONTROLLABLE_ACTION_COLUMNS,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if n_elite < 1:
            raise ValueError("n_elite must be >= 1")
        if n_elite > n_samples:
            raise ValueError("n_elite must be <= n_samples")

        self.predictor = predictor.eval()
        self.tokens_per_frame = int(tokens_per_frame)
        self.state_columns = tuple(state_columns)
        self.action_columns = tuple(action_columns)
        self.action_dim = len(self.action_columns)
        self.controllable_action_columns = tuple(controllable_action_columns)
        self.controllable_indices = controllable_action_indices(
            self.action_columns,
            self.controllable_action_columns,
        )
        self.fixed_indices = fixed_action_indices(
            self.action_columns,
            self.controllable_action_columns,
        )
        if not self.controllable_indices:
            raise ValueError("CEM planner needs at least one controllable action column")
        self.action_normalizer = action_normalizer
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.n_elite = int(n_elite)
        self.n_iter = int(n_iter)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.action_penalty = float(action_penalty)
        self.smooth_penalty = float(smooth_penalty)
        self.rollout_feedback_norm = bool(rollout_feedback_norm)
        self.device = torch.device(device)
        self.action_low = self._action_bound_tensor(action_low, "action_low")
        self.action_high = self._action_bound_tensor(action_high, "action_high")
        if torch.any(self.action_low >= self.action_high):
            raise ValueError("Every action_low value must be smaller than action_high")
        self._controllable_index_tensor = torch.tensor(self.controllable_indices, dtype=torch.long, device=self.device)
        self._fixed_index_tensor = torch.tensor(self.fixed_indices, dtype=torch.long, device=self.device)
        self.control_action_low = self.action_low.index_select(0, self._controllable_index_tensor)
        self.control_action_high = self.action_high.index_select(0, self._controllable_index_tensor)

    def _action_bound_tensor(
        self,
        value: Sequence[float] | torch.Tensor | float,
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.repeat(self.action_dim)
        if tensor.numel() != self.action_dim:
            raise ValueError(f"{name} must have {self.action_dim} values, got {tensor.numel()}")
        return tensor.view(self.action_dim)

    def _reference_action_sequence(self, reference_actions: torch.Tensor | None) -> torch.Tensor:
        if not self.fixed_indices:
            return torch.zeros(self.horizon, self.action_dim, dtype=torch.float32, device=self.device)
        if reference_actions is None:
            fixed_names = [self.action_columns[index] for index in self.fixed_indices]
            raise ValueError(
                "CEM planner found non-controllable action columns "
                f"{fixed_names}; pass reference_actions so they can be held fixed."
            )
        reference = reference_actions.to(self.device, dtype=torch.float32)
        if reference.ndim == 1:
            reference = reference.view(1, self.action_dim).expand(self.horizon, -1).contiguous()
        elif reference.ndim == 2:
            if reference.size(0) < self.horizon:
                raise ValueError(
                    f"reference_actions must have at least horizon={self.horizon} steps, got {reference.size(0)}"
                )
            reference = reference[: self.horizon]
        elif reference.ndim == 3:
            if reference.size(0) != 1:
                raise ValueError("reference_actions with 3 dims must have batch size 1")
            if reference.size(1) < self.horizon:
                raise ValueError(
                    f"reference_actions must have at least horizon={self.horizon} steps, got {reference.size(1)}"
                )
            reference = reference[0, : self.horizon]
        else:
            raise ValueError(f"Expected reference_actions [A], [H,A], or [1,H,A], got {tuple(reference.shape)}")
        if reference.size(-1) != self.action_dim:
            raise ValueError(f"reference_actions action_dim must be {self.action_dim}, got {reference.size(-1)}")
        return reference.contiguous()

    def _compose_full_action_samples(
        self,
        control_samples: torch.Tensor,
        reference_actions: torch.Tensor,
    ) -> torch.Tensor:
        samples = reference_actions.unsqueeze(0).expand(control_samples.size(0), -1, -1).clone()
        samples.index_copy_(-1, self._controllable_index_tensor, control_samples)
        return samples

    @torch.no_grad()
    def rollout(
        self,
        context_tokens: torch.Tensor,
        initial_state: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out future latent frames for raw action candidates.

        Args:
            context_tokens: one latent frame, shape ``[K,D]`` or ``[B,K,D]``.
            initial_state: normalized current state, shape ``[S]`` or ``[B,S]``.
            raw_actions: raw action candidates, shape ``[B,H,A]``.

        Returns:
            Predicted latent frames with shape ``[B,H,K,D]``.
        """
        raw_actions = raw_actions.to(self.device, dtype=torch.float32)
        if raw_actions.ndim != 3:
            raise ValueError(f"Expected raw_actions [B,H,A], got {tuple(raw_actions.shape)}")
        batch_size, horizon, action_dim = raw_actions.shape
        if horizon != self.horizon:
            raise ValueError(f"Expected horizon={self.horizon}, got {horizon}")
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")

        context_tokens = context_tokens.to(self.device, dtype=torch.float32)
        if context_tokens.ndim == 2:
            context_tokens = context_tokens.unsqueeze(0)
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected context_tokens [K,D] or [B,K,D], got {tuple(context_tokens.shape)}")
        if context_tokens.size(1) != self.tokens_per_frame:
            raise ValueError(
                f"Expected {self.tokens_per_frame} tokens/frame, got {context_tokens.size(1)}"
            )
        if context_tokens.size(0) == 1 and batch_size > 1:
            context_tokens = context_tokens.expand(batch_size, -1, -1).contiguous()
        if context_tokens.size(0) != batch_size:
            raise ValueError(f"context batch {context_tokens.size(0)} does not match actions batch {batch_size}")

        initial_state = initial_state.to(self.device, dtype=torch.float32)
        if initial_state.ndim == 1:
            initial_state = initial_state.unsqueeze(0)
        if initial_state.size(0) == 1 and batch_size > 1:
            initial_state = initial_state.expand(batch_size, -1).contiguous()
        if initial_state.size(0) != batch_size:
            raise ValueError(f"state batch {initial_state.size(0)} does not match actions batch {batch_size}")

        model_actions = normalize_action_tensor(
            raw_actions,
            action_columns=self.action_columns,
            action_normalizer=self.action_normalizer,
        )
        # Planning has no measured future state trajectory, so it intentionally
        # uses the fallback state approximation. Dynamic IMU columns remain
        # stale/approximated unless a separate state update model is added.
        rollout_states = build_rollout_state_context(
            initial_state=initial_state.unsqueeze(1),
            actions=model_actions,
            rollout_steps=self.horizon,
            state_columns=self.state_columns,
            action_columns=self.action_columns,
        )

        rollout_tokens = context_tokens
        predictions: list[torch.Tensor] = []
        for step in range(self.horizon):
            pred_tokens = self.predictor(
                latent_tokens=rollout_tokens,
                actions=model_actions[:, : step + 1],
                states=rollout_states[:, : step + 1],
                tokens_per_frame=self.tokens_per_frame,
            )
            next_tokens = pred_tokens[:, -self.tokens_per_frame :]
            predictions.append(next_tokens)
            feedback_tokens = normalize_rollout_feedback(next_tokens, enabled=self.rollout_feedback_norm)
            rollout_tokens = torch.cat([rollout_tokens, feedback_tokens], dim=1)
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def score(
        self,
        context_tokens: torch.Tensor,
        initial_state: torch.Tensor,
        goal_tokens: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return planner score for each raw action sequence."""
        goal_tokens = goal_tokens.to(self.device, dtype=torch.float32)
        if goal_tokens.ndim == 2:
            goal_tokens = goal_tokens.unsqueeze(0)

        predictions = self.rollout(context_tokens, initial_state, raw_actions)
        final_prediction = predictions[:, -1]
        if goal_tokens.size(0) == 1 and final_prediction.size(0) > 1:
            goal_tokens = goal_tokens.expand(final_prediction.size(0), -1, -1)
        if goal_tokens.shape != final_prediction.shape:
            raise ValueError(f"goal shape {tuple(goal_tokens.shape)} does not match prediction {tuple(final_prediction.shape)}")

        score = F.l1_loss(final_prediction, goal_tokens, reduction="none").mean(dim=(1, 2))
        raw_actions = raw_actions.to(self.device, dtype=torch.float32)
        control_actions = raw_actions.index_select(-1, self._controllable_index_tensor)
        if self.action_penalty > 0:
            score = score + self.action_penalty * control_actions.square().mean(dim=(1, 2))
        if self.smooth_penalty > 0 and control_actions.size(1) > 1:
            score = score + self.smooth_penalty * (control_actions[:, 1:] - control_actions[:, :-1]).square().mean(dim=(1, 2))
        return score

    @torch.no_grad()
    def plan(
        self,
        context_tokens: torch.Tensor,
        initial_state: torch.Tensor,
        goal_tokens: torch.Tensor,
        reference_actions: torch.Tensor | None = None,
    ) -> CEMPlanResult:
        """Run CEM and return the best raw action sequence."""
        reference_sequence = self._reference_action_sequence(reference_actions)
        control_dim = len(self.controllable_indices)
        mu = torch.zeros(self.horizon, control_dim, device=self.device)
        sigma = torch.full_like(mu, self.init_std)
        best_score: torch.Tensor | None = None
        best_sequence: torch.Tensor | None = None

        for _ in range(self.n_iter):
            eps = torch.randn(self.n_samples, self.horizon, control_dim, device=self.device)
            control_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            control_samples = torch.maximum(torch.minimum(control_samples, self.control_action_high), self.control_action_low)
            samples = self._compose_full_action_samples(control_samples, reference_sequence)
            scores = self.score(
                context_tokens=context_tokens,
                initial_state=initial_state,
                goal_tokens=goal_tokens,
                raw_actions=samples,
            )
            elite_indices = torch.topk(scores, self.n_elite, largest=False).indices
            elites = control_samples[elite_indices]
            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0, unbiased=False).clamp_min(self.min_std)

            current_score = scores[elite_indices[0]]
            if best_score is None or current_score < best_score:
                best_score = current_score
                best_sequence = samples[elite_indices[0]]

        if best_score is None or best_sequence is None:
            raise RuntimeError("CEM planner failed to produce an action sequence")

        best_sequence = best_sequence.detach().cpu()
        return CEMPlanResult(
            first_action=best_sequence[0],
            action_sequence=best_sequence,
            score=float(best_score.detach().cpu()),
            iterations=self.n_iter,
        )
