"""Numeric feature normalization helpers for state/action vectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from . import settings


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float


class FeatureNormalizer:
    """Normalize scalar feature dictionaries using train-split statistics."""

    def __init__(
        self,
        stats: dict[str, FeatureStats],
        clip_value: float = settings.NUMERIC_NORMALIZE_CLIP,
    ) -> None:
        self.stats = dict(stats)
        self.clip_value = clip_value

    def normalize_value(self, column: str, value: float) -> float:
        stats = self.stats.get(column)
        if stats is None:
            return float(value)
        normalized = (float(value) - stats.mean) / max(stats.std, settings.NUMERIC_NORMALIZE_EPS)
        if self.clip_value > 0:
            normalized = max(-self.clip_value, min(self.clip_value, normalized))
        return normalized

    def normalize_row(self, row: dict[str, float], columns: Sequence[str]) -> list[float]:
        return [self.normalize_value(column, row[column]) for column in columns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_value": self.clip_value,
            "stats": {
                column: {"mean": stats.mean, "std": stats.std}
                for column, stats in self.stats.items()
            },
        }


def build_feature_normalizer(
    samples: Sequence[dict[str, Any]],
    columns: Sequence[str],
    source_key: str,
) -> FeatureNormalizer:
    stats: dict[str, FeatureStats] = {}
    for column in columns:
        values = [
            float(sample[source_key][column])
            for sample in samples
            if isinstance(sample.get(source_key), dict) and column in sample[source_key]
        ]
        if not values:
            stats[column] = FeatureStats(mean=0.0, std=1.0)
            continue
        mean = sum(values) / len(values)
        std = compute_std(values, mean)
        stats[column] = FeatureStats(mean=float(mean), std=float(max(std, settings.NUMERIC_NORMALIZE_EPS)))
    return FeatureNormalizer(stats)


def compute_std(values: Sequence[float], mean: float) -> float:
    if len(values) <= 1:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def normalizer_to_dict(normalizer: FeatureNormalizer | None) -> dict[str, Any] | None:
    if normalizer is None:
        return None
    return normalizer.to_dict()
