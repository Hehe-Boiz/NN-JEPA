"""Small terminal progress bar for training scripts."""

from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True, width: int = 28) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.enabled = enabled
        self.width = width
        self.start_time = time.time()
        self.last_message = ""

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if not self.enabled:
            return

        step = min(max(int(step), 0), self.total)
        ratio = step / self.total
        filled = int(self.width * ratio)
        bar = "=" * filled + "." * (self.width - filled)
        elapsed = time.time() - self.start_time
        metrics_text = format_metrics(metrics or {})
        message = f"\r{self.label} [{bar}] {step}/{self.total} {elapsed:6.1f}s {metrics_text}"
        padding = " " * max(0, len(self.last_message) - len(message))
        sys.stderr.write(message + padding)
        sys.stderr.flush()
        self.last_message = message

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return ""
    return " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
