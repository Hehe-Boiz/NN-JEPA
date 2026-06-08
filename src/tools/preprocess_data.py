"""CLI entrypoint for preprocessing raw driving data."""

from __future__ import annotations

import json

from data.preprocess import preprocess_all_sessions


PROGRESS_PREFIX = "__JOB_PROGRESS__ "


def print_progress(current: int, total: int, label: str) -> None:
    percent = 100.0 if total <= 0 else 100.0 * current / total
    print(
        PROGRESS_PREFIX
        + json.dumps(
            {
                "percent": max(0.0, min(percent, 100.0)),
                "label": label,
                "current": current,
                "total": total,
                "indeterminate": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    try:
        print_progress(0, 1, "Starting preprocessing")
        summary = preprocess_all_sessions(progress_callback=print_progress)
        print_progress(1, 1, "Preprocessing complete")
    except (FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
