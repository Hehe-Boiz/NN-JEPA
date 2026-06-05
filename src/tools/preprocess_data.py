"""CLI entrypoint for preprocessing raw driving data."""

from __future__ import annotations

import json

from data.preprocess import preprocess_all_sessions


def main() -> None:
    try:
        summary = preprocess_all_sessions()
    except (FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
