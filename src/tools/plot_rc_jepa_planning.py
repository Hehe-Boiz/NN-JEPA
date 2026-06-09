"""Plot NN-JEPA offline planner outputs as dependency-free SVG charts."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PLANNING_JSONL = Path("checkpoints/rc_jepa_ac_vitb_features_newdata_tiny/planning/planning_test.jsonl")
CHART_WIDTH = 960
CHART_HEIGHT = 420
MARGIN_LEFT = 72
MARGIN_RIGHT = 190
MARGIN_TOP = 38
MARGIN_BOTTOM = 58
COLORS = (
    "#0f766e",
    "#dc2626",
    "#2563eb",
    "#ca8a04",
    "#7c3aed",
    "#475569",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NN-JEPA planning JSONL outputs as SVG charts.")
    parser.add_argument("--planning-jsonl", type=Path, default=DEFAULT_PLANNING_JSONL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sequence-records", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Planning JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Planning JSONL is empty: {path}")
    return records


def infer_action_columns(records: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    first = records[0]
    prefix = "planned_first_"
    for key in sorted(first):
        if key.startswith(prefix):
            columns.append(key[len(prefix) :])
    return tuple(columns)


def get_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_series(records: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[tuple[str, list[float | None]]]:
    return [(key, [get_float(record, key) for record in records]) for key in keys]


def safe_domain(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = max(abs(lo) * 0.1, 1e-3)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def make_svg_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[float | None]]],
) -> str:
    values = [float(value) for _, rows in series for value in rows if value is not None]
    y_min, y_max = safe_domain(values)
    sample_count = max((len(rows) for _, rows in series), default=0)
    plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    def x_pos(index: int) -> float:
        if sample_count <= 1:
            return MARGIN_LEFT + plot_width / 2.0
        return MARGIN_LEFT + (index / (sample_count - 1)) * plot_width

    def y_pos(value: float) -> float:
        return MARGIN_TOP + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_height

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_WIDTH}" height="{CHART_HEIGHT}" viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{MARGIN_LEFT}" y="24" font-family="sans-serif" font-size="18" font-weight="700" fill="#0f172a">{escape(title)}</text>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP + plot_height}" x2="{MARGIN_LEFT + plot_width}" y2="{MARGIN_TOP + plot_height}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{MARGIN_TOP + plot_height}" stroke="#334155" stroke-width="1"/>',
    ]

    for tick in range(5):
        ratio = tick / 4.0
        value = y_min + ratio * (y_max - y_min)
        y = y_pos(value)
        lines.append(f'<line x1="{MARGIN_LEFT - 5}" y1="{y:.2f}" x2="{MARGIN_LEFT + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{MARGIN_LEFT - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11" fill="#475569">{value:.4g}</text>')

    for label, rows in series:
        color = COLORS[len(lines) % len(COLORS)]
        points = []
        for index, value in enumerate(rows):
            if value is None:
                continue
            points.append(f"{x_pos(index):.2f},{y_pos(float(value)):.2f}")
        if points:
            color = COLORS[series.index((label, rows)) % len(COLORS)]
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="{" ".join(points)}"/>'
            )

    legend_x = MARGIN_LEFT + plot_width + 24
    legend_y = MARGIN_TOP + 8
    for index, (label, _) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        y = legend_y + index * 24
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{legend_x + 22}" y="{y + 2}" font-family="sans-serif" font-size="12" fill="#0f172a">{escape(label)}</text>')

    lines.append(f'<text x="{MARGIN_LEFT + plot_width / 2:.2f}" y="{CHART_HEIGHT - 14}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">{escape(x_label)}</text>')
    lines.append(f'<text x="18" y="{MARGIN_TOP + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {MARGIN_TOP + plot_height / 2:.2f})" font-family="sans-serif" font-size="12" fill="#475569">{escape(y_label)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def action_sequence_series(
    record: dict[str, Any],
    action_columns: Sequence[str],
) -> list[tuple[str, list[float | None]]]:
    planned = record.get("planned_action_sequence") or []
    groundtruth = record.get("groundtruth_action_sequence") or []
    series: list[tuple[str, list[float | None]]] = []
    for action_index, column in enumerate(action_columns):
        series.append(
            (
                f"planned_{column}",
                [float(row[action_index]) if len(row) > action_index else None for row in planned],
            )
        )
        series.append(
            (
                f"groundtruth_{column}",
                [float(row[action_index]) if len(row) > action_index else None for row in groundtruth],
            )
        )
    return series


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_summary(path: Path, output_files: Sequence[Path], records: Sequence[dict[str, Any]]) -> None:
    rows = [
        "# Planning Plots",
        "",
        f"- records: `{len(records)}`",
        "",
        "## Files",
        "",
    ]
    for output_file in output_files:
        rows.append(f"- `{output_file.name}`")
    rows.append("")
    write_text(path, "\n".join(rows))


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.planning_jsonl)
    output_dir = args.output_dir or args.planning_jsonl.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    action_columns = infer_action_columns(records)

    output_files: list[Path] = []

    latent_series = collect_series(
        records,
        ("planned_final_l1", "groundtruth_final_l1", "zero_action_final_l1"),
    )
    path = output_dir / "latent_l1_comparison.svg"
    write_text(
        path,
        make_svg_line_chart(
            title="Latent L1 tới goal",
            x_label="record_index",
            y_label="L1",
            series=latent_series,
        ),
    )
    output_files.append(path)

    if action_columns:
        first_action_series = []
        for column in action_columns:
            first_action_series.extend(
                collect_series(
                    records,
                    (f"planned_first_{column}", f"groundtruth_first_{column}"),
                )
            )
        path = output_dir / "first_action_planned_vs_groundtruth.svg"
        write_text(
            path,
            make_svg_line_chart(
                title="Action đầu tiên: planned vs ground-truth",
                x_label="record_index",
                y_label="raw action",
                series=first_action_series,
            ),
        )
        output_files.append(path)

        abs_error_series = collect_series(
            records,
            tuple(f"abs_error_first_{column}" for column in action_columns),
        )
        path = output_dir / "first_action_abs_error.svg"
        write_text(
            path,
            make_svg_line_chart(
                title="Sai số tuyệt đối của action đầu tiên",
                x_label="record_index",
                y_label="absolute error",
                series=abs_error_series,
            ),
        )
        output_files.append(path)

        for record in records[: max(args.sequence_records, 0)]:
            record_index = int(record.get("record_index", len(output_files)))
            path = output_dir / f"action_sequence_record_{record_index:06d}.svg"
            write_text(
                path,
                make_svg_line_chart(
                    title=f"Action sequence record {record_index:06d}",
                    x_label="planning step",
                    y_label="raw action",
                    series=action_sequence_series(record, action_columns),
                ),
            )
            output_files.append(path)

    summary_path = output_dir / "plots_summary.md"
    write_summary(summary_path, output_files, records)

    print(
        json.dumps(
            {
                "planning_jsonl": str(args.planning_jsonl),
                "output_dir": str(output_dir),
                "records": len(records),
                "plots": [str(path) for path in output_files],
                "summary": str(summary_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
