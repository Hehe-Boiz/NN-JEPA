from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.plot_rc_jepa_planning import infer_action_columns, main, read_jsonl


class PlanningPlotTests(unittest.TestCase):
    def test_infer_action_columns_from_planning_record(self) -> None:
        records = [
            {
                "planned_first_steering_cmd_t": 0.1,
                "planned_first_throttle_cmd_t": 0.2,
            }
        ]

        self.assertEqual(
            infer_action_columns(records),
            ("steering_cmd_t", "throttle_cmd_t"),
        )

    def test_plot_script_writes_svg_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            jsonl_path = root / "planning_test.jsonl"
            records = [
                make_planning_record(index=0, planned_l1=0.3, gt_l1=0.2),
                make_planning_record(index=1, planned_l1=0.25, gt_l1=0.21),
            ]
            jsonl_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output_dir = root / "plots"

            old_argv = __import__("sys").argv
            try:
                __import__("sys").argv = [
                    "plot_rc_jepa_planning",
                    "--planning-jsonl",
                    str(jsonl_path),
                    "--output-dir",
                    str(output_dir),
                    "--sequence-records",
                    "1",
                ]
                main()
            finally:
                __import__("sys").argv = old_argv

            self.assertTrue((output_dir / "latent_l1_comparison.svg").exists())
            self.assertTrue((output_dir / "first_action_planned_vs_groundtruth.svg").exists())
            self.assertTrue((output_dir / "first_action_abs_error.svg").exists())
            self.assertTrue((output_dir / "action_sequence_record_000000.svg").exists())
            self.assertTrue((output_dir / "plots_summary.md").exists())
            self.assertEqual(len(read_jsonl(jsonl_path)), 2)


def make_planning_record(index: int, planned_l1: float, gt_l1: float) -> dict[str, object]:
    return {
        "record_index": index,
        "planned_final_l1": planned_l1,
        "groundtruth_final_l1": gt_l1,
        "zero_action_final_l1": 0.4,
        "planned_first_steering_cmd_t": 0.1 + index * 0.01,
        "groundtruth_first_steering_cmd_t": 0.0,
        "abs_error_first_steering_cmd_t": 0.1 + index * 0.01,
        "planned_first_throttle_cmd_t": 0.2,
        "groundtruth_first_throttle_cmd_t": 0.1,
        "abs_error_first_throttle_cmd_t": 0.1,
        "planned_action_sequence": [[0.1, 0.2], [0.2, 0.1]],
        "groundtruth_action_sequence": [[0.0, 0.1], [0.1, 0.2]],
    }


if __name__ == "__main__":
    unittest.main()
