#!/usr/bin/env python3
"""Behavior tests for the checked-in benchmark SVG renderer."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "tools" / "render_benchmark_chart.py"


def record(arm, prefill_raftinfer, prefill_llama, generation_raftinfer,
           generation_llama, prefill_ratio, generation_ratio):
    return {
        "schema_version": 2,
        "arm": arm,
        "parity": {"passed": True},
        "performance_floor_passed": True,
        "raftinfer": {
            "prefill": {"tokens_per_second": prefill_raftinfer},
            "generation": {"tokens_per_second": generation_raftinfer},
        },
        "llama_cpp": {
            "prefill": {"tokens_per_second": prefill_llama},
            "generation": {"tokens_per_second": generation_llama},
        },
        "throughput_ratio": {
            "prefill": prefill_ratio,
            "generation": generation_ratio,
        },
    }


VALID_RECORDS = [
    record("pp128", 101, 51, 111, 61, 1.980, 1.820),
    record("pp512", 202, 102, 212, 112, 1.980, 1.890),
    record("tg128_pp512", 303, 153, 313, 163, 1.980, 1.920),
]

EXPECTED_GROUPS = {
    ("Prefill", "pp128"): ("101.00 tok/s", "51.00 tok/s", "RAFTInfer 1.980x"),
    ("Prefill", "pp512"): ("202.00 tok/s", "102.00 tok/s", "RAFTInfer 1.980x"),
    ("Generation", "pp128"): ("111.00 tok/s", "61.00 tok/s", "RAFTInfer 1.820x"),
    ("Generation", "pp512"): ("212.00 tok/s", "112.00 tok/s", "RAFTInfer 1.890x"),
    ("Generation", "tg128 pp512"): ("313.00 tok/s", "163.00 tok/s", "RAFTInfer 1.920x"),
}


class BenchmarkChartTest(unittest.TestCase):
    """A broken renderer must fail these observable SVG contracts."""

    def run_renderer(self, records):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        source = directory / "fixture.jsonl"
        output = directory / "chart.svg"
        source.write_text("\n".join(json.dumps(item) for item in records) + "\n")
        completed = subprocess.run(
            [sys.executable, str(RENDERER), str(source), str(output)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return completed, output

    def test_renders_hand_derived_values_in_two_panels(self):
        """Catches omitted metrics, wrong panels, and changed chart dimensions."""
        completed, output = self.run_renderer(VALID_RECORDS)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        root = ET.parse(output).getroot()
        self.assertEqual(root.attrib["width"], "1200")
        self.assertEqual(root.attrib["height"], "620")
        text = output.read_text()
        for expected in ("Prefill", "Generation", "RAFTInfer", "llama.cpp"):
            self.assertIn(expected, text)

    def test_scopes_each_series_label_to_its_panel_and_arm_group(self):
        """Catches labels swapped between chart groups despite matching global text."""
        completed, output = self.run_renderer(VALID_RECORDS)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        root = ET.parse(output).getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        texts = root.findall("svg:text", namespace)
        categories = [
            item for item in texts
            if item.attrib.get("y") == "534" and (item.text or "") in {"pp128", "pp512", "tg128 pp512"}
        ]
        self.assertEqual(len(categories), 5)
        for (panel, arm), expected in EXPECTED_GROUPS.items():
            category = next(
                item for item in categories
                if item.text == arm and ((panel == "Prefill") == (float(item.attrib["x"]) < 600))
            )
            center = float(category.attrib["x"])
            panel_categories = sorted(
                float(item.attrib["x"]) for item in categories
                if (float(item.attrib["x"]) < 600) == (panel == "Prefill")
            )
            position = panel_categories.index(center)
            left = 40 if panel == "Prefill" else 660
            right = 540 if panel == "Prefill" else 1160
            if position:
                left = (panel_categories[position - 1] + center) / 2
            if position + 1 < len(panel_categories):
                right = (center + panel_categories[position + 1]) / 2
            group_text = [
                item.text for item in texts
                if left < float(item.attrib.get("x", "-1")) < right
            ]
            for label in expected:
                self.assertEqual(group_text.count(label), 1, (panel, arm, label, group_text))

        value_labels = [item.text for item in texts if (item.text or "").endswith(" tok/s")]
        ratio_labels = [item.text for item in texts if (item.text or "").startswith("RAFTInfer ")]
        self.assertEqual(len(value_labels), 10)
        self.assertEqual(len(ratio_labels), 5)

    def test_rejects_invalid_schema(self):
        """Catches accepting an unsupported record schema."""
        records = [dict(item, schema_version=1) for item in VALID_RECORDS]
        completed, _ = self.run_renderer(records)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("schema_version", completed.stderr)

    def test_rejects_missing_arm(self):
        """Catches charts silently omitting a required benchmark arm."""
        completed, _ = self.run_renderer(VALID_RECORDS[:2])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("arms", completed.stderr)

    def test_rejects_failed_parity(self):
        """Catches publishing measurements that did not pass exact parity."""
        records = [dict(item) for item in VALID_RECORDS]
        records[0]["parity"] = {"passed": False}
        completed, _ = self.run_renderer(records)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("parity", completed.stderr)

    def test_rejects_failed_performance_floor(self):
        """Catches publishing an arm that failed its stated performance floor."""
        records = [dict(item) for item in VALID_RECORDS]
        records[0]["performance_floor_passed"] = False
        completed, _ = self.run_renderer(records)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("performance-floor", completed.stderr)

    def test_places_ratio_above_bars_and_value_labels_apart(self):
        """Catches ratios inside a bar and adjacent value labels colliding."""
        completed, output = self.run_renderer(VALID_RECORDS)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        root = ET.parse(output).getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        bars = root.findall("svg:rect", namespace)
        texts = root.findall("svg:text", namespace)
        for ratio in (item for item in texts if (item.text or "").startswith("RAFTInfer ")):
            ratio_x = float(ratio.attrib["x"])
            ratio_y = float(ratio.attrib["y"])
            group_bars = [
                item for item in bars
                if "x" in item.attrib
                and float(item.attrib.get("y", "0")) >= 185
                and abs((float(item.attrib["x"]) + 23) - ratio_x) < 60
                and item.attrib.get("fill") in {"#76B900", "#64748B"}
            ]
            self.assertEqual(len(group_bars), 2)
            highest_top = min(float(item.attrib["y"]) for item in group_bars)
            self.assertLess(ratio_y, highest_top - 12)

        labels = [item for item in texts if (item.text or "").endswith(" tok/s")]
        for left, right in zip(labels[::2], labels[1::2]):
            left_x = float(left.attrib["x"])
            right_x = float(right.attrib["x"])
            self.assertNotEqual(left.attrib.get("text-anchor"), right.attrib.get("text-anchor"))
            self.assertLess(left_x, right_x)


if __name__ == "__main__":
    unittest.main()
