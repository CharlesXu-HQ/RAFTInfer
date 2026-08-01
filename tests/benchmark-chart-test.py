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
    record("pp128", 10, 5, 11, 10, 2, 1.1),
    record("pp512", 20, 10, 21, 20, 2, 1.05),
    record("tg128_pp512", 30, 15, 31, 30, 2, 1.033),
]


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
        for expected in (
            "Prefill", "Generation", "RAFTInfer", "llama.cpp",
            "10.00 tok/s", "20.00 tok/s", "30.00 tok/s",
            "11.00 tok/s", "21.00 tok/s", "31.00 tok/s",
            "5.00 tok/s", "10.00 tok/s",
            "2.000x", "1.100x", "1.050x", "1.033x",
        ):
            self.assertIn(expected, text)

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
