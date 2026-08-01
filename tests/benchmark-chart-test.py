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


if __name__ == "__main__":
    unittest.main()
