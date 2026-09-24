"""The slide-figure script runs and writes the Beamer set, without a (c) curve."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_slides import (  # noqa: E402
    CALIBRATED_TAG,
    LABEL_A,
    LABEL_B,
    SLIDE_NAMES,
    write_slides,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SlideScriptTests(unittest.TestCase):
    def test_agreed_labels(self):
        self.assertEqual(LABEL_A, "(a) sterilised term premium")
        self.assertEqual(LABEL_B, "(b) path news")
        self.assertEqual(CALIBRATED_TAG, "Calibrated MVP, not estimated")
        self.assertNotIn("sovereign", SLIDE_NAMES)
        self.assertEqual(
            SLIDE_NAMES,
            (
                "a_macro",
                "a_consumption_mortgage",
                "a_vs_b",
                "ib_decomposition",
                "peg_robustness",
                "bank_rate_sterilisation",
            ),
        )

    def test_refuses_to_write_over_repo_figures(self):
        with self.assertRaises(ValueError):
            write_slides(ROOT / "output")

    def test_slide_script_runs(self):
        existing = sorted((ROOT / "output").glob("irf_*.png"))
        self.assertGreaterEqual(len(existing), 4)
        before = {path.name: _sha256(path) for path in existing}

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "src" / "plot_slides.py"), "--dest", str(dest)],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("P^h trough", proc.stdout)

            for name in SLIDE_NAMES:
                pdf = dest / f"{name}.pdf"
                png = dest / f"{name}.png"
                self.assertTrue(pdf.is_file() and pdf.stat().st_size > 2000, msg=name)
                self.assertTrue(png.is_file() and png.stat().st_size > 2000, msg=name)
                self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
                self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
                box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", pdf.read_bytes())
                self.assertIsNotNone(box, msg=name)
                nums = [float(part) for part in box.group(1).split()]
                width, height = nums[2] - nums[0], nums[3] - nums[1]
                self.assertAlmostEqual(width / height, 16 / 9, places=2)

            names = {path.name for path in dest.iterdir()}
            self.assertIn("README.md", names)
            self.assertFalse(any("sovereign" in name or name.startswith("c_") for name in names))

            readme = (dest / "README.md").read_text(encoding="utf-8")

        self.assertIn(LABEL_A, readme)
        self.assertIn(LABEL_B, readme)
        self.assertIn("`(c) sovereign` — not implemented", readme)
        self.assertIn("No (c) curve", readme)
        self.assertIn("+100 bp, anticipated Bank Rate peg H = 12", readme)
        self.assertIn("H = 12 applies to (a) only", readme)
        self.assertIn("lambda = 0.09", readme)
        self.assertIn(CALIBRATED_TAG, readme)

        house = re.search(r"House-price trough: ([0-9.+\-]+)% at q(\d+) \(about ([0-9.+\-]+)% at q(\d+)\)", readme)
        self.assertIsNotNone(house)
        self.assertEqual(house.group(2), "8")
        self.assertEqual(house.group(4), "8")
        self.assertAlmostEqual(float(house.group(1)), -2.597, places=2)
        self.assertEqual(house.group(3), "-2.6")

        output = re.search(r"Output trough: ([0-9.+\-]+)% at q(\d+)", readme)
        self.assertIsNotNone(output)
        self.assertEqual(output.group(2), "4")
        self.assertLess(float(output.group(1)), 0.0)

        ib = re.search(r"Debt interest IB peak: ([0-9.+\-]+) pp of GDP at q(\d+)", readme)
        self.assertIsNotNone(ib)
        self.assertGreater(float(ib.group(1)), 0.0)
        self.assertEqual(ib.group(2), "24")

        cb = re.search(r"C\^b trough: ([0-9.+\-]+)% at q(\d+)", readme)
        cs = re.search(r"C\^s trough: ([0-9.+\-]+)% at q(\d+)", readme)
        self.assertIsNotNone(cb)
        self.assertIsNotNone(cs)
        self.assertLess(float(cb.group(1)), float(cs.group(1)))

        after = {path.name: _sha256(path) for path in existing}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
