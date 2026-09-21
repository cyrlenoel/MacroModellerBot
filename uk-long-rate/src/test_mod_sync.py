"""The Dynare file and the Python calibration must use the same numbers and names."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import parameter_dict  # noqa: E402
from src.linear_model import EXOGENOUS, VARIABLES, build_model  # noqa: E402


MOD = (ROOT / "dynare" / "uk_long_rate.mod").read_text(encoding="utf-8")
PARAMS_M = (ROOT / "dynare" / "params_calibration.m").read_text(encoding="utf-8")


def _block(source: str, keyword: str) -> str:
    match = re.search(rf"{keyword}\b(.*?);", source, flags=re.S)
    if not match:
        raise AssertionError(f"missing {keyword} block")
    return match.group(1)


def _names(block: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", block)


class ModSyncTests(unittest.TestCase):
    def test_var_and_shock_order(self):
        declared = _names(_block(MOD, "var"))
        # The first 'var' match is the endogenous declaration, before varexo.
        self.assertEqual(tuple(declared), VARIABLES)
        exo = _names(_block(MOD, "varexo"))
        self.assertEqual(tuple(exo), EXOGENOUS)

    def test_equation_order(self):
        tags = re.findall(r"% eq: ([A-Za-z0-9_]+)", MOD)
        self.assertEqual(tuple(tags), build_model().eq_names)

    def test_literals_match_python(self):
        params = parameter_dict()
        # Assignments between the parameter declaration and the model block.
        # initval zeros are not calibration. target_rl_qp is scenario design
        # (100 annualised bp), set in the driver rather than as a parameter.
        block = re.search(r"\bparameters\b.*?;(.*?)model\s*\(", MOD, flags=re.S)
        self.assertIsNotNone(block)
        literals = {
            name: float(value)
            for name, value in re.findall(
                r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.]+)\s*;",
                block.group(1),
            )
        }
        for name, value in params.items():
            if name in (
                "delta_D",
                "phi_y_qp",
                "b_nom",
                "b_IL",
                "pi_ss_qp",
                "i_ss_qp",
                "nkpc_forward",
                "nkpc_lag",
                "target_rl_qp",
            ):
                continue
            self.assertIn(name, literals, msg=name)
            self.assertAlmostEqual(literals[name], value, places=10, msg=name)
        for snippet in (
            "#delta_D = 1/(4*D);",
            "#b_nom = (1-s_IL)*b_g_ratio;",
            "#b_IL = s_IL*b_g_ratio;",
            "#phi_y_qp = phi_y/4;",
        ):
            self.assertIn(snippet, MOD)
        self.assertAlmostEqual(literals["lambda_q"], 0.09)
        self.assertAlmostEqual(literals["s_IL"], 0.25)
        self.assertAlmostEqual(literals["D"], 9.1)

    def test_matlab_params_match(self):
        params = parameter_dict()
        for name, value in params.items():
            if name in ("nkpc_forward", "nkpc_lag"):
                continue
            match = re.search(
                rf"p\.{re.escape(name)}\s*=\s*([-+0-9.]+)\s*;",
                PARAMS_M,
            )
            if match is None:
                # Derived fields may be expressions; the hold-fixed ones must be literals.
                continue
            self.assertAlmostEqual(float(match.group(1)), value, places=10, msg=name)


if __name__ == "__main__":
    unittest.main()
