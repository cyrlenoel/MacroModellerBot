"""Success checks on the calibrated sterilised term-premium IRF."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import LAMBDA_Q, S_IL, D_YEARS, default_calibration  # noqa: E402
from src.linear_model import build_model  # noqa: E402
from src.solve_pf import eigenvalue_summary, factor_qz, solve_sterilised_qz  # noqa: E402
from src.success import evaluate_scenario_a  # noqa: E402


def _unit(T: int) -> np.ndarray:
    e = np.zeros(T)
    e[0] = 1.0
    return e


def _scaled_a(model, T: int):
    path = solve_sterilised_qz(model, {"e_tp": _unit(T)}, H=model.cal.H_peg, T=T)
    scale = model.cal.target_rl_qp / path.series(model, "rl")[0]
    path.y *= scale
    path.e_ster *= scale
    return path


class SuccessTests(unittest.TestCase):
    def test_hold_fixed_calibration(self):
        cal = default_calibration()
        self.assertEqual(cal.lambda_q, LAMBDA_Q)
        self.assertEqual(cal.s_IL, S_IL)
        self.assertEqual(cal.D, D_YEARS)
        self.assertGreater(cal.s_IL, 0.0)

    def test_blanchard_kahn(self):
        model = build_model()
        qz = factor_qz(model)
        roots = eigenvalue_summary(model)
        self.assertEqual(qz.ns, model.n)
        self.assertEqual(model.n, 36)
        self.assertEqual(roots["n_stable"], 36)
        self.assertEqual(roots["n_unstable_finite"], 6)

    def test_scenario_a_checks_and_invariance(self):
        model = build_model()
        path = _scaled_a(model, T=60)
        checks = evaluate_scenario_a(model, path, horizon=40)
        failed = [c for c in checks if not c.ok]
        self.assertFalse(failed, msg="; ".join(c.detail for c in failed))
        other = _scaled_a(model, T=100)
        gap = np.max(np.abs(path.y[:40] - other.y[:40]))
        self.assertEqual(gap, 0.0)


if __name__ == "__main__":
    unittest.main()
