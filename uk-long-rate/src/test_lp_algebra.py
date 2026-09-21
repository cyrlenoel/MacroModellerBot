"""Algebra checks for the LP / proxy-VAR helpers.

The series in this file are SYNTHETIC_NOT_UK_DATA. They are not gilt
surprises, and they must not be written out as UK results.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lp_proxyvar import (  # noqa: E402
    jorda_lp,
    orthogonalise_gilt_surprise,
    proxy_var_impact,
    var1_residuals,
)


class LPAlgebraTests(unittest.TestCase):
    def test_orthogonalisation_removes_bank_rate_and_ois(self):
        # SYNTHETIC_NOT_UK_DATA
        rng = np.random.default_rng(1)
        n = 400
        br = rng.normal(size=n)
        ois = rng.normal(size=n)
        noise = rng.normal(size=n)
        gilt = 2.0 * br + 3.0 * ois + noise
        resid = orthogonalise_gilt_surprise(gilt, br, ois)
        X = np.column_stack([np.ones(n), br, ois])
        self.assertLess(np.max(np.abs(X.T @ resid)), 1e-8)

    def test_horizon_zero_lp_recovers_the_impact(self):
        # SYNTHETIC_NOT_UK_DATA: y_t = 0.5 y_{t-1} + 0.8 z_t + noise
        rng = np.random.default_rng(2)
        T = 8000
        z = rng.normal(size=T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + 0.8 * z[t] + 0.01 * rng.normal()
        lag = np.zeros(T)
        lag[1:] = y[:-1]
        beta, _se = jorda_lp(y[1:], z[1:], lag[1:, None], [0])
        self.assertAlmostEqual(float(beta[0]), 0.8, delta=0.05)

    def test_proxy_impact_scales_to_one(self):
        # SYNTHETIC_NOT_UK_DATA
        rng = np.random.default_rng(3)
        T = 1500
        eps = rng.normal(size=T)
        other = rng.normal(size=T)
        y1 = np.zeros(T)
        y2 = np.zeros(T)
        for t in range(1, T):
            y1[t] = 0.4 * y1[t - 1] + eps[t]
            y2[t] = 0.2 * y2[t - 1] + 0.5 * eps[t] + other[t]
        resid, _coef = var1_residuals(np.column_stack([y1, y2]))
        impact = proxy_var_impact(resid, eps, normalise=0)
        self.assertAlmostEqual(float(impact[0]), 1.0, places=6)
        self.assertGreater(float(impact[1]), 0.2)


if __name__ == "__main__":
    unittest.main()
