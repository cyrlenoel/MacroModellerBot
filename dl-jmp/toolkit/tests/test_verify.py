"""Hand-computed neighbourhood for verify_against_linear."""

from __future__ import annotations

import numpy as np

from dleq.verify import verify_against_linear


def test_near_steady_state_subset_uses_relative_radius():
    states = np.array([[1.0, 1.0], [1.02, 1.0], [1.5, 1.0]])
    nn = np.array([1.0, 1.01, 2.0])
    linear = np.array([1.0, 1.0, 1.0])
    report = verify_against_linear(
        states,
        nn,
        linear,
        steady_state=np.array([1.0, 1.0]),
        radius=0.03,
        names=("consumption",),
    )
    assert report["n"] == 3
    assert report["n_near"] == 2
    assert report["max_rel"] == 1.0
    assert abs(report["near_max_rel"] - 0.01) < 1e-12
    assert abs(report["near_mean_abs"] - 0.005) < 1e-12
    assert report["variables"][0]["name"] == "consumption"
