"""Policy and Euler-error summaries used by the verification examples."""

from __future__ import annotations

import numpy as np


def policy_errors(predicted, truth, eps: float = 1e-12) -> dict[str, float]:
    """Max and mean absolute and relative errors of a policy against a benchmark.

    Relative error at a point is :math:`|\\hat y - y| / \\max(|y|, \\varepsilon)`.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(truth, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    absolute = np.abs(pred - ref)
    relative = absolute / np.maximum(np.abs(ref), eps)
    return {
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
        "max_rel": float(np.max(relative)),
        "mean_rel": float(np.mean(relative)),
    }


def log10_abs_stats(residuals) -> dict[str, float]:
    """Summaries of :math:`\\log_{10}|e|` for Euler residuals ``e``.

    A more negative log10 error is a smaller residual. The absolute floor is
    only there so a residual that underflows to zero still has a finite log.
    """
    absolute = np.abs(np.asarray(residuals, dtype=np.float64).ravel())
    if absolute.size == 0:
        raise ValueError("residuals are empty")
    logged = np.log10(np.maximum(absolute, 1e-300))
    return {
        "mean_abs": float(np.mean(absolute)),
        "max_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
        "mean_log10_abs": float(np.mean(logged)),
        "max_log10_abs": float(np.max(logged)),
        "median_log10_abs": float(np.median(logged)),
    }
