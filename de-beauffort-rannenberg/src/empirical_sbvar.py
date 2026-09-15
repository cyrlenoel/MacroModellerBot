"""S-BVAR comparison series digitised from WP 493 Figure 3.

These are *not* the authors' numerical IRFs. They are hand-read from the
published figure so the replication plot can overlay the qualitative
S-BVAR pattern (black circles in Figure 3). Marked [PROVISIONAL] in NOTES.md.

Horizon is 20 quarters. G is the model shock path (1% of GDP on impact).
"""

from __future__ import annotations

import numpy as np

# Government spending, percentage points of GDP. Paper: shock follows the
# empirical S-BVAR G path; "does not follow a hump-shaped path" (§4.3).
G_PP = np.array(
    [
        1.00,
        0.96,
        0.91,
        0.87,
        0.82,
        0.78,
        0.73,
        0.69,
        0.64,
        0.60,
        0.55,
        0.51,
        0.46,
        0.42,
        0.38,
        0.34,
        0.30,
        0.26,
        0.22,
        0.19,
    ]
)

# Remaining Figure 3 S-BVAR point estimates (same units as the paper figure).
SBVAR = {
    "C": np.array(
        [
            0.42,
            0.58,
            0.72,
            0.84,
            0.93,
            0.98,
            1.02,
            1.03,
            1.01,
            0.98,
            0.94,
            0.89,
            0.84,
            0.78,
            0.72,
            0.66,
            0.60,
            0.54,
            0.49,
            0.45,
        ]
    ),
    "C_T": np.array(
        [
            0.85,
            1.00,
            1.12,
            1.22,
            1.30,
            1.35,
            1.37,
            1.36,
            1.33,
            1.28,
            1.22,
            1.16,
            1.10,
            1.03,
            0.97,
            0.91,
            0.85,
            0.80,
            0.75,
            0.70,
        ]
    ),
    "C_NT": np.array(
        [
            0.20,
            0.30,
            0.38,
            0.44,
            0.48,
            0.51,
            0.52,
            0.51,
            0.48,
            0.44,
            0.40,
            0.35,
            0.29,
            0.23,
            0.17,
            0.11,
            0.05,
            -0.01,
            -0.06,
            -0.10,
        ]
    ),
    "Y": np.array(
        [
            1.20,
            1.28,
            1.35,
            1.38,
            1.36,
            1.32,
            1.26,
            1.18,
            1.10,
            1.02,
            0.94,
            0.86,
            0.79,
            0.72,
            0.66,
            0.60,
            0.54,
            0.49,
            0.44,
            0.40,
        ]
    ),
    "Y_T": np.array(
        [
            0.95,
            1.15,
            1.38,
            1.42,
            1.32,
            1.20,
            1.10,
            1.00,
            0.92,
            0.84,
            0.77,
            0.70,
            0.64,
            0.58,
            0.53,
            0.49,
            0.45,
            0.42,
            0.39,
            0.36,
        ]
    ),
    "Y_NT": np.array(
        [
            1.22,
            1.32,
            1.42,
            1.48,
            1.47,
            1.43,
            1.37,
            1.30,
            1.23,
            1.16,
            1.10,
            1.04,
            0.98,
            0.93,
            0.88,
            0.84,
            0.80,
            0.77,
            0.74,
            0.72,
        ]
    ),
    "p_ratio": np.array(
        [
            0.00,
            -0.08,
            -0.14,
            -0.19,
            -0.23,
            -0.27,
            -0.30,
            -0.32,
            -0.33,
            -0.34,
            -0.34,
            -0.33,
            -0.32,
            -0.31,
            -0.30,
            -0.29,
            -0.28,
            -0.27,
            -0.26,
            -0.25,
        ]
    ),
    "NX": np.array(
        [
            -0.11,
            -0.16,
            -0.19,
            -0.20,
            -0.17,
            -0.13,
            -0.09,
            -0.05,
            -0.02,
            0.01,
            0.03,
            0.04,
            0.045,
            0.05,
            0.05,
            0.048,
            0.045,
            0.042,
            0.04,
            0.038,
        ]
    ),
    "G": G_PP.copy(),
}


def g_shock_level(T: int, Y: float = 1.0) -> np.ndarray:
    """Level shock dG with impact 1% of output, then the Figure-3 G shape.

    After quarter 20, continue at the last observed decay rate, floored at 0.
    """
    dG = np.zeros(T)
    scale = 0.01 * Y  # 1% of GDP
    n = min(T, len(G_PP))
    dG[:n] = (G_PP[:n] / G_PP[0]) * scale
    if T > n:
        rho = G_PP[-1] / G_PP[-2]
        rho = float(np.clip(rho, 0.0, 0.999))
        for t in range(n, T):
            dG[t] = dG[t - 1] * rho
    return dG
