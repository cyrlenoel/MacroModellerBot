"""Compare a global nonlinear policy with a first-order perturbation policy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dleq.metrics import policy_errors


def verify_against_linear(
    states,
    nn_controls,
    linear_controls,
    *,
    steady_state=None,
    radius: float | None = None,
    distance: str = "relative",
    names: Sequence[str] | None = None,
    rel_eps: float = 1e-12,
) -> dict:
    """Compare neural-network controls with a log-linear (first-order) policy.

    The comparison is pointwise on a common set of states. Absolute and relative
    errors are reported for the whole sample. If ``steady_state`` and ``radius``
    are given, the same statistics are repeated on the neighbourhood

    * ``distance="relative"``: :math:`|s_i - s_i^*| \\le r \\max(|s_i^*|, \\varepsilon)`
      in every state dimension;
    * ``distance="absolute"``: :math:`|s_i - s_i^*| \\le r` in every dimension.

    Relative control error uses :math:`|\\hat c - c^{\\mathrm{lin}}| / \\max(|c^{\\mathrm{lin}}|, \\varepsilon)`.
    Overall ``max_rel`` is the largest per-control max, and overall ``mean_rel``
    is the average of the per-control means. Per-control rows are in ``variables``.

    This is the local check the heterogeneous-agent project uses: near the
    steady state the network should agree with perturbation; far away the
    first-order policy is not the benchmark.
    """
    if distance not in {"relative", "absolute"}:
        raise ValueError("distance must be 'relative' or 'absolute'")
    state_array = np.asarray(states, dtype=np.float64)
    nn = np.asarray(nn_controls, dtype=np.float64)
    linear = np.asarray(linear_controls, dtype=np.float64)
    if state_array.ndim == 1:
        state_array = state_array[:, None]
    if nn.ndim == 1:
        nn = nn[:, None]
    if linear.ndim == 1:
        linear = linear[:, None]
    if nn.shape != linear.shape:
        raise ValueError(f"control shapes differ: {nn.shape} vs {linear.shape}")
    if nn.shape[0] != state_array.shape[0]:
        raise ValueError("states and controls must have the same number of rows")
    n_controls = nn.shape[1]
    if names is None:
        control_names = [f"control_{i}" for i in range(n_controls)]
    else:
        if len(names) != n_controls:
            raise ValueError("names must have one entry per control")
        control_names = [str(name) for name in names]

    near_mask = np.ones(state_array.shape[0], dtype=bool)
    if radius is not None:
        if steady_state is None:
            raise ValueError("radius requires steady_state")
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        ss = np.asarray(steady_state, dtype=np.float64).reshape(-1)
        if ss.shape[0] != state_array.shape[1]:
            raise ValueError("steady_state has the wrong dimension")
        gap = np.abs(state_array - ss.reshape(1, -1))
        if distance == "relative":
            scale = np.maximum(np.abs(ss), rel_eps)
            near_mask = np.all(gap <= radius * scale.reshape(1, -1), axis=1)
        else:
            near_mask = np.all(gap <= radius, axis=1)

    rows = []
    for index, name in enumerate(control_names):
        full = policy_errors(nn[:, index], linear[:, index], eps=rel_eps)
        row = {"name": name, **full}
        if np.any(near_mask):
            near = policy_errors(nn[near_mask, index], linear[near_mask, index], eps=rel_eps)
            row.update(
                {
                    "near_max_abs": near["max_abs"],
                    "near_mean_abs": near["mean_abs"],
                    "near_max_rel": near["max_rel"],
                    "near_mean_rel": near["mean_rel"],
                }
            )
        else:
            row.update(
                {
                    "near_max_abs": None,
                    "near_mean_abs": None,
                    "near_max_rel": None,
                    "near_mean_rel": None,
                }
            )
        rows.append(row)

    report = {
        "n": int(state_array.shape[0]),
        "n_near": int(np.sum(near_mask)),
        "radius": None if radius is None else float(radius),
        "distance": distance,
        "max_abs": float(max(row["max_abs"] for row in rows)),
        "mean_abs": float(np.mean([row["mean_abs"] for row in rows])),
        "max_rel": float(max(row["max_rel"] for row in rows)),
        "mean_rel": float(np.mean([row["mean_rel"] for row in rows])),
        "variables": rows,
    }
    if rows and rows[0]["near_max_abs"] is not None:
        report["near_max_abs"] = float(max(row["near_max_abs"] for row in rows))
        report["near_mean_abs"] = float(np.mean([row["near_mean_abs"] for row in rows]))
        report["near_max_rel"] = float(max(row["near_max_rel"] for row in rows))
        report["near_mean_rel"] = float(np.mean([row["near_mean_rel"] for row in rows]))
    else:
        report["near_max_abs"] = None
        report["near_mean_abs"] = None
        report["near_max_rel"] = None
        report["near_mean_rel"] = None
    return report


def format_verification(report: dict) -> str:
    """Plain-text table of a :func:`verify_against_linear` report."""
    lines = [
        f"n={report['n']}  n_near={report['n_near']}  "
        f"radius={report['radius']}  distance={report['distance']}",
        f"{'variable':<16} {'mean_abs':>12} {'max_abs':>12} {'mean_rel':>12} {'max_rel':>12} "
        f"{'near_mean_rel':>14} {'near_max_rel':>14}",
    ]
    for row in report["variables"]:
        near_mean = _fmt(row["near_mean_rel"])
        near_max = _fmt(row["near_max_rel"])
        lines.append(
            f"{row['name']:<16} {row['mean_abs']:12.6e} {row['max_abs']:12.6e} "
            f"{row['mean_rel']:12.6e} {row['max_rel']:12.6e} {near_mean:>14} {near_max:>14}"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6e}"
