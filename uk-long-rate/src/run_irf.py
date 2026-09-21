#!/usr/bin/env python3
"""Calibrated IRFs for the UK long-rate twin.

Scenario (a) is a term-premium innovation scaled so the model long rate
rises 100 annualised basis points on impact, with Bank Rate held on its
steady-state path by anticipated Taylor residuals.

Scenario (b) is short-rate path news, scaled so the long rate also rises
100 annualised basis points on impact. Bank Rate is not sterilised.

Writes CSVs, a check log, and figures under ``output/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.linear_model import build_model  # noqa: E402
from src.plot_irf import plot_scenario_a, plot_a_versus_b  # noqa: E402
from src.solve_pf import (  # noqa: E402
    eigenvalue_summary,
    solve_qz,
    solve_sterilised_qz,
)
from src.success import STERILISATION_DEVICE, evaluate_scenario_a, format_checks  # noqa: E402


# Quantities in percent; rates converted to annualised basis points.
RATE_NAMES = ("i", "pi", "rl", "tp", "reh", "rm", "rms", "reff", "ril", "nu")
LEVEL_NAMES = (
    "y",
    "c",
    "cs",
    "cb",
    "inv",
    "hi",
    "q",
    "k",
    "h",
    "ph",
    "rer",
    "ib",
    "nx",
    "dg",
    "nfa",
    "bm",
)


def _scale_to_long_rate(model, path):
    scale = model.cal.target_rl_qp / path.series(model, "rl")[0]
    path.y *= scale
    path.e_ster *= scale
    return scale


def _frame(model, path, horizon: int) -> dict[str, np.ndarray]:
    out = {"quarter": np.arange(1, horizon + 1, dtype=int)}
    for name in RATE_NAMES:
        out[f"{name}_bp"] = path.series(model, name)[:horizon] * 400.0
    for name in LEVEL_NAMES:
        out[name] = path.series(model, name)[:horizon]
    out["e_ster_bp"] = path.e_ster[:horizon] * 400.0
    return out


def _write_csv(path: Path, frame: dict[str, np.ndarray], header_note: str) -> None:
    keys = list(frame)
    data = np.column_stack([frame[k] for k in keys])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {header_note}\n")
        fh.write(",".join(keys) + "\n")
        for row in data:
            fh.write(",".join(f"{v:.8g}" for v in row) + "\n")


def interior_residual(model, path, tp0: float) -> float:
    """Structural residual on dates whose lead lies inside the simulated path."""
    from src.solve_pf import _shock_matrix

    T = path.T
    e = _shock_matrix(model, {"e_tp": _impulse(T, tp0)}, T)
    e[:, model.ix_exo["e_ster"]] = path.e_ster
    worst = 0.0
    for t in range(T - 1):
        y_lag = path.y[t - 1] if t else np.zeros(model.n)
        res = model.Mp @ path.y[t + 1] + model.Mc @ path.y[t] + model.Ml @ y_lag + model.Me @ e[t]
        worst = max(worst, float(np.max(np.abs(res))))
    return worst


def _impulse(T: int, value: float) -> np.ndarray:
    e = np.zeros(T)
    e[0] = value
    return e


def main() -> int:
    model = build_model()
    cal = model.cal
    T = max(cal.irf_horizon, cal.H_peg) + 20
    horizons = cal.irf_horizon
    roots = eigenvalue_summary(model)

    unit = _impulse(T, 1.0)
    path_a = solve_sterilised_qz(model, {"e_tp": unit}, H=cal.H_peg, T=T)
    scale_a = _scale_to_long_rate(model, path_a)
    path_raw = solve_qz(model, {"e_tp": _impulse(T, scale_a)}, T=T)
    path_b = solve_qz(model, {"e_news": unit}, T=T)
    scale_b = _scale_to_long_rate(model, path_b)

    checks = evaluate_scenario_a(model, path_a, horizon=horizons)
    resid = interior_residual(model, path_a, scale_a)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)

    note_a = STERILISATION_DEVICE + f" | H={cal.H_peg} | TP innovation (qp)={scale_a:.6g}"
    _write_csv(out / "irf_tp_sterilised.csv", _frame(model, path_a, horizons), note_a)
    _write_csv(
        out / "irf_tp_unsterilised.csv",
        _frame(model, path_raw, horizons),
        "Same TP innovation as the sterilised experiment, Taylor rule left to operate. "
        + "Not scenario (a).",
    )
    _write_csv(
        out / "irf_short_rate_news.csv",
        _frame(model, path_b, horizons),
        f"Scenario (b): short-rate path news scaled to +100 bp on R^L. "
        f"News innovation (qp)={scale_b:.6g}. Bank Rate is not sterilised.",
    )

    log = format_checks(checks)
    log += (
        f"\nInterior residual (first {horizons - 1} leads): {resid:.3e}\n"
        f"Blanchard-Kahn stable roots: {roots['n_stable']:.0f} "
        f"(need {roots['n']:.0f})\n"
        f"TP innovation = {scale_a:.6g} quarterly percent; "
        f"news innovation = {scale_b:.6g} quarterly percent.\n"
    )
    (out / "success_checks.txt").write_text(log, encoding="utf-8")
    (out / "irf_a_summary.txt").write_text(log, encoding="utf-8")

    plot_scenario_a(model, path_a, path_raw, out / "irf_a_tp_sterilised.png")
    plot_a_versus_b(model, path_a, path_b, out / "irf_a_vs_b_y_ph.png")

    print(log)
    if resid > 1e-8:
        print("Residual check failed")
        return 1
    if abs(roots["n_stable"] - roots["n"]) > 0:
        print("Blanchard-Kahn check failed")
        return 1
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
