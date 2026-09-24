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

from src.linear_model import build_model, interest_burden_split  # noqa: E402
from src.plot_irf import (  # noqa: E402
    plot_a_versus_b,
    plot_ib_decomposition,
    plot_peg_robustness,
    plot_scenario_a,
)
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


def _frame(model, path, horizon: int, with_ib_split: bool = False) -> dict[str, np.ndarray]:
    out = {"quarter": np.arange(1, horizon + 1, dtype=int)}
    for name in RATE_NAMES:
        out[f"{name}_bp"] = path.series(model, name)[:horizon] * 400.0
    for name in LEVEL_NAMES:
        out[name] = path.series(model, name)[:horizon]
    out["e_ster_bp"] = path.e_ster[:horizon] * 400.0
    if with_ib_split:
        ib_nom, ib_il = interest_burden_split(
            model,
            path.series(model, "reff")[:horizon],
            path.series(model, "ril")[:horizon],
            path.series(model, "pi")[:horizon],
            path.series(model, "dg")[:horizon],
        )
        out["ib_nom"] = ib_nom
        out["ib_il"] = ib_il
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

    note_a = (
        "SCENARIO (a) = STERILISED TERM PREMIUM ONLY. "
        + STERILISATION_DEVICE
        + f" | H={cal.H_peg} | TP innovation (qp)={scale_a:.6g}"
    )
    _write_csv(
        out / "irf_tp_sterilised.csv",
        _frame(model, path_a, horizons, with_ib_split=True),
        note_a,
    )
    _write_csv(
        out / "irf_tp_unsterilised.csv",
        _frame(model, path_raw, horizons),
        "NOT SCENARIO (a). Same TP innovation as the sterilised experiment, "
        "Taylor rule left to operate (this is what stoch_simul of e_tp does). "
        "Scenario (a) is irf_tp_sterilised.csv only.",
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
    plot_ib_decomposition(model, path_a, out / "irf_a_ib_decomposition.png")

    # Peg-length robustness. H = H_peg is scenario (a). Other horizons are not.
    robust_rows = []
    robust_paths = {}
    for H in (8, 12, 20, 40):
        path_h = solve_sterilised_qz(model, {"e_tp": _impulse(T, 1.0)}, H=H, T=T)
        _scale_to_long_rate(model, path_h)
        robust_paths[H] = path_h
        frame_h = _frame(model, path_h, horizons, with_ib_split=True)
        for q in range(horizons):
            robust_rows.append(
                {
                    "H": H,
                    "scenario_a": int(H == cal.H_peg),
                    "quarter": q + 1,
                    "y": frame_h["y"][q],
                    "ph": frame_h["ph"][q],
                    "cs": frame_h["cs"][q],
                    "cb": frame_h["cb"][q],
                    "ib": frame_h["ib"][q],
                    "ib_nom": frame_h["ib_nom"][q],
                    "ib_il": frame_h["ib_il"][q],
                    "rl_bp": frame_h["rl_bp"][q],
                    "i_bp": frame_h["i_bp"][q],
                }
            )
    rob_keys = list(robust_rows[0])
    rob_path = out / "irf_peg_robustness.csv"
    with rob_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Peg-length robustness for the SAME sterilising device. "
            "scenario_a=1 is the default H only. H=40 is a robustness case, "
            "not the baseline. stoch_simul is not in this file.\n"
        )
        fh.write(",".join(rob_keys) + "\n")
        for row in robust_rows:
            fh.write(",".join(f"{row[k]:.8g}" if k not in ("H", "scenario_a", "quarter") else str(row[k]) for k in rob_keys) + "\n")
    plot_peg_robustness(model, robust_paths, out / "irf_peg_robustness.png")

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
