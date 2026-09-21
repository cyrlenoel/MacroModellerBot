"""Success criteria for the sterilised +100 bp term-premium IRF.

Quarter numbers are 1-indexed (quarter 1 = impact).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.linear_model import LinearModel, interest_burden_split
from src.solve_pf import Path, taylor_notional


STERILISATION_DEVICE = (
    "anticipated Bank Rate peg: the Taylor rule is left in place and the "
    "monetary residual is set to ε_ster,t = −i_t^TR for the first H quarters, "
    "with i_t^path = 0, so i_t = ρ_i i_{t−1} + (1−ρ_i)(φ_π π_t + φ_y Y_t) "
    "+ ψ_nfa E_t nfa_{t+1} + ν_t + ε_ster,t = 0; residuals are anticipated "
    "(perfect foresight) and the Taylor rule resumes after H. "
    "Scenario (a) is this sterilised path only. stoch_simul and any "
    "unsterilised TP run leave the Taylor rule free and are not scenario (a)"
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _q(index0: int) -> int:
    return int(index0) + 1


def evaluate_scenario_a(model: LinearModel, path: Path, horizon: int | None = None) -> list[CheckResult]:
    """Criteria the sketch asks for after a +100 bp sterilised TP move."""
    cal = model.cal
    h = int(horizon if horizon is not None else cal.irf_horizon)
    H = cal.H_peg
    y = path.series(model, "y")[:h]
    ph = path.series(model, "ph")[:h]
    cs = path.series(model, "cs")[:h]
    cb = path.series(model, "cb")[:h]
    ib = path.series(model, "ib")[:h]
    rl = path.series(model, "rl")[:h]
    reff = path.series(model, "reff")[:h]
    i = path.series(model, "i")[:H]
    rms = path.series(model, "rms")[:h]
    i_tr = taylor_notional(model, path)[:H]
    ster = path.e_ster[:H]

    q_ph = _q(int(np.argmin(ph)))
    q_cb = _q(int(np.argmin(cb)))
    q_rms = _q(int(np.argmax(rms)))
    q_ib = _q(int(np.argmax(ib)))
    q_reff = _q(int(np.argmax(reff)))
    ib_nom, ib_il = interest_burden_split(
        model,
        path.series(model, "reff")[:h],
        path.series(model, "ril")[:h],
        path.series(model, "pi")[:h],
        path.series(model, "dg")[:h],
    )

    full_pass = cal.b_nom * rl[0]  # IB if the whole conventional stock refixed at once
    both_down = float(np.max(cs)) <= 1e-3 and float(np.max(cb)) <= 1e-3
    checks = [
        CheckResult(
            "long rate +100 bp on impact",
            abs(rl[0] * 400.0 - 100.0) < 0.05,
            f"R^L impact = {rl[0] * 400:.3f} annualised bp",
        ),
        CheckResult(
            "Bank Rate pegged over H",
            float(np.max(np.abs(i)) * 400.0) < 1e-4,
            f"max |i| over {H}q = {np.max(np.abs(i)) * 400:.3e} annualised bp",
        ),
        CheckResult(
            "sterilising residual offsets the Taylor rule",
            float(np.max(np.abs(ster + i_tr))) < 1e-8,
            "max |ε_ster + i^TR| = "
            f"{np.max(np.abs(ster + i_tr)):.3e} quarterly percent",
        ),
        CheckResult(
            "output falls",
            bool(y[0] <= 0.0 and np.min(y) < -1e-3),
            f"Y impact = {y[0]:.3f}%, trough = {np.min(y):.3f}% at q{_q(int(np.argmin(y)))}",
        ),
        CheckResult(
            "house prices fall on impact",
            bool(ph[0] < -1e-4),
            f"P^h impact = {ph[0]:.3f}%",
        ),
        CheckResult(
            "house-price trough in quarters 4-12",
            4 <= q_ph <= 12 and q_ph < h,
            f"P^h trough = {np.min(ph):.3f}% at q{q_ph}",
        ),
        CheckResult(
            "house-price trough is a few percent, not a double-digit move",
            bool(-4.5 <= float(np.min(ph)) <= -1.0),
            f"P^h trough = {np.min(ph):.3f}% (provisional housing loadings)",
        ),
        CheckResult(
            "effective coupon refixes at 1/(4D)",
            abs(reff[0] / rl[0] - cal.delta_D) < 1e-8,
            f"R^eff_0 / R^L_0 = {reff[0] / rl[0]:.6f}, 1/(4D) = {cal.delta_D:.6f}",
        ),
        CheckResult(
            "interest burden rises, slower than full refix",
            bool(np.max(ib) > 0.0 and ib[0] < 0.25 * full_pass and q_reff >= 4),
            (
                f"IB impact = {ib[0]:.4f} pp of GDP, peak = {np.max(ib):.4f} at q{q_ib}, "
                f"full conventional refix would be {full_pass:.4f}; "
                f"R^eff peaks at q{q_reff}"
            ),
        ),
        CheckResult(
            "saver consumption does not rise",
            bool(float(np.max(cs)) <= 1e-3 and float(np.min(cs)) < -0.05),
            f"C^s ranges [{np.min(cs):.3f}%, {np.max(cs):.3f}%]",
        ),
        CheckResult(
            "borrowers are hit at least as hard as savers",
            bool(
                both_down
                and float(np.max(cb)) <= float(np.max(cs)) + 1e-6
                and float(np.min(cb)) <= float(np.min(cs)) + 1e-6
            ),
            (
                f"C^b ranges [{np.min(cb):.3f}%, {np.max(cb):.3f}%], "
                f"C^s ranges [{np.min(cs):.3f}%, {np.max(cs):.3f}%]"
            ),
        ),
        CheckResult(
            "long rate stays positive over the reported window",
            bool(float(np.min(rl)) > 0.0),
            f"min R^L over {h}q = {np.min(rl) * 400:.2f} annualised bp",
        ),
        CheckResult(
            "IL uplift dips on impact and the nominal coupon rises",
            bool(float(ib_il[0]) < 0.0 and float(np.max(ib_nom)) > 0.0),
            (
                f"IL component impact {ib_il[0]:.4f} pp of GDP, "
                f"nominal-coupon peak {np.max(ib_nom):.4f} at q{_q(int(np.argmax(ib_nom)))}"
            ),
        ),
        CheckResult(
            "borrower trough lines up with the mortgage-stock rate",
            abs(q_cb - q_rms) <= 2,
            f"C^b trough at q{q_cb} ({np.min(cb):.3f}%), R^m,stock peak at q{q_rms}",
        ),
    ]
    return checks


def format_checks(checks: list[CheckResult]) -> str:
    lines = [STERILISATION_DEVICE, ""]
    for c in checks:
        flag = "PASS" if c.ok else "FAIL"
        lines.append(f"[{flag}] {c.name}: {c.detail}")
    lines.append("")
    lines.append(
        "ALL PASS" if all(c.ok for c in checks) else "SOME CHECKS FAILED"
    )
    return "\n".join(lines) + "\n"
