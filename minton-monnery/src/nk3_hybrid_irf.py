"""3-eq NK IRFs: FIRE vs Minton–Monnery hybrid NKPC (FEDS 2026-053).

Provisional open-source smoke aligned with Section 4.1 / Fig 6 spirit:
hold the *real-rate* path exogenous so output is common across belief models,
then compare inflation under FIRE vs hybrid PC (eq. 42 + belief 37).

Supply: adverse AR(1) cost path a_t (instant in mc).
Demand: real-rate easing; costs follow a sticky-wage lag so MP is slow to
hit mc (paper's mechanism for demand underreaction). Not a full sticky-wage
block or Fig D.17 Taylor-rule replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nkpc_belief_skeleton import (
    BeliefParams,
    delta_fixed_point,
    fire_nkpc_slope,
    omega_b,
)


@dataclass(frozen=True)
class NK3Params:
    sigma: float = 1.0  # eis
    frisch: float = 0.5
    rho_r: float = 0.5  # MP / real-rate persistence (demand)
    rho_a: float = 0.9  # cost path persistence (supply)
    # Wage-stickiness proxy: mc_t = ρ_w mc_{t-1} + (1-ρ_w) φ y_t
    # Chosen so wage PC slope ≈ price FIRE slope under paper's κ_w = κ_p,FIRE idea.
    rho_w: float = 0.9
    T: int = 40

    @property
    def phi(self) -> float:
        return 1.0 / self.frisch + 1.0 / self.sigma


def _ar1(rho: float, T: int, impact: float = 1.0) -> np.ndarray:
    path = np.zeros(T)
    path[0] = impact
    for t in range(1, T):
        path[t] = rho * path[t - 1]
    return path


def solve_output_from_real_rate(r: np.ndarray, sigma: float) -> np.ndarray:
    """Forward-looking IS with terminal y_T = 0: y_t = -σ r_t + y_{t+1}."""
    T = len(r)
    y = np.zeros(T)
    for t in range(T - 1, -1, -1):
        y_next = 0.0 if t == T - 1 else y[t + 1]
        y[t] = -sigma * r[t] + y_next
    return y


def sticky_wage_mc(y: np.ndarray, phi: float, rho_w: float) -> np.ndarray:
    """Partial-adjustment mc toward flexible φy (sticky-wage proxy)."""
    T = len(y)
    mc = np.zeros(T)
    for t in range(T):
        target = phi * y[t]
        lag = 0.0 if t == 0 else mc[t - 1]
        mc[t] = rho_w * lag + (1.0 - rho_w) * target
    return mc


def solve_fire_pi(mc: np.ndarray, bp: BeliefParams) -> np.ndarray:
    """FIRE NKPC perfect foresight: π_t = β π_{t+1} + κ mc_t."""
    T = len(mc)
    kappa = fire_nkpc_slope(bp)
    pi = np.zeros(T)
    for t in range(T - 1, -1, -1):
        pi_next = 0.0 if t == T - 1 else pi[t + 1]
        pi[t] = bp.beta * pi_next + kappa * mc[t]
    return pi


def solve_hybrid_pi(
    mc: np.ndarray, bp: BeliefParams
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Hybrid PC (42) + belief (37) under perfect foresight given {mc_t}."""
    T = len(mc)
    delta = delta_fixed_point(bp)
    wb = omega_b(bp, delta)
    c_mc = (1.0 - bp.theta_p) / bp.theta_p + bp.beta * (bp.theta_p - delta)

    n = 2 * T
    A = np.zeros((n, n))
    rhs = np.zeros(n)

    for t in range(T):
        # (42)
        row = t
        A[row, t] = 1.0  # π_t
        A[row, T + t] = -wb  # -ω_b b_t
        if t < T - 1:
            A[row, t + 1] = -bp.beta * delta
            A[row, T + t + 1] = bp.beta * delta * wb
            rhs[row] = c_mc * mc[t] - (1.0 - delta) * bp.beta * mc[t + 1]
        else:
            rhs[row] = c_mc * mc[t]

        # Belief: b_t = K Δmc_t^nom + (1-Kρ) b_{t-1}, Δmc^nom ≈ Δmc + π
        row = T + t
        A[row, T + t] = 1.0
        A[row, t] = -bp.K_tilde
        dmc_exog = mc[t] - (0.0 if t == 0 else mc[t - 1])
        rhs[row] = bp.K_tilde * dmc_exog
        if t > 0:
            A[row, T + t - 1] = -(1.0 - bp.K_tilde * bp.rho_tilde)

    x = np.linalg.solve(A, rhs)
    return x[:T], x[T:], delta, wb


def run_experiment(shock: str, bp: BeliefParams, nk: NK3Params) -> dict[str, np.ndarray]:
    if shock == "demand":
        r = -_ar1(nk.rho_r, nk.T, impact=1.0)
        y = solve_output_from_real_rate(r, nk.sigma)
        a = np.zeros(nk.T)
        mc = sticky_wage_mc(y, nk.phi, nk.rho_w)
    elif shock == "supply":
        r = np.zeros(nk.T)
        y = np.zeros(nk.T)  # real-rate fixed at 0 ⇒ y=0 under this IS
        a = _ar1(nk.rho_a, nk.T, impact=1.0)
        mc = a.copy()
    else:
        raise ValueError(shock)

    pi_fire = solve_fire_pi(mc, bp)
    pi_hyb, bel, delta, wb = solve_hybrid_pi(mc, bp)
    return {
        "r": r,
        "a": a,
        "y": y,
        "mc": mc,
        "pi_fire": pi_fire,
        "pi_hybrid": pi_hyb,
        "b": bel,
        "delta": np.array([delta]),
        "omega_b": np.array([wb]),
    }


def main() -> None:
    bp = BeliefParams()
    nk = NK3Params()
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pack: dict[str, np.ndarray] = {}
    rows = [
        "shock,pi0_fire,pi0_hybrid,ratio_hybrid_over_fire,cum_pi_fire,cum_pi_hybrid",
    ]
    print("Minton–Monnery 3-eq NK IRF smoke OK (real-rate rule; sticky-wage demand)")
    dlt = delta_fixed_point(bp)
    print(f"  δ={dlt:.6g}, ω_b={omega_b(bp, dlt):.6g}, φ={nk.phi:.4g}, ρ_w={nk.rho_w}")

    for shock in ("supply", "demand"):
        res = run_experiment(shock, bp, nk)
        for k, v in res.items():
            pack[f"{shock}_{k}"] = v
        f0, h0 = float(res["pi_fire"][0]), float(res["pi_hybrid"][0])
        ratio = h0 / f0 if abs(f0) > 1e-14 else float("nan")
        cf, ch = float(np.sum(res["pi_fire"])), float(np.sum(res["pi_hybrid"]))
        rows.append(f"{shock},{f0:.6g},{h0:.6g},{ratio:.6g},{cf:.6g},{ch:.6g}")
        print(
            f"  {shock}: π₀ FIRE={f0:.4g}, hybrid={h0:.4g}, "
            f"hybrid/FIRE={ratio:.4g}; cum_π FIRE={cf:.4g}, hybrid={ch:.4g}"
        )
        # §4.1 / Fig 6 qualitative: supply overreacts; demand underreacts on impact
        if shock == "supply":
            assert h0 > f0 > 0, "hybrid should overreact to supply on impact"
        if shock == "demand":
            assert 0 < h0 < f0, "hybrid should underreact to demand on impact"

    np.savez(out_dir / "nk3_hybrid_irf.npz", **pack)
    (out_dir / "nk3_hybrid_irf_impact.csv").write_text("\n".join(rows) + "\n")
    print(f"  wrote {out_dir / 'nk3_hybrid_irf.npz'} and impact CSV")


if __name__ == "__main__":
    main()
