"""Minton–Monnery (FEDS 2026-053) NKPC belief-block skeleton.

Implements the quarterly Adaptive Learning / hybrid-FIRE belief recursion
and the δ fixed point / ω_b weight from Theorem 3. Smoke-tests the FIRE nest
(α_FIRE → 1 ⇒ δ → 1, ω_b → 0).

Not a full paper replication — parametric beliefs only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeliefParams:
    """Quarterly calibration from FEDS 2026-053 §2.2 (baseline instruments)."""

    alpha_fire: float = 0.25
    rho_tilde: float = 0.82
    K_tilde: float = 0.22
    beta: float = 0.99
    theta_p: float = 0.9


def belief_update(b_lag: float, dmc: float, p: BeliefParams) -> float:
    """Average persistent-cost belief b_t (paper eq. 37)."""
    return p.K_tilde * dmc + (1.0 - p.K_tilde * p.rho_tilde) * b_lag


def delta_fixed_point(p: BeliefParams, tol: float = 1e-12, maxit: int = 200) -> float:
    """Stable root δ ∈ (θ_p, 1] solving paper eq. (39).

    At α_FIRE = 1 the equation collapses to δ = 1.
    """
    if abs(p.alpha_fire - 1.0) < 1e-15:
        return 1.0

    a, r, K, b, th = p.alpha_fire, p.rho_tilde, p.K_tilde, p.beta, p.theta_p
    # Fixed-point iteration on δ; start near 1 (FIRE-adjacent).
    delta = 0.5 * (th + 1.0)
    for _ in range(maxit):
        denom = (1.0 - b * th * r) * (1.0 - b * r * (1.0 - K) * delta)
        if abs(denom) < 1e-18:
            raise RuntimeError("δ fixed-point denominator vanished")
        inner = th - ((1.0 - th) * b * r * K) / denom * (th - delta)
        new = a + (1.0 - a) * inner
        # Keep in (θ_p, 1]
        new = float(np.clip(new, th + 1e-12, 1.0))
        if abs(new - delta) < tol:
            return new
        delta = new
    raise RuntimeError(f"δ fixed point did not converge (last={delta})")


def omega_b(p: BeliefParams, delta: float) -> float:
    """Weight on average AL cost belief b_t (paper eq. 38)."""
    if abs(p.alpha_fire - 1.0) < 1e-15:
        return 0.0
    a, r, K, b, th = p.alpha_fire, p.rho_tilde, p.K_tilde, p.beta, p.theta_p
    num = (1.0 - a) * (1.0 - th) * b * r * (1.0 - b * r * (1.0 - K)) * th
    den = (1.0 - b * th * r) * (1.0 - b * r * (1.0 - K) * delta)
    return float(num / den)


def fire_nkpc_slope(p: BeliefParams) -> float:
    """κ_FIRE = (1-βθ_p)(1-θ_p)/θ_p for the static FIRE NKPC in mc_real."""
    return (1.0 - p.beta * p.theta_p) * (1.0 - p.theta_p) / p.theta_p


def hybrid_impact_mc_coeff(p: BeliefParams, delta: float) -> float:
    """Coefficient on current real MC in the recursive NKPC (42), π-ω_b b side."""
    return (1.0 - p.theta_p) / p.theta_p + p.beta * (p.theta_p - delta)


def smoke() -> dict[str, float]:
    p = BeliefParams()
    delta = delta_fixed_point(p)
    wb = omega_b(p, delta)
    # FIRE nest
    p_fire = BeliefParams(alpha_fire=1.0)
    delta_f = delta_fixed_point(p_fire)
    wb_f = omega_b(p_fire, delta_f)
    # Belief path: one-off dmc pulse then zeros
    b = 0.0
    path = []
    for t, dmc in enumerate([0.01] + [0.0] * 7):
        b = belief_update(b, dmc, p)
        path.append(b)

    out = {
        "alpha_fire": p.alpha_fire,
        "rho_tilde": p.rho_tilde,
        "K_tilde": p.K_tilde,
        "delta": delta,
        "omega_b": wb,
        "kappa_fire": fire_nkpc_slope(p),
        "mc_impact_coeff": hybrid_impact_mc_coeff(p, delta),
        "delta_fire_nest": delta_f,
        "omega_b_fire_nest": wb_f,
        "b_after_pulse": path[0],
        "b_t7": path[-1],
    }
    return out


def main() -> None:
    m = smoke()
    assert abs(m["delta_fire_nest"] - 1.0) < 1e-10
    assert abs(m["omega_b_fire_nest"]) < 1e-10
    assert 0.9 < m["delta"] <= 1.0
    assert m["omega_b"] >= 0.0
    print("Minton–Monnery NKPC belief skeleton smoke OK")
    for k, v in m.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
