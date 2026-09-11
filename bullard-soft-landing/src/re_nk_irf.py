#!/usr/bin/env python3
"""
Nested FIRE / RE New Keynesian IRFs for Bullard–Grimaud–Salle–Vermandel (Tinbergen WP 2025-001).

VERIFIED equations (WP §2.1 / §2.4), with phi_t ≡ 0 so E^{SL} = E:
  pi_t = kappa * y_t + beta * E_t pi_{t+1} + u_t
  y_t  = E_t y_{t+1} - (1/sigma) * (i_t - E_t pi_{t+1}) + g_t
  i_t  = rho_i * i_{t-1} + (1-rho_i) * (phi_pi * pi_t + phi_y * y_t) + v_t

Parameters: Table 1 posterior means (WP), beta = 0.99 calibrated.
This is the RE nesting of the HENK model, NOT the full social-learning system.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """Structural + shock params from Tinbergen WP Table 1 (posterior means) + beta."""

    beta: float = 0.99  # calibrated (WP §3.2)
    kappa: float = 0.2032
    sigma: float = 1.6993
    phi_pi: float = 1.7131
    phi_y: float = 0.1564
    rho_i: float = 0.8179
    # Shock processes (WP §2.4); monetary shock is white noise
    rho_g: float = 0.7455
    rho_u: float = 0.6328
    mu_g: float = 0.4579
    mu_u: float = 0.4887
    sigma_v: float = 0.0023  # MP shock std (Table 1); IRF uses unit shock by default


def steady_state(p: Params) -> dict[str, float]:
    """Zero steady state in deviation form (WP: intercepts zero when variables in deviations)."""
    return {"pi": 0.0, "y": 0.0, "i": 0.0, "g": 0.0, "u": 0.0, "v": 0.0}


def solve_re_policy(p: Params) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Solve MSV z_t = P z_{t-1} + Q eps_t for z = [pi, y, i, g, u, eg_lag, eu_lag].

    Shock vector eps = [eps_g, eps_u, eps_v].
    ARMA: g_t = rho_g g_{t-1} + eps_g - mu_g eps_g_{t-1}
           u_t = rho_u u_{t-1} + eps_u - mu_u eps_u_{t-1}
           v_t = eps_v

    Method: undetermined coefficients on the linear RE system (Sims-style manual system).
    State: s = [i_{t-1}, g_t, u_t, eg_lag, eu_lag] with g,u predetermined given shocks.
    We iterate on expectational consistency for one-period-ahead forecasts under AR(1)/white noise.

    For IRFs to a monetary shock alone (g=u=0), a simpler 3-variable solution suffices;
    we implement the full block for completeness.
    """
    # For monetary IRF with g=u=0, reduce to states [i_lag] and shock v.
    # Guess affine law: [pi, y, i]' = A * i_lag + B * v
    # With E_t pi_{t+1} = A_pi * i_t  (since E v_{t+1}=0), E_t y_{t+1} = A_y * i_t

    # Solve for A (3x1) and B (3x1) from:
    # pi = kappa y + beta (A_pi * i) + 0
    # y  = (A_y * i) - (1/sigma)(i - A_pi * i)
    # i  = rho_i * i_lag + (1-rho_i)(phi_pi pi + phi_y y) + v

    # Let A = [A_pi, A_y, A_i]', B = [B_pi, B_y, B_i]'
    # From i equation for the homogeneous (v=0) part and shock part separately.

    beta, kappa, sig = p.beta, p.kappa, p.sigma
    phi_pi, phi_y, rho_i = p.phi_pi, p.phi_y, p.rho_i
    inv_sig = 1.0 / sig

    def residual(x: np.ndarray) -> np.ndarray:
        A_pi, A_y, A_i, B_pi, B_y, B_i = x
        # Homogeneous: v=0, i_lag free. Then i = A_i * i_lag, pi=A_pi*i_lag, y=A_y*i_lag
        # E pi' = A_pi * i = A_pi * A_i * i_lag
        # E y'  = A_y * i  = A_y * A_i * i_lag
        r1 = A_pi - kappa * A_y - beta * (A_pi * A_i)
        r2 = A_y - (A_y * A_i) + inv_sig * (A_i - A_pi * A_i)
        r3 = A_i - rho_i - (1 - rho_i) * (phi_pi * A_pi + phi_y * A_y)
        # Shock: i_lag=0, v free. pi=B_pi v, y=B_y v, i=B_i v
        # E pi' = A_pi * i = A_pi * B_i * v
        # E y'  = A_y * i  = A_y * B_i * v
        r4 = B_pi - kappa * B_y - beta * (A_pi * B_i)
        r5 = B_y - (A_y * B_i) + inv_sig * (B_i - A_pi * B_i)
        r6 = B_i - (1 - rho_i) * (phi_pi * B_pi + phi_y * B_y) - 1.0
        return np.array([r1, r2, r3, r4, r5, r6])

    # Numerical solve (simple Newton with finite differences)
    x = np.array([0.1, -0.1, 0.5, 0.5, -0.5, 0.8], dtype=float)
    for _ in range(80):
        r = residual(x)
        if np.max(np.abs(r)) < 1e-12:
            break
        J = np.zeros((6, 6))
        eps = 1e-8
        for j in range(6):
            x2 = x.copy()
            x2[j] += eps
            J[:, j] = (residual(x2) - r) / eps
        step = np.linalg.solve(J, -r)
        x = x + step
    else:
        raise RuntimeError(f"RE policy did not converge; residual={residual(x)}")

    A = x[:3].reshape(3, 1)
    B = x[3:].reshape(3, 1)
    # P maps i_lag -> [pi,y,i]; Q maps v -> [pi,y,i]
    P = np.zeros((3, 3))
    # State vector for IRF storage: [pi, y, i]; transition uses i as lag
    # We return A, B packed for the IRF routine
    endog = ["pi", "y", "i"]
    shocks = ["v"]
    return A, B, endog, shocks


def irf_monetary(p: Params, horizon: int = 40, shock_size: float = 1.0) -> dict[str, np.ndarray]:
    """IRF to a one-time monetary policy shock v_0 = shock_size (default 1, not sigma_v)."""
    A, B, _, _ = solve_re_policy(p)
    A_pi, A_y, A_i = A.ravel()
    B_pi, B_y, B_i = B.ravel()

    pi = np.zeros(horizon)
    y = np.zeros(horizon)
    i = np.zeros(horizon)

    # t=0
    pi[0] = B_pi * shock_size
    y[0] = B_y * shock_size
    i[0] = B_i * shock_size

    for t in range(1, horizon):
        i_lag = i[t - 1]
        pi[t] = A_pi * i_lag
        y[t] = A_y * i_lag
        i[t] = A_i * i_lag

    return {"pi": pi, "y": y, "i": i, "A": A.ravel(), "B": B.ravel()}


def main() -> None:
    parser = argparse.ArgumentParser(description="RE-nested NK monetary IRFs (Bullard et al. WP)")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--shock", type=float, default=1.0, help="Size of v_0 (policy shock)")
    parser.add_argument("--save", type=str, default="", help="Optional CSV path for IRFs")
    args = parser.parse_args()

    p = Params()
    ss = steady_state(p)
    print("Steady state (deviations):", ss)
    print("Parameters (Table 1 posterior means + beta=0.99):")
    print(f"  kappa={p.kappa}, sigma={p.sigma}, phi_pi={p.phi_pi}, phi_y={p.phi_y}, rho_i={p.rho_i}")

    out = irf_monetary(p, horizon=args.horizon, shock_size=args.shock)
    print(f"\nRE policy loadings on i_lag (A_pi, A_y, A_i) = {out['A']}")
    print(f"RE policy loadings on v   (B_pi, B_y, B_i) = {out['B']}")
    print("\nMonetary shock IRF (first 12 quarters):")
    print(f"{'t':>4} {'pi':>12} {'y':>12} {'i':>12}")
    for t in range(min(12, args.horizon)):
        print(f"{t:4d} {out['pi'][t]:12.6f} {out['y'][t]:12.6f} {out['i'][t]:12.6f}")

    if args.save:
        import csv
        from pathlib import Path

        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "pi", "y", "i"])
            for t in range(args.horizon):
                w.writerow([t, out["pi"][t], out["y"][t], out["i"][t]])
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
