#!/usr/bin/env python3
"""
HENK prototype: couple RE NK policy loadings with social-learning phi_t wedge.

Paper: Bullard–Grimaud–Salle–Vermandel, Soft landing and inflation scares
       (Tinbergen WP 2025-001 / JME 2026).

VERIFIED nesting (WP eqs. 5, 8, Assumption 2):
  E^{SL}_t pi_{t+1} = phi_t + E_t pi_{t+1}
  E_t phi_{t+1} = phi_t   (martingale / random-walk beliefs)
  z_t = P z_{t-1} + Q eps_t + R phi_t

Approximation used here [PROVISIONAL — honest stepping stone, not author codes]:
  1. Import FIRE loadings A (on i_lag) and B (on v) from re_nk_irf.solve_re_policy.
  2. Solve additional linear loadings C on contemporaneous phi, D on cost-push u,
     and G on demand g under the same martingale-phi expectational structure
     (internal rationality / quasi-RE observer that takes phi_t as given).
  3. Each period: SL updates {a_j} from *past* inflation (WP timing) → phi_t;
     then macro block sets [pi,y,i] = A*i_lag + B*v + C*phi + D*u + G*g.
  4. Shock paths: ARMA(1,1) for u and g when use_ma=True (WP §2.4);
     loadings D/G still solved under AR(1) E x'=rho x (MA lag not in MSV state).
  5. Fitness common forecast (toward WP PLM (6)):
        common_fcast_s = A_pi*i_{s-1} + D_pi*u_s + G_pi*g_s [+ B_pi*v_s]
     Full P, Q̃ rows still blocked without Dynare SL toolbox.

FIRE nested case: disable SL (phi≡0) and compare IRFs to active SL.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow `python3 src/henk_sim.py` from repo root
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from re_nk_irf import Params, solve_re_policy  # noqa: E402
from social_learning import (  # noqa: E402
    SLParams,
    SocialLearning,
    build_plm_common_fcast,
)


def solve_phi_loadings(p: Params, A: np.ndarray) -> np.ndarray:
    """
    Loadings C = [C_pi, C_y, C_i]' such that [pi,y,i]' += C * phi
    when E phi' = phi and FIRE A loadings are used for E[z'|i,phi].

    [PROVISIONAL] Closed-form undetermined coefficients under Assumption 2;
    matches WP spirit (R column in eq. 5) but is not the Dynare toolbox matrix.
    """
    A_pi, A_y, A_i = float(A[0]), float(A[1]), float(A[2])
    beta, kappa, inv_sig = p.beta, p.kappa, 1.0 / p.sigma
    phi_pi, phi_y, rho_i = p.phi_pi, p.phi_y, p.rho_i

    def residual(x: np.ndarray) -> np.ndarray:
        C_pi, C_y, C_i = x
        r1 = C_pi - kappa * C_y - beta * (A_pi * C_i + C_pi + 1.0)
        r2 = (
            C_y
            - (A_y * C_i + C_y)
            + inv_sig * (C_i - 1.0 - A_pi * C_i - C_pi)
        )
        r3 = C_i - (1.0 - rho_i) * (phi_pi * C_pi + phi_y * C_y)
        return np.array([r1, r2, r3])

    x = np.array([1.0, 0.0, 0.5], dtype=float)
    for _ in range(80):
        r = residual(x)
        if np.max(np.abs(r)) < 1e-12:
            break
        J = np.zeros((3, 3))
        eps = 1e-8
        for j in range(3):
            x2 = x.copy()
            x2[j] += eps
            J[:, j] = (residual(x2) - r) / eps
        x = x + np.linalg.solve(J, -r)
    else:
        raise RuntimeError(f"phi loadings did not converge; residual={residual(x)}")
    return x


def solve_u_loadings(p: Params, A: np.ndarray) -> np.ndarray:
    """
    Loadings D on contemporaneous cost-push u (AR(1): E u' = rho_u u).
    [PROVISIONAL] Expectational structure ignores MA(1) mu_u; simulated u path
    may still follow ARMA(1,1) via arma_path().
    """
    A_pi, A_y, A_i = float(A[0]), float(A[1]), float(A[2])
    beta, kappa, inv_sig = p.beta, p.kappa, 1.0 / p.sigma
    phi_pi, phi_y, rho_i = p.phi_pi, p.phi_y, p.rho_i
    rho_u = p.rho_u

    def residual(x: np.ndarray) -> np.ndarray:
        D_pi, D_y, D_i = x
        r1 = D_pi - kappa * D_y - beta * (A_pi * D_i + D_pi * rho_u) - 1.0
        r2 = (
            D_y
            - (A_y * D_i + D_y * rho_u)
            + inv_sig * (D_i - A_pi * D_i - D_pi * rho_u)
        )
        r3 = D_i - (1.0 - rho_i) * (phi_pi * D_pi + phi_y * D_y)
        return np.array([r1, r2, r3])

    x = np.array([1.0, -0.5, 0.5], dtype=float)
    for _ in range(80):
        r = residual(x)
        if np.max(np.abs(r)) < 1e-12:
            break
        J = np.zeros((3, 3))
        eps = 1e-8
        for j in range(3):
            x2 = x.copy()
            x2[j] += eps
            J[:, j] = (residual(x2) - r) / eps
        x = x + np.linalg.solve(J, -r)
    else:
        raise RuntimeError(f"u loadings did not converge; residual={residual(x)}")
    return x


def solve_g_loadings(p: Params, A: np.ndarray) -> np.ndarray:
    """
    Loadings G on contemporaneous demand shock g (AR(1): E g' = rho_g g).
    Enters IS with +1. [PROVISIONAL] same AR(1) expectation caveat as D / MA.
    """
    A_pi, A_y, A_i = float(A[0]), float(A[1]), float(A[2])
    beta, kappa, inv_sig = p.beta, p.kappa, 1.0 / p.sigma
    phi_pi, phi_y, rho_i = p.phi_pi, p.phi_y, p.rho_i
    rho_g = p.rho_g

    def residual(x: np.ndarray) -> np.ndarray:
        G_pi, G_y, G_i = x
        # NKPC: no direct g; E pi' = A_pi i + G_pi rho_g g
        r1 = G_pi - kappa * G_y - beta * (A_pi * G_i + G_pi * rho_g)
        # IS: y = E y' - (1/sig)(i - E pi') + g
        r2 = (
            G_y
            - (A_y * G_i + G_y * rho_g)
            + inv_sig * (G_i - A_pi * G_i - G_pi * rho_g)
            - 1.0
        )
        r3 = G_i - (1.0 - rho_i) * (phi_pi * G_pi + phi_y * G_y)
        return np.array([r1, r2, r3])

    x = np.array([0.2, 1.0, 0.3], dtype=float)
    for _ in range(80):
        r = residual(x)
        if np.max(np.abs(r)) < 1e-12:
            break
        J = np.zeros((3, 3))
        eps = 1e-8
        for j in range(3):
            x2 = x.copy()
            x2[j] += eps
            J[:, j] = (residual(x2) - r) / eps
        x = x + np.linalg.solve(J, -r)
    else:
        raise RuntimeError(f"g loadings did not converge; residual={residual(x)}")
    return x


def arma_path(
    horizon: int,
    eps0: float,
    rho: float,
    mu: float,
    use_ma: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-time innovation at t=0 through ARMA(1,1):
      x_t = rho x_{t-1} + eps_t - mu eps_{t-1}
    If use_ma=False, mu is forced to 0 (pure AR(1)).
    Returns (x, eps).
    """
    if not use_ma:
        mu = 0.0
    eps = np.zeros(horizon)
    x = np.zeros(horizon)
    eps[0] = eps0
    x[0] = eps[0]  # x_{-1}=0, eps_{-1}=0
    for t in range(1, horizon):
        x[t] = rho * x[t - 1] + eps[t] - mu * eps[t - 1]
    return x, eps


def compute_loadings(p: Params) -> dict[str, np.ndarray]:
    """Solve A,B (FIRE) and C,D,G (provisional HENK) for given Taylor / structural params."""
    A, B, _, _ = solve_re_policy(p)
    A = A.ravel()
    B = B.ravel()
    C = solve_phi_loadings(p, A)
    D = solve_u_loadings(p, A)
    G = solve_g_loadings(p, A)
    return {"A": A, "B": B, "C": C, "D": D, "G": G}


def simulate(
    p: Params,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    horizon: int,
    shock: str,
    shock_size: float,
    use_sl: bool,
    sl_params: SLParams,
    seed: int,
    G: np.ndarray | None = None,
    use_ma: bool = True,
    u_path: np.ndarray | None = None,
    g_path: np.ndarray | None = None,
    v_path: np.ndarray | None = None,
    loadings_schedule: list[dict[str, np.ndarray]] | None = None,
    lambda_path: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Simulate paths under FIRE (use_sl=False, phi=0) or active SL.

    shock in {"monetary","costpush","demand","combined"}: impulse construction
    when exogenous paths are not supplied.
    use_ma: ARMA(1,1) vs AR(1) for u and g (WP mu_u, mu_g).
    Optional u_path / g_path / v_path override shock construction (same seed SL).
    loadings_schedule: optional length-horizon list of dicts with keys A,B,C,D,G
        for time-varying Taylor rules (delayed strength); default fixed loadings.
    """
    if G is None:
        G = np.zeros(3)
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    C = np.asarray(C, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    G = np.asarray(G, dtype=float).ravel()

    pi = np.zeros(horizon)
    y = np.zeros(horizon)
    i = np.zeros(horizon)
    phi = np.zeros(horizon)
    u = np.zeros(horizon)
    g = np.zeros(horizon)
    v = np.zeros(horizon)
    common_fcast = np.zeros(horizon)
    i_lag_hist = np.zeros(horizon)

    rng = np.random.default_rng(seed)
    sl = SocialLearning(sl_params, rng=rng) if use_sl else None

    # Build / accept exogenous shock paths
    if v_path is not None:
        v[:] = np.asarray(v_path, dtype=float)[:horizon]
    if u_path is not None:
        u[:] = np.asarray(u_path, dtype=float)[:horizon]
    if g_path is not None:
        g[:] = np.asarray(g_path, dtype=float)[:horizon]

    if u_path is None and g_path is None and v_path is None:
        if shock == "monetary":
            v[0] = shock_size
        elif shock == "costpush":
            u[:], _ = arma_path(horizon, shock_size, p.rho_u, p.mu_u, use_ma=use_ma)
        elif shock == "demand":
            g[:], _ = arma_path(horizon, shock_size, p.rho_g, p.mu_g, use_ma=use_ma)
        elif shock == "combined":
            # Unit cost-push + smaller demand (illustrative; not a paper calibration)
            u[:], _ = arma_path(horizon, shock_size, p.rho_u, p.mu_u, use_ma=use_ma)
            g[:], _ = arma_path(horizon, 0.5 * shock_size, p.rho_g, p.mu_g, use_ma=use_ma)
        else:
            raise ValueError(f"unknown shock={shock!r}")
    elif shock == "costpush" and u_path is None:
        u[:], _ = arma_path(horizon, shock_size, p.rho_u, p.mu_u, use_ma=use_ma)
    elif shock == "monetary" and v_path is None:
        v[0] = shock_size

    for t in range(horizon):
        if loadings_schedule is not None:
            L = loadings_schedule[t]
            A_t, B_t, C_t, D_t, G_t = L["A"], L["B"], L["C"], L["D"], L["G"]
        else:
            A_t, B_t, C_t, D_t, G_t = A, B, C, D, G

        A_pi, A_y, A_i = float(A_t[0]), float(A_t[1]), float(A_t[2])
        B_pi, B_y, B_i = float(B_t[0]), float(B_t[1]), float(B_t[2])
        C_pi, C_y, C_i = float(C_t[0]), float(C_t[1]), float(C_t[2])
        D_pi, D_y, D_i = float(D_t[0]), float(D_t[1]), float(D_t[2])
        G_pi, G_y, G_i = float(G_t[0]), float(G_t[1]), float(G_t[2])

        # SL timing (WP): beliefs update from history through t-1, then macro
        if use_sl:
            assert sl is not None
            lam = None if lambda_path is None else float(lambda_path[t])
            phi[t] = sl.step(
                pi[:t],
                common_fcast_hist=common_fcast[:t],
                lambda_t=lam,
            )
        else:
            phi[t] = 0.0

        i_lag = i[t - 1] if t > 0 else 0.0
        i_lag_hist[t] = i_lag
        pi[t] = A_pi * i_lag + B_pi * v[t] + C_pi * phi[t] + D_pi * u[t] + G_pi * g[t]
        y[t] = A_y * i_lag + B_y * v[t] + C_y * phi[t] + D_y * u[t] + G_y * g[t]
        i[t] = A_i * i_lag + B_i * v[t] + C_i * phi[t] + D_i * u[t] + G_i * g[t]

        # Richer common PLM component for *this* date (used in future fitness)
        # [PROVISIONAL] A_pi*i_lag + D_pi*u + G_pi*g + B_pi*v  (WP PLM (6) spirit)
        common_fcast[t] = build_plm_common_fcast(
            np.array([i_lag]),
            A_pi=A_pi,
            u=np.array([u[t]]),
            D_pi=D_pi,
            g=np.array([g[t]]),
            G_pi=G_pi,
            v=np.array([v[t]]),
            B_pi=B_pi,
        )[0]

    return {
        "pi": pi,
        "y": y,
        "i": i,
        "phi": phi,
        "u": u,
        "g": g,
        "v": v,
        "common_fcast": common_fcast,
        "re_fcast": common_fcast,  # alias for older callers
        "i_lag": i_lag_hist,
    }


def _write_csv(path: Path, data: dict[str, np.ndarray], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    H = len(data[keys[0]])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + keys)
        for t in range(H):
            w.writerow([t] + [float(data[k][t]) for k in keys])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HENK prototype: FIRE vs social-learning IRFs (Bullard et al.)"
    )
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="output")
    parser.add_argument("--J", type=int, default=100, help="Even SL population size")
    parser.add_argument(
        "--shock",
        type=str,
        default="both",
        choices=["monetary", "costpush", "demand", "combined", "both"],
        help="Impulse type(s) to simulate",
    )
    parser.add_argument(
        "--shock-size",
        type=float,
        default=1.0,
        help="Impulse size at t=0 (unit shock, not sigma)",
    )
    parser.add_argument(
        "--no-ma",
        action="store_true",
        help="Disable MA(1) on u/g (pure AR(1))",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = repo / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    p = Params()
    L = compute_loadings(p)
    A, B, C, D, G = L["A"], L["B"], L["C"], L["D"], L["G"]
    use_ma = not args.no_ma

    print("RE / HENK linear loadings (Table 1 params):")
    print(f"  A (i_lag) = {A}")
    print(f"  B (v)     = {B}")
    print(f"  C (phi)   = {C}   [PROVISIONAL undetermined coefficients]")
    print(f"  D (u)     = {D}   [PROVISIONAL; path uses ARMA if use_ma={use_ma}]")
    print(f"  G (g)     = {G}   [PROVISIONAL; path uses ARMA if use_ma={use_ma}]")

    sl_params = SLParams(J=args.J)
    if args.shock == "both":
        shocks = ["monetary", "costpush"]
    else:
        shocks = [args.shock]

    summary_lines: list[str] = []
    summary_lines.append(
        "FIRE vs SL comparison — Bullard et al. HENK Python prototype\n"
        f"horizon={args.horizon}, seed={args.seed}, J={args.J}, "
        f"shock_size={args.shock_size}, use_ma={use_ma}\n"
        "Approximation: quasi-RE observer with martingale phi; SL micro process "
        "from WP eqs (10)-(12); fitness common_fcast = A_pi*i_lag+D_pi*u+G_pi*g+B_pi*v.\n"
        "NOT a full Dynare-toolbox / author replication.\n"
    )

    for shock in shocks:
        fire = simulate(
            p, A, B, C, D, args.horizon, shock, args.shock_size,
            use_sl=False, sl_params=sl_params, seed=args.seed, G=G, use_ma=use_ma,
        )
        sl = simulate(
            p, A, B, C, D, args.horizon, shock, args.shock_size,
            use_sl=True, sl_params=sl_params, seed=args.seed, G=G, use_ma=use_ma,
        )

        keys = ["pi", "y", "i", "phi", "u", "g", "v"]
        _write_csv(save_dir / f"irf_{shock}_fire.csv", fire, keys)
        _write_csv(save_dir / f"irf_{shock}_sl.csv", sl, keys)

        diff = {k: sl[k] - fire[k] for k in keys}
        _write_csv(save_dir / f"irf_{shock}_sl_minus_fire.csv", diff, keys)

        header = f"\n=== {shock} shock (size={args.shock_size}, ma={use_ma}) ==="
        print(header)
        summary_lines.append(header)

        col = (
            f"{'t':>4} {'pi_FIRE':>12} {'pi_SL':>12} "
            f"{'y_FIRE':>12} {'y_SL':>12} {'phi_SL':>12}"
        )
        print(col)
        summary_lines.append(col)
        for t in range(min(12, args.horizon)):
            line = (
                f"{t:4d} {fire['pi'][t]:12.6f} {sl['pi'][t]:12.6f} "
                f"{fire['y'][t]:12.6f} {sl['y'][t]:12.6f} {sl['phi'][t]:12.6f}"
            )
            print(line)
            summary_lines.append(line)

        t_peak_fire = int(np.argmax(np.abs(fire["pi"])))
        t_peak_sl = int(np.argmax(np.abs(sl["pi"])))
        stats = (
            f"  |pi| peak FIRE at t={t_peak_fire}: {fire['pi'][t_peak_fire]:.6f}; "
            f"SL at t={t_peak_sl}: {sl['pi'][t_peak_sl]:.6f}; "
            f"phi_SL at t=12: {sl['phi'][min(12, args.horizon-1)]:.6f}; "
            f"max|phi|: {np.max(np.abs(sl['phi'])):.6f}"
        )
        print(stats)
        summary_lines.append(stats)

    out_txt = save_dir / "fire_vs_sl_summary.txt"
    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote CSVs and {out_txt}")


if __name__ == "__main__":
    main()
