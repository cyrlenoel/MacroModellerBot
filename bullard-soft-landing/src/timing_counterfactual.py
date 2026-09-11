#!/usr/bin/env python3
"""
Timing vs strength policy counterfactuals for the HENK Python prototype.

Inspired by Bullard–Grimaud–Salle–Vermandel Tinbergen WP 2025-001 §4
(soft landing / inflation scares): timing of tightening dominates strength
conditional on not being too weak.

This is a **transparent open-source stepping stone**, not a claim of author
replication. We do **not** have the estimated historical shock path / SPF
inversion filter; instead we compare Taylor-rule variants under a **common**
cost-push (optional combined) ARMA path and **identical SL seed**.

Scenarios [PROVISIONAL magnitudes inspired by WP §4.3–4.4 narrative]:
  baseline          — Table 1 posterior means (phi_pi=1.7131, rho_i=0.8179)
  earlier           — lower rho_i (×0.9) and higher phi_pi (+10%) from t=0
  stronger_delayed  — baseline loadings until delay D, then higher phi_pi

Default shock path: a multi-quarter cost-push *surge* (repeated innovations)
so delayed strength can still affect mid-path inflation / beliefs — closer in
spirit to WP §4 than a one-period IRF impulse. Use --surge-quarters 1 for a
classic single impulse.

Outputs under output/counterfactuals/:
  paths_{scenario}.csv, comparison_metrics.csv, comparison_summary.txt

Marking: numeric paths are [PROVISIONAL] prototype results — do not invent
paper Table 3 welfare numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from henk_sim import (  # noqa: E402
    compute_loadings,
    simulate,
    _write_csv,
)
from re_nk_irf import Params  # noqa: E402
from social_learning import SLParams  # noqa: E402


def build_arma_from_eps(
    eps: np.ndarray,
    rho: float,
    mu: float,
    use_ma: bool = True,
) -> np.ndarray:
    """x_t = rho x_{t-1} + eps_t - mu eps_{t-1} (mu=0 if not use_ma)."""
    if not use_ma:
        mu = 0.0
    horizon = eps.size
    x = np.zeros(horizon)
    x[0] = eps[0]
    for t in range(1, horizon):
        x[t] = rho * x[t - 1] + eps[t] - mu * eps[t - 1]
    return x


def build_surge_paths(
    p: Params,
    horizon: int,
    shock_size: float,
    surge_quarters: int,
    use_ma: bool,
    shock: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared fundamental paths: cost-push innovations for the first
    ``surge_quarters`` periods (default multi-quarter surge).
    Optional demand innovations at half size when shock=='combined'.
    """
    n = max(1, min(surge_quarters, horizon))
    eps_u = np.zeros(horizon)
    eps_u[:n] = shock_size
    u = build_arma_from_eps(eps_u, p.rho_u, p.mu_u, use_ma=use_ma)

    g = np.zeros(horizon)
    if shock == "combined":
        eps_g = np.zeros(horizon)
        eps_g[:n] = 0.5 * shock_size
        g = build_arma_from_eps(eps_g, p.rho_g, p.mu_g, use_ma=use_ma)

    v = np.zeros(horizon)
    return u, g, v


def _metrics(path: dict[str, np.ndarray], delay: int) -> dict[str, float]:
    pi, y, phi = path["pi"], path["y"], path["phi"]
    return {
        "peak_pi": float(np.max(pi)),
        "t_peak_pi": int(np.argmax(pi)),
        "min_y": float(np.min(y)),
        "t_min_y": int(np.argmin(y)),
        "max_abs_phi": float(np.max(np.abs(phi))),
        "t_max_abs_phi": int(np.argmax(np.abs(phi))),
        "mean_phi": float(np.mean(phi)),
        "peak_pi_after_delay": float(np.max(pi[delay:])) if delay < len(pi) else float("nan"),
        "min_y_after_delay": float(np.min(y[delay:])) if delay < len(y) else float("nan"),
    }


def _schedule_delayed_strength(
    p_base: Params,
    p_strong: Params,
    horizon: int,
    delay: int,
) -> list[dict[str, np.ndarray]]:
    """
    Baseline loadings for t < delay; stronger (higher phi_pi) loadings for t >= delay.

    [PROVISIONAL] Agents do not anticipate the future rule switch when forming
    expectations under the baseline loadings — transparent approximation only.
    """
    L0 = compute_loadings(p_base)
    L1 = compute_loadings(p_strong)
    return [L1 if t >= delay else L0 for t in range(horizon)]


def run_counterfactuals(
    horizon: int = 80,
    seed: int = 1,
    J: int = 100,
    shock_size: float = 0.01,
    delay: int = 8,
    surge_quarters: int = 12,
    use_ma: bool = True,
    shock: str = "costpush",
    save_dir: Path | None = None,
) -> dict[str, dict]:
    """
    Run baseline / earlier / stronger_delayed under the same u (and optional g) path.
    """
    repo = Path(__file__).resolve().parents[1]
    if save_dir is None:
        save_dir = repo / "output" / "counterfactuals"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    p_base = Params()
    # WP §4.3–4.4 inspired: ~10% less inertia + ~10% higher phi_pi for "earlier"
    p_earlier = replace(
        p_base,
        rho_i=round(p_base.rho_i * 0.9, 6),  # ~0.7361 vs Table 1 0.8179
        phi_pi=round(p_base.phi_pi * 1.10, 6),  # ~1.8844
    )
    # Strength only: higher phi_pi, same rho_i (applied after delay)
    p_strong = replace(p_base, phi_pi=round(p_base.phi_pi * 1.10, 6))

    L_base = compute_loadings(p_base)
    L_earlier = compute_loadings(p_earlier)

    u, g, v = build_surge_paths(
        p_base, horizon, shock_size, surge_quarters, use_ma, shock
    )

    sl_params = SLParams(J=J)
    keys = ["pi", "y", "i", "phi", "u", "g", "v"]

    scenarios: dict[str, dict] = {}

    # --- baseline ---
    base_path = simulate(
        p_base,
        L_base["A"], L_base["B"], L_base["C"], L_base["D"],
        horizon=horizon,
        shock=shock,
        shock_size=shock_size,
        use_sl=True,
        sl_params=sl_params,
        seed=seed,
        G=L_base["G"],
        use_ma=use_ma,
        u_path=u,
        g_path=g,
        v_path=v,
    )
    scenarios["baseline"] = {
        "path": base_path,
        "params": p_base,
        "note": f"Table 1: phi_pi={p_base.phi_pi}, rho_i={p_base.rho_i}",
    }

    # --- earlier (timing): lower rho_i + higher phi_pi from t=0 ---
    earlier_path = simulate(
        p_earlier,
        L_earlier["A"], L_earlier["B"], L_earlier["C"], L_earlier["D"],
        horizon=horizon,
        shock=shock,
        shock_size=shock_size,
        use_sl=True,
        sl_params=sl_params,
        seed=seed,
        G=L_earlier["G"],
        use_ma=use_ma,
        u_path=u,
        g_path=g,
        v_path=v,
    )
    scenarios["earlier"] = {
        "path": earlier_path,
        "params": p_earlier,
        "note": (
            f"Earlier from t=0: phi_pi={p_earlier.phi_pi} (+10%), "
            f"rho_i={p_earlier.rho_i} (×0.9)"
        ),
    }

    # --- stronger but delayed ---
    sched = _schedule_delayed_strength(p_base, p_strong, horizon, delay)
    delayed_path = simulate(
        p_base,
        L_base["A"], L_base["B"], L_base["C"], L_base["D"],
        horizon=horizon,
        shock=shock,
        shock_size=shock_size,
        use_sl=True,
        sl_params=sl_params,
        seed=seed,
        G=L_base["G"],
        use_ma=use_ma,
        u_path=u,
        g_path=g,
        v_path=v,
        loadings_schedule=sched,
    )
    scenarios["stronger_delayed"] = {
        "path": delayed_path,
        "params": p_strong,
        "note": (
            f"Baseline until t={delay}, then phi_pi={p_strong.phi_pi} (+10%); "
            f"rho_i stays {p_base.rho_i}"
        ),
    }

    # Write CSVs + summary
    lines: list[str] = []
    lines.append(
        "Timing vs strength counterfactuals — HENK Python prototype\n"
        f"Date context: 2026-09-11\n"
        f"horizon={horizon}, seed={seed}, J={J}, shock={shock}, "
        f"shock_size={shock_size}, surge_quarters={surge_quarters}, "
        f"use_ma={use_ma}, delay={delay}\n"
        "Same SL seed and same u/g/v paths across scenarios.\n"
        "[PROVISIONAL] Not author Table 3 / estimated historical shocks; "
        "illustrative cost-push surge under quasi-RE + SL.\n"
    )

    print("=== Timing vs strength counterfactuals ===")
    header = (
        f"{'scenario':<20} {'peak_pi':>12} {'min_y':>12} {'max|phi|':>12} "
        f"{'t_peak_pi':>10} {'t_min_y':>8}"
    )
    print(header)
    lines.append(header)

    metrics_all: dict[str, dict[str, float]] = {}
    for name, sc in scenarios.items():
        path = sc["path"]
        _write_csv(save_dir / f"paths_{name}.csv", path, keys)
        m = _metrics(path, delay)
        metrics_all[name] = m
        row = (
            f"{name:<20} {m['peak_pi']:12.6f} {m['min_y']:12.6f} "
            f"{m['max_abs_phi']:12.6f} {m['t_peak_pi']:10d} {m['t_min_y']:8d}"
        )
        print(row)
        lines.append(row)
        lines.append(f"  note: {sc['note']}")
        lines.append(
            f"  peak_pi_after_delay={m['peak_pi_after_delay']:.6f}, "
            f"min_y_after_delay={m['min_y_after_delay']:.6f}"
        )

    lines.append("\nRelative to baseline:")
    mb = metrics_all["baseline"]
    for name in ("earlier", "stronger_delayed"):
        m = metrics_all[name]
        lines.append(
            f"  {name}: Δpeak_pi={m['peak_pi'] - mb['peak_pi']:+.6f}, "
            f"Δmin_y={m['min_y'] - mb['min_y']:+.6f}, "
            f"Δmax|phi|={m['max_abs_phi'] - mb['max_abs_phi']:+.6f}"
        )

    lines.append(
        "\nInterpretation (prototype only, not paper claim):\n"
        "  WP §4 narrative: timing of tightening matters more than delayed strength\n"
        "  for containing an inflation scare under heterogeneous beliefs.\n"
        "  Compare peak_pi and max|phi| across earlier vs stronger_delayed here.\n"
        "  [PROVISIONAL] With a well-specified common PLM component, phi paths can be\n"
        "  news-dominated; policy still shifts pi/y via A,B,C,D,G loadings.\n"
    )

    summary_path = save_dir / "comparison_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote CSVs and {summary_path}")

    wide_path = save_dir / "comparison_metrics.csv"
    with wide_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "peak_pi",
                "min_y",
                "max_abs_phi",
                "t_peak_pi",
                "t_min_y",
                "peak_pi_after_delay",
                "min_y_after_delay",
                "note",
            ]
        )
        for name, sc in scenarios.items():
            m = metrics_all[name]
            w.writerow(
                [
                    name,
                    m["peak_pi"],
                    m["min_y"],
                    m["max_abs_phi"],
                    m["t_peak_pi"],
                    m["t_min_y"],
                    m["peak_pi_after_delay"],
                    m["min_y_after_delay"],
                    sc["note"],
                ]
            )
    print(f"Wrote {wide_path}")

    return {"scenarios": scenarios, "metrics": metrics_all, "save_dir": save_dir}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HENK timing vs strength counterfactuals (Python prototype)"
    )
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--J", type=int, default=100)
    parser.add_argument(
        "--shock-size",
        type=float,
        default=0.01,
        help="Per-quarter cost-push innovation during the surge (default 0.01)",
    )
    parser.add_argument(
        "--surge-quarters",
        type=int,
        default=12,
        help="Number of initial quarters with cost-push innovations (1 = IRF impulse)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=8,
        help="Quarters before stronger phi_pi kicks in (stronger_delayed)",
    )
    parser.add_argument(
        "--shock",
        type=str,
        default="costpush",
        choices=["costpush", "combined"],
        help="Shared fundamental path type",
    )
    parser.add_argument("--no-ma", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output/counterfactuals",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = repo / save_dir

    run_counterfactuals(
        horizon=args.horizon,
        seed=args.seed,
        J=args.J,
        shock_size=args.shock_size,
        delay=args.delay,
        surge_quarters=args.surge_quarters,
        use_ma=not args.no_ma,
        shock=args.shock,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
