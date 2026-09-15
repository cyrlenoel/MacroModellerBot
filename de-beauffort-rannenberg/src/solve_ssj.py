"""Steady state, Jacobians, and linear IRFs for WP 493 Figure 3."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sequence_jacobian.classes import SteadyStateDict

from .calibration import SteadyState, build_real_steady_state
from .empirical_sbvar import g_shock_level
from .households import (
    annual_impc,
    ha_calibration,
    household_ha,
    household_ha_sticky_block,
    household_rank,
    htm_share,
    solve_ha_steady_state,
)
from .model import TARGET_LIST, UNKNOWN_LIST, build_model, ss_to_calibration


DEFAULT_T = 200
PLOT_H = 20


def complete_ha_ss(phi_GT: float | None = None, psi_nfa: float = 1e-3) -> tuple[SteadyState, dict]:
    ss = build_real_steady_state(phi_GT=phi_GT, psi_nfa=psi_nfa)
    ss, hh_ss = solve_ha_steady_state(ss)
    return ss, hh_ss


def make_ss_dict(ss: SteadyState, hh_ss=None) -> SteadyStateDict:
    d = ss_to_calibration(ss)
    d["Y_T"] = ss.Y_T
    d["Y_NT"] = ss.Y_NT
    if hh_ss is not None:
        d.update({k: hh_ss[k] for k in hh_ss if k not in d})
        internals = getattr(hh_ss, "internals", None)
    else:
        internals = None
    ssdict = SteadyStateDict(d) if internals is None else SteadyStateDict(d, internals)
    return ssdict


def household_report(ss: SteadyState, hh_ss) -> dict:
    years = annual_impc(hh_ss, T=40)
    try:
        htm = htm_share(hh_ss)
    except Exception:
        htm = float("nan")
    return {
        "A": ss.A,
        "Z": ss.Z,
        "T": ss.T,
        "C_ha": ss.C,
        "C_goods": 1.0 - ss.G,
        "htm": htm,
        "impc_y1": years[0],
        "impc_y2": years[1],
        "impc_y3": years[2],
        "impc_y4": years[3],
        "impc_y5": years[4],
        "G": ss.G,
        "Y_T": ss.Y_T,
        "Y_NT": ss.Y_NT,
        "phi_l": ss.phi_l,
        "chi": ss.chi,
        "import_share": ss.M / ss.Y,
    }


def solve_variant(
    kind: str,
    phi_GT: float | None = None,
    T: int = DEFAULT_T,
    psi_nfa: float = 1e-3,
):
    """kind in {ha, ha_lb, ra, ra_lb, ha_fire}."""
    phi = 0.0 if kind.endswith("lb") else phi_GT
    ss, hh_ss = complete_ha_ss(phi_GT=phi, psi_nfa=psi_nfa)
    ssdict = make_ss_dict(ss, hh_ss)
    ssdict["beta"] = ss.beta_ha if kind.startswith("ha") else ss.beta_ra
    ss.beta = ssdict["beta"]

    if kind.startswith("ra"):
        # RANK uses the HA asset/tax SS so both models share liquidity and G.
        ssdict["beta"] = ss.beta_ra
        # Euler holds at r = 1/β_RA − 1, which is the common r.
        model = build_model(household_rank, f"rank_{kind}")
    elif kind == "ha_fire":
        model = build_model(household_ha, "hank_fire")
    else:
        sticky = household_ha_sticky_block(ssdict, T=T, theta=ss.vartheta)
        model = build_model(sticky, f"hank_{kind}")

    dG = g_shock_level(T, Y=ss.Y)
    irf = model.solve_impulse_linear(ssdict, UNKNOWN_LIST, TARGET_LIST, {"G": dG})
    return ss, ssdict, irf, household_report(ss, hh_ss)


def _pct(irf, name: str, ss_val: float, h: int) -> np.ndarray:
    path = np.array(irf[name])[:h]
    if ss_val == 0:
        return 100.0 * path
    return 100.0 * path / ss_val


def irf_to_figure3_series(irf, ss: SteadyState, h: int = PLOT_H) -> Dict[str, np.ndarray]:
    """Map model IRFs into Figure 3 units (see figure notes)."""
    return {
        "C": _pct(irf, "C", ss.C, h),
        "C_T": _pct(irf, "C_T", ss.C_T, h),
        "C_NT": _pct(irf, "C_NT", ss.C_NT, h),
        "Y": _pct(irf, "Y", ss.Y, h),
        "Y_T": _pct(irf, "Y_T", ss.Y_T, h),
        "Y_NT": _pct(irf, "Y_NT", ss.Y_NT, h),
        "p_ratio": _pct(irf, "p_ratio", 1.0, h),
        "NX": 100.0 * np.array(irf["NX"])[:h],  # pp of GDP, Y_ss=1
        "G": 100.0 * np.array(irf["G"])[:h] / ss.Y,
    }


def qualitative_flags(series: Dict[str, np.ndarray]) -> dict:
    """Paper Figure 3 / intro stylised facts (signs on impact / near term)."""
    h = slice(0, 8)
    return {
        "C_positive": float(np.mean(series["C"][h])) > 0,
        "Y_T_positive": float(np.mean(series["Y_T"][h])) > 0,
        "Y_NT_positive": float(np.mean(series["Y_NT"][h])) > 0,
        "p_ratio_negative": float(np.mean(series["p_ratio"][h])) < 0,
        "NX_negative": float(np.mean(series["NX"][h])) < 0,
        "C_T_gt_C_NT": float(np.mean(series["C_T"][h] - series["C_NT"][h])) > 0,
    }


def output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "output"
