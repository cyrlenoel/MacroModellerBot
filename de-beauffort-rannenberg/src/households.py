"""Household blocks: incomplete markets (HA), RANK Euler, sticky expectations.

HA problem is paper eqs. (2)–(4) with union-set hours (eq. 1): households
choose consumption/assets only. This is the SSJ `hh_sim` one-asset block,
which the authors thank Auclert for using.

Sticky expectations follow paper eq. (13) / footnote 4 and Auclert, Rognlie
& Straub (2020) mapping (24): apply after the FIRE household Jacobian.
"""

from __future__ import annotations

import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from sequence_jacobian.classes.sparse_jacobians import make_matrix

from .calibration import SteadyState


hh_core = sj.hetblocks.hh_sim.hh
make_grids = sj.hetblocks.hh_sim.make_grids


def income(Z, e_grid):
    """Post-tax labour income z(i) = Z * e(i), paper §3.1."""
    y = Z * e_grid
    return y


household_ha = hh_core.add_hetinputs([income, make_grids]).rename("hh")


@sj.solved(unknowns={"C": 1.0, "A": 1.0}, targets=["euler", "budget"])
def household_rank(C, A, Z, r, beta, eis):
    """Representative-agent counterpart of (2)+(4), union hours taken as given."""
    euler = C(+1) - ((1.0 + r(+1)) * beta) ** eis * C
    budget = (1.0 + r) * A(-1) + Z - C - A
    return euler, budget


def sticky_expect_jacobian(J: np.ndarray, theta: float) -> np.ndarray:
    """Map a FIRE Jacobian to sticky-information (MJMH eq. 24; paper fn. 4).

    θ = ϑ = share that does *not* update (Table 2: 0.935).
    J[t, s] = d output_t / d input_s.
    """
    J = np.asarray(J, dtype=float)
    T = J.shape[0]
    out = np.empty_like(J)
    out[:, 0] = J[:, 0]
    if T == 1:
        return out
    out[0, 1:] = (1.0 - theta) * J[0, 1:]
    for t in range(1, T):
        out[t, 1:] = theta * out[t - 1, :-1] + (1.0 - theta) * J[t, 1:]
    return out


def household_ha_sticky_block(ss: dict, T: int, theta: float) -> JacobianDictBlock:
    """FIRE HA Jacobians wrt (r, Z), then sticky-information mapping."""
    J = household_ha.jacobian(ss, inputs=["r", "Z"], outputs=["C", "A"], T=T)
    nested = {}
    for o in ("C", "A"):
        nested[o] = {}
        for i in ("r", "Z"):
            nested[o][i] = sticky_expect_jacobian(make_matrix(J[o][i], T), theta)
    return JacobianDictBlock(nested, name="hh")


def ha_calibration(ss: SteadyState) -> dict:
    """Inputs for the HA hetblock steady state (plus grids)."""
    return dict(
        beta=ss.beta_ha,
        eis=ss.eis,
        rho_e=ss.rho_e,
        sd_e=ss.sigma_e,
        n_e=ss.n_e,
        n_a=ss.n_a,
        min_a=ss.min_a,
        max_a=ss.max_a,
        r=ss.r,
        Z=1.0,  # placeholder; replaced after we know T, N
    )


def solve_ha_steady_state(ss: SteadyState) -> tuple[SteadyState, dict]:
    """Solve the incomplete-markets SS given r and a guess for Z, then iterate Z.

    Z = w_T N_T + w_NT N_NT - T and T = r A + G (nfa=0, B=A).
    A comes from the household, so we iterate on Z until A is consistent.
    """
    # First pass: ignore taxes, Z ≈ N_T + N_NT - G  (T≈G if A=0)
    Z = ss.N_T + ss.N_NT - ss.G
    calib = ha_calibration(ss)
    internals = None
    for _ in range(30):
        calib["Z"] = Z
        out = household_ha.steady_state(calib)
        A = float(out["A"])
        C_ha = float(out["C"])
        T = ss.r * A + ss.G
        Z_new = ss.N_T + ss.N_NT - T
        if abs(Z_new - Z) < 1e-10:
            Z = Z_new
            internals = out
            break
        Z = 0.5 * Z + 0.5 * Z_new
        internals = out
    else:
        internals = out

    ss = ss  # mutate below
    from .calibration import attach_assets

    ss = attach_assets(ss, A=float(internals["A"]))
    # Replace C with HA aggregate C; goods-market adding-up is imposed in GE,
    # but SS C should match the household. Small gaps are reported in NOTES.
    ss.C = float(internals["C"])
    ss.Z = Z
    return ss, internals


def htm_share(internals) -> float:
    """Mass at the borrowing constraint (paper reports 49.7% HtM)."""
    D = internals.internals["hh"]["D"]
    a = internals.internals["hh"]["a"]
    a_grid = internals.internals["hh"]["a_grid"]
    constrained = a <= a_grid[0] + 1e-8
    return float(np.sum(D * constrained))


def annual_impc(ss_ha, T: int = 40) -> np.ndarray:
    """Annual iMPCs from J^{C,Z} (paper Table 1: year-1 ≈ 0.507)."""
    J = household_ha.jacobian(ss_ha, inputs=["Z"], outputs=["C"], T=T)
    m = np.array(make_matrix(J["C"]["Z"], T))[:, 0]
    years = [float(m[4 * k : 4 * (k + 1)].sum()) for k in range(5)]
    return np.array(years)
