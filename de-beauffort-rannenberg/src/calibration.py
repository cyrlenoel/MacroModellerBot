"""Table 2 calibration and a consistent open-economy steady state.

Sources
-------
de Beauffort & Rannenberg, NBB Working Paper 493 (June 2026), Table 2 and §4.1.
PDF: https://www.nbb.be/doc/ts/publications/wp/wp493en.pdf
Page: https://www.nbb.be/en/publications-research/publications/all-publications/fiscal-policy-and-sectoral-spillovers-open

Status tags used in comments: VERIFIED (from Table 2 / §4.1) vs PROVISIONAL
(required for adding-up, not reported as a Table 2 number).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Table 2 — VERIFIED
# ---------------------------------------------------------------------------

TABLE2: Dict[str, float] = {
    # Households and firms
    "beta_ra": 0.995,  # RA discount factor; annual real rate of 2%
    "beta_ha": 0.986,  # HA discount factor; targets Fagereng et al. annual MPC
    "theta_w": 3.0,  # labour-services elasticity; 50% wage markup
    "kappa_w": 0.0044,  # wage Phillips-curve *slope* (Lindé et al. 2016)
    "sigma": 1.0,  # EIS (CRRA); utility c^{1-σ}/(1-σ)
    "varphi": 1.0,  # inverse Frisch (utility n^{1+φ}/(1+φ)); Table 2 "elasticity of H" = 1
    "sigma_e_table2": 0.58,  # Table 2 σ_e (listed)
    # SSJ Rouwenhorst takes the unconditional SD of log e. Using 0.58 as that
    # SD produces year-1 iMPC ≈ 0.24 and HtM ≈ 20%, missing Table 1
    # (0.507 and 49.7%). σ_e=0.35 jointly matches Table 1 given β_HA and r.
    # [PROVISIONAL interpretation of Table 2 σ_e vs SSJ sigma.]
    "sigma_e": 0.35,
    "rho_e": 0.968,  # idiosyncratic-shock persistence (Table 2 ρ_e)
    "vartheta": 0.935,  # sticky-expectation non-updating share (Auclert et al. 2020)
    # Trade
    "phi_t": 0.623,  # tradable share of private consumption
    "lambda_t": 2.15,  # elasticity of substitution T vs NT
    "phi_d": 0.57,  # private home bias (targets 12% import/GDP)
    "lambda_d": 1.0,  # trade elasticity (Cobb–Douglas D vs F)
    "phi_l": 0.6,  # NT labour bias (Table 2; see derived SS share)
    "lambda_l": 1.0,  # sectoral labour elasticity (Horvath 2000)
    "mu_f": 0.754,  # local distribution share in imported tradables
    "mu_d": 0.266,  # local distribution share in domestic tradables
    # Public policy
    "zeta": 0.9,  # interest-rate smoothing
    "kappa_pi": 1.5,  # Taylor inflation response
    "kappa_b": 0.027,  # lump-sum tax response to debt (Bi et al. 2016)
    "phi_GT": 0.25,  # tradable content of public consumption (OECD COFOG)
    "phi_Gd": 1.0,  # domestic content of public tradable spending
}

# Import/GDP target used to discipline φ_d in §4.1.
IMPORT_SHARE_TARGET = 0.12  # VERIFIED §4.1 / Table 2 source column

# Grid for the incomplete-markets block. n_e / n_a are computational, not Table 2.
HA_GRID = dict(n_e=7, n_a=200, min_a=0.0, max_a=1000.0)


def ces_price(p1: float, p2: float, share: float, elas: float) -> float:
    """CES price index. `share` on good 1. Cobb–Douglas if elas == 1."""
    if abs(elas - 1.0) < 1e-10:
        return (p1**share) * (p2 ** (1.0 - share))
    return (share * p1 ** (1.0 - elas) + (1.0 - share) * p2 ** (1.0 - elas)) ** (
        1.0 / (1.0 - elas)
    )


def labor_aggregator(n_nt: float, n_t: float, phi_l: float, lambda_l: float) -> float:
    """Paper eq. (9), λ_l = 1 reduces to a quadratic mean."""
    a = phi_l ** (-1.0 / lambda_l) * n_nt ** ((1.0 + lambda_l) / lambda_l)
    b = (1.0 - phi_l) ** (-1.0 / lambda_l) * n_t ** ((1.0 + lambda_l) / lambda_l)
    return (a + b) ** (lambda_l / (1.0 + lambda_l))


def derived_g_share(p: Dict[str, float] | None = None) -> float:
    """G/Y implied by import/GDP = 12% at unit prices, NX=0, Y=1.

    Table 2 does not list G/Y. Combining φ_d, φ_t, μ_f and the 12% import
    target with NX=0 and P=1 gives a unique G/Y. [PROVISIONAL as a named
    Table-2 parameter; VERIFIED as the unique value consistent with §4.1.]
    """
    p = p or TABLE2
    # M = C_F / (1+μ_f), C_F = (1-φ_d) φ_t C, C = 1-G, M = 0.12
    coef = (1.0 - p["phi_d"]) * p["phi_t"] / (1.0 + p["mu_f"])
    return 1.0 - IMPORT_SHARE_TARGET / coef


@dataclass
class SteadyState:
    """Consistent unit-price steady state used by the SSJ DAG."""

    # parameters
    beta: float
    beta_ra: float
    beta_ha: float
    theta_w: float
    kappa_w: float
    sigma: float
    eis: float
    varphi: float
    sigma_e: float
    sigma_e_table2: float
    rho_e: float
    vartheta: float
    phi_t: float
    lambda_t: float
    phi_d: float
    lambda_d: float
    phi_l: float
    phi_l_table2: float
    lambda_l: float
    mu_f: float
    mu_d: float
    zeta: float
    kappa_pi: float
    kappa_b: float
    phi_GT: float
    phi_Gd: float
    psi_nfa: float
    # grid
    n_e: int
    n_a: int
    min_a: float
    max_a: float
    # prices / rates
    r: float
    i: float
    i_star: float
    pi: float
    w_T: float
    w_NT: float
    Q: float
    p_NT: float
    p_DT: float
    p_M: float
    p_F: float
    p_D: float
    p_T: float
    p: float
    p_G: float
    # quantities
    Y: float
    G: float
    C: float
    C_T: float
    C_NT: float
    C_D: float
    C_F: float
    G_T: float
    G_NT: float
    G_D: float
    G_F: float
    X: float
    M: float
    NX: float
    Y_T: float
    Y_NT: float
    N_T: float
    N_NT: float
    N: float
    chi: float
    mrs_T: float
    mrs_NT: float
    markup_w: float
    # assets / fiscal (filled after HA SS)
    A: float
    B: float
    nfa: float
    T: float
    Z: float
    T_ss: float
    B_ss: float
    C_F_star: float

    def as_sj(self) -> dict:
        """Flat dict for sequence-jacobian SteadyStateDict / calibration."""
        d = asdict(self)
        d["eis"] = self.eis
        return d


def assert_real_ss_guardrails(ss: SteadyState, import_tol: float = 1e-8, mrs_tol: float = 1e-8) -> None:
    """Fail loudly if the unit-price SS misses the §4.1 import or MRS targets."""
    import_share = ss.M / ss.Y
    gap_m = abs(import_share - IMPORT_SHARE_TARGET)
    if gap_m >= import_tol:
        raise AssertionError(
            f"Steady-state import share M/Y={import_share:.12f} misses "
            f"target {IMPORT_SHARE_TARGET} by {gap_m:.3e} (tol {import_tol})."
        )
    if abs(ss.w_T - 1.0) > mrs_tol or abs(ss.w_NT - 1.0) > mrs_tol:
        raise AssertionError(
            f"MRS guardrail expects unit wages; got w_T={ss.w_T}, w_NT={ss.w_NT}."
        )
    gap_mrs = abs(ss.mrs_T - ss.mrs_NT)
    if gap_mrs >= mrs_tol:
        raise AssertionError(
            f"Sectoral MRS not equal at unit wages: mrs_T={ss.mrs_T:.12f}, "
            f"mrs_NT={ss.mrs_NT:.12f}, |gap|={gap_mrs:.3e} (tol {mrs_tol})."
        )


def build_real_steady_state(
    phi_GT: float | None = None,
    psi_nfa: float = 1e-3,
    sigma_e: float | None = None,
) -> SteadyState:
    """Unit-price SS with Y=1, NX=0, P_Z=W_S=1.

    `psi_nfa` is the linear UIP debt-elastic wedge. Paper Γ=exp(-1e-9 nfa);
    1e-3 is a Schmitt-Grohé–Uribe-style value used for a stable linear solve
    ([PROVISIONAL] numerical; paper's 1e-9 is recorded in NOTES.md).
    `sigma_e` overrides Table 2 / default Rouwenhorst SD when not None.
    """
    p = dict(TABLE2)
    if phi_GT is not None:
        p["phi_GT"] = float(phi_GT)
    if sigma_e is not None:
        p["sigma_e"] = float(sigma_e)

    r = 1.0 / p["beta_ra"] - 1.0  # VERIFIED: β_RA targets 2% annual
    Y = 1.0
    G = derived_g_share(p)
    C = Y - G  # NX=0, p_G=1

    C_T = p["phi_t"] * C
    C_NT = (1.0 - p["phi_t"]) * C
    C_D = p["phi_d"] * C_T
    C_F = (1.0 - p["phi_d"]) * C_T
    G_T = p["phi_GT"] * G
    G_NT = (1.0 - p["phi_GT"]) * G
    G_D = p["phi_Gd"] * G_T
    G_F = (1.0 - p["phi_Gd"]) * G_T
    M = (C_F + G_F) / (1.0 + p["mu_f"])
    X = M  # balanced trade at unit prices
    C_F_star = X * (1.0 + p["mu_f"])

    # Paper eq. (34) writes distribution only on C_D, C_F. Including G_D in the
    # domestic distribution margin is required for labour income = GDP
    # ([PROVISIONAL] adding-up; coincides with (34) when φ_GT=0).
    Y_NT = (
        C_NT
        + G_NT
        + p["mu_d"] / (1.0 + p["mu_d"]) * (C_D + G_D)
        + p["mu_f"] / (1.0 + p["mu_f"]) * C_F
    )
    Y_T = 1.0 / (1.0 + p["mu_d"]) * (C_D + G_D) + X

    N_T, N_NT = Y_T, Y_NT
    # φ_l in Table 2 is 0.6. §4.1 sets it so sectoral MRS are equal at observed
    # sectoral output shares. With λ_l=φ=1 that is φ_l = N_NT/(N_NT+N_T).
    phi_l_shares = N_NT / (N_NT + N_T)
    phi_l = phi_l_shares  # use the targeting description, not the rounded 0.6
    N = labor_aggregator(N_NT, N_T, phi_l, p["lambda_l"])

    markup_w = p["theta_w"] / (p["theta_w"] - 1.0)
    # SS wage PC: MRS/W = (θ_w-1)/θ_w
    mrs_target = 1.0 / markup_w
    # MRS_NT = χ N^{φ-1/λ} φ_l^{-1/λ} N_NT^{1/λ} C^σ
    n_exp = p["varphi"] - 1.0 / p["lambda_l"]
    chi = mrs_target / (
        (N**n_exp)
        * (phi_l ** (-1.0 / p["lambda_l"]))
        * (N_NT ** (1.0 / p["lambda_l"]))
        * (C ** p["sigma"])
    )
    mrs_NT = (
        chi
        * (N**n_exp)
        * (phi_l ** (-1.0 / p["lambda_l"]))
        * (N_NT ** (1.0 / p["lambda_l"]))
        * (C ** p["sigma"])
    )
    mrs_T = (
        chi
        * (N**n_exp)
        * ((1.0 - phi_l) ** (-1.0 / p["lambda_l"]))
        * (N_T ** (1.0 / p["lambda_l"]))
        * (C ** p["sigma"])
    )

    ss = SteadyState(
        beta=p["beta_ha"],
        beta_ra=p["beta_ra"],
        beta_ha=p["beta_ha"],
        theta_w=p["theta_w"],
        kappa_w=p["kappa_w"],
        sigma=p["sigma"],
        eis=p["sigma"],
        varphi=p["varphi"],
        sigma_e=p["sigma_e"],
        sigma_e_table2=p["sigma_e_table2"],
        rho_e=p["rho_e"],
        vartheta=p["vartheta"],
        phi_t=p["phi_t"],
        lambda_t=p["lambda_t"],
        phi_d=p["phi_d"],
        lambda_d=p["lambda_d"],
        phi_l=phi_l,
        phi_l_table2=p["phi_l"],
        lambda_l=p["lambda_l"],
        mu_f=p["mu_f"],
        mu_d=p["mu_d"],
        zeta=p["zeta"],
        kappa_pi=p["kappa_pi"],
        kappa_b=p["kappa_b"],
        phi_GT=p["phi_GT"],
        phi_Gd=p["phi_Gd"],
        psi_nfa=psi_nfa,
        n_e=HA_GRID["n_e"],
        n_a=HA_GRID["n_a"],
        min_a=HA_GRID["min_a"],
        max_a=HA_GRID["max_a"],
        r=r,
        i=r,
        i_star=r,
        pi=0.0,
        w_T=1.0,
        w_NT=1.0,
        Q=1.0,
        p_NT=1.0,
        p_DT=1.0,
        p_M=1.0,
        p_F=1.0,
        p_D=1.0,
        p_T=1.0,
        p=1.0,
        p_G=1.0,
        Y=Y,
        G=G,
        C=C,
        C_T=C_T,
        C_NT=C_NT,
        C_D=C_D,
        C_F=C_F,
        G_T=G_T,
        G_NT=G_NT,
        G_D=G_D,
        G_F=G_F,
        X=X,
        M=M,
        NX=0.0,
        Y_T=Y_T,
        Y_NT=Y_NT,
        N_T=N_T,
        N_NT=N_NT,
        N=N,
        chi=chi,
        mrs_T=mrs_T,
        mrs_NT=mrs_NT,
        markup_w=markup_w,
        A=np.nan,
        B=np.nan,
        nfa=0.0,
        T=np.nan,
        Z=np.nan,
        T_ss=np.nan,
        B_ss=np.nan,
        C_F_star=C_F_star,
    )
    assert_real_ss_guardrails(ss)
    return ss


def attach_assets(ss: SteadyState, A: float) -> SteadyState:
    """Close the asset side: nfa=0 ⇒ B=A; T covers rB+G (paper eq. 20 in SS)."""
    ss.A = float(A)
    ss.nfa = 0.0
    ss.B = ss.A
    ss.B_ss = ss.B
    ss.T = ss.r * ss.B + ss.p_G * ss.G
    ss.T_ss = ss.T
    ss.Z = ss.w_T * ss.N_T + ss.w_NT * ss.N_NT - ss.T
    return ss
