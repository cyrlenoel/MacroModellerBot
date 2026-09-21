"""Calibration for the UK long-rate / term-premium twin.

Three numbers are held fixed by the signed sketch. Everything else is a
provisional sketch parameter: free to retune so the linear IRFs are
qualitatively right, and not claimed as an estimate.

Units used throughout the linear system
---------------------------------------
* Quantities (Y, C, C^s, C^b, I, Q, K, H, P^h, rer): percent deviations.
* Interest rates and inflation (i, pi, R^L, TP, R^m, ...): quarterly percent.
  0.25 quarterly percent = 100 annualised basis points.
* Ratios (NX, IB, debt, NFA, taxes): percentage points of quarterly GDP.
  For a flow, that coincides with percentage points of annual GDP.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


# Sketch hold-fixed values. Do not retune these to chase IRF shapes.
LAMBDA_Q = 0.09
S_IL = 0.25
D_YEARS = 9.1


@dataclass(frozen=True)
class Calibration:
    # --- HOLD FIXED (sketch) -------------------------------------------------
    lambda_q: float = LAMBDA_Q  # quarterly remortgage / reset hazard
    s_IL: float = S_IL  # index-linked share of the gilt stock
    D: float = D_YEARS  # modified duration, years

    # --- Preferences / nominal block [PROVISIONAL] ---------------------------
    beta: float = 0.995  # ~2% annual real rate
    sigma: float = 6.0  # inverse EIS. Low EIS: MTM dominates short-rate substitution
    h_s: float = 0.40  # partial adjustment of saver consumption
    kappa: float = 0.004  # NKPC slope. Flat so a finite peg is not a Fisher spiral
    iota: float = 0.30  # inflation indexation weight
    phi_pi: float = 1.50  # Taylor, same units for i and pi
    phi_y: float = 0.125  # Taylor output weight, annualised-rate convention
    rho_i: float = 0.86  # Bank Rate smoothing
    psi_nfa: float = 0.08  # Taylor loading on E nfa (SOE closure; inside i^TR)

    # --- Term structure and mortgage pass-through [PROVISIONAL except lambda] -
    rho_tp: float = 0.90  # illustrative persistence of the TP scenario
    rho_nu: float = 0.75  # illustrative persistence of short-rate path news
    omega_S: float = 0.30  # weight of Bank Rate in the new-advances rate
    omega_L: float = 0.70  # weight of the long rate in the new-advances rate
    omega_f_S: float = 0.35  # firm discount, short real rate
    omega_f_L: float = 0.65  # firm discount, long real rate

    # --- HANK-lite budgets [PROVISIONAL] -------------------------------------
    mu_ds: float = 15.0  # borrower cashflow: percent cons. per qp of stock rate
    alpha_y: float = 0.15  # borrower income channel
    mu_coll: float = 0.10  # new-borrowing / LTV channel (scaled by lambda_q)
    omega_cs: float = 0.62  # saver share of aggregate consumption
    omega_cb: float = 0.38  # borrower share of aggregate consumption
    psi_tp: float = 2.0  # saver MTM: percent of Cs per qp of TP (positive cuts Cs)
    psi_y_cs: float = 0.10  # saver income channel from aggregate output
    psi_ib: float = 0.02  # small coupon pass-through; dominated by psi_tp
    psi_tau: float = 0.12  # saver consumption drag from the tax rule

    # --- Housing and capital [PROVISIONAL] -----------------------------------
    kappa_c: float = 0.055  # housing Phillips loading on borrower consumption
    phi_h: float = 0.75  # housing Phillips loading on real new-mortgage rate
    kappa_H: float = 0.03  # housing Phillips loading on the housing stock
    kappa_level: float = 0.05  # slow anchor of the house-price level
    phi_q: float = 0.90  # q-theory semi-elasticity on the firm real rate
    psi_k: float = 6.0  # investment adjustment (I = Q / psi_k)
    psi_h: float = 3.0  # housing investment (HI = P^h / psi_h)
    delta_k: float = 0.025
    delta_h: float = 0.007

    # --- Resource shares [PROVISIONAL], NX steady state = 0 ------------------
    s_C: float = 0.62
    s_I: float = 0.11
    s_HI: float = 0.04
    s_G: float = 0.23

    # --- Open economy [PROVISIONAL] ------------------------------------------
    eta_rer: float = 0.06  # NX (pp of GDP) per percent of real depreciation
    eta_y: float = 0.18  # import leakage
    eta_ys: float = 0.05
    chi_nfa: float = 0.01  # debt-elastic UIP premium, qp rate per pp of GDP

    # --- Fiscal [PROVISIONAL except s_IL]; debt/quarterly GDP ~ 100% of annual GDP
    b_g_ratio: float = 4.0
    phi_dg: float = 0.06  # tax (pp of GDP) per pp-of-GDP debt gap
    pi_ss_annual: float = 2.0

    # --- Other AR shocks (estimation slots; not used in the (a)/(b) IRFs) ----
    rho_zeta: float = 0.50
    rho_u: float = 0.60
    rho_g: float = 0.80
    rho_ystar: float = 0.90
    rho_fx: float = 0.75

    # Scenario design (not structural parameters)
    H_peg: int = 12  # quarters of anticipated Bank Rate peg under scenario (a)
    T: int = 220  # perfect-foresight horizon
    irf_horizon: int = 40
    target_rl_qp: float = 0.25  # 100 annualised bp

    @property
    def delta_D(self) -> float:
        """Quarterly refix / EH weight 1/(4D)."""
        return 1.0 / (4.0 * self.D)

    @property
    def phi_y_qp(self) -> float:
        """Output weight when the policy rate is in quarterly percent."""
        return self.phi_y / 4.0

    @property
    def b_nom(self) -> float:
        return (1.0 - self.s_IL) * self.b_g_ratio

    @property
    def b_IL(self) -> float:
        return self.s_IL * self.b_g_ratio

    @property
    def pi_ss_qp(self) -> float:
        return self.pi_ss_annual / 4.0

    @property
    def r_ss_qp(self) -> float:
        return (1.0 / self.beta - 1.0) * 100.0

    @property
    def i_ss_qp(self) -> float:
        return self.r_ss_qp + self.pi_ss_qp

    @property
    def nkpc_forward(self) -> float:
        return self.beta / (1.0 + self.beta * self.iota)

    @property
    def nkpc_lag(self) -> float:
        return self.iota / (1.0 + self.beta * self.iota)

    def assert_accounting(self) -> None:
        if abs(self.lambda_q - LAMBDA_Q) > 0 or abs(self.s_IL - S_IL) > 0 or abs(self.D - D_YEARS) > 0:
            raise ValueError("lambda_q, s_IL and D are hold-fixed sketch calibrations")
        if abs(self.s_C + self.s_I + self.s_HI + self.s_G - 1.0) > 1e-12:
            raise ValueError("resource shares must sum to 1 when NX_ss = 0")
        if abs(self.omega_S + self.omega_L - 1.0) > 1e-12:
            raise ValueError("mortgage weights omega_S + omega_L must sum to 1")
        if abs(self.omega_f_S + self.omega_f_L - 1.0) > 1e-12:
            raise ValueError("firm-discount weights must sum to 1")
        if abs(self.omega_cs + self.omega_cb - 1.0) > 1e-12:
            raise ValueError("consumption shares must sum to 1")
        if self.s_IL <= 0:
            raise ValueError("s_IL must stay strictly positive")


def default_calibration() -> Calibration:
    cal = Calibration()
    cal.assert_accounting()
    return cal


def parameter_dict(cal: Calibration | None = None) -> dict[str, float]:
    """Float fields only, for the Dynare parameter-sync test."""
    cal = cal or default_calibration()
    out: dict[str, float] = {}
    for f in fields(cal):
        val = getattr(cal, f.name)
        if isinstance(val, float):
            out[f.name] = val
    out["delta_D"] = cal.delta_D
    out["phi_y_qp"] = cal.phi_y_qp
    out["b_nom"] = cal.b_nom
    out["b_IL"] = cal.b_IL
    out["pi_ss_qp"] = cal.pi_ss_qp
    out["i_ss_qp"] = cal.i_ss_qp
    out["nkpc_forward"] = cal.nkpc_forward
    out["nkpc_lag"] = cal.nkpc_lag
    return out
