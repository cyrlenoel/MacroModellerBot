"""Linearised SOE NK + HANK-lite system, twin of ``dynare/uk_long_rate.mod``.

Unknowns are stacked as ``M_p E y_{t+1} + M_c y_t + M_l y_{t-1} + M_e e_t = 0``.
Rates are quarterly percent; quantities are percent; ratios are pp of quarterly GDP.
See ``calibration.py`` and NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calibration import Calibration, default_calibration


# Order is the Dynare ``var`` order in uk_long_rate.mod.
VARIABLES: tuple[str, ...] = (
    # predetermined
    "i",
    "k",
    "h",
    "rms",
    "bm",
    "dg",
    "nfa",
    "tp",
    "nu",
    "reff",
    "ril",
    "cslag",
    "pilag",
    "phlag",
    "zeta",
    "u",
    "g",
    "ystar",
    "rhofx",
    # forward-looking
    "pi",
    "cs",
    "q",
    "ph",
    "reh",
    "rer",
    # static
    "y",
    "c",
    "cb",
    "inv",
    "hi",
    "rm",
    "rl",
    "ib",
    "nx",
    "tau",
    "mpk",
)

EXOGENOUS: tuple[str, ...] = (
    "e_ster",
    "e_tp",
    "e_news",
    "e_zeta",
    "e_u",
    "e_g",
    "e_ystar",
    "e_rhofx",
)


@dataclass
class LinearModel:
    cal: Calibration
    names: tuple[str, ...]
    exo: tuple[str, ...]
    Mp: np.ndarray
    Mc: np.ndarray
    Ml: np.ndarray
    Me: np.ndarray
    eq_names: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def ix(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @property
    def ix_exo(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.exo)}


def _acc(mat: np.ndarray, eq: int, col: int, val: float) -> None:
    mat[eq, col] += val


def build_model(cal: Calibration | None = None) -> LinearModel:
    cal = cal or default_calibration()
    cal.assert_accounting()
    names = VARIABLES
    exo = EXOGENOUS
    ix = {n: i for i, n in enumerate(names)}
    jx = {n: i for i, n in enumerate(exo)}
    n = len(names)
    Mp = np.zeros((n, n))
    Mc = np.zeros((n, n))
    Ml = np.zeros((n, n))
    Me = np.zeros((n, len(exo)))
    eq_names: list[str] = []

    def eq(name: str) -> int:
        eq_names.append(name)
        return len(eq_names) - 1

    def c(e: int, var: str, val: float) -> None:
        _acc(Mc, e, ix[var], val)

    def p(e: int, var: str, val: float) -> None:
        _acc(Mp, e, ix[var], val)

    def lag(e: int, var: str, val: float) -> None:
        _acc(Ml, e, ix[var], val)

    def shock(e: int, name: str, val: float) -> None:
        _acc(Me, e, jx[name], val)

    dD = cal.delta_D
    rho = cal.rho_i
    eis = (1.0 - cal.h_s) / cal.sigma  # coefficient on the real rate in the Euler

    # --- predetermined -------------------------------------------------------
    e = eq("taylor")
    # i = rho i(-1) + (1-rho)(phi_pi pi + phi_y_qp y) + nu + e_ster
    c(e, "i", 1.0)
    lag(e, "i", -rho)
    c(e, "pi", -(1.0 - rho) * cal.phi_pi)
    c(e, "y", -(1.0 - rho) * cal.phi_y_qp)
    c(e, "nu", -1.0)
    shock(e, "e_ster", -1.0)

    e = eq("capital")
    # k = (1-delta_k) k(-1) + delta_k inv
    c(e, "k", 1.0)
    lag(e, "k", -(1.0 - cal.delta_k))
    c(e, "inv", -cal.delta_k)

    e = eq("housing_stock")
    c(e, "h", 1.0)
    lag(e, "h", -(1.0 - cal.delta_h))
    c(e, "hi", -cal.delta_h)

    e = eq("mortgage_stock_rate")
    # R^{m,stock} = (1-lambda) R^{m,stock}(-1) + lambda R^m
    c(e, "rms", 1.0)
    lag(e, "rms", -(1.0 - cal.lambda_q))
    c(e, "rm", -cal.lambda_q)

    e = eq("ltv")
    # B^m = (1-lambda) B^m(-1) + lambda (P^h + H)
    c(e, "bm", 1.0)
    lag(e, "bm", -(1.0 - cal.lambda_q))
    c(e, "ph", -cal.lambda_q)
    c(e, "h", -cal.lambda_q)

    e = eq("debt")
    # real debt (pp of quarterly GDP). Uplift is in IB, not in real IL debt.
    # Inflation erodes only the conventional stock.
    # dg = dg(-1)/beta + g_share - tau + b_nom*Reff + b_IL*rIL - b_nom*pi
    c(e, "dg", 1.0)
    lag(e, "dg", -(1.0 / cal.beta))
    c(e, "g", -cal.s_G)
    c(e, "tau", 1.0)
    c(e, "reff", -cal.b_nom)
    c(e, "ril", -cal.b_IL)
    c(e, "pi", cal.b_nom)

    e = eq("nfa")
    c(e, "nfa", 1.0)
    lag(e, "nfa", -(1.0 / cal.beta))
    c(e, "nx", -1.0)

    e = eq("tp")
    c(e, "tp", 1.0)
    lag(e, "tp", -cal.rho_tp)
    shock(e, "e_tp", -1.0)

    e = eq("news")
    c(e, "nu", 1.0)
    lag(e, "nu", -cal.rho_nu)
    shock(e, "e_news", -1.0)

    e = eq("reff")
    # conventional effective coupon refixes at speed 1/(4D)
    c(e, "reff", 1.0)
    lag(e, "reff", -(1.0 - dD))
    c(e, "rl", -dD)

    e = eq("ril")
    # real IL coupon refixes toward the real long rate R^L - E pi
    c(e, "ril", 1.0)
    lag(e, "ril", -(1.0 - dD))
    c(e, "rl", -dD)
    p(e, "pi", dD)

    e = eq("cslag")
    c(e, "cslag", 1.0)
    lag(e, "cs", -1.0)

    e = eq("pilag")
    c(e, "pilag", 1.0)
    lag(e, "pi", -1.0)

    e = eq("phlag")
    c(e, "phlag", 1.0)
    lag(e, "ph", -1.0)

    e = eq("zeta")
    c(e, "zeta", 1.0)
    lag(e, "zeta", -cal.rho_zeta)
    shock(e, "e_zeta", -1.0)

    e = eq("costpush")
    c(e, "u", 1.0)
    lag(e, "u", -cal.rho_u)
    shock(e, "e_u", -1.0)

    e = eq("gshock")
    c(e, "g", 1.0)
    lag(e, "g", -cal.rho_g)
    shock(e, "e_g", -1.0)

    e = eq("ystar")
    c(e, "ystar", 1.0)
    lag(e, "ystar", -cal.rho_ystar)
    shock(e, "e_ystar", -1.0)

    e = eq("rhofx")
    # scenario (c) slot: UIP risk-premium shock. Not fired in the MVP IRFs.
    c(e, "rhofx", 1.0)
    lag(e, "rhofx", -cal.rho_fx)
    shock(e, "e_rhofx", -1.0)

    # --- forward-looking -----------------------------------------------------
    e = eq("nkpc")
    # hybrid NKPC. Weights nkpc_forward + nkpc_lag = (beta+iota)/(1+beta*iota) < 1
    # when iota>0; the residual is the usual indexation normalisation.
    c(e, "pi", 1.0)
    p(e, "pi", -cal.nkpc_forward)
    c(e, "pilag", -cal.nkpc_lag)
    c(e, "y", -cal.kappa)
    c(e, "u", -1.0)

    e = eq("saver_euler")
    # Habit Euler, external habit. inc_pct is a percent-of-consumption shifter
    # (duration carry, mark-to-market, domestic coupons net of taxes), scaled so
    # a permanent inc_pct of x raises Cs by x:
    #   (1+h) Cs - h cslag - Cs(+1) + eis (i - pi(+1)) - (1-h) inc_pct = 0
    # inc_pct = carry*TP + mtm*(TP - TP(-1)) + fisc*(theta_dom*IB - tau)
    # mtm < 0: a positive TP innovation is a capital loss.
    inc_scale = 1.0 - cal.h_s
    c(e, "cs", 1.0 + cal.h_s)
    c(e, "cslag", -cal.h_s)
    p(e, "cs", -1.0)
    c(e, "i", eis)
    p(e, "pi", -eis)
    c(e, "tp", -inc_scale * (cal.carry + cal.mtm))
    lag(e, "tp", inc_scale * cal.mtm)
    c(e, "ib", -inc_scale * cal.fisc_coef * cal.theta_dom)
    c(e, "tau", inc_scale * cal.fisc_coef)

    e = eq("tobin_q")
    # Q = beta E Q(+1) + (1-beta) mpk - phi_q * r_firm
    # mpk is contemporaneous (end-of-period capital timing) so output is not
    # a second forward variable. r_firm = wS (i - E pi) + wL (R^L - E pi).
    c(e, "q", 1.0)
    p(e, "q", -cal.beta)
    c(e, "mpk", -(1.0 - cal.beta))
    c(e, "i", cal.phi_q * cal.omega_f_S)
    c(e, "rl", cal.phi_q * cal.omega_f_L)
    p(e, "pi", -cal.phi_q)  # weights sum to 1

    e = eq("housing_phillips")
    # ΔP^h = beta E ΔP^h(+1) + kappa_c Cb - phi_h (R^m - E pi) - kappa_H H - kappa_level P^h
    # (1+beta+kappa_level) Ph - Ph_lag - beta Ph(+1) - kappa_c Cb
    #     + phi_h (Rm - pi(+1)) + kappa_H H = 0
    c(e, "ph", 1.0 + cal.beta + cal.kappa_level)
    c(e, "phlag", -1.0)
    p(e, "ph", -cal.beta)
    c(e, "cb", -cal.kappa_c)
    c(e, "rm", cal.phi_h)
    p(e, "pi", -cal.phi_h)
    c(e, "h", cal.kappa_H)

    e = eq("expectations_hypothesis")
    # R^{EH} = (1/(4D)) i + (1-1/(4D)) E R^{EH}(+1)
    c(e, "reh", 1.0)
    c(e, "i", -dD)
    p(e, "reh", -(1.0 - dD))

    e = eq("uip")
    # rer = E rer(+1) - (i - E pi) + chi nfa + rhofx
    # (a rise in rer is a sterling depreciation)
    c(e, "rer", 1.0)
    p(e, "rer", -1.0)
    c(e, "i", 1.0)
    p(e, "pi", -1.0)
    c(e, "nfa", -cal.chi_nfa)
    c(e, "rhofx", -1.0)

    # --- static --------------------------------------------------------------
    e = eq("resource")
    c(e, "y", 1.0)
    c(e, "c", -cal.s_C)
    c(e, "inv", -cal.s_I)
    c(e, "hi", -cal.s_HI)
    c(e, "g", -cal.s_G)
    c(e, "nx", -1.0)

    e = eq("aggregate_c")
    c(e, "c", 1.0)
    c(e, "cs", -cal.omega_cs)
    c(e, "cb", -cal.omega_cb)

    e = eq("borrower")
    # C^b = alpha_y Y - mu_ds R^{m,stock} + mu_coll * lambda * (P^h - B^m(-1))
    c(e, "cb", 1.0)
    c(e, "y", -cal.alpha_y)
    c(e, "rms", cal.mu_ds)
    c(e, "ph", -cal.mu_coll * cal.lambda_q)
    lag(e, "bm", cal.mu_coll * cal.lambda_q)

    e = eq("investment")
    c(e, "inv", 1.0)
    c(e, "q", -(1.0 / cal.psi_k))

    e = eq("housing_investment")
    c(e, "hi", 1.0)
    c(e, "ph", -(1.0 / cal.psi_h))

    e = eq("new_mortgage_rate")
    # R^m = omega_S i + omega_L R^L + zeta
    c(e, "rm", 1.0)
    c(e, "i", -cal.omega_S)
    c(e, "rl", -cal.omega_L)
    c(e, "zeta", -1.0)

    e = eq("long_rate")
    # R^L = R^{EH} + TP
    c(e, "rl", 1.0)
    c(e, "reh", -1.0)
    c(e, "tp", -1.0)

    e = eq("interest_burden")
    # IB = b_nom R^{eff} + b_IL (r^{IL} + pi) + steady-state coupon × debt gap
    ss_nom = (cal.i_ss_qp / 100.0) * (1.0 - cal.s_IL)
    ss_il = ((cal.r_ss_qp + cal.pi_ss_qp) / 100.0) * cal.s_IL
    c(e, "ib", 1.0)
    c(e, "reff", -cal.b_nom)
    c(e, "ril", -cal.b_IL)
    c(e, "pi", -cal.b_IL)
    c(e, "dg", -(ss_nom + ss_il))

    e = eq("net_exports")
    c(e, "nx", 1.0)
    c(e, "rer", -cal.eta_rer)
    c(e, "y", cal.eta_y)
    c(e, "ystar", -cal.eta_ys)

    e = eq("tax_rule")
    c(e, "tau", 1.0)
    lag(e, "dg", -cal.phi_dg)

    e = eq("mpk")
    c(e, "mpk", 1.0)
    c(e, "y", -1.0)
    lag(e, "k", 1.0)

    if len(eq_names) != n:
        raise RuntimeError(f"equation count {len(eq_names)} != variable count {n}")

    # Habit must load on contemporaneous cslag, not on cslag(-1). Guard the sign.
    euler = eq_names.index("saver_euler")
    if abs(Mc[euler, ix["cslag"]] + cal.h_s) > 1e-12:
        raise RuntimeError("saver habit should enter as -h * cslag_t")
    if abs(Ml[euler, ix["cslag"]]) > 1e-12:
        raise RuntimeError("saver habit must not also load on cslag(-1)")

    return LinearModel(
        cal=cal,
        names=names,
        exo=exo,
        Mp=Mp,
        Mc=Mc,
        Ml=Ml,
        Me=Me,
        eq_names=tuple(eq_names),
    )
