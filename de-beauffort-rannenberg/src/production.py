"""Production, distribution, and wage Phillips curves.

Flexible prices / sticky wages (paper §3.1): P_Z = W_S so sectoral real
prices equal sectoral real wages. Distribution costs: Corsetti–Dedola
(paper eqs. 28–31). Wage PCs: linearized eqs. (12) with slope κ_w=0.0044.
"""

import sequence_jacobian as sj


@sj.simple
def prices(w_T, w_NT, Q, mu_d, mu_f, phi_d, phi_t, lambda_t, phi_GT):
    """Relative prices in CPI units. λ_d=1 ⇒ Cobb–Douglas D/F aggregator."""
    p_NT = w_NT
    p_DT = w_T
    p_M = Q  # P*_DT = 1
    p_F = mu_f / (1.0 + mu_f) * p_NT + 1.0 / (1.0 + mu_f) * p_M
    p_D = mu_d / (1.0 + mu_d) * p_NT + 1.0 / (1.0 + mu_d) * p_DT
    p_T = p_D**phi_d * p_F ** (1.0 - phi_d)
    p = (phi_t * p_T ** (1.0 - lambda_t) + (1.0 - phi_t) * p_NT ** (1.0 - lambda_t)) ** (
        1.0 / (1.0 - lambda_t)
    )
    p_G = (
        phi_GT * p_T ** (1.0 - lambda_t) + (1.0 - phi_GT) * p_NT ** (1.0 - lambda_t)
    ) ** (1.0 / (1.0 - lambda_t))
    p_ratio = p_T / p_NT
    cpi_res = p - 1.0
    return p_NT, p_DT, p_M, p_F, p_D, p_T, p, p_G, p_ratio, cpi_res


@sj.simple
def hours(Y_T, Y_NT, phi_l, lambda_l):
    """Hours = output (linear technology). Split from MRS to keep the DAG acyclic."""
    N_T = Y_T
    N_NT = Y_NT
    N = (
        phi_l ** (-1.0 / lambda_l) * N_NT ** ((1.0 + lambda_l) / lambda_l)
        + (1.0 - phi_l) ** (-1.0 / lambda_l) * N_T ** ((1.0 + lambda_l) / lambda_l)
    ) ** (lambda_l / (1.0 + lambda_l))
    return N_T, N_NT, N


@sj.simple
def mrs_block(N, N_T, N_NT, C, phi_l, lambda_l, varphi, chi, sigma):
    """Sectoral MRS from the union objective (paper eq. 12)."""
    n_exp = varphi - 1.0 / lambda_l
    mrs_NT = (
        chi
        * N**n_exp
        * phi_l ** (-1.0 / lambda_l)
        * N_NT ** (1.0 / lambda_l)
        * C**sigma
    )
    mrs_T = (
        chi
        * N**n_exp
        * (1.0 - phi_l) ** (-1.0 / lambda_l)
        * N_T ** (1.0 / lambda_l)
        * C**sigma
    )
    return mrs_T, mrs_NT


@sj.simple
def wage_pcs(pi, w_T, w_NT, mrs_T, mrs_NT, beta, kappa_w, theta_w):
    """Linear wage Phillips curves, slope κ_w on the markup gap."""
    pi_w_T = (1.0 + pi) * w_T / w_T(-1) - 1.0
    pi_w_NT = (1.0 + pi) * w_NT / w_NT(-1) - 1.0
    gap_T = mrs_T / w_T * theta_w / (theta_w - 1.0) - 1.0
    gap_NT = mrs_NT / w_NT * theta_w / (theta_w - 1.0) - 1.0
    pc_T = pi_w_T - kappa_w * gap_T - beta * pi_w_T(+1)
    pc_NT = pi_w_NT - kappa_w * gap_NT - beta * pi_w_NT(+1)
    return pi_w_T, pi_w_NT, pc_T, pc_NT


@sj.simple
def labor_income(w_T, w_NT, N_T, N_NT, T):
    Z = w_T * N_T + w_NT * N_NT - T
    return Z
