"""Open-economy / trade block: CES demands, distribution, UIP, NFA.

Paper §3.3–3.4. Foreign variables are held at SS (small-open-economy
closure). Export demand uses the foreign analogue of (8)+(30) with C*
constant — the paper writes X = C*_F / (1+μ_f) but not the foreign CES;
that demand is [PROVISIONAL].
"""

import sequence_jacobian as sj


@sj.simple
def trade_demand(
    C,
    G,
    p_T,
    p_NT,
    p_D,
    p_F,
    p_DT,
    Q,
    phi_t,
    lambda_t,
    phi_d,
    phi_GT,
    phi_Gd,
    mu_d,
    mu_f,
    C_F_star,
    lambda_d,
):
    """Private and public CES demands + distribution identities (32)–(35)."""
    C_T = phi_t * p_T ** (-lambda_t) * C
    C_NT = (1.0 - phi_t) * p_NT ** (-lambda_t) * C
    C_D = phi_d * (p_D / p_T) ** (-lambda_d) * C_T
    C_F = (1.0 - phi_d) * (p_F / p_T) ** (-lambda_d) * C_T
    G_T = phi_GT * p_T ** (-lambda_t) * G
    G_NT = (1.0 - phi_GT) * p_NT ** (-lambda_t) * G
    G_D = phi_Gd * (p_D / p_T) ** (-lambda_d) * G_T
    G_F = (1.0 - phi_Gd) * (p_F / p_T) ** (-lambda_d) * G_T
    # Foreign retail price of home goods (distribution abroad, P*_NT=1)
    p_F_star = mu_f / (1.0 + mu_f) * 1.0 + 1.0 / (1.0 + mu_f) * p_DT / Q
    C_F_star_t = C_F_star * p_F_star ** (-lambda_d)
    X = C_F_star_t / (1.0 + mu_f)
    M = (C_F + G_F) / (1.0 + mu_f)
    NX = p_DT * X - Q * M
    Y_T_d = 1.0 / (1.0 + mu_d) * (C_D + G_D) + X
    Y_NT_d = (
        C_NT
        + G_NT
        + mu_d / (1.0 + mu_d) * (C_D + G_D)
        + mu_f / (1.0 + mu_f) * C_F
    )
    tot = p_DT / p_F
    return C_T, C_NT, C_D, C_F, G_T, G_NT, G_D, G_F, X, M, NX, Y_T_d, Y_NT_d, tot, p_F_star


@sj.simple
def nfa_uip(nfa, Q, NX, r_ante, i_star, psi_nfa):
    """NFA residual (27) and UIP (26) with a linear Γ wedge."""
    nfa_res = nfa - ((1.0 + i_star) * Q / Q(-1) * nfa(-1) + NX)
    uip_res = (1.0 + r_ante) - (1.0 + i_star) * Q(+1) / Q * (1.0 - psi_nfa * nfa)
    return nfa_res, uip_res


@sj.simple
def market_clearing(A, B, nfa, Y_T, Y_NT, Y_T_d, Y_NT_d, C, p_G, G, NX):
    asset_mkt = A - B - nfa
    goods_T = Y_T - Y_T_d
    goods_NT = Y_NT - Y_NT_d
    Y = C + p_G * G + NX
    return asset_mkt, goods_T, goods_NT, Y
