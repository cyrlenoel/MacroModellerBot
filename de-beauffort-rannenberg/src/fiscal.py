"""Fiscal policy: G shock, tax rule (21), budget (20).

`B` is a GE unknown; `B_res` is the government-budget residual.
"""

import sequence_jacobian as sj


@sj.simple
def government(B, r, p_G, G, T_ss, kappa_b, B_ss):
    """Lump-sum tax rule + government flow budget residual."""
    T = T_ss + kappa_b * (B(-1) - B_ss)
    B_res = B - ((1.0 + r) * B(-1) + p_G * G - T)
    return T, B_res
