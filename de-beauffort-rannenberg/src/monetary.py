"""Monetary policy: Taylor rule (paper eq. 22) and Fisher equation (eq. 24).

`i` is a GE unknown; the Taylor residual is a target. That avoids a
same-block input/output cycle in the SSJ DAG. Timing follows SSJ
(ex-ante vs ex-post real rate). See NOTES.md.
"""

import sequence_jacobian as sj


@sj.simple
def taylor_res(i, pi, zeta, kappa_pi, i_star):
    """i_t = ζ i_{t-1} + (1-ζ)(i* + κ_π π_t), π net inflation (SS=0)."""
    i_res = i - (zeta * i(-1) + (1.0 - zeta) * (i_star + kappa_pi * pi))
    return i_res


@sj.simple
def fisher(i, pi):
    r_ante = (1.0 + i) / (1.0 + pi(+1)) - 1.0
    return r_ante


@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r
