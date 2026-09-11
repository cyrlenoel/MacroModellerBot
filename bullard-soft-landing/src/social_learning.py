#!/usr/bin/env python3
"""
Social-learning (SL) belief dynamics for Bullard–Grimaud–Salle–Vermandel
(Tinbergen WP 2025-001 / JME 2026 soft-landing paper).

VERIFIED from WP §2.3 eqs. (10)–(13):
  Mutation / news:  m_{j,t} = a_{j,t-1} + 1{varpi_{j,t}<=mu}*(iota_{j,t}+lambda_t)
  Fitness:          F_{j,t} = - sum_{tau=0}^{t-1} rho^tau * (pi_{t-tau} - E_j[pi_{t-tau}|m_{j,t}])^2
  Tournament:       paired agents; both adopt the higher-fitness belief (eq. 12)
  Aggregate:        phi_t = (1/J) * sum_j a_{j,t}

PLM / fitness forecast [PROVISIONAL]:
  WP eq. (6):  pî_t^{(j)} = a_{j,t} + P_{1,•} ẑ_{t-1} + Q̃_{1,•} E_t
  Full MSV rows P, Q̃ are not recoverable from the WP text alone (need Dynare SL
  toolbox). Callers therefore supply a *common* forecast component
  ``common_fcast`` (alias ``re_fcast_hist``) so that
      E[pi_s | m] = m + common_fcast_s
  Recommended common_fcast (wired in henk_sim):
      A_pi * i_{s-1} + D_pi * u_s + G_pi * g_s [+ optional B_pi * v_s]
  i.e. FIRE lag loading + cost-push / demand loadings. Omitting the R*phi
  column is intentional: under Assumption 1 agents proxy aggregate phi by
  their own intercept m. If common_fcast is omitted / zeros, fitness ranks
  beliefs by proximity of m to realised inflation — still nested when phi=0.

This module is NumPy-only and does NOT require Matlab / Dynare SL toolbox.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SLParams:
    """Panel C of Table 1 (Tinbergen WP) posterior means + population size."""

    mu: float = 0.4357  # news / mutation frequency (Calvo-like sticky info)
    rho: float = 0.7745  # fitness decay on past squared forecast errors
    sigma_iota: float = 0.0006  # idiosyncratic news std
    sigma_lambda: float = 0.0004  # aggregate news std
    J: int = 100  # even population size


def build_plm_common_fcast(
    i_lag: np.ndarray,
    A_pi: float,
    u: np.ndarray | None = None,
    D_pi: float = 0.0,
    g: np.ndarray | None = None,
    G_pi: float = 0.0,
    v: np.ndarray | None = None,
    B_pi: float = 0.0,
) -> np.ndarray:
    """
    Build the common (non-intercept) PLM component for fitness.

    [PROVISIONAL] Stand-in for P_{1,•} z_{t-1} + Q̃_{1,•} E_t excluding the
    idiosyncratic intercept a_j (WP eq. 6). Length must match the inflation
    history passed to ``fitness`` / ``step``.

    Parameters
    ----------
    i_lag : array
        Interest-rate lag at each history date s (i_{s-1}; 0 at s=0).
    A_pi : float
        FIRE loading of pi on i_lag.
    u, D_pi : optional cost-push path and loading.
    g, G_pi : optional demand path and loading.
    v, B_pi : optional monetary-shock path and loading.
    """
    i_lag = np.asarray(i_lag, dtype=float)
    out = A_pi * i_lag
    if u is not None:
        u = np.asarray(u, dtype=float)
        if u.shape != i_lag.shape:
            raise ValueError("u must match i_lag shape")
        out = out + D_pi * u
    if g is not None:
        g = np.asarray(g, dtype=float)
        if g.shape != i_lag.shape:
            raise ValueError("g must match i_lag shape")
        out = out + G_pi * g
    if v is not None:
        v = np.asarray(v, dtype=float)
        if v.shape != i_lag.shape:
            raise ValueError("v must match i_lag shape")
        out = out + B_pi * v
    return out


class SocialLearning:
    """Finite-population SL process; nests FIRE when a_j ≡ 0 and news shut off."""

    def __init__(self, params: SLParams | None = None, rng: np.random.Generator | None = None):
        self.p = params or SLParams()
        if self.p.J % 2 != 0:
            raise ValueError("Population J must be even (WP tournament pairing).")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.a = np.zeros(self.p.J, dtype=float)  # long-run inflation beliefs a_j

    @property
    def phi(self) -> float:
        """Cross-sectional mean belief — WP eq. (8) / (13)."""
        return float(np.mean(self.a))

    def mutate(self, lambda_t: float | None = None) -> np.ndarray:
        """
        News / mutation step — WP eq. (10).
        Returns candidate beliefs m_j before tournament.
        """
        p = self.p
        if lambda_t is None:
            lambda_t = float(self.rng.normal(0.0, p.sigma_lambda))
        iota = self.rng.normal(0.0, p.sigma_iota, size=p.J)
        varpi = self.rng.uniform(0.0, 1.0, size=p.J)
        news = (varpi <= p.mu).astype(float) * (iota + lambda_t)
        return self.a + news

    def fitness(
        self,
        m: np.ndarray,
        pi_hist: np.ndarray,
        re_fcast_hist: np.ndarray | None = None,
        common_fcast_hist: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Discounted squared forecast-error fitness — WP eq. (11).

        pi_hist: realised inflation deviations, oldest → newest (length T).
        common_fcast_hist / re_fcast_hist: common PLM component of each
            pi_hist[s] (same length); default zeros. Prefer common_fcast_hist;
            re_fcast_hist is kept as a backward-compatible alias.

        Forecast under candidate intercept m: E[pi_s|m] = m + common_fcast_s.
        [PROVISIONAL] See module docstring / build_plm_common_fcast.
        """
        pi_hist = np.asarray(pi_hist, dtype=float)
        T = pi_hist.size
        if T == 0:
            return np.zeros(m.shape[0], dtype=float)

        if common_fcast_hist is not None:
            common = np.asarray(common_fcast_hist, dtype=float)
        elif re_fcast_hist is not None:
            common = np.asarray(re_fcast_hist, dtype=float)
        else:
            common = np.zeros(T, dtype=float)
        if common.shape != pi_hist.shape:
            raise ValueError("common/re_fcast_hist must match pi_hist shape")

        # Weights: tau=0 is most recent → rho^0; tau=T-1 is oldest → rho^{T-1}
        tau = np.arange(T)[::-1]  # for s=0..T-1, tau = T-1-s
        w = (self.p.rho ** tau).astype(float)

        # error_{j,s} = pi_s - (m_j + common_s)
        target = pi_hist - common  # shape (T,)
        err = target[:, None] - m[None, :]
        F = -np.sum(w[:, None] * err**2, axis=0)
        return F

    def tournament(self, m: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Random pairing tournament — WP eq. (12).
        Winner's m is copied by both members of each pair.
        """
        J = m.shape[0]
        perm = self.rng.permutation(J)
        a_new = np.empty(J, dtype=float)
        for i in range(0, J, 2):
            k, ell = int(perm[i]), int(perm[i + 1])
            if F[k] > F[ell]:
                a_new[k] = m[k]
                a_new[ell] = m[k]
            elif F[ell] > F[k]:
                a_new[k] = m[ell]
                a_new[ell] = m[ell]
            else:
                # Tie: pick winner at random
                winner = k if self.rng.random() < 0.5 else ell
                a_new[k] = m[winner]
                a_new[ell] = m[winner]
        return a_new

    def step(
        self,
        pi_hist: np.ndarray,
        re_fcast_hist: np.ndarray | None = None,
        common_fcast_hist: np.ndarray | None = None,
        lambda_t: float | None = None,
    ) -> float:
        """
        One full SL period: mutate → fitness → tournament → update a_j.
        Returns new aggregate phi_t.
        Timing (WP): uses inflation history through t-1 before period-t macro.
        """
        m = self.mutate(lambda_t=lambda_t)
        F = self.fitness(
            m, pi_hist,
            re_fcast_hist=re_fcast_hist,
            common_fcast_hist=common_fcast_hist,
        )
        self.a = self.tournament(m, F)
        return self.phi

    def reset(self, a0: float | np.ndarray = 0.0) -> None:
        """Reset beliefs (default: perfect anchoring a_j=0 ⇒ FIRE nesting)."""
        if np.isscalar(a0):
            self.a[:] = float(a0)
        else:
            a0 = np.asarray(a0, dtype=float)
            if a0.shape != self.a.shape:
                raise ValueError("a0 shape must equal (J,)")
            self.a[:] = a0


def demo(seed: int = 0, T: int = 40) -> None:
    """Smoke test: SL on an exogenous inflation path with richer common fcast."""
    rng = np.random.default_rng(seed)
    sl = SocialLearning(SLParams(J=100), rng=rng)
    pi = np.concatenate(
        [np.linspace(0.02, -0.01, T // 2), np.linspace(-0.01, 0.03, T - T // 2)]
    )
    # Toy common component: A_pi*i_lag + D_pi*u with synthetic paths
    i_lag = np.zeros(T)
    u = 0.01 * np.exp(-0.1 * np.arange(T))
    common = build_plm_common_fcast(i_lag, A_pi=0.1, u=u, D_pi=0.8)
    phis = []
    for t in range(T):
        phi = sl.step(pi[:t], common_fcast_hist=common[:t])
        phis.append(phi)
    print(
        f"SL demo: phi_0={phis[0]:.6f}, phi_final={phis[-1]:.6f}, "
        f"mean|a|={np.mean(np.abs(sl.a)):.6f}"
    )


if __name__ == "__main__":
    demo()
