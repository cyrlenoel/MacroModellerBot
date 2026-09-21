"""Perfect-foresight solver for the linear twin, plus anticipated sterilisation.

The Bank Rate peg in scenario (a) is not a second Taylor rule. The structural
equation stays

    i_t = ρ_i i_{t-1} + (1-ρ_i)(φ_π π_t + φ_y^{qp} Y_t) + ν_t + ε^{ster}_t

and {ε^{ster}_t} is the anticipated path that enforces i_t = 0 for the first
H quarters. After H the residual is zero and the Taylor rule resumes, which
is what anchors inflation (a permanent peg would not).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.linear_model import LinearModel


@dataclass
class Path:
    """T x n simulation in model units, plus the sterilising residual if any."""

    y: np.ndarray  # (T, n)
    e_ster: np.ndarray  # (T,)
    T: int

    def series(self, model: LinearModel, name: str) -> np.ndarray:
        return self.y[:, model.ix[name]]


def _shock_matrix(model: LinearModel, shocks: dict[str, np.ndarray], T: int) -> np.ndarray:
    e = np.zeros((T, len(model.exo)))
    for name, series in shocks.items():
        arr = np.asarray(series, dtype=float)
        if arr.shape != (T,):
            raise ValueError(f"shock {name} must have length {T}, got {arr.shape}")
        e[:, model.ix_exo[name]] = arr
    return e


def solve_path(model: LinearModel, shocks: dict[str, np.ndarray], T: int | None = None) -> Path:
    """Unconstrained perfect foresight. Terminal y_T = 0, initial y_{-1} = 0."""
    T = int(T if T is not None else model.cal.T)
    e = _shock_matrix(model, shocks, T)
    y, _ = _solve_stacked(model, e, ster_unknown_until=0)
    return Path(y=y, e_ster=e[:, model.ix_exo["e_ster"]].copy(), T=T)


def solve_sterilised(
    model: LinearModel,
    shocks: dict[str, np.ndarray],
    H: int | None = None,
    T: int | None = None,
) -> Path:
    """Same shock paths, with ε^{ster} chosen so i_t = 0 for t = 0..H-1.

    ``shocks['e_ster']`` is ignored; the residual is an unknown.
    """
    T = int(T if T is not None else model.cal.T)
    H = int(H if H is not None else model.cal.H_peg)
    if not (1 <= H < T):
        raise ValueError("peg horizon H must lie strictly inside the simulation")
    e = _shock_matrix(model, shocks, T)
    e[:, model.ix_exo["e_ster"]] = 0.0
    y, ster = _solve_stacked(model, e, ster_unknown_until=H)
    e[:, model.ix_exo["e_ster"]] = ster
    return Path(y=y, e_ster=ster, T=T)


def _solve_stacked(
    model: LinearModel, e: np.ndarray, ster_unknown_until: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the linear system. If ``ster_unknown_until`` = H > 0, append the peg."""
    n = model.n
    T = e.shape[0]
    H = ster_unknown_until
    n_unk = T * n + H
    ix_i = model.ix["i"]
    ix_ster = model.ix_exo["e_ster"]
    Me_ster = model.Me[:, ix_ster]

    # Structural equations + H peg constraints.
    n_eq = T * n + H
    if n_eq != n_unk:
        raise RuntimeError("stacked system is not square")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    rhs = np.zeros(n_eq)

    def place(eq_offset: int, var_offset: int, mat: np.ndarray) -> None:
        ii, jj = np.nonzero(mat)
        if ii.size == 0:
            return
        rows.append(ii + eq_offset)
        cols.append(jj + var_offset)
        data.append(mat[ii, jj])

    for t in range(T):
        r0 = t * n
        place(r0, r0, model.Mc)
        if t > 0:
            place(r0, (t - 1) * n, model.Ml)
        if t + 1 < T:
            place(r0, (t + 1) * n, model.Mp)
        # M_p y_{t+1} + M_c y_t + M_l y_{t-1} + M_e e_t = 0, with y_T = y_{-1} = 0.
        rhs[r0 : r0 + n] = -model.Me @ e[t]

    if H > 0:
        ii = np.flatnonzero(Me_ster)
        vv = Me_ster[ii]
        for t in range(H):
            rows.append(ii + t * n)
            cols.append(np.full(ii.size, T * n + t, dtype=int))
            data.append(vv)
            # Peg: i_t = 0.
            rows.append(np.array([T * n + t], dtype=int))
            cols.append(np.array([t * n + ix_i], dtype=int))
            data.append(np.array([1.0]))

    A = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_eq, n_unk),
    ).tocsc()
    try:
        sol = spla.spsolve(A, rhs)
    except RuntimeError as exc:
        raise RuntimeError(
            "perfect-foresight system is singular or failed to factorise; "
            "the Taylor-rule model may be indeterminate at this calibration"
        ) from exc
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("perfect-foresight solution has non-finite entries")

    y = sol[: T * n].reshape(T, n)
    ster = np.zeros(T)
    if H > 0:
        ster[:H] = sol[T * n :]
    return y, ster


def max_residual(model: LinearModel, path: Path, shocks: dict[str, np.ndarray]) -> float:
    """Max absolute structural residual, including the solved ε^{ster}."""
    T = path.T
    e = _shock_matrix(model, shocks, T)
    e[:, model.ix_exo["e_ster"]] = path.e_ster
    worst = 0.0
    y = path.y
    for t in range(T):
        y_l = y[t - 1] if t else np.zeros(model.n)
        y_p = y[t + 1] if t + 1 < T else np.zeros(model.n)
        res = model.Mp @ y_p + model.Mc @ y[t] + model.Ml @ y_l + model.Me @ e[t]
        worst = max(worst, float(np.max(np.abs(res))))
    return worst


def taylor_notional(model: LinearModel, path: Path) -> np.ndarray:
    """i^{TR}_t without ε^{ster}, along a simulated path. Model units."""
    cal = model.cal
    i = path.series(model, "i")
    pi = path.series(model, "pi")
    y = path.series(model, "y")
    nu = path.series(model, "nu")
    i_lag = np.zeros_like(i)
    i_lag[1:] = i[:-1]
    return (
        cal.rho_i * i_lag
        + (1.0 - cal.rho_i) * (cal.phi_pi * pi + cal.phi_y_qp * y)
        + nu
    )


@dataclass
class QZFactor:
    """Complex QZ factorisation of the companion pencil. Stable roots first."""

    S: np.ndarray
    TT: np.ndarray
    Q: np.ndarray
    Z: np.ndarray
    ns: int
    n: int


def factor_qz(model: LinearModel) -> QZFactor:
    """Saddle-path factorisation. Evolution roots are eigenvalues of (B, A).

    The companion is ``A w_{t+1} = B w_t + d_t`` with ``w = [y_t; y_{t-1}]``.
    ``ordqz(B, A, sort='iuc')`` places roots inside the unit circle first.
    Determinacy for this pencil requires exactly ``n`` stable roots.
    """
    n = model.n
    eye = np.eye(n)
    A = np.block([[model.Mp, np.zeros((n, n))], [np.zeros((n, n)), eye]])
    B = np.block([[-model.Mc, -model.Ml], [eye, np.zeros((n, n))]])
    S, TT, alpha, beta, Q, Z = scipy.linalg.ordqz(B, A, output="complex", sort="iuc")
    mu = np.array(
        [np.inf if abs(b) < 1e-12 else a / b for a, b in zip(alpha, beta)],
        dtype=complex,
    )
    ns = int(np.sum(np.abs(mu) < 1.0 - 1e-8))
    if ns != n:
        raise RuntimeError(
            f"Blanchard-Kahn failed: {ns} stable roots, {2 * n - ns} unstable, "
            f"need {n} of each for a unique bounded path"
        )
    return QZFactor(S=S, TT=TT, Q=Q, Z=Z, ns=ns, n=n)


def simulate_qz(qz: QZFactor, Me: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    """Exact bounded path for a finite shock sequence (zero after the last row).

    ``shocks`` has shape (T, n_exo). Anticipated future shocks are allowed:
    the unstable block is solved backward from a zero terminal condition,
    which is exact once the shock sequence has ended.
    """
    n = qz.n
    T = shocks.shape[0]
    ns = qz.ns
    nu = 2 * n - ns
    d = np.zeros((T, 2 * n), dtype=complex)
    d[:, :n] = -(shocks @ Me.T)
    dhat = d @ qz.Q.conj().T.T
    S22 = qz.S[ns:, ns:]
    T22 = qz.TT[ns:, ns:]
    z2 = np.zeros((T + 1, nu), dtype=complex)
    for t in range(T - 1, -1, -1):
        z2[t] = np.linalg.solve(S22, T22 @ z2[t + 1] - dhat[t, ns:])
    Zu = qz.Z[:, ns:].conj().T
    y = np.zeros((T, n), dtype=complex)
    y_lag = np.zeros(n, dtype=complex)
    for t in range(T):
        y[t] = np.linalg.solve(Zu[:, :n], z2[t] - Zu[:, n:] @ y_lag)
        y_lag = y[t]
    imag = float(np.max(np.abs(np.imag(y)))) if y.size else 0.0
    if imag > 1e-7:
        raise RuntimeError(f"QZ path has a non-trivial imaginary part ({imag:.2e})")
    return np.real(y)


def solve_qz(
    model: LinearModel, shocks: dict[str, np.ndarray], T: int | None = None
) -> Path:
    """Bounded perfect-foresight path on the stable manifold."""
    T = int(T if T is not None else model.cal.irf_horizon)
    e = _shock_matrix(model, shocks, T)
    y = simulate_qz(factor_qz(model), model.Me, e)
    return Path(y=y, e_ster=e[:, model.ix_exo["e_ster"]].copy(), T=T)


def solve_sterilised_qz(
    model: LinearModel,
    shocks: dict[str, np.ndarray],
    H: int | None = None,
    T: int | None = None,
) -> Path:
    """Anticipated ε^{ster} so that i_t = 0 for t = 0..H-1, on the stable manifold.

    The Taylor equation is unchanged. Columns of the map from the sterilising
    residual to Bank Rate are unit anticipated impulses; the residual solves
    that map. Because the model is linear, one factorisation is enough.
    """
    H = int(H if H is not None else model.cal.H_peg)
    T = int(T if T is not None else max(model.cal.irf_horizon, H))
    if T < H:
        raise ValueError("simulation length must cover the peg")
    e = _shock_matrix(model, shocks, T)
    e[:, model.ix_exo["e_ster"]] = 0.0
    qz = factor_qz(model)
    base = simulate_qz(qz, model.Me, e)
    ix_i = model.ix["i"]
    ix_s = model.ix_exo["e_ster"]
    jac = np.zeros((H, H))
    for j in range(H):
        ej = np.zeros_like(e)
        ej[j, ix_s] = 1.0
        jac[:, j] = simulate_qz(qz, model.Me, ej)[:H, ix_i]
    ster_h = np.linalg.solve(jac, -base[:H, ix_i])
    e[:H, ix_s] = ster_h
    y = simulate_qz(qz, model.Me, e)
    ster = np.zeros(T)
    ster[:H] = ster_h
    return Path(y=y, e_ster=ster, T=T)


def eigenvalue_summary(model: LinearModel) -> dict[str, float]:
    """Evolution roots of the companion. A unique bounded path needs n stable roots.

    Infinite roots are the singular directions of variables that are not led.
    ``n_stable == n`` is the Blanchard–Kahn condition used by ``factor_qz``.
    """
    n = model.n
    eye = np.eye(n)
    A = np.block([[model.Mp, np.zeros((n, n))], [np.zeros((n, n)), eye]])
    B = np.block([[-model.Mc, -model.Ml], [eye, np.zeros((n, n))]])
    _S, _T, alpha, beta, _Q, _Z = scipy.linalg.ordqz(B, A, output="complex", sort="iuc")
    mu = np.array(
        [np.inf if abs(b) < 1e-12 else a / b for a, b in zip(alpha, beta)],
        dtype=complex,
    )
    mod = np.abs(mu)
    finite = mod[np.isfinite(mod)]
    return {
        "n": float(n),
        "n_stable": float(np.sum(mod < 1.0 - 1e-8)),
        "n_unstable_finite": float(np.sum((finite > 1.0 + 1e-8))),
        "n_infinite": float(np.isinf(mod).sum()),
        "max_stable_mod": float(np.max(finite[finite < 1.0])) if np.any(finite < 1) else float("nan"),
    }
