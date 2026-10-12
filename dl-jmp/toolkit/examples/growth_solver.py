"""Closed-form, log-linear and endogenous-grid solutions of the growth model.

The stochastic growth model is

.. math::

    \\max \\mathbb{E}_0 \\sum_t \\beta^t \\frac{c_t^{1-\\sigma}}{1-\\sigma}
    \\quad\\text{s.t.}\\quad
    c_t + k_{t+1} = (1-\\delta) k_t + z_t k_t^{\\alpha},

with :math:`\\log z_{t+1} = \\rho \\log z_t + \\sigma_z \\varepsilon_{t+1}` and
:math:`\\varepsilon_{t+1} \\sim N(0,1)`. Log utility is the limit :math:`\\sigma = 1`.

Full depreciation and log utility (:math:`\\delta = 1`, :math:`\\sigma = 1`) are
the Brock–Mirman case, whose saving rate is :math:`\\alpha\\beta`. The
log-linear policy is the first-order perturbation at the deterministic steady
state. The global benchmark is time iteration on the Euler equation, implemented
with Carroll's endogenous grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dleq.quadrature import gauss_hermite_normal


@dataclass(frozen=True)
class GrowthParams:
    alpha: float = 0.36
    beta: float = 0.99
    delta: float = 0.025
    sigma: float = 2.0
    rho: float = 0.95
    sigma_z: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if not 0.0 < self.delta <= 1.0:
            raise ValueError("delta must lie in (0, 1]")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if not -0.999 < self.rho < 0.999:
            raise ValueError("rho must lie in (-0.999, 0.999)")
        if self.sigma_z < 0.0:
            raise ValueError("sigma_z must be non-negative")


def steady_state(params: GrowthParams) -> dict[str, float]:
    """Deterministic steady state at :math:`z = 1`.

    :math:`1 = \\beta(1 - \\delta + \\alpha k^{\\alpha-1})` and
    :math:`c = k^{\\alpha} - \\delta k`.
    """
    marginal_product = 1.0 / params.beta - (1.0 - params.delta)
    capital = (params.alpha / marginal_product) ** (1.0 / (1.0 - params.alpha))
    output = capital ** params.alpha
    consumption = output - params.delta * capital
    if consumption <= 0.0:
        raise ValueError("steady-state consumption is not positive")
    return {
        "k": float(capital),
        "y": float(output),
        "c": float(consumption),
        "z": 1.0,
        "mpk": float(marginal_product),
        "R": float(1.0 / params.beta),
        "cash": float((1.0 - params.delta) * capital + output),
    }


def is_brock_mirman(params: GrowthParams) -> bool:
    return abs(params.delta - 1.0) < 1e-12 and abs(params.sigma - 1.0) < 1e-12


def closed_form_controls(params: GrowthParams, capital, productivity):
    """Brock–Mirman policy :math:`k' = \\alpha\\beta z k^{\\alpha}` and the implied consumption.

    Raises if the parameters are not log utility with full depreciation, because
    that is the only case in which this formula is the equilibrium policy.
    """
    if not is_brock_mirman(params):
        raise ValueError("closed form requires log utility and full depreciation")
    capital = np.asarray(capital, dtype=np.float64)
    productivity = np.asarray(productivity, dtype=np.float64)
    output = productivity * capital ** params.alpha
    capital_next = params.alpha * params.beta * output
    consumption = output - capital_next
    return consumption, capital_next


def stationary_variance(transition: np.ndarray, shock_covariance: np.ndarray) -> np.ndarray:
    """Solve :math:`V = A V A^\\top + S` for the stationary covariance ``V``."""
    matrix = np.asarray(transition, dtype=np.float64)
    shock = np.asarray(shock_covariance, dtype=np.float64)
    dimension = matrix.shape[0]
    identity = np.eye(dimension * dimension)
    kronecker = np.kron(matrix, matrix)
    vector = np.linalg.solve(identity - kronecker, shock.ravel(order="F"))
    return vector.reshape((dimension, dimension), order="F")


def loglinear_coefficients(params: GrowthParams, ss: dict[str, float] | None = None) -> dict[str, float]:
    """First-order coefficients in log deviations from the deterministic steady state.

    .. math::

        \\widehat{k}' = a \\widehat{k} + b \\widehat{z}, \\qquad
        \\widehat{c} = c_k \\widehat{k} + c_z \\widehat{z}.

    ``a`` is the stable root of the Riccati equation implied by the log-linear
    Euler equation and the log-linear resource constraint. The function checks
    that the returned coefficients plug back into those two linear equations.
    """
    if ss is None:
        ss = steady_state(params)
    alpha = params.alpha
    sigma = params.sigma
    rho = params.rho
    capital = ss["k"]
    consumption = ss["c"]
    output = ss["y"]
    gross = ss["R"]
    marginal = ss["mpk"]
    resource_k = (1.0 - params.delta) * capital + alpha * output
    phi_k = (alpha - 1.0) * marginal / gross
    phi_z = marginal / gross

    quad_a = sigma * capital
    quad_b = -sigma * capital - sigma * resource_k + phi_k * consumption
    quad_c = sigma * resource_k
    discriminant = quad_b ** 2 - 4.0 * quad_a * quad_c
    if discriminant < 0.0:
        raise RuntimeError(f"Riccati discriminant is negative ({discriminant})")
    root = np.sqrt(discriminant)
    roots = (
        (-quad_b + root) / (2.0 * quad_a),
        (-quad_b - root) / (2.0 * quad_a),
    )
    stable = [float(value) for value in roots if abs(value) < 1.0 - 1e-8]
    if len(stable) != 1:
        raise RuntimeError(f"expected one stable capital root, found {roots}")
    persistence = stable[0]
    consumption_k = (resource_k - capital * persistence) / consumption
    numerator = sigma * output * (1.0 - rho) + phi_z * rho * consumption
    denominator = sigma * capital * (1.0 - rho) + consumption * (sigma * consumption_k - phi_k)
    if abs(denominator) < 1e-14:
        raise RuntimeError("TFP loading denominator vanished")
    productivity_k = numerator / denominator
    consumption_z = (output - capital * productivity_k) / consumption
    coefficients = {
        "a": float(persistence),
        "b": float(productivity_k),
        "ck": float(consumption_k),
        "cz": float(consumption_z),
        "phi_k": float(phi_k),
        "phi_z": float(phi_z),
        "resource_k": float(resource_k),
    }
    mismatch = linearised_residuals(params, ss, coefficients)
    worst = max(abs(value) for value in mismatch.values())
    if worst > 1e-8:
        raise RuntimeError(f"log-linear coefficients do not satisfy the linear system: {mismatch}")
    return coefficients


def linearised_residuals(params: GrowthParams, ss: dict[str, float], coef: dict[str, float]) -> dict[str, float]:
    """Residuals of the log-linear resource constraint and Euler equation.

    Both residuals are written as coefficients on :math:`\\widehat{k}` and
    :math:`\\widehat{z}` and must be numerically zero at the solution. This is
    an algebraic check. Accuracy against the nonlinear model is a separate test.
    """
    sigma = params.sigma
    rho = params.rho
    capital = ss["k"]
    consumption = ss["c"]
    output = ss["y"]
    resource_k = coef["resource_k"]
    persistence = coef["a"]
    productivity_k = coef["b"]
    consumption_k = coef["ck"]
    consumption_z = coef["cz"]
    phi_k = coef["phi_k"]
    phi_z = coef["phi_z"]
    euler_k_lhs = -sigma * consumption_k
    euler_k_rhs = persistence * (-sigma * consumption_k + phi_k)
    euler_z_lhs = -sigma * consumption_z
    euler_z_rhs = (
        -sigma * (consumption_k * productivity_k + consumption_z * rho)
        + phi_k * productivity_k
        + phi_z * rho
    )
    return {
        "resource_k": float(consumption * consumption_k + capital * persistence - resource_k),
        "resource_z": float(consumption * consumption_z + capital * productivity_k - output),
        "euler_k": float(euler_k_lhs - euler_k_rhs),
        "euler_z": float(euler_z_lhs - euler_z_rhs),
    }


def linear_controls(ss: dict[str, float], coef: dict[str, float], capital, productivity):
    """Levels implied by the log-linear policy. Both controls stay positive."""
    capital = np.asarray(capital, dtype=np.float64)
    productivity = np.asarray(productivity, dtype=np.float64)
    log_capital = np.log(capital / ss["k"])
    log_productivity = np.log(productivity / ss["z"])
    capital_next = ss["k"] * np.exp(coef["a"] * log_capital + coef["b"] * log_productivity)
    consumption = ss["c"] * np.exp(coef["ck"] * log_capital + coef["cz"] * log_productivity)
    return consumption, capital_next


def euler_residual(params: GrowthParams, capital, productivity, policy, nodes: np.ndarray, weights: np.ndarray):
    """Nonlinear Euler residual :math:`\\beta\\mathbb{E}[(c'/c)^{-\\sigma} R'] - 1`.

    ``policy(k, z)`` returns ``(consumption, capital_next)``. The expectation
    uses the supplied Gauss–Hermite rule for the innovation to :math:`\\log z`.
    """
    capital = np.asarray(capital, dtype=np.float64)
    productivity = np.asarray(productivity, dtype=np.float64)
    consumption, capital_next = policy(capital, productivity)
    consumption = np.asarray(consumption, dtype=np.float64)
    capital_next = np.asarray(capital_next, dtype=np.float64)
    expectation = np.zeros(np.broadcast(capital, productivity).shape, dtype=np.float64)
    log_productivity = np.log(productivity)
    for node, weight in zip(nodes, weights, strict=True):
        productivity_next = np.exp(params.rho * log_productivity + params.sigma_z * node)
        consumption_next, _ = policy(capital_next, productivity_next)
        gross = _gross_return(params, capital_next, productivity_next)
        marginal = (np.asarray(consumption_next) / consumption) ** (-params.sigma) * gross
        expectation = expectation + weight * marginal
    return params.beta * expectation - 1.0


def solve_egm(
    params: GrowthParams,
    *,
    n_capital: int = 181,
    n_productivity: int = 25,
    n_hermite: int = 9,
    k_min_factor: float = 0.65,
    k_max_factor: float = 1.45,
    n_std: float = 4.0,
    max_iter: int = 400,
    tol: float = 1e-10,
    init_share: float | None = None,
) -> dict:
    """Euler-equation time iteration on an endogenous capital grid.

    The unknown is consumption on a fixed tensor of :math:`(k, z)`. Each
    iteration treats tomorrow's capital grid as the endogenous-grid nodes,
    inverts the Euler equation for today's consumption, backs out today's
    capital from the resource constraint, and interpolates back in
    :math:`(\\log k, \\log z)`. Interpolation is linear in logs with linear
    extrapolation, which is exact for the Brock–Mirman closed form.
    """
    ss = steady_state(params)
    if k_min_factor <= 0.0 or k_max_factor <= k_min_factor:
        raise ValueError("capital factors must satisfy 0 < min < max")
    log_capital = np.linspace(
        np.log(k_min_factor * ss["k"]),
        np.log(k_max_factor * ss["k"]),
        int(n_capital),
    )
    capital_grid = np.exp(log_capital)
    std_log_z = params.sigma_z / np.sqrt(1.0 - params.rho ** 2) if params.sigma_z > 0.0 else 0.0
    half_width = n_std * std_log_z if std_log_z > 0.0 else 0.0
    if n_productivity == 1 or half_width == 0.0:
        log_productivity = np.zeros(1)
    else:
        log_productivity = np.linspace(-half_width, half_width, int(n_productivity))
    productivity_grid = np.exp(log_productivity)
    nodes, weights = gauss_hermite_normal(n_hermite)
    share = ss["c"] / ss["cash"] if init_share is None else float(init_share)
    cash = _cash(params, capital_grid[:, None], productivity_grid[None, :])
    log_consumption = np.log(np.maximum(share * cash, 1e-12))

    converged = False
    difference = np.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        updated = np.empty_like(log_consumption)
        for index, log_z in enumerate(log_productivity):
            marginal_expectation = _expected_marginal(
                params, capital_grid, log_z, log_capital, log_productivity, log_consumption, nodes, weights
            )
            consumption_endo = np.maximum(marginal_expectation, 1e-16) ** (-1.0 / params.sigma)
            cash_endo = consumption_endo + capital_grid
            capital_endo = _invert_cash(params, cash_endo, productivity_grid[index])
            order = np.argsort(capital_endo)
            log_k_endo = np.log(np.maximum(capital_endo[order], 1e-16))
            log_c_endo = np.log(consumption_endo[order])
            log_k_endo, unique_index = np.unique(log_k_endo, return_index=True)
            log_c_endo = log_c_endo[unique_index]
            updated[:, index] = _interp_extrap(log_capital, log_k_endo, log_c_endo)
        difference = float(np.max(np.abs(np.exp(updated) - np.exp(log_consumption)) / np.exp(log_consumption)))
        log_consumption = updated
        if difference < tol:
            converged = True
            break

    solution = {
        "params": params,
        "ss": ss,
        "logk": log_capital,
        "logz": log_productivity,
        "k_grid": capital_grid,
        "z_grid": productivity_grid,
        "logc": log_consumption,
        "iterations": iterations,
        "difference": difference,
        "converged": converged,
        "nodes": nodes,
        "weights": weights,
        "init_share": share,
    }
    return solution


def egm_controls(solution: dict, capital, productivity):
    """Consumption and next capital from an endogenous-grid solution.

    Next capital uses the exact resource constraint, so the two controls are
    budget-consistent by construction.
    """
    capital = np.asarray(capital, dtype=np.float64)
    productivity = np.asarray(productivity, dtype=np.float64)
    consumption = np.exp(
        _bilinear(
            solution["logk"],
            solution["logz"],
            solution["logc"],
            np.log(capital),
            np.log(productivity),
        )
    )
    cash = _cash(solution["params"], capital, productivity)
    consumption = np.minimum(consumption, 0.999 * cash)
    capital_next = cash - consumption
    return consumption, capital_next


def _cash(params: GrowthParams, capital, productivity):
    return (1.0 - params.delta) * capital + productivity * np.power(capital, params.alpha)


def _gross_return(params: GrowthParams, capital_next, productivity_next):
    return 1.0 - params.delta + params.alpha * productivity_next * np.power(capital_next, params.alpha - 1.0)


def _expected_marginal(params, capital_grid, log_z, log_capital, log_productivity, log_consumption, nodes, weights):
    expectation = np.zeros_like(capital_grid)
    for node, weight in zip(nodes, weights, strict=True):
        log_z_next = params.rho * log_z + params.sigma_z * node
        log_c_next = _bilinear(log_capital, log_productivity, log_consumption, log_capital, log_z_next)
        consumption_next = np.exp(log_c_next)
        productivity_next = np.exp(log_z_next)
        gross = _gross_return(params, capital_grid, productivity_next)
        expectation = expectation + weight * np.power(consumption_next, -params.sigma) * gross
    return params.beta * expectation


def _invert_cash(params: GrowthParams, cash: np.ndarray, productivity: float) -> np.ndarray:
    cash = np.asarray(cash, dtype=np.float64)
    if abs(params.delta - 1.0) < 1e-14:
        return np.power(cash / productivity, 1.0 / params.alpha)
    capital = np.power(np.maximum(cash / max(productivity, 1e-12), 1e-10), 1.0 / params.alpha)
    for _ in range(30):
        residual = (1.0 - params.delta) * capital + productivity * np.power(capital, params.alpha) - cash
        slope = (1.0 - params.delta) + params.alpha * productivity * np.power(capital, params.alpha - 1.0)
        capital = np.maximum(capital - residual / slope, 1e-12)
    return capital


def _interp_extrap(x_query: np.ndarray, x_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """One-dimensional linear interpolation with linear extrapolation."""
    if x_grid.size < 2:
        return np.full_like(np.asarray(x_query, dtype=np.float64), float(values[0]), dtype=np.float64)
    interpolated = np.interp(x_query, x_grid, values)
    left = x_query < x_grid[0]
    right = x_query > x_grid[-1]
    if np.any(left):
        slope = (values[1] - values[0]) / (x_grid[1] - x_grid[0])
        interpolated = np.where(left, values[0] + slope * (x_query - x_grid[0]), interpolated)
    if np.any(right):
        slope = (values[-1] - values[-2]) / (x_grid[-1] - x_grid[-2])
        interpolated = np.where(right, values[-1] + slope * (x_query - x_grid[-1]), interpolated)
    return interpolated


def _bilinear(logk_grid, logz_grid, logc, logk_query, logz_query):
    """Interpolate ``logc[k, z]`` at query log-capital and log-productivity.

    Queries may be scalars or arrays. A scalar productivity returns a vector
    along capital; general broadcasting is done by raveling the queries.
    """
    logk_query = np.asarray(logk_query, dtype=np.float64)
    logz_query = np.asarray(logz_query, dtype=np.float64)
    query_shape = np.broadcast(logk_query, logz_query).shape
    logk_flat = np.broadcast_to(logk_query, query_shape).ravel()
    logz_flat = np.broadcast_to(logz_query, query_shape).ravel()
    if logz_grid.size == 1:
        column = _interp_extrap(logk_flat, logk_grid, logc[:, 0])
        return column.reshape(query_shape)
    gathered = np.column_stack(
        [_interp_extrap(logk_flat, logk_grid, logc[:, index]) for index in range(logz_grid.size)]
    )
    upper = np.searchsorted(logz_grid, logz_flat)
    upper = np.clip(upper, 1, logz_grid.size - 1)
    lower = upper - 1
    width = logz_grid[upper] - logz_grid[lower]
    weight = (logz_flat - logz_grid[lower]) / width
    row = np.arange(logk_flat.size)
    blended = (1.0 - weight) * gathered[row, lower] + weight * gathered[row, upper]
    return blended.reshape(query_shape)
