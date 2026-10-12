"""Log-linear algebra and the endogenous-grid benchmark."""

from __future__ import annotations

import numpy as np

from dleq.quadrature import gauss_hermite_normal
from examples.growth_solver import (
    GrowthParams,
    closed_form_controls,
    egm_controls,
    euler_residual,
    linear_controls,
    loglinear_coefficients,
    solve_egm,
    stationary_variance,
    steady_state,
)


def test_ar1_variance_matches_the_scalar_formula():
    rho = 0.95
    sigma = 0.02
    variance = stationary_variance(np.array([[rho]]), np.array([[sigma ** 2]]))
    assert abs(variance[0, 0] - sigma ** 2 / (1.0 - rho ** 2)) < 1e-12


def test_deterministic_steady_state_satisfies_the_euler_equation():
    params = GrowthParams()
    ss = steady_state(params)
    gross = 1.0 - params.delta + params.alpha * ss["k"] ** (params.alpha - 1.0)
    assert abs(params.beta * gross - 1.0) < 1e-12
    assert abs(ss["c"] + params.delta * ss["k"] - ss["y"]) < 1e-12


def test_loglinear_policy_is_first_order_accurate_when_shocks_vanish():
    params = GrowthParams(sigma_z=0.0)
    ss = steady_state(params)
    coef = loglinear_coefficients(params, ss)
    nodes, weights = gauss_hermite_normal(5)

    def policy(capital, productivity):
        return linear_controls(ss, coef, capital, productivity)

    def residual(log_k, log_z):
        capital = np.array([ss["k"] * np.exp(log_k)])
        productivity = np.array([np.exp(log_z)])
        return float(euler_residual(params, capital, productivity, policy, nodes, weights)[0])

    assert abs(residual(0.0, 0.0)) < 1e-10
    capital_step = residual(1e-3, 0.0)
    productivity_step = residual(0.0, 1e-3)
    # A wrong slope would leave an O(10^{-3}) residual. The linear solution is O(eps^2).
    assert abs(capital_step) < 5e-6
    assert abs(productivity_step) < 5e-6
    assert abs(residual(1e-4, 0.0)) < abs(capital_step) / 20.0


def test_endogenous_grid_recovers_brock_mirman():
    params = GrowthParams(beta=0.96, delta=1.0, sigma=1.0, rho=0.9, sigma_z=0.02)
    solution = solve_egm(
        params,
        n_capital=61,
        n_productivity=11,
        n_hermite=5,
        k_min_factor=0.45,
        k_max_factor=2.2,
        n_std=3.0,
        init_share=0.5,
        tol=1e-10,
        max_iter=80,
    )
    assert solution["converged"]
    ss = solution["ss"]
    capital = ss["k"] * np.linspace(0.7, 1.6, 12)
    productivity = np.exp(np.linspace(-0.04, 0.04, 7))
    kk, zz = np.meshgrid(capital, productivity, indexing="ij")
    _, k_egm = egm_controls(solution, kk, zz)
    _, k_true = closed_form_controls(params, kk, zz)
    relative = np.max(np.abs(k_egm - k_true) / np.abs(k_true))
    assert relative < 1e-4
