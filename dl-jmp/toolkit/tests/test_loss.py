"""The all-in-one product is unbiased for the squared conditional expectation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dleq.loss import all_in_one_loss


def test_product_of_two_draws_matches_squared_mean():
    key = jax.random.PRNGKey(7)
    # One state, many Monte Carlo replicates of f = a + b * epsilon.
    a = 0.4
    b = 1.5
    n = 400_000
    key_a, key_b = jax.random.split(key)
    eps_a = jax.random.normal(key_a, (n,), dtype=jnp.float64)
    eps_b = jax.random.normal(key_b, (n,), dtype=jnp.float64)
    residual_a = a + b * eps_a
    residual_b = a + b * eps_b
    product = float(all_in_one_loss(residual_a, residual_b))
    biased = float(jnp.mean(residual_a ** 2))
    assert abs(product - a ** 2) < 0.01
    assert abs(biased - (a ** 2 + b ** 2)) < 0.02
    # The squared single draw is larger by the residual variance.
    assert biased > product + 1.0


def test_weights_scale_each_equation():
    residual_a = jnp.array([[1.0, 2.0], [1.0, 2.0]])
    residual_b = jnp.array([[3.0, 4.0], [3.0, 4.0]])
    # Each row contributes 1*3 + 2*4 = 11, mean 11.
    assert float(all_in_one_loss(residual_a, residual_b)) == 11.0
    weighted = float(all_in_one_loss(residual_a, residual_b, weights=jnp.array([1.0, 0.0])))
    assert weighted == 3.0


def test_gauss_hermite_matches_normal_moments():
    from dleq.quadrature import gauss_hermite_normal

    nodes, weights = gauss_hermite_normal(11)
    assert abs(np.sum(weights) - 1.0) < 1e-12
    assert abs(np.sum(weights * nodes)) < 1e-12
    assert abs(np.sum(weights * nodes ** 2) - 1.0) < 1e-12
    assert abs(np.sum(weights * nodes ** 4) - 3.0) < 1e-10
