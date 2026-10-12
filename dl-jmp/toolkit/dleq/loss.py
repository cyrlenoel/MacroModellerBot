"""All-in-one squared-expectation loss (Maliar, Maliar and Winant, 2021)."""

from __future__ import annotations

import jax.numpy as jnp


def all_in_one_loss(
    residuals_1: jnp.ndarray,
    residuals_2: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Unbiased estimator of the squared conditional Euler residual.

    Let :math:`f(s, \\varepsilon)` be the vector of equilibrium residuals at state
    :math:`s` under one draw of next period's shock. A solution satisfies
    :math:`\\mathbb{E}[f(s, \\varepsilon) \\mid s] = 0`. The training target is

    .. math::

        \\mathbb{E}\\Big[\\sum_j w_j \\big(\\mathbb{E}[f_j \\mid s]\\big)^2\\Big].

    Squaring a single Monte Carlo draw estimates
    :math:`\\mathbb{E}[f_j^2 \\mid s] = (\\mathbb{E}[f_j \\mid s])^2 + \\mathrm{Var}(f_j \\mid s)`
    and is biased whenever the residual is noisy. Two independent draws
    :math:`\\varepsilon_1` and :math:`\\varepsilon_2` restore the square by independence:

    .. math::

        \\mathbb{E}[f_j(s, \\varepsilon_1) f_j(s, \\varepsilon_2) \\mid s]
        = \\big(\\mathbb{E}[f_j \\mid s]\\big)^2.

    ``residuals_1`` and ``residuals_2`` are those two evaluations. They have
    shape ``(batch,)`` or ``(batch, n_equations)`` and must be computed at the
    same states. ``weights`` is an optional ``(n_equations,)`` vector; the
    default weights every equation equally with weight one.

    The batch mean of the weighted products is unbiased for the expectation
    over states as well, which is the all-in-one operator: one Monte Carlo
    object covers both the shock and the state distribution.
    """
    left = _as_columns(residuals_1)
    right = _as_columns(residuals_2)
    if left.shape != right.shape:
        raise ValueError(f"residual shapes differ: {left.shape} vs {right.shape}")
    product = left * right
    if weights is not None:
        weight_row = jnp.asarray(weights, dtype=product.dtype).reshape(1, -1)
        if weight_row.shape[-1] != product.shape[-1]:
            raise ValueError("weights must have one entry per equilibrium condition")
        product = product * weight_row
    return jnp.mean(jnp.sum(product, axis=-1))


def _as_columns(residuals: jnp.ndarray) -> jnp.ndarray:
    if residuals.ndim == 1:
        return residuals[:, None]
    if residuals.ndim != 2:
        raise ValueError(f"residuals must have ndim 1 or 2, got shape {residuals.shape}")
    return residuals
