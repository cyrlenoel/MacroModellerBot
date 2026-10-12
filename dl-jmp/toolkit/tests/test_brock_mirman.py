"""Closed-form Brock–Mirman residual, and a short training run for CI."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import dleq
from examples.brock_mirman import RunConfig, _operators, _simulate_closed_form, run
from examples.growth_solver import GrowthParams, steady_state


def test_constant_true_share_solves_the_euler_equation():
    model = GrowthParams(alpha=0.36, beta=0.96, delta=1.0, sigma=1.0, rho=0.95, sigma_z=0.02)
    ss = steady_state(model)
    true_share = 1.0 - model.alpha * model.beta
    raw = dleq.inverse_output_transform("sigmoid", 1e-4, 1.0 - 1e-4, true_share)
    policy = dleq.MLPPolicy(
        input_dim=2,
        output_dim=1,
        width=8,
        depth=1,
        output_transform="sigmoid",
        output_lower=1e-4,
        output_upper=1.0 - 1e-4,
        final_bias=raw,
        last_weight_scale=0.0,
    )
    theta = policy.init(jax.random.PRNGKey(0))
    nodes, weights = dleq.gauss_hermite_normal(7)
    std = model.sigma_z / np.sqrt(1.0 - model.rho ** 2)
    bundle = _operators(
        policy, model, ss, 0.15, std,
        0.3 * ss["k"], 3.0 * ss["k"], -6.0 * std, 6.0 * std,
        nodes, weights, 0.4, 3.0,
    )
    states = _simulate_closed_form(jax.random.PRNGKey(4), 512, 80, model, ss)
    euler = np.asarray(bundle["euler_quad"](theta, states))
    assert np.max(np.abs(euler)) < 1e-8


def test_short_training_reaches_a_loose_tolerance():
    summary = run(
        RunConfig(
            n_steps=40,
            batch_size=512,
            width=16,
            depth=2,
            last_weight_scale=0.01,
            lr=1e-3,
            lr_decay_steps=40,
            lr_end=3e-4,
            warmup_steps=5,
            n_pairs=2,
            lbfgs_steps=20,
            lbfgs_pairs=4,
            polish_n=1024,
            burn_in=5,
            mix_fraction=0.0,
            grad_clip=5.0,
            log_every=0,
            score_every=40,
            n_hermite=7,
            eval_n=1024,
            eval_burn=80,
            seed=0,
            quiet=True,
            save=False,
        )
    )
    assert summary["evaluation"]["initial_policy"]["max_rel"] > 0.1
    assert summary["evaluation"]["policy"]["max_rel"] < 1e-2
    assert summary["evaluation"]["closed_form_euler"]["max_abs"] < 1e-8
    # The learned network must move. A no-op trainer would still be at the 0.5 share.
    assert summary["evaluation"]["policy"]["max_rel"] < summary["evaluation"]["initial_policy"]["max_rel"] / 5.0
