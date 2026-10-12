"""Output transforms keep controls inside the bounds they were given."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dleq.networks import MLPPolicy, inverse_output_transform
from dleq.simulate import simulate_forward


def test_sigmoid_and_softplus_respect_bounds():
    key = jax.random.PRNGKey(0)
    sigmoid = MLPPolicy(
        input_dim=3,
        output_dim=2,
        width=8,
        depth=2,
        output_transform="sigmoid",
        output_lower=(0.1, -0.5),
        output_upper=(0.9, 0.25),
    )
    softplus = MLPPolicy(
        input_dim=3,
        output_dim=1,
        width=8,
        depth=1,
        output_transform="softplus",
        output_lower=0.3,
    )
    key, k1, k2, k3 = jax.random.split(key, 4)
    inputs = jax.random.normal(k1, (64, 3), dtype=jnp.float64)
    bounded = np.asarray(sigmoid.apply(sigmoid.init(k2), inputs))
    positive = np.asarray(softplus.apply(softplus.init(k3), inputs))
    assert np.all(bounded[:, 0] > 0.1) and np.all(bounded[:, 0] < 0.9)
    assert np.all(bounded[:, 1] > -0.5) and np.all(bounded[:, 1] < 0.25)
    assert np.all(positive[:, 0] > 0.3)


def test_inverse_transform_round_trip():
    for target in (0.2, 0.6544, 0.067):
        raw = inverse_output_transform("sigmoid", 1e-4, 1.0 - 1e-4, target)
        unit = 1.0 / (1.0 + np.exp(-raw))
        restored = 1e-4 + (1.0 - 2e-4) * unit
        assert abs(restored - target) < 1e-10
    raw = inverse_output_transform("softplus", 0.5, 0.0, 1.25)
    assert abs(0.5 + np.log1p(np.exp(raw)) - 1.25) < 1e-10


def test_init_is_deterministic():
    policy = MLPPolicy(input_dim=2, output_dim=1, width=4, depth=1)
    key = jax.random.PRNGKey(3)
    first = policy.init(key)
    second = policy.init(key)
    for left, right in zip(first, second, strict=True):
        assert np.allclose(left["w"], right["w"])
        assert np.allclose(left["b"], right["b"])


def test_simulate_forward_applies_the_transition():
    def transition(params, states, shocks):
        del params
        return states + shocks

    def shocks(key, n_agents):
        return jax.random.normal(key, (n_agents, 1), dtype=jnp.float64)

    states = jnp.zeros((5, 1), dtype=jnp.float64)
    final, path = simulate_forward(
        {},
        states,
        transition,
        shocks,
        jax.random.PRNGKey(1),
        n_steps=3,
        save_path=True,
    )
    assert final.shape == (5, 1)
    assert path.shape == (3, 5, 1)
    assert not np.allclose(final, 0.0)
