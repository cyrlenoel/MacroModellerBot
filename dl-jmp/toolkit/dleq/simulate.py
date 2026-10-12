"""Ergodic-set simulation in the sense of Azinovic, Gaegauf and Scheidegger (2022)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

Params = list | dict
ShockFn = Callable[[jax.Array, int], jnp.ndarray]
TransitionFn = Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def simulate_forward(
    params: Params,
    states: jnp.ndarray,
    transition_fn: TransitionFn,
    shock_fn: ShockFn,
    key: jax.Array,
    n_steps: int,
    *,
    save_path: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Simulate a panel forward under the current policy.

    ``transition_fn(params, states, shocks)`` returns the next state of every
    agent. Parameters are held fixed and detached: the simulated cloud is
    training data, and the distribution is not differentiated. This is the
    ergodic-set update used by deep equilibrium nets. Call it between Adam
    steps, passing the latest parameters, and the panel tracks the distribution
    implied by the policy currently being trained.

    Returns the final cross-section and, if ``save_path`` is set, an array of
    shape ``(n_steps, n_agents, state_dim)`` containing the state after each step.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_steps == 0:
        return states, None

    frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
    if not save_path:
        def body(step_index: jnp.ndarray, carry: tuple[jax.Array, jnp.ndarray]):
            del step_index
            step_key, current = carry
            step_key, sub = jax.random.split(step_key)
            shocks = shock_fn(sub, current.shape[0])
            nxt = transition_fn(frozen, current, shocks)
            return step_key, nxt

        _, final = jax.lax.fori_loop(0, n_steps, body, (key, states))
        return jax.lax.stop_gradient(final), None

    def scan_body(current: jnp.ndarray, step_key: jax.Array):
        shocks = shock_fn(step_key, current.shape[0])
        nxt = transition_fn(frozen, current, shocks)
        return nxt, nxt

    keys = jax.random.split(key, n_steps)
    final, path = jax.lax.scan(scan_body, states, keys)
    return jax.lax.stop_gradient(final), jax.lax.stop_gradient(path)


def mix_states(
    states: jnp.ndarray,
    fresh: jnp.ndarray,
    fraction: float,
    key: jax.Array,
) -> jnp.ndarray:
    """Replace a fraction of a panel with freshly drawn states.

    The deep-equilibrium update simulates every agent. Mixing a few draws from
    a fixed domain back in stops a badly initialised policy from collapsing the
    training cloud onto a clipping boundary, and it is how a global comparison
    grid is kept inside the training distribution. ``fraction=0`` leaves the
    simulated panel unchanged.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    n_agents = int(states.shape[0])
    n_mix = int(round(fraction * n_agents))
    if n_mix <= 0:
        return states
    if fresh.shape[0] < n_mix:
        raise ValueError("fresh sample is smaller than the mix count")
    index = jax.random.choice(key, n_agents, shape=(n_mix,), replace=False)
    return states.at[index].set(fresh[:n_mix])
