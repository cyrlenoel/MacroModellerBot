"""Multilayer perceptron policies with bound-preserving output transforms."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


def inverse_output_transform(
    transform: str,
    lower: float,
    upper: float,
    target: float,
) -> float:
    """Raw network output that the chosen transform maps to ``target``.

    Useful for initialising the last-layer bias at a steady-state guess
    without putting that guess into the training loss.
    """
    if transform == "sigmoid":
        if not upper > lower:
            raise ValueError("sigmoid bounds must satisfy upper > lower")
        unit = (float(target) - float(lower)) / (float(upper) - float(lower))
        unit = float(np.clip(unit, 1e-8, 1.0 - 1e-8))
        return float(np.log(unit / (1.0 - unit)))
    if transform == "softplus":
        gap = float(target) - float(lower)
        if gap <= 0.0:
            raise ValueError("softplus target must be strictly above lower")
        return float(np.log(np.expm1(gap)))
    if transform == "none":
        return float(target)
    raise ValueError(f"unknown output transform {transform!r}")


def _as_bounds(values: float | Sequence[float], output_dim: int, name: str) -> tuple[float, ...]:
    array = np.broadcast_to(np.asarray(values, dtype=np.float64), (output_dim,))
    return tuple(float(v) for v in array)


def _glorot(key: jax.Array, fan_in: int, fan_out: int, dtype: jnp.dtype) -> jnp.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=dtype)


class MLPPolicy:
    """Feed-forward policy network.

    Hidden layers use a pointwise activation (``tanh`` by default). The last
    layer is affine. An output transform then enforces bounds:

    * ``sigmoid`` maps onto ``(lower, upper)``;
    * ``softplus`` maps onto ``(lower, infinity)``;
    * ``none`` leaves the raw output unchanged.

    Parameters are a list of ``{"w", "b"}`` dictionaries, one per affine layer,
    which is a pytree Optax can differentiate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        output_transform: str = "sigmoid",
        output_lower: float | Sequence[float] = 0.0,
        output_upper: float | Sequence[float] = 1.0,
        final_bias: float | Sequence[float] = 0.0,
        last_weight_scale: float = 1.0,
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if depth > 0 and width < 1:
            raise ValueError("width must be positive when depth > 0")
        if activation not in {"tanh", "silu", "relu"}:
            raise ValueError(f"unsupported activation {activation!r}")
        if output_transform not in {"sigmoid", "softplus", "none"}:
            raise ValueError(f"unsupported output transform {output_transform!r}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.activation = activation
        self.output_transform = output_transform
        self.output_lower = _as_bounds(output_lower, self.output_dim, "output_lower")
        self.output_upper = _as_bounds(output_upper, self.output_dim, "output_upper")
        self.final_bias = _as_bounds(final_bias, self.output_dim, "final_bias")
        self.last_weight_scale = float(last_weight_scale)
        if output_transform == "sigmoid":
            if any(hi <= lo for lo, hi in zip(self.output_lower, self.output_upper, strict=True)):
                raise ValueError("sigmoid requires output_upper > output_lower for every output")
        hidden = [self.width] * self.depth
        self.layer_sizes = [self.input_dim, *hidden, self.output_dim]

    def init(self, key: jax.Array) -> list[dict[str, jnp.ndarray]]:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        params: list[dict[str, jnp.ndarray]] = []
        n_layers = len(self.layer_sizes) - 1
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)):
            key, sub = jax.random.split(key)
            weight = _glorot(sub, n_in, n_out, dtype)
            if i == n_layers - 1:
                weight = weight * self.last_weight_scale
                bias = jnp.asarray(self.final_bias, dtype=dtype)
            else:
                bias = jnp.zeros((n_out,), dtype=dtype)
            params.append({"w": weight, "b": bias})
        return params

    def apply(self, params: Sequence[dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the policy. ``x`` has shape ``(n, input_dim)`` or ``(input_dim,)``."""
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")
        hidden = x
        for layer in params[:-1]:
            hidden = _activate(hidden @ layer["w"] + layer["b"], self.activation)
        raw = hidden @ params[-1]["w"] + params[-1]["b"]
        output = _transform(raw, self.output_transform, self.output_lower, self.output_upper)
        if squeeze:
            return output[0]
        return output


def _activate(x: jnp.ndarray, name: str) -> jnp.ndarray:
    if name == "tanh":
        return jnp.tanh(x)
    if name == "silu":
        return jax.nn.silu(x)
    if name == "relu":
        return jax.nn.relu(x)
    raise ValueError(name)


def _transform(
    raw: jnp.ndarray,
    name: str,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> jnp.ndarray:
    if name == "none":
        return raw
    lo = jnp.asarray(lower, dtype=raw.dtype)
    if name == "softplus":
        return lo + jax.nn.softplus(raw)
    if name == "sigmoid":
        hi = jnp.asarray(upper, dtype=raw.dtype)
        return lo + (hi - lo) * jax.nn.sigmoid(raw)
    raise ValueError(name)
