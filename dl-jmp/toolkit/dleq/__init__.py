"""Deep equilibrium nets for dynamic economic models.

Importing this package selects the CPU JAX platform unless ``JAX_PLATFORMS``
is already set, and turns on 64-bit arithmetic. Both choices are deliberate:
the toolkit is a numerical solver, and float32 roundoff is visible in Euler
errors long before it is visible in a policy plot.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

from dleq.loss import all_in_one_loss
from dleq.metrics import log10_abs_stats, policy_errors
from dleq.networks import MLPPolicy, inverse_output_transform
from dleq.quadrature import gauss_hermite_normal
from dleq.simulate import mix_states, simulate_forward
from dleq.train import TrainConfig, TrainResult, train
from dleq.verify import format_verification, verify_against_linear

__version__ = "0.1.0"

__all__ = [
    "MLPPolicy",
    "TrainConfig",
    "TrainResult",
    "all_in_one_loss",
    "format_verification",
    "gauss_hermite_normal",
    "inverse_output_transform",
    "log10_abs_stats",
    "mix_states",
    "policy_errors",
    "simulate_forward",
    "train",
    "verify_against_linear",
]
