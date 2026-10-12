"""Gauss–Hermite nodes for expectations under a standard normal shock."""

from __future__ import annotations

import numpy as np


def gauss_hermite_normal(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for :math:`\\mathbb{E}[g(\\varepsilon)]`, :math:`\\varepsilon \\sim N(0, 1)`.

    ``numpy.polynomial.hermite.hermgauss`` returns physicist nodes for the
    weight :math:`e^{-x^2}`. The change of variables :math:`\\varepsilon = \\sqrt{2} x`
    converts them to a standard normal expectation.
    """
    if n < 1:
        raise ValueError("n must be positive")
    nodes_x, weights_x = np.polynomial.hermite.hermgauss(int(n))
    nodes = np.sqrt(2.0) * nodes_x
    weights = weights_x / np.sqrt(np.pi)
    return nodes.astype(np.float64), weights.astype(np.float64)
