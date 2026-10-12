"""Small writers shared by the verification scripts."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import jax
import numpy as np
import optax


def environment_info() -> dict:
    import jaxlib

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "optax": optax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "x64": bool(jax.config.jax_enable_x64),
    }


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body]) + "\n"


def sci(value: float) -> str:
    return f"{value:.6e}"


def plain(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"
