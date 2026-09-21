"""IRF figures. Scenario (a) panels carry the sterilisation device in the title."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.linear_model import LinearModel
from src.solve_pf import Path
from src.success import STERILISATION_DEVICE


def _bp(path: Path, model: LinearModel, name: str, horizon: int) -> np.ndarray:
    return path.series(model, name)[:horizon] * 400.0


def _pct(path: Path, model: LinearModel, name: str, horizon: int) -> np.ndarray:
    return path.series(model, name)[:horizon]


def plot_scenario_a(
    model: LinearModel,
    sterilised: Path,
    unsterilised: Path,
    dest: Path,
) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2), sharex=True)
    series = [
        (axes[0, 0], "Long rate", _bp(sterilised, model, "rl", h), "bp", None),
        (axes[0, 1], "Bank Rate", _bp(sterilised, model, "i", h), "bp", _bp(unsterilised, model, "i", h)),
        (axes[0, 2], "Sterilising residual", sterilised.e_ster[:h] * 400.0, "bp", None),
        (axes[1, 0], "New and stock mortgage rates", _bp(sterilised, model, "rm", h), "bp", _bp(sterilised, model, "rms", h)),
        (axes[1, 1], "Output", _pct(sterilised, model, "y", h), "%", None),
        (axes[1, 2], "House prices", _pct(sterilised, model, "ph", h), "%", None),
        (axes[2, 0], "Consumption", _pct(sterilised, model, "cs", h), "%", _pct(sterilised, model, "cb", h)),
        (axes[2, 1], "Debt interest IB", _pct(sterilised, model, "ib", h), "pp of GDP", None),
        (axes[2, 2], "Effective conventional coupon", _bp(sterilised, model, "reff", h), "bp", None),
    ]
    labels = {
        axes[1, 0]: ("new advances $R^m$", "stock $R^{m,stock}$"),
        axes[2, 0]: ("savers $C^s$", "borrowers $C^b$"),
        axes[0, 1]: ("sterilised (a)", "same TP shock, Taylor rule"),
    }
    for ax, title, main, unit, extra in series:
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.plot(q, main, color="#8c2f39", lw=1.6, label=labels.get(ax, (None, None))[0])
        if extra is not None:
            ax.plot(q, extra, color="#1f4e79", lw=1.3, ls="--", label=labels.get(ax, (None, None))[1])
            ax.legend(frameon=False, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(1, h)
    for ax in axes[2, :]:
        ax.set_xlabel("quarter")
    fig.suptitle(
        "Scenario (a): +100 bp term premium, Bank Rate sterilised",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.005,
        "Sterilisation: " + STERILISATION_DEVICE,
        fontsize=6.5,
        ha="left",
        va="bottom",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def plot_a_versus_b(
    model: LinearModel,
    scenario_a: Path,
    scenario_b: Path,
    dest: Path,
) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    panels = [
        (axes[0], "Output", "y"),
        (axes[1], "House prices", "ph"),
    ]
    for ax, title, name in panels:
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.plot(q, scenario_a.series(model, name)[:h], color="#8c2f39", lw=1.7, label="(a) sterilised TP")
        ax.plot(
            q,
            scenario_b.series(model, name)[:h],
            color="#1f4e79",
            lw=1.5,
            ls="--",
            label="(b) short-rate news",
        )
        ax.set_title(title)
        ax.set_ylabel("percent")
        ax.set_xlabel("quarter")
        ax.set_xlim(1, h)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Output and house prices: sterilised term premium versus short-rate news\n"
        "(both scaled to +100 bp on the model long rate)",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        "Sterilisation on (a) only: " + STERILISATION_DEVICE,
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
