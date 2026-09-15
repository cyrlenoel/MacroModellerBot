"""Figure 3 layout: 3×3 IRFs, HANK vs RANK vs S-BVAR overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .empirical_sbvar import SBVAR

PANELS = [
    ("C", "Aggregate consumption"),
    ("C_T", "Tradable consumption"),
    ("C_NT", "Non-tradable consumption"),
    ("Y", "Aggregate output"),
    ("Y_T", "Tradable output"),
    ("Y_NT", "Non-tradable output"),
    ("p_ratio", "Price ratio T/NT"),
    ("NX", "Net exports"),
    ("G", "Gov spending"),
]


def plot_figure3(
    ha: Dict[str, np.ndarray],
    ra: Dict[str, np.ndarray],
    ha_lb: Dict[str, np.ndarray] | None,
    ra_lb: Dict[str, np.ndarray] | None,
    path: Path,
    include_sbvar: bool = True,
):
    h = len(ha["C"])
    q = np.arange(1, h + 1)
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        if include_sbvar and key in SBVAR:
            ax.plot(
                q,
                SBVAR[key][:h],
                color="black",
                marker="o",
                markerfacecolor="white",
                markersize=4.5,
                linewidth=1.1,
                label="S-BVAR",
            )
        ax.plot(q, ha[key], color="#1f77b4", lw=2.0, label="HA")
        ax.plot(q, ra[key], color="#d4a017", lw=1.7, ls="--", label="RA")
        if ha_lb is not None:
            ax.plot(q, ha_lb[key], color="#1f77b4", lw=1.4, ls="-.", label="HA-LB")
        if ra_lb is not None:
            ax.plot(q, ra_lb[key], color="#d4a017", lw=1.3, ls=":", label="RA-LB")
        ax.axhline(0.0, color="0.55", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(1, h)
        if ax in axes[-1]:
            ax.set_xlabel("Quarters")
    axes[0, 0].set_ylabel("Percentage points (dev from ss)")
    axes[1, 0].set_ylabel("Percentage points (dev from ss)")
    axes[2, 0].set_ylabel("pp of GDP / percent")
    hnd, lab = axes[0, 0].get_legend_handles_labels()
    # unique labels
    seen = {}
    for k, v in zip(lab, hnd):
        seen.setdefault(k, v)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.suptitle(
        "Impulse responses to a government spending shock (WP 493 Figure 3 layout)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
    fig.text(
        0.5,
        0.012,
        "S-BVAR markers are hand-digitised from WP 493 Figure 3 [PROVISIONAL], "
        "not the authors' numerical series.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
