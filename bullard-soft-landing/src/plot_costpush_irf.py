#!/usr/bin/env python3
"""
Plot FIRE vs social-learning cost-push (inflation) IRFs for the Bullard HENK prototype.

Runs henk_sim cost-push paths if CSVs are missing, then writes a 2x2 chart:
  pi, y, i, phi — FIRE (solid) vs SL (dashed).

Example:
  python3 src/plot_costpush_irf.py
  python3 src/plot_costpush_irf.py --horizon 40 --seed 1 --shock-size 1.0
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    out: dict[str, np.ndarray] = {}
    keys = [k for k in rows[0].keys() if k != "t"]
    out["t"] = np.array([int(r["t"]) for r in rows], dtype=int)
    for k in keys:
        out[k] = np.array([float(r[k]) for r in rows], dtype=float)
    return out


def ensure_costpush_csvs(
    save_dir: Path,
    horizon: int,
    seed: int,
    shock_size: float,
) -> tuple[Path, Path]:
    fire_path = save_dir / "irf_costpush_fire.csv"
    sl_path = save_dir / "irf_costpush_sl.csv"
    if fire_path.is_file() and sl_path.is_file():
        return fire_path, sl_path

    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_SRC / "henk_sim.py"),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--shock",
        "costpush",
        "--shock-size",
        str(shock_size),
        "--save-dir",
        str(save_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    if not fire_path.is_file() or not sl_path.is_file():
        raise FileNotFoundError(
            f"expected {fire_path} and {sl_path} after henk_sim"
        )
    return fire_path, sl_path


def plot_fire_vs_sl(
    fire: dict[str, np.ndarray],
    sl: dict[str, np.ndarray],
    out_path: Path,
    title_extra: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "Bullard HENK prototype — cost-push (inflation) IRF\n"
        f"FIRE vs social learning{title_extra}",
        fontsize=12,
    )
    series = [
        ("pi", r"Inflation $\pi$", axes[0, 0]),
        ("y", r"Output gap $y$", axes[0, 1]),
        ("i", r"Policy rate $i$", axes[1, 0]),
        ("phi", r"Belief wedge $\phi$", axes[1, 1]),
    ]
    for key, title, ax in series:
        ax.plot(fire["t"], fire[key], label="FIRE", lw=2)
        ax.plot(sl["t"], sl[key], label="SL", lw=2, ls="--")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Quarters")
    axes[1, 1].set_xlabel("Quarters")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FIRE vs SL cost-push IRFs (Bullard HENK prototype)"
    )
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shock-size", type=float, default=1.0)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output",
        help="Directory for henk_sim CSVs (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/irf_costpush_fire_vs_sl.png",
        help="Output PNG path (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Do not re-run henk_sim if CSVs already exist (default behaviour)",
    )
    parser.add_argument(
        "--force-sim",
        action="store_true",
        help="Re-run henk_sim even if CSVs exist",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = _REPO / save_dir
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO / out_path

    if args.force_sim:
        for p in (
            save_dir / "irf_costpush_fire.csv",
            save_dir / "irf_costpush_sl.csv",
        ):
            if p.is_file():
                p.unlink()

    fire_path, sl_path = ensure_costpush_csvs(
        save_dir, args.horizon, args.seed, args.shock_size
    )
    fire = _load_csv(fire_path)
    sl = _load_csv(sl_path)
    extra = f" (seed={args.seed}, Table 1 params, shock_size={args.shock_size})"
    written = plot_fire_vs_sl(fire, sl, out_path, title_extra=extra)
    print(f"Wrote {written}")
    print(
        f"peak pi FIRE={float(np.max(fire['pi'])):.6f}, "
        f"SL={float(np.max(sl['pi'])):.6f}; "
        f"max|phi| SL={float(np.max(np.abs(sl['phi']))):.6f}"
    )


if __name__ == "__main__":
    main()
