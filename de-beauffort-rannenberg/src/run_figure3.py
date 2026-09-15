#!/usr/bin/env python3
"""Solve HANK/RANK IRFs and write WP 493 Figure 3 plus CSV/summary artefacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plot_figure3 import plot_figure3  # noqa: E402
from src.solve_ssj import (  # noqa: E402
    irf_to_figure3_series,
    output_dir,
    qualitative_flags,
    solve_variant,
)


def _save_csv(path: Path, series: dict) -> None:
    keys = list(series)
    h = len(series[keys[0]])
    header = "quarter," + ",".join(keys)
    rows = [header]
    for t in range(h):
        rows.append(str(t + 1) + "," + ",".join(f"{series[k][t]:.8g}" for k in keys))
    path.write_text("\n".join(rows) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--skip-lb", action="store_true", help="skip φ_GT=0 variants")
    p.add_argument("--psi-nfa", type=float, default=1e-3)
    args = p.parse_args()

    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    variants = ["ha", "ra"] if args.skip_lb else ["ha", "ra", "ha_lb", "ra_lb"]
    packed = {}
    reports = {}
    flags = {}
    for kind in variants:
        print(f"Solving {kind} ...", flush=True)
        ss, ssdict, irf, report = solve_variant(
            kind, T=args.T, psi_nfa=args.psi_nfa
        )
        series = irf_to_figure3_series(irf, ss, h=args.horizon)
        packed[kind] = series
        reports[kind] = {k: (float(v) if np.isscalar(v) else v) for k, v in report.items()}
        flags[kind] = qualitative_flags(series)
        _save_csv(out / f"irf_{kind}.csv", series)
        print(f"  flags {kind}: {flags[kind]}", flush=True)
        print(
            f"  G={report['G']:.4f}  A={report['A']:.4f}  HtM={report['htm']:.3f}  "
            f"iMPC1={report['impc_y1']:.3f}",
            flush=True,
        )

    png = out / "figure3.png"
    plot_figure3(
        packed["ha"],
        packed["ra"],
        packed.get("ha_lb"),
        packed.get("ra_lb"),
        png,
        include_sbvar=True,
    )
    summary = {"reports": reports, "flags": flags, "figure": str(png)}
    (out / "figure3_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    txt = []
    txt.append("de Beauffort & Rannenberg (NBB WP 493) — Figure 3 IRF summary")
    txt.append("Shock: 1% of GDP government spending, empirical-shaped path.")
    for kind, fl in flags.items():
        txt.append(f"\n[{kind}] qualitative co-movements:")
        for k, v in fl.items():
            txt.append(f"  {k}: {v}")
        r = reports[kind]
        txt.append(
            f"  SS: G/Y={r['G']:.4f} A={r['A']:.4f} HtM={r['htm']:.3f} "
            f"year-1 iMPC={r['impc_y1']:.3f}"
        )
    (out / "figure3_summary.txt").write_text("\n".join(txt) + "\n")
    print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
