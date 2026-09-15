#!/usr/bin/env python3
"""Smoke test: HA Figure-3 co-movements and RANK sign contrast."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solve_ssj import irf_to_figure3_series, qualitative_flags, solve_variant


def main() -> int:
    ss, _, irf, report = solve_variant("ha", T=60)
    flags = qualitative_flags(irf_to_figure3_series(irf, ss, 20))
    print("HA flags", flags)
    print(
        f"HtM={report['htm']:.3f} iMPC1={report['impc_y1']:.3f} "
        f"import_share={report['import_share']:.3f}"
    )
    assert all(flags.values()), flags
    assert 0.4 < report["htm"] < 0.6
    assert 0.4 < report["impc_y1"] < 0.65
    assert abs(report["import_share"] - 0.12) < 1e-6

    ss_ra, _, irf_ra, _ = solve_variant("ra", T=60)
    flags_ra = qualitative_flags(irf_to_figure3_series(irf_ra, ss_ra, 20))
    print("RA flags", flags_ra)
    assert flags_ra["C_positive"] is False
    assert flags_ra["Y_NT_positive"] is True

    ss_lb, _, irf_lb, _ = solve_variant("ra_lb", T=60)
    flags_lb = qualitative_flags(irf_to_figure3_series(irf_lb, ss_lb, 20))
    print("RA-LB flags", flags_lb)
    assert flags_lb["Y_T_positive"] is False
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
