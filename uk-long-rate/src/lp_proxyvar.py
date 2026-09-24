#!/usr/bin/env python3
"""HF gilt-surprise local projections and a small external-instrument VAR.

Preferred UK series (not shipped in this repo)
----------------------------------------------
* High-frequency gilt yield changes around MPC / monetary-event windows,
  in the style of Cesa-Bianchi, Thwaites and Vicondoa (2020), European
  Economic Review, "Monetary policy transmission in the United Kingdom:
  a high frequency identification approach".
* Bank of England yield curves:
  https://www.bankofengland.co.uk/statistics/yield-curves
* Outcomes, once a real file is supplied: ONS monthly GDP (Y) and the
  UK House Price Index, or Halifax / Nationwide (P^h).

Identification, kept separate
-----------------------------
* Proxy (a): the high-frequency change in the 10-year gilt, orthogonalised
  to the Bank Rate surprise and the near-term OIS path. This is the
  gilt-surprise analogue of a term-premium / long-rate move.
* Proxy (b): the OIS-path / expected Bank Rate factor itself.

Do not invent surprise data. ``data/gilt_surprises.TEMPLATE.csv`` is a
header only. If ``data/gilt_surprises.csv`` is absent, this script writes
``output/lp_irf_stub_Y_Ph.csv`` with NaN coefficients and exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# TODO: replace the template with a real event-level file before treating
# any coefficient as a UK estimate. Column units are basis points for
# surprises and percent for the outcome series.
REQUIRED_COLUMNS = (
    "date",
    "event_type",
    "bank_rate_surprise_bp",
    "ois_path_bp",
    "gilt_10y_hf_bp",
)
OPTIONAL_COLUMNS = ("gilt_2y_hf_bp",)
OUTCOME_Y = "y"  # ONS monthly GDP, percent
OUTCOME_PH = "ph"  # house-price index, percent
HORIZONS = tuple(range(0, 13))  # months if the file is monthly; document in the CSV

DATA_PATH = ROOT / "data" / "gilt_surprises.csv"
STUB_PATH = ROOT / "output" / "lp_irf_stub_Y_Ph.csv"


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def orthogonalise_gilt_surprise(
    gilt_10y_hf_bp: np.ndarray,
    bank_rate_surprise_bp: np.ndarray,
    ois_path_bp: np.ndarray,
) -> np.ndarray:
    """Proxy (a): residual of the 10y HF change on a constant, Bank Rate, and OIS.

    Proxy (b) is ``ois_path_bp`` itself and is not returned here.
    """
    y = _as_float(gilt_10y_hf_bp)
    br = _as_float(bank_rate_surprise_bp)
    ois = _as_float(ois_path_bp)
    if not (y.shape == br.shape == ois.shape):
        raise ValueError("surprise series must be aligned and the same length")
    X = np.column_stack([np.ones(y.shape[0]), br, ois])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and homoskedastic standard errors."""
    y = _as_float(y)
    X = np.asarray(X, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    if dof <= 0:
        return beta, np.full(k, np.nan)
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0.0))
    return beta, se


def jorda_lp(
    outcome: np.ndarray,
    shock: np.ndarray,
    controls: np.ndarray | None,
    horizons: tuple[int, ...] | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Jordà local projection. Returns the shock coefficient and its SE at each horizon.

    ``controls`` is aligned with ``outcome`` and ``shock`` at date t. Lags must
    already be built by the caller. Horizon h regresses outcome[t+h] on the
    date-t shock and date-t controls.
    """
    y = _as_float(outcome)
    z = _as_float(shock)
    if controls is None:
        C = np.zeros((y.shape[0], 0))
    else:
        C = np.atleast_2d(np.asarray(controls, dtype=float))
        if C.shape[0] != y.shape[0]:
            C = C.T
        if C.shape[0] != y.shape[0]:
            raise ValueError("controls must have one row per observation")
    betas = np.full(len(horizons), np.nan)
    ses = np.full(len(horizons), np.nan)
    for i, h in enumerate(horizons):
        if h < 0:
            raise ValueError("horizons must be non-negative")
        if h == 0:
            yy, zz, CC = y, z, C
        else:
            yy, zz, CC = y[h:], z[:-h], C[:-h]
        mask = np.isfinite(yy) & np.isfinite(zz)
        if CC.shape[1]:
            mask &= np.all(np.isfinite(CC), axis=1)
        if int(mask.sum()) <= CC.shape[1] + 2:
            continue
        X = np.column_stack([np.ones(int(mask.sum())), zz[mask], CC[mask]])
        beta, se = ols(yy[mask], X)
        betas[i] = beta[1]
        ses[i] = se[1]
    return betas, ses


def var1_residuals(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """VAR(1) with a constant. Returns residuals (T-1, n) and the companion coefficients."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 4:
        raise ValueError("VAR sample is too short")
    lag = Y[:-1]
    now = Y[1:]
    X = np.column_stack([np.ones(lag.shape[0]), lag])
    coef, *_ = np.linalg.lstsq(X, now, rcond=None)
    resid = now - X @ coef
    return resid, coef


def proxy_var_impact(residuals: np.ndarray, proxy: np.ndarray, normalise: int = 0) -> np.ndarray:
    """External-instrument impact vector, scaled so component ``normalise`` equals 1.

    Mertens–Ravn / Stock–Watson: the impact is proportional to the covariance
    of the reduced-form residuals with the proxy.
    """
    U = np.asarray(residuals, dtype=float)
    z = _as_float(proxy)
    if z.shape[0] == U.shape[0] + 1:
        z = z[1:]
    if z.shape[0] != U.shape[0]:
        raise ValueError("proxy length must match the VAR sample or the residual sample")
    mask = np.isfinite(z) & np.all(np.isfinite(U), axis=1)
    cov = U[mask].T @ z[mask] / float(mask.sum())
    scale = cov[normalise]
    if abs(scale) < 1e-14:
        raise ValueError("proxy does not load on the normalised residual")
    return cov / scale


def load_surprise_csv(path: Path) -> dict[str, np.ndarray]:
    """Load a real surprise file. Refuses the header-only template.

    TODO: point this at the maintained BoE / Cesa-Bianchi–Thwaites–Vicondoa
    event file once it is available. Do not synthesise that file here.
    """
    text = path.read_text(encoding="utf-8").strip().splitlines()
    text = [ln for ln in text if ln.strip() and not ln.startswith("#")]
    if len(text) < 2:
        raise FileNotFoundError(
            f"{path} has no data rows. Copy the template and fill it with "
            "real event surprises; do not invent them."
        )
    header = [h.strip() for h in text[0].split(",")]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    cols = {name: [] for name in header}
    for line in text[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            raise ValueError(f"row has {len(parts)} fields, header has {len(header)}")
        for name, raw in zip(header, parts):
            cols[name].append(raw)
    out: dict[str, np.ndarray] = {}
    for name, raw in cols.items():
        if name in ("date", "event_type"):
            out[name] = np.asarray(raw)
        else:
            out[name] = np.array([np.nan if v == "" else float(v) for v in raw])
    return out


def write_stub(path: Path, horizons: tuple[int, ...] = HORIZONS) -> None:
    """IRF-ready schema for Y and P^h under (a) and (b). Coefficients stay NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "horizon,beta_y_a,se_y_a,beta_ph_a,se_ph_a,"
        "beta_y_b,se_y_b,beta_ph_b,se_ph_b"
    )
    lines = [
        "# IRF stub only. No UK gilt-surprise file was loaded. "
        "Coefficients are NaN on purpose.",
        "# Proxy (a) = 10y gilt HF change orthogonalised to Bank Rate and the OIS path.",
        "# Proxy (b) = OIS path / expected Bank Rate factor.",
        "# Outcomes: y = ONS monthly GDP, ph = UK house price index.",
        header,
    ]
    for h in horizons:
        lines.append(f"{h}," + ",".join(["nan"] * 8))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def irfs_from_frame(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """LP IRFs for Y and P^h. Requires outcome columns; otherwise the caller keeps NaNs."""
    proxy_a = orthogonalise_gilt_surprise(
        frame["gilt_10y_hf_bp"],
        frame["bank_rate_surprise_bp"],
        frame["ois_path_bp"],
    )
    proxy_b = _as_float(frame["ois_path_bp"])
    out: dict[str, np.ndarray] = {}
    for outcome, key in ((OUTCOME_Y, "y"), (OUTCOME_PH, "ph")):
        if outcome not in frame:
            for tag in ("a", "b"):
                out[f"beta_{key}_{tag}"] = np.full(len(HORIZONS), np.nan)
                out[f"se_{key}_{tag}"] = np.full(len(HORIZONS), np.nan)
            continue
        y = frame[outcome]
        lag = np.full(y.shape[0], np.nan)
        lag[1:] = y[:-1]
        for tag, shock in (("a", proxy_a), ("b", proxy_b)):
            beta, se = jorda_lp(y, shock, lag[:, None], HORIZONS)
            out[f"beta_{key}_{tag}"] = beta
            out[f"se_{key}_{tag}"] = se
    out["horizon"] = np.asarray(HORIZONS, dtype=float)
    return out


def write_irfs(path: Path, irfs: dict[str, np.ndarray], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "horizon",
        "beta_y_a",
        "se_y_a",
        "beta_ph_a",
        "se_ph_a",
        "beta_y_b",
        "se_y_b",
        "beta_ph_b",
        "se_ph_b",
    ]
    lines = ["# " + note, ",".join(keys)]
    n = len(irfs["horizon"])
    for i in range(n):
        row = []
        for k in keys:
            v = irfs[k][i]
            row.append(f"{int(v)}" if k == "horizon" else f"{v:.8g}")
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not DATA_PATH.exists():
        write_stub(STUB_PATH)
        print(
            "TODO: load a real UK HF surprise file at data/gilt_surprises.csv "
            "(see data/gilt_surprises.TEMPLATE.csv and data/README.md)."
        )
        print(f"Wrote NaN IRF stub {STUB_PATH}")
        return 0
    frame = load_surprise_csv(DATA_PATH)
    irfs = irfs_from_frame(frame)
    note = (
        "LP coefficients from data/gilt_surprises.csv. "
        "Proxy (a) is the 10y gilt residual; proxy (b) is the OIS path. "
        "Not a structural model IRF."
    )
    write_irfs(STUB_PATH, irfs, note)
    print(f"Wrote {STUB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
