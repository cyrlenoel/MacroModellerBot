#!/usr/bin/env python3
"""Slide-ready IRF figures for the Beamer deck.

Reuses the calibrated QZ paths from ``src/run_irf.py`` (same scaling, same
peg). Writes PDF and PNG under ``output/slides/`` and does not touch the
repo figures in ``output/`` or the model.

Scenario (c), "(c) sovereign", is not in the MVP. This script does not
draw a (c) curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.linear_model import build_model, interest_burden_split  # noqa: E402
from src.plot_irf import _bp, _pct  # noqa: E402
from src.run_irf import _impulse, _scale_to_long_rate  # noqa: E402
from src.solve_pf import solve_qz, solve_sterilised_qz  # noqa: E402
from src.success import evaluate_scenario_a  # noqa: E402


# Agreed deck labels. "(c) sovereign" is the name for a shock that is not
# implemented; it is intentionally not a plotted series.
LABEL_A = "(a) sterilised term premium"
LABEL_B = "(b) path news"
LABEL_TAYLOR_FREE = "Taylor rule free"
CALIBRATED_TAG = "Calibrated MVP, not estimated"

SLIDE_NAMES = (
    "a_macro",
    "a_consumption_mortgage",
    "a_vs_b",
    "ib_decomposition",
    "peg_robustness",
    "bank_rate_sterilisation",
)

# Beamer aspectratio=169 paper is 16 cm × 9 cm. Draw at that size so a
# \includegraphics[width=\paperwidth] keeps these font sizes.
FIGSIZE = (16.0 / 2.54, 9.0 / 2.54)

COLOUR_A = "#8c2f39"
COLOUR_B = "#1f4e79"
COLOUR_FREE = "#5e6a71"
COLOUR_TOTAL = "#1a1a1a"
PEG_COLOURS = {8: "#4c78a8", 12: "#8c2f39", 20: "#f58518", 40: "#54a24b"}

SLIDE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "axes.titlepad": 4,
    "axes.labelpad": 2,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
}


def _quarter_ticks(horizon: int) -> list[int]:
    if horizon >= 40:
        return [1, 10, 20, 30, horizon]
    return [1, horizon]


def _prepare(ax, title: str, ylabel: str, horizon: int, xlabel: bool) -> None:
    ax.axhline(0.0, color="#b5b5b5", lw=0.7, zorder=0)
    ax.set_xlim(1, horizon)
    ax.set_xticks(_quarter_ticks(horizon))
    ax.set_title(title, loc="left", pad=3)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Quarter")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.grid(True, linestyle=":", color="#e4e4e4", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)


def _line(ax, q, values, *, color: str, label: str | None, ls: str = "-", lw: float | None = None, zorder: int = 2) -> None:
    ax.plot(q, values, color=color, ls=ls, lw=1.9 if lw is None else lw, label=label, zorder=zorder, solid_capstyle="round")


def _dress(fig, title: str, *, top: float = 0.82, bottom: float = 0.14, wspace: float = 0.40, hspace: float = 0.50) -> None:
    """Left-aligned title lines. The calibration tag sits on the top-right."""
    fig._slide_lines = []
    y = 0.988
    for i, line in enumerate(title.split("\n")):
        artist = fig.text(
            0.015,
            y,
            line,
            ha="left",
            va="top",
            fontsize=13.5 if i == 0 else 12.0,
            color="#1a1a1a",
        )
        fig._slide_lines.append(artist)
        y -= 0.055
    fig.text(
        0.985,
        0.988,
        CALIBRATED_TAG,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#5c5c5c",
        style="italic",
    )
    fig.subplots_adjust(left=0.125, right=0.985, top=top, bottom=bottom, wspace=wspace, hspace=hspace)


def _intersects(a, b, pad: float = 1.0) -> bool:
    return not (a.x1 < b.x0 + pad or b.x1 < a.x0 + pad or a.y1 < b.y0 + pad or b.y1 < a.y0 + pad)


def _layout_problems(fig) -> list[str]:
    """Text that collides, or that falls outside the canvas. Legend labels may sit inside their legend."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tag = next(t for t in fig.texts if t.get_text() == CALIBRATED_TAG)
    tag_bb = tag.get_window_extent(renderer)
    problems = []
    others = list(fig._slide_lines)
    for ax in fig.axes:
        for attr in ("title", "_left_title", "_right_title"):
            text = getattr(ax, attr, None)
            if text is not None and text.get_text():
                others.append(text)
        others.extend([ax.xaxis.label, ax.yaxis.label])
        others.extend(ax.texts)
        if ax.get_legend() is not None:
            others.append(ax.get_legend())
    others.extend(fig.legends)
    for artist in others:
        bb = artist.get_window_extent(renderer)
        label = artist.get_text() if hasattr(artist, "get_text") else "legend"
        if _intersects(tag_bb, bb):
            problems.append(f"tag overlaps {label!r}")
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > fig.bbox.width + 2 or bb.y1 > fig.bbox.height + 2:
            problems.append(f"clipped {label!r}")
    for legend in fig.legends:
        leg_bb = legend.get_window_extent(renderer)
        for line in fig._slide_lines:
            if _intersects(leg_bb, line.get_window_extent(renderer), pad=2):
                problems.append(f"legend overlaps title {line.get_text()!r}")
        for ax in fig.axes:
            panel = getattr(ax, "_left_title", None)
            if panel is not None and panel.get_text() and _intersects(leg_bb, panel.get_window_extent(renderer), pad=2):
                problems.append(f"legend overlaps panel {panel.get_text()!r}")
    return problems


def _save(fig, dest: Path, name: str) -> None:
    problems = _layout_problems(fig)
    if problems:
        plt.close(fig)
        raise RuntimeError(f"{name}: " + "; ".join(problems))
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.pdf")
    fig.savefig(dest / f"{name}.png", dpi=300)
    plt.close(fig)


def _legend(ax, loc: str = "best") -> None:
    ax.legend(frameon=False, loc=loc, handlelength=2.4, borderaxespad=0.2)


def _legend_under_title(fig, handles, labels, ncol: int) -> None:
    """A single legend row under the title, with the panels dropped below it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lowest = min(line.get_window_extent(renderer).transformed(inv).y0 for line in fig._slide_lines)
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, lowest - 0.025),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.15,
        fontsize=9.5,
        borderaxespad=0.0,
    )
    fig.canvas.draw()
    legend_bb = legend.get_window_extent(renderer).transformed(inv)
    # Panel titles sit above the axes, so the gap has to clear that line of type.
    fig.subplots_adjust(top=legend_bb.y0 - 0.078)


def compute_paths(model=None):
    """Scenario (a), Taylor-rule-free twin, scenario (b), and peg lengths.

    Scaling matches ``src/run_irf.py``: (a) and (b) are each scaled so the
    model long rate is +100 annualised bp on impact. The Taylor-rule-free
    path uses the same term-premium innovation as (a) and is not rescaled.
    Peg lengths other than H = 12 are robustness, not a third shock.
    """
    model = build_model() if model is None else model
    cal = model.cal
    horizon = cal.irf_horizon
    T = max(horizon, cal.H_peg, 40) + 20
    unit = _impulse(T, 1.0)

    path_a = solve_sterilised_qz(model, {"e_tp": unit}, H=cal.H_peg, T=T)
    scale_a = _scale_to_long_rate(model, path_a)
    path_free = solve_qz(model, {"e_tp": _impulse(T, scale_a)}, T=T)
    path_b = solve_qz(model, {"e_news": unit.copy()}, T=T)
    _scale_to_long_rate(model, path_b)

    robust: dict[int, object] = {}
    for H in (8, 12, 20, 40):
        if H == cal.H_peg:
            robust[H] = path_a
            continue
        path_h = solve_sterilised_qz(model, {"e_tp": _impulse(T, 1.0)}, H=H, T=T)
        _scale_to_long_rate(model, path_h)
        robust[H] = path_h
    return model, path_a, path_free, path_b, robust, horizon


def scenario_numbers(model, path_a, path_b) -> dict[str, float | int]:
    """Key slide numbers, read off the scaled paths. Quarters are 1-indexed."""
    h = model.cal.irf_horizon
    y = _pct(path_a, model, "y", h)
    ph = _pct(path_a, model, "ph", h)
    ib = _pct(path_a, model, "ib", h)
    cs = _pct(path_a, model, "cs", h)
    cb = _pct(path_a, model, "cb", h)

    def trough(series: np.ndarray) -> tuple[float, int]:
        i = int(np.argmin(series))
        return float(series[i]), i + 1

    def peak(series: np.ndarray) -> tuple[float, int]:
        i = int(np.argmax(series))
        return float(series[i]), i + 1

    y_trough, y_q = trough(y)
    ph_trough, ph_q = trough(ph)
    ib_peak, ib_q = peak(ib)
    cs_trough, cs_q = trough(cs)
    cb_trough, cb_q = trough(cb)
    return {
        "H": int(model.cal.H_peg),
        "lambda_q": float(model.cal.lambda_q),
        "y_trough": y_trough,
        "y_trough_q": y_q,
        "ph_trough": ph_trough,
        "ph_trough_q": ph_q,
        "ib_peak": ib_peak,
        "ib_peak_q": ib_q,
        "cs_trough": cs_trough,
        "cs_trough_q": cs_q,
        "cb_trough": cb_trough,
        "cb_trough_q": cb_q,
        "rl_a_bp": float(_bp(path_a, model, "rl", 1)[0]),
        "rl_b_bp": float(_bp(path_b, model, "rl", 1)[0]),
    }


def _readme(numbers: dict[str, float | int]) -> str:
    n = numbers
    return f"""# Slide-ready IRFs

PDF and PNG figures for a 16:9 Beamer deck. Generated by `src/plot_slides.py` from the calibrated Python QZ twin (same paths and +100 bp scaling as `src/run_irf.py`). They do not replace the figures in `output/`.

Calibrated MVP, not estimated. The figures have no sterilisation footnote; the slide text carries that explanation.

Agreed shock labels:

- `{LABEL_A}` — plotted
- `{LABEL_B}` — plotted
- `(c) sovereign` — not implemented in the MVP. No (c) curve and no (c) figure.

Scenario (a) is a term-premium innovation scaled to +100 bp on the long rate, with an anticipated Bank Rate peg of H = {n['H']} quarters. Scenario (b) is short-rate path news scaled to the same +100 bp long-rate move. H = {n['H']} applies to (a) only. The mortgage-stock lag uses lambda = {n['lambda_q']:.2f}.

Regenerate from `uk-long-rate/`:

```bash
python3 src/plot_slides.py
```

## Key numbers from this run

Scenario (a), H = {n['H']}, read off the scaled path:

- Long rate on impact, (a): {n['rl_a_bp']:.3f} annualised bp
- Long rate on impact, (b): {n['rl_b_bp']:.3f} annualised bp
- Output trough: {n['y_trough']:.3f}% at q{n['y_trough_q']}
- House-price trough: {n['ph_trough']:.3f}% at q{n['ph_trough_q']} (about {n['ph_trough']:.1f}% at q{n['ph_trough_q']})
- Debt interest IB peak: {n['ib_peak']:.4f} pp of GDP at q{n['ib_peak_q']}
- Borrower consumption C^b trough: {n['cb_trough']:.3f}% at q{n['cb_trough_q']}
- Saver consumption C^s trough: {n['cs_trough']:.3f}% at q{n['cs_trough_q']}

## Files

### `a_macro.pdf`, `a_macro.png`

Output, house prices, and debt interest IB under {LABEL_A}. The title states +100 bp, anticipated Bank Rate peg H = {n['H']}.

### `a_consumption_mortgage.pdf`, `a_consumption_mortgage.png`

Savers C^s against borrowers C^b, next to new-advance R^m against stock R^{{m,stock}}. The stock lags new advances because lambda = {n['lambda_q']:.2f}. H = {n['H']} is in the title.

### `a_vs_b.pdf`, `a_vs_b.png`

Output and house prices. {LABEL_A} against {LABEL_B}, both scaled to +100 bp on the long rate. The title states that H = {n['H']} applies to (a) only.

### `ib_decomposition.pdf`, `ib_decomposition.png`

Debt interest IB under {LABEL_A}, split into the nominal coupon and the index-linked uplift (including inflation uplift). H = {n['H']}.

### `peg_robustness.pdf`, `peg_robustness.png`

The same sterilised term premium at H = 8, 12, 20 and 40. H = {n['H']} is marked as the default. The other peg lengths are robustness cases, not scenario (b) or (c).

### `bank_rate_sterilisation.pdf`, `bank_rate_sterilisation.png`

Bank Rate under {LABEL_A} against the Taylor rule free path (same term-premium innovation, peg off), plus the sterilising residual. H = {n['H']}.
"""


def plot_a_macro(model, path_a, dest: Path) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    panels = [
        (axes[0], "Output", _pct(path_a, model, "y", h), "percent"),
        (axes[1], "House prices", _pct(path_a, model, "ph", h), "percent"),
        (axes[2], "Debt interest IB", _pct(path_a, model, "ib", h), "pp of GDP"),
    ]
    for ax, title, series, unit in panels:
        _prepare(ax, title, unit, h, xlabel=True)
        _line(ax, q, series, color=COLOUR_A, label=None)
    _dress(
        fig,
        f"{LABEL_A}\n+100 bp, anticipated Bank Rate peg H = {model.cal.H_peg}",
    )
    _save(fig, dest, "a_macro")


def plot_a_consumption_mortgage(model, path_a, dest: Path) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    lam = model.cal.lambda_q
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    _prepare(axes[0], "Consumption", "percent", h, xlabel=True)
    _line(axes[0], q, _pct(path_a, model, "cs", h), color=COLOUR_B, label="savers $C^{s}$")
    _line(axes[0], q, _pct(path_a, model, "cb", h), color=COLOUR_A, label="borrowers $C^{b}$")
    _legend(axes[0], loc="lower right")

    _prepare(axes[1], f"Mortgage rates, λ = {lam:.2f}", "annualised bp", h, xlabel=True)
    _line(axes[1], q, _bp(path_a, model, "rm", h), color=COLOUR_A, label="new advances $R^{m}$")
    _line(
        axes[1],
        q,
        _bp(path_a, model, "rms", h),
        color=COLOUR_B,
        label="stock $R^{m,\\mathrm{stock}}$",
        ls="--",
    )
    # Upper right is empty: new advances jump on impact and then decay.
    _legend(axes[1], loc="upper right")
    _dress(
        fig,
        f"{LABEL_A}, H = {model.cal.H_peg}\nsavers $C^{{s}}$ vs borrowers $C^{{b}}$, mortgage-stock lag λ = {lam:.2f}",
        wspace=0.32,
    )
    _save(fig, dest, "a_consumption_mortgage")


def plot_a_vs_b(model, path_a, path_b, dest: Path) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    panels = [
        (axes[0], "Output", "y"),
        (axes[1], "House prices", "ph"),
    ]
    for ax, title, name in panels:
        _prepare(ax, title, "percent", h, xlabel=True)
        _line(ax, q, _pct(path_a, model, name, h), color=COLOUR_A, label=LABEL_A)
        _line(ax, q, _pct(path_b, model, name, h), color=COLOUR_B, label=LABEL_B, ls="--")
    handles, labels = axes[0].get_legend_handles_labels()
    _dress(
        fig,
        f"{LABEL_A} vs {LABEL_B}\n+100 bp on the long rate. H = {model.cal.H_peg} applies to (a) only.",
        wspace=0.32,
    )
    _legend_under_title(fig, handles, labels, ncol=2)
    _save(fig, dest, "a_vs_b")


def plot_ib_decomposition(model, path_a, dest: Path) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    ib_nom, ib_il = interest_burden_split(
        model,
        path_a.series(model, "reff")[:h],
        path_a.series(model, "ril")[:h],
        path_a.series(model, "pi")[:h],
        path_a.series(model, "dg")[:h],
    )
    total = path_a.series(model, "ib")[:h]
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    _prepare(ax, "Debt interest IB", "pp of GDP", h, xlabel=True)
    _line(ax, q, total, color=COLOUR_TOTAL, label="debt interest IB", lw=2.1)
    _line(ax, q, ib_nom, color=COLOUR_A, label="nominal coupon")
    _line(ax, q, ib_il, color=COLOUR_B, label="index-linked uplift", ls="--")
    _legend(ax, loc="upper left")
    _dress(
        fig,
        f"{LABEL_A}, H = {model.cal.H_peg}\nnominal coupon vs index-linked uplift",
    )
    _save(fig, dest, "ib_decomposition")


def plot_peg_robustness(model, paths: dict, dest: Path) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    default_h = model.cal.H_peg
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    panels = [
        (axes[0, 0], "Output", "y", False),
        (axes[0, 1], "House prices", "ph", False),
        (axes[1, 0], "Savers $C^{s}$", "cs", True),
        (axes[1, 1], "Borrowers $C^{b}$", "cb", True),
    ]
    for ax, title, name, xlabel in panels:
        _prepare(ax, title, "percent", h, xlabel=xlabel)
        for H in (8, 12, 20, 40):
            path = paths[H]
            is_default = H == default_h
            label = f"H = {H} (default)" if is_default else f"H = {H}"
            _line(
                ax,
                q,
                _pct(path, model, name, h),
                color=PEG_COLOURS[H],
                label=label,
                lw=2.4 if is_default else 1.35,
                zorder=3 if is_default else 2,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _dress(
        fig,
        f"{LABEL_A}\nH = 8, 12, 20 and 40. H = {default_h} is the default.",
        top=0.78,
        hspace=0.55,
        wspace=0.34,
    )
    _legend_under_title(fig, handles, labels, ncol=4)
    _save(fig, dest, "peg_robustness")


def plot_bank_rate(model, path_a, path_free, dest: Path) -> None:
    h = model.cal.irf_horizon
    q = np.arange(1, h + 1)
    H = model.cal.H_peg
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    _prepare(axes[0], "Bank Rate", "annualised bp", h, xlabel=True)
    _line(axes[0], q, _bp(path_a, model, "i", h), color=COLOUR_A, label=LABEL_A)
    _line(axes[0], q, _bp(path_free, model, "i", h), color=COLOUR_FREE, label=LABEL_TAYLOR_FREE, ls="--")
    handles, labels = axes[0].get_legend_handles_labels()

    _prepare(axes[1], "Sterilising residual", "annualised bp", h, xlabel=True)
    _line(axes[1], q, path_a.e_ster[:h] * 400.0, color=COLOUR_A, label=None)
    for ax in axes:
        ax.axvline(H, color="#8c2f39", lw=0.7, ls=":", zorder=1)
        ax.text(
            H + 0.6,
            0.96,
            f"H = {H}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8.5,
            color="#8c2f39",
        )
    _dress(
        fig,
        f"{LABEL_A}, H = {H}\nBank Rate vs Taylor rule free, plus the sterilising residual",
        wspace=0.32,
    )
    _legend_under_title(fig, handles, labels, ncol=2)
    _save(fig, dest, "bank_rate_sterilisation")


def _check_dest(dest: Path) -> Path:
    dest = Path(dest)
    output = (ROOT / "output").resolve()
    if dest.resolve() == output:
        raise ValueError("refusing to write slide figures into output/ (use output/slides)")
    return dest


def write_slides(dest: Path | None = None) -> dict[str, float | int]:
    """Solve the merged model and write the slide figures plus README."""
    dest = _check_dest(ROOT / "output" / "slides" if dest is None else dest)
    with plt.rc_context(SLIDE_RC):
        model, path_a, path_free, path_b, robust, _horizon = compute_paths()
        checks = evaluate_scenario_a(model, path_a, horizon=model.cal.irf_horizon)
        failed = [c for c in checks if not c.ok]
        if failed:
            detail = "; ".join(c.detail for c in failed)
            raise RuntimeError(f"scenario (a) checks failed: {detail}")
        numbers = scenario_numbers(model, path_a, path_b)
        plot_a_macro(model, path_a, dest)
        plot_a_consumption_mortgage(model, path_a, dest)
        plot_a_vs_b(model, path_a, path_b, dest)
        plot_ib_decomposition(model, path_a, dest)
        plot_peg_robustness(model, robust, dest)
        plot_bank_rate(model, path_a, path_free, dest)
        (dest / "README.md").write_text(_readme(numbers), encoding="utf-8")
    return numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write slide-ready IRF figures.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="output directory (default: output/slides)",
    )
    args = parser.parse_args(argv)
    numbers = write_slides(args.dest)
    dest = ROOT / "output" / "slides" if args.dest is None else args.dest
    print(f"Wrote slide figures to {dest}")
    print(
        f"Y trough {numbers['y_trough']:.3f}% at q{numbers['y_trough_q']}; "
        f"P^h trough {numbers['ph_trough']:.3f}% at q{numbers['ph_trough_q']}; "
        f"IB peak {numbers['ib_peak']:.4f} at q{numbers['ib_peak_q']}; "
        f"C^b trough {numbers['cb_trough']:.3f}% at q{numbers['cb_trough_q']}; "
        f"C^s trough {numbers['cs_trough']:.3f}% at q{numbers['cs_trough_q']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
