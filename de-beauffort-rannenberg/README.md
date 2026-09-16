# de Beauffort & Rannenberg: Fiscal policy and sectoral spillovers in open-economy HANK

Open-source **starter** replication of:

> de Beauffort, C. & Rannenberg, A. (June 2026). Fiscal policy and sectoral spillovers in open-economy HANK. *NBB Working Paper* No. 493.
>
> PDF: https://www.nbb.be/doc/ts/publications/wp/wp493en.pdf  
> NBB page: https://www.nbb.be/en/publications-research/publications/all-publications/fiscal-policy-and-sectoral-spillovers-open

**Do not commit the PDF.** Cite the links above.

**Toolkit:** Python + [Sequence-Space Jacobian](https://github.com/shade-econ/sequence-jacobian) (Auclert, Bardóczy, Rognlie, Straub 2021). The authors thank Adrien Auclert for guidance on SSJ. Checked-in `output/figure3.png` was produced with **sequence-jacobian 1.0.0**.

See **NOTES.md** for verified vs provisional claims, calibration, and blockers.

## What runs now

Linear MIT-shock IRFs of a **two-sector open-economy HANK** (and the nested RANK) to a government spending shock biased toward non-tradables (Table 2 \(\phi_{GT}=0.25\)), plotted in the **Figure 3** layout:

aggregate / tradable / non-tradable consumption and output, the T/NT price ratio, net exports, and \(G\).

```bash
cd de-beauffort-rannenberg
python3 -m pip install -r requirements.txt
python3 src/run_figure3.py
# optional knobs: --sigma-e 0.35 --psi-nfa 1e-3
# faster smoke (no φ_GT=0 variants):
python3 src/run_figure3.py --T 80 --skip-lb
python3 src/test_qualitative.py
```

Writes:

- `output/figure3.png` — main figure
- `output/irf_{ha,ra,ha_lb,ra_lb}.csv` — 20-quarter series in figure units
- `output/figure3_summary.txt` — qualitative sign checks

Seminar slides: `beamer/dbr_seminar.tex` (compile with `pdflatex dbr_seminar.tex` or `latexmk -pdf dbr_seminar.tex` from `beamer/`). Compiled review copy: `beamer/dbr_seminar.pdf`.

## Parameters (cited)

| Symbol | Role | Value | Source |
|---|---|---|---|
| \(\beta\) | discount (RA / HA) | 0.995 / 0.986 | Table 2 |
| \(\theta_w\) | labour-variety elasticity | 3 | Table 2 (50% markup) |
| \(\kappa_w\) | wage PC **slope** | 0.0044 | Table 2 / Lindé et al. (2016) |
| \(\sigma,\varphi\) | EIS, inverse Frisch | 1, 1 | Table 2 |
| \(\rho_e\) | idiosyncratic persistence | 0.968 | Table 2 |
| \(\vartheta\) | sticky-expectation share | 0.935 | Table 2 / Auclert et al. (2020) |
| \(\phi_t\) | tradable consumption share | 0.623 | Table 2 |
| \(\lambda_t\) | T vs NT elasticity | 2.15 | Table 2 |
| \(\phi_d\) | private home bias | 0.57 | Table 2 (12% import/GDP) |
| \(\lambda_d\) | trade elasticity | 1 | Table 2 (Cobb–Douglas) |
| \(\mu_f,\mu_d\) | distribution shares | 0.754, 0.266 | Table 2 |
| \(\zeta,\kappa_\pi\) | Taylor smoothing, inflation | 0.9, 1.5 | Table 2 |
| \(\kappa_b\) | tax response to debt | 0.027 | Table 2 |
| \(\phi_{GT}\) | tradable content of \(G\) | 0.25 | Table 2; HA-LB uses 0 |

Rouwenhorst \(\sigma_e\): Table 2 lists 0.58; this code uses **0.35** so that Table 1 year-1 iMPC and HtM share are matched (see NOTES.md).

## What matches Figure 3 (qualitative)

HANK: **positive** consumption (goods > services), **positive** goods and services output, **decline** in the relative goods price, **deterioration** of net exports, and **positive tradable output even when \(\phi_{GT}=0\)**.

RANK: consumption **falls**; with \(\phi_{GT}=0\), tradable output no longer co-moves positively.

Magnitudes and some dynamic shapes (especially the S-BVAR overlay and the NX recovery) are **not** a claim of exact author numbers.

## Layout

```
de-beauffort-rannenberg/
  NOTES.md
  EXTENDING.md     # how to add another domestic sector later
  README.md
  requirements.txt
  beamer/          # academic seminar slides (dbr_seminar.tex + dbr_seminar.pdf)
  src/
    sectors.py          # TRADABLE / NONTRADABLE labels
    calibration.py      # Table 2 + consistent SS
    households.py       # HA hetblock, RANK Euler, sticky Jacobians
    production.py       # T / NT / distribution prices, wage PCs
    open_economy.py     # trade, UIP, NFA
    fiscal.py           # G, tax rule, debt
    monetary.py         # Taylor + Fisher
    model.py            # SSJ DAG
    empirical_sbvar.py  # G path + digitised S-BVAR overlay
    solve_ssj.py        # SS + linear IRFs
    plot_figure3.py
    run_figure3.py
    test_qualitative.py
  output/
    figure3.png
    irf_*.csv
```
