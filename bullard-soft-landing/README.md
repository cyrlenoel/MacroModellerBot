# Bullard–Grimaud–Salle–Vermandel: Soft landing and inflation scares

Open-source **starter** replication for the heterogeneous-expectations New Keynesian (HENK) model in:

> Bullard, J., Grimaud, A., Salle, I. & Vermandel, G. (2026). Soft landing and inflation scares. *Journal of Monetary Economics* 157, 103871.
> DOI: https://doi.org/10.1016/j.jmoneco.2025.103871

**Primary open source used here:** Tinbergen Institute Discussion Paper TI 2025-001/VI (7 Jan 2025),
https://papers.tinbergen.nl/25001.pdf — saved under `papers/`.

See **NOTES.md** for verified vs provisional claims, search for replication materials, and related digest papers.

## What runs now

1. Nested **FIRE / RE** 3-equation NK monetary IRFs (`phi_t = 0`).
2. **HENK prototype** — social-learning belief layer + FIRE-vs-SL IRFs with ARMA(1,1) cost-push/demand and richer PLM common forecast (Python / NumPy only; no Matlab).
3. **Timing vs strength counterfactuals** (WP §4 spirit) under a shared cost-push surge and identical SL seed.

```bash
cd /workspace/macro-models/bullard-soft-landing
# NumPy required (often already present); else:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 src/re_nk_irf.py
python3 src/re_nk_irf.py --horizon 40 --shock 1.0 --save output/irf_monetary.csv
python3 src/henk_sim.py --horizon 80 --seed 1 --save-dir output
python3 src/timing_counterfactual.py --horizon 80 --seed 1 --surge-quarters 12 --delay 8
# cost-push IRF chart (needs matplotlib):
python3 src/plot_costpush_irf.py
# optional micro smoke test:
python3 src/social_learning.py
```

`henk_sim.py` writes `output/irf_{monetary,costpush}_{fire,sl,sl_minus_fire}.csv` and `output/fire_vs_sl_summary.txt`.
`timing_counterfactual.py` writes `output/counterfactuals/paths_*.csv`, `comparison_metrics.csv`, and `comparison_summary.txt`.
FIRE paths nest `re_nk_irf` on the monetary A,B block. SL uses WP eqs. (10)–(12) with a documented quasi-RE / martingale-phi approximation — **not** a full Dynare-toolbox replication (see NOTES.md STATUS).

## Parameters (cited)

| Symbol | Role | Value | Source |
|---|---|---|---|
| beta | discount factor | 0.99 | WP section 3.2 calibrated |
| kappa | NKPC slope | 0.2032 | Table 1 posterior mean |
| sigma | inv. intertemporal elasticity | 1.6993 | Table 1 |
| phi_pi | Taylor inflation weight | 1.7131 | Table 1 |
| phi_y | Taylor output weight | 0.1564 | Table 1 |
| rho_i | interest-rate smoothing | 0.8179 | Table 1 |

Shock ARMA coefficients (`rho_u`, `mu_u`, `rho_g`, `mu_g`, …) are listed in NOTES.md / Table 1; `henk_sim` and the counterfactual module use them when `--no-ma` is not set.

## What is still missing (blockers)

1. **Author-comparable SL solution** still needs the Dynare SL toolbox (Grimaud–Salle–Vermandel; Mendeley Data fzmx3vkt66) — Matlab/Octave. Our Python SL is a transparent prototype (see NOTES.md STATUS for provisional choices).
2. **No official replication package** for this paper was found (GitHub / journal supplement / author download).
3. Bayesian **inversion-filter** estimation and SPF observables.
4. JME published PDF not downloaded (paywalled); WP may differ slightly from the journal version.

Equation checklist: `src/henk_equation_skeleton.md`.

Seminar slides: `beamer/henk_seminar.tex` (compile with `pdflatex henk_seminar.tex` or `latexmk -pdf henk_seminar.tex` from `beamer/`).

## Recommended next step

1. Cross-check provisional C/D/G loadings against Dynare SL toolbox P, Q̃, R if Matlab becomes available.
2. Request or locate author replication files for the exact estimated HENK system used in section 4 (historical shocks).
3. Optional: UK / BoE timing analogue (see NOTES.md applied extensions).

## Layout

```
bullard-soft-landing/
  NOTES.md
  README.md
  requirements.txt
  papers/          # Tinbergen WP PDF + text extract
  beamer/          # academic seminar slides (henk_seminar.tex)
  src/
    re_nk_irf.py
    social_learning.py
    henk_sim.py
    timing_counterfactual.py
    plot_costpush_irf.py
    henk_equation_skeleton.md
  output/          # FIRE / SL IRF CSVs + fire_vs_sl_summary.txt
    counterfactuals/  # timing vs strength CSVs + summary
```
