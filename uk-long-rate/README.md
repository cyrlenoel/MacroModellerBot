# UK long rate and term premium

MVP twin of a linearised UK small-open-economy New Keynesian model with two households (savers and mortgagors), plus a high-frequency gilt-surprise local-projection scaffold.

The structural engine is Dynare (`.mod`). The paths under `output/` were produced by a Python QZ twin of that system, because this environment has no MATLAB, Octave, or Dynare. See **NOTES.md** for the calibration, the sterilisation device, and the success checks.

Three numbers are held fixed:

| Symbol | Value | Role |
|---|---:|---|
| $\lambda_q$ | 0.09 | quarterly remortgage / reset hazard |
| $s^{IL}$ | 0.25 | index-linked share of the gilt stock |
| $D$ | 9.1 | modified duration, years |

## What runs now

```bash
cd uk-long-rate
python3 -m pip install -r requirements.txt   # numpy, scipy, matplotlib
python3 src/run_irf.py
python3 src/lp_proxyvar.py
python3 -m unittest discover -s src -p 'test_*.py'
```

`src/run_irf.py` writes:

- `output/irf_tp_sterilised.csv` and `output/irf_a_tp_sterilised.png` — scenario (a), +100 bp term premium, Bank Rate sterilised
- `output/irf_tp_unsterilised.csv` — the same term-premium innovation with the Taylor rule left on
- `output/irf_short_rate_news.csv` — scenario (b), short-rate news scaled to the same +100 bp long-rate move
- `output/irf_a_vs_b_y_ph.png` — output and house prices, (a) against (b)
- `output/success_checks.txt`

`src/lp_proxyvar.py` does not ship surprise data. With no `data/gilt_surprises.csv` it writes `output/lp_irf_stub_Y_Ph.csv` full of NaNs and exits 0.

## Dynare on a laptop

```matlab
cd('/path/to/uk-long-rate/dynare')
run_uk_long_rate
```

Details are in `dynare/README.md`. Full Bayesian estimation is stubbed. The calibrated steady state is the zero deviation point of the linear model, and the IRF path above is the check.

## Layout

```
uk-long-rate/
  NOTES.md
  README.md
  requirements.txt
  data/            # header-only surprise template; no invented UK data
  dynare/          # uk_long_rate.mod, driver, parameter file
  src/             # Python twin, LP scaffold, tests
  output/          # calibrated IRFs and the NaN LP stub
```

Scenario (c), a UIP risk-premium shock, is an unused exogenous in the `.mod`. The MVP does not add buy-to-let, corporate credit, or a second mortgage cohort.
