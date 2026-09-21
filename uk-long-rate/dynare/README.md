# Dynare twin — UK long rate

Linear small-open-economy NK model with a saver / mortgagor split. The equation system is the same object as `src/linear_model.py`.

This environment has no MATLAB, Octave, or Dynare. The IRFs checked into `../output/` come from the Python QZ solver. Run the `.mod` on a laptop to cross-check.

## Files

| File | Role |
|---|---|
| `uk_long_rate.mod` | `model(linear)`, 36 equations, hold-fixed `lambda_q`, `s_IL`, `D` |
| `params_calibration.m` | The same numbers as `src/calibration.py` |
| `run_uk_long_rate.m` | Anticipated Bank Rate peg for scenario (a) |

`stoch_simul` inside the `.mod` is an unsterilised preview. Scenario (a) is the perfect-foresight peg in the driver, not the `e_tp` impulse response from `stoch_simul`.

## How to run

```matlab
cd('/path/to/uk-long-rate/dynare')
% Dynare's matlab/ folder must be on the path.
run_uk_long_rate
```

The driver writes `../output/dynare_irf_tp_sterilised.csv`. Compare Bank Rate (should be numerically zero over 40 quarters) and the long rate (100 annualised bp on impact) with `../output/irf_tp_sterilised.csv`.

Dynare 4.6 and 5 store `oo_.exo_simul` with `T+2` rows: period 0, the `T` simulation dates, and a terminal row. The driver detects that layout and also accepts a `T`-row layout.

## Sterilisation

Anticipated Bank Rate peg: the Taylor rule stays in the model. For the first `H = 40` quarters the driver solves

$$
\varepsilon^{\mathrm{ster}}_t = -i^{\mathrm{TR}}_t
$$

so that Bank Rate stays on the steady-state path. The residuals are known at date 0. After quarter 40 the residual is zero and the Taylor rule resumes. A permanent peg is not used, because it leaves inflation undetermined.

## Estimation

Not run. `lambda_q`, `s_IL`, and `D` are not estimated parameters. A commented note at the bottom of the `.mod` says what a later `estimated_params` block is allowed to free.

## Backup engines

Dynare.jl can target the same `.mod` later. It is not required. Sequence-space Jacobians are not the engine for this MVP.
