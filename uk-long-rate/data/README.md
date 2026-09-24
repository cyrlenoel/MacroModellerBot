# Gilt-surprise data

`gilt_surprises.TEMPLATE.csv` is a header only. It is not a dataset.

Do not fill it with simulated UK surprises and then read the local-projection output as evidence. `src/lp_proxyvar.py` writes a NaN IRF stub until a real file is saved as `gilt_surprises.csv` (that name is gitignored if you create it locally; it is not in the repo).

## Columns

| Column | Role |
|---|---|
| `date` | Event date |
| `event_type` | MPC decision, speech, or other monetary window |
| `bank_rate_surprise_bp` | High-frequency Bank Rate surprise, basis points |
| `ois_path_bp` | Near-term OIS / expected Bank Rate path factor, basis points. This is proxy (b). |
| `gilt_10y_hf_bp` | High-frequency change in the 10-year gilt, basis points |
| `gilt_2y_hf_bp` | Optional. Not required for the MVP orthogonalisation. |
| `y` | Outcome: ONS monthly GDP, percent. Optional until the LP is run for real. |
| `ph` | Outcome: UK house price index (or Halifax / Nationwide), percent. |

Proxy (a) is computed in code: the residual of `gilt_10y_hf_bp` on a constant, `bank_rate_surprise_bp`, and `ois_path_bp`.

## Where the series come from

* Cesa-Bianchi, Thwaites and Vicondoa (2020), *European Economic Review*, “Monetary policy transmission in the United Kingdom: a high frequency identification approach”. Gilt yield changes inside monetary-event windows.
* Bank of England yield curves: https://www.bankofengland.co.uk/statistics/yield-curves
* Bank Rate and OIS surprises around the same windows, so the 10-year move can be orthogonalised to the expected short-rate path.

The model scenarios and these proxies are different objects. Proxy (a) is the empirical analogue of a long-rate move that is not the OIS path. Proxy (b) is the empirical analogue of short-rate news. Neither file is estimated inside the Dynare model in this MVP.
