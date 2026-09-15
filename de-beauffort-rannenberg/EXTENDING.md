# Extending the two-sector DAG

This starter is a **T / NT** open-economy HANK, not a generic N-sector solver.
`src/sectors.py` only names the current sectors:

```python
TRADABLE = "T"
NONTRADABLE = "NT"
DOMESTIC_SECTORS = (TRADABLE, NONTRADABLE)
```

## What is hard-coded T/NT today

| Location | What to duplicate for a third domestic sector `S` |
|---|---|
| `src/model.py` `UNKNOWN_LIST` | `Y_S`, `w_S` (plus existing `pi, Q, i, B, nfa`) |
| `src/model.py` `TARGET_LIST` | `pc_S`; keep **one** goods market dropped by Walras (today `goods_NT`) |
| `src/production.py` `hours`, `mrs_block`, `wage_pcs` | extra hours/MRS/wage-PC residual |
| `src/production.py` `prices` | extra relative price in the CPI CES |
| `src/open_economy.py` `trade_demand` | extra CES demand and an output identity like (34)–(35) |
| `src/calibration.py` `SteadyState` | `Y_S`, `N_S`, `w_S`, `mrs_S`, and any new share/elasticity |
| `src/households.py` income | `Z = Σ_s w_s N_s − T` (already a sum of two terms) |

Labour aggregator (paper eq. 9), government CES (`φ_GT` vs `1-φ_GT`), and the distribution block (NT services bundled into tradable retail prices) are two-sector objects: a new sector needs an explicit nesting (which CES, which distribution, tradable or not).

## Suggested next step (not in this PR)

Keep SSJ simple blocks, but generate the per-sector wage PC / hours / demand lines from `DOMESTIC_SECTORS` instead of copy-pasting `T`/`NT`. Do not collapse trade, UIP, and the HA Jacobian into one monolithic N-sector script.

Walras reminder: household budget + government budget + NFA law of motion imply aggregate goods clearing, so **one** sectoral goods residual is omitted from `TARGET_LIST` (currently `goods_NT`). Adding a sector adds one `Y_S` unknown **and** one extra goods residual, still dropping exactly one by Walras.
