# HENK equation skeleton (Bullard–Grimaud–Salle–Vermandel, Tinbergen WP 2025-001)

All equations below are **`[VERIFIED]`** from the WP text extract (`papers/wp25001.txt`, §2.4), except items marked **`[TODO]`** / **`[PROVISIONAL]`**.

## Aggregate NK block

```
pi_t = kappa * y_t + beta * E_SL_pi_{t+1} + u_t

y_t  = E_y_{t+1} - (1/sigma) * (i_t - E_SL_pi_{t+1}) + g_t
       # WP groups sigma^{-1}(i_t - E^{SL} pi_{t+1}); see NOTES.md

i_t  = rho_i * i_{t-1} + (1 - rho_i) * (phi_pi * pi_t + phi_y * y_t) + v_t
```

## Belief / SL block

```
E_SL_pi_{t+1} = phi_t + E_pi_{t+1}          # WP eq. (8)
phi_t = (1/J) * sum_j a_{j,t}
E_t phi_{t+1} = phi_t                         # Assumption 2 (martingale / random-walk beliefs)

# Individual PLM — WP eq. (6) [VERIFIED form]:
#   pî_t^{(j)} = a_{j,t} + P_{1,•} ẑ_{t-1} + Q̃_{1,•} E_t
# Prototype fitness uses [PROVISIONAL] stand-in:
#   E[pi_s|m] = m + A_pi*i_{s-1} + D_pi*u_s + G_pi*g_s + B_pi*v_s

# Individual PLM intercept update — WP eqs. (10)–(13):
m_{j,t} = a_{j,t-1} + 1{varpi_{j,t} <= mu} * (iota_{j,t} + lambda_t)
F_{j,t} = - sum_{tau=0}^{t-1} rho^tau * (pi_{t-tau} - forecast(m_{j,t}))^2
# tournament pairing → a_{k,t}, a_{ell,t}
phi_t = phi_{t-1} + S(lambda_t, {iota_j,t}, pi history)
```

## Shock processes

```
g_t = rho_g * g_{t-1} + eps_g_t - mu_g * eps_g_{t-1}   # wired in henk_sim / counterfactuals
u_t = rho_u * u_{t-1} + eps_u_t - mu_u * eps_u_{t-1}   # wired in henk_sim / counterfactuals
v_t = eps_v_t
```

## TODO / blockers for a full open-source HENK solver

1. **`[TODO]`** Port or wrap Grimaud–Salle–Vermandel Dynare Social Learning toolbox  
   (Mendeley: https://data.mendeley.com/datasets/fzmx3vkt66/1) — authors use this for P, Q̃.
2. ~~**`[TODO]`** Implement tournament + fitness on a finite population J (even); store full `{a_j}`~~  
   → **Done (prototype):** `src/social_learning.py` (WP eqs. 10–12; PLM common forecast provisional).
3. **`[TODO]`** Inversion filter (Cuba-Borda et al. 2019) for Bayesian estimation on US + SPF data.
4. **`[TODO]`** Measurement equations / data construction (WP §3.1, App. A) — not yet transcribed into code.
5. ~~**`[TODO]`** Counterfactual policy experiments (timing vs strength of ρ_ι, φ_π) as in WP §4~~  
   → **Done (prototype):** `src/timing_counterfactual.py` (illustrative surge path; not author historical shocks / Table 3).
6. ~~**`[TODO]`** Compare IRFs: FIRE nesting (`src/re_nk_irf.py`) vs HENK with identical structural params~~  
   → **Done (prototype):** `src/henk_sim.py` writes FIRE vs SL CSVs + `output/fire_vs_sl_summary.txt`.
7. ~~**`[TODO]`** Wire MA(1) for `g`,`u`; richer PLM common forecast inside fitness~~  
   → **Done (prototype, 2026-09-11):** ARMA paths + `build_plm_common_fcast`; full P/Q̃ still blocked.
8. **`[TODO]`** Cross-check C/D/G (and MA state) vs Dynare toolbox loadings; optional historical-shock counterfactuals if data appear.

## Runnable today

- `src/re_nk_irf.py`: FIRE nesting (`phi_t = 0`) steady state + monetary IRFs with Table 1 params.
- `src/social_learning.py`: finite-J news → fitness → tournament → `phi_t` (+ PLM common-fcast helper).
- `src/henk_sim.py`: quasi-RE + martingale-phi coupling; FIRE vs SL monetary and cost-push paths  
  (`python3 src/henk_sim.py --horizon 80 --seed 1 --save-dir output`).
- `src/timing_counterfactual.py`: baseline / earlier / stronger_delayed under shared surge  
  (`python3 src/timing_counterfactual.py --surge-quarters 12 --delay 8`).

See **NOTES.md STATUS (2026-09-11)** for provisional modelling choices and remaining blockers.
