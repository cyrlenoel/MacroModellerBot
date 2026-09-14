# Dynare / MATLAB scaffold — Bullard HENK soft-landing model

Open-source-friendly Dynare code for the linear NK block of
**Bullard, Grimaud, Salle & Vermandel**, *Soft landing and inflation scares*
(Tinbergen Institute Discussion Paper **TI 2025-001**, Jan 7 2025), targeting the
**Grimaud–Salle–Vermandel Dynare Social Learning toolbox**
(Mendeley `fzmx3vkt66` / JEDC 2025 DOI [10.1016/j.jedc.2024.104984](https://doi.org/10.1016/j.jedc.2024.104984)).

Status legend:

| Tag | Meaning |
|---|---|
| **[VERIFIED]** | Confirmed from open sources (Tinbergen WP text, WU WP 339 PDF, or the Mendeley toolbox zip `V3`) |
| **[PROVISIONAL]** | Reasonable mapping / scaffolding; confirm on your laptop against the toolbox comments |
| **[BLOCKED]** | Needs author replication materials / estimation stack not shipped here |

No PDF is required to *run* once the toolbox is installed. PDFs/text extracts under
`../papers/` are documentation only.

---

## Files in this folder

| File | Role |
|---|---|
| `bullard_henk.mod` | Linear NK + ARMA(1,1) shocks; FIRE/RE nesting of WP eqs (1)–(3) |
| `run_bullard_henk.m` | Path setup → `dynare` → `SL_get_pq` → `SL_chain` → RE vs SL IRFs |
| `params_table1.m` | Table 1 posterior means + `beta=0.99` |
| `README.md` | This file |

---

## Install the Social Learning toolbox (laptop)

1. Download **Code for a Dynare Toolbox for Social Learning Expectations**  
   - Mendeley: <https://data.mendeley.com/datasets/fzmx3vkt66/1>  
   - DOI: [10.17632/fzmx3vkt66.1](https://doi.org/10.17632/fzmx3vkt66.1)  
   - Licence: **CC BY 4.0**  
   - Direct file (zip name `V3.zip`, observed 2024-06-20):  
     `https://data.mendeley.com/public-files/datasets/fzmx3vkt66/files/f9feb7e9-81b9-47cd-8885-250e4d34e6f8/file_downloaded`
2. Unzip. You should see `V3/Toolbox/` (contains `SL_get_pq.m`, `SL_chain.m`, …)
   and example `V3/GSV2023.mod`.
3. Requirements from toolbox `read_me.txt` **[VERIFIED]**:
   - MATLAB known-good: **2021b–2023a**
   - Dynare known-good: **5.00–5.3**
4. Put Dynare’s `matlab/` directory on the MATLAB path (Dynare installer docs).

The Mendeley landing page itself does **not** document function signatures
(description only). The API below is from the zip’s `read_me.txt`, `GSV2023.mod`,
and `Toolbox/SL_*.m`.

---

## How to run (MATLAB)

```matlab
cd('/path/to/bullard-soft-landing/dynare')
% Edit TOOLBOX_PATH inside run_bullard_henk.m first
run_bullard_henk
```

Expected outputs under `dynare/output/`:

- `irf_monetary_re_vs_sl.csv` / `.png`
- `irf_costpush_re_vs_sl.csv` / `.png`

---

## Parameter sources **[VERIFIED]**

From Tinbergen WP TI 2025-001, Table 1 (1985Q1–2023Q4), plus calibrated `beta=0.99`
(WP §3.2). See `params_table1.m`.

| Param | Value | Source |
|---|---:|---|
| β | 0.99 | calibrated |
| κ | 0.2032 | Table 1 mean |
| σ | 1.6993 | Table 1 mean |
| φ_π | 1.7131 | Table 1 mean |
| φ_y | 0.1564 | Table 1 mean |
| ρ_ι | 0.8179 | Table 1 mean |
| ρ_g, ρ_u | 0.7455, 0.6328 | Table 1 |
| μ_g, μ_u | 0.4579, 0.4887 | Table 1 |
| σ_g, σ_u, σ_v | 0.0067, 0.0043, 0.0023 | Table 1 |
| σ_ι, σ_λ | 0.0006, 0.0004 | Table 1 Panel C |
| ρ (fitness), μ (news) | 0.7745, 0.4357 | Table 1 Panel C |

Narrative aside in the WP that cites φ_π≈1.82 is **not** used; **use Table 1**.

---

## Toolbox API — VERIFIED vs PROVISIONAL

### VERIFIED (from WU WP 339 §4.1 + Mendeley `V3`)

Workflow (no special Dynare “SL expectation” syntax):

1. Write a **standard** `.mod` with RE leads (`pi(+1)`, `y(+1)`, …), parameters, shocks.
2. Run `stoch_simul` so Dynare fills `oo_`, `M_`, `options_`.
3. Set `options_.SL.*` and optional `options_.seed_block`.
4. Call sequentially:

```matlab
[mo] = SL_get_pq(oo_, M_, options_);
% cc: Tsample-by-nexo matrix of standardized innovations
[oo] = SL_chain(mo, oo_, M_, options_, cc);
```

| Item | Detail | Source |
|---|---|---|
| `SL_get_pq(oo_,M_,options_)` | Builds policy `mo.PP`, `mo.QQ`, `mo.RR` (+ Jacobians) | `SL_get_pq.m` |
| `SL_chain(mo,oo_,M_,options_,cc)` | Simulates RE + SL paths for shock chain `cc` | `SL_chain.m` |
| `ee = (cc * sqrt(M_.Sigma_e))'` | Scaling inside `SL_chain` (elementwise `sqrt`) | `SL_chain.m` |
| Outputs | `oo.irfs.re`, `oo.irfs.sl`, `oo.irfs.Esl` | `SL_simulxm.m` |
| `options_.SL.N_agents` | Population size J | `GSV2023.mod` |
| `options_.SL.mut_p` | Mutation / news frequency | `GSV2023.mod` |
| `options_.SL.mut_sd` | Mutation std **vector** (length = # forward vars) | `GSV2023.mod` |
| `options_.SL.wed_p` | Tournament participation share | `GSV2023.mod` |
| `options_.SL.rhoGN` | Fitness-decay **vector** (length = # forward vars) | `GSV2023.mod` |
| `options_.SL.Tsample` | Simulation length | `GSV2023.mod` |
| `options_.seed_block` | 1 = freeze SL RNG for optimization | `GSV2023.mod` |
| Model syntax | Same as RE `.mod`; SL applied to **all** forward endo | WU WP 339 §4.1; `GSV2023.mod` |

Example from toolbox `GSV2023.mod` (abridged):

```matlab
options_.SL.N_agents = 300;
options_.SL.mut_p    = .3;
options_.SL.mut_sd   = [0.3; 0.3];
options_.SL.wed_p    = 1;
options_.SL.rhoGN    = [0.8; .8];
options_.SL.Tsample  = 1300;
options_.seed_block  = 0;

[mo] = SL_get_pq(oo_, M_, options_);
cc   = randn(M_.exo_nbr, options_.SL.Tsample)';
[oo] = SL_chain(mo, oo_, M_, options_, cc);
```

IRFs: zero `cc`, place an impulse at a burn-in date, Monte Carlo optional
(see `GSV2023.mod` IRF block). Toolbox also ships `SL_IRF.m` (less complete than
the `GSV2023` pattern; our driver follows `SL_chain`).

### PROVISIONAL (our Bullard-specific choices)

| Choice | Why provisional |
|---|---|
| `mut_sd = [sigma_iota; 0]` | WP: SL on **inflation only**; toolbox mutates every forward var. Zeroing the y-component keeps y-beliefs at SS. |
| Ignoring separate `sigma_lambda` | Toolbox mutations are idiosyncratic (`mut_sd .* randn`); no distinct aggregate λ shock in `options_.SL`. |
| `N_agents = 300` | Toolbox default; WP requires J even but does not estimate J in Table 1. |
| Burn-in `run_init = 300` | Copied from `GSV2023.mod` IRF block, not from Bullard WP. |
| One-std IRFs via `cc(...)=1` | Matches `SL_chain` scaling by `sqrt(Sigma_e)`; not author figures. |
| Fitness uses toolbox PLM | Not a re-implementation of WP eqs (10)–(12); trust `SL_simulxm.m`. |

### Not invented

We did **not** invent Dynare keywords for `E^{SL}`. The `.mod` uses ordinary
`pi(+1)` / `y(+1)`. If a future toolbox release adds markup syntax, update
`bullard_henk.mod` only after confirming against that release’s example `.mod`.

---

## Model equations in the `.mod` **[VERIFIED]**

Matches WP (1)–(3) under the FIRE nesting (`E^{SL}=E`):

```
pi = kappa*y + beta*pi(+1) + u;
y  = y(+1) - (1/sigma)*(i - pi(+1)) + g;
i  = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + ev;
g  = rho_g*g(-1) + eg - mu_g*eg_lag;   % ARMA(1,1)
u  = rho_u*u(-1) + eu - mu_u*eu_lag;   % ARMA(1,1)
```

with `eg_lag = eg`, `eu_lag = eu`, and white-noise `ev`.

---

## Blockers (not included)

| Item | Status |
|---|---|
| Bayesian estimation + inversion filter (Cuba-Borda et al. 2019) | **[BLOCKED]** |
| SPF measurement equations / data construction (WP §3.1, App. A) | **[BLOCKED]** |
| Table 3 historical-shock counterfactuals | **[BLOCKED]** (no public replication package found) |
| Author-comparable P, Q̃ matrices beyond toolbox `mo.PP/QQ/RR` | Use toolbox after install |
| Aggregate news shock λ as separate process | **[PROVISIONAL]** gap vs toolbox |

Related open Python prototypes (FIRE IRFs, finite-J SL micro block, timing
counterfactuals) live under `../src/` — see `../NOTES.md`.

---

## Exact next step for Cyrille (laptop)

1. Install / confirm **Dynare 5.x** on MATLAB path.
2. Download & unzip Mendeley `fzmx3vkt66` → note path to `V3/Toolbox`.
3. Open `run_bullard_henk.m`, set `TOOLBOX_PATH` to that `Toolbox` folder.
4. In MATLAB: `cd` to this `dynare/` folder → run `run_bullard_henk`.
5. Open toolbox `GSV2023.mod` side-by-side; confirm `nfwrd==2` warning does not fire;
   if Dynare lists more forward variables, resize `mut_sd` / `rhoGN`.
6. Optional: run stock `dynare GSV2023` from the toolbox folder once as a smoke test.
7. Only then consider estimation / SPF / Table 3 (still blocked without author codes).

---

## References

- Bullard, Grimaud, Salle, Vermandel (2025), Tinbergen WP TI 2025-001  
  <https://papers.tinbergen.nl/25001.pdf>
- Grimaud, Salle, Vermandel (2025), *A Dynare toolbox for social learning expectations*,  
  JEDC 172, DOI 10.1016/j.jedc.2024.104984
- Open WP for the toolbox: WU Economics WP 339 (2023)  
  <https://research.wu.ac.at/ws/portalfiles/portal/44832177/WP339.pdf>
- Toolbox code: Mendeley DOI 10.17632/fzmx3vkt66.1 (CC BY 4.0)
