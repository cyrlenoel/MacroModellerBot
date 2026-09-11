# Soft landing and inflation scares — research notes

**Status legend:** `[VERIFIED]` = taken from fetched open sources (Tinbergen WP PDF, RePEc/abstract pages, author sites). `[PROVISIONAL]` = inference needed for replication scaffolding, not a claim that the published JME version matches verbatim. `[BLOCKED]` = not available from public materials found so far.

---

## Bibliographic record

| Field | Value | Source |
|---|---|---|
| Title | Soft landing and inflation scares | JME / Tinbergen WP |
| Authors | James (Jim) Bullard; Alex Grimaud; Isabelle Salle; Gauthier Vermandel | Crossref / WP |
| Venue | *Journal of Monetary Economics*, Vol. 157, Jan 2026, Article 103871 | Crossref DOI |
| DOI | https://doi.org/10.1016/j.jmoneco.2025.103871 | Crossref |
| Open WP | Tinbergen Institute Discussion Paper **TI 2025-001/VI**, dated **January 7, 2025** | https://papers.tinbergen.nl/25001.pdf |
| RePEc (WP) | RePEc:tin:wpaper:20250001 | IDEAS |
| RePEc (JME) | RePEc:eee:moneco:v:157:y:2026:i:c:s0304393225001424 | IDEAS |
| SSRN | https://doi.org/10.2139/ssrn.6384399 (listing found; not used as primary text) | WebSearch |

Open WP PDF: https://papers.tinbergen.nl/25001.pdf (not in this repo).

**JME full text:** paywalled / ScienceDirect returned 403 from this environment. All equation and parameter claims below are tied to the **Tinbergen WP** unless marked otherwise. Author pages (Grimaud, Vermandel) reproduce the abstract in language consistent with the WP/JME abstract.

---

## Core question `[VERIFIED]`

Why did the post-pandemic US inflation surge end in a **soft landing** (disinflation without a major recession), unlike Volcker-era disinflation? The authors ask whether the **timing** and **strength** of the Fed’s reaction matter when **long-run inflation expectations** are **heterogeneous** and can **lose their anchoring** to the target (an “inflation scare”).

---

## Abstract (verbatim from Tinbergen WP / Tinbergen news / IDEAS) `[VERIFIED]`

> We discuss the timing and strength of the Fed’s reaction to the recent inflation surge within an estimated macroeconomic model where long-run inflation expectations are heterogeneous and can lose their anchoring to the target. The resulting inflation scare worsens the real cost of disinflation. We derive a closed-form solution that retains the entire time-varying cross-sectional distribution of subjective inflation beliefs. We estimate the model using Bayesian techniques on both US macroeconomic time series and forecast data from the Survey of Professional Forecasters. Counterfactual simulations show that the timing – rather than the strength – of the policy reaction to the inflation surge is critical to contain the development of an inflation scare and prevent the entrenchment of above-target inflation. We show that the Fed fell behind the curve in 2021 since an earlier tightening could have reduced the inflation peak without triggering a recession. However, further delays would have unanchored inflation expectations, aggravated the inflation scare and strengthened the inflation surge, resulting in larger output losses.

---

## Model ingredients `[VERIFIED` from WP §2]

### Agents / structure
- Stylized **New Keynesian** economy (NKPC + IS + Taylor-type rule).
- Population of **J** agents (J even), identical except for **idiosyncratic subjective beliefs about long-run (steady-state) inflation** \(a_{j,t}\).
- Aggregate subjective inflation expectation \(E_t^{SL}(\hat\pi_{t+1})\) is the cross-sectional average of individual forecasts.
- **Only departure from RE:** information friction on long-run inflation via **social learning (SL)**. Output-gap expectations in the IS equation are treated as **rational / model-consistent** (internal rationality / quasi-RE observer).

### Log-linear block (WP eqs. 1–3 / §2.4 summary)

Notation: hats = deviations from steady state / target.

1. **NK Phillips curve**
   \[
   \hat\pi_t = \kappa\,\hat y_t + \beta\, E_t^{SL}(\hat\pi_{t+1}) + u_t
   \]

2. **IS / aggregate demand** (standard grouping; WP typesets \(\sigma^{-1}(\hat\iota_t - E_t^{SL}\hat\pi_{t+1})\))
   \[
   \hat y_t = E_t(\hat y_{t+1}) - \sigma^{-1}\big(\hat\iota_t - E_t^{SL}(\hat\pi_{t+1})\big) + g_t
   \]

3. **Interest-rate rule**
   \[
   \hat\iota_t = \rho_\iota\,\hat\iota_{t-1} + (1-\rho_\iota)\big(\phi_\pi\,\hat\pi_t + \phi_y\,\hat y_t\big) + v_t
   \]

4. **Subjective vs RE inflation expectations** (WP eq. 8 / §2.4)
   \[
   E_t^{SL}(\hat\pi_{t+1}) = \phi_t + E_t(\hat\pi_{t+1}),
   \qquad
   \phi_t \equiv \tfrac1J\sum_j a_{j,t},\quad
   E_t(\phi_{t+1})=\phi_t
   \]
   Nested FIRE case: \(\phi_t=0\) for all \(t\) ⇒ \(E^{SL}=E\).

### Expectation formation — social learning `[VERIFIED` WP §2.3]
Three-step recursive nonlinear process each period:
1. **News / mutation** (sticky information, Calvo-like probability \(\mu\)):
   \[
   m_{j,t} = a_{j,t-1} + \mathbf{1}_{\varpi_{j,t}\le\mu}(\iota_{j,t}+\lambda_t)
   \]
2. **Fitness** from discounted squared forecast errors over inflation history (decay \(\rho\)).
3. **Tournament**: agents paired; the more accurate belief is copied by both.

Aggregate belief update: \(\phi_t = \phi_{t-1} + S(\cdot)\) with \(S\) nonlinear (no closed form); solved in practice with the authors’ **Dynare SL toolbox** (Grimaud, Salle & Vermandel, JEDC 2024 / forthcoming toolbox paper).

### Shocks `[VERIFIED` WP §2.4]
- Demand: \(g_t = \rho_g g_{t-1} + \varepsilon_t^g - \mu_g\varepsilon_{t-1}^g\) (ARMA(1,1))
- Cost-push: \(u_t = \rho_u u_{t-1} + \varepsilon_t^u - \mu_u\varepsilon_{t-1}^u\) (ARMA(1,1))
- Monetary: \(v_t = \varepsilon_t^v\) (i.i.d.)
- SL news: aggregate \(\lambda_t\), idiosyncratic \(\iota_{j,t}\)

### Estimation sample / observables `[VERIFIED` WP §3]
- Bayesian estimation with **inversion filter** (Cuba-Borda et al. 2019) for the nonlinear SL block.
- Sample cited for Table 1: **1985Q1–2023Q4**.
- Observables include macro series **and** SPF inflation expectations (exact measurement equations in WP §3.1 / App. A).
- Calibrated: \(\beta=0.99\).

### Key posterior means (Table 1, WP) `[VERIFIED]`

| Parameter | Meaning | Posterior mean |
|---|---|---|
| \(\kappa\) | NKPC slope | 0.2032 |
| \(\sigma\) | inv. IES | 1.6993 |
| \(\phi_\pi\) | inflation stance | 1.7131 |
| \(\phi_y\) | output stance | 0.1564 |
| \(\rho_\iota\) | MPR smoothing | 0.8179 |
| \(\rho_g,\rho_u\) | AR demand / cost-push | 0.7455 / 0.6328 |
| \(\mu_g,\mu_u\) | MA demand / cost-push | 0.4579 / 0.4887 |
| \(\sigma_g,\sigma_u,\sigma_v\) | shock stds | 0.0067 / 0.0043 / 0.0023 |
| \(\sigma_\iota,\sigma_\lambda\) | idio. / agg. news std | 0.0006 / 0.0004 |
| \(\rho\) (fitness decay) | | 0.7745 |
| \(\mu\) (news frequency) | | 0.4357 |

---

## Key results `[VERIFIED` from abstract + WP intro/§4 narrative]

1. Model generates **time-varying cross-sectional dispersion** of long-run inflation beliefs; matches (untargeted) SPF dispersion moments reasonably well (WP validation / Table 2).
2. Belief dispersion **amplifies** shock transmission relative to homogeneous / FIRE benchmarks.
3. **Timing of tightening dominates strength** (conditional on not being too weak) in managing expectation unanchoring.
4. Counterfactuals: Fed **behind the curve** in 2021 — earlier tightening could have cut the inflation peak **without a recession**; further delay (~3 quarters in WP narrative) risks **expectation-driven entrenchment**; an early rate cut would worsen the scare.
5. Soft landing is interpreted as successful (if late) management of an inflation scare under HENK, not as FIRE-only dynamics.

---

## Replication materials search `[VERIFIED` search outcome]

| Resource | Found? | Notes |
|---|---|---|
| Tinbergen WP PDF | **Yes** | https://papers.tinbergen.nl/25001.pdf |
| JME PDF | **No** (paywalled / 403) | DOI known |
| Official replication package for *this* paper | **Not found** | No GitHub / journal supplement / author “replication files” link located for Bullard et al. (unlike some other Vermandel papers) |
| Related **Dynare Social Learning toolbox** | **Yes (related)** | Grimaud–Salle–Vermandel Mendeley Data: https://data.mendeley.com/datasets/fzmx3vkt66/1 (DOI 10.17632/fzmx3vkt66.1); JEDC toolbox paper. Authors state they use this toolbox for the HENK solution. |
| Author sites | Abstract + links | https://www.alexgrimaud.com/publications ; https://vermandel.org/travaux-de-recherche/ |

---

## Open questions / blockers for full replication

1. **`[BLOCKED]`** Author replication codes (estimation, filters, counterfactual scripts, data construction) not publicly linked in materials found.
2. **`[BLOCKED]`** Full SL tournament + inversion-filter estimation requires the **Dynare SL toolbox** + Matlab/Octave Dynare stack; not yet vendored here.
3. **`[PROVISIONAL]`** Exact observable definitions, HP/Hamilton filter choices, and measurement of SPF moments — need WP §3.1 / App. A carefully transcribed (text extract is messy for tables/matrices).
4. **`[PROVISIONAL]`** JME published version may differ slightly from Jan 7, 2025 WP (e.g. narrative cites \(\phi_\pi\approx 1.82\) in one place while Table 1 posterior mean is 1.7131 — **use Table 1** until JME PDF is available).
5. Microfoundations App. B (Rotemberg costs ↔ \(\kappa\)) not yet coded.

---

## What this repo implements now

- **Runnable (Python / NumPy only):**
  - nested **FIRE / RE** 3-equation NK IRFs (`src/re_nk_irf.py`);
  - finite-J **social learning** micro block (`src/social_learning.py`);
  - **HENK prototype** FIRE vs SL IRFs with ARMA(1,1) `u`,`g` and richer PLM common forecast (`src/henk_sim.py`);
  - **timing vs strength** counterfactuals (`src/timing_counterfactual.py` → `output/counterfactuals/`).
- **Skeleton / TODOs:** remaining Dynare / estimation items in `src/henk_equation_skeleton.md`.
- **Not runnable yet:** Bayesian inversion-filter estimation, SPF observables, author Dynare SL toolbox matrices.

---

## Applied extension ideas (tied to the paper)

1. **UK / BoE timing exercise.** Re-estimate or re-calibrate the same HENK structure on UK macro + UK SPF/CBI expectations and ask whether **timing vs strength** of BoE hikes 2021–23 similarly dominates inflation-scare containment (direct analogue of WP §4 counterfactuals).
2. **Communication as anchoring device.** Follow the authors’ related SL/ELB work (Arifovic–Grimaud–Salle–Vermandel, *JMCB*): add CB announcements of the target / CB inflation forecasts into the fitness/tournament stage and quantify how much communication substitutes for earlier rate hikes in preventing \(\phi_t\) drift.

---

## Related digest papers (for later; not part of this replication)

### 1. Kaplan & Miyahara — NBER WP 35400 `[VERIFIED` NBER abstract]
- **Title:** *How Does Monetary and Fiscal Policy Affect the Economy in the Face of Large Shocks?*
- **Authors:** Greg Kaplan, Ken Miyahara
- **Link:** https://www.nber.org/papers/w35400 (PDF: https://www.nber.org/system/files/working_papers/w35400/w35400.pdf)
- **One-liner:** HANK (incomplete markets) **plus state-dependent pricing with strategic complementarities**; large 2020 shocks make both ingredients matter for fiscal/monetary transmission (non-Ricardian effects; nonlinear inflation). Quantifies 2020–22 policy paths and distributional welfare.

### 2. Angeletos, Lian, Wolf & Zhang — NBER WP 35642 `[VERIFIED` NBER abstract]
- **Title:** *Monetary-Fiscal Interactions: A Reappraisal*
- **Authors:** George-Marios Angeletos, Chen Lian, Christian K. Wolf, Dalton Rongxuan Zhang
- **Link:** https://www.nber.org/papers/w35642
- **One-liner:** Fiscal dominance channels in **RANK** rely on fragile infinite-horizon demand determination; under a short-run refinement, fiscal effects should be studied via **classical non-Ricardian / HANK** mechanisms (finite horizons, liquidity constraints), not beliefs-at-infinity selection.

**Contrast with Bullard et al.:** Bullard–Grimaud–Salle–Vermandel stress **heterogeneous long-run inflation beliefs / SL** in a (otherwise) representative-agent NK; Kaplan–Miyahara and Angeletos et al. stress **household balance-sheet heterogeneity and fiscal–monetary interaction**. Natural later synthesis: HENK beliefs + HANK fiscal non-Ricardians for post-COVID soft-landing accounting.

---

## Sources fetched this session
- https://papers.tinbergen.nl/25001.pdf (downloaded)
- https://ideas.repec.org/a/eee/moneco/v157y2026ics0304393225001424.html
- https://ideas.repec.org/p/tin/wpaper/20250001.html
- https://tinbergen.nl/news/1172/...
- https://www.alexgrimaud.com/publications
- https://vermandel.org/travaux-de-recherche/
- https://doi.org/10.1016/j.jmoneco.2025.103871 (Crossref metadata)
- https://data.mendeley.com/datasets/fzmx3vkt66/1 (related SL toolbox)
- https://www.nber.org/papers/w35400 ; https://www.nber.org/papers/w35642

---

## STATUS — Python HENK prototype (2026-09-11)

### What works now `[VERIFIED` runnable]

| Artefact | Role |
|---|---|
| `src/re_nk_irf.py` | FIRE / RE 3-eq NK monetary IRFs; Table 1 params |
| `src/social_learning.py` | Finite-J SL: news/mutation (eq. 10), fitness (eq. 11), tournament (eq. 12); `build_plm_common_fcast` helper |
| `src/henk_sim.py` | Couples RE loadings A,B with provisional C(phi), D(u), G(g); ARMA(1,1) paths for u,g; richer fitness common forecast |
| `src/timing_counterfactual.py` | Baseline vs earlier vs stronger-delayed Taylor rules under shared surge path + SL seed |
| `output/irf_*_{fire,sl,sl_minus_fire}.csv` | Monetary + cost-push IRFs |
| `output/fire_vs_sl_summary.txt` | First-12-quarter FIRE vs SL comparison |
| `output/counterfactuals/` | `paths_*.csv`, `comparison_metrics.csv`, `comparison_summary.txt` |

Run:
```bash
python3 src/henk_sim.py --horizon 80 --seed 1 --save-dir output
python3 src/timing_counterfactual.py --horizon 80 --seed 1 --surge-quarters 12 --delay 8
python3 src/social_learning.py   # SL micro smoke test
```

FIRE paths from `henk_sim` **nest** `re_nk_irf` on the monetary A,B block (phi=0, u=g=0).

### Prototype metrics snapshot (2026-09-11, illustrative — `[PROVISIONAL]`)

Cost-push **surge** (12 quarters × 0.01 innovations, MA(1) on, seed=1, J=100, delay=8):

| scenario | peak π | min y | max\|φ\| |
|---|---:|---:|---:|
| baseline (Table 1) | 0.0931 | −0.1687 | 0.0369 |
| earlier (φ_π +10%, ρ_ι ×0.9 from t=0) | 0.0681 | −0.1584 | 0.0338 |
| stronger_delayed (φ_π +10% from t=8) | 0.0752 | −0.1708 | 0.0369 |

Qualitative pattern is **consistent with** WP §4 narrative that **timing** beats delayed **strength** for peak inflation / scare containment in this prototype — **not** a claim of matching author Table 3 / historical-shock counterfactuals.

### Approximation / modelling judgements `[PROVISIONAL]`

1. **Quasi-RE observer:** each period SL sets `phi_t` from past inflation; macro block is linear in `(i_{t-1}, v_t, phi_t, u_t, g_t)` with martingale `E phi'=phi` (WP Assumption 2). Not a simultaneous nonlinear RE+SL fixed point each period.
2. **Phi / cost-push / demand loadings C, D, G:** undetermined coefficients under that martingale structure; intended as R / shock-column analogues of WP eq. (5), **not** the Dynare SL toolbox matrices.
3. **Fitness forecast (toward WP PLM (6)):** `E[pi_s|m] = m + common_fcast_s` with
   `common_fcast = A_pi*i_{s-1} + D_pi*u_s + G_pi*g_s + B_pi*v_s`.
   Full MSV rows P, Q̃ still **not** recoverable from the WP text alone.
4. **Shocks:** simulated `u`,`g` use ARMA(1,1) (`rho`,`mu` from Table 1) when `use_ma=True`; loadings D/G still solved under AR(1) expectations `E x'=rho x`.
5. **Delayed strength:** loadings switch at delay D without agents anticipating the future rule (transparent approximation).
6. **Unit / surge IRFs:** henk_sim default unit impulses; counterfactuals default to a 12-quarter 0.01 surge (not `sigma_u` historical path).

### Still blocked / next

1. **`[BLOCKED]`** Dynare Social Learning toolbox (Mendeley `fzmx3vkt66`) + Matlab — needed for author-comparable P, Q̃, R and inversion-filter estimation.
2. **`[BLOCKED]`** Official replication package / SPF data construction still not found.
3. **`[TODO]`** Cross-check C/D/G vs toolbox; optional MA state in loadings; richer historical-shock counterfactuals if data become available.
4. This remains a **transparent open-source stepping stone**, not a claim of full author replication.
