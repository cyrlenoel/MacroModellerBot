# How Firms Form Beliefs and the Implications for Inflation — research notes

**Status legend:** `[VERIFIED]` = from FEDS 2026-053 PDF. `[PROVISIONAL]` = scaffolding for open-source replication. `[BLOCKED]` / `[TODO]` = next work.

Updated: 2026-09-21 (Europe/London). Pipeline step: Bullard Dynare SL still blocked → 3-eq NK IRF under hybrid PC (real-rate rule).

---

## Bibliographic record `[VERIFIED]`

| Field | Value |
|---|---|
| Title | How Firms Form Beliefs and the Implications for Inflation |
| Authors | Robert Minton; Hugo Monnery |
| Venue | Finance and Economics Discussion Series **2026-053** (Fed Board) |
| Date | July 1, 2026 |
| DOI | https://doi.org/10.17016/FEDS.2026.053 |
| HTML | https://www.federalreserve.gov/econres/feds/how-firms-form-beliefs-and-the-implications-for-inflation.htm |
| PDF | https://www.federalreserve.gov/econres/feds/files/2026053pap.pdf |

Local PDF and text extract are intentionally omitted from git. Download the PDF from https://www.federalreserve.gov/econres/feds/files/2026053pap.pdf.

---

## Core mechanism `[VERIFIED]`

Firms’ **own nominal marginal-cost forecasts** (Atlanta Fed BIE), not CPI beliefs, are the primitive for Calvo pricing. Five facts: CPI ↔ own-cost disconnect; overreaction to own past costs; stable across sectors/size; inertial beliefs; underreaction to aggregate signals until own costs move.

Beliefs mix a small FIRE weight with Adaptive Learning (AL) on a perceived AR(1) persistent cost component (Kalman gain \(\tilde K\)). Under this microfoundation the NKPC is **steeper and less forward-looking** than FIRE: supply shocks more inflationary, demand shocks less so; long-horizon forward guidance loses bite, near-term guidance strengthens.

Team one-liner (EconWriter 2026-09-15): firms forecast own MC, overreact to those and underreact to aggregates until costs move — NKPC steepens, less forward-looking; supply shocks inflate more; long-horizon FG loses bite.

---

## Applied extension idea

Drop the calibrated hybrid NKPC (eqs. 36–42) into the Bullard HE-NK block in place of the FIRE/SL inflation equation, then re-run timing-vs-strength counterfactuals: does earlier tightening still dominate when price-setters learn from own costs rather than (or in addition to) social learning over long-run \(\pi\) beliefs?

---

## Belief block (quarterly) `[VERIFIED` §2.1–2.2]

Perceived PLM for firm \(i\) cost growth:
$$
\Delta mc_{i,t} = \Delta\tilde{mc}_{i,t}^{p} + \tilde\eta_{i,t},
\qquad
\Delta\tilde{mc}_{i,t}^{p} = \tilde\rho\,\Delta\tilde{mc}_{i,t-1}^{p} + \tilde\varepsilon_{i,t}
$$

AL forecast and Kalman update:
$$
\tilde{E}_{i,t}[\Delta mc_{i,t+\tau}] = \tilde\rho^{\tau}\,\tilde{E}_{i,t}[\Delta\tilde{mc}_{i,t}^{p}]
$$
$$
\tilde{E}_{i,t}[\Delta\tilde{mc}_{i,t}^{p}] = \tilde K\,\Delta mc_{i,t} + (1-\tilde K\tilde\rho)\,\tilde{E}_{i,t-1}[\Delta\tilde{mc}_{i,t-1}^{p}]
$$

Hybrid forecast:
$$
E_{i,t}[\Delta mc_{i,t+\tau}] = \alpha_{\mathrm{FIRE}} E_t[\Delta mc_{i,t+\tau}] + (1-\alpha_{\mathrm{FIRE}})\,\tilde{E}_{i,t}[\Delta mc_{i,t+\tau}]
$$

**Quarterly calibration** (baseline instruments; §2.2):
$$
\alpha_{\mathrm{FIRE}} = 0.25,\quad \tilde\rho = 0.82,\quad \tilde K = 0.22
$$
Equivalent MA form (paper eq. 21):
$$
E_{i,t}[\Delta mc_{i,t+\tau}] = 0.25\,E_t[\Delta mc_{i,t+\tau}] + 0.16\times 0.82^{\tau}\sum_{\ell=0}^{\infty} 0.64^{\ell}\,\Delta mc_{i,t-\ell}
$$

---

## Aggregate NKPC under AL `[VERIFIED` Theorem 3 / eqs 36–42]

Average persistent-cost belief:
$$
b_t = \tilde K\,\Delta mc_t + (1-\tilde K\tilde\rho)\,b_{t-1}
$$

Extra future discount \(\delta\in(\theta_p,1)\) solves the fixed point (39); weight on \(b_t\) is \(\omega_b\) (38). Recursive form (42):
$$
\pi_t - \omega_b b_t = \Bigl(\tfrac{1-\theta_p}{\theta_p} + \beta(\theta_p-\delta)\Bigr) mc_t^{\mathrm{real}} - (1-\delta)\beta E_t mc_{t+1}^{\mathrm{real}} + \beta\delta E_t[\pi_{t+1}-\omega_b b_{t+1}]
$$

FIRE recovered at \(\alpha_{\mathrm{FIRE}}=1\) (\(\delta=1\), \(\omega_b=0\)).

Default Calvo numbers used in paper illustrations: \(\beta=0.99\), \(\theta_p=0.9\).

---

## Incremental advance 2026-09-18

- `papers/README.md` points at the Fed PDF; the PDF and text extract are not in git.
- `src/nkpc_belief_skeleton.py`: belief recursion $b_t$, $\delta$ fixed-point solver, $\omega_b$, and a one-period hybrid NKPC residual smoke (FIRE nest check).

## Incremental advance 2026-09-21

- `src/nk3_hybrid_irf.py`: sequence-space IRFs under a **real-rate rule** (common $y$ across belief models), FIRE vs hybrid PC (42)+(37).
- Supply: AR(1) cost path — impact $\pi$ hybrid/FIRE $\approx 1.43$ (overreaction).
- Demand: real-rate easing + sticky-wage lag $\rho_w=0.9$ so MP is slow to hit mc — impact $\pi$ hybrid/FIRE $\approx 0.86$ (underreaction). Matches §4.1 / Fig 6 qualitative signs; not a full sticky-wage or Taylor-rule (D.17) replication.
- Artefacts: `output/nk3_hybrid_irf.npz`, `output/nk3_hybrid_irf_impact.csv`.

### Still next

1. Optional: Taylor-rule ($\phi_\pi=1.5$) IRF (paper Fig D.17) or tighten sticky-wage block toward $\kappa_w=\kappa_{p,\mathrm{FIRE}}$.
2. Optional bridge into `bullard-soft-landing` once Dynare SL toolbox / TOOLBOX_PATH unblocks (or in parallel as Python-only hybrid PC swap).
3. Do **not** start full BIE micro replication — beliefs stay parametric at calibrated $(\alpha_{\mathrm{FIRE}},\tilde\rho,\tilde K)$.
