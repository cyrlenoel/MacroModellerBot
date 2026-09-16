# Fiscal policy and sectoral spillovers in open-economy HANK — research notes

**Status legend:** `[VERIFIED]` = taken from NBB WP 493 (June 2026). `[PROVISIONAL]` = modelling choice required to run a complete SSJ DAG, not a claim that the authors’ files match verbatim. `[BLOCKED]` = not available from the public WP.

Paper (do **not** commit the PDF):

- https://www.nbb.be/doc/ts/publications/wp/wp493en.pdf
- https://www.nbb.be/en/publications-research/publications/all-publications/fiscal-policy-and-sectoral-spillovers-open

---

## Bibliographic record `[VERIFIED]`

| Field | Value |
|---|---|
| Title | Fiscal policy and sectoral spillovers in open-economy HANK |
| Authors | Charles de Beauffort (NBB); Ansgar Rannenberg (ECB) |
| Venue | NBB Working Paper No. 493, June 2026 |
| JEL | E62, F41, E21, C11, C32 |

The authors thank Adrien Auclert for guidance on the sequence-space Jacobian method.

---

## Core question `[VERIFIED]`

Government spending is concentrated in non-tradable services. Empirically, a spending shock still raises private consumption, raises goods **and** services output, **lowers** the relative price of goods, and **worsens** net exports. A two-sector open-economy HANK with imperfect labour mobility and distribution costs can replicate those co-movements; the representative-agent version cannot.

---

## Empirical S-BVAR `[VERIFIED` §2]

Blanchard–Perotti identification, US 1954Q1–2019Q4, nine variables, four lags, Normal-Inverse-Wishart prior. Shock: orthogonal **1% of output** increase in government spending. Figure 1 / Figure 3 black line: the four stylised facts listed in the abstract.

The theoretical $G$ path is the empirical IRF (“does not follow a hump-shaped path”, §4.3). This repo uses a 20-quarter path digitised from Figure 3’s $G$ panel `[PROVISIONAL]`.

---

## Model ingredients `[VERIFIED` §3]

### Households
- Incomplete markets, borrowing constraint $a\geq 0$, idiosyncratic $e_t$ AR(1).
- Hours set by sectoral unions, $n_{S,t}(i)=N_{S,t}$ (eq. 1). Post-tax labour income $z_t(i)=(w_T N_T+w_{NT}N_{NT}-T_t)e_t(i)$.
- Budget (2), utility (4) with $\sigma=\varphi=1$.
- CES demands (5)–(8): $T$ vs $NT$ with $\lambda_t$; home vs foreign with trade elasticity $\lambda_d=1$ (Table 2). Eqs. (7)–(8) typeset $\lambda_t$; we use $\lambda_d$ for D/F as in the Table 2 / §4.1 discussion `[PROVISIONAL` parsing].
- Sticky expectations, matrix (13), Jacobian map in footnote 4 / Auclert–Rognlie–Straub (2020).

### Production and labour
- Linear technology $Y_S=N_S$, **flexible prices**, $P_Z=W_S$ (eq. 14). Sticky wages only (baseline). Sticky prices: Appendix C, not coded.
- Imperfect labour mobility: hours aggregator (9), Horvath $\lambda_l=1$.
- Sectoral wage Phillips curves (12), slope 0.0044 (not the Rotemberg $\psi_w$ level).

### Fiscal / monetary
- $G$ exogenous; CES split (16)–(19) with $\phi_{GT}=0.25$, $\phi^G_d=1$.
- Budget (20); tax rule (21) $\kappa_b=0.027$.
- Taylor (22) $\zeta=0.9$, $\kappa_\pi=1.5$.

### Open economy
- Mutual fund / UIP (23)–(26) with $\Gamma_t=\exp(-10^{-9} nfa_t)$.
- Distribution (Corsetti–Dedola) (28)–(31); $\mu_f=0.754$, $\mu_d=0.266$.
- $X_t=C^*_{F,t}/(1+\mu_f)$, $M_t=(C_{F,t}+G_{F,t})/(1+\mu_f)$.
- Output identities (34)–(35); goods/asset clearing (36)–(37).

### Solution
Perfect-foresight MIT shocks, sequence-space Jacobians, linearised (Auclert et al. 2021).

---

## Table 2 vs what we feed the computer

All Table 2 numbers are stored in `src/calibration.py:TABLE2`.

| Item | Status |
|---|---|
| $\beta_{RA},\beta_{HA},\theta_w,\kappa_w,\sigma,\varphi,\rho_e,\vartheta$ | `[VERIFIED]` |
| Trade shares/elasticities $\phi_t,\lambda_t,\phi_d,\lambda_d,\lambda_l,\mu_f,\mu_d$ | `[VERIFIED]` |
| Taylor / tax / $\phi_{GT},\phi^G_d$ | `[VERIFIED]` |
| $\sigma_e=0.58$ as SSJ Rouwenhorst **unconditional** SD of $\log e$ | **Misses Table 1** ($iMPC_1\approx 0.24$, $HtM\approx 20\%$) |
| $\sigma_e=0.35$ used here | `[PROVISIONAL]` chosen to hit Table 1 $iMPC_1\approx 0.51$ and $HtM\approx 49.7\%$ given $\beta_{HA}$ and $r=1/\beta_{RA}-1$ |
| $G/Y\approx 0.214$ | `[PROVISIONAL` as a named parameter] uniquely implied by 12% import/GDP, $\phi_d,\phi_t,\mu_f$, NX=0, $Y=1$ |
| $\phi_l$ | Table 2 = 0.6; we set $\phi_l=N_{NT}/(N_{NT}+N_T)\approx 0.617$ so both wage PCs hold at $w=1$ (same targeting sentence as §4.1) |
| Eq. (34) distribution on $G_D$ | `[PROVISIONAL]` included so labour income = GDP; coincides with (34) when $\phi_{GT}=0$ |
| UIP wedge $\psi_{nfa}=10^{-3}$ | `[PROVISIONAL]` numerical; paper $10^{-9}$ |
| Foreign export demand | `[PROVISIONAL]` SOE: $C^*$ fixed, foreign retail price of home goods includes distribution |
| SSJ timing of $r$ (ex-post = lagged ex-ante) | `[PROVISIONAL]` Auclert convention; WP typesetting mixes $r_t$ and $r_{t-1}$ |
| S-BVAR overlay in `empirical_sbvar.py` | `[PROVISIONAL]` hand-read from Figure 3, not the authors’ series |

---

## Figure 3 — what this repo matches

Paper Figure 3 (3×3): $C, C_T, C_{NT}, Y, Y_T, Y_{NT}$, price ratio $T/NT$, NX, $G$. Lines: HA, RA, HA-LB ($\phi_{GT}=0$), RA-LB, S-BVAR.

**Qualitative co-movements (this code, `output/figure3_summary.txt`):**

| Fact | HA | HA-LB | RA | RA-LB |
|---|---|---|---|---|
| $C>0$ | yes | yes | no (falls) | no |
| $Y_T>0$ | yes | yes | yes (from $\phi_{GT}$) | **no** |
| $Y_{NT}>0$ | yes | yes | yes | yes |
| $P_T/P_{NT}$ down | yes | yes | yes (smaller) | yes |
| NX down | yes | yes | no (tiny +) | no |
| $C_T>C_{NT}$ response | yes | yes | yes (less negative $C_T$) | yes |

This is the paper’s main message: heterogeneity delivers the consumption-led spillover to tradables; RANK does not, especially when $G$ is purely non-tradable.

**Not claimed:** exact author IRF numbers, the S-BVAR 68% bands, Figure 4 international panel (coded internally but not the main artefact), Appendices B–D.

Known shape gaps vs the published figure: HA consumption is a bit too front-loaded (sticky-info hump is milder than in the paper); HA $Y_T$ does not reproduce the empirical hump; HA NX stays negative too long (the authors note the same medium-run NX undershoot); RANK NX is a small surplus rather than a small deficit.

---

## Blockers for a full author replication

1. `[BLOCKED]` No official replication package / S-BVAR data / exact $G$ IRF vector.
2. `[BLOCKED]` Foreign block only partially specified in the WP (export demand).
3. `[PROVISIONAL]` $\sigma_e$ vs Table 1 tension under the SSJ Rouwenhorst convention.
4. Appendix C sticky prices and Appendix D 25% import share: not in the baseline script (the DAG is extensible).

---

## Sources
- NBB WP 493 PDF (read from the attached upload; not stored in git)
- https://github.com/shade-econ/sequence-jacobian (`hh_sim` hetblock)
- Auclert, Rognlie, Straub (2020), *Micro jumps, macro humps*, sticky-Jacobian mapping (24)
