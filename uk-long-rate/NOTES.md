# UK long rate — sketch notes

**Status legend:** `[FIXED]` = held at the signed-sketch value. `[SKETCH]` = implemented at sketch fidelity, not estimated. `[PROVISIONAL]` = a modelling choice that makes the linear IRFs qualitatively usable, not a UK estimate. `[CHECK]` = a number produced by the Python QZ twin of `dynare/uk_long_rate.mod` on the default calibration.

The system is linear. Quantities ($Y$, $C^s$, $C^b$, $C$, $I$, $Q$, $K$, $H$, $P^h$, the real exchange rate) are percent deviations. Rates and inflation are quarterly percent, so $0.25$ quarterly percent is $100$ annualised basis points. Ratios ($NX$, $IB$, debt, NFA, taxes) are percentage points of quarterly GDP, which for a flow is the same as percentage points of annual GDP.

---

## What the two experiments are

They are different shocks.

**(a) Term premium, sterilised only.** $TP_t$ jumps and $R^L$ rises with it. Bank Rate is sterilised for $H = 12$ quarters by an anticipated monetary residual. This is the baseline. `stoch_simul` and `output/irf_tp_unsterilised.csv` are **not** scenario (a): both leave the Taylor rule free.

**(b) Short-rate news.** The news state $\nu_t$ shifts the Taylor rule. Bank Rate is not sterilised. The innovation is scaled so $R^L$ still rises $100$ bp on impact, which makes the Bank Rate path much larger than in (a), because the expectations-hypothesis weight on the current short rate is only $1/(4D)$.

**(c)** A UIP risk-premium shock, $\rho^{fx}_t$, is in the `.mod` and is not fired.

---

## Sterilisation `[SKETCH]`

Every scenario-(a) figure and CSV carries this device:

> anticipated Bank Rate peg: the Taylor rule is left in place and the monetary residual is set to $\varepsilon^{\mathrm{ster}}_t = -i^{TR}_t$ for the first $H$ quarters, with $i^{\mathrm{path}}_t = 0$, so $i_t = \rho_i i_{t-1} + (1-\rho_i)(\varphi_\pi \pi_t + \varphi_y Y_t) + \psi_{\mathrm{nfa}} E_t nfa_{t+1} + \nu_t + \varepsilon^{\mathrm{ster}}_t = 0$; residuals are anticipated (perfect foresight) and the Taylor rule resumes after $H$. Scenario (a) is this sterilised path only.

In quarterly-percent units the rule in the model is

$$
i_t = \rho_i i_{t-1} + (1-\rho_i)\big(\varphi_\pi \pi_t + \tfrac{1}{4}\varphi_y Y_t\big) + \psi_{\mathrm{nfa}} E_t nfa_{t+1} + \nu_t + \varepsilon^{\mathrm{ster}}_t.
$$

$\varphi_y = 0.125$ is the annual convention, so the output coefficient in quarterly percent is $\varphi_y/4$. $\psi_{\mathrm{nfa}} = 0.08$ `[PROVISIONAL]` is an external-balance term inside $i^{TR}$. It is part of the rule the residual offsets; it is not a second instrument. Default $H = 12$. The reported window is still 40 quarters, so the Taylor rule is back on for the last 28 quarters of the figure. A permanent peg is not used: it would leave the inflation path undetermined.

The same term-premium innovation with $\varepsilon^{\mathrm{ster}} = 0$ is written to `output/irf_tp_unsterilised.csv` so the peg is visible. **That file is not scenario (a).** Neither is `stoch_simul`.

---

## Hold-fixed calibration

$$
\lambda_q = 0.09, \qquad s^{IL} = 0.25, \qquad D = 9.1.
$$

These are not free parameters in the `.mod` or in `src/calibration.py`.

### $\lambda_q$ `[FIXED]`

$\lambda_q = 0.09$ is the sketch’s quarterly remortgage / reset hazard. A constant hazard of that size has mean spell $1/0.09 \approx 11.1$ quarters, and an annual reset probability $1 - 0.91^4 \approx 31\%$.

The sketch points at three public sources for the magnitude. It does not claim that any one table cell equals $0.09$.

* MLAR, 2024 Q4 (Bank of England publication, FCA commentary of 11 March 2025): the great majority of gross advances were fixed-rate, a large share of balances outstanding were still on fixes, and owner-occupier remortgages were a substantial part of new advances. https://www.bankofengland.co.uk/statistics/mortgage-lenders-and-administrators/2024/2024-q4 and https://www.fca.org.uk/data/commentary-mortgage-lending-statistics-q4-2024
* Financial Stability Report, July 2025: a large minority of mortgage accounts had not yet refinanced, and a further substantial share was expected to refinance onto higher rates between mid-2025 and 2028 Q2. https://www.bankofengland.co.uk/financial-stability-report/2025/july-2025
* UK Finance is the sketch’s third source for product-transfer and remortgage flow. No UK Finance statistic is invented here.

One hazard updates both the stock mortgage rate and the LTV debt stock. There is no second cohort.

### $s^{IL}$ `[FIXED]`

$s^{IL} = 0.25$ is a stock share, rounded from the Debt Management Report 2026-27 figure of about $25.2\%$ of the wholesale debt portfolio in index-linked gilts (nominal uplifted stock about £688.5bn at end-2025).

It is not the planned issuance share. That share is $9.3\%$ (£23.5bn of £252.1bn in the report, later a revised financing remit that is still about $9.3\%$). Setting $s^{IL} = 0$ would drop the index-linked block the sketch asks to keep.

Sources: https://www.gov.uk/government/publications/debt-management-report-2026-27/debt-management-report-2026-27 and https://www.dmo.gov.uk/media/0uuiqi10/drmr2627.pdf (section A.11).

### $D$ `[FIXED]`

$D = 9.1$ years is the sketch’s modified duration. The DMO Quarterly Review for January–March 2026 does not print a single “9.1” cell. The 9.1 figure is the conventional-plus-index-linked weighted average, rounded.

Net market values at 31 December 2025, from https://www.dmo.gov.uk/media/dqlnqypb/jan-mar-2026.pdf :

* conventional gilts £1,674.59bn, modified duration 7.63 years
* index-linked gilts £564.55bn, modified duration 13.61 years

$$
\frac{1674.59 \times 7.63 + 564.55 \times 13.61}{1674.59 + 564.55} \approx 9.14.
$$

The sketch rounds that to $9.1$. End-March 2026 net weights in the same review give a weighted duration of about $9.0$ years. The model holds $9.1$.

The quarterly refix weight is

$$
\delta_D = \frac{1}{4D} = \frac{1}{36.4} \approx 0.027473.
$$

---

## Core relations `[SKETCH]`

**Long rate.** Expectations hypothesis plus a term premium:

$$
R^{EH}_t = \delta_D\, i_t + (1-\delta_D)\, E_t R^{EH}_{t+1}, \qquad R^L_t = R^{EH}_t + TP_t.
$$

Shock (a) innovates $TP$. Shock (b) innovates the short-rate news state inside the Taylor rule. $TP$ persistence $\rho_{TP} = 0.93$ is the illustrative scenario, not a hold-fixed parameter.

**Mortgage flow and stock.**

$$
R^m_t = \omega_S i_t + \omega_L R^L_t + \zeta_t, \qquad \omega_S + \omega_L = 1,
$$

$$
R^{m,\mathrm{stock}}_t = (1-\lambda_q) R^{m,\mathrm{stock}}_{t-1} + \lambda_q R^m_t.
$$

Debt service uses $R^{m,\mathrm{stock}}$. New borrowing and the housing user cost use $R^m$. Provisional weights: $\omega_S = 0.30$, $\omega_L = 0.70$.

**Borrower.** Cashflow, income, and a slow LTV / collateral term:

$$
C^b_t = \alpha_y Y_t - \mu_{ds} R^{m,\mathrm{stock}}_t + \mu_{\mathrm{coll}} \lambda_q \big(P^h_t - B^m_{t-1}\big).
$$

**Saver.** A level rule, not a forward habit Euler. The Euler

$$
(1+h_s) C^s_t = h_s C^s_{t-1} + E_t C^s_{t+1} - \mathrm{eis}\,(i_t - E_t\pi_{t+1}) + \mathrm{inc}_t
$$

has roots $1$ and $h_s$ when $\mathrm{inc}$ is shut off. A negative mark-to-market dip is then echoed by a positive tail, and a one-period $\Delta TP$ term books a capital gain as soon as the premium decays. Under the peg the short real rate is not the tightening: $i_t = 0$ while $\pi$ falls, so intertemporal substitution postpones consumption and the near-unit root carries that postponement into a later boom. Coupon income in the same Euler integrates the slow refix of $IB$ into the same boom. That is why the previous calibration had peak $C^s \approx +0.5\%$ on a pure term-premium rise.

The rule in the model holds the level:

$$
C^s_t = h_s C^s_{t-1} - \psi^{TP} TP_t - \mathrm{eis}\,(i_t - E_t\pi_{t+1}) + \psi^{y} Y_t + \psi^{IB} IB_t - \psi^{\tau} \tau_t,
$$

with $\mathrm{eis} = (1-h_s)/\sigma$. Provisional values: $h_s = 0.40$, $\sigma = 6$ so $\mathrm{eis} = 0.10$, $\psi^{TP} = 2$, $\psi^{y} = 0.10$, $\psi^{IB} = 0.02$, $\psi^{\tau} = 0.12$. $\psi^{TP} > 0$ means a higher term premium is a duration loss and cuts $C^s$. $\psi^{IB}$ is a small coupon pass-through and does not overturn that loss. There is no $\Delta TP$ capital-gain term.

Taking $C^s$ off the unit-root Euler drops one stable root. $\psi_{\mathrm{nfa}} E_t nfa_{t+1}$ inside the Taylor rule puts it back (36 stable roots, 6 finite unstable). That loading is a provisional SOE closure, not a description of the MPC's remit. Sterilisation still sets $\varepsilon^{\mathrm{ster}}_t = -i^{TR}_t$, and $i^{TR}$ includes the term.

**Housing** `[PROVISIONAL]`. A sticky user-cost / housing Phillips curve, so the house-price trough is not forced onto the impact date the way a pure flexible asset price would be. $\varphi_h = 0.75$, $\kappa_c = 0.055$, $\mu_{\mathrm{coll}} = 0.10$ and $\kappa_{\mathrm{level}} = 0.05$ are dialled down from the first pass, which produced a trough of about $-9.5\%$ per $+100$ bp. That was far too large next to UK housing responses of a few percent per percentage point of Bank Rate. These loadings are not estimates.

$$
(1+\beta+\kappa_{\mathrm{level}}) P^h_t = P^h_{t-1} + \beta E_t P^h_{t+1} + \kappa_c C^b_t - \varphi_h \big(R^m_t - E_t\pi_{t+1}\big) - \kappa_H H_t.
$$

**Capital.** End-of-period $q$, with contemporaneous marginal product. A lead of output in this equation added an extra explosive root and broke Blanchard–Kahn.

$$
Q_t = \beta E_t Q_{t+1} + (1-\beta)\, mpk_t - \varphi_q \big(\omega^f_S (i_t - E_t\pi_{t+1}) + \omega^f_L (R^L_t - E_t\pi_{t+1})\big).
$$

**Interest burden.** Do not double-count $s^{IL}$. $B^{\mathrm{nom}} = (1-s^{IL}) B^g$ and $B^{IL} = s^{IL} B^g$, with $B^g$ equal to $4$ quarters of GDP (about $100\%$ of annual GDP), so $b^{\mathrm{nom}} = 3$ and $b^{IL} = 1$ in pp-of-quarterly-GDP units per quarterly-percent of coupon.

$$
IB_t = b^{\mathrm{nom}} R^{\mathrm{eff}}_t + b^{IL}\big(r^{IL}_t + \pi_t\big) + \frac{i_{ss}}{100} \widetilde{B}^g_t.
$$

The conventional coupon refixes at the duration speed,

$$
R^{\mathrm{eff}}_t = (1-\delta_D) R^{\mathrm{eff}}_{t-1} + \delta_D R^L_t,
$$

so on impact, from a zero lag, $R^{\mathrm{eff}}_0 / R^L_0 = 1/(4D)$. The index-linked real coupon refixes toward $R^L - E\pi$. Real debt accumulates conventional coupons and the real IL coupon, and inflation erodes only the conventional stock. The inflation uplift is inside reported $IB$, not a second addition to real IL debt.

**UIP.** A rise in $rer$ is a sterling depreciation.

$$
rer_t = E_t rer_{t+1} - (i_t - E_t \pi_{t+1}) + \chi_{\mathrm{nfa}}\, nfa_t + \rho^{fx}_t.
$$

$\chi_{\mathrm{nfa}} > 0$ and a passive tax rule $\tau_t = \varphi_{dg} \widetilde{B}^g_{t-1}$ close the open-economy and fiscal loops. $\varphi_\pi = 1.5$. $\beta = 0.995$.

**Phillips curve.** Hybrid, with a flat slope $\kappa = 0.004$ `[PROVISIONAL]`. Under a peg that is long relative to $D = 9.1$ years, a steep Phillips curve is a Fisher spiral: expected easing after the peg can drive $R^L$ through zero. The flat slope keeps the sterilised long rate positive at $H \in \{8, 12, 20, 40\}$ and holds the Blanchard–Kahn count at 36 stable roots and 6 finite unstable roots. It is not an estimate of the UK Phillips curve. $H = 40$ with the old $\kappa = 0.015$ was a fragile way to get the same sign pattern; it is no longer the baseline.

State count: 19 predetermined variables, 6 forward-looking variables in the declaration, 11 static definitions. Saver consumption is now a level rule (it does not lead $C^s$). That is the “about 15–25 states” linear system.

---

## What changed after the review `[CHECK]`

Five corrections, same branch. $\lambda_q$, $s^{IL}$ and $D$ were not retuned.

1. **$C^s$ sign.** The forward habit Euler plus a $\Delta TP$ mark-to-market term produced peak $C^s \approx +0.5\%$ on a sterilised $+100$ bp term premium. Saver consumption is now the level rule above. On scenario (a), $C^s$ stays negative.
2. **$P^h$ scale.** The trough is about $-2.6\%$ per $+100$ bp, not $-9.5\%$. $\kappa$ and the housing loadings are provisional.
3. **$IB$ timing and units.** The conventional coupon still refixes at $1/(4D)$ per quarter. $IB$ is pp of GDP. The early dip is the index-linked uplift. The total peaks later because $\rho_{TP} = 0.90$ convolved with a coupon half-life of $\ln 2 / \delta_D \approx 25$ quarters.
4. **Peg length.** Default $H = 12$. $H \in \{8, 20, 40\}$ are the same sign pattern. $H = 40$ is robustness only.
5. **Labels.** Scenario (a) means the sterilised path only.

Scenario (a), $H = 12$, innovation about $0.252$ quarterly percent, $\rho_{TP} = 0.90$, scaled so $R^L$ rises $100$ annualised bp on impact:

| Variable | Scenario (a) |
|---|---|
| $Y$ | impact $-0.391\%$, trough $-0.545\%$ at quarter 4 |
| $P^h$ | impact $-0.752\%$, trough $-2.60\%$ at quarter 8 |
| $C^s$ | ranges $[-0.768\%,\ -0.081\%]$ (does not rise) |
| $C^b$ | trough $-1.038\%$ at quarter 9, least negative $-0.227\%$ |
| $IB$ | impact $-0.034$ pp of GDP, peak $+0.202$ at quarter 24 |

Borrowers are hit harder than savers at both the trough and the least-negative quarter. $C^b$ lines up with $R^{m,\mathrm{stock}}$, which peaks at quarter 10.

## Success criteria `[CHECK]`

Horizon 40. Interior equation residual about $10^{-14}$. The first 40 quarters are identical at simulation lengths 60 and 100.

| Check | Result |
|---|---|
| $R^L$ impact | $100$ annualised bp; minimum over 40 quarters about $+2.4$ bp (stays positive) |
| Bank Rate over the 12-quarter peg | numerically zero (max abs. about $10^{-13}$ annualised bp) |
| $\varepsilon^{\mathrm{ster}} + i^{TR}$ | numerically zero |
| Output | impact $-0.391\%$, trough $-0.545\%$ at quarter 4 |
| House prices | impact $-0.752\%$ (no rise), trough $-2.60\%$ at quarter 8 |
| $R^{\mathrm{eff}}_0 / R^L_0$ | $0.027473 = 1/(4D)$ |
| Interest burden | impact $-0.034$ pp of GDP, peak $+0.202$ at quarter 24 |
| $R^{\mathrm{eff}}$ | peaks at quarter 17; impact pass-through $1/(4D)$ of the long-rate move |
| Consumption | $C^s \in [-0.768,\ -0.081]$, $C^b \in [-1.038,\ -0.227]$ |
| Timing | $C^b$ trough at quarter 9, $R^{m,\mathrm{stock}}$ peak at quarter 10 |

They were not tuned by changing $\lambda_q$, $s^{IL}$, or $D$.

**Interest burden, units and split.** $IB$ is percentage points of GDP, not a levels series that needs dividing by 100. Quarterly-percent coupons times a debt ratio in quarters of GDP cancel the 100. A full immediate refix of the conventional stock would be $b^{\mathrm{nom}} \times R^L_0 = 0.75$ pp of GDP. On impact $R^{\mathrm{eff}}_0 = R^L_0 /(4D)$, so the conventional book has only just started to refix.

$$
IB_t = \underbrace{b^{\mathrm{nom}} R^{\mathrm{eff}}_t + \tfrac{i_{ss}}{100}(1-s^{IL})\widetilde{B}^g_t}_{\text{nominal coupon}} + \underbrace{b^{IL}(r^{IL}_t + \pi_t) + \tfrac{i_{ss}}{100}s^{IL}\widetilde{B}^g_t}_{\text{index-linked, including the uplift}}.
$$

The index-linked piece is about $-0.056$ pp of GDP on impact (disinflation cuts the uplift). The nominal-coupon piece peaks near $+0.16$ pp at quarter 19. Their sum dips, then peaks at quarter 24. That is later than a one-quarter pass-through and earlier than a permanent level shift of the whole book, whose half-life is $\ln 2 / (1/(4D)) \approx 25$ quarters. $R^L$ itself decays at $\rho_{TP} = 0.90$, so the convolution peaks before that half-life. The one-panel split is `output/irf_a_ib_decomposition.png`. Columns `ib_nom` and `ib_il` are on `output/irf_tp_sterilised.csv`.

**Peg length** `[PROVISIONAL]`. Same sterilised device, each path scaled to $+100$ bp on impact. $H = 12$ is scenario (a).

| $H$ | $Y$ trough | $P^h$ trough | $C^s$ max | $C^b$ trough | $IB$ peak | min $R^L$ |
|---|---:|---:|---:|---:|---:|---:|
| 8 | $-0.540\%$ | $-2.54\%$ q8 | $-0.082\%$ | $-1.024\%$ | $+0.202$ q25 | $+2.4$ bp |
| 12 (a) | $-0.545\%$ | $-2.60\%$ q8 | $-0.081\%$ | $-1.038\%$ | $+0.202$ q24 | $+2.4$ bp |
| 20 | $-0.549\%$ | $-2.64\%$ q9 | $-0.080\%$ | $-1.046\%$ | $+0.204$ q24 | $+2.3$ bp |
| 40 | $-0.550\%$ | $-2.65\%$ q9 | $-0.079\%$ | $-1.048\%$ | $+0.205$ q24 | $+2.1$ bp |

The sign pattern does not depend on a decade-long peg. Paths are in `output/irf_peg_robustness.csv`. $\kappa = 0.004$ and $\psi_{\mathrm{nfa}}$ are what keep $R^L$ from being driven through zero when the peg is short relative to duration; both are provisional.

**Same $TP$ innovation, Taylor rule free** `[CHECK]`. **Not scenario (a).** With the innovation that delivers $+100$ bp when sterilised, unsterilised $R^L$ impact is about $95$ bp. Bank Rate falls about $6$ bp on impact and later troughs near $-24$ bp. Output impact is about $-0.35\%$, trough about $-0.49\%$ at quarter 4. House prices trough about $-2.2\%$. Saver consumption stays negative. This file is `output/irf_tp_unsterilised.csv`. `stoch_simul(e_tp)` is this experiment, not (a).

**Scenario (b), same $+100$ bp long-rate scaling** `[CHECK]`. Not scenario (a). News persistence $\rho_\nu = 0.75$. Bank Rate impact is about $+315$ annualised bp (the expectations hypothesis only puts weight $\delta_D$ on the current short rate, and the news decays). Output impact is about $-0.87\%$, trough about $-1.18\%$ at quarter 5. House prices fall on impact (about $-1.95\%$) and trough about $-6.6\%$ at quarter 7. This is a different experiment from (a).

---

## High-frequency companion

`src/lp_proxyvar.py` is the empirical scaffold. It does not contain a surprise series.

* Proxy (a): high-frequency change in the 10-year gilt, residualised on a constant, the Bank Rate surprise, and the near-term OIS path.
* Proxy (b): the OIS path itself.
* Outcomes the stub is shaped for: ONS monthly GDP and a UK house-price index.

The identification reference is Cesa-Bianchi, Thwaites and Vicondoa (2020), *European Economic Review*, monetary-event windows for UK gilt yields. Yield-curve source: https://www.bankofengland.co.uk/statistics/yield-curves . Column definitions are in `data/README.md`. The template CSV is a header. A synthetic DGP is used only inside `src/test_lp_algebra.py`, labelled `SYNTHETIC_NOT_UK_DATA`, and is not written to `output/` as a UK result.

---

## What is not in the MVP

No CRE, buy-to-let, or corporate credit block. No second mortgage cohort. No social learning. No sequence-space Jacobian. Dynare.jl is a possible later reader of the same `.mod`, not a dependency. gEconpy and puremacro are not used. Bayesian estimation is not run; the check is the calibrated linear path above.
