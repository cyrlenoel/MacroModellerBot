# UK long rate — sketch notes

**Status legend:** `[FIXED]` = held at the signed-sketch value. `[SKETCH]` = implemented at sketch fidelity, not estimated. `[PROVISIONAL]` = a modelling choice that makes the linear IRFs qualitatively usable, not a UK estimate. `[CHECK]` = a number produced by the Python QZ twin of `dynare/uk_long_rate.mod` on the default calibration.

The system is linear. Quantities ($Y$, $C^s$, $C^b$, $C$, $I$, $Q$, $K$, $H$, $P^h$, the real exchange rate) are percent deviations. Rates and inflation are quarterly percent, so $0.25$ quarterly percent is $100$ annualised basis points. Ratios ($NX$, $IB$, debt, NFA, taxes) are percentage points of quarterly GDP, which for a flow is the same as percentage points of annual GDP.

---

## What the two experiments are

They are different shocks.

**(a) Term premium.** $TP_t$ jumps and $R^L$ rises with it. Bank Rate is sterilised for $H = 40$ quarters by an anticipated monetary residual. This is the baseline.

**(b) Short-rate news.** The news state $\nu_t$ shifts the Taylor rule. Bank Rate is not sterilised. The innovation is scaled so $R^L$ still rises $100$ bp on impact, which makes the Bank Rate path much larger than in (a), because the expectations-hypothesis weight on the current short rate is only $1/(4D)$.

**(c)** A UIP risk-premium shock, $\rho^{fx}_t$, is in the `.mod` and is not fired.

---

## Sterilisation `[SKETCH]`

Every scenario-(a) figure and CSV carries this device:

> anticipated Bank Rate peg: the Taylor rule is left in place and the monetary residual is set to $\varepsilon^{\mathrm{ster}}_t = -i^{TR}_t$ for the first $H$ quarters, with $i^{\mathrm{path}}_t = 0$, so $i_t = \rho_i i_{t-1} + (1-\rho_i)(\varphi_\pi \pi_t + \varphi_y Y_t) + \nu_t + \varepsilon^{\mathrm{ster}}_t = 0$; residuals are anticipated (perfect foresight) and the Taylor rule resumes after $H$.

In quarterly-percent units the rule in the model is

$$
i_t = \rho_i i_{t-1} + (1-\rho_i)\big(\varphi_\pi \pi_t + \tfrac{1}{4}\varphi_y Y_t\big) + \nu_t + \varepsilon^{\mathrm{ster}}_t.
$$

$\varphi_y = 0.125$ is the annual convention, so the output coefficient in quarterly percent is $\varphi_y/4$. $H = 40$, and the reported IRFs are also 40 quarters, so the plotted Bank Rate path is inside the peg. A permanent peg is not used: it would leave the inflation path undetermined.

The same term-premium innovation with $\varepsilon^{\mathrm{ster}} = 0$ is written to `output/irf_tp_unsterilised.csv` so the peg is visible. It is not scenario (a).

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

**Saver.** External habit, plus a percent-of-consumption shifter for duration carry, mark-to-market, and domestic coupons net of taxes:

$$
(1+h_s) C^s_t = h_s C^s_{t-1} + E_t C^s_{t+1} - \frac{1-h_s}{\sigma}\big(i_t - E_t\pi_{t+1}\big) + (1-h_s)\,\mathrm{inc}_t,
$$

$$
\mathrm{inc}_t = (\mathrm{carry}+\mathrm{mtm})\, TP_t - \mathrm{mtm}\, TP_{t-1} + \mathrm{fisc}\,\big(\theta^{\mathrm{dom}} IB_t - \tau_t\big).
$$

$\mathrm{mtm} < 0$, so a positive term-premium innovation is a capital loss on impact and a carry gain while $TP$ stays high. Putting the carry inside the elasticity of substitution as a rate does not separate $C^s$ from $C^b$: the terminal condition kills a pure rate wedge. The shifter is scaled so a permanent $\mathrm{inc}$ of $x$ percent raises $C^s$ by $x$ percent.

**Housing.** A sticky user-cost / housing Phillips curve, so the house-price trough is not forced onto the impact date the way a pure flexible asset price would be:

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

**Phillips curve.** Hybrid, with a flat slope $\kappa = 0.015$ `[PROVISIONAL]`. Under a 40-quarter peg a steep Phillips curve is a Fisher spiral: the sign of output can flip with the horizon, or an extra unstable root appears. The flat slope is what keeps the sterilised term-premium shock contractionary and the Blanchard–Kahn count at 36 stable and 36 unstable roots (6 of the unstable roots finite: $\pi$, $C^s$, $Q$, $P^h$, $R^{EH}$, $rer$). It is not an estimate of the UK Phillips curve.

State count: 19 predetermined variables, 6 jumps, 11 static definitions. That is the “about 15–25 states” linear system.

---

## Success criteria `[CHECK]`

Illustrative size: a term-premium innovation of about $0.236$ quarterly percent, persistence $0.93$, scaled so $R^L$ rises $100$ annualised bp on impact. Horizon 40. Interior equation residual about $10^{-14}$. The first 40 quarters are identical at simulation lengths 60 and 100.

| Check | Result |
|---|---|
| $R^L$ impact | $100$ annualised bp |
| Bank Rate over the 40-quarter peg | numerically zero (max abs. about $10^{-12}$ annualised bp) |
| $\varepsilon^{\mathrm{ster}} + i^{TR}$ | numerically zero |
| Output | impact $-0.187\%$, trough $-0.364\%$ at quarter 9 |
| House prices | impact $-1.81\%$ (no rise), trough $-9.46\%$ at quarter 12 |
| $R^{\mathrm{eff}}_0 / R^L_0$ | $0.027473 = 1/(4D)$ |
| Interest burden | impact $-0.093$ pp of GDP, peak $+0.346$ at quarter 31 |
| $R^{\mathrm{eff}}$ | peaks at quarter 29, impact pass-through $1/(4D)$ of the long-rate move |
| Consumption peaks | peak $C^b = -0.34\%$, peak $C^s = +0.51\%$ |
| Timing | $C^b$ trough at quarter 11 ($-1.48\%$), $R^{m,\mathrm{stock}}$ peak at quarter 13 |

The house-price trough sits on the last admissible quarter of the 4–12 window. The consumption-timing gap is exactly two quarters, which is the edge of the check. Both pass on this calibration and both move if the provisional housing or cashflow loadings are nudged. They were not tuned by changing $\lambda_q$, $s^{IL}$, or $D$.

$IB$ is slightly negative on impact because the index-linked uplift falls with inflation, then rises as the conventional coupon refixes. A full immediate refix of the conventional stock would be $b^{\mathrm{nom}} \times R^L_0 = 0.75$ pp of GDP. The model peak is about half of that and arrives late, which is the $D$ and $s^{IL}$ timing the check asks for.

Inflation falls by about $52$ annualised bp on impact under (a). That is large next to a $-0.2\%$ output gap because the Phillips curve is forward-looking and the peg lasts the whole reported window. It is part of the same provisional flat-curve choice.

**Same $TP$ innovation, Taylor rule free** `[CHECK]`. $R^L$ impact is about $99$ bp. Bank Rate rises about $12$ bp on impact and later troughs near $-17$ bp. Output rises about $0.51\%$ on impact. Sterilisation is what makes the output response negative in this calibration: saver carry and the endogenous Bank Rate path offset the mortgagor cashflow channel when the rule is left on.

**Scenario (b), same $+100$ bp long-rate scaling** `[CHECK]`. News persistence $\rho_\nu = 0.75$. Bank Rate impact is about $+295$ annualised bp (the expectations hypothesis only puts weight $\delta_D$ on the current short rate, and the news decays). Output impact is about $-4.2\%$, trough about $-5.7\%$ at quarter 3. House prices fall on impact (about $-5.9\%$) and trough near quarter 8. This is a different experiment from (a), not a larger version of the same residual.

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
