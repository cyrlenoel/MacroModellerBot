/*
  uk_long_rate.mod
  ---------------------------------------------------------------
  Linearised UK long-rate / term-premium twin.
  Small-open-economy NK + HANK-lite (savers vs mortgagors).

  Sketch fidelity, not a full paper. Twin of src/linear_model.py.
  Units: quantities in percent; rates and inflation in quarterly percent
  (0.25 qp = 100 annualised bp); ratios in pp of quarterly GDP.

  HOLD FIXED (not free parameters):
    lambda_q = 0.09   quarterly remortgage / reset hazard
    s_IL     = 0.25   index-linked share of the gilt STOCK
    D        = 9.1    modified duration, years

  Shocks, not to be conflated:
    (a) e_tp    pure term premium. Structural IRFs sterilise Bank Rate.
    (b) e_news  short-rate path news. Taylor rule left free.
    (c) e_rhofx UIP risk-premium slot. Not fired in the MVP IRFs.

  stoch_simul below is an UNSTERILISED preview. Scenario (a) is the
  anticipated Bank Rate peg in run_uk_long_rate.m and in
  src/solve_pf.py:solve_sterilised_qz. Do not read the e_tp IRF from
  stoch_simul as scenario (a).

  Sterilisation device:
    anticipated Bank Rate peg: the Taylor rule is left in place and the
    monetary residual is set to e_ster,t = -i_t^TR for the first H
    quarters, with i_t^path = 0, so
      i_t = rho_i*i_{t-1} + (1-rho_i)*(phi_pi*pi_t + phi_y*Y_t) + nu_t
            + e_ster,t = 0;
    residuals are anticipated (perfect foresight) and the Taylor rule
    resumes after H.
*/

var
  i k h rms bm dg nfa tp nu reff ril cslag pilag phlag zeta u g ystar rhofx
  pi cs q ph reh rer
  y c cb inv hi rm rl ib nx tau mpk
;

varexo e_ster e_tp e_news e_zeta e_u e_g e_ystar e_rhofx;

parameters
  lambda_q s_IL D
  beta sigma h_s kappa iota phi_pi phi_y rho_i
  rho_tp rho_nu omega_S omega_L omega_f_S omega_f_L
  mu_ds alpha_y mu_coll omega_cs omega_cb carry mtm fisc_coef theta_dom
  kappa_c phi_h kappa_H kappa_level phi_q psi_k psi_h delta_k delta_h
  s_C s_I s_HI s_G
  eta_rer eta_y eta_ys chi_nfa
  b_g_ratio phi_dg pi_ss_annual
  rho_zeta rho_u rho_g rho_ystar rho_fx
;

% --- HOLD FIXED -----------------------------------------------------------
lambda_q = 0.09;
s_IL     = 0.25;
D        = 9.1;

% --- Provisional sketch calibration (see NOTES.md). Not estimates. --------
beta = 0.995;
sigma = 1.0;
h_s = 0.50;
kappa = 0.015;
iota = 0.30;
phi_pi = 1.50;
phi_y = 0.125;
rho_i = 0.80;
rho_tp = 0.93;
rho_nu = 0.75;
omega_S = 0.30;
omega_L = 0.70;
omega_f_S = 0.35;
omega_f_L = 0.65;
mu_ds = 16.0;
alpha_y = 0.10;
mu_coll = 0.40;
omega_cs = 0.62;
omega_cb = 0.38;
carry = 1.2;
mtm = -0.8;
fisc_coef = 0.30;
theta_dom = 0.70;
kappa_c = 0.16;
phi_h = 0.70;
kappa_H = 0.04;
kappa_level = 0.03;
phi_q = 0.90;
psi_k = 6.0;
psi_h = 3.0;
delta_k = 0.025;
delta_h = 0.007;
s_C = 0.62;
s_I = 0.11;
s_HI = 0.04;
s_G = 0.23;
eta_rer = 0.06;
eta_y = 0.18;
eta_ys = 0.05;
chi_nfa = 0.01;
b_g_ratio = 4.0;
phi_dg = 0.06;
pi_ss_annual = 2.0;
rho_zeta = 0.50;
rho_u = 0.60;
rho_g = 0.80;
rho_ystar = 0.90;
rho_fx = 0.75;

model(linear);

  #delta_D = 1/(4*D);
  #phi_y_qp = phi_y/4;
  #b_nom = (1-s_IL)*b_g_ratio;
  #b_IL = s_IL*b_g_ratio;
  #pi_ss_qp = pi_ss_annual/4;
  #i_ss_qp = (1/beta - 1)*100 + pi_ss_qp;
  #nkpc_forward = beta/(1+beta*iota);
  #nkpc_lag = iota/(1+beta*iota);
  #eis = (1-h_s)/sigma;
  #inc_scale = 1-h_s;

  % eq: taylor
  i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y_qp*y) + nu + e_ster;
  % eq: capital
  k = (1-delta_k)*k(-1) + delta_k*inv;
  % eq: housing_stock
  h = (1-delta_h)*h(-1) + delta_h*hi;
  % eq: mortgage_stock_rate
  rms = (1-lambda_q)*rms(-1) + lambda_q*rm;
  % eq: ltv
  bm = (1-lambda_q)*bm(-1) + lambda_q*(ph + h);
  % eq: debt
  dg = (1/beta)*dg(-1) + s_G*g - tau + b_nom*reff + b_IL*ril - b_nom*pi;
  % eq: nfa
  nfa = (1/beta)*nfa(-1) + nx;
  % eq: tp
  tp = rho_tp*tp(-1) + e_tp;
  % eq: news
  nu = rho_nu*nu(-1) + e_news;
  % eq: reff
  reff = (1-delta_D)*reff(-1) + delta_D*rl;
  % eq: ril
  ril = (1-delta_D)*ril(-1) + delta_D*(rl - pi(+1));
  % eq: cslag
  cslag = cs(-1);
  % eq: pilag
  pilag = pi(-1);
  % eq: phlag
  phlag = ph(-1);
  % eq: zeta
  zeta = rho_zeta*zeta(-1) + e_zeta;
  % eq: costpush
  u = rho_u*u(-1) + e_u;
  % eq: gshock
  g = rho_g*g(-1) + e_g;
  % eq: ystar
  ystar = rho_ystar*ystar(-1) + e_ystar;
  % eq: rhofx
  rhofx = rho_fx*rhofx(-1) + e_rhofx;
  % eq: nkpc
  pi = nkpc_forward*pi(+1) + nkpc_lag*pilag + kappa*y + u;
  % eq: saver_euler
  % Habit loads on contemporaneous cslag (= Cs_{t-1}), not on cslag(-1).
  (1+h_s)*cs = h_s*cslag + cs(+1) - eis*(i - pi(+1)) + inc_scale*((carry+mtm)*tp - mtm*tp(-1) + fisc_coef*(theta_dom*ib - tau));
  % eq: tobin_q
  % mpk is contemporaneous. A lead of output here adds an explosive root.
  q = beta*q(+1) + (1-beta)*mpk - phi_q*(omega_f_S*(i - pi(+1)) + omega_f_L*(rl - pi(+1)));
  % eq: housing_phillips
  (1+beta+kappa_level)*ph = phlag + beta*ph(+1) + kappa_c*cb - phi_h*(rm - pi(+1)) - kappa_H*h;
  % eq: expectations_hypothesis
  reh = delta_D*i + (1-delta_D)*reh(+1);
  % eq: uip
  rer = rer(+1) - (i - pi(+1)) + chi_nfa*nfa + rhofx;
  % eq: resource
  y = s_C*c + s_I*inv + s_HI*hi + s_G*g + nx;
  % eq: aggregate_c
  c = omega_cs*cs + omega_cb*cb;
  % eq: borrower
  cb = alpha_y*y - mu_ds*rms + mu_coll*lambda_q*(ph - bm(-1));
  % eq: investment
  inv = q/psi_k;
  % eq: housing_investment
  hi = ph/psi_h;
  % eq: new_mortgage_rate
  rm = omega_S*i + omega_L*rl + zeta;
  % eq: long_rate
  rl = reh + tp;
  % eq: interest_burden
  ib = b_nom*reff + b_IL*(ril + pi) + (i_ss_qp/100)*dg;
  % eq: net_exports
  nx = eta_rer*rer - eta_y*y + eta_ys*ystar;
  % eq: tax_rule
  tau = phi_dg*dg(-1);
  % eq: mpk
  mpk = y - k(-1);

end;

initval;
  i = 0; k = 0; h = 0; rms = 0; bm = 0; dg = 0; nfa = 0;
  tp = 0; nu = 0; reff = 0; ril = 0; cslag = 0; pilag = 0; phlag = 0;
  zeta = 0; u = 0; g = 0; ystar = 0; rhofx = 0;
  pi = 0; cs = 0; q = 0; ph = 0; reh = 0; rer = 0;
  y = 0; c = 0; cb = 0; inv = 0; hi = 0; rm = 0; rl = 0;
  ib = 0; nx = 0; tau = 0; mpk = 0;
end;

steady;
check;

% Positive shock variances so stoch_simul runs. These IRFs are NOT scenario (a):
% Bank Rate is free to follow the Taylor rule. The sterilised experiment is
% run_uk_long_rate.m (and the Python QZ path).
shocks;
  var e_ster; stderr 0.01;
  var e_tp; stderr 0.01;
  var e_news; stderr 0.01;
  var e_zeta; stderr 0.01;
  var e_u; stderr 0.01;
  var e_g; stderr 0.01;
  var e_ystar; stderr 0.01;
  var e_rhofx; stderr 0.01;
end;

stoch_simul(order=1, irf=40, nograph, nomoments);

% Estimation is intentionally not executed.
% lambda_q, s_IL and D stay fixed. A later estimated_params block may free
% kappa, rho_tp, rho_nu, omega_S, mu_ds and the housing loadings only after
% the observables and the HF instruments are actually loaded. Do not estimate
% the three hold-fixed numbers.
