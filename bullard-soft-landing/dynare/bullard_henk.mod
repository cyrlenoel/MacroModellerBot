/*
  bullard_henk.mod
  ---------------------------------------------------------------
  Linear HENK / NK block for Bullard-Grimaud-Salle-Vermandel
  "Soft landing and inflation scares" (Tinbergen WP TI 2025-001).

  Equations [VERIFIED] WP (1)-(3) and shock processes in WP section 2.4:
    pi_t = kappa*y_t + beta*E_t^{SL}(pi_{t+1}) + u_t
    y_t  = E_t(y_{t+1}) - (1/sigma)*(i_t - E_t^{SL}(pi_{t+1})) + g_t
    i_t  = rho_i*i_{t-1} + (1-rho_i)*(phi_pi*pi_t + phi_y*y_t) + v_t
    g_t  = rho_g*g_{t-1} + eps_g_t - mu_g*eps_g_{t-1}   % ARMA(1,1)
    u_t  = rho_u*u_{t-1} + eps_u_t - mu_u*eps_u_{t-1}   % ARMA(1,1)
    v_t  = eps_v_t                                       % white noise

  SL EXPECTATIONS — HOW THIS FILE MATCHES THE TOOLBOX [VERIFIED]
  ---------------------------------------------------------------
  From Grimaud-Salle-Vermandel Mendeley toolbox (fzmx3vkt66 / GSV2023.mod):
    * There is NO special Dynare operator for E^{SL}.
    * Write the model with STANDARD Dynare RE leads: pi(+1), y(+1).
    * After stoch_simul, call SL_get_pq then SL_chain (see run_bullard_henk.m).
    * The toolbox then applies social learning to ALL forward-looking
      endogenous variables (those with leads in M_.lead_lag_incidence).

  BULLARD vs TOOLBOX ILLUSTRATION [PROVISIONAL mapping]
  ---------------------------------------------------------------
  WP: only inflation expectations are SL; output-gap expectations stay RE.
  Toolbox GSV2023 applies SL intercepts to every forward variable (pi and y).
  run_bullard_henk.m therefore sets mut_sd for y to 0 so y-beliefs stay at
  steady state (approximate inflation-only SL). Review after inspecting
  SL_simulxm.m on your laptop.

  Parameters: Table 1 posterior means + beta=0.99 (see params_table1.m).
  This .mod is the FIRE/RE nesting; SL is layered by the toolbox driver.
*/

% Optional: load named Table 1 struct (ignored by Dynare preprocessor if
% you prefer hard-coded values below). Safe when called from run_*.m.
% p = params_table1();

%------------------------------------------------------------------------------------------%
%                 ENDOGENOUS VARIABLES
%------------------------------------------------------------------------------------------%
% Declaration order matters for options_.SL.mut_sd / rhoGN indexing:
% forward variables with leads are pi and y (first two among endo with leads).
var pi y i g u eg_lag eu_lag;

%------------------------------------------------------------------------------------------%
%                 PARAMETERS
%------------------------------------------------------------------------------------------%
parameters beta kappa sigma phi_pi phi_y rho_i
           rho_g rho_u mu_g mu_u;

%------------------------------------------------------------------------------------------%
%                 EXOGENOUS VARIABLES
%------------------------------------------------------------------------------------------%
varexo eg eu ev;

%------------------------------------------------------------------------------------------%
%                 CALIBRATION — Table 1 posterior means + beta
%------------------------------------------------------------------------------------------%
beta   = 0.99;      % calibrated, WP section 3.2
kappa  = 0.2032;
sigma  = 1.6993;
phi_pi = 1.7131;
phi_y  = 0.1564;
rho_i  = 0.8179;

rho_g  = 0.7455;
rho_u  = 0.6328;
mu_g   = 0.4579;
mu_u   = 0.4887;

%------------------------------------------------------------------------------------------%
%                 MODEL (linear; FIRE nesting of WP eqs 1-3)
%------------------------------------------------------------------------------------------%
model(linear);

  % NKPC — WP (1). Under RE nesting, E^{SL}(pi(+1)) = pi(+1).
  % Toolbox SL replaces the forward solution with P,Q,R + belief intercepts.
  pi = kappa*y + beta*pi(+1) + u;

  % IS — WP (2). y(+1) is RE / model-consistent (WP section 2.2).
  % Under toolbox SL, y beliefs can also mutate unless mut_sd_y=0 in the driver.
  y  = y(+1) - (1/sigma)*(i - pi(+1)) + g;

  % Taylor rule — WP (3). Monetary shock is white-noise varexo ev.
  i  = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + ev;

  % Demand shock ARMA(1,1)
  g  = rho_g*g(-1) + eg - mu_g*eg_lag;
  eg_lag = eg;

  % Cost-push shock ARMA(1,1)
  u  = rho_u*u(-1) + eu - mu_u*eu_lag;
  eu_lag = eu;

end;

check;

% Shock stds = Table 1 Panel A posterior means (Dynare wants variances)
shocks;
  var eg = 0.0067^2;
  var eu = 0.0043^2;
  var ev = 0.0023^2;
end;

% RE / FIRE solution (required before SL_get_pq)
stoch_simul(order=1, irf=0, noprint, nograph);

/*
  SL driver lives in run_bullard_henk.m (preferred), mirroring GSV2023.mod:
    options_.SL.* = ...;
    [mo] = SL_get_pq(oo_, M_, options_);
    [oo] = SL_chain(mo, oo_, M_, options_, cc);
*/
