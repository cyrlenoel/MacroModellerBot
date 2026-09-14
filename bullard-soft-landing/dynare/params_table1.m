% params_table1.m
% Posterior means from Bullard-Grimaud-Salle-Vermandel Tinbergen WP TI 2025-001,
% Table 1 (sample 1985Q1-2023Q4). Calibrated: beta = 0.99 (WP section 3.2).
% [VERIFIED] from papers/wp25001.txt Table 1.

function p = params_table1()
p = struct();

% Calibrated
p.beta = 0.99;

% Panel B — structural (posterior means)
p.kappa   = 0.2032;
p.sigma   = 1.6993;   % inverse IES
p.phi_pi  = 1.7131;
p.phi_y   = 0.1564;
p.rho_i   = 0.8179;   % MPR smoothing (rho_iota)

% Panel A — shock processes (posterior means)
p.sigma_g = 0.0067;
p.sigma_u = 0.0043;
p.sigma_v = 0.0023;
p.rho_g   = 0.7455;
p.rho_u   = 0.6328;
p.mu_g    = 0.4579;   % MA(1) demand
p.mu_u    = 0.4887;   % MA(1) cost-push

% Panel C — social learning (posterior means)
p.sigma_iota   = 0.0006;  % idiosyncratic news std
p.sigma_lambda = 0.0004;  % aggregate news std (see README: toolbox mapping)
p.rho_fit      = 0.7745;  % decay in past forecast errors
p.mu_news      = 0.4357;  % frequency of news shocks

% Simulation choice (not in Table 1; toolbox default in GSV2023.mod)
p.N_agents = 300;  % even; [PROVISIONAL] WP requires J even, no estimated J
end
