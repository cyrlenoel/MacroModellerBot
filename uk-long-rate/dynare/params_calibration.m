function p = params_calibration()
% Sketch calibration for uk_long_rate.mod. Mirror of src/calibration.py.
% HOLD FIXED: lambda_q, s_IL, D. Everything else is provisional.

p.lambda_q = 0.09;
p.s_IL = 0.25;
p.D = 9.1;

p.beta = 0.995;
p.sigma = 6.0;
p.h_s = 0.40;
p.kappa = 0.004;
p.iota = 0.30;
p.phi_pi = 1.50;
p.phi_y = 0.125;
p.rho_i = 0.86;
p.psi_nfa = 0.08;
p.rho_tp = 0.90;
p.rho_nu = 0.75;
p.omega_S = 0.30;
p.omega_L = 0.70;
p.omega_f_S = 0.35;
p.omega_f_L = 0.65;
p.mu_ds = 15.0;
p.alpha_y = 0.15;
p.mu_coll = 0.10;
p.omega_cs = 0.62;
p.omega_cb = 0.38;
p.psi_tp = 2.0;
p.psi_y_cs = 0.10;
p.psi_ib = 0.02;
p.psi_tau = 0.12;
p.kappa_c = 0.055;
p.phi_h = 0.75;
p.kappa_H = 0.03;
p.kappa_level = 0.05;
p.phi_q = 0.90;
p.psi_k = 6.0;
p.psi_h = 3.0;
p.delta_k = 0.025;
p.delta_h = 0.007;
p.s_C = 0.62;
p.s_I = 0.11;
p.s_HI = 0.04;
p.s_G = 0.23;
p.eta_rer = 0.06;
p.eta_y = 0.18;
p.eta_ys = 0.05;
p.chi_nfa = 0.01;
p.b_g_ratio = 4.0;
p.phi_dg = 0.06;
p.pi_ss_annual = 2.0;
p.rho_zeta = 0.50;
p.rho_u = 0.60;
p.rho_g = 0.80;
p.rho_ystar = 0.90;
p.rho_fx = 0.75;

p.delta_D = 1 / (4 * p.D);
p.phi_y_qp = p.phi_y / 4;
p.b_nom = (1 - p.s_IL) * p.b_g_ratio;
p.b_IL = p.s_IL * p.b_g_ratio;
p.pi_ss_qp = p.pi_ss_annual / 4;
p.r_ss_qp = (1 / p.beta - 1) * 100;
p.i_ss_qp = p.r_ss_qp + p.pi_ss_qp;
p.H_peg = 12;
p.target_rl_qp = 0.25;
end
