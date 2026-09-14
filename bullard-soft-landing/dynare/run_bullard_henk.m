% run_bullard_henk.m
% Driver for bullard_henk.mod + Grimaud-Salle-Vermandel Dynare Social Learning toolbox.
%
% API status:
%   [VERIFIED] from Mendeley dataset fzmx3vkt66 (V3/GSV2023.mod, Toolbox/SL_*.m)
%     and WU WP 339 section 4.1 / JEDC 2025 DOI 10.1016/j.jedc.2024.104984 abstract.
%   [PROVISIONAL] inflation-only SL via mut_sd_y=0; mut_sd mapping of sigma_lambda;
%     IRF burn-in length; exact plot indexing if endo order differs.
%
% Prerequisites (laptop):
%   1. MATLAB + Dynare on path (toolbox read_me.txt: MATLAB 2021b-2023a, Dynare 5.0-5.3)
%   2. Download Mendeley dataset DOI 10.17632/fzmx3vkt66.1 and unzip
%   3. Set TOOLBOX_PATH below to the unzipped .../V3/Toolbox folder

clear; close all; clc;

%% ========== PATHS (edit these) ==========
% [PLACEHOLDER] Absolute path to the unzipped toolbox Toolbox/ directory:
TOOLBOX_PATH = '/PATH/TO/fzmx3vkt66/V3/Toolbox';

% Dynare matlab folder if not already on the MATLAB path, e.g.:
% addpath('/usr/local/dynare/5.x/matlab');

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(TOOLBOX_PATH);

%% ========== Table 1 parameters (for SL options) ==========
p = params_table1();

%% ========== Solve RE nesting via Dynare ==========
cd(this_dir);
dynare bullard_henk noclearall nolog

% After dynare, workspace holds oo_, M_, options_

%% ========== SL options [VERIFIED field names from GSV2023.mod] ==========
% Forward endo with leads in bullard_henk.mod: pi, y  (declaration order).
% mut_sd and rhoGN must be nfwrd x 1 column vectors matching that order.

options_.SL.N_agents = p.N_agents;          % [VERIFIED] field; J even preferred
options_.SL.mut_p    = p.mu_news;           % [VERIFIED] mutation / news frequency
% [PROVISIONAL] mut_sd: toolbox has idiosyncratic mutations only (no separate
% aggregate lambda shock). Use sigma_iota for pi; set y-component to 0 so
% output-gap beliefs stay at SS (WP: only inflation is SL).
options_.SL.mut_sd   = [p.sigma_iota; 0];
options_.SL.wed_p    = 1;                   % [VERIFIED] tournament participation share
options_.SL.rhoGN    = [p.rho_fit; p.rho_fit];  % [VERIFIED] fitness-decay vector
options_.SL.Tsample  = 400;                 % total simulated length (incl. burn-in)
options_.seed_block  = 0;                   % [VERIFIED] 0 = draw SL news; 1 = freeze seed

% Sanity: nfwrd should be 2 (pi, y). Adjust mut_sd/rhoGN if Dynare reports more leads.
nfwrd_expected = 2;
id_frw = find(M_.lead_lag_incidence(3,:) > 0);
if numel(id_frw) ~= nfwrd_expected
    warning(['nfwrd=%d (expected %d). Resize options_.SL.mut_sd and rhoGN to match. ' ...
             'Forward names: %s'], numel(id_frw), nfwrd_expected, ...
            strjoin(cellstr(M_.endo_names(id_frw,:)), ', '));
end

%% ========== Build P, Q, R then simulate [VERIFIED signatures] ==========
% From Toolbox/SL_get_pq.m:
%   [mo] = SL_get_pq(oo_, M_, options_);
%   mo.PP, mo.QQ, mo.RR (and FF,GG,HH,MM)
% From Toolbox/SL_chain.m:
%   [oo] = SL_chain(mo, oo_, M_, options_, cc);
%   cc is Tsample x nexo (then scaled inside by sqrt(M_.Sigma_e))
%   oo.irfs.re, oo.irfs.sl, oo.irfs.Esl

[mo] = SL_get_pq(oo_, M_, options_);

%% ========== IRFs: monetary (ev) and cost-push (eu), RE vs SL ==========
% Pattern [VERIFIED] from GSV2023.mod IRF block: zero chain + impulse at burn-in.
% SL_chain scales cc by sqrt(M_.Sigma_e) (elementwise), so cc_impulse=1 is ~1 std.

run_init = 300;                             % burn-in [PROVISIONAL; GSV uses 300]
T_irf    = 40;                              % plotted horizon
options_.SL.Tsample = run_init + T_irf;

% Endogenous name lookup (declaration order in .mod)
endo = cellstr(M_.endo_names);
idx_pi = find(strcmp(endo, 'pi'), 1);
idx_y  = find(strcmp(endo, 'y'), 1);
idx_i  = find(strcmp(endo, 'i'), 1);

% Exogenous order in .mod: eg, eu, ev
exo = cellstr(M_.exo_names);
idx_eg = find(strcmp(exo, 'eg'), 1);
idx_eu = find(strcmp(exo, 'eu'), 1);
idx_ev = find(strcmp(exo, 'ev'), 1);

shock_specs = { ...
    struct('name', 'monetary',  'exo_idx', idx_ev, 'title', 'Monetary policy shock (ev)'), ...
    struct('name', 'costpush',  'exo_idx', idx_eu, 'title', 'Cost-push shock (eu)') ...
    };

out_dir = fullfile(this_dir, 'output');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

rng(2023);  % [VERIFIED] GSV2023 uses rng(2023) for reproducibility

for s = 1:numel(shock_specs)
    spec = shock_specs{s};

    cc = zeros(options_.SL.Tsample, M_.exo_nbr);
    % One-std impulse at t = run_init+1 (1-based in MATLAB chain)
    cc(run_init + 1, spec.exo_idx) = 1;

    [oo] = SL_chain(mo, oo_, M_, options_, cc);

    t_plot = (run_init + 1):(run_init + T_irf);
    tt = 1:T_irf;

    figure('Name', spec.title);
    subplot(2,2,1);
    plot(tt, oo.irfs.re(idx_pi, t_plot), 'k-', tt, oo.irfs.sl(idx_pi, t_plot), 'b--');
    title('\pi'); legend('RE','SL'); grid on;
    subplot(2,2,2);
    plot(tt, oo.irfs.re(idx_y, t_plot), 'k-', tt, oo.irfs.sl(idx_y, t_plot), 'b--');
    title('y'); legend('RE','SL'); grid on;
    subplot(2,2,3);
    plot(tt, oo.irfs.re(idx_i, t_plot), 'k-', tt, oo.irfs.sl(idx_i, t_plot), 'b--');
    title('i'); legend('RE','SL'); grid on;
    subplot(2,2,4);
    % Mean subjective inflation expectation across agents [VERIFIED oo.irfs.Esl]
    % Esl dims: nfwrd x time x N_agents; first forward var is pi if declaration order holds
    if ~isempty(oo.irfs.Esl)
        Epi_sl = squeeze(mean(oo.irfs.Esl(1, t_plot, :), 3));
        plot(tt, Epi_sl, 'b--');
        title('mean E^{SL}(\pi_{t+1})'); grid on;
    end
    sgtitle(spec.title);

    saveas(gcf, fullfile(out_dir, ['irf_' spec.name '_re_vs_sl.png']));

    % CSV export
    Ttab = table(tt(:), ...
        oo.irfs.re(idx_pi, t_plot).', oo.irfs.sl(idx_pi, t_plot).', ...
        oo.irfs.re(idx_y, t_plot).',  oo.irfs.sl(idx_y, t_plot).', ...
        oo.irfs.re(idx_i, t_plot).',  oo.irfs.sl(idx_i, t_plot).', ...
        'VariableNames', {'t','pi_re','pi_sl','y_re','y_sl','i_re','i_sl'});
    writetable(Ttab, fullfile(out_dir, ['irf_' spec.name '_re_vs_sl.csv']));
end

fprintf('Done. IRFs written under %s\n', out_dir);
fprintf(['Next: compare mut_sd / mut_p / rhoGN to Toolbox defaults; ' ...
         'if SL on y is undesired, keep mut_sd(2)=0 (current setting).\n']);

%% ========== Optional notes (TODOs not implemented) ==========
% TODO [BLOCKED]: Bayesian inversion-filter estimation + SPF observables (WP section 3).
% TODO [BLOCKED]: Table 3 historical-shock counterfactuals (need author data/replication).
% TODO [PROVISIONAL]: Map aggregate news sigma_lambda into toolbox (no separate lambda).
% TODO: Demand-shock IRFs (eg) — same pattern as above if needed.
