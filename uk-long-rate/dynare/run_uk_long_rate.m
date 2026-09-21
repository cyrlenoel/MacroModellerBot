% run_uk_long_rate.m
% Sterilised scenario (a) on top of uk_long_rate.mod.
%
% Requires MATLAB or Octave with Dynare 4.6+ / 5 on the path.
% This repository environment has neither. The IRFs under ../output/ were
% produced by the Python QZ twin:
%
%   cd uk-long-rate
%   python3 src/run_irf.py
%
% Scenario (a) is NOT oo_.irfs.e_tp_*. stoch_simul leaves the Taylor rule
% free. The block below builds the anticipated residual that pegs Bank Rate.
%
% Sterilisation device (report this on every scenario-(a) table):
%   anticipated Bank Rate peg: the Taylor rule is left in place and the
%   monetary residual is set to e_ster,t = -i_t^TR for the first H quarters,
%   with i_path = 0. Residuals are anticipated. The Taylor rule resumes after H.

this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir)
    this_dir = pwd;
end
cd(this_dir);

dynare uk_long_rate noclearall;

H = 40;
T = 80;
target_rl = 0.25;  % quarterly percent = 100 annualised bp

ix_i = endo_index(M_, 'i');
ix_rl = endo_index(M_, 'rl');
jx_tp = exo_index(M_, 'e_tp');
jx_ster = exo_index(M_, 'e_ster');

shocks0 = zeros(T, M_.exo_nbr);
shocks0(1, jx_tp) = 1;
base = pf_response(T, shocks0);
i_base = base(ix_i, 1:H).';

J = zeros(H, H);
for j = 1:H
    ej = zeros(T, M_.exo_nbr);
    ej(j, jx_ster) = 1;
    resp_j = pf_response(T, ej);
    J(:, j) = resp_j(ix_i, 1:H).';
end
ster = -J \ i_base;

shocks_a = zeros(T, M_.exo_nbr);
shocks_a(1, jx_tp) = 1;
shocks_a(1:H, jx_ster) = ster;
resp = pf_response(T, shocks_a);
scale = target_rl / resp(ix_rl, 1);

out_dir = fullfile(this_dir, '..', 'output');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
out_file = fullfile(out_dir, 'dynare_irf_tp_sterilised.csv');
fid = fopen(out_file, 'w');
fprintf(fid, ['# anticipated Bank Rate peg: e_ster,t = -i_t^TR for H=%d quarters. ' ...
    'Compare with output/irf_tp_sterilised.csv from the Python twin.\n'], H);
fprintf(fid, 'quarter,i_bp,rl_bp,y,ph,e_ster_bp\n');
for t = 1:H
    fprintf(fid, '%d,%.8g,%.8g,%.8g,%.8g,%.8g\n', t, ...
        scale * resp(ix_i, t) * 400, ...
        scale * resp(ix_rl, t) * 400, ...
        scale * resp(endo_index(M_, 'y'), t), ...
        scale * resp(endo_index(M_, 'ph'), t), ...
        scale * shocks_a(t, jx_ster) * 400);
end
fclose(fid);
fprintf('Wrote %s\n', out_file);
fprintf('R^L impact = %.4f annualised bp (target 100).\n', scale * resp(ix_rl, 1) * 400);
fprintf('max |Bank Rate| over the peg = %.3e annualised bp.\n', ...
    max(abs(scale * resp(ix_i, 1:H))) * 400);

function idx = endo_index(M_, name)
idx = name_index(M_.endo_names, name);
end

function idx = exo_index(M_, name)
idx = name_index(M_.exo_names, name);
end

function idx = name_index(names, name)
if iscell(names)
    list = names(:);
else
    list = cellstr(names);
end
list = strtrim(list);
idx = find(strcmp(list, name), 1);
if isempty(idx)
    error('uk_long_rate:name', 'Name %s not found.', name);
end
end

function y = pf_response(T, shocks)
% shocks is T x exo_nbr over simulation dates (not including period 0).
% Dynare 4.6/5: oo_.exo_simul has T+2 rows (period 0, periods 1..T, terminal)
% and oo_.endo_simul has T+2 columns in the same order.
% Older builds use T rows/columns. Both are accepted.
global M_ oo_ options_
options_.periods = T;
perfect_foresight_setup;
nrow = size(oo_.exo_simul, 1);
if nrow == T + 2
    oo_.exo_simul(2:T+1, :) = shocks;
elseif nrow == T
    oo_.exo_simul(:, :) = shocks;
else
    error('uk_long_rate:exoSimul', ...
        'oo_.exo_simul has %d rows; expected T=%d or T+2=%d.', nrow, T, T+2);
end
perfect_foresight_solver;
ncol = size(oo_.endo_simul, 2);
if ncol == T + 2
    y = oo_.endo_simul(:, 2:T+1);
elseif ncol == T
    y = oo_.endo_simul;
else
    error('uk_long_rate:endoSimul', ...
        'oo_.endo_simul has %d columns; expected T or T+2.', ncol);
end
end
