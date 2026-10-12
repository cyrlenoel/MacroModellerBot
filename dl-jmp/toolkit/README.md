# Deep equilibrium nets (JAX)

A small, reusable toolkit for solving dynamic equilibrium models with neural networks, and two verification models. It is the numerical foundation for a nonlinear heterogeneous-household New Keynesian model with a preferred-habitat gilt market. Equilibrium residuals are minimised by stochastic gradient descent on a simulated state distribution, following Maliar, Maliar and Winant (2021) and Azinovic, Gaegauf and Scheidegger (2022).

The package lives in `dleq/`. Verification scripts live in `examples/`. Nothing outside `dl-jmp/toolkit/` is part of this folder.

## Install

Python 3.10 or newer. The pins in `requirements.txt` are a CPU stack: JAX's default wheel, not `jax[cuda]`. Importing `dleq` selects the CPU platform unless `JAX_PLATFORMS` is already set, and it turns on 64-bit arithmetic.

```bash
cd dl-jmp/toolkit
python -m pip install -r requirements.txt
```

Versions used for the tables below: Python 3.12.3, JAX 0.11.2, jaxlib 0.11.2, Optax 0.2.8, NumPy 2.2.6, backend `cpu`, float64.

## Run

```bash
python -m examples.brock_mirman
python -m examples.stochastic_growth
python -m pytest
```

Both examples write tables, figures and parameter archives under `output/`. Pytest (14 tests, about 10 seconds on this CPU) includes a short Brock–Mirman training run held to a loose tolerance of \(10^{-2}\) in the max relative policy error. It does not repeat the published training budgets.

## API

```python
import jax
import dleq

policy = dleq.MLPPolicy(
    input_dim=2,
    output_dim=1,
    width=64,
    depth=2,
    activation="tanh",
    output_transform="sigmoid",   # or "softplus" or "none"
    output_lower=1e-4,
    output_upper=1.0 - 1e-4,
)
theta = policy.init(jax.random.PRNGKey(0))
controls = policy.apply(theta, states)  # states shape (batch, input_dim)
```

`sigmoid` maps the raw head onto \((\ell, u)\):

$$
y = \ell + (u - \ell)\,\sigma(z).
$$

`softplus` maps onto \((\ell, \infty)\):

$$
y = \ell + \log(1 + e^{z}).
$$

`dleq.inverse_output_transform` returns the raw value that hits a chosen target, which is how the stochastic-growth example places the initial consumption share at the deterministic steady state.

The user writes the equilibrium conditions. `residual_fn(params, states, shocks)` returns an array of shape `(batch,)` or `(batch, n_equations)`. The same states and two independent shock draws are scored by the all-in-one loss:

```python
loss = dleq.all_in_one_loss(residual_fn(theta, states, eps1),
                            residual_fn(theta, states, eps2))
```

Optional `weights` scale the equations. `transition_fn(params, states, shocks)` is the state transition used only to move the training panel. `sample_states(key, n)` draws the initial panel and any states mixed back in.

```python
result = dleq.train(
    theta,
    residual_fn,
    transition_fn,
    sample_states,
    dleq.TrainConfig(n_steps=2000, batch_size=2048, seed=0, lbfgs_steps=50),
    score_fn=quadrature_mean_square,      # deterministic checkpoint score
    polish_loss_fn=quadrature_mean_square,  # optional; default polish is all-in-one
)
```

`dleq.simulate_forward` advances a panel under the current policy with the parameters detached, which is the ergodic-set update. `dleq.verify_against_linear` compares a nonlinear policy with a first-order policy on a common set of states, including a steady-state neighbourhood.

## All-in-one expectation

An equilibrium condition is a residual \(f(s, \varepsilon')\) with

$$
\mathbb{E}\big[f(s, \varepsilon') \mid s\big] = 0.
$$

The training target is the squared conditional expectation. A single Monte Carlo draw estimates \(\mathbb{E}[f^2 \mid s] = (\mathbb{E}[f \mid s])^2 + \mathrm{Var}(f \mid s)\) and is biased by the variance. Two independent draws \(\varepsilon_1, \varepsilon_2\) undo that, because independence pulls the expectation out of the product (Maliar, Maliar and Winant, 2021):

$$
\mathbb{E}\big[f_j(s, \varepsilon_1)\, f_j(s, \varepsilon_2) \mid s\big]
  = \big(\mathbb{E}[f_j \mid s]\big)^2.
$$

The loss implemented in `dleq.all_in_one_loss` is the batch mean of \(\sum_j w_j f_j(\varepsilon_1) f_j(\varepsilon_2)\). Averaging several independent pairs, which `TrainConfig.n_pairs` does, does not change the target.

## Ergodic-set training

Azinovic, Gaegauf and Scheidegger (2022) train on states drawn from the distribution the current policy actually generates. After each Adam step the panel is moved by

$$
s_{i,t+1} = \mathcal{T}(s_{i,t}, \varepsilon_{i,t}; \theta),
$$

and those states are the next minibatch. Gradients do not flow through \(\mathcal{T}\): the cloud is data. `TrainConfig.mix_fraction` replaces that fraction of the panel with fresh draws from `sample_states` (zero leaves a pure simulated panel). Adam uses a linear warmup and then exponential decay. An optional L-BFGS polish follows.

For these scalar-shock examples the polish minimises the Gauss–Hermite mean square of the same Euler residual on a fixed state sample. With one shock, an 9- or 11-point rule is essentially the conditional expectation, so the polish is the low-variance limit of the squared-residual objective. The stochastic Adam steps themselves always use the two-shock product. Maliar, Maliar and Winant discuss this hybrid (quadrature for a low-dimensional shock, Monte Carlo for the states) as the practical choice for small models; the all-in-one product is the device that survives when the shock vector is high-dimensional.

Checkpoints are kept by a deterministic score, here the same quadrature mean square, not by the closed form.

## Verification 1: Brock–Mirman

Log utility and full depreciation,

$$
\max \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \log c_t,
\qquad
c_t + k_{t+1} = z_t k_t^{\alpha},
\qquad
\log z_{t+1} = \rho \log z_t + \sigma_z \varepsilon_{t+1},
$$

with \(\alpha = 0.36\), \(\beta = 0.96\), \(\rho = 0.95\), \(\sigma_z = 0.02\). The closed form does not depend on the law of motion of \(z\):

$$
k' = \alpha\beta\, z k^{\alpha} = 0.3456\, z k^{0.36}.
$$

The network outputs the consumption share of output through a sigmoid. It is initialised at a share of \(1/2\), not at the steady-state share \(1 - \alpha\beta = 0.6544\), so the initial max relative error on \(k'\) is \(0.448\). Capital and productivity on the reported ergodic set are a cross-section of 8192 agents simulated for 500 periods under the closed form (seed 0). The Euler residual is

$$
e = \beta\, \mathbb{E}\big[(c / c')\, \alpha z' (k')^{\alpha - 1} \big] - 1,
$$

with an 11-point Gauss–Hermite rule. On the closed form itself, \(\max |e| = 6.66\times 10^{-16}\).

Training is 500 Adam steps on a panel of 2048 (width 32, depth 2, four shock pairs, 10% of the panel redrawn from the state domain each step) and 100 L-BFGS steps. Seed 0.

| quantity | value |
| --- | --- |
| max absolute policy error | 9.685243e-05 |
| mean absolute policy error | 5.984188e-05 |
| max relative policy error | 7.028064e-04 |
| mean relative policy error | 3.111198e-04 |
| mean Euler error, \(\log_{10}\|e\|\) | −3.598 |
| max Euler error, \(\log_{10}\|e\|\) | −3.135 |
| max absolute Euler error | 7.336086e-04 |
| Adam time | 5.11 s |
| L-BFGS time | 6.87 s |
| total runtime | 14.48 s |

The success criterion is a max relative policy error below \(10^{-3}\). The run meets it: \(7.028064\times 10^{-4}\). Relative error is \(|k'_{\mathrm{NN}} - k'_{\mathrm{CF}}| / |k'_{\mathrm{CF}}|\). Because \(k' = (1 - s) z k^{\alpha}\), that ratio is the relative error in the saving rate.

Figures: `output/brock_mirman_policy.png`, `output/brock_mirman_errors.png`. The same numbers are in `output/brock_mirman_table.md` and `output/brock_mirman_summary.json`.

## Verification 2: stochastic growth

CRRA utility and partial depreciation, with no closed form:

$$
u(c) = \frac{c^{1-\sigma}}{1-\sigma},
\qquad
c_t + k_{t+1} = (1-\delta)k_t + z_t k_t^{\alpha},
$$

and the same AR(1) for \(\log z\). Parameters: \(\beta = 0.99\), \(\alpha = 0.36\), \(\delta = 0.025\), \(\sigma = 2\), \(\rho = 0.95\), \(\sigma_z = 0.02\). The deterministic steady state is \(k = 37.9893\), \(c = 2.7543\), and the consumption share of cash-on-hand is \(0.06760\). The network is initialised at that share and has to learn the slopes.

Three objects are compared.

1. The network, width 64 and depth 2, trained for 2500 Adam steps (panel 2048, four shock pairs, half the panel redrawn uniformly on the training rectangle) and 100 L-BFGS steps. Seed 1.
2. The log-linear policy at the deterministic steady state,
   \(\widehat{k}' = a\widehat{k} + b\widehat{z}\) and \(\widehat{c} = c_k\widehat{k} + c_z\widehat{z}\), with \(a = 0.976540\), \(b = 0.071970\), \(c_k = 0.462887\), \(c_z = 0.352161\).
3. Euler-equation time iteration on an endogenous grid (Carroll, 2006): 201 capital nodes from \(0.45k\) to \(2.05k\), 31 productivity nodes over \(\pm 5.5\) unconditional standard deviations, 9-point Gauss–Hermite. It converged in 412 iterations. On an interior subset the max Euler residual is \(7.34\times 10^{-8}\). Finite-difference slopes at the steady state differ from \((a, b)\) by \(-4.51\times 10^{-5}\) and \(2.03\times 10^{-5}\).

`dleq.verify_against_linear` is the local comparison. The near-steady-state sample is a \(31\times 31\) grid with capital and productivity each within 1% of steady state. The global grid is \(\pm 3\) unconditional standard deviations of \((\log k, \log z)\) under the log-linear law of motion (\(k/k_{ss}\) from 0.718 to 1.393). The ergodic set is 4096 agents simulated for 400 periods under the endogenous-grid policy. Unconditional standard deviations are 0.111 for \(\log k\) and 0.064 for \(\log z\).

### Policy agreement

| comparison | domain | variable | mean relative | max relative |
| --- | --- | --- | --- | --- |
| NN vs linear | within 1% of steady state | consumption | 1.543498e-04 | 4.841863e-04 |
| NN vs linear | within 1% of steady state | capital_next | 1.225527e-05 | 3.541031e-05 |
| NN vs linear | global grid (±3 s.d.) | consumption | 2.378174e-03 | 9.595239e-03 |
| NN vs linear | global grid (±3 s.d.) | capital_next | 8.923547e-04 | 5.539021e-03 |
| NN vs EGM | ergodic set | consumption | 1.397094e-03 | 2.968717e-03 |
| NN vs EGM | ergodic set | capital_next | 1.022310e-04 | 2.264049e-04 |
| linear vs EGM | ergodic set | consumption | 8.601131e-04 | 3.995862e-03 |
| linear vs EGM | ergodic set | capital_next | 1.628711e-04 | 1.925601e-03 |
| NN vs EGM | global grid (±3 s.d.) | consumption | 1.049561e-03 | 2.962963e-03 |
| NN vs EGM | global grid (±3 s.d.) | capital_next | 7.703076e-05 | 2.255653e-04 |
| linear vs EGM | global grid (±3 s.d.) | consumption | 1.825830e-03 | 8.985272e-03 |
| linear vs EGM | global grid (±3 s.d.) | capital_next | 9.083234e-04 | 5.555928e-03 |
| NN vs EGM | within 1% of steady state | consumption | 7.153806e-04 | 1.137463e-03 |
| NN vs EGM | within 1% of steady state | capital_next | 5.183987e-05 | 8.259297e-05 |
| linear vs EGM | within 1% of steady state | consumption | 7.843922e-04 | 7.912326e-04 |
| linear vs EGM | within 1% of steady state | capital_next | 5.863338e-05 | 6.606402e-05 |

### Euler errors

The residual is \(e = \beta\mathbb{E}[(c'/c)^{-\sigma} R'] - 1\), with \(R' = 1 - \delta + \alpha z' (k')^{\alpha-1}\), again by 9-point Gauss–Hermite. A more negative \(\log_{10}|e|\) is a smaller residual.

| method | domain | mean \(\log_{10}\|e\|\) | max \(\log_{10}\|e\|\) | max \(\|e\|\) |
| --- | --- | --- | --- | --- |
| network | within 1% of steady state | −4.789 | −4.149 | 7.088256e-05 |
| time iteration | within 1% of steady state | −5.444 | −5.238 | 5.782855e-06 |
| log-linear | within 1% of steady state | −4.013 | −3.999 | 1.003301e-04 |
| network | ergodic set | −4.005 | −3.459 | 3.479249e-04 |
| time iteration | ergodic set | −5.278 | −4.687 | 2.057117e-05 |
| log-linear | ergodic set | −3.843 | −2.847 | 1.421597e-03 |

### Runtime

| stage | seconds |
| --- | --- |
| endogenous-grid time iteration | 29.85 |
| Adam | 24.13 |
| L-BFGS | 6.51 |
| total | 64.25 |

Within 1% of steady state the network and the log-linear rule differ by at most \(4.84\times 10^{-4}\) in consumption and \(3.54\times 10^{-5}\) in next capital. On the \(\pm 3\) standard-deviation grid the network stays within \(0.30\%\) of time iteration in consumption, while the log-linear rule reaches \(0.90\%\). On the ergodic set the ordering is closer, because \(\sigma_z = 0.02\) does not push the economy far: the log-linear consumption gap versus time iteration averages \(8.60\times 10^{-4}\), and the network averages \(1.40\times 10^{-3}\). Figures: `output/stochastic_growth_policy.png` and `output/stochastic_growth_errors.png`.

## Tests

`tests/test_loss.py` checks that the two-shock product recovers \((\mathbb{E} f)^2\) rather than \(\mathbb{E}[f^2]\), and that the Gauss–Hermite rule matches the first normal moments. `tests/test_network.py` checks the output bounds. `tests/test_growth_solver.py` checks the log-linear coefficients against the nonlinear Euler equation at first order (the residual is \(O(\varepsilon^2)\), not \(O(\varepsilon)\)) and checks that time iteration reproduces the Brock–Mirman closed form to a relative \(10^{-4}\). `tests/test_brock_mirman.py` checks that a network pinned to the true saving rate has an Euler residual below \(10^{-8}\), then trains a small network and requires the max relative policy error to fall below \(10^{-2}\), starting from above \(0.1\).

## Caveats

- The published Brock–Mirman error is the maximum over one simulated cross-section of 8192 agents, not a proven uniform bound on the whole state space. The policy slice at \(z = 1\) is in `output/brock_mirman_policy_slice.csv`.
- Brock–Mirman saving is a constant, so this example tests the residual, the bound, and the training loop. It is not a test of approximating a richly state-dependent policy. The stochastic-growth example is that test.
- Adam minimises the all-in-one product. L-BFGS then minimises a Gauss–Hermite mean square of the same residual. That polish is the right objective for one shock and is not a substitute for the all-in-one operator in a model with many shocks.
- The log-linear policy is a first-order object. Its consumption and next capital satisfy the resource constraint only up to second order. On the \(\pm 3\) standard-deviation grid the max relative budget gap is \(5.84\times 10^{-3}\); within 1% of steady state it is \(9.14\times 10^{-6}\). The network and the endogenous-grid solution satisfy the budget exactly, because next capital is cash-on-hand minus consumption.
- Time iteration is a numerical benchmark. Its interior Euler residual is at most \(7.34\times 10^{-8}\), and its steady-state slopes match the Riccati coefficients to about \(5\times 10^{-5}\).
- Training clips capital and productivity to the rectangle the endogenous grid covers, and the stochastic-growth example redraws half the panel from that rectangle so the global comparison is interpolation. `mix_fraction=0` is a pure simulated panel.
- Runtimes include JAX compilation on this CPU. A second process that hits the same cache is faster. Seeds are 0 (Brock–Mirman) and 1 (stochastic growth).

## References

Azinovic, M., Gaegauf, L., and Scheidegger, S. (2022). Deep equilibrium nets. *International Economic Review*, 63(4), 1471–1525.

Brock, W. A. and Mirman, L. J. (1972). Optimal economic growth and uncertainty: The discounted case. *Journal of Economic Theory*, 4(3), 479–513.

Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.

Maliar, L., Maliar, S., and Winant, P. (2021). Deep learning for solving dynamic economic models. *Journal of Monetary Economics*, 122, 76–101.
