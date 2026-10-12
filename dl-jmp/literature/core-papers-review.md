# Deep-learning solution methods — core paper review (JMP)

*Prepared for Cyrille Lenoel, 12 October 2026. Starting point: Jesús Fernández-Villaverde, "Deep Learning for Solving Economic Models" (lecture slides, dated 8 October 2026), <https://www.sas.upenn.edu/~jesusfv/Slides_Deep_Learning_Solving_Harvard.pdf>.*

**How the references were checked.** The slides cite papers by author and year only. They have **no bibliography**: the slides say the lecture follows Fernández-Villaverde (2026a), *Journal of Economic Literature* 64(3), 829–875, which I have not checked separately. So every full citation below comes from the publisher, NBER/CEPR or the authors' own pages, and I matched each one to the slides' author–year cite and description. Method details and numbers come from the paper versions named in each entry. Where a newer version may differ, I say so.

---

## 1. The common method in one page

**Equilibrium conditions as a loss.** We approximate the unknown policy functions (and, where needed, value functions, prices and Kuhn–Tucker multipliers) with a neural network g_θ(x) of the state x. Every equilibrium condition becomes a residual: Euler equations, Bellman/HJB equations, market clearing and complementary slackness. Training picks θ to minimise a weighted mean of squared residuals over a set of states. These are simulated model states, not data. Some restrictions are built into the output layer rather than added as penalties: a sigmoid for a consumption share, a softplus for a positive price or multiplier. Occasionally binding constraints enter through a complementarity residual, either μ·g(x) or a Fischer–Burmeister function.

**The equilibrium loop.** The right training states are draws from the ergodic distribution, but that distribution depends on the solution. So we simulate with the current network, train on the simulated states, simulate again, and repeat. Each step uses fresh points, so a held-out test set costs nothing.

**The all-in-one (AiO) expectation operator.** An Euler residual contains a conditional expectation. If we square a residual built from one shock draw, we get E[R²] rather than (E[R])², which is biased by the conditional variance. Maliar, Maliar and Winant (2021) take two **independent** draws ε₁, ε₂ at each state and minimise R(x, ε₁)·R(x, ε₂). Conditional on x, its expectation is exactly (E[R | x])². One Monte Carlo average then covers both the integral over states and the integral over future shocks, whatever the number of shocks. The cost is extra noise. The alternatives are quadrature (natural with discrete or few shocks) and, in continuous time, Itô's lemma, which removes the expectation altogether.

**Training: Adam, then L-BFGS.** The slides' recipe has these steps:
1. Scale inputs to [−1, 1] and residuals to comparable economic units.
2. Run Adam with a learning rate around 10⁻³, a decaying schedule, gradient clipping and fresh minibatches. Adam alone typically gets about three digits of accuracy.
3. Once the loss flattens, fix a large set of points, switch to float64 and polish with full-batch L-BFGS. The slides' last example adds a Levenberg–Marquardt step.
4. Judge the solution by Euler errors on held-out points in economic units, and repeat from several seeds.

In fairness, among the papers below only Ebrahimi Kahou et al. (2024) uses L-BFGS. The rest use Adam or AdamW. "Adam then L-BFGS" is the slides' recommendation, drawn mainly from the physics-informed neural network literature, not yet standard practice in economics.

---

## 2. The papers

### 2.1 Maliar, Maliar & Winant (2021)

**Citation.** Maliar, L., Maliar, S. and Winant, P. (2021). "Deep learning for solving dynamic economic models." *Journal of Monetary Economics*, 122, 76–101. <https://doi.org/10.1016/j.jmoneco.2021.07.004>. Slides: "Maliar et al. (2021)".

**Question.** Is there one deep-learning approach that solves dynamic models, heterogeneous-agent models included, when grids are infeasible?

**Method.** The paper writes three objects as expectation functions to minimise: lifetime reward, Bellman residuals and Euler residuals. Expectations use the AiO operator: two independent draws (or batches) for the Euler method, and random initial conditions for the lifetime-reward method. Kuhn–Tucker conditions use the Fischer–Burmeister function. In the consumption–savings application the network has two hidden layers of leaky ReLU units (8×8 up to 64×64). It outputs a consumption-to-wealth share through a sigmoid and a unit-free multiplier. Training uses Adam for 50,000 iterations with 64 random points each, in TensorFlow.

**Result.** The three methods give very similar solutions. Euler residuals are around 0.1–1% in the kinked consumption–savings model, and cost rises roughly linearly with dimension. In Krusell–Smith the network takes the whole panel as input and handles 1,000 agents (2,001 state variables) on a desktop. A solution using only the first moment, Krusell–Smith style, is shifted relative to the full-state solution.

**Reusable.** The AiO operator for continuous shocks. Fischer–Burmeister for non-negativity of gilt purchases and sales. Feeding the household panel directly into the network.

**Limitations.** Accuracy is modest by projection standards, Adam's noise sets a floor, and there is no convergence guarantee.

### 2.2 Azinovic, Gaegauf & Scheidegger (2022) — Deep equilibrium nets

**Citation.** Azinovic, M., Gaegauf, L. and Scheidegger, S. (2022). "Deep equilibrium nets." *International Economic Review*, 63(4), 1471–1525. <https://doi.org/10.1111/iere.12575>. Code: <https://github.com/sischei/DeepEquilibriumNets>. Slides: "Azinovic et al. (2022)".

**Question.** Can we compute recursive equilibria of discrete-time models with heavy heterogeneity, aggregate risk and occasionally binding constraints?

**Method.** One network maps the state to all equilibrium functions: policies, multipliers and the bond price. Firm FOCs, budget constraints and state transitions are imposed exactly, and non-negativity comes from softplus outputs. The loss is the mean squared error of relative Euler errors, KKT complementarity and bond market clearing. Shocks are discrete, so expectations are exact sums. Training alternates between simulating episodes and running Adam on minibatches, and each simulated state is used once. In the benchmark (a 56-generation OLG with illiquid capital and a collateralised bond, 110-dimensional endogenous state) the network has 236 inputs, two hidden layers of 1,000 ReLU units and 221 softplus outputs. Training then moves from a learning rate of 10⁻⁵ with batch 64 to 10⁻⁶ with batch 1,000.

**Result.** Mean relative errors in all equilibrium conditions are below 0.1%, with the 99.9th percentile and maximum below 1%. Appendices cover an analytic-solution test and a Bewley model with Epstein–Zin preferences, which uses a histogram state à la Young.

**Reusable.** The template for a gilt-market block. Market clearing goes straight into the loss and multipliers are network outputs. Adding redundant inputs (prices, aggregates) helps, according to the authors.

**Limitations (the authors').** No convergence guarantee. Training can blow up when simulated states drift outside the region already learned. Softplus multipliers can get "stuck" near zero. Hyperparameters are ad hoc.

### 2.3 Fernández-Villaverde, Hurtado & Nuño (2023)

**Citation.** Fernández-Villaverde, J., Hurtado, S. and Nuño, G. (2023). "Financial frictions and the wealth distribution." *Econometrica*, 91(3), 869–901. <https://doi.org/10.3982/ECTA18180>. Working paper: NBER WP 26302 (2019). Slides: "Fernández-Villaverde, Hurtado, and Nuño (2023): law of motion for the distribution of assets using neural networks in continuous time".

**Question.** How do household heterogeneity and a leveraged financial sector interact to produce endogenous aggregate risk and state-dependent dynamics?

**Method.** This is a continuous-time Krusell–Smith design. Households forecast aggregates through a perceived law of motion (PLM) for aggregate debt, dB_t = h(B_t, N_t) dt, where N is expert net worth. The household HJB and Kolmogorov forward equations are solved by upwind finite differences. The neural network approximates **only** the PLM h, fitted to simulated data. In the 2019 NBER version I read, it is a single hidden layer of 16 softplus units. Training uses full-batch gradient descent with a line search, on simulated points averaged onto a 101×101 grid to remove noise. The first iteration starts from ten random initialisations, later ones warm-start, and the PLM update is damped. The model is also estimated by likelihood.

**Result.** The model has multiple stochastic steady states (high- and low-leverage regions), state-dependent impulse responses, and approximate aggregation fails. In the 2019 version the neural-network PLM has R² = 0.9922 against 0.8275 for a linear Krusell–Smith regression. I have not checked whether the published figures differ.

**Reusable.** A neural-network PLM for aggregate gilt-market states (dealer or arbitrageur net worth, term premium), with households solved by standard methods. Generalised IRFs that depend on the state.

**Limitations.** Only the PLM is learned. Finite differences limit the household block to a few states. Expectations rest on a small set of aggregates.

### 2.4 Ebrahimi Kahou, Fernández-Villaverde, Perla & Sood (2021) — Exploiting symmetry

**Citation.** Ebrahimi Kahou, M., Fernández-Villaverde, J., Perla, J. and Sood, A. (2021). "Exploiting symmetry in high-dimensional dynamic programming." NBER Working Paper 28981 (revised November 2021). <https://doi.org/10.3386/w28981>. Also CEPR DP16285. I found no journal version. Slides: "Ebrahimi Kahou et al. (2021) exploit the symmetric group for models with many heterogeneous agents".

**Question.** How can we solve recursive equilibria with a large but finite number N of heterogeneous agents without the cost exploding in N?

**Method.** If payoffs are permutation-invariant and the law of motion is permutation-equivariant, the optimal policy is permutation-invariant. It can then be written as u(x, X) = ρ(x, (1/N) Σᵢ φ(Xᵢ)), a "deep sets" form. The paper tries three versions of φ: one moment, four moments, and a ReLU network with two layers of 128 units. ρ has four layers of 128 units. Concentration of measure lets one Monte Carlo draw of idiosyncratic shocks stand in for the high-dimensional expectation. Adam minimises squared Euler residuals along 16 simulated trajectories over 64 periods. The application is a multi-firm Lucas–Prescott investment model.

**Result.** Against the exact linear–quadratic solution the method is highly accurate. Solve time is about a minute and does not grow systematically with N. The nonlinear case is no harder to solve. Transversality is never imposed.

**Reusable.** A permutation-invariant embedding for the household panel. It suits exchangeable households and possibly exchangeable habitat investors within a maturity bucket. The LQ benchmark is the model for a linearised validation check.

**Limitations.** It needs exchangeable agents. The moments version sometimes reaches poor local minima. It is still a working paper.

### 2.5 Ebrahimi Kahou, Fernández-Villaverde, Gómez-Cardona, Perla & Rosa (2024) — Spooky boundaries

**Citation.** Ebrahimi Kahou, M., Fernández-Villaverde, J., Gómez-Cardona, S., Perla, J. and Rosa, J. (2024). "Spooky boundaries at a distance: Inductive bias, dynamic models, and behavioral macro." NBER Working Paper 32850. <https://doi.org/10.3386/w32850>. Also CEPR DP19386. The authors' August 2025 revision is retitled "Spooky Boundaries at a Distance: Inductive Bias and Dynamic Macroeconomic Models". I found no journal version. Slides: "Ebrahimi Kahou et al. (2024)", cited for transversality not being imposed and for full-batch L-BFGS.

**Question.** Why do neural-network solvers that never impose transversality still find the stable, saddle-path solution?

**Method.** Over-parameterised networks trained by gradient methods are biased towards min-norm, "flat" solutions. Paths that violate transversality explode and so have large norms. The examples are deterministic and set in sequence space: linear asset pricing, where bubble solutions are excluded, and the neoclassical growth model, including a version with multiple steady states. The network approximates the path (e.g. k(t)) with four hidden tanh layers of 128 units and a softplus output. Training minimises equilibrium residuals on a finite set of dates with full-batch L-BFGS and extrapolates beyond them.

**Result.** The solutions are accurate in the short and medium run without any boundary condition. The method even picks the right path when there are multiple steady states. Errors grow slowly in the extrapolation region.

**Reusable.** Highly relevant to long-gilt pricing, where explosive or bubble price paths satisfy the pricing recursions. The paper also gives a concrete optimiser recipe (L-BFGS).

**Limitations.** **Formulation matters.** Approximating consumption c(t) instead of capital can select the path that violates transversality, even with tiny Euler residuals. The authors suggest approximating states or co-states (marginal utility). The examples are small and deterministic, and the boundary conditions hold only approximately.

### 2.6 Kase, Melosi & Rottner — Estimating nonlinear HA models with neural networks

**Citation.** Kase, H., Melosi, L. and Rottner, M. "Estimating nonlinear heterogeneous agent models with neural networks." Versions: Federal Reserve Bank of Chicago WP 2022-26 (2022); CEPR DP17391; Warwick Economics Research Paper 1499 (2024); BIS Working Paper 1241 (January 2025), <https://www.bis.org/publ/work1241.pdf>; latest author version 17 August 2026. I found no journal version. Slides: "Kase et al. (2024)", which most likely refers to the 2024 version. Details below are from the August 2026 version.

**Question.** Can a nonlinear HANK model with a zero lower bound (ZLB) be solved globally **and** estimated by likelihood?

**Method.** Parameters to be estimated enter as **pseudo-state variables**, so one training run covers the whole (truncated) parameter box. There are two networks, one for individual and one for aggregate policies, plus an auxiliary network for the parameter-dependent steady state. The loss is a weighted sum of squared equilibrium residuals over a batch of parallel economies, each tracking L agents (usually 100; 400 in the quantitative HANK). Expectations use Monte Carlo with antithetic draws (usually 100), and states come from simulation as in Maliar et al. A second "neural-network particle filter" learns the mapping from parameters to the likelihood, which then drives Metropolis–Hastings. Typical settings: five hidden CELU layers of 128 units, AdamW, cosine schedule.

**Result.** The method recovers true parameters in simulations and matches a non-ML time-iteration-plus-particle-filter benchmark. It closely replicates the linearised sequence-space Jacobian (SSJ) solution of a one-asset HANK. In US data, the ZLB interacting with idiosyncratic risk is a key driver of output volatility. The full quantitative pipeline takes about 70 hours on one high-end GPU.

**Reusable.** The pseudo-state approach for IRF matching. **The SSJ comparison is the model for the linearised validation.**

**Limitations.** Accuracy is weak in rare states, for example agents jumping from the borrowing limit to top productivity, or very high wealth. Results depend on the parameter box. Compute is heavy.

### 2.7 Nuño, Renner & Scheidegger (2024) — Monetary policy with persistent supply shocks

**Citation.** Nuño, G., Renner, P. and Scheidegger, S. (2024). "Monetary policy with persistent supply shocks." CEPR Discussion Paper 19678. <https://cepr.org/publications/dp19678>. Also Banco de España WP 2529 (June 2025), <https://doi.org/10.53479/40165>. Code: <https://github.com/phrenner/DEQN_Keras>. Slides: "Nuño et al. (2024): optimal monetary policy with persistent supply shocks". The title itself does not contain "optimal".

**Question.** How should monetary policy respond to a persistent cost-push regime, under a Taylor rule, under discretion and under commitment, with and without the ZLB?

**Method.** The model is a nonlinear NK with a two-state Markov chain for the cost-push shock ("normal" and "bad" times) plus three AR shocks. The authors extend DEQN to Markov switching. Continuous shocks use quadrature with fixed nodes and weights that depend on the regime, computed in batches. Under commitment (timeless perspective) the lagged Lagrange multipliers on the forward-looking equations become two extra state variables. All models use two hidden layers of 512 SELU units and Adam with a learning rate of 10⁻⁵, first in single and then in double precision.

**Result.** Taylor rules fail to stabilise long-run inflation because the natural rate shifts with the regime. Discretion produces an inflation bias in bad times. Commitment leans against the wind but "bygones are bygones". The ZLB strengthens this (the CEPR abstract adds pre-emptive easing).

**Reusable.** Optimal policy with lagged multipliers as states is the route to optimal purchases and QT. Markov regimes can represent QT "on/off". The float32-then-float64 schedule.

**Limitations.** Representative agent and few states. I did not extract a summary accuracy statistic from the paper, so none is quoted here.

### 2.8 Duarte, Duarte & Silva (2024) — Machine learning for continuous-time finance

**Citation.** Duarte, V., Duarte, D. and Silva, D. H. (2024). "Machine learning for continuous-time finance." *Review of Financial Studies*, 37(11), 3217–3271. <https://doi.org/10.1093/rfs/hhae043>. Working paper: CESifo WP 10909 (January 2024). Slides: "Duarte et al. (2024): continuous-time finance".

**Question.** Can high-dimensional nonlinear continuous-time asset pricing, corporate finance and portfolio problems be solved globally?

**Method.** The method is "deep policy iteration". It samples states (uniformly, or from a guess of the ergodic distribution), takes one gradient step of the policy on the HJB (policy improvement), and runs an SGD step on the squared HJB residual for the value network (policy evaluation). Itô's lemma plus automatic differentiation give the exact drift of the value function. A directional second-derivative trick means the cost does not grow with the number of states and grows less than linearly with the number of shocks. All applications use the same networks: 256 units (layer norm), then 128, then 64. Policies use ReLU and values use SiLU, which is twice differentiable as Itô requires. Adam runs at 10⁻³ for the value and 10⁻⁴ for the policy, with batch 2,048.

**Result.** It solves a Lucas orchard economy with up to 100 trees, while Smolyak breaks down beyond 25. It also handles corporate-finance problems with parameters as states, portfolio choice, and jumps.

**Reusable.** If the gilt block is written in continuous time (as in the original Vayanos–Vila preferred-habitat formulation), Itô plus autodiff removes the expectation problem altogether.

**Limitations.** Discrete-time NK and HA blocks do not map straight across. Policy iteration can diverge, which is why the policy learning rate is smaller. Hyperparameters are not tuned.

---

## 3. How this maps onto the gilt preferred-habitat model

**D1. State and architecture.**
- *States:* exogenous shocks; lagged Bank Rate; the central bank's gilt stock by maturity bucket; arbitrageur wealth or risk-bearing capacity; a household block.
- *Household block:* either a panel of L ≈ 100–400 households (Kase et al.), passed through a permutation-invariant embedding (Ebrahimi Kahou et al. 2021), or a histogram (DEQN).
- *Outputs:* yields or prices per bucket (softplus where positivity is needed), household policies as shares (sigmoid), and multipliers.
- *Redundant inputs:* add aggregates (DEQN).

**D2. Purchases and active QT as separate tools.** Model them as two non-negative instruments, each with its own process or rule and its own cost or pass-through parameter. Do not use one signed balance-sheet variable. Non-negativity and "no buying and selling at once" go into the loss as Fischer–Burmeister terms (Maliar et al.). This asymmetry is a real source of nonlinearity that the linearised model cannot capture. Add the Bank Rate lower bound as a second occasionally binding constraint. That also builds the first fallback.

**D3. Expectations.**
- *Euler and pricing equations with continuous shocks:* use AiO with two independent draws, plus antithetic draws to cut noise.
- *Regime shocks (QT on/off):* use exact sums.
- *Risk:* the term-structure recursion has many expectation terms, so AiO noise may dominate. Compare against low-order quadrature early on.
- *Continuous-time gilt block:* consider Itô plus autodiff (Duarte et al.).

**D4. Estimation via pseudo-states and IRF matching.** Treat the key preferred-habitat parameters as pseudo-states over a prior box (Kase et al.): habitat demand elasticities, arbitrageur risk aversion, purchase and QT pass-through. Minimum-distance matching to the high-frequency gilt local-projection IRFs is then cheap. *Risk:* local projections estimate an average response across states. The model counterpart should therefore be generalised IRFs averaged over the ergodic distribution (and, separately, by state), not responses at the deterministic steady state. The short active-QT sample may weakly identify the QT parameters, so report identification diagnostics.

**D5. Training and accuracy.**
- Adam in float32 with cosine decay and clipping, then a fixed large set of points, float64 and an L-BFGS polish (slides; Nuño et al. on precision).
- Multi-seed runs.
- Residuals in bp of yield or % consumption.
- Oversample rare but policy-relevant states (large QT with rates near the floor). Kase et al. and DEQN both show accuracy degrading where simulated data are thin.
- Optimal-policy extensions follow Nuño et al. (lagged multipliers as states).

**D6. Validation against the linearised solution.** Use a subcase where the linearised solution is valid. Switch off the occasionally binding constraints and the purchase/QT asymmetry, use symmetric quadratic costs and small shocks. Then solve the linearised model (SSJ for the HANK version; a standard perturbation solver for a representative-agent preferred-habitat version, e.g. against Kamdar & Ray's setup). Compare four things:
- (i) policy and price functions at the deterministic steady state;
- (ii) the network's Jacobian at the steady state (by autodiff) against the linear decision-rule coefficients;
- (iii) IRFs to small shocks, as Kase et al. do against SSJ;
- (iv) convergence of network IRFs to linear IRFs as shock volatility σ → 0.

Then switch the nonlinearities back on step by step and record where and why the two solutions diverge. That divergence map is itself a result. Also report Euler and pricing errors on held-out simulated states.

**Open risks.**
- (a) Explosive or bubble solutions in long-gilt pricing. Ebrahimi Kahou et al. (2024) suggest the inductive bias helps, but formulation matters: approximate prices or co-states rather than consumption levels, and check long simulations.
- (b) Compute: Kase et al. report about 70 GPU-hours for a full pipeline.
- (c) Identification of the preferred-habitat parameters from a few high-frequency events.
- (d) No convergence guarantee, so multi-seed agreement and the linear benchmark are essential.

**Fallbacks.** The same toolkit carries over: a sectoral input–output NK model with a binding lower bound (DEQN with exact or quadrature expectations, Fischer–Burmeister for the bound), or neural-network estimation of a nonlinear UK model (Kase et al.'s pseudo-states and neural-network particle filter).

---

### Verification notes
- **Slides:** no bibliography, so the full references were matched by author–year and description (see top).
- **Fernández-Villaverde, Hurtado & Nuño (2023):** figures are from the 2019 NBER version and may differ in the published *Econometrica* paper.
- **Kase, Melosi & Rottner:** the slides' "2024" is not tied to a specific version; details are from the August 2026 version.
- **Nuño, Renner & Scheidegger:** the slides' word "optimal" is a description, not part of the title.
- **Not checked:** the JEL reference for the lecture itself (Fernández-Villaverde 2026a), and the Vayanos–Vila mention, which is background and not part of this review's checks.
