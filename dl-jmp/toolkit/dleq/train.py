"""Adam training on the ergodic set, with an optional L-BFGS polish."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from dleq.loss import all_in_one_loss
from dleq.simulate import mix_states, simulate_forward

Params = list | dict
ResidualFn = Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
TransitionFn = Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
ShockFn = Callable[[jax.Array, int], jnp.ndarray]
SampleFn = Callable[[jax.Array, int], jnp.ndarray]
ScoreFn = Callable[[Params], float]
EvalFn = Callable[[int, Params, float], dict]


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train`.

    ``batch_size`` is the number of agents in the simulated panel. Each Adam
    step draws ``n_pairs`` independent shock pairs, forms the all-in-one loss,
    and then moves the panel ``simulate_steps`` periods under the updated
    policy. ``mix_fraction`` redraws that share of the panel from
    ``sample_states`` after the simulation; leave it at zero for a pure
    ergodic-set update.

    The learning rate warms up linearly from ``0.1 * lr`` to ``lr``, then
    decays exponentially toward ``lr_end``. L-BFGS, when requested, minimises
    the same all-in-one loss on one frozen panel and one frozen set of shock
    pairs, so the polish objective is deterministic.
    """

    n_steps: int = 2000
    batch_size: int = 1024
    lr: float = 1e-3
    lr_decay_steps: int = 500
    lr_decay_rate: float = 0.5
    lr_end: float = 1e-5
    warmup_steps: int = 100
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    grad_clip: float | None = 1.0
    n_pairs: int = 4
    n_shocks: int = 1
    simulate_steps: int = 1
    refresh_every: int = 1
    burn_in: int = 0
    mix_fraction: float = 0.0
    lbfgs_steps: int = 0
    lbfgs_pairs: int = 16
    lbfgs_memory: int = 10
    seed: int = 0
    log_every: int = 100
    score_every: int = 100
    quiet: bool = False


@dataclass
class TrainResult:
    """Parameters and the training log.

    If a ``score_fn`` was supplied, ``params`` is the checkpoint with the
    lowest score, including the L-BFGS candidate when that candidate improves
    the score. ``history`` entries are plain Python floats.
    """

    params: Params
    history: dict
    elapsed_sec: float
    adam_sec: float
    lbfgs_sec: float
    final_loss: float
    best_score: float | None
    best_step: int | None
    nan_resets: int
    lbfgs_accepted: bool
    lbfgs_message: str
    config: TrainConfig = field(repr=False)


def train(
    params: Params,
    residual_fn: ResidualFn,
    transition_fn: TransitionFn,
    sample_states: SampleFn,
    config: TrainConfig,
    *,
    shock_fn: ShockFn | None = None,
    score_fn: ScoreFn | None = None,
    eval_fn: EvalFn | None = None,
    polish_states: jnp.ndarray | None = None,
    polish_loss_fn: Callable[[Params], jnp.ndarray] | None = None,
) -> TrainResult:
    """Minimise equilibrium residuals by Adam, then optionally L-BFGS.

    ``residual_fn(params, states, shocks)`` returns an array of shape
    ``(batch,)`` or ``(batch, n_equations)``. The same states and two
    independent shock draws are passed to :func:`dleq.loss.all_in_one_loss`.
    ``transition_fn`` is the state transition used only to refresh the panel.
    ``sample_states(key, n)`` draws the initial panel and any mixed-in states.

    ``score_fn(params)`` should be a deterministic scalar that is smaller when
    the equilibrium conditions are closer to zero (a quadrature mean square is
    the natural choice). It is used to keep the best Adam checkpoint and to
    accept or reject the L-BFGS polish. It is not differentiated.

    By default the L-BFGS polish minimises the all-in-one loss on one frozen
    panel and one frozen set of shock pairs. Pass ``polish_loss_fn`` to polish
    a different scalar built from the same residuals — for a single shock,
    Gauss–Hermite quadrature of the squared residual is the low-variance limit
    of that objective. ``polish_loss_fn`` must be differentiable.
    """
    if config.n_steps < 0 or config.n_pairs < 1 or config.batch_size < 1:
        raise ValueError("n_steps, n_pairs and batch_size must be positive (n_steps may be 0)")
    if shock_fn is None:
        shock_fn = _standard_normal_shocks(config.n_shocks)

    schedule = _learning_rate_schedule(config)
    transforms: list = []
    if config.grad_clip is not None:
        transforms.append(optax.clip_by_global_norm(config.grad_clip))
    transforms.append(
        optax.adam(schedule, b1=config.adam_b1, b2=config.adam_b2, eps=1e-8)
    )
    optimizer = optax.chain(*transforms)
    opt_state = optimizer.init(params)

    update = _adam_step(optimizer, residual_fn, shock_fn, config.n_pairs)
    refresh = _refresh_step(transition_fn, shock_fn, sample_states, config)

    key = jax.random.PRNGKey(config.seed)
    key, sample_key = jax.random.split(key)
    states = sample_states(sample_key, config.batch_size)
    if config.burn_in > 0:
        key, sim_key = jax.random.split(key)
        states, _ = simulate_forward(
            params, states, transition_fn, shock_fn, sim_key, config.burn_in
        )

    history: dict = {"step": [], "loss": [], "lr": [], "score": [], "elapsed_sec": []}
    nan_resets = 0
    best_params = params
    best_score: float | None = None
    best_step: int | None = None
    final_loss = math.nan
    t0 = time.perf_counter()

    if score_fn is not None:
        best_score = float(score_fn(params))
        best_step = 0
        best_params = _copy_params(params)
        history["step"].append(0)
        history["loss"].append(math.nan)
        history["lr"].append(float(schedule(0)))
        history["score"].append(best_score)
        history["elapsed_sec"].append(0.0)
        if not config.quiet:
            print(f"step      0  score {best_score:.6e}  (initial)", flush=True)

    for step in range(config.n_steps):
        key, step_key = jax.random.split(key)
        candidate, candidate_state, loss = update(params, opt_state, states, step_key)
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            nan_resets += 1
            key, reset_key = jax.random.split(key)
            states = sample_states(reset_key, config.batch_size)
            if not config.quiet:
                print(f"step {step + 1:6d}  non-finite loss; panel reset", flush=True)
            continue
        params = candidate
        opt_state = candidate_state
        final_loss = loss_value

        if config.simulate_steps > 0 and config.refresh_every > 0 and (step + 1) % config.refresh_every == 0:
            key, sim_key = jax.random.split(key)
            states = refresh(params, states, sim_key)

        should_log = config.log_every > 0 and (step + 1) % config.log_every == 0
        should_score = score_fn is not None and config.score_every > 0 and (step + 1) % config.score_every == 0
        score_value: float | None = None
        if should_score or (should_log and score_fn is not None and config.score_every <= 0):
            score_value = float(score_fn(params))
            if best_score is None or score_value < best_score:
                best_score = score_value
                best_step = step + 1
                best_params = _copy_params(params)
        if should_log or step + 1 == config.n_steps:
            elapsed = time.perf_counter() - t0
            lr_now = float(schedule(step))
            history["step"].append(step + 1)
            history["loss"].append(loss_value)
            history["lr"].append(lr_now)
            history["score"].append(math.nan if score_value is None else score_value)
            history["elapsed_sec"].append(elapsed)
            extra = ""
            if eval_fn is not None and should_log:
                metrics = eval_fn(step + 1, params, loss_value)
                if metrics:
                    extra = "  " + "  ".join(f"{name} {value:.6e}" for name, value in metrics.items())
            if not config.quiet and should_log:
                score_text = "" if score_value is None else f"  score {score_value:.6e}"
                print(
                    f"step {step + 1:6d}  loss {loss_value:.6e}  lr {lr_now:.3e}{score_text}{extra}",
                    flush=True,
                )

    adam_sec = time.perf_counter() - t0
    if score_fn is not None:
        params = best_params
        # The panel was produced by the last Adam policy. Resimulate under the
        # retained checkpoint so the L-BFGS sample belongs to that policy.
        key, sim_key = jax.random.split(key)
        states, _ = simulate_forward(
            params,
            states,
            transition_fn,
            shock_fn,
            sim_key,
            max(config.burn_in, config.simulate_steps, 1),
        )

    lbfgs_sec = 0.0
    lbfgs_accepted = False
    lbfgs_message = "not requested"
    if config.lbfgs_steps > 0:
        key, polish_key = jax.random.split(key)
        t1 = time.perf_counter()
        if polish_loss_fn is None:
            frozen_states = states if polish_states is None else polish_states
            polished, polish_loss, lbfgs_message = _lbfgs_polish(
                params,
                residual_fn,
                shock_fn,
                frozen_states,
                polish_key,
                n_steps=config.lbfgs_steps,
                n_pairs=config.lbfgs_pairs,
                memory_size=config.lbfgs_memory,
            )
        else:
            polished, polish_loss, lbfgs_message = _lbfgs_minimize(
                params,
                polish_loss_fn,
                n_steps=config.lbfgs_steps,
                memory_size=config.lbfgs_memory,
            )
        lbfgs_sec = time.perf_counter() - t1
        accept = False
        polished_score = None
        if polished is not None and math.isfinite(polish_loss):
            if score_fn is None:
                accept = True
            else:
                polished_score = float(score_fn(polished))
                improved = best_score is None or polished_score < best_score - 1e-15
                if improved:
                    accept = True
                    best_score = polished_score
                    best_step = config.n_steps
                    best_params = _copy_params(polished)
                lbfgs_message += f"; score {polished_score:.6e}"
        if accept and polished is not None:
            params = polished
            lbfgs_accepted = True
            final_loss = polish_loss
            if not config.quiet:
                print(
                    f"lbfgs accepted  loss {polish_loss:.6e}  ({lbfgs_message})",
                    flush=True,
                )
        else:
            params = best_params if score_fn is not None else params
            if not config.quiet:
                print(f"lbfgs rejected  ({lbfgs_message})", flush=True)
        history["lbfgs_loss"] = polish_loss
        history["lbfgs_sec"] = lbfgs_sec

    total = time.perf_counter() - t0
    return TrainResult(
        params=params,
        history=history,
        elapsed_sec=total,
        adam_sec=adam_sec,
        lbfgs_sec=lbfgs_sec,
        final_loss=float(final_loss),
        best_score=best_score,
        best_step=best_step,
        nan_resets=nan_resets,
        lbfgs_accepted=lbfgs_accepted,
        lbfgs_message=lbfgs_message,
        config=config,
    )


def _learning_rate_schedule(config: TrainConfig):
    decay = optax.exponential_decay(
        init_value=config.lr,
        transition_steps=max(config.lr_decay_steps, 1),
        decay_rate=config.lr_decay_rate,
        staircase=False,
        end_value=config.lr_end,
    )
    if config.warmup_steps <= 0:
        return decay
    warmup = optax.linear_schedule(
        init_value=0.1 * config.lr,
        end_value=config.lr,
        transition_steps=config.warmup_steps,
    )
    return optax.join_schedules([warmup, decay], [config.warmup_steps])


def _standard_normal_shocks(n_shocks: int) -> ShockFn:
    def shock_fn(key: jax.Array, n_agents: int) -> jnp.ndarray:
        return jax.random.normal(key, (n_agents, n_shocks), dtype=jnp.float64)

    return shock_fn


def _adam_step(optimizer, residual_fn: ResidualFn, shock_fn: ShockFn, n_pairs: int):
    @jax.jit
    def update(params, opt_state, states, key):
        pair_keys = jax.random.split(key, n_pairs * 2)

        def loss_fn(current):
            total = jnp.zeros((), dtype=states.dtype)
            for index in range(n_pairs):
                shocks_a = shock_fn(pair_keys[2 * index], states.shape[0])
                shocks_b = shock_fn(pair_keys[2 * index + 1], states.shape[0])
                residual_a = residual_fn(current, states, shocks_a)
                residual_b = residual_fn(current, states, shocks_b)
                total = total + all_in_one_loss(residual_a, residual_b)
            return total / n_pairs

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return update


def _refresh_step(transition_fn, shock_fn, sample_states, config: TrainConfig):
    n_sim = config.simulate_steps
    fraction = config.mix_fraction

    @jax.jit
    def refresh(params, states, key):
        key, sim_key, fresh_key, mix_key = jax.random.split(key, 4)
        states, _ = simulate_forward(
            params, states, transition_fn, shock_fn, sim_key, n_sim
        )
        if fraction > 0.0:
            fresh = sample_states(fresh_key, states.shape[0])
            states = mix_states(states, fresh, fraction, mix_key)
        return states

    return refresh


def _lbfgs_minimize(params: Params, loss_fn: Callable[[Params], jnp.ndarray], n_steps: int, memory_size: int):
    """Deterministic L-BFGS. Keeps the iterate with the smallest loss value."""
    baseline = float(loss_fn(params))
    if not math.isfinite(baseline):
        return None, baseline, "non-finite loss before L-BFGS"
    solver = optax.lbfgs(memory_size=memory_size)
    opt_state = solver.init(params)

    @jax.jit
    def step(current, state):
        value, grad = jax.value_and_grad(loss_fn)(current)
        updates, state = solver.update(
            grad, state, current, value=value, grad=grad, value_fn=loss_fn
        )
        current = optax.apply_updates(current, updates)
        return current, state, value

    message = f"started from loss {baseline:.6e}"
    best_loss = baseline
    best_params = params
    try:
        for _ in range(n_steps):
            params, opt_state, value = step(params, opt_state)
            current_loss = float(value)
            if not math.isfinite(current_loss):
                message += "; non-finite step, keeping best finite iterate"
                break
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = _copy_params(params)
    except Exception as exc:  # zoom linesearch can fail on a non-convex residual
        message += f"; L-BFGS raised {type(exc).__name__}: {exc}"
    if best_loss >= baseline * (1.0 - 1e-12):
        return None, best_loss, message + "; polish did not improve the frozen loss"
    return best_params, best_loss, message + f"; best loss {best_loss:.6e}"


def _lbfgs_polish(
    params: Params,
    residual_fn: ResidualFn,
    shock_fn: ShockFn,
    states: jnp.ndarray,
    key: jax.Array,
    *,
    n_steps: int,
    n_pairs: int,
    memory_size: int,
) -> tuple[Params | None, float, str]:
    keys = jax.random.split(key, n_pairs * 2)
    frozen_pairs = [
        (shock_fn(keys[2 * index], states.shape[0]), shock_fn(keys[2 * index + 1], states.shape[0]))
        for index in range(n_pairs)
    ]

    def loss_fn(current):
        total = jnp.zeros((), dtype=states.dtype)
        for shocks_a, shocks_b in frozen_pairs:
            total = total + all_in_one_loss(
                residual_fn(current, states, shocks_a),
                residual_fn(current, states, shocks_b),
            )
        return total / n_pairs

    baseline = float(loss_fn(params))
    if not math.isfinite(baseline):
        return None, baseline, "non-finite AiO loss before L-BFGS"
    solver = optax.lbfgs(memory_size=memory_size)
    opt_state = solver.init(params)

    @jax.jit
    def step(current, state):
        value, grad = jax.value_and_grad(loss_fn)(current)
        updates, state = solver.update(
            grad, state, current, value=value, grad=grad, value_fn=loss_fn
        )
        current = optax.apply_updates(current, updates)
        return current, state, value

    message = f"started from AiO loss {baseline:.6e}"
    last_loss = baseline
    try:
        for _ in range(n_steps):
            params, opt_state, value = step(params, opt_state)
            last_loss = float(value)
            if not math.isfinite(last_loss):
                return None, last_loss, message + "; non-finite L-BFGS step"
    except Exception as exc:  # linesearch can fail on a non-convex residual
        return None, last_loss, message + f"; L-BFGS raised {type(exc).__name__}: {exc}"
    if last_loss > baseline * (1.0 + 1e-8):
        return None, last_loss, message + f"; polish loss {last_loss:.6e} did not improve"
    return params, last_loss, message + f"; polish loss {last_loss:.6e}"


def _copy_params(params: Params) -> Params:
    return jax.tree_util.tree_map(lambda array: jnp.array(array), params)
