"""Stochastic growth with partial depreciation: network vs linear vs time iteration.

There is no closed form. The network is compared with

* a log-linear policy at the deterministic steady state, on a 1% neighbourhood
  and on a wider grid, through :func:`dleq.verify_against_linear`;
* an endogenous-grid time iteration on a fine ``(k, z)`` tensor, which is the
  global benchmark.

Run from ``dl-jmp/toolkit``::

    python -m examples.stochastic_growth
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dleq
from dleq.metrics import log10_abs_stats, policy_errors
from dleq.networks import inverse_output_transform
from examples.growth_solver import (
    GrowthParams,
    egm_controls,
    euler_residual,
    linear_controls,
    loglinear_coefficients,
    solve_egm,
    stationary_variance,
    steady_state,
)
from examples.reporting import dump_json, environment_info, markdown_table, plain, sci

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


@dataclass
class RunConfig:
    alpha: float = 0.36
    beta: float = 0.99
    delta: float = 0.025
    sigma: float = 2.0
    rho: float = 0.95
    sigma_z: float = 0.02
    width: int = 64
    depth: int = 2
    last_weight_scale: float = 0.05
    n_steps: int = 2500
    batch_size: int = 2048
    polish_n: int = 4096
    lr: float = 1e-3
    lr_decay_steps: int = 500
    lr_decay_rate: float = 0.5
    lr_end: float = 1e-4
    warmup_steps: int = 50
    n_pairs: int = 4
    lbfgs_steps: int = 100
    burn_in: int = 40
    mix_fraction: float = 0.5
    simulate_steps: int = 1
    grad_clip: float = 5.0
    seed: int = 1
    log_every: int = 400
    score_every: int = 400
    n_hermite: int = 9
    n_capital: int = 201
    n_productivity: int = 31
    k_min_factor: float = 0.45
    k_max_factor: float = 2.05
    n_std: float = 5.5
    ergodic_n: int = 4096
    ergodic_burn: int = 400
    quiet: bool = False
    save: bool = True


def run(config: RunConfig | None = None, **overrides) -> dict:
    if config is None:
        config = RunConfig()
    if overrides:
        config = replace(config, **overrides)
    started = time.perf_counter()
    model = GrowthParams(
        alpha=config.alpha,
        beta=config.beta,
        delta=config.delta,
        sigma=config.sigma,
        rho=config.rho,
        sigma_z=config.sigma_z,
    )
    ss = steady_state(model)
    coefficients = loglinear_coefficients(model, ss)
    std_log_z = model.sigma_z / np.sqrt(1.0 - model.rho ** 2)
    share_ss = ss["c"] / ss["cash"]
    if not config.quiet:
        print(
            f"steady state k={ss['k']:.6f}  c={ss['c']:.6f}  share={share_ss:.6f}",
            flush=True,
        )
        print(
            f"log-linear a={coefficients['a']:.6f}  b={coefficients['b']:.6f}  "
            f"ck={coefficients['ck']:.6f}  cz={coefficients['cz']:.6f}",
            flush=True,
        )

    t_egm = time.perf_counter()
    solution = solve_egm(
        model,
        n_capital=config.n_capital,
        n_productivity=config.n_productivity,
        n_hermite=config.n_hermite,
        k_min_factor=config.k_min_factor,
        k_max_factor=config.k_max_factor,
        n_std=config.n_std,
        tol=1e-10,
        max_iter=600,
        init_share=share_ss,
    )
    egm_sec = time.perf_counter() - t_egm
    if not solution["converged"]:
        raise RuntimeError(f"endogenous grid did not converge (diff={solution['difference']})")
    if not config.quiet:
        print(
            f"EGM converged in {solution['iterations']} iterations "
            f"(diff {solution['difference']:.3e}) in {egm_sec:.2f}s",
            flush=True,
        )

    benchmark = _benchmark_egm(solution, coefficients, ss, model)
    if not config.quiet:
        print(
            f"EGM slope vs linear: a {benchmark['a_fd']:.6f} vs {coefficients['a']:.6f}, "
            f"b {benchmark['b_fd']:.6f} vs {coefficients['b']:.6f}",
            flush=True,
        )
        print(
            f"EGM interior Euler max |e| {benchmark['euler_max_abs']:.3e} "
            f"(log10 {benchmark['euler_max_log10']:.3f})",
            flush=True,
        )
    if abs(benchmark["a_fd"] - coefficients["a"]) > 0.02:
        raise RuntimeError("endogenous-grid capital slope disagrees with the log-linear coefficient")
    if benchmark["euler_max_abs"] > 1e-4:
        raise RuntimeError("endogenous-grid Euler residual is too large to use as a benchmark")

    raw_bias = inverse_output_transform("sigmoid", 1e-4, 1.0 - 1e-4, share_ss)
    policy = dleq.MLPPolicy(
        input_dim=2,
        output_dim=1,
        width=config.width,
        depth=config.depth,
        activation="tanh",
        output_transform="sigmoid",
        output_lower=1e-4,
        output_upper=1.0 - 1e-4,
        final_bias=raw_bias,
        last_weight_scale=config.last_weight_scale,
    )
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    theta = policy.init(init_key)
    nodes_np, weights_np = dleq.gauss_hermite_normal(config.n_hermite)
    bundle = _operators(
        policy,
        model,
        ss,
        std_log_z,
        config.k_min_factor,
        config.k_max_factor,
        config.n_std,
        nodes_np,
        weights_np,
    )
    key, score_key = jax.random.split(key)
    score_states = bundle["sample_states"](score_key, config.polish_n)

    def polish_loss(current):
        return jnp.mean(bundle["euler_quad"](current, score_states) ** 2)

    def score_fn(current):
        return float(polish_loss(current))

    monitor_k, monitor_z = _monitor_grid(ss, std_log_z)

    def eval_fn(step, current, loss):
        del step, loss
        consumption, _ = _nn_controls(bundle, current, monitor_k, monitor_z)
        reference, _ = egm_controls(solution, monitor_k, monitor_z)
        error = policy_errors(consumption, reference)
        return {"mean_rel_c": error["mean_rel"], "max_rel_c": error["max_rel"]}

    train_config = dleq.TrainConfig(
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        lr=config.lr,
        lr_decay_steps=config.lr_decay_steps,
        lr_decay_rate=config.lr_decay_rate,
        lr_end=config.lr_end,
        warmup_steps=config.warmup_steps,
        grad_clip=config.grad_clip,
        n_pairs=config.n_pairs,
        simulate_steps=config.simulate_steps,
        burn_in=config.burn_in,
        mix_fraction=config.mix_fraction,
        lbfgs_steps=config.lbfgs_steps,
        seed=config.seed,
        log_every=config.log_every,
        score_every=config.score_every,
        quiet=config.quiet,
    )
    trained = dleq.train(
        theta,
        bundle["residual_fn"],
        bundle["transition_fn"],
        bundle["sample_states"],
        train_config,
        score_fn=score_fn,
        eval_fn=None if config.quiet else eval_fn,
        polish_loss_fn=polish_loss if config.lbfgs_steps > 0 else None,
    )
    evaluation = _evaluate(trained.params, bundle, solution, coefficients, ss, model, nodes_np, weights_np, config)
    elapsed = time.perf_counter() - started
    summary = {
        "model": "stochastic_growth",
        "environment": environment_info(),
        "parameters": {
            "alpha": model.alpha,
            "beta": model.beta,
            "delta": model.delta,
            "sigma": model.sigma,
            "rho": model.rho,
            "sigma_z": model.sigma_z,
            "k_steady": ss["k"],
            "c_steady": ss["c"],
            "y_steady": ss["y"],
            "consumption_share_steady": share_ss,
        },
        "loglinear": {key: float(value) for key, value in coefficients.items()},
        "egm": {
            "n_capital": config.n_capital,
            "n_productivity": config.n_productivity,
            "n_hermite": config.n_hermite,
            "iterations": solution["iterations"],
            "difference": solution["difference"],
            "seconds": egm_sec,
            **benchmark,
        },
        "training": {
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "width": config.width,
            "depth": config.depth,
            "seed": config.seed,
            "lbfgs_steps": config.lbfgs_steps,
            "lbfgs_accepted": trained.lbfgs_accepted,
            "lbfgs_message": trained.lbfgs_message,
            "best_step": trained.best_step,
            "best_score": trained.best_score,
            "nan_resets": trained.nan_resets,
            "adam_sec": trained.adam_sec,
            "lbfgs_sec": trained.lbfgs_sec,
            "train_sec": trained.elapsed_sec,
            "initial_consumption_share": share_ss,
        },
        "evaluation": evaluation,
        "runtime_sec": elapsed,
        "history": _clean_history(trained.history),
    }
    if not config.quiet:
        local = evaluation["nn_vs_linear_local"]
        ergodic = evaluation["nn_vs_egm_ergodic"]["consumption"]
        print(
            f"near SS, NN vs linear: max rel c {local['variables'][0]['max_rel']:.6e}  "
            f"max rel k' {local['variables'][1]['max_rel']:.6e}",
            flush=True,
        )
        print(
            f"ergodic set, NN vs EGM consumption: mean rel {ergodic['mean_rel']:.6e}  "
            f"max rel {ergodic['max_rel']:.6e}",
            flush=True,
        )
        print(f"runtime {elapsed:.2f}s", flush=True)
    if config.save:
        _save(summary, trained.params, bundle, solution, coefficients, ss, std_log_z)
    summary["params"] = trained.params
    return summary


def _operators(policy, model, ss, std_log_z, k_min_factor, k_max_factor, n_std, nodes, weights):
    scale_log_k = 0.10
    scale_log_z = std_log_z
    k_min = k_min_factor * ss["k"]
    k_max = k_max_factor * ss["k"]
    logz_half = n_std * std_log_z
    node_list = tuple(float(value) for value in np.asarray(nodes))
    weight_list = tuple(float(value) for value in np.asarray(weights))
    # Train and evaluate inside the endogenous-grid rectangle.
    logk_low = np.log(ss["k"] * (k_min_factor + 0.03))
    logk_high = np.log(ss["k"] * (k_max_factor - 0.03))
    logz_draw = 4.5 * std_log_z

    def features(capital, log_z):
        return jnp.stack(
            [(jnp.log(capital) - np.log(ss["k"])) / scale_log_k, log_z / scale_log_z],
            axis=-1,
        )

    def controls(theta, capital, log_z):
        share = policy.apply(theta, features(capital, log_z))[..., 0]
        productivity = jnp.exp(log_z)
        cash = (1.0 - model.delta) * capital + productivity * capital ** model.alpha
        consumption = share * cash
        return consumption, cash - consumption

    def residual_fn(theta, states, shocks):
        capital = states[:, 0]
        log_z = states[:, 1]
        consumption, capital_next = controls(theta, capital, log_z)
        log_z_next = model.rho * log_z + model.sigma_z * shocks[:, 0]
        consumption_next, _ = controls(theta, capital_next, log_z_next)
        gross = _jax_return(model, capital_next, jnp.exp(log_z_next))
        euler = model.beta * (consumption_next / consumption) ** (-model.sigma) * gross - 1.0
        return euler[:, None]

    def transition_fn(theta, states, shocks):
        capital = states[:, 0]
        log_z = states[:, 1]
        _, capital_next = controls(theta, capital, log_z)
        log_z_next = model.rho * log_z + model.sigma_z * shocks[:, 0]
        capital_next = jnp.clip(capital_next, k_min, k_max)
        log_z_next = jnp.clip(log_z_next, -logz_half, logz_half)
        return jnp.stack([capital_next, log_z_next], axis=-1)

    def sample_states(key, n_agents):
        key_k, key_z = jax.random.split(key)
        draw_k = jax.random.uniform(key_k, (n_agents,), dtype=jnp.float64)
        draw_z = jax.random.uniform(key_z, (n_agents,), dtype=jnp.float64)
        log_k = logk_low + (logk_high - logk_low) * draw_k
        log_z = -logz_draw + 2.0 * logz_draw * draw_z
        return jnp.stack([jnp.exp(log_k), log_z], axis=-1)

    @jax.jit
    def euler_quad(theta, states):
        capital = states[:, 0]
        log_z = states[:, 1]
        consumption, capital_next = controls(theta, capital, log_z)
        expectation = jnp.zeros_like(consumption)
        for node, weight in zip(node_list, weight_list, strict=True):
            log_z_next = model.rho * log_z + model.sigma_z * node
            consumption_next, _ = controls(theta, capital_next, log_z_next)
            gross = _jax_return(model, capital_next, jnp.exp(log_z_next))
            expectation = expectation + weight * (consumption_next / consumption) ** (-model.sigma) * gross
        return model.beta * expectation - 1.0

    @jax.jit
    def controls_of_states(theta, states):
        return controls(theta, states[:, 0], states[:, 1])

    return {
        "residual_fn": residual_fn,
        "transition_fn": transition_fn,
        "sample_states": sample_states,
        "euler_quad": euler_quad,
        "controls_of_states": controls_of_states,
    }


def _jax_return(model, capital_next, productivity_next):
    return 1.0 - model.delta + model.alpha * productivity_next * capital_next ** (model.alpha - 1.0)


def _nn_controls(bundle, theta, capital, productivity):
    capital = np.asarray(capital, dtype=np.float64)
    productivity = np.asarray(productivity, dtype=np.float64)
    shape = np.broadcast(capital, productivity).shape
    capital_flat = np.broadcast_to(capital, shape).ravel()
    productivity_flat = np.broadcast_to(productivity, shape).ravel()
    states = jnp.stack([jnp.asarray(capital_flat), jnp.log(jnp.asarray(productivity_flat))], axis=-1)
    consumption, capital_next = bundle["controls_of_states"](theta, states)
    return np.asarray(consumption).reshape(shape), np.asarray(capital_next).reshape(shape)


def _benchmark_egm(solution, coefficients, ss, model) -> dict:
    eps = 1e-3

    def capital_next(capital, productivity):
        _, k_next = egm_controls(solution, np.asarray(capital), np.asarray(productivity))
        return float(np.asarray(k_next).reshape(-1)[0])

    a_fd = (
        np.log(capital_next(ss["k"] * np.exp(eps), 1.0) / ss["k"])
        - np.log(capital_next(ss["k"] * np.exp(-eps), 1.0) / ss["k"])
    ) / (2.0 * eps)
    b_fd = (
        np.log(capital_next(ss["k"], np.exp(eps)) / ss["k"])
        - np.log(capital_next(ss["k"], np.exp(-eps)) / ss["k"])
    ) / (2.0 * eps)
    # Interior Euler residual of the time-iteration solution.
    k_grid = solution["k_grid"][2:-2:4]
    z_grid = solution["z_grid"][1:-1:2]
    kk, zz = np.meshgrid(k_grid, z_grid, indexing="ij")

    def policy(capital, productivity):
        return egm_controls(solution, capital, productivity)

    residual = euler_residual(model, kk, zz, policy, solution["nodes"], solution["weights"])
    stats = log10_abs_stats(residual)
    return {
        "a_fd": float(a_fd),
        "b_fd": float(b_fd),
        "a_gap": float(a_fd - coefficients["a"]),
        "b_gap": float(b_fd - coefficients["b"]),
        "euler_max_abs": stats["max_abs"],
        "euler_mean_abs": stats["mean_abs"],
        "euler_max_log10": stats["max_log10_abs"],
        "euler_mean_log10": stats["mean_log10_abs"],
        "fd_step": eps,
    }


def _monitor_grid(ss, std_log_z):
    capital = ss["k"] * np.exp(np.linspace(-0.12, 0.12, 16))
    productivity = np.exp(np.linspace(-1.5 * std_log_z, 1.5 * std_log_z, 9))
    kk, zz = np.meshgrid(capital, productivity, indexing="ij")
    return kk.ravel(), zz.ravel()


def _evaluate(theta, bundle, solution, coefficients, ss, model, nodes, weights, config) -> dict:
    std_log_z = model.sigma_z / np.sqrt(1.0 - model.rho ** 2)
    local_k = ss["k"] * (1.0 + np.linspace(-0.01, 0.01, 31))
    local_z = 1.0 + np.linspace(-0.01, 0.01, 31)
    local = _mesh(local_k, local_z)
    # ±3 unconditional standard deviations of the log-linear state, which is
    # inside the endogenous-grid rectangle (about ±6 standard deviations).
    transition = np.array(
        [[coefficients["a"], coefficients["b"]], [0.0, model.rho]],
        dtype=np.float64,
    )
    shock = np.array([[0.0, 0.0], [0.0, model.sigma_z ** 2]], dtype=np.float64)
    variance = stationary_variance(transition, shock)
    std_log_k = float(np.sqrt(variance[0, 0]))
    global_k = ss["k"] * np.exp(np.linspace(-3.0 * std_log_k, 3.0 * std_log_k, 41))
    global_z = np.exp(np.linspace(-3.0 * std_log_z, 3.0 * std_log_z, 21))
    wide = _mesh(global_k, global_z)
    rng = np.random.default_rng(config.seed + 11)
    ergodic_k, ergodic_z = _simulate_egm(solution, config.ergodic_n, config.ergodic_burn, rng)
    domains = {"local": local, "global_grid": wide, "ergodic": (ergodic_k, ergodic_z)}
    out = {
        "ergodic_n": int(ergodic_k.size),
        "ergodic_burn": config.ergodic_burn,
        "k_ergodic_min": float(np.min(ergodic_k)),
        "k_ergodic_max": float(np.max(ergodic_k)),
        "z_ergodic_min": float(np.min(ergodic_z)),
        "z_ergodic_max": float(np.max(ergodic_z)),
        "std_log_k": std_log_k,
        "std_log_z": float(std_log_z),
        "global_k_over_ss": [float(np.exp(-3.0 * std_log_k)), float(np.exp(3.0 * std_log_k))],
        "local_radius": 0.01,
    }
    policies = {}
    for name, (capital, productivity) in domains.items():
        policies[name] = {
            "nn": _nn_controls(bundle, theta, capital, productivity),
            "egm": egm_controls(solution, capital, productivity),
            "linear": linear_controls(ss, coefficients, capital, productivity),
            "states": np.stack([np.asarray(capital).ravel(), np.asarray(productivity).ravel()], axis=1),
        }
    out["nn_vs_linear_local"] = _linear_report(policies["local"], ss)
    out["nn_vs_linear_global"] = _linear_report(policies["global_grid"], ss)
    for domain in ("local", "global_grid", "ergodic"):
        out[f"nn_vs_egm_{domain}"] = _pair(policies[domain]["nn"], policies[domain]["egm"])
        out[f"linear_vs_egm_{domain}"] = _pair(policies[domain]["linear"], policies[domain]["egm"])
        out[f"linear_budget_gap_{domain}"] = _budget_gap(model, policies[domain]["states"], policies[domain]["linear"])
    for domain in ("local", "ergodic"):
        capital = policies[domain]["states"][:, 0]
        productivity = policies[domain]["states"][:, 1]
        # The network's state vector is (k, log z). The stored states are levels.
        log_states = np.stack([capital, np.log(productivity)], axis=1)
        out[f"euler_{domain}"] = {
            "nn": log10_abs_stats(np.asarray(bundle["euler_quad"](theta, jnp.asarray(log_states)))),
            "egm": log10_abs_stats(euler_residual(model, capital, productivity, lambda k, z: egm_controls(solution, k, z), nodes, weights)),
            "linear": log10_abs_stats(euler_residual(model, capital, productivity, lambda k, z: linear_controls(ss, coefficients, k, z), nodes, weights)),
        }
    out["_plot"] = {
        "k_over_ss": global_k / ss["k"],
        "z": np.ones_like(global_k),
    }
    # Store the z=1 slice used in the figure, computed here so saving does not retrain.
    slice_k = global_k
    slice_z = np.ones_like(slice_k)
    out["slice"] = {
        "k": slice_k,
        "nn": _nn_controls(bundle, theta, slice_k, slice_z),
        "egm": egm_controls(solution, slice_k, slice_z),
        "linear": linear_controls(ss, coefficients, slice_k, slice_z),
    }
    return out


def _mesh(capital, productivity):
    kk, zz = np.meshgrid(capital, productivity, indexing="ij")
    return kk.ravel(), zz.ravel()


def _linear_report(bundle, ss):
    nn_c, nn_k = bundle["nn"]
    lin_c, lin_k = bundle["linear"]
    return dleq.verify_against_linear(
        bundle["states"],
        np.stack([np.asarray(nn_c).ravel(), np.asarray(nn_k).ravel()], axis=1),
        np.stack([np.asarray(lin_c).ravel(), np.asarray(lin_k).ravel()], axis=1),
        steady_state=np.array([ss["k"], ss["z"]]),
        radius=0.01,
        distance="relative",
        names=("consumption", "capital_next"),
    )


def _pair(left, right):
    return {
        "consumption": policy_errors(np.asarray(left[0]).ravel(), np.asarray(right[0]).ravel()),
        "capital_next": policy_errors(np.asarray(left[1]).ravel(), np.asarray(right[1]).ravel()),
    }


def _budget_gap(model, states, controls) -> dict[str, float]:
    capital = states[:, 0]
    productivity = states[:, 1]
    consumption = np.asarray(controls[0]).ravel()
    capital_next = np.asarray(controls[1]).ravel()
    cash = (1.0 - model.delta) * capital + productivity * capital ** model.alpha
    gap = np.abs(consumption + capital_next - cash) / np.maximum(np.abs(cash), 1e-12)
    return {"mean_rel": float(np.mean(gap)), "max_rel": float(np.max(gap))}


def _simulate_egm(solution, n_agents, burn_in, rng):
    model = solution["params"]
    ss = solution["ss"]
    std = model.sigma_z / np.sqrt(1.0 - model.rho ** 2)
    log_k = np.log(ss["k"]) + 0.01 * rng.normal(size=n_agents)
    log_z = std * rng.normal(size=n_agents)
    for _ in range(burn_in):
        _, capital_next = egm_controls(solution, np.exp(log_k), np.exp(log_z))
        shock = rng.normal(size=n_agents)
        log_z = model.rho * log_z + model.sigma_z * shock
        log_k = np.log(np.maximum(capital_next, 1e-10))
    return np.exp(log_k), np.exp(log_z)


def _clean_history(history: dict) -> dict:
    cleaned = {}
    for key, values in history.items():
        if isinstance(values, list):
            cleaned[key] = [None if isinstance(value, float) and not np.isfinite(value) else value for value in values]
        else:
            cleaned[key] = values
    return cleaned


def _save(summary, theta, bundle, solution, coefficients, ss, std_log_z) -> None:
    del bundle, solution, coefficients, std_log_z
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    evaluation = dict(payload["evaluation"])
    plot_slice = evaluation.pop("slice")
    evaluation.pop("_plot", None)
    payload["evaluation"] = evaluation
    dump_json(OUTPUT / "stochastic_growth_summary.json", payload)
    _write_table(payload)
    _write_slice(plot_slice)
    _figures(plot_slice, ss)
    layers = {}
    for index, layer in enumerate(theta):
        layers[f"w{index}"] = np.asarray(layer["w"])
        layers[f"b{index}"] = np.asarray(layer["b"])
    np.savez(OUTPUT / "stochastic_growth_params.npz", **layers)


def _write_slice(plot_slice) -> None:
    capital = np.asarray(plot_slice["k"])
    columns = {
        "k": capital,
        "c_nn": np.asarray(plot_slice["nn"][0]),
        "c_egm": np.asarray(plot_slice["egm"][0]),
        "c_linear": np.asarray(plot_slice["linear"][0]),
        "kprime_nn": np.asarray(plot_slice["nn"][1]),
        "kprime_egm": np.asarray(plot_slice["egm"][1]),
        "kprime_linear": np.asarray(plot_slice["linear"][1]),
    }
    path = OUTPUT / "stochastic_growth_policy_slice.csv"
    header = list(columns)
    with path.open("w") as handle:
        handle.write(",".join(header) + "\n")
        for index in range(capital.size):
            handle.write(",".join(f"{columns[name][index]:.10e}" for name in header) + "\n")


def _write_table(summary) -> None:
    evaluation = summary["evaluation"]
    rows = []
    local = evaluation["nn_vs_linear_local"]
    for variable in local["variables"]:
        rows.append([
            "NN vs linear",
            "within 1% of steady state",
            variable["name"],
            sci(variable["mean_rel"]),
            sci(variable["max_rel"]),
        ])
    wide = evaluation["nn_vs_linear_global"]
    for variable in wide["variables"]:
        rows.append([
            "NN vs linear",
            "global grid (±3 s.d.)",
            variable["name"],
            sci(variable["mean_rel"]),
            sci(variable["max_rel"]),
        ])
    for label, key in (
        ("NN vs EGM", "nn_vs_egm_ergodic"),
        ("linear vs EGM", "linear_vs_egm_ergodic"),
        ("NN vs EGM", "nn_vs_egm_global_grid"),
        ("linear vs EGM", "linear_vs_egm_global_grid"),
        ("NN vs EGM", "nn_vs_egm_local"),
        ("linear vs EGM", "linear_vs_egm_local"),
    ):
        domain = {
            "nn_vs_egm_ergodic": "ergodic set",
            "linear_vs_egm_ergodic": "ergodic set",
            "nn_vs_egm_global_grid": "global grid (±3 s.d.)",
            "linear_vs_egm_global_grid": "global grid (±3 s.d.)",
            "nn_vs_egm_local": "within 1% of steady state",
            "linear_vs_egm_local": "within 1% of steady state",
        }[key]
        for variable in ("consumption", "capital_next"):
            stats = evaluation[key][variable]
            rows.append([label, domain, variable, sci(stats["mean_rel"]), sci(stats["max_rel"])])
    table = markdown_table(
        ["comparison", "domain", "variable", "mean relative", "max relative"],
        rows,
    )
    euler_rows = []
    for domain in ("local", "ergodic"):
        for method in ("nn", "egm", "linear"):
            stats = evaluation[f"euler_{domain}"][method]
            euler_rows.append([
                method,
                "within 1% of steady state" if domain == "local" else "ergodic set",
                plain(stats["mean_log10_abs"], 3),
                plain(stats["max_log10_abs"], 3),
                sci(stats["max_abs"]),
            ])
    euler = markdown_table(
        ["method", "domain", "mean log10 |e|", "max log10 |e|", "max |e|"],
        euler_rows,
    )
    runtime = markdown_table(
        ["stage", "seconds"],
        [
            ["endogenous-grid time iteration", plain(summary["egm"]["seconds"], 2)],
            ["Adam", plain(summary["training"]["adam_sec"], 2)],
            ["L-BFGS", plain(summary["training"]["lbfgs_sec"], 2)],
            ["total", plain(summary["runtime_sec"], 2)],
        ],
    )
    (OUTPUT / "stochastic_growth_table.md").write_text(
        "### Policy agreement\n\n" + table + "\n### Euler errors\n\n" + euler + "\n### Runtime\n\n" + runtime
    )


def _figures(plot_slice, ss) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    capital = np.asarray(plot_slice["k"]) / ss["k"]
    c_nn = np.asarray(plot_slice["nn"][0]) / ss["c"]
    c_egm = np.asarray(plot_slice["egm"][0]) / ss["c"]
    c_lin = np.asarray(plot_slice["linear"][0]) / ss["c"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    axes[0].plot(capital, c_egm, color="black", lw=2.0, label="time iteration (EGM)")
    axes[0].plot(capital, c_lin, color="C1", lw=1.4, label="log-linear")
    axes[0].plot(capital, c_nn, color="C0", lw=1.4, ls="--", label="neural network")
    axes[0].set_xlabel(r"$k / k_{ss}$")
    axes[0].set_ylabel(r"$c / c_{ss}$")
    axes[0].set_title(r"Consumption policy at $z = 1$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)
    nn_gap = np.abs(np.asarray(plot_slice["nn"][0]) - np.asarray(plot_slice["egm"][0])) / np.asarray(plot_slice["egm"][0])
    lin_gap = np.abs(np.asarray(plot_slice["linear"][0]) - np.asarray(plot_slice["egm"][0])) / np.asarray(plot_slice["egm"][0])
    axes[1].plot(capital, lin_gap, color="C1", lw=1.4, label="log-linear vs EGM")
    axes[1].plot(capital, nn_gap, color="C0", lw=1.4, label="network vs EGM")
    axes[1].set_xlabel(r"$k / k_{ss}$")
    axes[1].set_ylabel("relative consumption gap")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Gap from the global solution at $z = 1$")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(OUTPUT / "stochastic_growth_policy.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(capital, nn_gap, color="C0", lw=1.6, label="network")
    axis.plot(capital, lin_gap, color="C1", lw=1.6, label="log-linear")
    axis.set_xlabel(r"$k / k_{ss}$")
    axis.set_ylabel(r"$|c - c_{\mathrm{EGM}}| / c_{\mathrm{EGM}}$")
    axis.set_yscale("log")
    axis.set_title("Where the linear solution leaves the global policy")
    axis.legend(frameon=False)
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(OUTPUT / "stochastic_growth_errors.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stochastic growth model and compare benchmarks.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    overrides = {"save": not args.no_save, "quiet": args.quiet}
    if args.steps is not None:
        overrides["n_steps"] = args.steps
    if args.lbfgs_steps is not None:
        overrides["lbfgs_steps"] = args.lbfgs_steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    run(**overrides)


if __name__ == "__main__":
    main()
