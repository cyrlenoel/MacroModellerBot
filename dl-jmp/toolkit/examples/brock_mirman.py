"""Brock–Mirman verification: log utility, full depreciation, closed-form saving.

The equilibrium saving rate is the constant :math:`\\alpha\\beta`, or

.. math::

    k' = \\alpha\\beta\\, z k^{\\alpha}.

The network is not shown this formula. It minimises Euler residuals and is scored
against the formula afterwards. The initial consumption share is one half, not
the steady-state share, so a successful run has to move the policy.

Run from ``dl-jmp/toolkit``::

    python -m examples.brock_mirman
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
from examples.growth_solver import (
    GrowthParams,
    closed_form_controls,
    steady_state,
)
from examples.reporting import dump_json, environment_info, markdown_table, plain, sci

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

@dataclass
class RunConfig:
    alpha: float = 0.36
    beta: float = 0.96
    delta: float = 1.0
    sigma: float = 1.0
    rho: float = 0.95
    sigma_z: float = 0.02
    width: int = 32
    depth: int = 2
    last_weight_scale: float = 0.01
    n_steps: int = 500
    batch_size: int = 2048
    polish_n: int = 4096
    lr: float = 1e-3
    lr_decay_steps: int = 250
    lr_decay_rate: float = 0.5
    lr_end: float = 3e-4
    warmup_steps: int = 40
    n_pairs: int = 4
    lbfgs_steps: int = 100
    lbfgs_pairs: int = 24
    burn_in: int = 20
    mix_fraction: float = 0.10
    simulate_steps: int = 1
    grad_clip: float = 5.0
    seed: int = 0
    log_every: int = 250
    score_every: int = 250
    n_hermite: int = 11
    eval_n: int = 8192
    eval_burn: int = 500
    k_min_factor: float = 0.30
    k_max_factor: float = 3.00
    sample_logk_span: float = 0.45
    sample_logz_std: float = 4.0
    quiet: bool = False
    save: bool = True


def run(config: RunConfig | None = None, **overrides) -> dict:
    """Train the Brock–Mirman policy and score it on the closed-form ergodic set."""
    if config is None:
        config = RunConfig()
    if overrides:
        config = replace(config, **overrides)
    started = time.perf_counter()
    params_model = GrowthParams(
        alpha=config.alpha,
        beta=config.beta,
        delta=config.delta,
        sigma=config.sigma,
        rho=config.rho,
        sigma_z=config.sigma_z,
    )
    ss = steady_state(params_model)
    true_share = 1.0 - params_model.alpha * params_model.beta
    std_log_z = params_model.sigma_z / np.sqrt(1.0 - params_model.rho ** 2)
    scale_log_k = 0.15
    scale_log_z = std_log_z
    k_min = config.k_min_factor * ss["k"]
    k_max = config.k_max_factor * ss["k"]
    logz_min = -6.0 * std_log_z
    logz_max = 6.0 * std_log_z

    policy = dleq.MLPPolicy(
        input_dim=2,
        output_dim=1,
        width=config.width,
        depth=config.depth,
        activation="tanh",
        output_transform="sigmoid",
        output_lower=1e-4,
        output_upper=1.0 - 1e-4,
        final_bias=0.0,
        last_weight_scale=config.last_weight_scale,
    )
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    theta = policy.init(init_key)

    nodes_np, weights_np = dleq.gauss_hermite_normal(config.n_hermite)
    bundle = _operators(
        policy,
        params_model,
        ss,
        scale_log_k,
        scale_log_z,
        k_min,
        k_max,
        logz_min,
        logz_max,
        nodes_np,
        weights_np,
        config.sample_logk_span,
        config.sample_logz_std,
    )

    eval_states = _simulate_closed_form(
        jax.random.PRNGKey(config.seed + 2000),
        config.eval_n,
        config.eval_burn,
        params_model,
        ss,
    )
    monitor_states = _simulate_closed_form(
        jax.random.PRNGKey(config.seed + 1000),
        min(4096, config.eval_n),
        min(300, config.eval_burn),
        params_model,
        ss,
    )
    initial = _score_policy(theta, eval_states, bundle, params_model)
    if not config.quiet:
        print(
            f"closed form k' = {params_model.alpha * params_model.beta:.6f} * z * k^{params_model.alpha}",
            flush=True,
        )
        print(
            f"initial max relative policy error {initial['policy']['max_rel']:.6e}",
            flush=True,
        )

    # Quadrature on a fixed domain sample. Adam still minimises the all-in-one
    # product; this deterministic residual is the L-BFGS polish and the checkpoint
    # score. It does not use the closed form.
    key, score_key = jax.random.split(key)
    score_states = bundle["sample_states"](score_key, config.polish_n)

    def polish_loss(current):
        residual = bundle["euler_quad"](current, score_states)
        return jnp.mean(residual ** 2)

    def score_fn(current):
        return float(polish_loss(current))

    def eval_fn(step, current, loss):
        del step, loss
        metrics = _score_policy(current, monitor_states, bundle, params_model)
        return {"max_rel": metrics["policy"]["max_rel"], "mean_abs": metrics["policy"]["mean_abs"]}

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
        n_shocks=1,
        simulate_steps=config.simulate_steps,
        burn_in=config.burn_in,
        mix_fraction=config.mix_fraction,
        lbfgs_steps=config.lbfgs_steps,
        lbfgs_pairs=config.lbfgs_pairs,
        seed=config.seed,
        log_every=config.log_every,
        score_every=config.score_every,
        quiet=config.quiet,
    )
    result = dleq.train(
        theta,
        bundle["residual_fn"],
        bundle["transition_fn"],
        bundle["sample_states"],
        train_config,
        score_fn=score_fn,
        eval_fn=None if config.quiet else eval_fn,
        polish_loss_fn=polish_loss if config.lbfgs_steps > 0 else None,
    )
    final = _score_policy(result.params, eval_states, bundle, params_model)
    closed_form_euler = _closed_form_euler(eval_states, params_model, nodes_np, weights_np)
    elapsed = time.perf_counter() - started
    summary = {
        "model": "brock_mirman",
        "environment": environment_info(),
        "parameters": {
            "alpha": params_model.alpha,
            "beta": params_model.beta,
            "delta": params_model.delta,
            "sigma": params_model.sigma,
            "rho": params_model.rho,
            "sigma_z": params_model.sigma_z,
            "k_steady": ss["k"],
            "c_steady": ss["c"],
            "true_savings_rate": params_model.alpha * params_model.beta,
            "true_consumption_share": true_share,
            "initial_consumption_share": 0.5,
        },
        "training": {
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "width": config.width,
            "depth": config.depth,
            "lr": config.lr,
            "n_pairs": config.n_pairs,
            "lbfgs_steps": config.lbfgs_steps,
            "lbfgs_pairs": config.lbfgs_pairs,
            "lbfgs_accepted": result.lbfgs_accepted,
            "lbfgs_message": result.lbfgs_message,
            "seed": config.seed,
            "best_step": result.best_step,
            "best_score": result.best_score,
            "nan_resets": result.nan_resets,
            "adam_sec": result.adam_sec,
            "lbfgs_sec": result.lbfgs_sec,
            "train_sec": result.elapsed_sec,
        },
        "evaluation": {
            "distribution": "cross-section of a closed-form simulation after burn-in",
            "n": int(eval_states.shape[0]),
            "burn_in": config.eval_burn,
            "n_hermite": config.n_hermite,
            "initial_policy": initial["policy"],
            "policy": final["policy"],
            "euler": final["euler"],
            "closed_form_euler": log10_abs_stats(closed_form_euler),
        },
        "runtime_sec": elapsed,
        "success_max_rel_below_1e-3": bool(final["policy"]["max_rel"] < 1e-3),
        "history": _clean_history(result.history),
    }
    if not config.quiet:
        print(
            "policy errors on the ergodic set: "
            f"max abs {final['policy']['max_abs']:.6e}  "
            f"mean abs {final['policy']['mean_abs']:.6e}  "
            f"max rel {final['policy']['max_rel']:.6e}  "
            f"mean rel {final['policy']['mean_rel']:.6e}",
            flush=True,
        )
        print(
            "Euler log10 |e|: "
            f"mean {final['euler']['mean_log10_abs']:.3f}  "
            f"max {final['euler']['max_log10_abs']:.3f}",
            flush=True,
        )
        print(f"runtime {elapsed:.2f}s  success {summary['success_max_rel_below_1e-3']}", flush=True)
    if config.save:
        _save(summary, result.params, eval_states, bundle, params_model, ss)
    summary["params"] = result.params
    return summary


def _operators(
    policy,
    model,
    ss,
    scale_log_k,
    scale_log_z,
    k_min,
    k_max,
    logz_min,
    logz_max,
    nodes,
    weights,
    sample_logk_span,
    sample_logz_std,
):
    k_ref = ss["k"]
    logk_low = np.log(k_ref) - sample_logk_span
    logk_high = np.log(k_ref) + sample_logk_span
    logz_draw = sample_logz_std * (model.sigma_z / np.sqrt(1.0 - model.rho ** 2))
    # Python floats, so the quadrature loop unrolls as constants inside jit.
    node_list = tuple(float(value) for value in np.asarray(nodes))
    weight_list = tuple(float(value) for value in np.asarray(weights))

    def features(capital, log_z):
        return jnp.stack(
            [
                (jnp.log(capital) - np.log(k_ref)) / scale_log_k,
                log_z / scale_log_z,
            ],
            axis=-1,
        )

    def controls(theta, capital, log_z):
        share = policy.apply(theta, features(capital, log_z))[..., 0]
        productivity = jnp.exp(log_z)
        cash = (1.0 - model.delta) * capital + productivity * capital ** model.alpha
        consumption = share * cash
        capital_next = cash - consumption
        return consumption, capital_next

    def residual_fn(theta, states, shocks):
        capital = states[:, 0]
        log_z = states[:, 1]
        consumption, capital_next = controls(theta, capital, log_z)
        log_z_next = model.rho * log_z + model.sigma_z * shocks[:, 0]
        consumption_next, _ = controls(theta, capital_next, log_z_next)
        productivity_next = jnp.exp(log_z_next)
        gross = (
            1.0
            - model.delta
            + model.alpha * productivity_next * capital_next ** (model.alpha - 1.0)
        )
        euler = model.beta * (consumption_next / consumption) ** (-model.sigma) * gross - 1.0
        return euler[:, None]

    def transition_fn(theta, states, shocks):
        capital = states[:, 0]
        log_z = states[:, 1]
        _, capital_next = controls(theta, capital, log_z)
        log_z_next = model.rho * log_z + model.sigma_z * shocks[:, 0]
        capital_next = jnp.clip(capital_next, k_min, k_max)
        log_z_next = jnp.clip(log_z_next, logz_min, logz_max)
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
            productivity_next = jnp.exp(log_z_next)
            gross = (
                1.0
                - model.delta
                + model.alpha * productivity_next * capital_next ** (model.alpha - 1.0)
            )
            expectation = expectation + weight * (consumption_next / consumption) ** (-model.sigma) * gross
        return model.beta * expectation - 1.0

    @jax.jit
    def capital_next(theta, states):
        _, k_next = controls(theta, states[:, 0], states[:, 1])
        return k_next

    return {
        "residual_fn": residual_fn,
        "transition_fn": transition_fn,
        "sample_states": sample_states,
        "euler_quad": euler_quad,
        "capital_next": capital_next,
        "k_ref": k_ref,
    }


def _simulate_closed_form(key, n_agents, burn_in, model, ss):
    std = model.sigma_z / jnp.sqrt(1.0 - model.rho ** 2)
    key, key_k, key_z = jax.random.split(key, 3)
    log_k = jnp.log(ss["k"]) + 0.05 * jax.random.normal(key_k, (n_agents,), dtype=jnp.float64)
    log_z = std * jax.random.normal(key_z, (n_agents,), dtype=jnp.float64)

    def body(step, carry):
        del step
        step_key, capital, log_productivity = carry
        step_key, shock_key = jax.random.split(step_key)
        productivity = jnp.exp(log_productivity)
        capital_next = model.alpha * model.beta * productivity * capital ** model.alpha
        shock = jax.random.normal(shock_key, (n_agents,), dtype=jnp.float64)
        log_next = model.rho * log_productivity + model.sigma_z * shock
        return step_key, capital_next, log_next

    _, capital, log_z = jax.lax.fori_loop(0, burn_in, body, (key, jnp.exp(log_k), log_z))
    return jnp.stack([capital, log_z], axis=-1)


def _score_policy(theta, states, bundle, model) -> dict:
    k_hat = np.asarray(bundle["capital_next"](theta, states))
    capital = np.asarray(states[:, 0])
    productivity = np.exp(np.asarray(states[:, 1]))
    _, k_true = closed_form_controls(model, capital, productivity)
    euler = np.asarray(bundle["euler_quad"](theta, states))
    return {"policy": policy_errors(k_hat, k_true), "euler": log10_abs_stats(euler)}


def _closed_form_euler(states, model, nodes, weights) -> np.ndarray:
    capital = np.asarray(states[:, 0])
    productivity = np.exp(np.asarray(states[:, 1]))

    def policy(k, z):
        return closed_form_controls(model, k, z)

    from examples.growth_solver import euler_residual

    return euler_residual(model, capital, productivity, policy, nodes, weights)


def _clean_history(history: dict) -> dict:
    cleaned = {}
    for key, values in history.items():
        if isinstance(values, list):
            cleaned[key] = [None if isinstance(value, float) and not np.isfinite(value) else value for value in values]
        else:
            cleaned[key] = values
    return cleaned


def _save(summary, theta, states, bundle, model, ss) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dump_json(OUTPUT / "brock_mirman_summary.json", summary)
    capital = np.asarray(states[:, 0])
    productivity = np.exp(np.asarray(states[:, 1]))
    k_hat = np.asarray(bundle["capital_next"](theta, states))
    _, k_true = closed_form_controls(model, capital, productivity)
    order = np.argsort(capital)
    # Policy slice at z = 1, spanning the central ergodic range of capital.
    k_lo, k_hi = np.quantile(capital, [0.01, 0.99])
    k_plot = np.linspace(k_lo, k_hi, 200)
    z_plot = np.ones_like(k_plot)
    slice_states = jnp.stack([jnp.asarray(k_plot), jnp.log(z_plot)], axis=-1)
    k_slice = np.asarray(bundle["capital_next"](theta, slice_states))
    _, k_slice_true = closed_form_controls(model, k_plot, z_plot)
    slice_path = OUTPUT / "brock_mirman_policy_slice.csv"
    with slice_path.open("w") as handle:
        handle.write("k,z,kprime_nn,kprime_closed_form,rel_error\n")
        for k_i, knn, kcf in zip(k_plot, k_slice, k_slice_true, strict=True):
            handle.write(f"{k_i:.10e},{1.0:.1f},{knn:.10e},{kcf:.10e},{abs(knn - kcf) / abs(kcf):.10e}\n")

    policy = summary["evaluation"]["policy"]
    euler = summary["evaluation"]["euler"]
    table = markdown_table(
        ["quantity", "value"],
        [
            ["ergodic sample size", summary["evaluation"]["n"]],
            ["max absolute policy error", sci(policy["max_abs"])],
            ["mean absolute policy error", sci(policy["mean_abs"])],
            ["max relative policy error", sci(policy["max_rel"])],
            ["mean relative policy error", sci(policy["mean_rel"])],
            ["mean Euler error, log10", plain(euler["mean_log10_abs"], 3)],
            ["max Euler error, log10", plain(euler["max_log10_abs"], 3)],
            ["max absolute Euler error", sci(euler["max_abs"])],
            ["Adam time (s)", plain(summary["training"]["adam_sec"], 2)],
            ["L-BFGS time (s)", plain(summary["training"]["lbfgs_sec"], 2)],
            ["total runtime (s)", plain(summary["runtime_sec"], 2)],
            ["max relative error below 1e-3", str(summary["success_max_rel_below_1e-3"])],
        ],
    )
    (OUTPUT / "brock_mirman_table.md").write_text(table)
    _figures(k_plot, k_slice, k_slice_true, k_hat[order], k_true[order])
    layers = {}
    for index, layer in enumerate(theta):
        layers[f"w{index}"] = np.asarray(layer["w"])
        layers[f"b{index}"] = np.asarray(layer["b"])
    np.savez(OUTPUT / "brock_mirman_params.npz", **layers)


def _figures(k_plot, k_slice, k_true, k_hat, k_cf) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].plot(k_plot, k_true, color="black", lw=2.0, label=r"closed form $\alpha\beta z k^{\alpha}$")
    axes[0].plot(k_plot, k_slice, color="C0", lw=1.5, ls="--", label="neural network")
    axes[0].set_xlabel(r"capital $k$")
    axes[0].set_ylabel(r"next capital $k'$")
    axes[0].set_title(r"Policy at $z = 1$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)
    relative = np.abs(k_slice - k_true) / np.abs(k_true)
    axes[1].plot(k_plot, relative, color="C0", lw=1.5)
    axes[1].axhline(1e-3, color="C3", ls=":", lw=1.0, label=r"$10^{-3}$")
    axes[1].set_xlabel(r"capital $k$")
    axes[1].set_ylabel(r"$|k'_{NN} - k'_{CF}| / |k'_{CF}|$")
    axes[1].set_title("Relative policy error")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(OUTPUT / "brock_mirman_policy.png", dpi=160)
    plt.close(figure)

    ergodic_rel = np.abs(k_hat - k_cf) / np.abs(k_cf)
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.hist(np.log10(np.maximum(ergodic_rel, 1e-16)), bins=40, color="C0", edgecolor="none")
    axis.axvline(-3.0, color="C3", ls=":", label=r"$10^{-3}$")
    axis.set_xlabel(r"$\log_{10}$ relative policy error")
    axis.set_ylabel("count on the ergodic set")
    axis.set_title("Brock–Mirman ergodic-set errors")
    axis.legend(frameon=False)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT / "brock_mirman_errors.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify the Brock–Mirman model.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    overrides = {"save": not args.no_save, "quiet": args.quiet}
    if args.steps is not None:
        overrides["n_steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.width is not None:
        overrides["width"] = args.width
    if args.lbfgs_steps is not None:
        overrides["lbfgs_steps"] = args.lbfgs_steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    summary = run(**overrides)
    if not summary["success_max_rel_below_1e-3"]:
        raise SystemExit("Brock–Mirman max relative policy error is not below 1e-3")


if __name__ == "__main__":
    main()
