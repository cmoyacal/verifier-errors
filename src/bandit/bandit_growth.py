#!/usr/bin/env python3
"""Simulate acceptance and hacking growth in a Gaussian contextual bandit. See README.md for usage."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_features(n_prompts, n_per_group, dim, noise_std, mu_n, seed):
    """Paper's centered Gaussian features; shape (prompt, group, response, d)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, noise_std, (n_prompts, n_per_group, dim))
    phi_H = eps - eps.mean(axis=1, keepdims=True)
    phi_H[..., :2] += [1.0, 1.0]
    phi_G = phi_H.copy()
    phi_G[..., 1] *= -1

    xi = rng.normal(0, noise_std, (n_prompts, n_per_group // 2, dim))
    reflected = xi.copy()
    reflected[..., 1] *= -1
    phi_N = np.concatenate([xi, reflected], axis=1)
    phi_N -= phi_N.mean(axis=1, keepdims=True)
    phi_N[..., 1] += mu_n
    return np.stack([phi_G, phi_H, phi_N], axis=1)


def make_offsets(p0, q0, n_per_group):
    """Each member of S receives a_S = log(p_S(0) / n_per_group)."""
    return np.log(np.array([p0 * (1 - q0), p0 * q0, 1 - p0]) / n_per_group)


def population(theta, phi, offsets):
    """Return p_S and grad p_S in G,H,N order, plus g_R = grad p_G+grad p_H."""
    logits = np.einsum("xskd,d->xsk", phi, theta) + offsets[None, :, None]
    pi = softmax(logits.reshape(phi.shape[0], -1), axis=1).reshape(logits.shape)

    mean_phi = np.einsum("xsk,xskd->xd", pi, phi)
    score = phi - mean_phi[:, None, None, :]
    p_S = pi.sum(axis=2).mean(axis=0)
    grad_p_S = np.einsum("xsk,xskd->sd", pi, score) / phi.shape[0]
    g_R = grad_p_S[0] + grad_p_S[1]
    return p_S, grad_p_S, g_R


def metrics(theta, phi, offsets):
    """Current-policy probabilities and the paper's analytic time derivatives."""
    p_S, grad_p_S, g_R = population(theta, phi, offsets)
    p_G, p_H, p_N = p_S
    if np.any(p_S <= 0):
        raise FloatingPointError("A group probability underflowed; shorten the horizon.")
    p = p_G + p_H
    q = p_H / p
    s_G, s_H, s_N = grad_p_S / p_S[:, None]
    difference = s_H - s_G
    leakage = difference @ (s_G - s_N)
    bias = q * (difference @ difference)
    z_dot = difference @ g_R
    z_dot_decomposition = p * (1 - p) * (leakage + bias)
    p_dot = g_R @ g_R
    q_dot = q * (1 - q) * z_dot
    p_G_dot = (1 - q) * p_dot - p * q_dot
    p_H_dot = q * p_dot + p * q_dot
    return dict(p=p, q=q, p_G=p_G, p_H=p_H, p_N=p_N,
                z=np.log(p_H) - np.log(p_G), leakage=leakage, bias=bias,
                z_dot=z_dot, z_dot_decomposition=z_dot_decomposition,
                p_dot=p_dot, q_dot=q_dot, p_G_dot=p_G_dot, p_H_dot=p_H_dot,
                p_G_dot_direct=grad_p_S[0] @ g_R,
                correctness_decline=p * q_dot - (1 - q) * p_dot)


def integrate(phi, offsets, args):
    """Numerical approximation to dot theta = g_R, with exact population RHS."""
    def flow(t, theta):
        return population(theta, phi, offsets)[2]

    def stationary_correctness(t, theta):
        _, grad_p_S, g_R = population(theta, phi, offsets)
        return grad_p_S[0] @ g_R

    solution = solve_ivp(
        flow, (0, args.horizon), np.zeros(args.dim), method="DOP853",
        t_eval=np.linspace(0, args.horizon, args.n_times), dense_output=True,
        events=stationary_correctness, rtol=args.rtol, atol=args.atol,
        max_step=args.max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    rows = [metrics(theta, phi, offsets) for theta in solution.y.T]
    data = {"t": solution.t}
    data.update({key: np.array([row[key] for row in rows]) for key in rows[0]})
    for j in range(args.dim):
        data[f"theta_{j}"] = solution.y[j]
    if not all(np.isfinite(values).all() for values in data.values()):
        raise FloatingPointError("Non-finite trajectory values.")

    data["p_G_declining"] = np.where(data["correctness_decline"] > 0,
                                      data["p_G"], np.nan)
    events = []
    for t in solution.t_events[0]:
        row = metrics(solution.sol(t), phi, offsets)
        events.append([t, row["p"], row["p_G"]])
    return data, np.asarray(events).reshape(-1, 3), solution


def save_table(path, data):
    """Whitespace-separated, named columns: directly readable by PGFPlots."""
    np.savetxt(path, np.column_stack(list(data.values())), fmt="%.16e",
               header=" ".join(data), comments="")


def plot_figures(runs, args, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})

    a_run = next(run for run in runs if run["q0"] == args.a_q0)
    data = a_run["data"]
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
    for key, label, style in [("p", r"Acceptance $p$", "-"),
                              ("q", r"Hacked share $q$", "--"),
                              ("p_G", r"Correctness $p_G$", "-.")]:
        ax.plot(data["t"], data[key], style, label=label, linewidth=1.8)
    ax.set(xlabel="Training time", ylabel="Probability", ylim=(0, 1))
    ax.legend(frameon=False)
    fig.savefig(output / "growth_A.pdf")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
    for run in runs:
        data, events = run["data"], run["events"]
        line, = ax.plot(data["p"], data["p_G"], linewidth=1.1,
                        label=rf"$q(0)={run['q0']:g}$")
        color = line.get_color()
        ax.plot(data["p"], data["p_G_declining"], color=color, linewidth=2.8)
        if len(events):
            ax.plot(events[:, 1], events[:, 2], "o", color=color,
                    markerfacecolor="white", markersize=5)

        distance = np.r_[0, np.cumsum(np.hypot(np.diff(data["p"]),
                                               np.diff(data["p_G"])))]
        if distance[-1] > 1e-10:
            i, j = np.searchsorted(distance, [0.45 * distance[-1], 0.55 * distance[-1]])
            j = min(max(j, i + 1), len(distance) - 1)
            ax.annotate("", xy=(data["p"][j], data["p_G"][j]),
                        xytext=(data["p"][i], data["p_G"][i]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    ax.set(xlabel=r"Verifier reward $J_R=p$", ylabel=r"True correctness $J_C=p_G$")
    ax.legend(frameon=False)
    fig.savefig(output / "growth_B.pdf")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("."), help="Output folder")
    parser.add_argument("--p0", type=float, default=2 / 3, help="Initial acceptance")
    parser.add_argument("--q0", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                        help="Initial hacked shares for B; reuse identical features")
    parser.add_argument("--a-q0", type=float, default=0.5,
                        help="Initial share for A; must occur in --q0")
    parser.add_argument("--mu-n", type=float, default=-0.5,
                        help="Second coordinate of rejected-group mean; other coordinates zero")
    parser.add_argument("--n-prompts", type=int, default=8, help="Equally weighted prompts")
    parser.add_argument("--n-per-group", type=int, default=16, help="Responses per group per prompt; even")
    parser.add_argument("--dim", type=int, default=4, help="Feature dimension, at least two")
    parser.add_argument("--noise-std", type=float, default=0.25, help="Gaussian noise standard deviation")
    parser.add_argument("--seed", type=int, default=0, help="Seed for one fixed feature realization")
    parser.add_argument("--horizon", type=float, default=50, help="Final flow time")
    parser.add_argument("--n-times", type=int, default=501, help="Saved evaluation times, not solver steps")
    parser.add_argument("--rtol", type=float, default=1e-10, help="DOP853 relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-12, help="DOP853 absolute tolerance")
    parser.add_argument("--max-step", type=float, default=0.25, help="Maximum internal integration step")
    parser.add_argument("--width", type=float, default=4.2, help="Each PDF width in inches")
    parser.add_argument("--height", type=float, default=3.2, help="Each PDF height in inches")
    args = parser.parse_args()
    numeric = [args.p0, *args.q0, args.a_q0, args.mu_n, args.noise_std,
               args.horizon, args.rtol, args.atol, args.max_step, args.width, args.height]
    if not np.isfinite(numeric).all():
        parser.error("All numerical arguments must be finite.")
    if not 0 < args.p0 < 1 or any(not 0 < q < 1 for q in args.q0):
        parser.error("p0 and all q0 values must lie strictly between 0 and 1.")
    if len(set(args.q0)) != len(args.q0) or args.a_q0 not in args.q0:
        parser.error("q0 values must be unique, and a-q0 must occur in q0.")
    if args.n_prompts < 1 or args.n_per_group < 2 or args.n_per_group % 2 or args.dim < 2:
        parser.error("Use positive n-prompts, even n-per-group >= 2, and dim >= 2.")
    if args.noise_std < 0 or args.seed < 0 or args.n_times < 3:
        parser.error("Use noise-std >= 0, seed >= 0, and n-times >= 3.")
    if min(args.horizon, args.rtol, args.atol, args.max_step, args.width, args.height) <= 0:
        parser.error("Horizon, solver tolerances, max-step, width and height must be positive.")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    phi = make_features(args.n_prompts, args.n_per_group, args.dim,
                        args.noise_std, args.mu_n, args.seed)
    runs, summaries, offsets_all = [], [], []
    for index, q0 in enumerate(args.q0):
        offsets = make_offsets(args.p0, q0, args.n_per_group)
        data, events, solution = integrate(phi, offsets, args)
        filename = f"trajectory_{index:02d}.dat"
        save_table(args.output / filename, data)
        np.savetxt(args.output / f"stationary_{index:02d}.dat", events,
                   fmt="%.16e", header="t p p_G", comments="")
        if q0 == args.a_q0:
            save_table(args.output / "growth_A.dat", data)
        runs.append(dict(q0=q0, data=data, events=events))
        offsets_all.append(offsets)
        summaries.append(dict(
            q0=q0, trajectory_file=filename, stationary_file=f"stationary_{index:02d}.dat",
            initial={key: float(data[key][0]) for key in ("p", "q", "p_G", "leakage", "bias")},
            final={key: float(data[key][-1]) for key in ("p", "q", "p_G", "p_H")},
            stationary_times=events[:, 0].tolist(),
            min_p_G_dot=float(data["p_G_dot"].min()),
            min_acceptance_increment=float(np.diff(data["p"]).min()),
            max_z_identity_error=float(np.max(np.abs(data["z_dot"] - data["z_dot_decomposition"]))),
            max_correctness_identity_error=float(np.max(np.abs(data["p_G_dot"] - data["p_G_dot_direct"]))),
            rhs_evaluations=solution.nfev))
        print(f"q0={q0:g}: p {data['p'][0]:.4f}->{data['p'][-1]:.4f}, "
              f"q {q0:.4f}->{data['q'][-1]:.4f}, "
              f"p_G {data['p_G'][0]:.4f}->{data['p_G'][-1]:.4f}; "
              f"stationary correctness at t={events[:, 0].round(5).tolist()}")
    plot_figures(runs, args, args.output)

    np.savez(args.output / "bandit_inputs.npz", phi=phi, offsets=np.stack(offsets_all),
             q0=np.asarray(args.q0), group_order=np.array(["G", "H", "N"]))
    config = {**vars(args), "output": str(args.output)}
    (args.output / "run.json").write_text(json.dumps(dict(parameters=config, runs=summaries),
                                                    indent=2) + "\n")
    print(f"Saved separate PDFs, PGFPlots tables, inputs and run.json to {args.output.resolve()}")


if __name__ == "__main__":
    main()
