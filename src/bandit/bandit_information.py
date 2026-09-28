#!/usr/bin/env python3
"""Compare hidden correctness rules along the same verifier-driven trajectory. See README.md for usage."""

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


def save_table(path, data):
    """Whitespace-separated, named columns: directly readable by PGFPlots."""
    np.savetxt(path, np.column_stack(list(data.values())), fmt="%.16e",
               header=" ".join(data), comments="")


def integrate(phi, offsets, args):
    """Verifier-only flow. Correctness is deliberately absent from this function."""
    solution = solve_ivp(
        lambda t, theta: population(theta, phi, offsets)[2],
        (0, args.horizon), np.zeros(args.dim), method="DOP853",
        t_eval=np.linspace(0, args.horizon, args.n_times),
        rtol=args.rtol, atol=args.atol, max_step=args.max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def evaluate(solution, phi, offsets):
    """Evaluate two fixed correctness rules along ONE shared policy trajectory."""
    R = np.array([1, 1, 0])
    c_eq_R = R.copy()
    c1 = np.array([1, 0, 0])
    states = [population(theta, phi, offsets) for theta in solution.y.T]
    p_S = np.stack([state[0] for state in states])
    grad_p_S = np.stack([state[1] for state in states])
    g_R = np.stack([state[2] for state in states])
    p = p_S @ R
    p_G_eq_R = p_S @ c_eq_R
    p_G_c1 = p_S @ c1
    p_H_c1 = p_S @ (R - c1)
    data = dict(t=solution.t, p=p, p_G_c_eq_R=p_G_eq_R,
                p_G_c_le_R=p_G_c1, p_H_c_eq_R=p_S @ (R - c_eq_R),
                p_H_c_le_R=p_H_c1, q_c_le_R=p_H_c1 / p,
                p_dot=np.einsum("td,td->t", g_R, g_R))
    for j in range(phi.shape[-1]):
        data[f"theta_{j}"] = solution.y[j]

    logits = np.einsum("xskd,td->txsk", phi, solution.y.T)
    logits += offsets[None, None, :, None]
    pi = softmax(logits.reshape(len(solution.t), phi.shape[0], -1), axis=2)
    pi = pi.reshape(logits.shape)
    checks = dict(
        max_acceptance_correctness_gap_eq_R=float(np.max(np.abs(p - p_G_eq_R))),
        max_mass_identity_error=float(np.max(np.abs(p - p_G_c1 - p_H_c1))),
        max_policy_normalization_error=float(np.max(np.abs(pi.sum(axis=(2, 3)) - 1))),
        min_hack_probability_c1=float(p_H_c1.min()),
        min_acceptance_increment=float(np.diff(p).min()))
    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError("Non-finite trajectory values.")
    if max(checks[k] for k in checks if k.startswith("max_")) > 1e-10:
        raise RuntimeError(f"Probability identity check failed: {checks}")
    record = dict(t=solution.t, theta=solution.y.T, pi=pi, g_R=g_R,
                  grad_p_S=grad_p_S)
    labels = dict(R=R, c_eq_R=c_eq_R, c1=c1)
    return data, record, labels, checks


def plot_figure(data, markers, args):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
    ax.plot(data["t"], data["p"], color="tab:blue", lw=1.8, label=r"$p$")
    ax.plot(markers["t"], markers["p_G_c_eq_R"], linestyle="none", marker="s",
            markerfacecolor="none", markeredgecolor="black", markersize=5,
            label=r"$p_G\ (c=R)$", clip_on=False)
    ax.plot(data["t"], data["p_G_c_le_R"], color="tab:orange", ls="--", lw=1.8,
            label=r"$p_G\ (c\leq R)$")
    ax.set(xlabel="Training time", ylabel="Probability", xlim=(0, args.horizon),
           ylim=(0, 1))
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(args.output / "information_C.pdf")
    plt.close(fig)


def save_tikz(args):


    template = r"""% Uses your existing 'growth axis' style and groupplot.
% Paths are relative to the LaTeX compilation directory. Override if moved:
% \newcommand{\InformationDataDir}{path/to/data}
\providecommand{\InformationDataDir}{OUTPUT}
\newcommand{\InformationPanel}[1]{%
  \nextgroupplot[growth axis,title={#1},
    xmin=0,xmax=HORIZON,ymin=0,ymax=1,
    xlabel={Training time},ylabel={Probability},
    legend style={at={(0.03,0.03)},anchor=south west,
                  font=\small,draw=none,fill=none,row sep=-1pt}]
  \addplot[blue,line width=1.1pt]
    table[x=t,y=p] {\InformationDataDir/information_C.dat};
  \addlegendentry{$p$}
  \addplot[black,only marks,mark=square,mark size=2pt]
    table[x=t,y=p_G_c_eq_R] {\InformationDataDir/information_C_markers.dat};
  \addlegendentry{$p_G\ (c=R)$}
  \addplot[orange,dashed,line width=1.1pt]
    table[x=t,y=p_G_c_le_R] {\InformationDataDir/information_C.dat};
  \addlegendentry{$p_G\ (c\le R)$}
}
"""
    (args.output / "information_C.tex").write_text(
        template.replace("OUTPUT", args.output.as_posix()).replace("HORIZON", f"{args.horizon:g}"))

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("."), help="Output folder")
    parser.add_argument("--p0", type=float, default=2 / 3, help="Initial acceptance")
    parser.add_argument("--q0", type=float, default=0.3,
                        help="Initial hacked share under c1; offsets stay fixed in both worlds")
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
    numeric = [args.p0, args.q0, args.mu_n, args.noise_std,
               args.horizon, args.rtol, args.atol, args.max_step, args.width, args.height]
    if not np.isfinite(numeric).all():
        parser.error("All numerical arguments must be finite.")
    if not 0 < args.p0 < 1 or not 0 < args.q0 < 1:
        parser.error("p0 and q0 must lie strictly between 0 and 1.")
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
    offsets = make_offsets(args.p0, args.q0, args.n_per_group)
    solution = integrate(phi, offsets, args)
    data, record, labels, checks = evaluate(solution, phi, offsets)
    indices = np.unique(np.linspace(0, args.n_times - 1, min(11, args.n_times), dtype=int))
    markers = {key: data[key][indices] for key in ("t", "p_G_c_eq_R")}
    save_table(args.output / "information_C.dat", data)
    save_table(args.output / "information_C_markers.dat", markers)
    np.savez_compressed(args.output / "verifier_record.npz", **record)
    np.savez(args.output / "bandit_inputs.npz", phi=phi, offsets=offsets,
             group_order=np.array(["G", "H", "N"]), **labels)
    plot_figure(data, markers, args)
    save_tikz(args)
    summary = dict(parameters={**vars(args), "output": str(args.output)}, checks=checks,
                   rhs_evaluations=solution.nfev,
                   initial={key: float(value[0]) for key, value in data.items()},
                   final={key: float(value[-1]) for key, value in data.items()})
    (args.output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"c=R: p_G=p at every saved time (max gap {checks['max_acceptance_correctness_gap_eq_R']:.2e})")
    print(f"c1: p_G {data['p_G_c_le_R'][0]:.4f}->{data['p_G_c_le_R'][-1]:.4f}; "
          f"p_H {data['p_H_c_le_R'][0]:.4f}->{data['p_H_c_le_R'][-1]:.4f}")
    print(f"Saved PDF, TikZ tables and macro, shared record, inputs and run.json to {args.output.resolve()}")


if __name__ == "__main__":
    main()
