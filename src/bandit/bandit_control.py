#!/usr/bin/env python3
"""Compare verifier, regularized, raw-audit, and projected-audit bandit flows. See README.md for usage."""
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

def reward_hessian_vector(theta, phi, offsets, vector):
    """B_R v = E_x E_pi[(R-p_x) s (s^T v)], with prompt-local p_x."""
    logits = np.einsum('xskd,d->xsk', phi, theta) + offsets[None, :, None]
    pi = softmax(logits.reshape(phi.shape[0], -1), axis=1).reshape(logits.shape)
    score = phi - np.einsum('xsk,xskd->xd', pi, phi)[:, None, None, :]
    R = np.array([1., 1., 0.])[None, :, None]
    p_x = (pi * R).sum(axis=(1, 2))
    weights = pi * (R - p_x[:, None, None])
    return np.einsum('xsk,xskd,xsk->d', weights, score,
                     np.einsum('xskd,d->xsk', score, vector)) / phi.shape[0]


def direction(theta, phi, offsets, method, args):
    """Recompute gradients and correction at the current policy."""
    masses, grads, g = population(theta, phi, offsets)
    good, hack = (0, 1) if args.rule == 'c1' else (1, 0)
    h = grads[hack]
    g2 = g @ g
    projected_h = h - g * ((g @ h) / g2) if g2 > 0 else h.copy()
    strength = projected_h @ projected_h
    drift = h @ g
    gain = 0.
    if method == 'rlvr':
        u = np.zeros_like(g)
    elif method == 'gr':
        u = -2 * args.gamma * reward_hessian_vector(theta, phi, offsets, g)
    else:
        projected = method.startswith('projected')
        correction_gradient = projected_h if projected else h
        gain = args.gain
        if method.endswith('adaptive'):
            denominator = strength if projected else h @ h


            if denominator <= args.strength_tol:
                raise FloatingPointError(
                    f'{method}: correction strength {denominator:.3e} <= '
                    f'{args.strength_tol:.3e}; shorten the horizon or inspect the geometry.')
            gain = max(0., drift) / denominator + args.adaptive_delta
        u = -gain * correction_gradient
    return g + u, (masses, grads, g, h, projected_h, u, gain, drift, strength, good, hack)


def metrics(theta, phi, offsets, method, args):
    v, (masses, grads, g, h, ph, u, gain, drift, strength, good, hack) = direction(
        theta, phi, offsets, method, args)
    p_G, p_H = masses[good], masses[hack]
    p = p_G + p_H
    if min(p_G, p_H, masses[2]) <= 0:
        raise FloatingPointError('Group mass underflow; shorten the horizon.')
    p_G_dot, p_H_dot, p_dot = grads[good] @ v, h @ v, g @ v
    d = h / p_H - grads[good] / p_G
    leakage = d @ (grads[good] / p_G - grads[2] / masses[2])
    q = p_H / p
    bias = q * (d @ d)
    row = dict(p=p, p_G=p_G, p_H=p_H, p_N=masses[2], q=q,
               p_dot=p_dot, p_G_dot=p_G_dot, p_H_dot=p_H_dot,
               q_dot=(p_H_dot - q * p_dot) / p, z_dot=d @ v,
               leakage=leakage, bias=bias, verifier_z_dot=d @ g,
               gain=gain, drift=drift, projected_strength=strength,
               projected_margin=gain * strength - drift,
               correctness_audit_alignment=grads[good] @ h,
               correctness_correction=grads[good] @ u,
               raw_correctness_margin=gain * (grads[good] @ h) - grads[good] @ g,
               raw_hack_margin=gain * (h @ h) - drift,
               delta_R=g @ u, g_R_norm_sq=g @ g, control_norm=np.linalg.norm(u),
               F_GR=p - args.gamma * (g @ g),
               selective=float(p_H_dot < -args.selectivity_tol and
                               p_G_dot >= -args.selectivity_tol),
               collateral_suppression=float(p_H_dot < -args.selectivity_tol and
                                             p_G_dot < -args.selectivity_tol),
               swapped_p_G=p_H, swapped_p_H=p_G,
               swapped_p_G_dot=p_H_dot, swapped_p_H_dot=p_G_dot)
    for j in range(args.dim):
        row[f'theta_{j}'] = theta[j]
        row[f'u_{j}'] = u[j]
    return row


def integrate(theta0, phi, offsets, method, args):
    flow = lambda t, theta: direction(theta, phi, offsets, method, args)[0]
    def hack_boundary(t, theta):
        return metrics(theta, phi, offsets, method, args)['p_H_dot'] + args.selectivity_tol
    def good_boundary(t, theta):
        return metrics(theta, phi, offsets, method, args)['p_G_dot'] + args.selectivity_tol
    solution = solve_ivp(flow, (0, args.horizon), theta0, method='DOP853',
                         t_eval=np.linspace(0, args.horizon, args.n_times),
                         dense_output=True, events=(hack_boundary, good_boundary),
                         rtol=args.rtol, atol=args.atol, max_step=args.max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    rows = [metrics(theta, phi, offsets, method, args) for theta in solution.y.T]
    data = dict(t=solution.t, training_time=solution.t + args.warmup)
    data.update({k: np.asarray([row[k] for row in rows]) for k in rows[0]})
    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError(f'Non-finite data in {method}.')
    boundaries = np.unique(np.r_[0., *solution.t_events, args.horizon])
    intervals, collateral_intervals = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        row = metrics(solution.sol((left + right) / 2), phi, offsets, method, args)
        if row['selective']:
            intervals.append([float(left), float(right)])
        if row['collateral_suppression']:
            collateral_intervals.append([float(left), float(right)])
    summary = dict(method=method, rhs_evaluations=solution.nfev,
                   initial=rows[0], final=rows[-1],
                   selective_fraction_saved=float(data['selective'].mean()),
                   selective_intervals=intervals,
                   selective_duration=float(sum(b-a for a, b in intervals)),
                   collateral_intervals=collateral_intervals,
                   collateral_duration=float(sum(b-a for a, b in collateral_intervals)),
                   min_correctness_audit_alignment=float(data['correctness_audit_alignment'].min()),
                   max_p_G_dot=float(data['p_G_dot'].max()),
                   min_p_G_dot=float(data['p_G_dot'].min()),
                   max_p_H_dot=float(data['p_H_dot'].max()),
                   min_p_H_dot=float(data['p_H_dot'].min()),
                   hack_rate_boundary_times=solution.t_events[0].tolist(),
                   good_rate_boundary_times=solution.t_events[1].tolist(),
                   max_gain=float(data['gain'].max()),
                   min_projected_margin=float(data['projected_margin'].min()),
                   max_abs_delta_R=float(np.abs(data['delta_R']).max()),
                   max_mass_rate_error=float(np.max(np.abs(
                       data['p_dot'] - data['p_G_dot'] - data['p_H_dot']))))
    return data, summary


def make_checkpoint(phi, offsets, args):
    theta0 = np.zeros(args.dim)
    if args.warmup == 0:
        return theta0
    sol = solve_ivp(lambda t, th: population(th, phi, offsets)[2],
                    (0, args.warmup), theta0, method='DOP853',
                    rtol=args.rtol, atol=args.atol, max_step=args.max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1]


STYLES = {'rlvr': ('black', 'solid', 'RLVR'),
          'gr': ('tab:orange', 'dotted', 'Gradient reg.'),
          'raw': ('tab:green', 'dashed', 'Audit'),
          'projected': ('tab:blue', 'solid', 'Projected audit')}


def export_comparison(runs, mode, args):
    """One PDF per gain mode. Individual seeds plus a mean parametric curve."""
    methods = ['rlvr', 'gr', f'raw_{mode}', f'projected_{mode}']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout='constrained')
    tex = [r'\providecommand{\ControlDataDir}{' + args.output.as_posix() + '}',
           r'\def\ControlPanel#1{%',
           r'  \nextgroupplot[growth axis,title={#1},xlabel={$p_H$},ylabel={$p_G$},',
           r'    legend style={at={(0.97,0.97)},anchor=north east,',
           r'      font=\small,draw=none,fill=none,row sep=-1pt}]']
    tikz_colors = dict(rlvr='black', gr='orange', raw='green!60!black', projected='blue')
    for method in methods:
        base = method.split('_')[0]
        color, ls, label = STYLES[base]
        samples = [run[method] for run in runs]
        aggregate = {'t': samples[0]['t']}
        for key in samples[0]:
            if key == 't':
                continue
            values = np.stack([s[key] for s in samples])
            aggregate[key + '_mean'] = values.mean(axis=0)
            aggregate[key + '_std'] = values.std(axis=0, ddof=1) if len(samples)>1 else np.zeros_like(values[0])
        save_table(args.output / f'mean_{method}.dat', aggregate)
        if len(samples) > 1:
            for seed, sample in zip(args.seeds, samples):
                ax.plot(sample['p_H'], sample['p_G'], color=color, ls=ls, lw=.6, alpha=.2)
                tex += [rf'  \addplot[{tikz_colors[base]},{ls},thin,opacity=0.2,forget plot]',
                        rf'    table[x=p_H,y=p_G] {{\ControlDataDir/seed_{seed:03d}/{method}.dat}};']
        x, y = aggregate['p_H_mean'], aggregate['p_G_mean']
        ax.plot(x, y, color=color, ls=ls, lw=1.8, label=label)
        length = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        if length[-1] > 1e-10:
            i, j = np.searchsorted(length, [.45*length[-1], .55*length[-1]])
            j = min(max(i+1, j), len(x)-1)
            ax.annotate('', (x[j], y[j]), (x[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        tex += [rf'  \addplot[{tikz_colors[base]},{ls},line width=1.1pt,',
                r'    postaction={decorate},decoration={markings,mark=at position 0.55 with {\arrow{stealth}}}]',
                rf'    table[x=p_H_mean,y=p_G_mean] {{\ControlDataDir/mean_{method}.dat}};',
                rf'  \addlegendentry{{{label}}}']
    starts = np.array([[run['rlvr']['p_H'][0], run['rlvr']['p_G'][0]] for run in runs])
    ax.scatter(starts[:, 0], starts[:, 1], s=16, facecolors='white', edgecolors='black', zorder=5)
    save_table(args.output / 'checkpoints.dat', {'p_H':starts[:, 0], 'p_G':starts[:, 1]})
    tex += [r'  \addplot[black,only marks,mark=o,mark options={fill=white},forget plot]',
            r'    table[x=p_H,y=p_G] {\ControlDataDir/checkpoints.dat};', '}']
    (args.output / f'control_D_{mode}.tex').write_text('\n'.join(tex) + '\n')
    ax.set(xlabel=r'Accepted errors $p_H$', ylabel=r'Correct responses $p_G$')
    ax.legend(frameon=False, loc='upper right')
    fig.savefig(args.output / f'control_D_{mode}.pdf')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--output', type=Path, default=Path('.'), help='Output folder')
    p.add_argument('--seed', type=int, default=0, help='First feature seed')
    p.add_argument('--n-seeds', type=int, default=10, help='Consecutive feature seeds')
    p.add_argument('--gain-mode', choices=['fixed','adaptive','both'], default='both', help='Audit gain schedule')
    p.add_argument('--lambda', dest='gain', type=float, default=1., help='Fixed audit gain')
    p.add_argument('--adaptive-delta', type=float, default=1., help='Positive additive gain above the local threshold')
    p.add_argument('--gamma', type=float, default=16., help='Gradient regularization weight')
    p.add_argument('--rule', choices=['c1','c2'], default='c1', help='c1: original G correct; c2: original H correct')
    p.add_argument('--p0', type=float, default=2/3, help='Acceptance BEFORE warmup')
    p.add_argument('--q0', type=float, default=.3, help='Original H share BEFORE warmup, even with rule c2')
    p.add_argument('--mu-n', type=float, default=-.5, help='Rejected-group second-coordinate mean')
    p.add_argument('--accepted-mean', type=float, default=1.,
                   help='Shared first-coordinate mean of G and H; use 4 for the projection comparison')
    p.add_argument('--n-prompts', type=int, default=8, help='Equally weighted prompts')
    p.add_argument('--n-per-group', type=int, default=16, help='Even response count per group')
    p.add_argument('--dim', type=int, default=4, help='Feature dimension >=2')
    p.add_argument('--noise-std', type=float, default=.25, help='Gaussian feature standard deviation')
    p.add_argument('--warmup', type=float, default=10., help='Verifier-only flow before the common checkpoint')
    p.add_argument('--horizon', type=float, default=10., help='Control duration AFTER warmup')
    p.add_argument('--n-times', type=int, default=501, help='Recorded times, not solver steps')
    p.add_argument('--rtol', type=float, default=1e-10, help='DOP853 relative tolerance')
    p.add_argument('--atol', type=float, default=1e-12, help='DOP853 absolute tolerance')
    p.add_argument('--max-step', type=float, default=.25, help='Maximum internal solver step')
    p.add_argument('--selectivity-tol', type=float, default=1e-8, help='Rate tolerance for empirical selectivity')
    p.add_argument('--strength-tol', type=float, default=1e-14, help='Adaptive denominator failure threshold')
    p.add_argument('--width', type=float, default=4.2, help='PDF width in inches')
    p.add_argument('--height', type=float, default=3.2, help='PDF height in inches')
    a = p.parse_args()
    if not all(np.isfinite(v) for v in vars(a).values() if isinstance(v, (float,int))):
        p.error('All numerical arguments must be finite.')
    if not 0<a.p0<1 or not 0<a.q0<1:
        p.error('p0 and q0 must lie strictly between zero and one.')
    if a.n_prompts<1 or a.n_per_group<2 or a.n_per_group%2 or a.dim<2:
        p.error('Require n-prompts>=1, even n-per-group>=2, dim>=2.')
    if a.seed<0 or a.n_seeds<1 or a.n_times<3:
        p.error('Require seed>=0, n-seeds>=1, n-times>=3.')
    if min(a.noise_std,a.gain,a.gamma,a.warmup,a.selectivity_tol,a.strength_tol)<0:
        p.error('Noise, gains, warmup and diagnostic tolerances must be nonnegative.')
    if min(a.horizon,a.rtol,a.atol,a.max_step,a.width,a.height,a.adaptive_delta)<=0:
        p.error('Horizon, integration settings, dimensions and adaptive-delta must be positive.')
    a.seeds = list(range(a.seed, a.seed+a.n_seeds))
    return a


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ['fixed','adaptive'] if args.gain_mode=='both' else [args.gain_mode]
    methods = ['rlvr','gr'] + [f'{base}_{mode}' for mode in modes for base in ['raw','projected']]
    offsets = make_offsets(args.p0, args.q0, args.n_per_group)
    runs, summaries, comparisons = [], [], []
    for seed in args.seeds:
        folder = args.output / f'seed_{seed:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        phi = make_features(args.n_prompts,args.n_per_group,args.dim,args.noise_std,args.mu_n,seed)

        phi[:, :2, :, 0] += args.accepted_mean - 1.
        theta0 = make_checkpoint(phi, offsets, args)
        np.savez(folder/'bandit_inputs.npz', phi=phi, offsets=offsets, theta_checkpoint=theta0,
                 group_order=np.array(['G','H','N']), rule=args.rule)
        run, method_summaries = {}, {}
        for method in methods:
            data, summary = integrate(theta0, phi, offsets, method, args)
            run[method] = data
            save_table(folder/f'{method}.dat', data)
            summaries.append(dict(seed=seed, **summary))
            method_summaries[method] = summary
            print(f"seed={seed} {method}: p_H {data['p_H'][0]:.4f}->{data['p_H'][-1]:.4f}, "
                  f"p_G {data['p_G'][0]:.4f}->{data['p_G'][-1]:.4f}; "
                  f"selective duration={summary['selective_duration']:.4f}/{args.horizon:g}", flush=True)
        if 'fixed' in modes:

            contrast = []
            for left, right in method_summaries['raw_fixed']['collateral_intervals']:
                for start, end in method_summaries['projected_fixed']['selective_intervals']:
                    if max(left, start) < min(right, end):
                        contrast.append([max(left, start), min(right, end)])
            duration = sum(right-left for left, right in contrast)
            comparisons.append(dict(seed=seed, fixed_gain=args.gain,
                                    contrast_intervals=contrast, contrast_duration=duration,
                                    description='Raw decreases both masses; projected decreases hacks and preserves correctness.'))
            print(f'seed={seed}: raw/projected contrast duration={duration:.4f}/{args.horizon:g}', flush=True)
        runs.append(run)
    for mode in modes:
        export_comparison(runs, mode, args)
    (args.output/'run.json').write_text(json.dumps(
        dict(parameters={**vars(args),'output':str(args.output)},runs=summaries,
             fixed_gain_comparisons=comparisons),indent=2)+'\n')
    print(f'Saved standalone PDFs, TikZ tables/macros, inputs and diagnostics to {args.output.resolve()}')


if __name__ == '__main__':
    main()
