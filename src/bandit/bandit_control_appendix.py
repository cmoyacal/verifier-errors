#!/usr/bin/env python3
"""Collect verifier-only control and gradient-regularization ablations. See README.md for usage."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
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


def save_table(path, data):
    """Whitespace-separated, named columns: directly readable by PGFPlots."""
    np.savetxt(path, np.column_stack(list(data.values())), fmt="%.16e",
               header=" ".join(data), comments="")


def population(theta, phi, offsets):
    """Exact group probabilities/gradients, policy, and reward Hessian."""
    logits = np.einsum('xskd,d->xsk', phi, theta) + offsets[None, :, None]
    pi = softmax(logits.reshape(phi.shape[0], -1), axis=1).reshape(logits.shape)
    score = phi - np.einsum('xsk,xskd->xd', pi, phi)[:, None, None, :]
    p_S = pi.sum(axis=2).mean(axis=0)
    grad_p_S = np.einsum('xsk,xskd->sd', pi, score) / phi.shape[0]
    R = np.array([1., 1., 0.])[None, :, None]
    p_x = (pi * R).sum(axis=(1, 2))
    weights = pi * (R - p_x[:, None, None])
    B_R = np.einsum('xsk,xskd,xske->de', weights, score, score) / phi.shape[0]
    return p_S, grad_p_S, pi, B_R


def direction(theta, phi, offsets, gamma):
    """dot theta = g_R - 2 gamma B_R g_R, with no correctness feedback."""
    _, grad, _, B = population(theta, phi, offsets)
    g = grad[0] + grad[1]
    return g - 2 * gamma * (B @ g)


def integrate(phi, offsets, gamma, args, dt=None):
    """Classical RK4, including recomputation at all four intermediate states."""
    dt = args.dt if dt is None else dt
    n = round(args.horizon / dt)
    stride = round(args.record_step / dt)
    states = [np.zeros(phi.shape[-1])]
    theta = states[0].copy()
    for j in range(n):
        k1 = direction(theta, phi, offsets, gamma)
        k2 = direction(theta + dt*k1/2, phi, offsets, gamma)
        k3 = direction(theta + dt*k2/2, phi, offsets, gamma)
        k4 = direction(theta + dt*k3, phi, offsets, gamma)
        theta = theta + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        if not np.isfinite(theta).all():
            raise FloatingPointError('Non-finite RK4 state; reduce --dt.')
        if (j+1) % stride == 0:
            states.append(theta.copy())
    return np.linspace(0, args.horizon, len(states)), np.asarray(states)


def numerical_checks(record, data, phi, offsets, gamma, args):
    """Independent parameter finite differences and step-halving refinement."""
    h = args.fd_step
    errors = dict(group_gradient=0., reward_hessian=0., controlled_gradient=0.,
                  group_rates=0., F_rate=0.)
    for i in sorted(set([0, len(record['t'])//2, len(record['t'])-1])):
        theta = record['theta'][i]
        _, grad, _, B = population(theta, phi, offsets)
        v = direction(theta, phi, offsets, gamma)
        for j in range(phi.shape[-1]):
            e = np.eye(phi.shape[-1])[j] * h
            plus, minus = population(theta+e, phi, offsets), population(theta-e, phi, offsets)
            gp, gm = plus[1][:2].sum(axis=0), minus[1][:2].sum(axis=0)
            fp = plus[0][:2].sum()-gamma*(gp@gp)
            fm = minus[0][:2].sum()-gamma*(gm@gm)
            errors['group_gradient'] = max(errors['group_gradient'], float(np.max(np.abs((plus[0]-minus[0])/(2*h)-grad[:, j]))))
            errors['reward_hessian'] = max(errors['reward_hessian'], float(np.max(np.abs((gp-gm)/(2*h)-B[:, j]))))
            errors['controlled_gradient'] = max(errors['controlled_gradient'], float(abs((fp-fm)/(2*h)-v[j])))
        plus, minus = population(theta+h*v, phi, offsets), population(theta-h*v, phi, offsets)
        gp, gm = plus[1][:2].sum(axis=0), minus[1][:2].sum(axis=0)
        fp = plus[0][:2].sum()-gamma*(gp@gp)
        fm = minus[0][:2].sum()-gamma*(gm@gm)
        errors['group_rates'] = max(errors['group_rates'], float(np.max(np.abs((plus[0]-minus[0])/(2*h)-grad@v))))
        errors['F_rate'] = max(errors['F_rate'], float(abs((fp-fm)/(2*h)-v@v)))
    _, fine = integrate(phi, offsets, gamma, args, dt=args.dt/2)
    max_theta = float(np.max(np.abs(fine-record['theta'])))
    max_mass = max_F = 0.
    for i, th in enumerate(fine):
        mass, grad, _, _ = population(th, phi, offsets)
        g = grad[0]+grad[1]
        coarse_mass = [data['p_G_c1'][i], data['p_H_c1'][i], data['p_N'][i]]
        max_mass = max(max_mass, float(np.max(np.abs(mass-coarse_mass))))
        max_F = max(max_F, float(abs(mass[:2].sum()-gamma*(g@g)-data['F_GR'][i])))
    return dict(derivative_errors=errors,
                derivative_passed=max(errors.values()) <= args.derivative_tol,
                refinement_errors=dict(theta=max_theta, group_probabilities=max_mass, F_GR=max_F),
                refinement_passed=max(max_theta,max_mass,max_F) <= args.refinement_tol,
                min_F_increment=float(np.min(np.diff(data['F_GR']))))


def aggregate(samples, x='t'):
    out = {x:samples[0][x]}
    for key in samples[0]:
        if key == x:continue
        values = np.stack([d[key] for d in samples])
        out[key+'_mean'] = values.mean(axis=0)
        out[key+'_std'] = values.std(axis=0,ddof=1) if len(samples)>1 else np.zeros_like(values[0])
    return out


def evaluate(time, states, phi, offsets, gamma, eps):
    """Evaluate c1=1_G and c2=1_H on the SAME controlled trajectory."""
    reward = np.array([1., 1., 0.])
    rules = {'c1': np.array([1., 0., 0.]), 'c2': np.array([0., 1., 0.])}
    rows = []
    for theta in states:
        mass, grad, _, B = population(theta, phi, offsets)
        if np.any(mass <= 0):
            raise FloatingPointError('Group mass underflowed; shorten the horizon.')
        g = grad[0] + grad[1]
        k = -2 * gamma * (B @ g)
        v = g + k
        rates = grad @ v
        p = reward @ mass
        row = dict(p=p, p_N=mass[2], p_dot=g@v,
                   F_GR=p-gamma*(g@g), F_dot=v@v,
                   grad_norm_sq=g@g, correction_norm=np.linalg.norm(k))
        for name, c in rules.items():
            p_G, p_H = c@mass, (reward-c)@mass
            r_G, r_H = c@rates, (reward-c)@rates
            row.update({f'p_G_{name}': p_G, f'p_H_{name}': p_H,
                        f'q_{name}': p_H/p, f'r_G_{name}': r_G,
                        f'r_H_{name}': r_H,
                        f'selective_{name}': float(r_H < -eps and r_G >= -eps)})
        for j in range(len(theta)):
            row.update({f'theta_{j}': theta[j], f'g_R_{j}': g[j], f'k_{j}': k[j]})
        rows.append(row)
    return {'t': time, **{key: np.array([r[key] for r in rows]) for key in rows[0]}}


def summarize(data, seed, gamma, folder):
    """Classify each seed before averaging. Switch times are grid brackets."""
    row = dict(seed=seed, gamma=gamma)
    switches = []
    for world in ('c1', 'c2'):
        mask = data[f'selective_{world}']
        row[f'fraction_{world}'] = float(mask.mean())
        row[f'all_recorded_{world}'] = int(np.all(mask))
        for i in np.flatnonzero(np.diff(mask)):
            switches.append([int(world[-1]), data['t'][i], data['t'][i+1], mask[i], mask[i+1]])
    np.savetxt(folder/'switches.dat', np.asarray(switches).reshape(-1, 5),
               header='rule t_left t_right selective_left selective_right', comments='')
    return row


def plot_band(ax, data, x, key, label, color, style, folder, filename):
    """Save the exact mean +/- sample SD polygon used in the PDF for TikZ."""
    mean, sd = data[key+'_mean'], data[key+'_std']
    lower, upper = mean-sd, mean+sd
    ax.fill_between(data[x], lower, upper, color=color, alpha=.2, linewidth=0)
    ax.plot(data[x], mean, color=color, ls=style, lw=1.6, label=label,
            marker='o' if x == 'gamma' else None, ms=4)
    save_table(folder/filename, {'x': np.r_[data[x], data[x][::-1]],
                                'y': np.r_[upper, lower[::-1]]})


def save_figure(fig, path):

    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf', bbox_inches='tight')
    path.write_bytes(buffer.getvalue())
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--experiment', required=True, choices=['control', 'regularization'])
    p.add_argument('--output', type=Path, default=Path('.'), help='Parent folder; an experiment subfolder is added')
    p.add_argument('--n-seeds', type=int, default=10)
    p.add_argument('--seed', type=int, default=0, help='First consecutive feature seed')
    p.add_argument('--gamma', type=float, default=16., help='Penalty weight for the control experiment')
    p.add_argument('--gammas', type=float, nargs='+', default=[0., 1., 4., 16.])
    p.add_argument('--p0', type=float, default=2/3)
    p.add_argument('--q0', type=float, default=.5)
    p.add_argument('--mu-n', type=float, default=-.5)
    p.add_argument('--n-prompts', type=int, default=8)
    p.add_argument('--n-per-group', type=int, default=16)
    p.add_argument('--dim', type=int, default=4)
    p.add_argument('--noise-std', type=float, default=.25)
    p.add_argument('--horizon', type=float, default=1.)
    p.add_argument('--dt', type=float, default=.01)
    p.add_argument('--record-step', type=float, default=.01)
    p.add_argument('--fd-step', type=float, default=1e-5)
    p.add_argument('--derivative-tol', type=float, default=1e-8)
    p.add_argument('--refinement-tol', type=float, default=1e-8)
    p.add_argument('--identity-tol', type=float, default=1e-10)
    p.add_argument('--selectivity-tol', type=float, default=1e-8)
    p.add_argument('--width', type=float, default=4.8)
    p.add_argument('--height', type=float, default=3.2)
    p.add_argument('--smoke', action='store_true', help='Use one seed, keeping the full horizon and gamma sweep')
    a = p.parse_args()
    if a.smoke:
        a.n_seeds = 1
    numeric = [v for v in vars(a).values() if isinstance(v, (int, float))] + a.gammas
    if not np.isfinite(numeric).all():
        p.error('Numeric parameters must be finite.')
    if not 0 < a.p0 < 1 or not 0 < a.q0 < 1:
        p.error('Require 0 < p0,q0 < 1.')
    if min(a.seed, a.gamma, *a.gammas, a.noise_std, a.selectivity_tol) < 0:
        p.error('Seeds, gamma, noise and selectivity tolerance must be nonnegative.')
    if min(a.n_seeds, a.n_prompts) < 1 or a.dim < 2 or a.n_per_group < 2 or a.n_per_group % 2:
        p.error('Require positive counts, dim >= 2 and an even group size >= 2.')
    if min(a.dt, a.horizon, a.record_step, a.fd_step, a.derivative_tol,
           a.refinement_tol, a.identity_tol, a.width, a.height) <= 0:
        p.error('Times, numerical tolerances and figure dimensions must be positive.')
    if len(set(a.gammas)) != len(a.gammas):
        p.error('Gamma values must be unique.')
    for ratio in (a.horizon/a.dt, a.record_step/a.dt, a.horizon/a.record_step):
        if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
            p.error('Horizon and record-step must be multiples of dt; horizon must be divisible by record-step.')
    return a


def main():
    a = parse_args()
    root = a.output/a.experiment
    root.mkdir(parents=True, exist_ok=True)
    gammas = [a.gamma] if a.experiment == 'control' else sorted(a.gammas)
    seeds = list(range(a.seed, a.seed+a.n_seeds))
    offsets = make_offsets(a.p0, a.q0, a.n_per_group)
    summaries, checks, means = [], [], []
    for index, gamma in enumerate(gammas):
        folder = root if a.experiment == 'control' else root/f'gamma_{index:02d}'
        folder.mkdir(exist_ok=True)
        samples = []
        for seed in seeds:
            run_folder = folder/f'seed_{seed:03d}'
            run_folder.mkdir(exist_ok=True)
            phi = make_features(a.n_prompts, a.n_per_group, a.dim, a.noise_std, a.mu_n, seed)
            time, states = integrate(phi, offsets, gamma, a)
            data = evaluate(time, states, phi, offsets, gamma, a.selectivity_tol)
            if not all(np.isfinite(v).all() for v in data.values()):
                raise FloatingPointError('Non-finite evaluation metric.')
            swap_error = max(float(np.max(np.abs(data[x]-data[y]))) for x, y in
                             [('r_H_c1', 'r_G_c2'), ('r_G_c1', 'r_H_c2'),
                              ('p_H_c1', 'p_G_c2'), ('p_G_c1', 'p_H_c2')])
            if swap_error > a.identity_tol or np.any(data['selective_c1']*data['selective_c2']):
                raise RuntimeError('Correctness-rule exchange check failed.')
            check = numerical_checks({'t': time, 'theta': states}, data, phi, offsets, gamma, a)
            if not (check['derivative_passed'] and check['refinement_passed']):
                raise RuntimeError(f'Numerical check failed for seed={seed}, gamma={gamma}: {check}')
            checks.append(dict(seed=seed, gamma=gamma, swap_error=swap_error, **check))
            save_table(run_folder/'trajectory.dat', data)
            np.savez_compressed(run_folder/'inputs.npz', phi=phi, offsets=offsets)
            summaries.append(summarize(data, seed, gamma, run_folder))
            samples.append(data)
        mean = aggregate(samples)
        save_table(folder/'mean.dat', mean)
        means.append(mean)
        print(f'gamma={gamma:g}: {a.n_seeds} seed(s), derivative and refinement checks passed', flush=True)

    save_table(root/'selectivity_by_seed.dat', {k: [r[k] for r in summaries] for k in summaries[0]})
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(a.width, a.height), layout='constrained')
    if a.experiment == 'control':
        data = means[0]
        save_table(root/'control_rates.dat', {k: data[k] for k in
                   ('t', 'r_G_c1_mean', 'r_G_c1_std', 'r_H_c1_mean', 'r_H_c1_std',
                    'r_G_c2_mean', 'r_G_c2_std', 'r_H_c2_mean', 'r_H_c2_std')})
        plot_band(ax, data, 't', 'r_G_c1', r'$\dot p_{G,c_1}=\dot p_{H,c_2}$',
                  'tab:blue', '-', root, 'rate_G_band.dat')
        plot_band(ax, data, 't', 'r_H_c1', r'$\dot p_{H,c_1}=\dot p_{G,c_2}$',
                  'tab:orange', '--', root, 'rate_H_band.dat')
        ax.axhline(0, color='black', lw=.7)
        ax.set(xlabel='Training time', ylabel='Probability rate')
        filename = 'control_rates.pdf'
    else:
        data = {'gamma': np.array(gammas)}
        for rule in ('c1', 'c2'):
            values = np.array([[r[f'fraction_{rule}'] for r in summaries if r['gamma'] == g] for g in gammas])
            data[f'fraction_{rule}_mean'] = values.mean(axis=1)
            data[f'fraction_{rule}_std'] = values.std(axis=1, ddof=1) if a.n_seeds > 1 else np.zeros(len(gammas))
        save_table(root/'selectivity_fraction.dat', data)
        for rule, color, style in [('c1', 'tab:blue', '-'), ('c2', 'tab:orange', '--')]:
            plot_band(ax, data, 'gamma', f'fraction_{rule}', '$c_'+rule[-1]+'$',
                      color, style, root, f'fraction_{rule}_band.dat')
        ax.axhline(0, color='gray', lw=.5)
        ax.axhline(1, color='gray', lw=.5)
        ax.set(xlabel=r'Regularization weight $\gamma$', ylabel='Fraction of recorded times selective', xticks=gammas)
        filename = 'selectivity_fraction.pdf'
    ax.legend(frameon=False, fontsize=9, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2)
    save_figure(fig, root/filename)
    report = dict(parameters={**vars(a), 'output': str(a.output)}, gamma_order=gammas,
                  feature_seeds=seeds, checks=checks,
                  note='Both correctness rules evaluate one shared trajectory. Fractions classify each seed before averaging. Switches are grid brackets, not certified continuous-time intervals. SD is sample SD; one-seed SD is zero.')
    (root/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved {filename} and TikZ data in {root.resolve()}')


if __name__ == '__main__':
    main()
