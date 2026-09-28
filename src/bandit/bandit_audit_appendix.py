#!/usr/bin/env python3
"""Collect audit control, coverage, projection-error, and trajectory-bound ablations. See README.md for usage."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
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


def population(theta, phi, offsets, audit):
    """G,H,N masses, gradients, reward Hessian, and fixed audited-hack gradient."""
    logits = np.einsum('xskd,d->xsk', phi, theta) + offsets[None, :, None]
    pi = softmax(logits.reshape(phi.shape[0], -1), axis=1).reshape(logits.shape)
    score = phi - np.einsum('xsk,xskd->xd', pi, phi)[:, None, None, :]
    mass = pi.sum(axis=2).mean(axis=0)
    grad = np.einsum('xsk,xskd->sd', pi, score)/phi.shape[0]
    reward = np.array([1., 1., 0.])[None, :, None]
    p_x = (pi*reward).sum(axis=(1, 2))
    B = np.einsum('xsk,xskd,xske->de', pi*(reward-p_x[:, None, None]), score, score)/phi.shape[0]

    weights = pi[:, 1, :]*audit[:, 1, :]
    p_HA = weights.sum()/phi.shape[0]
    h_A = np.einsum('xk,xkd->d', weights, score[:, 1, :, :])/phi.shape[0]
    return mass, grad, grad[0]+grad[1], h_A, p_HA, B, pi, score


def project(h, g):
    """Orthogonal to g; the paper sets P_perp=I when g=0."""
    return h-g*((g@h)/(g@g)) if g@g > 0 else h.copy()


def audit_mask(fraction, a):
    """Nested fixed sets, sampled within G and H; identical indices across seeds."""
    rng = np.random.default_rng(a.audit_seed)
    order = np.argsort(rng.random((a.n_prompts, 2, a.n_per_group)), axis=-1)
    mask = np.zeros((a.n_prompts, 3, a.n_per_group))
    count = round(fraction*a.n_per_group)
    for x in range(a.n_prompts):
        for group in (0, 1):
            mask[x, group, order[x, group, :count]] = 1.
    return mask


def state(theta, phi, offsets, audit, method, a):
    m, grad, g, h_A, p_HA, B, _, _ = population(theta, phi, offsets, audit)
    p_G, p_H, p_N = m
    if np.any(m <= 0):
        raise FloatingPointError('Group mass underflow; shorten the horizon.')
    h = grad[1]
    ph, ph_A = project(h, g), project(h_A, g)
    if method == 'rlvr':
        u = np.zeros_like(g)
    elif method == 'gr':
        u = -2*a.gamma*(B@g)
    elif method == 'raw':
        u = -a.gain*h_A
    else:
        u = -a.gain*ph_A
    v = g+u
    p, q = p_G+p_H, p_H/(p_G+p_H)
    p_dot, p_G_dot, p_H_dot = g@v, grad[0]@v, h@v
    drift = (h/p_H-grad[0]/p_G)@g
    row = dict(p=p, p_G=p_G, p_H=p_H, p_N=p_N, q=q,
               p_HA=p_HA, p_H_unseen=p_H-p_HA, coverage=p_HA/p_H,
               p_dot=p_dot, p_G_dot=p_G_dot, p_H_dot=p_H_dot,
               p_HA_dot=h_A@v, q_dot=(p_H_dot-q*p_dot)/p,
               verifier_z_dot=drift, projected_strength=ph@ph,
               audited_strength=ph_A@ph_A, alignment=ph@ph_A,
               kappa_local=(ph@ph)/p_H, partial_strength_ratio=(ph@ph_A)/p_H,
               delta_R=g@u, grad_norm_sq=g@g, correction_norm=np.linalg.norm(u),
               correctness_audit_alignment=grad[0]@h_A,
               F_GR=p-a.gamma*(g@g),
               selective=float(p_H_dot < -a.selectivity_tol and p_G_dot >= -a.selectivity_tol))
    for j in range(len(theta)):
        row[f'theta_{j}'], row[f'u_{j}'] = theta[j], u[j]
    return v, row


def sample_solution(sol, time, phi, offsets, audit, method, a):
    rows = [state(th, phi, offsets, audit, method, a)[1] for th in sol.sol(time).T]
    return {'t': np.asarray(time), **{k: np.array([r[k] for r in rows]) for k in rows[0]}}


def integrate(phi, offsets, audit, method, a):
    sol = solve_ivp(lambda t, th: state(th, phi, offsets, audit, method, a)[0],
                    (0, a.horizon), np.zeros(a.dim), method='DOP853', dense_output=True,
                    rtol=a.rtol, atol=a.atol, max_step=a.max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    time = np.linspace(0, a.horizon, round(a.horizon/a.record_step)+1)
    return sample_solution(sol, time, phi, offsets, audit, method, a), sol


def check_run(data, sol, phi, offsets, audit, method, a):
    """Parameter finite differences at three policies; check the actual rates."""
    error = 0.
    h = a.fd_step
    for t in (0., a.horizon/2, a.horizon):
        theta = sol.sol(t)
        m, grad, g, h_A, p_HA, B, _, _ = population(theta, phi, offsets, audit)
        v, _ = state(theta, phi, offsets, audit, method, a)
        for j in range(a.dim):
            step = np.eye(a.dim)[j]*h
            plus = population(theta+step, phi, offsets, audit)
            minus = population(theta-step, phi, offsets, audit)
            error = max(error, np.max(np.abs((plus[0]-minus[0])/(2*h)-grad[:, j])),
                        np.max(np.abs((plus[2]-minus[2])/(2*h)-B[:, j])),
                        abs((plus[4]-minus[4])/(2*h)-h_A[j]))

        scale = max(1., np.linalg.norm(v))
        plus = population(theta+h*v/scale, phi, offsets, audit)[0]
        minus = population(theta-h*v/scale, phi, offsets, audit)[0]
        error = max(error, np.max(np.abs(scale*(plus-minus)/(2*h)-grad@v)))
    identity = np.max(np.abs(data['p_dot']-data['p_G_dot']-data['p_H_dot']))
    if method == 'projected':
        identity = max(identity, np.max(np.abs(data['delta_R'])))
    if error > a.derivative_tol or identity > a.identity_tol:
        raise RuntimeError(f'Numerical check failed: derivative={error:g}, identity={identity:g}')
    return dict(derivative_error=float(error), identity_error=float(identity))


def mean_sd(samples, x='t'):
    out = {x: samples[0][x]}
    for key in samples[0]:
        if key == x:
            continue
        arr = np.stack([s[key] for s in samples])
        out[key+'_mean'] = arr.mean(axis=0)
        out[key+'_std'] = arr.std(axis=0, ddof=1) if len(samples) > 1 else np.zeros_like(arr[0])
    return out


def save_rows(path, rows):
    save_table(path, {k: [r[k] for r in rows] for k in rows[0]})


def band(ax, x, y, sd, color, label, path, style='-'):
    """Export the exact SD polygon; TikZ can use table[x=x,y=y] ... closedcycle."""
    ax.fill_between(x, y-sd, y+sd, color=color, alpha=.2, linewidth=0)
    line, = ax.plot(x, y, color=color, ls=style, lw=1.6, label=label)
    save_table(path, {'x': np.r_[x, x[::-1]], 'y': np.r_[y+sd, (y-sd)[::-1]]})
    return line


def finish(fig, ax, path, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(frameon=False, fontsize=9, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf', bbox_inches='tight')
    path.write_bytes(buffer.getvalue())
    plt.close(fig)


def inputs(seed, a, root):
    phi = make_features(a.n_prompts, a.n_per_group, a.dim, a.noise_std, a.mu_n, seed)
    phi[:, :2, :, 0] += a.accepted_mean-1.
    offsets = make_offsets(a.p0, a.q0, a.n_per_group)
    folder = root/f'seed_{seed:03d}'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'inputs.npz', phi=phi, offsets=offsets)
    return phi, offsets, folder


def control(a, root):
    styles = {'rlvr': ('black', '-', 'RLVR'), 'gr': ('tab:orange', ':', 'Gradient reg.'),
              'raw': ('tab:green', '--', 'Audit'), 'projected': ('tab:blue', '-', 'Projected audit')}
    samples = {method: [] for method in styles}
    checks, summaries = [], []
    audit = audit_mask(1., a)
    for seed in a.seeds:
        phi, offsets, folder = inputs(seed, a, root)
        for method in styles:
            data, sol = integrate(phi, offsets, audit, method, a)
            checks.append(dict(seed=seed, method=method, **check_run(data, sol, phi, offsets, audit, method, a)))
            save_table(folder/f'{method}.dat', data)
            samples[method].append(data)
            summaries.append(dict(seed=seed, method_index=list(styles).index(method),
                                  p_H_final=data['p_H'][-1], p_G_final=data['p_G'][-1],
                                  selective_fraction=data['selective'].mean(),
                                  min_p_G_dot=data['p_G_dot'].min(), max_p_G_dot=data['p_G_dot'].max()))
        print(f'control: seed {seed} complete', flush=True)
    fig, ax = plt.subplots(figsize=(a.width, a.height), layout='constrained')
    for method, (color, style, label) in styles.items():
        avg = mean_sd(samples[method])
        save_table(root/f'phase_{method}.dat', avg)
        band(ax, avg['p_H_mean'], avg['p_G_mean'], avg['p_G_std'], color, label,
             root/f'phase_{method}_band.dat', style)
        j = max(1, min(len(avg['t'])-2, round(.1*(len(avg['t'])-1))))
        ax.annotate('', xy=(avg['p_H_mean'][j+1], avg['p_G_mean'][j+1]),
                    xytext=(avg['p_H_mean'][j-1], avg['p_G_mean'][j-1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.plot(a.p0*a.q0, a.p0*(1-a.q0), 'ko', mfc='white', ms=4)
    finish(fig, ax, root/'control_phase.pdf', r'$p_H$', r'$p_G$')
    save_rows(root/'control_summary.dat', summaries)
    return dict(checks=checks, methods=list(styles), phase_band='Vertical SD in p_G at matched times; x is mean p_H.')


def coverage(a, root):
    fractions = sorted(a.coverage)
    samples, checks = [], []
    for seed in a.seeds:
        phi, offsets, folder = inputs(seed, a, root)
        rows = []
        for j, fraction in enumerate(fractions):
            audit = audit_mask(fraction, a)
            np.save(folder/f'audit_{j:02d}.npy', audit)
            data, sol = integrate(phi, offsets, audit, 'projected', a)
            checks.append(dict(seed=seed, coverage=fraction, **check_run(data, sol, phi, offsets, audit, 'projected', a)))
            save_table(folder/f'coverage_{j:02d}.dat', data)
            rows.append(dict(coverage=fraction, initial_coverage=data['coverage'][0],
                             p_H_final=data['p_H'][-1], p_G_final=data['p_G'][-1],
                             p_HA_final=data['p_HA'][-1], p_H_unseen_final=data['p_H_unseen'][-1],
                             final_coverage=data['coverage'][-1], delta_p_H=data['p_H'][-1]-data['p_H'][0],
                             min_alignment=data['alignment'].min(), min_partial_strength=data['partial_strength_ratio'].min(),
                             selective_fraction=data['selective'].mean()))
        save_rows(folder/'coverage_summary.dat', rows)
        samples.append({k: np.array([r[k] for r in rows]) for k in rows[0]})
        print(f'coverage: seed {seed} complete', flush=True)
    avg = mean_sd(samples, x='coverage')
    save_table(root/'coverage.dat', avg)
    fig, ax = plt.subplots(figsize=(a.width, a.height), layout='constrained')
    line = band(ax, avg['coverage'], avg['p_H_final_mean'], avg['p_H_final_std'],
                'tab:blue', r'Total $p_H(T)$', root/'coverage_band.dat')
    line.set_marker('o')
    ax.axhline(a.p0*a.q0, color='black', ls='--', lw=1., label=r'Initial $p_H(0)$')
    finish(fig, ax, root/'audit_coverage.pdf', r'Initial audited fraction $p_{H_A}(0)/p_H(0)$', r'$p_H(T)$')
    return dict(checks=checks, coverage_order=fractions, mask='Nested stratified random subsets, fixed through training and shared across feature seeds.')


def iss(a, root):
    samples, constants, checks = [], [], []
    audit = audit_mask(1., a)
    for seed in a.seeds:
        phi, offsets, folder = inputs(seed, a, root)
        coarse, sol = integrate(phi, offsets, audit, 'projected', a)
        fine_args = SimpleNamespace(**vars(a))
        fine_args.rtol /= 10
        fine_args.atol /= 10
        fine_args.max_step /= 2
        data, fine_sol = integrate(phi, offsets, audit, 'projected', fine_args)
        refinement = max(np.max(np.abs(data[k]-coarse[k])) for k in ('q', 'p_H', 'p_G'))
        if refinement > a.refinement_tol:
            raise RuntimeError(f'ISS integration refinement failed: {refinement:g}')
        coarse_grid = np.linspace(0, a.horizon, a.bound_points)
        fine_grid = np.linspace(0, a.horizon, 2*a.bound_points-1)
        c = sample_solution(sol, coarse_grid, phi, offsets, audit, 'projected', a)
        f = sample_solution(fine_sol, fine_grid, phi, offsets, audit, 'projected', a)
        D0, D1 = max(0., c['verifier_z_dot'].max()), max(0., f['verifier_z_dot'].max())
        k0, k1 = c['kappa_local'].min(), f['kappa_local'].min()

        D = (1+a.bound_slack)*max(D0, D1)
        kappa = (1-a.bound_slack)*min(k0, k1)
        if kappa <= 0:
            raise RuntimeError('No positive numerical kappa on this interval.')
        rate = a.gain*kappa
        envelope = lambda t: np.exp(-rate*t)*a.q0 + D/(4*rate)*(-np.expm1(-rate*t))
        excess = np.max(f['q']-envelope(fine_grid))
        inequality_error = np.max(f['q_dot']-(D/4-rate*f['q']))
        if max(excess, inequality_error) > a.identity_tol:
            raise RuntimeError('Numerical ISS envelope or differential inequality failed on the refined grid.')
        data['envelope'] = envelope(data['t'])
        data['envelope_gap'] = data['envelope']-data['q']
        f['envelope'] = envelope(fine_grid)
        save_table(folder/'iss_grid.dat', f)
        save_table(folder/'iss.dat', data)
        samples.append(data)
        checks.append(dict(seed=seed, refinement_error=float(refinement), max_q_minus_envelope=float(excess),
                           **check_run(data, fine_sol, phi, offsets, audit, 'projected', a)))
        constants.append(dict(seed=seed, D_coarse=D0, D_fine=D1, kappa_coarse=k0, kappa_fine=k1,
                              D=D, kappa=kappa, residual=D/(4*rate), q_final=data['q'][-1],
                              envelope_final=data['envelope'][-1], max_inequality_error=inequality_error))
        print(f'iss: seed {seed}, kappa={kappa:.4g}, D={D:.4g}', flush=True)
    avg = mean_sd(samples)
    save_table(root/'iss.dat', avg)
    save_rows(root/'iss_constants.dat', constants)
    fig, ax = plt.subplots(figsize=(a.width, a.height), layout='constrained')
    band(ax, avg['t'], avg['q_mean'], avg['q_std'], 'tab:blue', r'$q(t)$', root/'q_band.dat')
    band(ax, avg['t'], avg['envelope_mean'], avg['envelope_std'], 'black', 'ISS envelope', root/'envelope_band.dat', '--')
    finish(fig, ax, root/'iss_bound.pdf', 'Training time', 'Hacked share')
    return dict(checks=checks, interpretation='Each trajectory has its own numerically estimated envelope; the figure averages paired trajectories and envelopes. Sample SD bands need not satisfy an envelope ordering. Extrema and bounds are numerical, not certified.')


def projection(a, root):
    sizes = sorted(a.batch_sizes)
    samples, checks = [], []
    audit = audit_mask(1., a)
    for seed in a.seeds:
        phi, offsets, folder = inputs(seed, a, root)
        m, grad, g, h_A, _, _, pi, score = population(np.zeros(a.dim), phi, offsets, audit)
        reward = np.broadcast_to(np.array([1., 1., 0.])[None, :, None], pi.shape)
        reward_score = (reward[..., None]*score).reshape(-1, a.dim)
        probability = pi.ravel()/a.n_prompts
        probability /= probability.sum()
        rng = np.random.default_rng(np.random.SeedSequence([a.mc_seed, seed]))
        rows = []
        for n in sizes:

            counts = rng.multinomial(n, probability, size=a.repetitions)
            g_hat = counts@reward_score/n
            u = np.array([-a.gain*project(h_A, gh) for gh in g_hat])
            signed = u@g
            true_p_dot = g@g+signed
            h_dot = (g+u)@grad[1]
            good_dot = (g+u)@grad[0]
            error = np.abs(signed)
            bound = np.linalg.norm(g-g_hat, axis=1)*np.linalg.norm(u, axis=1)
            identity = np.max(np.abs(signed-np.einsum('ij,ij->i', g-g_hat, u)))
            if identity > a.identity_tol or np.max(error-bound) > a.identity_tol:
                raise RuntimeError('Projection-error identity failed.')
            batches = dict(batch=np.arange(a.repetitions), abs_error=error, signed_error=signed,
                           true_p_dot=true_p_dot, p_H_dot=h_dot, p_G_dot=good_dot, cauchy_bound=bound)
            save_table(folder/f'batches_{n}.dat', batches)
            rows.append(dict(batch_size=n, abs_error=error.mean(), batch_sd=error.std(ddof=1),
                             mcse=error.std(ddof=1)/np.sqrt(a.repetitions), signed_error=signed.mean(),
                             acceptance_decrease_fraction=np.mean(true_p_dot < -a.selectivity_tol),
                             selective_fraction=np.mean((h_dot < -a.selectivity_tol)&(good_dot >= -a.selectivity_tol)),
                             zero_gradient_fraction=np.mean(np.sum(g_hat*g_hat, axis=1)==0)))
            checks.append(dict(seed=seed, batch_size=n, projection_identity_error=float(identity)))
        save_rows(folder/'projection_summary.dat', rows)
        samples.append({k: np.array([r[k] for r in rows]) for k in rows[0]})
        print(f'projection: seed {seed}, {a.repetitions} independent batches per size', flush=True)
    avg = mean_sd(samples, x='batch_size')
    save_table(root/'projection_error.dat', avg)
    fig, ax = plt.subplots(figsize=(a.width, a.height), layout='constrained')
    line = band(ax, avg['batch_size'], avg['abs_error_mean'], avg['abs_error_std'],
                'tab:blue', 'Estimated projection', root/'projection_error_band.dat')
    line.set_marker('o')
    ax.axhline(0, color='black', ls='--', lw=1., label='Exact projection')
    ax.set_xscale('log')
    ax.set_xticks(sizes, labels=[str(n) for n in sizes])
    finish(fig, ax, root/'projection_error.pdf', 'Samples for reward-gradient estimate', r'$|\dot p-\|g_R\|^2|$')
    return dict(checks=checks, interpretation='Fixed initial policy; exact baseline flow and audited-hack gradient. Average absolute errors over batches within each seed, then mean/sample SD across feature seeds. Batch SD and MCSE are saved separately.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--experiment', required=True, choices=['control', 'coverage', 'projection', 'iss'])
    p.add_argument('--output', type=Path, default=Path('.'))
    p.add_argument('--n-seeds', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lambda', dest='gain', type=float, default=6.)
    p.add_argument('--gamma', type=float, default=16.)
    p.add_argument('--p0', type=float, default=2/3)
    p.add_argument('--q0', type=float, default=.5)
    p.add_argument('--accepted-mean', type=float, default=4.)
    p.add_argument('--mu-n', type=float, default=-.5)
    p.add_argument('--n-prompts', type=int, default=8)
    p.add_argument('--n-per-group', type=int, default=16)
    p.add_argument('--dim', type=int, default=4)
    p.add_argument('--noise-std', type=float, default=.25)
    p.add_argument('--horizon', type=float, help='Default: 10 for control, 1 for coverage/ISS; projection does not integrate')
    p.add_argument('--record-step', type=float, default=.01)
    p.add_argument('--rtol', type=float, default=1e-10)
    p.add_argument('--atol', type=float, default=1e-12)
    p.add_argument('--max-step', type=float, default=.25)
    p.add_argument('--coverage', type=float, nargs='+', default=[0., .125, .25, .5, .75, 1.])
    p.add_argument('--audit-seed', type=int, default=2026)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 128, 512, 2048])
    p.add_argument('--repetitions', type=int, default=1000)
    p.add_argument('--mc-seed', type=int, default=2027)
    p.add_argument('--bound-points', type=int, default=1001, help='ISS extrema grid; refinement uses 2N-1 points')
    p.add_argument('--bound-slack', type=float, default=.01, help='Numerical outward padding: D*(1+s), kappa*(1-s)')
    p.add_argument('--fd-step', type=float, default=1e-5)
    p.add_argument('--derivative-tol', type=float, default=1e-8)
    p.add_argument('--refinement-tol', type=float, default=1e-8)
    p.add_argument('--identity-tol', type=float, default=1e-10)
    p.add_argument('--selectivity-tol', type=float, default=1e-8)
    p.add_argument('--width', type=float, default=4.8)
    p.add_argument('--height', type=float, default=3.2)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.horizon is None:
        a.horizon = 10. if a.experiment == 'control' else 1.
    if a.smoke:
        a.n_seeds = 1
        a.repetitions = 32
    vals = [v for v in vars(a).values() if isinstance(v, (int, float))]+a.coverage+a.batch_sizes
    if not np.isfinite(vals).all(): p.error('Numeric values must be finite.')
    if not 0 < a.p0 < 1 or not 0 < a.q0 < 1: p.error('Require p0,q0 strictly between 0 and 1.')
    if min(a.seed, a.audit_seed, a.mc_seed, a.noise_std, a.gain, a.gamma, a.selectivity_tol) < 0: p.error('Seeds, noise, gains and selectivity tolerance must be nonnegative.')
    if a.experiment == 'iss' and a.gain == 0: p.error('ISS requires positive constant gain.')
    if min(a.n_seeds, a.n_prompts, *a.batch_sizes) < 1 or a.n_per_group < 2 or a.n_per_group % 2 or a.dim < 2: p.error('Invalid model or sample counts; require even group size and dim>=2.')
    if a.repetitions < 2 or a.bound_points < 3: p.error('Need at least 2 batches and 3 bound-grid points.')
    if min(a.horizon, a.record_step, a.rtol, a.atol, a.max_step, a.fd_step, a.derivative_tol, a.refinement_tol, a.identity_tol, a.width, a.height) <= 0: p.error('Times, solver tolerances and figure sizes must be positive.')
    if not 0 <= a.bound_slack < 1: p.error('Bound slack must lie in [0,1).')
    if any(not 0 <= f <= 1 for f in a.coverage): p.error('Coverage must lie in [0,1].')
    if a.experiment == 'coverage' and any(not np.isclose(f*a.n_per_group, round(f*a.n_per_group)) for f in a.coverage): p.error('Each coverage fraction times group size must be an integer.')
    if len(a.coverage) != len(set(a.coverage)) or len(a.batch_sizes) != len(set(a.batch_sizes)): p.error('Sweep values must be unique.')
    ratio = a.horizon/a.record_step
    if a.experiment != 'projection' and (ratio < 1 or not np.isclose(ratio, round(ratio))): p.error('Horizon must be a positive integer multiple of record-step.')
    a.seeds = list(range(a.seed, a.seed+a.n_seeds))
    return a


def main():
    a = parse_args()
    root = a.output/a.experiment
    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    result = globals()[a.experiment](a, root)
    report = dict(parameters={**vars(a), 'output': str(a.output)}, diagnostics=result)
    (root/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved one standalone PDF and TikZ data in {root.resolve()}')


if __name__ == '__main__':
    main()
