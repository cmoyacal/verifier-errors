#!/usr/bin/env python3
"""Collect bandit growth, geometry, and discrete-step ablations. See README.md for usage."""
import argparse
import io
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
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


VARIABLES = ('z', 'p', 'q', 'p_G')
RATE_KEYS = ('z_dot_decomposition', 'p_dot', 'q_dot', 'p_G_dot')


def save_rows(path, rows):
    """Named, whitespace-delimited rows for PGFPlots; identifiers are numeric."""
    if rows:
        save_table(path, {key: np.asarray([row[key] for row in rows]) for key in rows[0]})


def aggregate(samples):
    """Sample SD across feature seeds at matched times, never standard error."""
    out = {'t': samples[0]['t']}
    for key in samples[0]:
        if key in ('t', 'p_G_declining'):
            continue
        values = np.stack([sample[key] for sample in samples])
        out[key + '_mean'] = values.mean(axis=0)
        out[key + '_std'] = (values.std(axis=0, ddof=1) if len(samples) > 1
                            else np.zeros_like(values[0]))
    return out


def figure(args):
    return plt.subplots(figsize=(args.width, args.height), layout='constrained')


def finish(fig, ax, path, xlabel, ylabel, *, log=False, legend=True):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
    if legend:
        ax.legend(frameon=False, fontsize=9, loc='lower center',
                  bbox_to_anchor=(.5, 1.01), ncol=2, borderaxespad=0)
    buffer=io.BytesIO()
    fig.savefig(buffer,format='pdf')
    temporary=path.with_suffix('.pdf.tmp')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)
    plt.close(fig)


def band(ax, x, mean, std, label, n_seeds):
    line, = ax.plot(x, mean, label=label, lw=1.6)
    if n_seeds > 1:
        ax.fill_between(x, mean-std, mean+std, color=line.get_color(), alpha=.18, linewidth=0)
    return line


def run_case(case_id, mu, p0, q0, args, root):
    """One fixed geometry/initialization, paired across all requested seeds."""
    folder = root / f'case_{case_id:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    samples, solutions, contexts, summaries = [], [], [], []
    for seed in args.seeds:
        phi = make_features(args.n_prompts, args.n_per_group, args.dim,
                            args.noise_std, mu, seed)
        offsets = make_offsets(p0, q0, args.n_per_group)
        data, events, sol = integrate(phi, offsets, args)
        data['bracket'] = data['leakage'] + data['bias']
        data['acceptance_factor'] = data['p']*(1-data['p'])
        save_table(folder / f'seed_{seed:03d}.dat', data)
        np.savez_compressed(folder / f'inputs_{seed:03d}.npz', phi=phi, offsets=offsets)
        np.savetxt(folder / f'stationary_{seed:03d}.dat', events, fmt='%.16e',
                   header='t p p_G', comments='')
        samples.append(data); solutions.append(sol); contexts.append((phi, offsets))
        summaries.append(dict(seed=seed, initial={k:float(data[k][0]) for k in ('p','q','p_G','p_H')},
                              final={k:float(data[k][-1]) for k in ('p','q','p_G','p_H')},
                              max_q_departure=float(np.max(np.abs(data['q']-q0))),
                              stationary_correctness_times=events[:,0].tolist(),
                              min_acceptance_increment=float(np.diff(data['p']).min()),
                              saved_growth_fraction=float(np.mean(data['q_dot']>args.rate_tol)),
                              saved_correctness_decline_fraction=float(np.mean(data['p_G_dot'] < -args.rate_tol))))
    mean = aggregate(samples)
    save_table(folder / 'mean.dat', mean)
    print(f"case={case_id:03d}: mu_N={mu:g}, p0={p0:g}, q0={q0:g}, seeds={len(samples)}", flush=True)
    return dict(id=case_id, mu=mu, p0=p0, q0=q0, folder=folder, samples=samples,
                solutions=solutions, contexts=contexts, mean=mean, summaries=summaries)


def initial_checks(cases, root):
    rows=[]
    for case in cases:
        for seed, d in zip(case['seeds'], case['samples']):
            expected = -2*(1+case['mu']) + 4*case['q0']
            rows.append(dict(case=case['id'], seed=seed, mu_n=case['mu'], p0=case['p0'],
                             q0=case['q0'], leakage=d['leakage'][0], bias=d['bias'][0],
                             bracket=d['bracket'][0], predicted_bracket=expected,
                             normalized_z_dot=d['z_dot'][0]/(case['p0']*(1-case['p0'])),
                             bracket_error=abs(d['bracket'][0]-expected),
                             scaling_error=abs(d['z_dot'][0]/(case['p0']*(1-case['p0']))-expected)))
    save_rows(root/'initial_checks.dat', rows)
    return rows


def validate_rates(cases, args, root):
    """Compare formula predictions with central time differences of probabilities."""
    rows, refinement = [], []
    for case in cases:
        for seed, sol, (phi, off) in zip(args.seeds, case['solutions'], case['contexts']):
            times=np.linspace(max(args.fd_steps), args.horizon-max(args.fd_steps), args.check_times)
            for t in times:
                predicted=metrics(sol.sol(t), phi, off)
                for h in args.fd_steps:
                    def values(at):
                        pg,ph,pn=population(sol.sol(at),phi,off)[0]
                        return np.array([np.log(ph)-np.log(pg),pg+ph,ph/(pg+ph),pg])
                    measured=(values(t+h)-values(t-h))/(2*h)
                    row=dict(case=case['id'],seed=seed,t=t,fd_step=h)
                    for key,rate,val in zip(VARIABLES,RATE_KEYS,measured):
                        row[key+'_predicted']=predicted[rate]
                        row[key+'_measured']=val
                        row[key+'_abs_error']=abs(predicted[rate]-val)
                    rows.append(row)
            fine=SimpleNamespace(**vars(args))
            fine.rtol=args.rtol/10; fine.atol=args.atol/10; fine.max_step=args.max_step/2
            refined,_,_=integrate(phi,off,fine)
            coarse=case['samples'][args.seeds.index(seed)]
            refinement.append(dict(case=case['id'],seed=seed,**{
                key+'_max_difference':np.max(np.abs(refined[key]-coarse[key])) for key in VARIABLES}))
    save_rows(root/'rate_checks.dat',rows)
    save_rows(root/'integration_refinement.dat',refinement)
    errors=[]
    for h in args.fd_steps:
        subset=[r for r in rows if r['fd_step']==h]
        errors.append(dict(fd_step=h,**{key+'_max_error':max(r[key+'_abs_error'] for r in subset)
                                        for key in VARIABLES},**{
            key+'_rmse':np.sqrt(np.mean([r[key+'_abs_error']**2 for r in subset])) for key in VARIABLES}))
    save_rows(root/'rate_errors.dat',errors)
    h=min(args.fd_steps,key=lambda value:abs(np.log10(value)+4))
    subset=[r for r in rows if r['fd_step']==h]
    marker_samples=[]
    for seed in args.seeds:
        selected=[r for r in subset if r['seed']==seed]
        marker_samples.append(dict(t=np.array([r['t'] for r in selected]),
                                   predicted=np.array([r['z_predicted'] for r in selected]),
                                   measured=np.array([r['z_measured'] for r in selected]),
                                   abs_error=np.array([r['z_abs_error'] for r in selected])))
    save_table(root/'z_dot_checks.dat',aggregate(marker_samples))
    return dict(max_integration_refinement=max(r[k] for r in refinement for k in r if k.endswith('_max_difference')),
                max_rate_error_by_increment=errors,plotted_fd_step=h)


def mechanisms(cases,args,root):
    """Two linked views of the SAME trajectories; validation is tabulated."""
    result=validate_rates(cases,args,root)
    d=cases[0]['mean']
    save_table(root/'mechanisms.dat',d)
    fig,ax=figure(args)
    for key,label,style in [('leakage','Leakage','-'),('bias','Hack bias','--'),('bracket','Sum','-.')]:
        line=band(ax,d['t'],d[key+'_mean'],d[key+'_std'],label,args.n_seeds)
        line.set_linestyle(style)
    ax.axhline(0,color='black',lw=.7)
    finish(fig,ax,root/'mechanisms.pdf','Training time','Growth bracket terms')
    fig,ax=figure(args)
    band(ax,d['t'],d['z_dot_decomposition_mean'],d['z_dot_decomposition_std'],
         'Predicted rate',args.n_seeds)
    markers=np.genfromtxt(root/'z_dot_checks.dat',names=True)
    ax.plot(markers['t'],markers['measured_mean'],ls='none',marker='o',ms=4,
            markerfacecolor='none',color='black',label='Time differences')
    ax.axhline(0,color='gray',lw=.7)
    finish(fig,ax,root/'z_dot.pdf','Training time',r'$\dot z$')
    result['max_seed_std']={key:float(np.max(d[key+'_std'])) for key in ('leakage','bias','bracket','z_dot_decomposition')}
    result['max_growth_identity_error']=float(max(np.max(np.abs(s['z_dot']-s['z_dot_decomposition'])) for s in cases[0]['samples']))
    return result


def geometry(cases,args,root):
    """Correctness versus acceptance, with seeds aligned at equal acceptance."""
    lower=max(float(s['p'][0]) for c in cases for s in c['samples'])
    upper=min(float(s['p'][-1]) for c in cases for s in c['samples'])
    if upper<=lower:raise ValueError('No common increasing acceptance range; increase the horizon.')
    acceptance=np.linspace(lower,upper,args.n_phase)
    fig,ax=figure(args)
    curves=[]
    for index,case in enumerate(cases):
        aligned=[]
        for seed,sample in zip(args.seeds,case['samples']):
            if np.any(np.diff(sample['p']) < -1e-12):
                raise ValueError('Acceptance decreased along the reference flow; refine the solve.')
            p,indices=np.unique(sample['p'],return_index=True)
            good=np.interp(acceptance,p,sample['p_G'][indices])
            save_table(case['folder']/f'phase_seed_{seed:03d}.dat',dict(p=acceptance,p_G=good))
            aligned.append(good)
        values=np.asarray(aligned)
        mean=values.mean(axis=0)
        std=values.std(axis=0,ddof=1) if args.n_seeds>1 else np.zeros_like(mean)
        filename=f'phase_{index:02d}.dat'
        save_table(root/filename,dict(p=acceptance,p_G_mean=mean,p_G_std=std,
                                     p_G_lower=mean-std,p_G_upper=mean+std))
        line=band(ax,acceptance,mean,std,rf"$\mu_{{N,2}}={case['mu']:g}$",args.n_seeds)
        line.set_linestyle(['-','--','-.',':'][index%4])
        j=round(.55*(len(acceptance)-1))
        ax.annotate('',xy=(acceptance[j+1],mean[j+1]),xytext=(acceptance[j-1],mean[j-1]),
                    arrowprops=dict(arrowstyle='->',color=line.get_color(),lw=1.5))
        curves.append(dict(file=filename,case=case['id'],mu_n=case['mu'],max_seed_std=float(std.max())))
    finish(fig,ax,root/'geometry_reward_correctness.pdf',r'$J_R=p$',r'$J_C=p_G$')
    return dict(common_acceptance_range=[lower,upper],phase_curves=curves,
                alignment='Linear interpolation at common acceptance values, without extrapolation.')


def steps(cases,args,root):
    """Exact Euler ascent: rates at each CURRENT Euler policy and flow at k*eta."""
    summaries=[]
    for case in cases:
        for ie,eta in enumerate(args.eta):
            samples=[]
            for seed,sol,(phi,off) in zip(args.seeds,case['solutions'],case['contexts']):
                n=int(round(args.horizon/eta))
                time=np.arange(n+1)*eta
                theta=np.zeros((n+1,args.dim));vals=[]
                for k in range(n+1):
                    vals.append(metrics(theta[k],phi,off))
                    if k<n: theta[k+1]=theta[k]+eta*population(theta[k],phi,off)[2]
                d={'t':time}
                d.update({key:np.asarray([r[key] for r in vals]) for key in vals[0]})
                reference=[metrics(th,phi,off) for th in sol.sol(time).T]
                rates={'t':time[:-1]}
                summary=dict(case=case['id'],seed=seed,eta=eta,
                             acceptance_decreases=np.sum(np.diff(d['p']) < -args.rate_tol),
                             max_q_departure=np.max(np.abs(d['q']-case['q0'])))
                for key,rate in zip(VARIABLES,RATE_KEYS):
                    measured=np.diff(d[key])/eta
                    predicted=d[rate][:-1]
                    rates[key+'_predicted']=predicted;rates[key+'_measured']=measured
                    rates[key+'_abs_error']=np.abs(measured-predicted)
                    ref=np.array([r[key] for r in reference])
                    d[key+'_flow']=ref
                    d[key+'_flow_abs_error']=np.abs(d[key]-ref)
                    summary[key+'_max_rate_error']=np.max(np.abs(measured-predicted))
                    summary[key+'_max_flow_error']=np.max(np.abs(d[key]-ref))

                    active=(np.abs(measured)>args.rate_tol)&(np.abs(predicted)>args.rate_tol)
                    summary[key+'_sign_disagreements']=np.sum(active&(np.sign(measured)!=np.sign(predicted)))
                folder=case['folder']/f'eta_{ie:02d}';folder.mkdir(exist_ok=True)
                save_table(folder/f'seed_{seed:03d}.dat',d)
                save_table(folder/f'rates_{seed:03d}.dat',rates)
                samples.append(d);summaries.append(summary)
            save_table(case['folder']/f'eta_{ie:02d}'/'mean.dat',aggregate(samples))
    save_rows(root/'step_errors.dat',summaries)

    errors=[]
    for eta in args.eta:
        per_seed=[]
        for seed in args.seeds:
            subset=[r for r in summaries if r['eta']==eta and r['seed']==seed]
            per_seed.append({key+'_'+kind:max(r[key+'_'+kind] for r in subset)
                             for key in VARIABLES for kind in ('max_rate_error','max_flow_error')})
        row={'eta':eta}
        for key in per_seed[0]:
            arr=np.array([r[key] for r in per_seed]);row[key+'_mean']=arr.mean()
            row[key+'_std']=arr.std(ddof=1) if len(arr)>1 else 0.
        errors.append(row)
    save_rows(root/'step_error_summary.dat',errors)
    fig,ax=figure(args)
    for key in VARIABLES:
        values=np.array([r[key+'_max_flow_error_mean'] for r in errors])
        std=np.array([r[key+'_max_flow_error_std'] for r in errors])
        line,=ax.plot(args.eta,np.where(values>0,values,np.nan),'o-',label='$'+key+'$')
        if args.n_seeds>1:
            ax.fill_between(args.eta,values-std,values+std,where=values-std>0,
                            color=line.get_color(),alpha=.18,linewidth=0)
    finish(fig,ax,root/'max_flow_error.pdf',r'Learning rate $\eta$',
           'Maximum deviation from flow',log=True)
    return dict(total_acceptance_decreases=int(sum(r['acceptance_decreases'] for r in summaries)))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--experiment',required=True,choices=['mechanisms','geometry','steps'])
    p.add_argument('--output',type=Path,default=Path('.'),help='Parent output folder; a selector subfolder is added')
    p.add_argument('--smoke',action='store_true',help='One seed, horizon 2, 41 recorded times; steps uses three learning rates')
    p.add_argument('--seed',type=int,default=0,help='First feature seed')
    p.add_argument('--n-seeds',type=int,default=10,help='Number of consecutive feature seeds')
    p.add_argument('--p0',type=float,nargs='+',help='Initial acceptance values; defaults depend on experiment')
    p.add_argument('--q0',type=float,nargs='+',help='Initial hacked shares; defaults depend on experiment')
    p.add_argument('--mu-n',type=float,nargs='+',help='Rejected second-coordinate means; defaults depend on experiment')
    p.add_argument('--eta',type=float,nargs='+',default=[.4,.2,.1,.05,.025],help='Exact ascent learning rates')
    p.add_argument('--n-prompts',type=int,default=8)
    p.add_argument('--n-per-group',type=int,default=16)
    p.add_argument('--dim',type=int,default=4)
    p.add_argument('--noise-std',type=float,default=.25)
    p.add_argument('--horizon',type=float,default=50.)
    p.add_argument('--n-times',type=int,default=501,help='Saved flow times, not solver steps')
    p.add_argument('--n-phase',type=int,default=201,help='Common acceptance values for the geometry plot')
    p.add_argument('--rtol',type=float,default=1e-10)
    p.add_argument('--atol',type=float,default=1e-12)
    p.add_argument('--max-step',type=float,default=.25)
    p.add_argument('--fd-steps',type=float,nargs='+',default=[1e-3,1e-4,1e-5],help='Central time-difference increments')
    p.add_argument('--check-times',type=int,default=31,help='Interior times for rate validation')
    p.add_argument('--rate-tol',type=float,default=1e-8,help='Threshold for empirical rate signs')
    p.add_argument('--width',type=float,default=4.2)
    p.add_argument('--height',type=float,default=3.2)
    a=p.parse_args()
    defaults={
        'mechanisms':([2/3],[.3],[-.5]),
        'geometry':([2/3],[.3],[-1.5,-.5,.5]),
        'steps':([2/3],[.5],[-.5,0.,.5])}
    for name,value in zip(('p0','q0','mu_n'),defaults[a.experiment]):
        if getattr(a,name) is None:setattr(a,name,value)
    if a.smoke:
        a.n_seeds=1;a.horizon=2.;a.n_times=41;a.check_times=9
        if a.experiment=='steps':a.eta=[.4,.2,.1]
    numerical=[]
    for v in vars(a).values():
        if isinstance(v,(int,float)):numerical.append(v)
        elif isinstance(v,list):numerical.extend(v)
    if not np.isfinite(numerical).all():p.error('All numerical arguments must be finite.')
    for name in ('p0','q0','mu_n','eta','fd_steps'):
        v=getattr(a,name)
        if len(v)!=len(set(v)):p.error(f'{name} values must be unique.')
    if any(not 0<x<1 for x in a.p0+a.q0):p.error('p0 and q0 must lie strictly between 0 and 1.')
    if a.n_prompts<1 or a.n_per_group<2 or a.n_per_group%2 or a.dim<2:p.error('Require positive prompts, even group size>=2 and dim>=2.')
    if a.seed<0 or a.n_seeds<1 or a.n_times<3 or a.check_times<2 or a.n_phase<5:p.error('Invalid seed/count setting.')
    if a.experiment in ('mechanisms','geometry') and (len(a.p0)!=1 or len(a.q0)!=1):
        p.error('Mechanisms and geometry require one p0 and one q0; only geometry sweeps mu-n.')
    if a.experiment=='mechanisms' and len(a.mu_n)!=1:p.error('Mechanisms requires one mu-n.')
    if a.noise_std<0 or a.rate_tol<0:p.error('Noise and rate tolerance must be nonnegative.')
    if min(a.horizon,a.rtol,a.atol,a.max_step,a.width,a.height,*a.eta,*a.fd_steps)<=0:p.error('Time, solver, plot dimensions and increments must be positive.')
    if 2*max(a.fd_steps)>=a.horizon:p.error('fd increments must be smaller than half the horizon.')
    if a.experiment=='steps' and any(not np.isclose(a.horizon/e,round(a.horizon/e),rtol=0,atol=1e-8) for e in a.eta):
        p.error('For steps, horizon must be an integer multiple of every eta.')
    a.seeds=list(range(a.seed,a.seed+a.n_seeds))
    return a


def main():
    args=parse_args()
    root=args.output/args.experiment;root.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    if args.smoke:print('SMOKE ONLY: one feature seed, horizon 2; SD columns are zero placeholders.',flush=True)
    cases=[]
    for case_id,(mu,p0,q0) in enumerate(product(args.mu_n,args.p0,args.q0)):
        case=run_case(case_id,mu,p0,q0,args,root);case['seeds']=args.seeds;cases.append(case)
    checks=initial_checks(cases,root)
    result=globals()[args.experiment](cases,args,root)
    manifest=dict(parameters={**vars(args),'output':str(args.output)},
                  case_map=[dict(id=c['id'],folder=c['folder'].name,mu_n=c['mu'],p0=c['p0'],q0=c['q0'],runs=c['summaries']) for c in cases],
                  diagnostics=result,max_initial_bracket_error=float(max(r['bracket_error'] for r in checks)),
                  max_initial_scaling_error=float(max(r['scaling_error'] for r in checks)),
                  note='Means and sample standard deviations across feature seeds. Trajectories use matched times; phase curves use matched acceptance; step errors use per-seed maxima. One-seed SD columns are zero placeholders.')
    (root/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(list(root.rglob("*.pdf")))} independent PDFs and numeric tables in {root.resolve()}')


if __name__=='__main__':main()
