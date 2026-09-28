#!/usr/bin/env python3
"""Collect neural-bandit growth, feedback, coverage, and projection ablations. See README.md for usage."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
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


class NeuralBandit:
    """Trainable W, hidden biases b, and output weights w; no output bias."""
    def __init__(self, phi, seed, width=16):
        self.phi = phi
        self.width = width
        self.dim = phi.shape[-1]

        rng = np.random.default_rng(np.random.SeedSequence([seed, 2026]))
        W = rng.normal(0, 1 / np.sqrt(self.dim), (width, self.dim))
        b = np.zeros(width)
        w = rng.normal(0, 1 / np.sqrt(width), width)
        self.theta_initial = np.r_[W.ravel(), b, w]
        self.f_initial = self.network(self.theta_initial)[0].copy()

    def unpack(self, theta):
        m, d = self.width, self.dim
        return theta[:m*d].reshape(m, d), theta[m*d:m*(d+1)], theta[m*(d+1):]

    def network(self, theta, vector=None):
        W, b, w = self.unpack(theta)
        t = np.tanh(np.einsum('...d,md->...m', self.phi, W) + b)
        slope = 1 - t*t
        a = slope * w
        J_W = (a[..., :, None] * self.phi[..., None, :]).reshape(*t.shape[:-1], -1)
        J = np.concatenate([J_W, a, t], axis=-1)
        f = t @ w
        if vector is None:
            return f, J
        dW, db, dw = self.unpack(vector)
        dz = np.einsum('...d,md->...m', self.phi, dW) + db
        dt = slope * dz
        da = slope * dw - 2 * w * t * dt
        dJ_W = (da[..., :, None] * self.phi[..., None, :]).reshape(*t.shape[:-1], -1)
        dJ = np.concatenate([dJ_W, da, dt], axis=-1)
        return f, J, dJ

    def distribution(self, f, offsets):
        logits = f - self.f_initial + offsets[None, :, None]
        return softmax(logits.reshape(logits.shape[0], -1), axis=1).reshape(logits.shape)


def population(theta, model, offsets):
    f, J = model.network(theta)
    pi = model.distribution(f, offsets)
    mean_J = np.einsum('xsk,xskd->xd', pi, J)
    score = J - mean_J[:, None, None, :]
    masses = pi.sum(axis=2).mean(axis=0)
    grads = np.einsum('xsk,xskd->sd', pi, score) / pi.shape[0]
    return masses, grads, grads[0] + grads[1]


def save_table(path, data):
    """Whitespace-separated, named columns: directly readable by PGFPlots."""
    np.savetxt(path, np.column_stack(list(data.values())), fmt="%.16e",
               header=" ".join(data), comments="")

def reward_hessian_vector(theta, model, offsets, vector):
    """Exact neural B_R v, including changes in the network Jacobian."""
    f, J, dJ = model.network(theta, vector)
    pi = model.distribution(f, offsets)
    R = np.array([1., 1., 0.])[None, :, None]
    p_x = (pi * R).sum(axis=(1, 2))
    df = np.einsum('xskd,d->xsk', J, vector)
    dpi = pi * (df - (pi * df).sum(axis=(1, 2))[:, None, None])
    dp_x = (dpi * R).sum(axis=(1, 2))
    weights = pi * (R - p_x[:, None, None])
    dweights = dpi * (R - p_x[:, None, None]) - pi * dp_x[:, None, None]
    return (np.einsum('xsk,xskd->d', dweights, J) +
            np.einsum('xsk,xskd->d', weights, dJ)) / pi.shape[0]


def project(h, g):
    return h-g*((g@h)/(g@g)) if g@g > 0 else h.copy()


def policy_arrays(theta, model, offsets):
    f, J = model.network(theta)
    pi = model.distribution(f, offsets)
    score = J-np.einsum('xsk,xskd->xd', pi, J)[:, None, None, :]
    return pi, score


def audited_population(theta, model, offsets, audit):
    pi, score = policy_arrays(theta, model, offsets)
    masses = pi.sum(axis=2).mean(axis=0)
    grads = np.einsum('xsk,xskd->sd', pi, score)/pi.shape[0]
    weights = pi[:, 1, :]*audit[:, 1, :]
    p_HA = weights.sum()/pi.shape[0]
    h_A = np.einsum('xk,xkd->d', weights, score[:, 1])/pi.shape[0]
    return masses, grads, grads[0]+grads[1], p_HA, h_A


def field(theta, model, offsets, args, audit=None):
    if audit is None:
        masses, grads, g = population(theta, model, offsets)
        h_A, p_HA = grads[1], masses[1]
        u = (-2*args.gamma*reward_hessian_vector(theta, model, offsets, g)
             if args.gamma else np.zeros_like(g))
    else:
        masses, grads, g, p_HA, h_A = audited_population(theta, model, offsets, audit)
        u = -args.gain*project(h_A, g)
    return g+u, (masses, grads, g, p_HA, h_A, u)


def metrics(theta, model, offsets, args, audit=None):
    v, (m, grad, g, p_HA, h_A, u) = field(theta, model, offsets, args, audit)
    if np.any(m<=0):
        raise FloatingPointError('Group mass underflow; shorten the horizon.')
    p_G, p_H, p_N = m
    p=p_G+p_H; q=p_H/p; d=grad[1]/p_H-grad[0]/p_G
    leakage=d@(grad[0]/p_G-grad[2]/p_N); bias=q*(d@d)
    p_dot=g@v; G_dot=grad[0]@v; H_dot=grad[1]@v
    ph, ph_A = project(grad[1],g), project(h_A,g)
    return dict(p=p,q=q,p_G=p_G,p_H=p_H,p_N=p_N,z=np.log(p_H/p_G),
                p_dot=p_dot,p_G_dot=G_dot,p_H_dot=H_dot,
                q_dot=(H_dot-q*p_dot)/p,z_dot=d@v,
                verifier_z_dot=d@g,z_dot_decomposition=p*(1-p)*(leakage+bias),
                leakage=leakage,bias=bias,grad_norm_sq=g@g,
                delta_R=g@u,correction_norm=np.linalg.norm(u),
                correctness_decline=float(G_dot < -args.selectivity_tol),
                simultaneous_growth=float(p_dot>args.selectivity_tol and (H_dot-q*p_dot)/p>args.selectivity_tol),
                selective=float(H_dot < -args.selectivity_tol and G_dot>=-args.selectivity_tol),
                p_HA=p_HA,p_H_unseen=p_H-p_HA,coverage=p_HA/p_H,
                p_HA_dot=h_A@v,alignment=ph@ph_A,
                partial_strength_ratio=(ph@ph_A)/p_H,
                projected_strength=ph@ph,F_GR=p-args.gamma*(g@g))


def sample_solution(sol, time, model, offsets, args, audit=None):
    rows=[metrics(th, model, offsets, args, audit) for th in sol.sol(time).T]
    return dict(t=np.asarray(time),**{k:np.array([r[k] for r in rows]) for k in rows[0]})


def integrate(model, offsets, args, audit=None):
    def good_boundary(t, th):
        return metrics(th,model,offsets,args,audit)['p_G_dot']
    sol=solve_ivp(lambda t,th:field(th,model,offsets,args,audit)[0],
                  (0,args.horizon),model.theta_initial.copy(),method='DOP853',
                  dense_output=True,rtol=args.rtol,atol=args.atol,
                  max_step=args.max_step,events=good_boundary)
    if not sol.success:
        raise RuntimeError(sol.message)
    time=np.linspace(0,args.horizon,args.n_times)
    data=sample_solution(sol,time,model,offsets,args,audit)
    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError('Nonfinite trajectory.')
    return data,sol


def numeric_checks(data, sol, model, offsets, args, audit=None):
    """Independent parameter and integrated-time finite differences; one refinement."""
    rng=np.random.default_rng(818); errors=dict(gradient=0.,hessian_vector=0.,
        regularized_gradient=0.,audit_gradient=0.,flow_rate=0.,time_rate=0.,
        half_time_rate=0.,identity=0.,refinement=0.)
    full=np.ones(model.phi.shape[:-1]) if audit is None else audit
    for t in (0.,args.horizon/2,args.horizon):
        th=sol.sol(t); m,gr,g,pA,hA=audited_population(th,model,offsets,full)
        v=field(th,model,offsets,args,audit)[0]
        w=rng.normal(size=len(th)); w/=np.linalg.norm(w); eps=args.fd_step
        plus=audited_population(th+eps*w,model,offsets,full)
        minus=audited_population(th-eps*w,model,offsets,full)
        errors['gradient']=max(errors['gradient'],np.max(np.abs((plus[0]-minus[0])/(2*eps)-gr@w)))
        errors['audit_gradient']=max(errors['audit_gradient'],abs((plus[3]-minus[3])/(2*eps)-hA@w))
        hv=reward_hessian_vector(th,model,offsets,w)
        errors['hessian_vector']=max(errors['hessian_vector'],np.max(np.abs((plus[2]-minus[2])/(2*eps)-hv)))
        if audit is None and args.gamma:
            fp=plus[0][:2].sum()-args.gamma*(plus[2]@plus[2])
            fm=minus[0][:2].sum()-args.gamma*(minus[2]@minus[2])
            errors['regularized_gradient']=max(errors['regularized_gradient'],abs((fp-fm)/(2*eps)-v@w))
        scale=max(1.,np.linalg.norm(v))
        mp=population(th+eps*v/scale,model,offsets)[0]
        mm=population(th-eps*v/scale,model,offsets)[0]
        errors['flow_rate']=max(errors['flow_rate'],np.max(np.abs((mp-mm)/(2*eps)*scale-gr@v)))
    for t in np.linspace(.1,.9,7)*args.horizon:
        row=metrics(sol.sol(t),model,offsets,args,audit)
        for factor,name in [(1.,'time_rate'),(.5,'half_time_rate')]:
            dt=min(args.time_fd_step,args.horizon/100)*factor
            plus=metrics(sol.sol(t+dt),model,offsets,args,audit)
            minus=metrics(sol.sol(t-dt),model,offsets,args,audit)
            for key in ['p','q','z','p_G','p_H']:
                measured=(plus[key]-minus[key])/(2*dt)
                errors[name]=max(errors[name],abs(measured-row[key+'_dot']))
    errors['identity']=max(np.max(np.abs(data['p_dot']-data['p_G_dot']-data['p_H_dot'])),
        np.max(np.abs(data['verifier_z_dot']-data['z_dot_decomposition'])))
    if audit is not None:
        errors['identity']=max(errors['identity'],np.max(np.abs(data['delta_R'])),
            np.max(np.abs(data['p_H_dot']-(data['q']*data['grad_norm_sq']+
                       data['p']*data['q']*(1-data['q'])*data['verifier_z_dot']-args.gain*data['alignment']))))
    elif args.gamma==0:
        errors['identity']=max(errors['identity'],np.max(np.abs(data['p_dot']-data['grad_norm_sq'])))
    fine=copy.copy(args);fine.rtol/=10;fine.atol/=10;fine.max_step/=2
    refined,_=integrate(model,offsets,fine,audit)
    errors['refinement']=max(np.max(np.abs(data[k]-refined[k])) for k in ['p','q','p_G','p_H'])
    tolerances={k:args.derivative_tol for k in errors}
    tolerances.update(time_rate=args.time_derivative_tol,half_time_rate=args.time_derivative_tol,
                      identity=args.identity_tol,refinement=args.refinement_tol)
    if any(errors[k]>tolerances[k] for k in errors):
        raise RuntimeError(f'Numerical checks failed: {errors}')
    return {k:float(v) for k,v in errors.items()}


def save_rows(path,rows):
    save_table(path,{k:np.array([r[k] for r in rows]) for k in rows[0]})


def aggregate(samples,x='t'):
    for d in samples[1:]:
        np.testing.assert_array_equal(d[x],samples[0][x])
    out={x:samples[0][x]}
    for key in samples[0]:
        if key==x:continue
        values=np.stack([d[key] for d in samples])
        out[key+'_mean']=values.mean(axis=0)
        out[key+'_std']=values.std(axis=0,ddof=1) if len(samples)>1 else np.zeros_like(values[0])
    return out


def setup(seed,args,root,q0=None):
    phi=make_features(args.n_prompts,args.n_per_group,4,args.noise_std,args.mu_n,seed)
    phi[:,:2,:,0]+=args.accepted_mean-1.
    model=NeuralBandit(phi,seed,args.hidden_width)
    offsets=make_offsets(args.p0,args.q0 if q0 is None else q0,args.n_per_group)
    folder=root/f'seed_{seed:03d}';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'inputs.npz',phi=phi,offsets=offsets,
        theta_initial=model.theta_initial,f_initial=model.f_initial)
    return model,offsets,folder


def curve(data,x,y,label,color='blue',style='solid',sd=None,arrows=False,**kwargs):
    return dict(x=np.asarray(data[x]),y=np.asarray(data[y]),
        sd=np.asarray(data[sd]) if sd else np.zeros_like(data[y]),
        label=label,color=color,style=style,arrows=arrows,**kwargs)


def figure(root,name,curves,args,xlabel,ylabel,xlog=False,ylim=None):
    """One untitled axis. Save independent PGFPlots data and uncompiled source."""
    mplcolor={'blue':'tab:blue','orange':'tab:orange','green!60!black':'tab:green',
              'black':'black','gray':'gray'}
    fig,ax=plt.subplots(figsize=(args.width,args.height),layout='constrained')
    bands=[];lines=[]
    for j,c in enumerate(curves):
        x,y,sd=c['x'],c['y'],c['sd'];col=c['color'];style=c['style']
        filename=f'{name}_{j:02d}.dat'
        save_table(root/filename,dict(x=x,y=y,std=sd))
        if np.any(sd>0):
            bandname=f'{name}_{j:02d}_band.dat'
            bx,by=np.r_[x,x[::-1]],np.r_[y+sd,(y-sd)[::-1]]
            save_table(root/bandname,dict(x=bx,y=by))
            ax.fill(bx,by,color=mplcolor[col],alpha=.18,lw=0,zorder=1)
            bands += [rf'\addplot[draw=none,fill={col},fill opacity=0.18,forget plot] table[x=x,y=y] {{\NeuralDataDir/{bandname}}} \closedcycle;']
        if c.get('marks'):
            ax.plot(x,y,linestyle='none',marker='s',markerfacecolor='none',color=mplcolor[col],label=c['label'],ms=4,zorder=5)
            option=col+',only marks,mark=square,mark options={fill=none}'
        else:
            ax.plot(x,y,color=mplcolor[col],ls=style,lw=args.line_width,label=c['label'],zorder=3)
            option=f'{col},{style},line width={args.line_width}pt'
        if c['arrows']:
            length=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
            if length[-1]>0:
                i,jj=np.searchsorted(length,[.45*length[-1],.55*length[-1]])
                jj=min(max(i+1,jj),len(x)-1)
                ax.annotate('',(x[jj],y[jj]),(x[i],y[i]),arrowprops=dict(arrowstyle='->',color=mplcolor[col],lw=args.line_width))
            option+=r',postaction={decorate},decoration={markings,mark=at position 0.55 with {\arrow{stealth}}}'
        lines += [rf'\addplot[{option}] table[x=x,y=y] {{\NeuralDataDir/{filename}}};',
                  r'\addlegendentry{'+c['label']+'}']
        if 'declining' in c:
            yd=np.where(c['declining'],y,np.nan)
            decname=f'{name}_{j:02d}_declining.dat';save_table(root/decname,dict(x=x,y=yd))
            ax.plot(x,yd,color=mplcolor[col],lw=args.line_width*1.8,zorder=4)
            lines += [rf'\addplot[{col},line width={args.line_width*1.8}pt,forget plot,unbounded coords=jump] table[x=x,y=y] {{\NeuralDataDir/{decname}}};']
    ax.set(xlabel=xlabel,ylabel=ylabel)
    if xlog:ax.set_xscale('log')
    if ylim is not None:ax.set_ylim(*ylim)
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='best')
    fig.savefig(root/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(root/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    options=[f'width={args.width}in',f'height={args.height}in','font=\\small',
        'xlabel={'+xlabel+'}','ylabel={'+ylabel+'}',
        r'legend style={font=\small,draw=none,fill=none}',
        'grid=major','grid style={gray!15}']
    if xlog:options+=['xmode=log']
    if ylim is not None:options += [f'ymin={ylim[0]}',f'ymax={ylim[1]}']
    tex=[r'\providecommand{\NeuralDataDir}{.}',r'\begin{tikzpicture}',
        r'\begin{axis}['+','.join(options)+']']+bands+lines+[r'\end{axis}',r'\end{tikzpicture}']
    (root/f'{name}.tex').write_text('\n'.join(tex)+'\n')
    return name+'.pdf'


def growth(args,root):
    runs={q:[] for q in args.q_values};checks=[];summary=[]
    for seed in args.seeds:
        for i,q0 in enumerate(args.q_values):
            model,offsets,folder=setup(seed,args,root,q0)
            np.save(folder/f'offsets_{i:02d}.npy',offsets)
            d,sol=integrate(model,offsets,args)
            save_table(folder/f'trajectory_{i:02d}.dat',d)
            np.savez_compressed(folder/f'parameters_{i:02d}.npz',t=d['t'],theta=sol.sol(d['t']).T)
            stationary=sol.t_events[0]
            sd=sample_solution(sol,stationary,model,offsets,args) if len(stationary) else None
            save_table(folder/f'stationary_{i:02d}.dat',dict(t=stationary,p=sd['p'] if sd else [],p_G=sd['p_G'] if sd else []))
            if args.checks:
                checks.append(dict(seed=seed,q0=q0,**numeric_checks(d,sol,model,offsets,args)))
            runs[q0].append(d)
            summary.append(dict(seed=seed,q0=q0,p_final=d['p'][-1],q_final=d['q'][-1],
                p_G_final=d['p_G'][-1],decline_fraction=d['correctness_decline'].mean(),
                growth_fraction=d['simultaneous_growth'].mean()))
            print(f'growth seed={seed}, q0={q0:g}: q {q0:.3f}->{d["q"][-1]:.3f}, p_G {d["p_G"][0]:.3f}->{d["p_G"][-1]:.3f}',flush=True)
    avgs={q:aggregate(v) for q,v in runs.items()}
    for i,q in enumerate(args.q_values):save_table(root/f'trajectory_{i:02d}_mean.dat',avgs[q])
    a=avgs[args.q0];save_table(root/'growth_A.dat',a)
    plots=[figure(root,'growth_A',[
        curve(a,'t','p_mean',r'$p$','blue',sd='p_std'),
        curve(a,'t','q_mean',r'$q$','orange','dashed',sd='q_std'),
        curve(a,'t','p_G_mean',r'$p_G$','black','dotted',sd='p_G_std')],args,'Training time', 'Probability / hacked share',ylim=(0,1))]
    colors=['gray','orange','black','blue','green!60!black']
    plots.append(figure(root,'growth_B',[
        curve(avgs[q],'p_mean','p_G_mean',rf'$q(0)={q:g}$',colors[i%len(colors)],
            sd='p_G_std',arrows=True,declining=avgs[q]['p_G_dot_mean'] < -args.selectivity_tol)
        for i,q in enumerate(args.q_values)],args,r'$J_R=p$',r'$J_C=p_G$',ylim=(0,1)))
    save_rows(root/'summary.dat',summary)
    return dict(plots=plots,checks=checks,trajectory_q0=args.q_values,
                thick_segments='Negative mean correctness rate, not a guarantee for every seed.')


def feedback(args,root):
    samples=[];checks=[]
    for seed in args.seeds:
        model,offsets,folder=setup(seed,args,root)
        d,sol=integrate(model,offsets,args)

        theta=sol.sol(d['t']).T
        pis=[];gs=[]
        for th in theta:
            pi,_=policy_arrays(th,model,offsets);pis.append(pi)
            gs.append(population(th,model,offsets)[2])
        np.savez_compressed(folder/'verifier_record.npz',t=d['t'],theta=theta,pi=np.array(pis),g_R=np.array(gs))
        digest=hashlib.sha256(theta.tobytes()+np.asarray(pis).tobytes()+np.asarray(gs).tobytes()).hexdigest()
        d.update(p_G_cR=d['p'].copy(),p_G_c1=d['p_G'].copy(),p_G_c2=d['p_H'].copy(),
                 p_H_cR=np.zeros_like(d['p']),p_H_c1=d['p_H'].copy(),p_H_c2=d['p_G'].copy(),
                 r_H_c1=d['p_H_dot'].copy(),r_G_c1=d['p_G_dot'].copy(),
                 r_H_c2=d['p_G_dot'].copy(),r_G_c2=d['p_H_dot'].copy())
        np.savez_compressed(folder/'evaluation_labels.npz',cR=np.array([1,1,0]),c1=np.array([1,0,0]),c2=np.array([0,1,0]))
        identity=max(np.max(np.abs(d['p_G_cR']-d['p'])),
                     np.max(np.abs(d['p_H_c1']-d['p_G_c2'])),
                     np.max(np.abs(d['r_H_c1']-d['r_G_c2'])))
        check=dict(seed=seed,shared_record_sha256=digest,world_identity_error=float(identity),
                   construction='One verifier record; three evaluator assignments. Labels do not enter the vector field.')
        if args.checks:check.update(numeric_checks(d,sol,model,offsets,args))
        checks.append(check);save_table(folder/'trajectory.dat',d);samples.append(d)
        print(f'feedback seed={seed}: one shared record, three assignments',flush=True)
    a=aggregate(samples);save_table(root/'information_C.dat',a)
    idx=np.unique(np.linspace(0,len(a['t'])-1,min(11,len(a['t']))).astype(int))
    markers={k:v[idx] for k,v in a.items()};save_table(root/'information_C_markers.dat',markers)
    plots=[figure(root,'information_C',[
        curve(a,'t','p_mean',r'$p$','blue',sd='p_std'),
        curve(markers,'t','p_G_cR_mean',r'$p_G\ (c=R)$','black',marks=True),
        curve(a,'t','p_G_c1_mean',r'$p_G\ (c_1)$','orange','dashed',sd='p_G_c1_std'),
        curve(a,'t','p_G_c2_mean',r'$p_G\ (c_2)$','green!60!black','dotted',sd='p_G_c2_std')
        ],args,'Training time',r'Acceptance / correctness',ylim=(0,1))]
    return dict(plots=plots,checks=checks,worlds={'cR':[1,1,0],'c1':[1,0,0],'c2':[0,1,0]},
        interpretation='Paired compatible assignments illustrate the information limit; no monitor benchmark.')


def audit_mask(fraction,args):
    rng=np.random.default_rng(args.audit_seed)
    order=np.argsort(rng.random((args.n_prompts,2,args.n_per_group)),axis=-1)
    mask=np.zeros((args.n_prompts,3,args.n_per_group))
    count=round(fraction*args.n_per_group)
    for x in range(args.n_prompts):
        for group in (0,1):mask[x,group,order[x,group,:count]]=1.
    return mask


def coverage(args,root):
    samples=[];checks=[];trajectories={f:[] for f in args.coverage}
    masks=[audit_mask(f,args) for f in args.coverage]
    for left,right in zip(masks[:-1],masks[1:]):
        if np.any(left>right):raise RuntimeError('Audit sets must be nested.')
    for seed in args.seeds:
        model,offsets,folder=setup(seed,args,root);rows=[]
        for i,(fraction,audit) in enumerate(zip(args.coverage,masks)):
            np.save(folder/f'audit_{i:02d}.npy',audit)
            d,sol=integrate(model,offsets,args,audit)
            save_table(folder/f'coverage_{i:02d}.dat',d)
            np.savez_compressed(folder/f'parameters_{i:02d}.npz',t=d['t'],theta=sol.sol(d['t']).T)
            if args.checks:checks.append(dict(seed=seed,coverage=fraction,**numeric_checks(d,sol,model,offsets,args,audit)))
            rows.append(dict(coverage=fraction,initial_coverage=d['coverage'][0],
                p_H_final=d['p_H'][-1],p_G_final=d['p_G'][-1],p_HA_final=d['p_HA'][-1],
                p_H_unseen_final=d['p_H_unseen'][-1],final_coverage=d['coverage'][-1],
                min_alignment=d['alignment'].min(),min_partial_strength=d['partial_strength_ratio'].min(),
                selective_fraction=d['selective'].mean(),max_abs_delta_R=np.abs(d['delta_R']).max()))
            trajectories[fraction].append(d)
        save_rows(folder/'summary.dat',rows)
        samples.append({k:np.array([r[k] for r in rows]) for k in rows[0]})
        print(f'coverage seed={seed}: {len(masks)} nested fixed masks',flush=True)
    a=aggregate(samples,'coverage');save_table(root/'coverage.dat',a)
    for i,f in enumerate(args.coverage):save_table(root/f'coverage_{i:02d}_mean.dat',aggregate(trajectories[f]))
    initial={'x':a['coverage'],'y':np.full_like(a['coverage'],args.p0*args.q0)}
    plots=[figure(root,'audit_coverage',[
        curve(a,'coverage','p_H_final_mean',r'Total $p_H(T)$','blue',sd='p_H_final_std'),
        curve(initial,'x','y',r'Initial $p_H(0)$','black','dashed')],args,
        r'Initial audited fraction $p_{H_A}(0)/p_H(0)$',r'$p_H(T)$')]
    return dict(plots=plots,checks=checks,coverage_order=args.coverage,
        interpretation='Fixed stratified masks, no inverse-probability correction. Same gain; correction norms may differ.')


def projection(args,root):
    samples=[];checks=[]
    for seed in args.seeds:
        model,offsets,folder=setup(seed,args,root)
        th=model.theta_initial; mass,grad,g=population(th,model,offsets)
        pi,score=policy_arrays(th,model,offsets)
        reward=np.broadcast_to(np.array([1.,1.,0.])[None,:,None],pi.shape)
        reward_score=(reward[...,None]*score).reshape(-1,len(th))
        probability=pi.ravel()/args.n_prompts;probability/=probability.sum()
        rng=np.random.default_rng(np.random.SeedSequence([args.mc_seed,seed]));rows=[]
        h=grad[1];exact_u=-args.gain*project(h,g);exact_error=abs(g@exact_u)
        for n in args.batch_sizes:
            counts=rng.multinomial(n,probability,size=args.repetitions)
            ghat=counts@reward_score/n
            u=np.array([-args.gain*project(h,gh) for gh in ghat])
            signed=u@g;error=np.abs(signed)
            orthogonality=np.einsum('ij,ij->i',ghat,u)
            residual=np.einsum('ij,ij->i',g-ghat,u)
            bound=np.linalg.norm(g-ghat,axis=1)*np.linalg.norm(u,axis=1)
            H_dot=(g+u)@h;G_dot=(g+u)@grad[0];p_dot=g@g+signed
            check=max(np.abs(signed-residual).max(),np.abs(orthogonality).max(),exact_error)
            if check>args.identity_tol or (error-bound).max()>args.identity_tol:
                raise RuntimeError('Projection-error identity or Cauchy bound failed.')
            save_table(folder/f'batches_{n}.dat',dict(batch=np.arange(args.repetitions),
                signed_error=signed,abs_error=error,cauchy_bound=bound,p_dot=p_dot,
                p_H_dot=H_dot,p_G_dot=G_dot,orthogonality=orthogonality,
                gradient_error=np.linalg.norm(g-ghat,axis=1),correction_norm=np.linalg.norm(u,axis=1)))
            rows.append(dict(batch_size=n,abs_error=error.mean(),batch_sd=error.std(ddof=1),
                mcse=error.std(ddof=1)/np.sqrt(args.repetitions),signed_error=signed.mean(),
                acceptance_decrease_fraction=np.mean(p_dot < -args.selectivity_tol),
                selective_fraction=np.mean((H_dot < -args.selectivity_tol)&(G_dot>=-args.selectivity_tol)),
                zero_gradient_fraction=np.mean(np.sum(ghat*ghat,axis=1)==0),exact_error=exact_error))
            checks.append(dict(seed=seed,batch_size=n,identity=float(check),
                max_cauchy_excess=float((error-bound).max())))
        save_rows(folder/'summary.dat',rows)
        samples.append({k:np.array([r[k] for r in rows]) for k in rows[0]})
        print(f'projection seed={seed}: {args.repetitions} batches per sample size',flush=True)
    a=aggregate(samples,'batch_size');save_table(root/'projection_error.dat',a)
    exact={'x':a['batch_size'],'y':np.zeros_like(a['batch_size'])}
    plots=[figure(root,'projection_error',[
        curve(a,'batch_size','abs_error_mean','Estimated projection','blue',sd='abs_error_std'),
        curve(exact,'x','y','Exact projection','black','dashed')],args,
        'Samples for reward-gradient estimate',r'$|\dot p-\|g_R\|^2|$',xlog=True)]
    return dict(plots=plots,checks=checks,
        interpretation='At the initial neural policy. Only the projector uses a sampled gradient. Average within seed first; SD across seeds.')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--experiment',required=True,choices=['growth','feedback','coverage','projection'])
    p.add_argument('--output',type=Path,default=Path('./output/neural_appendix'))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--n-seeds',type=int,default=10)
    p.add_argument('--smoke',action='store_true',help='One seed, 32 projection batches; same horizon. Enables numerical checks.')
    p.add_argument('--checks',action='store_true',help='Directional/time derivative checks and a refined solve for every trajectory')
    p.add_argument('--q0',type=float,default=None,help='A/C initial hacked share (default .3); ablation default .5')
    p.add_argument('--q-values',type=float,nargs='+',default=[.1,.3,.5],help='B initial shares; must include q0')
    p.add_argument('--p0',type=float,default=2/3)
    p.add_argument('--accepted-mean',type=float,default=None,help='Default 1 for growth/feedback, 4 for ablations')
    p.add_argument('--mu-n',type=float,default=-.5)
    p.add_argument('--n-prompts',type=int,default=8);p.add_argument('--n-per-group',type=int,default=16)
    p.add_argument('--hidden-width',type=int,default=16);p.add_argument('--noise-std',type=float,default=.25)
    p.add_argument('--horizon',type=float,default=None,help='Default 50 for growth/feedback, 1 for coverage; projection has no flow')
    p.add_argument('--n-times',type=int,default=None,help='Default 501 for growth/feedback, 101 for coverage')
    p.add_argument('--lambda',dest='gain',type=float,default=6.)
    p.add_argument('--gamma',type=float,default=0.,help='Optional gradient regularization in feedback only')
    p.add_argument('--coverage',type=float,nargs='+',default=[0.,.125,.25,.5,.75,1.])
    p.add_argument('--audit-seed',type=int,default=2026)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[32,128,512,2048])
    p.add_argument('--repetitions',type=int,default=1000);p.add_argument('--mc-seed',type=int,default=2027)
    p.add_argument('--rtol',type=float,default=1e-9);p.add_argument('--atol',type=float,default=1e-11)
    p.add_argument('--max-step',type=float,default=.25);p.add_argument('--selectivity-tol',type=float,default=1e-10)
    p.add_argument('--fd-step',type=float,default=1e-5);p.add_argument('--time-fd-step',type=float,default=1e-5)
    p.add_argument('--derivative-tol',type=float,default=1e-7)
    p.add_argument('--time-derivative-tol',type=float,default=1e-6)
    p.add_argument('--identity-tol',type=float,default=1e-10);p.add_argument('--refinement-tol',type=float,default=1e-7)
    p.add_argument('--width',type=float,default=4.2);p.add_argument('--height',type=float,default=3.2)
    p.add_argument('--line-width',type=float,default=1.4)
    a=p.parse_args();basic=a.experiment in ['growth','feedback']
    if a.q0 is None:a.q0=.3 if basic else .5
    if a.accepted_mean is None:a.accepted_mean=1. if basic else 4.
    if a.horizon is None:a.horizon=50. if basic else 1.
    if a.n_times is None:a.n_times=501 if basic else 101
    if a.smoke:a.n_seeds=1;a.repetitions=min(a.repetitions,32);a.checks=True
    a.q_values=sorted(set(a.q_values));a.coverage=sorted(set(a.coverage));a.batch_sizes=sorted(set(a.batch_sizes))
    nums=[v for v in vars(a).values() if isinstance(v,(float,int))]+a.q_values+a.coverage+a.batch_sizes
    if not np.isfinite(nums).all():p.error('All numbers must be finite.')
    if not 0<a.p0<1 or not 0<a.q0<1 or not all(0<q<1 for q in a.q_values):p.error('p0 and hacked shares must lie in (0,1).')
    if a.experiment=='growth' and a.q0 not in a.q_values:p.error('q-values must include q0 for panel A.')
    if a.n_prompts<1 or a.n_per_group<2 or a.n_per_group%2 or a.hidden_width<1:p.error('Require positive prompts/width and even n-per-group >=2.')
    if a.seed<0 or a.n_seeds<1 or a.n_times<3 or min(a.audit_seed,a.mc_seed)<0:p.error('Invalid seeds or recording count.')
    if a.repetitions<2 or min(a.batch_sizes)<1:p.error('Require repetitions>=2 and positive batch sizes.')
    if not all(0<=f<=1 and abs(f*a.n_per_group-round(f*a.n_per_group))<1e-10 for f in a.coverage):p.error('Coverage counts must be exact integers per group.')
    if min(a.noise_std,a.gamma,a.gain,a.selectivity_tol)<0:p.error('Noise, gains, and selectivity tolerance must be nonnegative.')
    if min(a.horizon,a.rtol,a.atol,a.max_step,a.fd_step,a.time_fd_step,a.width,a.height,a.line_width,a.derivative_tol,a.time_derivative_tol,a.identity_tol,a.refinement_tol)<=0:p.error('Time, solver, plotting settings, and check tolerances must be positive.')
    if a.experiment!='feedback' and a.gamma!=0:p.error('gamma is supported only in feedback; growth tests verifier flow and ablations isolate audit effects.')
    a.seeds=list(range(a.seed,a.seed+a.n_seeds));return a


def main():
    args=parse_args();root=args.output/args.experiment;root.mkdir(parents=True,exist_ok=True)
    settings={**vars(args),'output':str(args.output)}
    (root/'run.json').write_text(json.dumps(dict(status='running',parameters=settings),indent=2)+'\n')
    plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    result=globals()[args.experiment](args,root)
    (root/'run.json').write_text(json.dumps(dict(status='completed',parameters=settings,**result),indent=2)+'\n')
    print(f'Saved {len(result["plots"])} individual figure(s), data, diagnostics and uncompiled TikZ to {root.resolve()}',flush=True)


if __name__=='__main__':main()
