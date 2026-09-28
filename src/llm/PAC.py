#!/usr/bin/env python3
"""Apply projected audit correction after each GRPO optimizer step. See README.md for usage."""
import math
import hashlib
import inspect
from pathlib import Path
import GRPO as grpo

if "extra_arguments" not in inspect.signature(grpo.arguments).parameters:
    raise RuntimeError("PAC.py requires the updated GRPO.py. Save both files beside SFT.py.")


def centered_coefficients(rewards, hacks, audit_probability=1.):
    """Leave-one-out baselines: unbiased gradients of p and p_H, averaged over ALL rows."""
    n = len(rewards)
    if n < 2 or len(hacks) != n:
        raise ValueError("Invalid group size")
    if not math.isfinite(audit_probability) or not 0 < audit_probability <= 1:
        raise ValueError("audit_probability must be in (0, 1]")
    inverse = 1./audit_probability
    if any(r not in (0, 1) or h not in (0, inverse) or (h and not r)
           for r, h in zip(rewards, hacks)):
        raise ValueError("Expected binary rewards and inverse-probability-weighted audit indicators")
    mean_r, mean_h = sum(rewards)/n, sum(hacks)/n
    factor = n/(n-1)
    cr = [factor*(r-mean_r) for r in rewards]
    ch = [factor*(h-mean_h) for h in hacks]
    std = math.sqrt(sum((r-mean_r)**2 for r in rewards)/n)

    grpo_factor = (n-1)/n/(std+1e-4) if std else 0.
    return cr, ch, grpo_factor


def flat(values, params, torch):
    return torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1).float()
                      for v, p in zip(values, params)])


def estimated_gradients(model, rows, tok, scaler, args, torch, device, bf16, audit_batch=None):
    """Two backward passes sharing one forward, instead of three separate passes."""
    params = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in params)
    g_r = torch.zeros(size, device=device)
    h = torch.zeros_like(g_r)
    g_grpo = torch.zeros_like(g_r)
    model.train()
    model.config.use_cache = False
    scale = scaler.get_scale()
    audits = 0
    if audit_batch is not None and (len(audit_batch["weighted_hacks"]) != len(rows)
                                   or len(audit_batch["audited"]) != len(rows)):
        raise ValueError("Audit observations must align with every rollout")
    for start in range(0, len(rows), args.group_size):
        group = rows[start:start+args.group_size]
        if len(group) != args.group_size or len({r["group"] for r in group}) != 1:
            raise ValueError("Rollouts must contain complete, consecutive prompt groups")
        rewards = [int(r["accepted"]) for r in group]
        if audit_batch is None:
            hacks = [int(not r["correct"]) if r["accepted"] else 0 for r in group]
            audits += sum(rewards)
            cr, ch, factor = centered_coefficients(rewards, hacks)
        else:

            end = start+len(group)
            hacks = audit_batch["weighted_hacks"][start:end]
            audits += sum(audit_batch["audited"][start:end])
            cr, ch, factor = centered_coefficients(rewards, hacks, audit_batch["probability"])
        for offset in range(0, len(group), args.micro_batch_size):
            batch = group[offset:offset+args.micro_batch_size]
            wr, wh = cr[offset:offset+len(batch)], ch[offset:offset+len(batch)]
            need_r, need_h = any(v != 0 for v in wr), any(v != 0 for v in wh)
            if not need_r and not need_h:
                continue
            inputs, targets, mask = grpo.pack_completions(batch, tok.pad_token_id, torch, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if bf16 else torch.float16,
                                enabled=device == "cuda"):
                logits = model(**inputs).logits[:, :-1, :]
            if logits.shape[:2] != targets.shape:
                raise AssertionError("Completion-logit alignment failed")
            logp = -torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")

            scores = (logp*mask).sum(-1)/(len(rows)*args.max_new_tokens)
            if need_r:
                objective = (scores*torch.tensor(wr, device=device, dtype=torch.float32)).sum()
                grads = torch.autograd.grad(scaler.scale(objective), params, retain_graph=need_h, allow_unused=True)
                vector = flat(grads, params, torch)/scale
                g_r.add_(vector, alpha=args.max_new_tokens)
                g_grpo.add_(vector, alpha=factor)
                del grads, vector, objective
            if need_h:
                objective = (scores*torch.tensor(wh, device=device, dtype=torch.float32)).sum()
                grads = torch.autograd.grad(scaler.scale(objective), params, allow_unused=True)
                vector = flat(grads, params, torch)/scale
                h.add_(vector, alpha=args.max_new_tokens)
                del grads, vector, objective
            del logits, logp, scores
    if not all(bool(torch.isfinite(v).all()) for v in (g_grpo, g_r, h)):
        raise FloatingPointError("Nonfinite PAC gradient estimate")
    return g_grpo, g_r, h, audits


def dot(a, b):
    import torch
    return (a*b).sum(dtype=torch.float64)


def projected_correction(g, h, base_step, nominal_budget, strength, max_ratio, torch):
    """Construct k orthogonal to estimated g, AFTER the optimizer's base step."""
    gg, hh = float(dot(g, g)), float(dot(h, h))
    base_norm = math.sqrt(max(float(dot(base_step, base_step)), 0.))
    budget = max(base_norm, nominal_budget)
    u = h.clone()
    zero_g = gg <= 1e-24
    if not zero_g:
        u.add_(g, alpha=-float(dot(g, h))/gg)
        u.add_(g, alpha=-float(dot(g, u))/gg)
    uu = max(float(dot(u, u)), 0.)
    info = {"zero_acceptance_estimate": zero_g, "g_norm": math.sqrt(max(gg, 0.)),
            "h_norm": math.sqrt(max(hh, 0.)), "projected_h_norm": math.sqrt(uu),
            "base_step_norm": base_norm, "step_budget": budget, "gain_capped": False}
    if strength == 0 or hh <= 1e-24 or uu <= max(1e-24, hh*1e-12):
        return torch.zeros_like(h), {**info, "gain": 0., "planned_correction_norm": 0.}
    unorm = math.sqrt(uu)
    hb, gb = float(dot(h, base_step)), float(dot(g, base_step))
    needed = max(0., hb-min(0., gb))/unorm
    requested_norm = needed + strength*budget
    correction_norm = min(requested_norm, max_ratio*budget)
    gain = correction_norm/unorm
    info.update(gain=gain, gain_capped=correction_norm < requested_norm,
                planned_correction_norm=correction_norm)
    return -gain*u, info


def apply_flat_add(params, vector, torch):
    offset = 0
    with torch.no_grad():
        for p in params:
            end = offset+p.numel()
            p.add_(vector[offset:end].view_as(p).to(p.dtype))
            offset = end


def budget_correction(g, h, base_step, nominal_budget, strength, max_ratio, torch, direction):
    """Projection ablation: identical norm-budget rule, with no sign-based gain."""
    if direction not in ("projected", "raw"):
        raise ValueError("Unknown audit direction")
    gg, hh = float(dot(g, g)), float(dot(h, h))
    base_norm = math.sqrt(max(float(dot(base_step, base_step)), 0.))
    budget = max(base_norm, nominal_budget)
    u = h.clone()
    if direction == "projected" and gg > 1e-24:
        u.add_(g, alpha=-float(dot(g, h))/gg)
        u.add_(g, alpha=-float(dot(g, u))/gg)
    uu = max(float(dot(u, u)), 0.)
    requested = strength*budget
    size = min(requested, max_ratio*budget)
    info = {"zero_acceptance_estimate": gg <= 1e-24,
            "g_norm": math.sqrt(max(gg, 0.)), "h_norm": math.sqrt(max(hh, 0.)),
            "direction_norm": math.sqrt(uu), "base_step_norm": base_norm,
            "step_budget": budget, "gain_capped": size < requested,
            "audit_direction": direction, "gain_rule": "budget"}
    if strength == 0 or hh <= 1e-24 or uu <= max(1e-24, hh*1e-12):
        return torch.zeros_like(h), {**info, "gain": 0., "planned_correction_norm": 0.}
    gain = size/math.sqrt(uu)
    return -gain*u, {**info, "gain": gain, "planned_correction_norm": size}


def update_policy(model, rows, tok, optimizer, scaler, args, torch, device, bf16, *, audit_batch=None):

    if args.correction_strength == 0:
        result = grpo.update_policy(model, rows, tok, optimizer, scaler, args, torch, device, bf16)
        return {**result, "reward_updated": result["updated"], "pac_applied": False,
                "audits": 0, "estimated_selective": False}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    g_grpo, g_r, h, audits = estimated_gradients(model, rows, tok, scaler, args, torch, device, bf16,
                                              audit_batch=audit_batch)
    original = flat([p.detach() for p in params], params, torch)
    reward_updated = any(r["advantage"] != 0 for r in rows)
    norm = 0.
    if reward_updated:
        scale, offset = scaler.get_scale(), 0
        for p in params:
            end = offset+p.numel()

            p.grad = (-scale*g_grpo[offset:end]).view_as(p).to(p.dtype).clone()
            offset = end
        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite reward gradient")
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    base_parameters = flat([p.detach() for p in params], params, torch)
    base_step = base_parameters-original

    nominal = .25*args.lr*math.sqrt(original.numel())
    direction = getattr(args, "audit_direction", "projected")
    gain_rule = getattr(args, "gain_rule", "adaptive")
    if gain_rule == "budget":
        correction, geometry = budget_correction(g_r, h, base_step, nominal,
            args.correction_strength, args.max_correction_ratio, torch, direction)
    elif gain_rule == "adaptive" and direction == "projected":
        correction, geometry = projected_correction(g_r, h, base_step, nominal,
            args.correction_strength, args.max_correction_ratio, torch)
    else:
        raise ValueError("Raw audit requires the shared budget gain rule")
    apply_flat_add(params, correction, torch)
    actual = flat([p.detach() for p in params], params, torch)
    actual_correction, total_step = actual-base_parameters, actual-original
    correction_norm = math.sqrt(max(float(dot(actual_correction, actual_correction)), 0.))
    acceptance_effect = float(dot(g_r, total_step))
    hack_effect = float(dot(h, total_step))
    correct_effect = acceptance_effect-hack_effect
    orthogonality_error = abs(float(dot(g_r, actual_correction)))
    relative_error = orthogonality_error/(geometry["g_norm"]*correction_norm+1e-30)
    applied = correction_norm > 0
    return {"updated": reward_updated or applied, "reward_updated": reward_updated,
            "pac_applied": applied and direction == "projected", "correction_applied": applied,
            "grad_norm": norm, "audits": audits,
            "active_completions": sum(r["advantage"] != 0 for r in rows), **geometry,
            "actual_correction_norm": correction_norm,
            "projection_relative_error": relative_error if direction == "projected" else None,
            "correction_acceptance_effect": float(dot(g_r, actual_correction)),
            "estimated_acceptance_change": acceptance_effect,
            "estimated_H_change": hack_effect, "estimated_G_change": correct_effect,
            "estimated_selective": hack_effect < 0 and correct_effect > 0}


def arguments():
    def extra(parser):
        parser.set_defaults(output_dir="PAC_output")
        parser.add_argument("--correction-strength", type=float, default=1.)
        parser.add_argument("--max-correction-ratio", type=float, default=2.)
    args = grpo.arguments(extra_arguments=extra)
    if not math.isfinite(args.correction_strength) or args.correction_strength < 0:
        raise ValueError("correction-strength must be finite and nonnegative")
    if not math.isfinite(args.max_correction_ratio) or args.max_correction_ratio <= 0:
        raise ValueError("max-correction-ratio must be finite and positive")
    return args


if __name__ == "__main__":
    grpo.run(arguments(), update_fn=update_policy, method="PAC", extra_provenance={
        "correction_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_probability_on_accepted": 1.,
        "acceptance_gradient": "score estimator with leave-one-out baseline; not GRPO's std-weighted gradient",
        "audit_gradient": "full exact accepted-response audits; leave-one-out baseline; all-rollout denominator",
        "projection": "Euclidean in trainable LoRA coordinates; applied after Adam displacement",
        "gain": "same-batch adaptive margin, capped correction norm; not the fixed-gain expectation theorem",
        "validation_scope": "batch-estimated directional signs and before/after probe; no independent local certificate",
    })
