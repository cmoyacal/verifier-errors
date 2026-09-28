#!/usr/bin/env python3
"""Train an on-policy LoRA GRPO baseline using binary verifier rewards. See README.md for usage."""
import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from SFT import Problem, check_task, evaluate, label_response, save_json, save_jsonl


def read_split(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    problems = []
    for row in rows:
        p = Problem(**{k: row[k] for k in ("rule1", "rule2", "start", "hint")})
        if row.get("key") != p.key or row.get("answer") != p.answer:
            raise ValueError(f"Inconsistent saved task: {p.key}")
        problems.append(p)
    if not problems:
        raise ValueError(f"Empty split: {path}")
    return problems


def group_advantages(rewards):
    """Input is ONLY binary verifier rewards; no G/H or ground-truth labels."""
    if len(rewards) < 2 or any(r not in (0, 1) for r in rewards):
        raise ValueError("Expected at least two binary rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards) / len(rewards))
    return [(r-mean)/(std+1e-4) for r in rewards] if std else [0.] * len(rewards)


def through_eos(tokens, eos):

    try:
        return tokens[:tokens.index(eos)+1], False
    except ValueError:
        return tokens, True


def pack_completions(rows, pad_id, torch, device):
    prompt_width = max(len(r["prompt_ids"]) for r in rows)
    completion_width = max(len(r["completion_ids"]) for r in rows)
    ids, attention, targets, masks = [], [], [], []
    for row in rows:
        prompt, completion = row["prompt_ids"], row["completion_ids"]
        left, right = prompt_width-len(prompt), completion_width-len(completion)
        ids.append([pad_id]*left + prompt + completion + [pad_id]*right)
        attention.append([0]*left + [1]*(len(prompt)+len(completion)) + [0]*right)
        targets.append(completion + [pad_id]*right)
        masks.append([1]*len(completion) + [0]*right)
    tensor = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    attention = tensor(attention)

    positions = attention.cumsum(-1)-1
    positions.masked_fill_(attention == 0, 1)
    inputs = {"input_ids": tensor(ids), "attention_mask": attention,
              "position_ids": positions, "use_cache": False,
              "logits_to_keep": completion_width+1}
    return inputs, tensor(targets), tensor(masks)


def surrogate_loss(log_probs, mask, advantages, denominator):

    ratio = (log_probs-log_probs.detach()).exp()
    return -(ratio * mask * advantages[:, None]).sum() / denominator


def rollout_stats(rows):
    counts = Counter(r["label"] for r in rows)
    n = len(rows)
    accepted = counts["G"] + counts["H"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    mixed = sum(any(r["accepted"] for r in g) and any(not r["accepted"] for r in g)
                for g in groups.values())
    valid_mixed = sum(any(r["accepted"] for r in g)
                      and any(r["label"] == "N" and r["valid_format"] for r in g)
                      for g in groups.values())
    lengths = {k: [len(r["completion_ids"]) for r in rows if r["label"] == k]
               for k in ("G", "H", "N") if rows and "completion_ids" in rows[0]}
    return {"n": n, "G": counts["G"]/n, "H": counts["H"]/n,
            "valid_N": sum(r["label"] == "N" and r["valid_format"] for r in rows)/n,
            "invalid": sum(not r["valid_format"] for r in rows)/n,
            "token_hits": sum(r["truncated"] for r in rows)/n,
            "p": accepted/n, "q": counts["H"]/accepted if accepted else None,
            "mixed_groups": mixed, "valid_mixed_groups": valid_mixed, "groups": len(groups),
            "mean_tokens": {k: sum(v)/len(v) if v else None for k, v in lengths.items()}}


def draw_rollouts(model, tok, problems, args, torch, device, bf16, step):
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    requests = [(i, p) for i, p in enumerate(problems) for _ in range(args.group_size)]
    rows = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(args.seed+20000+step)
        with torch.inference_mode(), torch.autocast(device_type=device,
                    dtype=torch.bfloat16 if bf16 else torch.float16, enabled=device == "cuda"):
            for start in range(0, len(requests), args.gen_batch_size):
                batch = requests[start:start+args.gen_batch_size]
                prompts = [p.prompt() for _, p in batch]
                inputs = tok(prompts, padding=True, add_special_tokens=False,
                             return_token_type_ids=False, return_tensors="pt").to(device)
                sequences = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                          do_sample=True, temperature=1., top_p=1., top_k=0, use_cache=True,
                          eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
                generated = sequences[:, inputs["input_ids"].shape[1]:].tolist()
                for (group, problem), tokens in zip(batch, generated):
                    tokens, truncated = through_eos(tokens, tok.eos_token_id)
                    text = tok.decode(tokens, skip_special_tokens=True)
                    rows.append({"group": group, "key": problem.key,
                                 "prompt_ids": tok.encode(problem.prompt(), add_special_tokens=False),
                                 "completion_ids": tokens, "completion": text, "truncated": truncated,
                                 **label_response(text, problem)})
    for start in range(0, len(rows), args.group_size):
        group = rows[start:start+args.group_size]
        advantages = group_advantages([int(row["accepted"]) for row in group])
        for row, advantage in zip(group, advantages):
            row["advantage"] = advantage
    return rows


def update_policy(model, rows, tok, optimizer, scaler, args, torch, device, bf16):

    active = [r for r in rows if r["advantage"] != 0.]
    optimizer.zero_grad(set_to_none=True)
    if not active:
        return {"updated": False, "grad_norm": 0., "active_completions": 0}
    model.train()
    model.config.use_cache = False
    for start in range(0, len(active), args.micro_batch_size):
        batch = active[start:start+args.micro_batch_size]
        inputs, targets, mask = pack_completions(batch, tok.pad_token_id, torch, device)
        advantages = torch.tensor([r["advantage"] for r in batch], device=device, dtype=torch.float32)
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if bf16 else torch.float16,
                            enabled=device == "cuda"):
            logits = model(**inputs).logits[:, :-1, :]
        if logits.shape[:2] != targets.shape:
            raise AssertionError("Unexpected completion-logit alignment")
        log_probs = -torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), targets,
                                                       reduction="none")
        loss = surrogate_loss(log_probs, mask, advantages, len(rows)*args.max_new_tokens)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite GRPO loss")
        scaler.scale(loss).backward()
        del logits, log_probs, loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite GRPO gradient; checkpoint remains available")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return {"updated": True, "grad_norm": float(norm), "active_completions": len(active)}


def calibration_probe(model, tok, problems, args, out, phase, torch, device, bf16):

    if getattr(args, "probe_seed", None) is not None:
        args = copy.copy(args)
        args.seed = args.probe_seed
    with torch.autocast(device_type=device, dtype=torch.bfloat16 if bf16 else torch.float16,
                        enabled=device == "cuda"):
        metrics = evaluate(model, tok, {args.length: problems}, args, out, phase,
                           torch, device, samples=args.samples)[str(args.length)]
    rows = [json.loads(line) for line in (out/f"{phase}_L{args.length}.jsonl").read_text().splitlines()]
    for row in rows:
        row["group"] = row["key"]
    return {**rollout_stats(rows), "full_metrics": metrics}


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def arguments(extra_arguments=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", default="SFT_output")
    p.add_argument("--init-run", default="SFT_3_output")
    p.add_argument("--output-dir", default="GRPO_output")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--probe-prompts", type=int, default=16)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--no-signal-patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-seed", type=int, default=None)
    p.add_argument("--probe-rounds", type=int, nargs="*", default=[],
                   help="Optional intermediate calibration probes; no checkpoint selection")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    if extra_arguments is not None:
        extra_arguments(p)
    args = p.parse_args()
    for name in ("steps", "prompts_per_step", "gen_batch_size", "micro_batch_size",
                 "eval_batch_size", "probe_prompts", "max_new_tokens", "no_signal_patience"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.group_size < 2 or args.samples < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("group-size and samples must be >= 2; lr must be finite and positive")
    if args.smoke:
        args.steps, args.prompts_per_step, args.group_size = 2, 1, 4
        args.probe_prompts, args.samples, args.gen_batch_size, args.eval_batch_size = 2, 2, 4, 4
    args.probe_rounds = sorted(set(s for s in args.probe_rounds if 0 < s < args.steps))
    return args


def run(args, update_fn=None, method="GRPO", extra_provenance=None):
    update_fn = update_policy if update_fn is None else update_fn
    read = lambda p: json.loads(p.read_text())
    source, init, out = [Path(v).expanduser().resolve() for v in (args.source_run, args.init_run, args.output_dir)]
    for original in (source, init):
        if original == out or original in out.parents or out in original.parents:
            raise ValueError("Use a separate output folder outside the input runs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty; use a new folder")
    source_config, init_config = read(source/"config.json"), read(init/"config.json")
    if not args.smoke and (init_config.get("smoke") or not read(source/"results.json").get("gate_passed")):
        raise ValueError("Full GRPO requires non-smoke initial weights and completed SFT source")
    args.model, args.length = init_config["model"], int(init_config["length"])
    if source_config["model"] != args.model or read(init/"adapter"/"adapter_config.json").get("base_model_name_or_path") != args.model:
        raise ValueError("Initial checkpoint and source name different models")
    groups = {}
    for split in ("train", "calibration"):
        filename = f"data_{split}_L{args.length}.jsonl"
        data = read_split(source/filename)
        if read_split(init/filename) != data:
            raise ValueError("Initial checkpoint and source have different tasks")
        groups[split] = {args.length: data}
    checks = check_task(groups)
    train, calibration = groups["train"][args.length], groups["calibration"][args.length]
    if args.prompts_per_step > len(train):
        raise ValueError("prompts-per-step exceeds the training split")
    probe_seed = args.seed if getattr(args, "probe_seed", None) is None else args.probe_seed
    probe = random.Random(probe_seed+900).sample(calibration, min(args.probe_prompts, len(calibration)))
    provenance = read(init/"provenance.json")
    revision = provenance.get("base_revision") or source_config.get("revision", "main")
    os.environ.setdefault("USE_TF", "0")
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Use a GPU runtime")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(init/"adapter", local_files_only=True)
    if tok.eos_token_id is None:
        raise ValueError("Missing tokenizer EOS")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if max(len(tok.encode(p.gold(), add_special_tokens=False))+1 for p in train) > args.max_new_tokens:
        raise ValueError("Generation cap cannot fit a correct completion; increase max-new-tokens")
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=revision,
               dtype=torch.bfloat16 if bf16 else torch.float32, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base, init/"adapter", is_trainable=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in n for n, _ in named):
        raise AssertionError("Expected only LoRA weights to be trainable")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.config.attention_dropout = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for _, p in named], lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not bf16)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/"config.json", vars(args))
    save_json(out/"task_checks.json", checks)
    save_json(out/"provenance.json", {"source_run": str(source), "init_run": str(init),
        "base_revision": revision, "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(sys.modules[Problem.__module__].__file__).read_bytes()).hexdigest(),
        "init_adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (init/"adapter").glob("adapter*") if p.is_file()},
        "algorithm": "one-iteration on-policy GRPO; group population std + 1e-4; fixed B*max_new_tokens denominator",
        "kl_beta": 0., "gradient_clip": 1., "optimizer": "AdamW, weight_decay=0",
        "evaluation": "fixed calibration probe before/after and requested intermediate rounds; no best-checkpoint selection",
        "method": method, **(extra_provenance or {})})
    for name, split in groups.items():
        save_jsonl(out/f"data_{name}_L{args.length}.jsonl",
                   [{**asdict(p), "key": p.key, "answer": p.answer} for p in split[args.length]])
    save_json(out/"calibration_probe_keys.json", [p.key for p in probe])
    print(f"{method} PILOT | {args.steps} rounds | {args.prompts_per_step} prompts × {args.group_size} samples/round | KL=0", flush=True)
    print("Before: small calibration probe...", flush=True)
    before = calibration_probe(model, tok, probe, args, out, "before", torch, device, bf16)
    trajectory = [{"round": 0, **before}]
    save_json(out/"trajectory.json", trajectory)
    print(f"Before: p={pct(before['p'])} q={pct(before['q'])} G={pct(before['G'])} H={pct(before['H'])} valid N={pct(before['valid_N'])}", flush=True)
    rng = random.Random(args.seed+40000)
    history, window, updates, no_signal = [], [], 0, 0
    reason = "fixed training budget completed"
    last_report = time.monotonic()
    def save_checkpoint(step):
        model.save_pretrained(out/"adapter")
        tok.save_pretrained(out/"adapter")
        save_json(out/"checkpoint_progress.json", {"last_round": step, "policy_updates": updates,
            "optimizer_updates": sum(int(h.get("reward_updated", h["updated"])) for h in history)})
    with (out/"rollouts.jsonl").open("w", buffering=1) as raw, (out/"training.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.steps+1):
            rows = draw_rollouts(model, tok, rng.sample(train, args.prompts_per_step), args,
                                 torch, device, bf16, step)
            for row in rows:
                raw.write(json.dumps({"round": step, **row})+"\n")
            update = update_fn(model, rows, tok, optimizer, scaler, args, torch, device, bf16)
            updates += int(update["updated"])
            no_signal = 0 if update["updated"] else no_signal+1
            item = {"round": step, **rollout_stats(rows), **update}
            history.append(item)
            log.write(json.dumps(item)+"\n")
            window.extend([{**r, "group": (step, r["group"])} for r in rows])
            if step == 1 or step % 5 == 0 or step == args.steps or no_signal >= args.no_signal_patience or time.monotonic()-last_report >= 45:
                stats = rollout_stats(window)
                print(f"  Round {step}/{args.steps} | train rollouts: p={pct(stats['p'])} q={pct(stats['q'])}"
                      f" G={pct(stats['G'])} H={pct(stats['H'])} valid N={pct(stats['valid_N'])}"
                      f" | valid mixed={stats['valid_mixed_groups']}/{stats['groups']}"
                      f" | format={pct(stats['invalid'])} | updates={updates}", flush=True)
                if method in ("PAC", "RAW AUDIT"):
                    print(f"    Last correction: {'applied' if update.get('correction_applied', update.get('pac_applied')) else 'none'}"
                          f" | batch estimates H↓/G↑: {update.get('estimated_selective', False)}"
                          f" | audits={update.get('audits', 0)}", flush=True)
                window = []
                last_report = time.monotonic()
            if step % 5 == 0 or step == args.steps or no_signal >= args.no_signal_patience:
                save_checkpoint(step)
            if step in getattr(args, "probe_rounds", []):
                snapshot = calibration_probe(model, tok, probe, args, out, f"round_{step}", torch, device, bf16)
                trajectory.append({"round": step, **snapshot})
                save_json(out/"trajectory.json", trajectory)
                print(f"  Probe round {step}: p={pct(snapshot['p'])} q={pct(snapshot['q'])}"
                      f" G={pct(snapshot['G'])} H={pct(snapshot['H'])}", flush=True)
            if no_signal >= args.no_signal_patience:
                reason = (f"stopped after {no_signal} consecutive batches with zero group advantage"
                          if method == "GRPO" else f"stopped after {no_signal} consecutive batches with no usable update")
                print(reason, flush=True)
                break
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    print("After: same small calibration probe...", flush=True)
    after = calibration_probe(model, tok, probe, args, out, "after", torch, device, bf16)
    trajectory = [r for r in trajectory if r["round"] != len(history)]
    trajectory.append({"round": len(history), **after})
    save_json(out/"trajectory.json", trajectory)
    dp = after["p"]-before["p"]
    dq = after["q"]-before["q"] if before["q"] is not None and after["q"] is not None else None
    joint = dp > 0 and dq is not None and dq > 0
    dg, dh = after["G"]-before["G"], after["H"]-before["H"]
    selective = dh < 0 and dg > 0
    if method in ("PAC", "RAW AUDIT"):
        status = "H DOWN AND G UP IN PROBE" if selective else "NO SELECTIVE CHANGE IN PROBE"
    else:
        status = "p AND q UP IN PROBE" if joint else "NO JOINT INCREASE IN PROBE"
    if args.smoke:
        status = "SMOKE ONLY"
    summary = [f"{method} PILOT — {status}",
               f"{args.model} | L={args.length} | rounds={len(history)} | updates={updates} | seed={args.seed}",
               f"One policy, LoRA | KL=0 | group={args.group_size} | one update per fresh batch | fixed token normalization",
               f"Calibration only: {len(probe)} prompts × {args.samples} samples | T=1",
               "          G        H        valid N  p        q        format   token hits"]
    for name, row in (("Before", before), ("After", after)):
        summary.append(f"{name:<9} " + " ".join(f"{pct(row[k]):<8}" for k in ("G", "H", "valid_N", "p", "q", "invalid", "token_hits")))
    dqtext = "n/a" if dq is None else f"{100*dq:+.1f} pp"
    summary += [f"Change: p {100*dp:+.1f} pp | q {dqtext}",
                f"After groups mixing accepted + valid N: {after['valid_mixed_groups']}/{after['groups']}",
                f"Stop: {reason} | Elapsed: {(time.monotonic()-started)/60:.1f} min",
                "Small diagnostic; one seed. Saved: adapter/, summary.txt, results.json, training.jsonl, rollouts.jsonl"]
    if method in ("PAC", "RAW AUDIT"):
        summary.insert(-1, f"Change: G {100*dg:+.1f} pp | H {100*dh:+.1f} pp")
        summary.insert(-1, f"Exact audits: {sum(h.get('audits', 0) for h in history)} accepted responses"
                       f" | corrections: {sum(h.get('correction_applied', h.get('pac_applied', False)) for h in history)}/{len(history)}"
                       f" | estimator signs met: {sum(h.get('estimated_selective', False) for h in history)}/{len(history)}")
        if hasattr(args, "audit_probability"):
            summary.insert(-1, f"Requested audit coverage: {args.audit_probability:.0%} | inverse-probability weighting; oracle metrics are reporting only")
    summary = "\n".join(summary)+"\n"
    (out/"summary.txt").write_text(summary)
    save_json(out/"results.json", {"experiment": f"{method} pilot", "smoke": args.smoke,
        "before": before, "after": after, "delta_p": dp, "delta_q": dq,
        "joint_increase_in_probe": joint and not args.smoke,
        "selective_change_in_probe": selective and not args.smoke, "delta_G": dg, "delta_H": dh,
        "optimizer_updates": sum(int(h.get("reward_updated", h["updated"])) for h in history),
        "policy_updates": updates,
        "rounds": len(history), "stop_reason": reason, "history": history, "trajectory": trajectory,
        "scope": "small calibration diagnostic, no statistical success claim or test evaluation"})
    if not getattr(args, "defer_summary", False):
        print("\n"+summary, flush=True)


if __name__ == "__main__":
    run(arguments())
