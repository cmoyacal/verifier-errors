#!/usr/bin/env python3
"""Collect paired runs with partial random audits and reuse the full-audit baseline. See README.md for usage."""
import copy
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import GRPO as grpo
import PAC as pac
import evaluation as comparison

if "audit_batch" not in inspect.signature(pac.update_policy).parameters:
    raise RuntimeError("Save the updated PAC.py, GRPO.py, and evaluation.py beside audit_coverage.py")
if not hasattr(comparison, "reusable_evaluations"):
    raise RuntimeError("audit_coverage.py requires the updated evaluation.py")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_audits(rows, probability, rng):
    """Read correctness ONLY after the independent audit coin selects a row."""
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("audit-probability must be in (0, 1]")
    audited, weighted_hacks, records, clean = [], [], [], []
    for i, row in enumerate(rows):
        draw = rng.random()
        accepted = bool(row["accepted"])
        reveal = accepted and draw < probability
        correct = bool(row["correct"]) if reveal else None
        weight = 1./probability if reveal and not correct else 0.
        audited.append(reveal)
        weighted_hacks.append(weight)
        records.append({"slot": i, "group": row["group"], "key": row.get("key"),
                        "accepted": accepted, "audited": reveal,
                        "observed_correct": correct, "weighted_hack": weight})
        clean.append({key: row[key] for key in
                      ("group", "prompt_ids", "completion_ids", "accepted", "advantage")})
    return clean, {"probability": probability, "audited": audited,
                   "weighted_hacks": weighted_hacks}, records


class CoverageUpdate:
    def __init__(self, probability, seed, log_path):
        self.probability, self.seed, self.log_path, self.round = probability, seed, log_path, 0

    def __call__(self, model, rows, tok, optimizer, scaler, args, torch, device, bf16):
        self.round += 1

        rng = random.Random(f"audit-coverage-v1:{self.seed}:{self.round}")
        clean, observations, records = sample_audits(rows, self.probability, rng)
        with self.log_path.open("a") as stream:
            for row in records:
                stream.write(json.dumps({"round": self.round, **row})+"\n")
        result = pac.update_policy(model, clean, tok, optimizer, scaler, args, torch, device, bf16,
                                   audit_batch=observations)
        audited = sum(observations["audited"])
        assert result["audits"] == audited
        return {**result, "audit_probability": self.probability,
                "accepted_candidates": sum(bool(r["accepted"]) for r in clean),
                "audited_hacks": sum(r["audited"] and r["observed_correct"] is False for r in records),
                "no_audits": audited == 0}


def arguments():
    def extra(parser):
        parser.description = __doc__
        parser.set_defaults(init_run="SFT_3_output", output_dir=None,
                            probe_prompts=8, probe_seed=1234, probe_rounds=[5, 10])
        parser.add_argument("--full-audit-run", default=None,
                            help="Matching saved 100%% pair; default audit_seedN")
        parser.add_argument("--audit-probability", type=float, default=.5)
        parser.add_argument("--audit-seed", type=int, default=None,
                            help="Independent audit RNG seed; defaults to the training seed")
        parser.add_argument("--correction-strength", type=float, default=1.)
        parser.add_argument("--max-correction-ratio", type=float, default=2.)
        parser.add_argument("--test-prompts", type=int, default=32)
        parser.add_argument("--test-samples", type=int, default=4)
        parser.add_argument("--test-seed", type=int, default=1234)
    args = grpo.arguments(extra_arguments=extra)
    if not math.isfinite(args.audit_probability) or not 0 < args.audit_probability <= 1:
        raise ValueError("audit-probability must be in (0, 1]; use GRPO as the zero-audit baseline")
    if any(not math.isfinite(v) or v <= 0 for v in (args.correction_strength, args.max_correction_ratio)):
        raise ValueError("Correction strength and maximum ratio must be finite and positive")
    if args.test_prompts < 1 or args.test_samples < 2:
        raise ValueError("test-prompts must be positive; test-samples must be >= 2")
    args.full_audit_run = args.full_audit_run or f"audit_seed{args.seed}"
    tag = f"{100*args.audit_probability:g}".replace(".", "p")
    args.output_dir = args.output_dir or f"audit{tag}_seed{args.seed}"
    args.audit_seed = args.seed if args.audit_seed is None else args.audit_seed
    args.no_signal_patience = args.steps+1
    if args.smoke:
        args.probe_rounds, args.test_prompts, args.test_samples = [1], 2, 2
    return args


def validate_baseline(args):
    full = Path(args.full_audit_run).expanduser().resolve()
    config = comparison.read_json(full/"config.json")
    saved = comparison.read_json(full/"results.json")
    if saved.get("seed") != args.seed or saved.get("experiment") != "raw versus projected audit ablation":
        raise ValueError("Use the matching training seed's original full-audit pair")
    args.grpo_run = config["grpo_run"]
    source = Path(args.source_run).expanduser().resolve()
    initial = Path(args.init_run).expanduser().resolve()
    runs = {"initial": initial, "grpo": Path(args.grpo_run).expanduser().resolve(),
            "raw": full/"raw", "pac": full/"pac"}
    checked = comparison.validate_inputs(source, runs, args.smoke)
    groups, length, model, _, _, hashes, _, configs = checked
    fields = ("seed", "steps", "prompts_per_step", "group_size", "gen_batch_size", "micro_batch_size",
              "eval_batch_size", "lr", "max_new_tokens", "correction_strength", "max_correction_ratio",
              "no_signal_patience", "probe_prompts", "samples", "probe_seed", "probe_rounds")
    for arm in ("raw", "pac"):
        prov = comparison.read_json(runs[arm]/"provenance.json")
        if prov.get("audit_probability_on_accepted") != 1. or configs[arm].get("gain_rule") != "budget":
            raise ValueError("Reference must use full audits and the same norm-budget rule")
        if not args.smoke:
            for field in fields:
                if configs[arm].get(field) != getattr(args, field):
                    raise ValueError(f"Full-audit {arm} differs in {field}; match the reference setting")
    if args.test_prompts > len(groups["test"][length]) or args.probe_prompts > len(groups["calibration"][length]):
        raise ValueError("Requested probes exceed the saved splits")
    if not args.smoke:
        eval_args = SimpleNamespace(length=length, model=model, seed=args.test_seed,
            prompts=args.test_prompts, samples=args.test_samples, eval_batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens, smoke=False)
        keys = [p.key for p in random.Random(args.test_seed).sample(groups["test"][length], args.test_prompts)]
        comparison.reusable_evaluations(full/"heldout", eval_args, hashes, keys)
        prov = comparison.read_json(full/"heldout"/"provenance.json")
        if any(prov["adapter_sha256"][arm] != hashes[arm] for arm in ("raw", "pac")):
            raise ValueError("Full-audit held-out results refer to different frozen adapters")
        if comparison.read_json(full/"heldout"/"results.json").get("smoke"):
            raise ValueError("Full run requires non-smoke baseline evaluation")
    return full, runs, hashes


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cost(result):
    history = result["history"]
    eligible = sum(h.get("accepted_candidates", round(h["p"]*h["n"])) for h in history)
    audits = sum(h["audits"] for h in history)
    return {"rollouts": sum(h["n"] for h in history), "accepted_candidates": eligible,
            "audits": audits, "realized_coverage": audits/eligible if eligible else None,
            "corrections": sum(h.get("correction_applied", h.get("pac_applied", False)) for h in history),
            "rounds_without_observed_hacks": (sum(h["audited_hacks"] == 0 for h in history)
                                               if all("audited_hacks" in h for h in history) else None),
            "updates": result["policy_updates"], "adam_steps": result["optimizer_updates"]}


def summarize(root, args, full, elapsed):
    train = {arm: comparison.read_json(root/arm/"results.json") for arm in ("raw", "pac")}
    held = comparison.read_json(root/"heldout"/"results.json")
    costs = {arm: cost(result) for arm, result in train.items()}
    label = f"{args.audit_probability:.0%}"
    lines = ["AUDIT COVERAGE — " + ("SMOKE ONLY" if args.smoke else "PAIRED RUN"),
        f"Seed {args.seed} | audit probability {label} | fixed {args.steps} rounds/arm | KL=0",
        f"Same initial adapter | group={args.group_size} | acceptance gradient uses every rollout",
        "Only independently revealed correctness labels enter the correction; inverse-probability weighting.",
        f"Calibration: {args.probe_prompts} prompts × {args.samples} samples | T=1",
        "Arm   Round  G        H        p        q"]
    for arm in ("raw", "pac"):
        for t in train[arm]["trajectory"]:
            lines.append(f"{arm.upper():<5} {t['round']:<6} " + " ".join(f"{grpo.pct(t[k]):<8}" for k in ("G", "H", "p", "q")))
        c = costs[arm]
        lines.append(f"{arm.upper()} cost: audits={c['audits']}/{c['accepted_candidates']} accepted"
            f" ({grpo.pct(c['realized_coverage'])}) | rollouts={c['rollouts']} | corrections={c['corrections']}/{args.steps}"
            f" | no observed H={c['rounds_without_observed_hacks']}/{args.steps}")
    lines += ["", "Held-out policy   G        H        valid N  p        q        format   token hits"]
    policies = {}
    baseline = None if args.smoke else comparison.read_json(full/"heldout"/"results.json")
    if baseline:
        for arm in ("raw", "pac"):
            policies[f"{arm}_100"] = baseline["policies"][arm]
    for arm in ("raw", "pac"):
        policies[arm] = held["policies"][arm]
    for name, m in policies.items():
        display = name.replace("_100", " 100%") if name.endswith("_100") else f"{name} {label}"
        lines.append(f"{display.upper():<17}" + " ".join(f"{grpo.pct(m[k]):<8}" for k in
                     ("G", "H", "valid_N", "p", "q", "invalid", "truncated")))
    contrasts = {}
    baseline_costs = {}
    if baseline:
        records = {arm: read_rows(root/"heldout"/f"{arm}_L{args.length}.jsonl") for arm in ("raw", "pac")}
        for arm in ("raw", "pac"):
            records[f"{arm}_100"] = read_rows(full/"heldout"/f"{arm}_L{args.length}.jsonl")
            baseline_costs[arm] = cost(comparison.read_json(full/arm/"results.json"))
        specifications = {f"PAC minus raw ({label})": ("pac", "raw", ("G", "H", "p", "q")),
            f"Raw {label} minus 100%": ("raw", "raw_100", ("G", "H", "p", "q")),
            f"PAC {label} minus 100%": ("pac", "pac_100", ("G", "H", "p", "q"))}
        contrasts = comparison.paired_differences(records, held["prompt_keys"], args.test_samples,
                                                   args.test_seed+50000, contrasts=specifications)
        lines.append("Differences, percentage points [approx. paired prompt-bootstrap 95% interval]:")
        for name, values in contrasts.items():
            parts = []
            for key in ("G", "H"):
                m = values[key]
                parts.append(f"{key} {100*m['delta']:+.1f} [{100*m['ci95'][0]:+.1f}, {100*m['ci95'][1]:+.1f}]")
            lines.append(f"{name}: " + " | ".join(parts))
        for arm in ("raw", "pac"):
            lines.append(f"{arm.upper()} audits: 100%={baseline_costs[arm]['audits']} | {label}={costs[arm]['audits']}")
    runtime_fields = ("torch", "transformers", "peft")
    current = comparison.read_json(root/"pac"/"provenance.json")
    previous = comparison.read_json(full/"pac"/"provenance.json")
    runtime_differences = {k:{"current":current.get(k),"baseline":previous.get(k)}
                           for k in runtime_fields if current.get(k) != previous.get(k)}
    if runtime_differences:
        lines.append("Runtime versions differ from the 100% baseline; exact fields saved in results.json.")
    lines += ["Small probes; one training seed. Same-batch signs are diagnostics, not certificates.",
              "Coverage restricts labels used for training; exact oracle labels in reporting are separate.",
              f"Elapsed: {elapsed/60:.1f} min | Saved: {root}/summary.txt, results.json, audits.jsonl in each arm"]
    grpo.save_json(root/"results.json", {"experiment":"random audit coverage ablation", "seed":args.seed,
        "smoke":args.smoke, "audit_probability":args.audit_probability, "audit_seed":args.audit_seed,
        "trajectories":{arm:r["trajectory"] for arm,r in train.items()}, "costs":costs,
        "baseline_costs":baseline_costs, "full_audit_run":str(full), "heldout":held,
        "policies":policies, "coverage_contrasts":contrasts, "runtime_differences":runtime_differences,
        "elapsed_minutes":elapsed/60,
        "scope":"uniform random audit coverage; shared norm-budget rule; full oracle diagnostics are not training inputs"})
    text = "\n".join(lines)+"\n"
    (root/"summary.txt").write_text(text)
    return text


def run(args):
    full, runs, hashes = validate_baseline(args)
    root = Path(args.output_dir).expanduser().resolve()
    for original in [full, Path(args.source_run).expanduser().resolve(), *runs.values()]:
        if root == original or root in original.parents or original in root.parents:
            raise ValueError("Use a new output folder outside all input runs")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output folder is not empty; use a new folder")
    root.mkdir(parents=True, exist_ok=True)
    grpo.save_json(root/"config.json", vars(args))
    protocol = {"audit_probability_on_accepted":args.audit_probability, "audit_seed":args.audit_seed,
        "initial_adapter_sha256":hashes["initial"], "full_audit_run":str(full),
        "full_audit_adapter_sha256":hashes, "gain_rule":"budget",
        "audit_rng":"independent Python RNG keyed by seed, round, rollout slot; paired across arms",
        "audit_gradient":"A*R*(1-c)/rho; leave-one-out baseline; all-rollout denominator",
        "acceptance_gradient":"all verifier rewards; LOO probability gradient, not group-standardized GRPO",
        "label_access":"unaudited correctness is absent from updater inputs; oracle diagnostics are reporting only",
        "training_budget":"fixed rounds; no early stop, checkpoint selection, or forced audit discoveries",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation":"unbiased audit gradient does not imply unbiased normalized correction"}
    grpo.save_json(root/"protocol.json", protocol)
    print(f"AUDIT COVERAGE | seed {args.seed} | {args.audit_probability:.0%} | RAW + PAC | {args.steps} rounds each", flush=True)
    print("Saved 100% baseline stays frozen. Only two new audit policies are trained.", flush=True)
    start = time.monotonic()
    for arm, direction, method in (("raw","raw","RAW AUDIT"),("pac","projected","PAC")):
        branch = copy.copy(args)
        branch.output_dir, branch.defer_summary = str(root/arm), True
        branch.gain_rule, branch.audit_direction = "budget", direction
        update = CoverageUpdate(args.audit_probability, args.audit_seed, root/arm/"audits.jsonl")
        grpo.run(branch, update_fn=update, method=method, extra_provenance={
            **{k:v for k,v in protocol.items() if k != "script_sha256"},
            "projection":"none" if arm == "raw" else "Euclidean orthogonal to estimated grad p",
            "placement":"direct correction after Adam", "gain":"shared norm-budget rule",
            "correction_script_sha256":hashlib.sha256(Path(pac.__file__).read_bytes()).hexdigest(),
            "coverage_script_sha256":protocol["script_sha256"]})
        release_gpu()
    args.length = comparison.read_json(root/"pac"/"config.json")["length"]
    ev = SimpleNamespace(source_run=args.source_run, initial_run=args.init_run, grpo_run=args.grpo_run,
        raw_run=str(root/"raw"), pac_run=str(root/"pac"), output_dir=str(root/"heldout"),
        prompts=args.test_prompts, samples=args.test_samples, eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens, seed=args.test_seed, smoke=args.smoke,
        allow_cpu=args.allow_cpu, defer_summary=True,
        reuse_evaluations_from=None if args.smoke else str(full/"heldout"))
    comparison.run(ev)
    release_gpu()
    if any(comparison.adapter_hashes(path) != hashes[name] for name,path in runs.items()):
        raise RuntimeError("An input checkpoint changed during the experiment")
    print("\n"+summarize(root, args, full, time.monotonic()-start), flush=True)


if __name__ == "__main__":
    run(arguments())
