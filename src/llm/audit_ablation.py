#!/usr/bin/env python3
"""Collect paired raw and projected audit runs with a shared correction-norm rule. See README.md for usage."""
import copy
import gc
import hashlib
import math
from pathlib import Path
import time
from types import SimpleNamespace

import GRPO as grpo
import PAC as pac
import evaluation as comparison


def arguments():
    def extra(parser):
        parser.set_defaults(output_dir=None, probe_prompts=8, probe_seed=1234,
                            probe_rounds=[5, 10])
        parser.add_argument("--grpo-run", default=None,
                            help="Existing baseline; default GRPO_seedN (GRPO_output for seed 0)")
        parser.add_argument("--correction-strength", type=float, default=1.,
                            help="Correction norm / step budget in BOTH arms; no sign-based tuning")
        parser.add_argument("--max-correction-ratio", type=float, default=2.)
        parser.add_argument("--test-prompts", type=int, default=32)
        parser.add_argument("--test-samples", type=int, default=4)
        parser.add_argument("--test-seed", type=int, default=1234)
    args = grpo.arguments(extra_arguments=extra)
    if not math.isfinite(args.correction_strength) or args.correction_strength < 0:
        raise ValueError("correction-strength must be finite and nonnegative")
    if not math.isfinite(args.max_correction_ratio) or args.max_correction_ratio <= 0:
        raise ValueError("max-correction-ratio must be finite and positive")
    if args.test_prompts < 1 or args.test_samples < 2:
        raise ValueError("test-prompts must be positive; test-samples must be >= 2")
    args.output_dir = args.output_dir or f"audit_seed{args.seed}"
    args.grpo_run = args.grpo_run or (f"GRPO_seed{args.seed}" if args.seed else "GRPO_output")

    args.no_signal_patience = args.steps+1
    if args.smoke:
        args.probe_rounds = [1]
        args.test_prompts, args.test_samples = 2, 2
    return args


def release_gpu():

    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compact_summary(root, args, elapsed):
    results = {name: comparison.read_json(root/name/"results.json") for name in ("raw", "pac")}
    lines = ["AUDIT ABLATION" + (" — SMOKE ONLY" if args.smoke else " — PAIRED RUN"),
             f"Seed {args.seed} | same SFT_3 start | KL=0 | fixed {args.steps} rounds/arm",
             "Same correction-norm budget rule; PAC projects, raw audit does not.",
             "This PAC arm uses budget gain; earlier adaptive-gain PAC checkpoints are unchanged.",
             f"Fixed calibration probe: {args.probe_prompts} prompts × {args.samples} samples | T=1",
             "Policy   Round  G        H        p        q        format   token hits"]
    diagnostics = {}
    for name, result in results.items():
        trajectory = result["trajectory"]
        for row in trajectory:
            lines.append(f"{name.upper():<8} {row['round']:<6} " + " ".join(
                f"{grpo.pct(row[key]):<8}" for key in ("G", "H", "p", "q", "invalid", "token_hits")))
        diagnostics[name] = {}
        for key in ("p", "G"):
            low = min(trajectory, key=lambda r: r[key])
            diagnostics[name][key] = {"minimum_observed": low[key], "round": low["round"],
                "change_from_start_at_minimum": low[key]-trajectory[0][key],
                "final_minus_minimum": trajectory[-1][key]-low[key]}
        low_p, low_g = diagnostics[name]["p"], diagnostics[name]["G"]
        lines.append(f"{name.upper()} sampled minima: p={grpo.pct(low_p['minimum_observed'])}"
                     f" @ {low_p['round']}, G={grpo.pct(low_g['minimum_observed'])} @ {low_g['round']}")
    lines.append("Sparse, noisy probes: sampled minima do not establish a population U-turn.")
    for name, result in results.items():
        hist = result["history"]
        lines.append(f"{name.upper()} cost: rollouts={sum(h['n'] for h in hist)}"
                     f" | audits={sum(h.get('audits', 0) for h in hist)}"
                     f" | updates={result['policy_updates']} | Adam steps={result['optimizer_updates']}")
    heldout = (root/"heldout"/"summary.txt").read_text().strip()
    lines += ["", heldout, f"Pair total: {elapsed/60:.1f} min | Saved: {root.name}/summary.txt, results.json"]
    text = "\n".join(lines)+"\n"
    (root/"summary.txt").write_text(text)
    grpo.save_json(root/"results.json", {"experiment": "raw versus projected audit ablation",
        "smoke": args.smoke, "seed": args.seed, "calibration_diagnostics": diagnostics,
        "trajectories": {name: result["trajectory"] for name, result in results.items()},
        "heldout": comparison.read_json(root/"heldout"/"results.json"),
        "elapsed_minutes": elapsed/60,
        "scope": "one training seed; full accepted-response audits; shared norm-budget rule, not identical scalar gains; sparse calibration diagnostics"})
    return text


def run(args):
    root = Path(args.output_dir).expanduser().resolve()
    source, initial, baseline = [Path(p).expanduser().resolve()
                                for p in (args.source_run, args.init_run, args.grpo_run)]
    for path in (source, initial, baseline):
        if root == path or root in path.parents or path in root.parents:
            raise ValueError("Use a new output folder outside the source and checkpoint folders")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output folder is not empty; use a new folder")

    checked = comparison.validate_inputs(source, {"initial": initial, "grpo": baseline}, args.smoke)
    groups, length, _, _, _, hashes, _, configs = checked
    if args.test_prompts > len(groups["test"][length]):
        raise ValueError("test-prompts exceeds the saved test split")
    if args.probe_prompts > len(groups["calibration"][length]):
        raise ValueError("probe-prompts exceeds the saved calibration split")
    if configs["grpo"].get("seed") != args.seed:
        raise ValueError("Use the GRPO baseline with the same training seed")
    root.mkdir(parents=True, exist_ok=True)
    grpo.save_json(root/"config.json", vars(args))
    grpo.save_json(root/"protocol.json", {
        "initial_adapter_sha256": hashes["initial"], "gain_rule": "budget",
        "correction_size": "min(strength, max_ratio) * max(norm(actual Adam displacement), .25*lr*sqrt(trainable parameters))",
        "directions": {"raw": "-h / norm(h)", "pac": "-P_g h / norm(P_g h)"},
        "degeneracy": "zero/nearly degenerate direction is skipped; no forced signal",
        "audit_coverage": "1 on accepted outputs; costs vary with acceptance and are reported",
        "training_budget": "same fixed rounds; no signal-based early stop or checkpoint selection",
        "probes": [0, *args.probe_rounds, args.steps], "probe_seed": args.probe_seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    started = time.monotonic()
    print(f"AUDIT ABLATION | seed {args.seed} | RAW + PAC | {args.steps} rounds each", flush=True)
    print("Shared norm budget, fixed rollout budget; earlier checkpoints remain frozen.", flush=True)
    for name, direction, method in (("raw", "raw", "RAW AUDIT"), ("pac", "projected", "PAC")):
        arm = copy.copy(args)
        arm.output_dir = str(root/name)
        arm.defer_summary = True
        arm.gain_rule, arm.audit_direction = "budget", direction
        grpo.run(arm, update_fn=pac.update_policy, method=method, extra_provenance={
            "correction_script_sha256": hashlib.sha256(Path(pac.__file__).read_bytes()).hexdigest(),
            "ablation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audit_probability_on_accepted": 1.,
            "acceptance_gradient": "LOO probability-gradient estimate; no group-std weighting",
            "audit_gradient": "full accepted-response audits; LOO baseline; all-rollout denominator",
            "projection": "none" if direction == "raw" else "Euclidean projection orthogonal to estimated grad p",
            "placement": "direct parameter correction after Adam",
            "gain": "shared norm-budget rule; no sign-based adaptation; scalar gain depends on direction norm",
            "validation_scope": "estimated geometry and sparse fixed probes; no population monotonicity guarantee"})
        release_gpu()
    if comparison.adapter_hashes(initial) != hashes["initial"] or comparison.adapter_hashes(baseline) != hashes["grpo"]:
        raise RuntimeError("An input checkpoint changed; keep the input folders frozen")
    compare_args = SimpleNamespace(source_run=str(source), initial_run=str(initial),
        grpo_run=str(baseline), pac_run=str(root/"pac"), raw_run=str(root/"raw"),
        output_dir=str(root/"heldout"), prompts=args.test_prompts, samples=args.test_samples,
        eval_batch_size=args.eval_batch_size, max_new_tokens=args.max_new_tokens,
        seed=args.test_seed, smoke=args.smoke, allow_cpu=args.allow_cpu, defer_summary=True)
    comparison.run(compare_args)
    release_gpu()
    print("\n"+compact_summary(root, args, time.monotonic()-started), flush=True)


if __name__ == "__main__":
    run(arguments())
