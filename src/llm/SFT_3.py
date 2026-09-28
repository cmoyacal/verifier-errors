#!/usr/bin/env python3
"""Continue SFT with correct, hacked, and valid rejected demonstrations. See README.md for usage."""
import argparse
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

from SFT import Problem, check_task, collate, evaluate, label_response, save_json, save_jsonl


def completion(problem, kind):
    if kind == "G":
        return problem.gold()
    if kind == "H":
        return f"Use the suggested answer.\nFinal answer: {problem.hint}"
    if kind != "N":
        raise ValueError("Expected G, H, or N")

    replacements = [problem.rules[d] for d in problem.start]
    last = replacements[-1]
    replacements[-1] = last[:-1] + ("2" if last[-1] == "1" else "1")
    steps = [f"Position {i}: {digit} -> {value}"
             for i, (digit, value) in enumerate(zip(problem.start, replacements), 1)]
    return "\n".join(steps) + f"\nFinal answer: {''.join(replacements)}"


def read_split(path):
    problems = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            p = Problem(**{k: row[k] for k in ("rule1", "rule2", "start", "hint")})
            if row.get("key") != p.key or row.get("answer") != p.answer:
                raise ValueError(f"Inconsistent saved task: {p.key}")
            problems.append(p)
    if not problems:
        raise ValueError(f"Empty split: {path}")
    return problems


def probe_details(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["key"]].append(row)
    valid_n = sum(r["label"] == "N" and r["valid_format"] for r in records)
    valid_mixed = sum(any(r["accepted"] for r in group)
                      and any(r["label"] == "N" and r["valid_format"] for r in group)
                      for group in groups.values())
    mixed = sum(any(r["accepted"] for r in group) and any(not r["accepted"] for r in group)
                for group in groups.values())
    return {"valid_N_count": valid_n, "valid_N_rate": valid_n / len(records),
            "valid_mixed_groups": valid_mixed, "mixed_reward_groups": mixed,
            "n_groups": len(groups)}


def candidate(metrics, details):

    return (metrics["G"] >= .1 and metrics["H"] >= .1
            and .1 <= details["valid_N_rate"] <= .4
            and metrics["invalid"] <= .05
            and details["valid_mixed_groups"] / details["n_groups"] >= .2)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", default="SFT_output")
    p.add_argument("--init-run", default="SFT_2_output")
    p.add_argument("--output-dir", default="SFT_3_output")
    p.add_argument("--mix", type=float, nargs=3, default=[.4, .4, .2], metavar=("G", "H", "N"))
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--probe-prompts", type=int, default=16)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    for name in ("steps", "batch_size", "grad_accum", "eval_batch_size", "probe_prompts", "max_new_tokens", "max_seq_len"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.samples < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("samples must be >= 2 and lr must be finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in args.mix) or not math.isclose(sum(args.mix), 1., abs_tol=1e-8):
        p.error("--mix must contain three positive fractions summing to 1")
    if args.smoke:
        args.steps, args.batch_size, args.grad_accum = 2, 2, 1
        args.probe_prompts, args.samples, args.eval_batch_size = 4, 2, 4

    return args


def run(args):
    read = lambda path: json.loads(path.read_text())
    source, init, out = [Path(v).expanduser().resolve() for v in (args.source_run, args.init_run, args.output_dir)]
    for original in (source, init):
        if original == out or original in out.parents or out in original.parents:
            raise ValueError("Use a new output folder outside both input runs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty; use a new folder")
    source_config, init_config = read(source / "config.json"), read(init / "config.json")
    source_result = read(source / "results.json")
    if not args.smoke and (not source_result.get("gate_passed") or source_config.get("smoke") or init_config.get("smoke")):
        raise ValueError("Full training requires completed SFT and non-smoke initial weights")
    args.model, args.length = source_config["model"], int(init_config["length"])
    if init_config["model"] != args.model:
        raise ValueError("Initial adapter and SFT source name different models")
    adapter_config = read(init / "adapter" / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != args.model:
        raise ValueError("Adapter base model does not match source config")
    groups = {}
    for name in ("train", "calibration"):
        filename = f"data_{name}_L{args.length}.jsonl"
        data = read_split(source / filename)
        if read_split(init / filename) != data:
            raise ValueError("Initial checkpoint's tasks differ from the supplied SFT source")
        groups[name] = {args.length: data}
    checks = check_task(groups)
    for splits in groups.values():
        for problem in splits[args.length]:
            for kind in ("G", "H", "N"):
                observed = label_response(completion(problem, kind), problem)
                if observed["label"] != kind or not observed["valid_format"]:
                    raise AssertionError(f"Incorrect {kind} demonstration")
    train = groups["train"][args.length]
    calibration = groups["calibration"][args.length]
    probe = random.Random(args.seed + 900).sample(calibration, min(args.probe_prompts, len(calibration)))
    provenance = read(source / "model_provenance.json") if (source / "model_provenance.json").exists() else {}
    revision = provenance.get("resolved_commit") or source_config.get("revision", "main")

    os.environ.setdefault("USE_TF", "0")
    import torch
    import transformers
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
    tok = AutoTokenizer.from_pretrained(init / "adapter", local_files_only=True)
    if tok.eos_token_id is None:
        raise ValueError("Missing EOS token")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=revision,
               dtype=torch.bfloat16 if bf16 else torch.float32, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base, init / "adapter", is_trainable=True)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in name for name, _ in named):
        raise AssertionError("Only LoRA weights should train")
    encoded = {kind: [] for kind in ("G", "H", "N")}
    for problem in train:
        prefix = tok.encode(problem.prompt(), add_special_tokens=False)
        for kind in encoded:
            target = tok.encode(completion(problem, kind), add_special_tokens=False) + [tok.eos_token_id]
            if len(prefix) + len(target) > args.max_seq_len or len(target) > args.max_new_tokens:
                raise ValueError("Token budget cannot fit a complete training target; increase it")
            encoded[kind].append((prefix + target, [-100] * len(prefix) + target))
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", vars(args))
    save_json(out / "task_checks.json", checks)
    save_json(out / "provenance.json", {
        "source_run": str(source), "init_run": str(init), "base_revision": revision,
        "init_adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (init / "adapter").glob("adapter*") if p.is_file()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(sys.modules[Problem.__module__].__file__).read_bytes()).hexdigest(),
        "evaluation": "One fixed calibration probe; no test evaluation or checkpoint selection",
    })
    for name, splits in groups.items():
        save_jsonl(out / f"data_{name}_L{args.length}.jsonl",
                   [{**asdict(p), "key": p.key, "answer": p.answer} for p in splits[args.length]])
    save_json(out / "calibration_probe_keys.json", [p.key for p in probe])
    save_jsonl(out / "demonstrations.jsonl", [{"key": p.key, "prompt": p.prompt(),
               **{kind: completion(p, kind) for kind in encoded}} for p in train])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not bf16)
    rng, seen = random.Random(args.seed + 30000), Counter()
    print(f"SFT 3 | {args.steps} updates | G/H/N mix: {args.mix} | ONE final probe: {len(probe)} × {args.samples}", flush=True)
    with (out / "training.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for _ in range(args.grad_accum):
                examples = []
                for _ in range(args.batch_size):
                    kind = (["G", "H", "N"][sum(seen.values())]
                            if args.smoke and sum(seen.values()) < 3
                            else rng.choices(["G", "H", "N"], weights=args.mix, k=1)[0])
                    seen[kind] += 1
                    examples.append(encoded[kind][rng.randrange(len(train))])
                with torch.autocast(device_type=device, dtype=torch.bfloat16 if bf16 else torch.float16,
                                    enabled=device == "cuda"):
                    loss = model(**collate(examples, tok.pad_token_id, torch, device)).loss / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                loss_value += float(loss.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            scaler.step(optimizer)
            scaler.update()
            log.write(json.dumps({"step": step, "loss": loss_value, "seen": dict(seen)}) + "\n")
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(f"  SFT {step}/{args.steps} | loss {loss_value:.3f}", flush=True)
            if step % 20 == 0 or step == args.steps:
                model.save_pretrained(out / "adapter")
                tok.save_pretrained(out / "adapter")
                save_json(out / "checkpoint_progress.json", {"last_step": step})
    save_json(out / "selection.json", {"selected_step": args.steps, "source": "fixed training budget"})
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    evaluation_start = time.monotonic()
    print("Training saved. Running the single calibration probe at T=1...", flush=True)

    with torch.autocast(device_type=device, dtype=torch.bfloat16 if bf16 else torch.float16,
                        enabled=device == "cuda"):
        metrics = evaluate(model, tok, {args.length: probe}, args, out, "quick_calibration",
                           torch, device, samples=args.samples)[str(args.length)]
    records = [json.loads(line) for line in (out / f"quick_calibration_L{args.length}.jsonl").read_text().splitlines()]
    details = probe_details(records)
    promising = candidate(metrics, details) and not args.smoke
    status = "SMOKE ONLY" if args.smoke else ("PROMISING INITIALIZATION" if promising else "NEEDS CALIBRATION")
    counts, total = metrics["counts"], metrics["n_completions"]
    q = "n/a" if metrics["q"] is None else f"{metrics['q']:.1%}"
    summary = "\n".join([
        f"SFT 3 — {status}",
        f"{args.model} | L={args.length} | steps={args.steps} | seed={args.seed}",
        "Demo mix G/H/N: " + "/".join(f"{v:.0%}" for v in args.mix)
        + " | observed: " + "/".join(f"{seen[k]/sum(seen.values()):.1%}" for k in ("G", "H", "N")),
        f"Calibration only: {len(probe)} prompts × {args.samples} samples | T=1",
        f"G={counts.get('G',0)}/{total} ({metrics['G']:.1%}) | H={counts.get('H',0)}/{total} ({metrics['H']:.1%})"
        f" | valid N={details['valid_N_count']}/{total} ({details['valid_N_rate']:.1%})",
        f"p={metrics['acceptance']:.1%} | q={q} | copy hint={metrics['hint_copy']:.1%}",
        f"Groups mixing accepted + valid N: {details['valid_mixed_groups']}/{len(probe)}",
        f"Format errors={counts.get('invalid',0)}/{total} | token hits={counts.get('truncated',0)}/{total}",
        f"Probe: {(time.monotonic()-evaluation_start)/60:.1f} min | Total: {(time.monotonic()-started)/60:.1f} min",
        "Small diagnostic only; GRPO readiness not certified. Saved: adapter/, summary.txt, results.json",
    ]) + "\n"
    (out / "summary.txt").write_text(summary)
    save_json(out / "results.json", {"experiment": 3, "smoke": args.smoke, "candidate": promising,
               "rlvr_ready": False, "selected_step": args.steps, "selected_length": args.length,
               "probe": metrics, "details": details, "training_demonstrations": dict(seen),
               "scope": "Controlled G/H/N SFT initialization; one calibration probe only"})
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    run(arguments())
