#!/usr/bin/env python3
"""Shared checkpoint validation and held-out evaluation for audit experiments."""
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

from SFT import Problem, check_task, evaluate, save_json, save_jsonl


def read_json(path):
    return json.loads(path.read_text())


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


def adapter_hashes(run):
    adapter = run/"adapter"
    names = ["adapter_config.json"]
    weights = [name for name in ("adapter_model.safetensors", "adapter_model.bin") if (adapter/name).is_file()]
    if len(weights) != 1:
        raise ValueError(f"Expected one adapter weight file: {adapter}")
    return {name: hashlib.sha256((adapter/name).read_bytes()).hexdigest() for name in names+weights}


def validate_inputs(source, runs, smoke):
    configs = {name: read_json(run/"config.json") for name, run in runs.items()}
    results = {name: read_json(run/"results.json") for name, run in runs.items()}
    expected = {"initial": 3, "grpo": "GRPO pilot", "pac": "PAC pilot", "raw": "RAW AUDIT pilot"}
    for name, result in results.items():
        if result.get("experiment") != expected[name]:
            raise ValueError(f"{name}: wrong run type; expected {expected[name]}")
        if not smoke and (result.get("smoke") or configs[name].get("smoke")):
            raise ValueError(f"{name}: use the completed non-smoke checkpoint")
    source_config = read_json(source/"config.json")
    length, model = int(configs["initial"]["length"]), configs["initial"]["model"]
    if not smoke and not read_json(source/"results.json").get("gate_passed"):
        raise ValueError("Source SFT run has not passed its task-learning gate")
    if source_config["model"] != model:
        raise ValueError("Source and checkpoints use different base models")
    groups = {name: {length: read_split(source/f"data_{name}_L{length}.jsonl")}
              for name in ("train", "calibration", "test")}
    checks = check_task(groups)
    hashes = {name: adapter_hashes(run) for name, run in runs.items()}
    provenance = {name: read_json(run/"provenance.json") for name, run in runs.items()}
    revision = provenance["initial"]["base_revision"]
    helper_hash = hashlib.sha256(Path(sys.modules[Problem.__module__].__file__).read_bytes()).hexdigest()
    for name, run in runs.items():
        if int(configs[name]["length"]) != length or configs[name]["model"] != model:
            raise ValueError(f"{name}: different task length or base model")
        if provenance[name].get("base_revision") != revision:
            raise ValueError(f"{name}: different base-model revision")
        if provenance[name].get("helper_sha256") != helper_hash:
            raise ValueError(f"{name}: SFT.py differs from the helper used in training")
        if read_json(run/"adapter"/"adapter_config.json").get("base_model_name_or_path") != model:
            raise ValueError(f"{name}: adapter names a different base model")
        for split in ("train", "calibration"):
            if read_split(run/f"data_{split}_L{length}.jsonl") != groups[split][length]:
                raise ValueError(f"{name}: different saved training/calibration tasks")
        if name != "initial":
            if provenance[name].get("init_adapter_sha256") != hashes["initial"]:
                raise ValueError(f"{name}: did not start from the supplied SFT_3 adapter")
    matched_fields = ("seed", "steps", "prompts_per_step", "group_size", "lr", "max_new_tokens",
                      "no_signal_patience")
    trained = {k: v for k, v in configs.items() if k != "initial"}
    differences = {field: {k: v.get(field) for k, v in trained.items()}
                   for field in matched_fields if len({v.get(field) for v in trained.values()}) > 1}
    if "raw" in configs and "pac" in configs:
        for field in ("gain_rule", "correction_strength", "max_correction_ratio", "probe_seed",
                      "probe_prompts", "samples", "gen_batch_size", "micro_batch_size", "eval_batch_size"):
            if configs["pac"].get(field) != configs["raw"].get(field):
                differences[field] = {k: configs[k].get(field) for k in ("pac", "raw")}
    return groups, length, model, revision, checks, hashes, differences, configs


def select_checkpoint(model, name):
    model.set_adapter(name)
    if model.active_adapter != name:
        raise AssertionError("Failed to activate the requested checkpoint")

    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True


def pool_counts(counts):
    g, h, n = [sum(c[i] for c in counts) for i in range(3)]
    if not n:
        raise ValueError("No completions")
    return {"G": g/n, "H": h/n, "p": (g+h)/n, "q": h/(g+h) if g+h else None}


def paired_differences(records_by_policy, keys, samples, seed, draws=2000, contrasts=None):
    """Resample the SAME prompt indices for every policy; pool counts before q."""
    counts = {}
    for name, records in records_by_policy.items():
        grouped = defaultdict(Counter)
        for row in records:
            grouped[row["key"]][row["label"]] += 1
        if set(grouped) != set(keys) or any(sum(v.values()) != samples for v in grouped.values()):
            raise ValueError("Policies were not evaluated on matching prompt/sample sets")
        counts[name] = [(grouped[k]["G"], grouped[k]["H"], sum(grouped[k].values())) for k in keys]
    if contrasts is None:
        contrasts = {"GRPO minus initial": ("grpo", "initial", ("p", "q")),
                     "PAC minus initial": ("pac", "initial", ("G", "H", "p", "q")),
                     "PAC minus GRPO": ("pac", "grpo", ("G", "H", "p", "q"))}
        if "raw" in counts:
            contrasts.update({"Raw audit minus initial": ("raw", "initial", ("G", "H", "p", "q")),
                              "PAC minus raw audit": ("pac", "raw", ("G", "H", "p", "q"))})
    point = {name: pool_counts(rows) for name, rows in counts.items()}
    draws_by_contrast = {label: defaultdict(list) for label in contrasts}
    rng = random.Random(seed)
    for _ in range(draws):
        indices = [rng.randrange(len(keys)) for _ in keys]
        rates = {name: pool_counts([values[i] for i in indices]) for name, values in counts.items()}
        for label, (a, b, metrics) in contrasts.items():
            for metric in metrics:
                if rates[a][metric] is not None and rates[b][metric] is not None:
                    draws_by_contrast[label][metric].append(rates[a][metric]-rates[b][metric])
    output = {}
    for label, (a, b, metrics) in contrasts.items():
        output[label] = {}
        for metric in metrics:
            values = sorted(draws_by_contrast[label][metric])
            interval = ([values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]]
                        if values else [None, None])
            delta = point[a][metric]-point[b][metric] if point[a][metric] is not None and point[b][metric] is not None else None
            output[label][metric] = {"delta": delta, "ci95": interval, "valid_bootstrap_draws": len(values)}
    return output


def reusable_evaluations(directory, args, hashes, keys):
    """Reuse only frozen initial/GRPO evaluations with identical sampling settings."""
    directory = Path(directory)
    config, prov = read_json(directory/"config.json"), read_json(directory/"provenance.json")
    result = read_json(directory/"results.json")
    for field in ("seed", "prompts", "samples", "eval_batch_size", "max_new_tokens", "length", "model", "smoke"):
        if config.get(field) != getattr(args, field):
            raise ValueError(f"Cached evaluation differs in {field}; use matching settings")
    if result["prompt_keys"] != keys:
        raise ValueError("Cached evaluation uses different test prompts/order")
    cached = {}
    for name in ("initial", "grpo"):
        if prov["adapter_sha256"][name] != hashes[name]:
            raise ValueError(f"Cached {name} checkpoint differs")
        path = directory/f"{name}_L{args.length}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        grouped = defaultdict(list)
        for row in records:
            grouped[row["key"]].append(row)
        if set(grouped) != set(keys) or any(len(v) != args.samples for v in grouped.values()):
            raise ValueError("Cached completion counts do not match")
        measured = pool_counts([(sum(r["label"] == "G" for r in records),
                                 sum(r["label"] == "H" for r in records), len(records))])
        metrics = result["policies"][name]
        for key, value in measured.items():
            if value is None:
                if metrics[key] is not None:
                    raise ValueError("Cached metrics disagree with saved completions")
            elif metrics[key] is None or not math.isclose(value, metrics[key], abs_tol=1e-12):
                raise ValueError("Cached metrics disagree with saved completions")
        cached[name] = {"metrics": metrics, "records": records, "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return cached




def run(args):
    source = Path(args.source_run).expanduser().resolve()
    runs = {name: Path(path).expanduser().resolve() for name, path in
            (("initial", args.initial_run), ("grpo", args.grpo_run), ("pac", args.pac_run))}
    if getattr(args, "raw_run", None):
        runs["raw"] = Path(args.raw_run).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    for original in [source, *runs.values()]:
        if out == original or out in original.parents or original in out.parents:
            raise ValueError("Use a separate output folder outside all input runs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty; choose a new folder")
    groups, args.length, args.model, revision, checks, hashes, differences, configs = validate_inputs(source, runs, args.smoke)
    test = groups["test"][args.length]
    if args.prompts > len(test):
        raise ValueError(f"Only {len(test)} test prompts exist; lower --prompts")
    problems = random.Random(args.seed).sample(test, args.prompts)
    reuse_dir = getattr(args, "reuse_evaluations_from", None)
    cached = reusable_evaluations(reuse_dir, args, hashes, [p.key for p in problems]) if reuse_dir else {}
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
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    torch.manual_seed(args.seed)
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(runs["initial"]/"adapter", local_files_only=True)
    if tok.eos_token_id is None:
        raise ValueError("Missing EOS token")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    check_texts = [text for problem in problems for text in (problem.prompt(), problem.gold())]
    expected_tokens = [tok.encode(text, add_special_tokens=False) for text in check_texts]
    vocabulary = tok.get_vocab()
    for name in (n for n in runs if n != "initial"):
        other = AutoTokenizer.from_pretrained(runs[name]/"adapter", local_files_only=True)
        if other.pad_token_id is None:
            other.pad_token = other.eos_token
        if other.get_vocab() != vocabulary or (other.eos_token_id, other.pad_token_id) != (tok.eos_token_id, tok.pad_token_id):
            raise ValueError(f"{name}: tokenizer vocabulary or special tokens differ")
        if [other.encode(t, add_special_tokens=False) for t in check_texts] != expected_tokens:
            raise ValueError(f"{name}: tokenizer encodes the task differently")
        del other
    del vocabulary
    if max(len(tok.encode(p.gold(), add_special_tokens=False))+1 for p in problems) > args.max_new_tokens:
        raise ValueError("Generation cap cannot fit a correct answer; increase max-new-tokens")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/"config.json", vars(args))
    save_json(out/"task_checks.json", checks)
    save_jsonl(out/"selected_test_prompts.jsonl", [{**asdict(p), "key": p.key, "answer": p.answer} for p in problems])
    save_json(out/"provenance.json", {"base_model": args.model, "base_revision": revision,
        "runs": {k: str(v) for k, v in runs.items()}, "adapter_sha256": hashes,
        "same_initial_adapter_verified": True, "training_setting_differences": differences,
        "training_seeds": {k: v.get("seed") for k, v in configs.items()},
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_evaluations": {k: {f: v[f] for f in ("path", "sha256")} for k, v in cached.items()},
        "protocol": "frozen checkpoints; same test prompts, sampling seed, batch size, temperature and token cap; no checkpoint selection"})
    print(f"HELD-OUT COMPARISON | {args.prompts} test prompts × {args.samples} samples × {len(runs)} policies | T=1", flush=True)
    print(f"Evaluating {len(runs)-len(cached)} policies; reusing {len(cached)} verified frozen evaluations...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=revision,
            dtype=torch.bfloat16 if bf16 else torch.float32, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base, runs["initial"]/"adapter", adapter_name="initial", is_trainable=False)
    for name in (n for n in runs if n != "initial" and n not in cached):
        model.load_adapter(runs[name]/"adapter", adapter_name=name, is_trainable=False, torch_device=device)
    policies, records = {}, {}
    for name in runs:
        if name in cached:
            policies[name], records[name] = cached[name]["metrics"], cached[name]["records"]
            save_jsonl(out/f"{name}_L{args.length}.jsonl", records[name])
            print(f"  {name}: reused verified evaluation", flush=True)
            continue
        select_checkpoint(model, name)
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if bf16 else torch.float16,
                            enabled=device == "cuda"):
            metrics = evaluate(model, tok, {args.length: problems}, args, out, name,
                               torch, device, samples=args.samples)[str(args.length)]
        rows = [json.loads(line) for line in (out/f"{name}_L{args.length}.jsonl").read_text().splitlines()]
        policies[name] = {**metrics, "p": metrics["acceptance"],
                          "valid_N": sum(r["label"] == "N" and r["valid_format"] for r in rows)/len(rows)}
        records[name] = rows
        m = policies[name]
        q = "n/a" if m["q"] is None else f"{m['q']:.1%}"
        print(f"  {name}: G={m['G']:.1%} H={m['H']:.1%} p={m['p']:.1%} q={q}", flush=True)
    if any(adapter_hashes(run) != hashes[name] for name, run in runs.items()):
        raise RuntimeError("An input checkpoint changed during evaluation; keep the input folders frozen")
    contrasts = paired_differences(records, [p.key for p in problems], args.samples, args.seed+50000)
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    summary = ["HELD-OUT COMPARISON" + (" — SMOKE ONLY" if args.smoke else " — FROZEN CHECKPOINTS"),
        f"{args.model} | L={args.length} | {args.prompts} test prompts × {args.samples} samples/policy | T=1",
        "Same initial adapter verified | train/cal/test overlap: 0",
        "Policy     G        H        valid N  p        q        format   token hits"]
    labels = {"initial": "SFT_3", "grpo": "GRPO", "pac": "PAC", "raw": "Raw audit"}
    for name in runs:
        label = labels[name]
        summary.append(f"{label:<10} " + " ".join(f"{pct(policies[name][key]):<8}"
                       for key in ("G", "H", "valid_N", "p", "q", "invalid", "truncated")))
    summary.append("Differences in percentage points [approx. paired prompt-bootstrap 95% interval]:")
    for label in contrasts:
        metrics = ("p", "q") if label == "GRPO minus initial" else ("G", "H")
        if label == "PAC minus raw audit":
            metrics = ("G", "H", "p")
        parts = []
        for metric in metrics:
            item = contrasts[label][metric]
            if item["delta"] is None or item["ci95"][0] is None:
                parts.append(f"{metric}: n/a")
            else:
                lo, hi = item["ci95"]
                parts.append(f"{metric}: {100*item['delta']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        summary.append(f"{label}: " + " | ".join(parts))
    if differences:
        summary.append("Training settings differ; see provenance.json for the exact fields.")
    if "raw" in configs:
        summary.append(f"Audit gain rules: PAC={configs['pac'].get('gain_rule', 'adaptive')}, raw={configs['raw'].get('gain_rule', 'unknown')}.")
    summary += [f"Elapsed: {(time.monotonic()-started)/60:.1f} min | {(len(runs)-len(cached))*args.prompts*args.samples} new completions; {len(cached)*args.prompts*args.samples} reused",
                "Approximate prompt-bootstrap intervals; one training run per method. Saved: summary.txt, results.json, raw completions."]
    text = "\n".join(summary)+"\n"
    (out/"summary.txt").write_text(text)
    save_json(out/"results.json", {"experiment": "held-out checkpoint comparison", "smoke": args.smoke,
        "policies": policies, "paired_contrasts": contrasts, "prompt_keys": [p.key for p in problems],
        "reused_policies": list(cached),
        "training_setting_differences": differences,
        "scope": "existing disjoint test split; fixed small subset; marginal approximate prompt-cluster intervals, not simultaneous bounds or training-seed uncertainty",
        "bootstrap_note": "2000 paired prompt resamples; unobserved events can produce degenerate intervals, not proof of zero population probability"})
    if not getattr(args, "defer_summary", False):
        print("\n"+text, flush=True)
