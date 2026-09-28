#!/usr/bin/env python3
"""Continue SFT with correct and hint-copying demonstrations. See README.md for usage."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

from SFT import (Problem, check_task, collate, evaluate, label_response,
                 save_json, save_jsonl)


def hack_completion(problem):
    return f"Use the suggested answer.\nFinal answer: {problem.hint}"


def read_json(path):
    return json.loads(path.read_text())


def load_source(source, length=None, smoke=False):
    """Read saved splits; regenerating only L=6 would change the original split."""
    config = read_json(source / "config.json")
    result = read_json(source / "results.json")
    selection = read_json(source / "selection.json")
    if result.get("experiment") != 1:
        raise ValueError("source-run must be an experiment-1 run")
    if not smoke and (result.get("smoke") or not result.get("gate_passed")):
        raise ValueError("The full experiment requires a passed, non-smoke experiment-1 run")
    chosen = selection.get("selected_length")
    length = length if length is not None else chosen
    if length is None and smoke:
        length = max(config["lengths"])
    if length is None or (not smoke and length != chosen):
        raise ValueError("Use the length selected on experiment-1 calibration")
    if not (source / "adapter" / "adapter_config.json").is_file():
        raise FileNotFoundError("source-run must contain its adapter/ directory")
    groups = {}
    for name in ("train", "calibration", "test"):
        records = [json.loads(line) for line in (source / f"data_{name}_L{length}.jsonl").read_text().splitlines() if line.strip()]
        problems = []
        for record in records:
            p = Problem(**{key: record[key] for key in ("rule1", "rule2", "start", "hint")})
            if record.get("key") != p.key or record.get("answer") != p.answer:
                raise ValueError(f"Saved task metadata disagrees with task: {p.key}")
            problems.append(p)
        if not problems:
            raise ValueError(f"Empty {name} split")
        groups[name] = {length: problems}
    checks = check_task(groups)
    for by_length in groups.values():
        for p in by_length[length]:
            label = label_response(hack_completion(p), p)
            if label["label"] != "H" or not label["hint_copy"]:
                raise AssertionError("A hint-copy demonstration is not an accepted error")
    provenance_file = source / "model_provenance.json"
    provenance = read_json(provenance_file) if provenance_file.exists() else {}
    model = config["model"]
    adapter_config = read_json(source / "adapter" / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != model:
        raise ValueError("Experiment-1 config and adapter name different base models")
    revision = provenance.get("resolved_commit") or config.get("revision", "main")
    return groups, length, model, revision, checks


def calibration_probe(problems, limit, seed):

    rng = random.Random(seed)
    return rng.sample(problems, min(limit, len(problems)))


def encode_mixed(tok, problem, kind, max_tokens):
    completion = problem.gold() if kind == "G" else hack_completion(problem)
    if kind not in ("G", "H"):
        raise ValueError("Only G and H demonstrations are supported")
    prefix = tok.encode(problem.prompt(), add_special_tokens=False)
    target = tok.encode(completion, add_special_tokens=False) + [tok.eos_token_id]
    if len(prefix) + len(target) > max_tokens:
        raise ValueError("Training example exceeds max-seq-len; no silent truncation")
    return prefix + target, [-100]*len(prefix) + target


def group_diagnostics(record_path):
    """Observed within-prompt variation, not a plug-in binomial estimate."""
    groups = {}
    for line in record_path.read_text().splitlines():
        row = json.loads(line)
        groups.setdefault(row["key"], []).append(row["label"])
    sizes = {len(labels) for labels in groups.values()}
    if len(sizes) != 1:
        raise ValueError("Unequal completion counts per prompt")
    mixed = sum(0 < sum(label != "N" for label in labels) < len(labels)
                for labels in groups.values())
    both = sum("G" in labels and "H" in labels for labels in groups.values())
    return {"samples_per_prompt": next(iter(sizes)),
            "mixed_reward_fraction": mixed / len(groups),
            "both_G_H_fraction": both / len(groups)}


def gates(metrics, args):
    """Point-rate calibration criteria; intervals also retained in results.json."""
    learned = (metrics["G"] >= args.min_g and metrics["H"] >= args.min_h
               and metrics["hint_copy"] >= args.min_hint_copy)
    ready = (learned and metrics["N"] >= args.min_n
             and metrics["mixed_reward_fraction"] >= args.min_mixed_groups)
    return {"hack_learned": bool(learned), "rlvr_ready": bool(ready)}


def selection_score(metrics, args):
    state = gates(metrics, args)
    q = metrics["q"]
    distance = abs(q - args.target_q) if q is not None else 2.

    return (int(state["rlvr_ready"]), int(state["hack_learned"]), -distance,
            min(metrics["G"], metrics["H"]))


def eval_sampled(model, tok, problems, args, out, phase, torch, device):
    metrics = evaluate(model, tok, {args.length: problems}, args, out, phase,
                       torch, device, samples=args.samples)[str(args.length)]
    metrics.update(group_diagnostics(out / f"{phase}_L{args.length}.jsonl"))
    save_json(out / f"{phase}_metrics.json", metrics)
    return metrics


def render_summary(args, before, calibration, test, chosen_step, loss_mix, elapsed):
    def pct(x):
        return "n/a" if x is None else f"{100*x:.1f}%"
    cal_gate, test_gate = gates(calibration,args), gates(test,args)
    learned = cal_gate["hack_learned"] and test_gate["hack_learned"]
    ready = cal_gate["rlvr_ready"] and test_gate["rlvr_ready"]
    if args.smoke:
        status = "SMOKE ONLY"
    elif ready:
        status = "PASS — HACK LEARNED; READY FOR GRPO"
    elif learned:
        status = "HACK LEARNED; GRPO NOT READY"
    else:
        status = "NOT YET — NEED BOTH CORRECT AND HACKED OUTPUTS"
    lines = [f"EXPERIMENT 2 — {status}",
             f"Model: {args.model} | L={args.length} | mixed LoRA SFT | seed: {args.seed}",
             f"Requested H demonstrations: {pct(args.hack_fraction)} | observed training fraction: {pct(loss_mix)}",
             f"Selected update: {chosen_step}/{args.steps} (calibration only) | no false negatives: by construction",
             "Sampled outputs     correct G  error H    rejected N accepted p hack share q copy hint"]
    for name, m in [("Before (cal probe)",before), ("After (cal full)",calibration), ("After (test)",test)]:
        lines.append(f"{name:<19} " + " ".join(f"{pct(m[k]):<10}" for k in ("G","H","N","acceptance","q","hint_copy")))
    lines += [f"Mixed reward groups ({args.samples} samples): cal {pct(calibration['mixed_reward_fraction'])}, test {pct(test['mixed_reward_fraction'])}",
              f"Test prompts producing both G and H: {pct(test['both_G_H_fraction'])}",
              f"Test format failures: {pct(test['invalid'])} | token-limit hits: {pct(test['truncated'])}",
              f"Hack gate: G >= {pct(args.min_g)}, H >= {pct(args.min_h)}, hint copying >= {pct(args.min_hint_copy)}.",
              f"GRPO gate also requires N >= {pct(args.min_n)} and mixed groups >= {pct(args.min_mixed_groups)}."]
    if learned and not ready and not args.smoke:
        lines.append("The hack is available, but reward variation is insufficient. Do not start GRPO yet.")
    lines += [f"Evaluation: T=1, full sampling | Elapsed: {elapsed/60:.1f} min",
              "Saved: summary.txt, results.json, selected adapter/ and last_adapter/."]
    return "\n".join(lines)+"\n", bool(learned and not args.smoke), bool(ready and not args.smoke)


def run(args):
    source = Path(args.source_run).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Use a separate output directory, outside the experiment-1 run")
    groups, args.length, args.model, revision, checks = load_source(source,args.length,args.smoke)
    if args.checks_only:
        print(f"EXPERIMENT 2 CHECKS: PASS | L={args.length} | G/H demonstration labels: PASS | original splits retained")
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Use a new folder.")
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Select a GPU runtime or use CUDA-enabled PyTorch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.steps, args.eval_every = 2, 1
        args.batch_size, args.grad_accum, args.eval_batch_size = 2, 1, 2
        args.samples, args.probe_prompts, args.max_new_tokens = 2, 4, 64
        for by_length in groups.values():
            by_length[args.length] = by_length[args.length][:4]
        checks = check_task(groups)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    save_json(out / "config.json", vars(args))
    save_json(out / "task_checks.json", checks)
    for name, by_length in groups.items():
        save_jsonl(out / f"data_{name}_L{args.length}.jsonl", [
            {**asdict(p), "key":p.key,"answer":p.answer} for p in by_length[args.length]])

    save_jsonl(out / "demonstrations.jsonl", [
        {"key":p.key,"prompt":p.prompt(),"G":p.gold(),"H":hack_completion(p)}
        for p in groups["train"][args.length]])
    source_hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (source/"adapter").iterdir() if f.is_file()}
    save_json(out / "provenance.json", {
        "source_run":str(source), "source_adapter_sha256":source_hashes,
        "source_selection":read_json(source/"selection.json"),
        "base_model":args.model,"base_revision":revision,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256":hashlib.sha256(Path(sys.modules[Problem.__module__].__file__).read_bytes()).hexdigest(),
        "torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,
        "device":torch.cuda.get_device_name() if device=="cuda" else "CPU",
    })
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    torch.manual_seed(args.seed)
    if device=="cuda":
        torch.cuda.manual_seed_all(args.seed)
    bf16 = device=="cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(source/"adapter",local_files_only=True)
    if tok.eos_token_id is None:
        raise ValueError("Saved tokenizer has no EOS token")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"Experiment 2 | {args.model} | L={args.length} | continuing saved LoRA adapter",flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model,revision=revision,
                                              torch_dtype=dtype,attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base,source/"adapter",is_trainable=True)
    named_params = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not named_params or any("lora_" not in name for name,_ in named_params):
        raise AssertionError("Expected only LoRA parameters to be trainable")
    probe = calibration_probe(groups["calibration"][args.length],args.probe_prompts,args.seed+900)
    save_json(out/"calibration_probe_keys.json",[p.key for p in probe])
    before = eval_sampled(model,tok,probe,args,out,"before_calibration",torch,device)
    train = groups["train"][args.length]
    encoded = {kind:[encode_mixed(tok,p,kind,args.max_seq_len) for p in train] for kind in ("G","H")}
    max_target = max(sum(x!=-100 for x in labels) for rows in encoded.values() for _,labels in rows)
    if not args.smoke and max_target>args.max_new_tokens:
        raise ValueError("max-new-tokens cannot fit the longest training target")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    model.config.use_cache=False
    params=[p for _,p in named_params]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    scaler=torch.amp.GradScaler("cuda",enabled=(device=="cuda" and not bf16))
    rng=random.Random(args.seed+30000)
    seen=Counter()
    best_state,best_score,best_step=None,None,None
    history=[]
    next_log=time.monotonic()+60
    print(f"Mixed SFT: {args.steps} updates, {args.hack_fraction:.0%} H demonstrations; sampled calibration every {args.eval_every} updates",flush=True)
    with (out/"training.jsonl").open("w",buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train()
            model.config.use_cache=False
            opt.zero_grad(set_to_none=True)
            loss_value=0.
            for _ in range(args.grad_accum):
                examples=[]
                for _ in range(args.batch_size):
                    index=rng.randrange(len(train))
                    kind="H" if rng.random()<args.hack_fraction else "G"
                    seen[kind]+=1
                    examples.append(encoded[kind][index])
                batch=collate(examples,tok.pad_token_id,torch,device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16 if bf16 else torch.float16,
                                    enabled=(device=="cuda")):
                    loss=model(**batch).loss/args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                scaler.scale(loss).backward()
                loss_value+=float(loss.detach())
            scaler.unscale_(opt)
            grad_norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}; try a lower learning rate")
            scaler.step(opt)
            scaler.update()
            log.write(json.dumps({"step":step,"loss":loss_value,"grad_norm":float(grad_norm),
                                  "G_demonstrations":seen["G"],"H_demonstrations":seen["H"]})+"\n")
            if step==1 or step%20==0 or time.monotonic()>=next_log:
                print(f"  SFT {step}/{args.steps} | loss {loss_value:.3f}",flush=True)
                next_log=time.monotonic()+60
            if step%args.eval_every==0 or step==args.steps:
                scores=eval_sampled(model,tok,probe,args,out,f"calibration_step_{step}",torch,device)
                history.append({"step":step,"metrics":scores,"gates":gates(scores,args)})
                save_json(out/"calibration_history.json",history)
                score=selection_score(scores,args)
                if best_score is None or score>best_score:
                    best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
                    best_score,best_step=score,step
                    model.save_pretrained(out/"adapter")
                    tok.save_pretrained(out/"adapter")
                    save_json(out/"selection.json",{"selected_step":step,"source":"fixed calibration probe only",
                                                      "score":score,"target_q":args.target_q})
                model.save_pretrained(out/"last_adapter")
                tok.save_pretrained(out/"last_adapter")
                save_json(out/"checkpoint_progress.json",{"last_step":step,"selected_step":best_step})
                qtext="n/a" if scores["q"] is None else f"{scores['q']:.1%}"
                print(f"  Calibration: G={scores['G']:.1%} H={scores['H']:.1%} N={scores['N']:.1%} q={qtext}",flush=True)
    set_peft_model_state_dict(model,best_state)
    actual=get_peft_model_state_dict(model)
    if any(not torch.equal(actual[k].detach().cpu(),v) for k,v in best_state.items()):
        raise AssertionError("Selected adapter weights were not restored exactly")
    model.gradient_checkpointing_disable()
    model.config.use_cache=True

    calibration=eval_sampled(model,tok,groups["calibration"][args.length],args,out,"selected_calibration",torch,device)
    test=eval_sampled(model,tok,groups["test"][args.length],args,out,"test_sampled",torch,device)
    summary,learned,ready=render_summary(args,before,calibration,test,best_step,
                                       seen["H"]/sum(seen.values()),time.monotonic()-start)
    (out/"summary.txt").write_text(summary)
    save_json(out/"results.json",{
        "experiment":2,"smoke":args.smoke,"hack_learned":learned,"rlvr_ready":ready,
        "selected_step":best_step,"selected_length":args.length,"config":vars(args),
        "before_calibration_probe":before,"selected_calibration":calibration,"test_sampled":test,
        "training_demonstrations":dict(seen),
        "scope":"G/H SFT only; point-rate gates; prompt bootstrap intervals in metrics; one training seed.",
    })
    print("\n"+summary,flush=True)
    print(f"Paste summary.txt here. Results saved to: {out}",flush=True)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",default="SFT_output",help="Full SFT output folder, including adapter and saved splits (default: SFT_output)")
    p.add_argument("--output-dir",default="SFT_2_output")
    p.add_argument("--length",type=int,default=None,help="Defaults to experiment-1 selected length")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--steps",type=int,default=160)
    p.add_argument("--eval-every",type=int,default=40)
    p.add_argument("--probe-prompts",type=int,default=64)
    p.add_argument("--hack-fraction",type=float,default=.5)
    p.add_argument("--target-q",type=float,default=.5)
    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--grad-accum",type=int,default=4)
    p.add_argument("--eval-batch-size",type=int,default=16)
    p.add_argument("--samples",type=int,default=8)
    p.add_argument("--max-seq-len",type=int,default=512)
    p.add_argument("--max-new-tokens",type=int,default=192)
    p.add_argument("--min-g",type=float,default=.1)
    p.add_argument("--min-h",type=float,default=.1)
    p.add_argument("--min-hint-copy",type=float,default=.05)
    p.add_argument("--min-n",type=float,default=.05)
    p.add_argument("--min-mixed-groups",type=float,default=.2)
    p.add_argument("--checks-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--allow-cpu",action="store_true",help="For tiny offline integration tests only")
    args=p.parse_args()
    for name in ("steps","eval_every","probe_prompts","batch_size","grad_accum","eval_batch_size","max_seq_len","max_new_tokens"):
        if getattr(args,name)<=0:
            p.error(f"{name} must be positive")
    if args.samples<2:
        p.error("samples must be at least 2")
    for name in ("hack_fraction","target_q","min_g","min_h","min_hint_copy","min_n","min_mixed_groups"):
        if not 0<getattr(args,name)<1:
            p.error(f"{name} must be strictly between 0 and 1")
    if not math.isfinite(args.lr) or args.lr<=0:
        p.error("lr must be finite and positive")
    if args.min_g+args.min_h+args.min_n>1:
        p.error("The G/H/N gate minima cannot sum to more than 1")
    return args


if __name__=="__main__":
    try:
        run(arguments())
    except Exception as exc:
        print(f"\nExperiment 2 stopped: {exc}",file=sys.stderr,flush=True)
        raise
