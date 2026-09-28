# LLM experiments

Synthetic task generation, LoRA training, and audit-control experiments.
Run from this folder with CUDA-enabled PyTorch:

```bash
python -m pip install -r requirements.txt
python SFT.py --output-dir SFT_output
python SFT_2.py --source-run SFT_output --output-dir SFT_2_output
python SFT_3.py --source-run SFT_output --init-run SFT_2_output --output-dir SFT_3_output
python GRPO.py --seed 1 --output-dir GRPO_seed1
python PAC.py --seed 1 --output-dir PAC_seed1
python audit_ablation.py --seed 1 --grpo-run GRPO_seed1 --output-dir audit_seed1
python audit_coverage.py --seed 1 --full-audit-run audit_seed1 --audit-probability 0.5 --output-dir audit50_seed1
```

| File | Purpose |
| --- | --- |
| `SFT.py` | Generate disjoint splits and train correct-only demonstrations |
| `SFT_2.py` | Add hint-copying demonstrations |
| `SFT_3.py` | Train a mixture of correct, hacked, and valid rejected responses |
| `GRPO.py` | Optimize binary verifier reward |
| `PAC.py` | Apply projected audit correction with adaptive bounded gain |
| `audit_ablation.py` | Compare raw/projected directions with a shared norm budget |
| `audit_coverage.py` | Compare partial random audits against a saved full-audit pair |
| `evaluation.py` | Shared validation and held-out evaluation used by audit experiments |

Keep all files together. Downstream stages need the complete earlier run
folders, including splits, configuration, and adapters. SFT must pass its
readiness check before a normal downstream run. Standalone PAC uses adaptive
gain; the audit ablation uses a shared norm-budget rule. These are distinct
experiment settings.

Key arguments: `--model` on SFT selects the base model; `--source-run` selects
the original task/SFT run; `--init-run` selects starting weights; `--output-dir`
selects a new run folder; `--seed`, `--steps`, and `--lr` control training.
SFT_3's `--mix G H N` sets demonstration fractions. GRPO/PAC use
`--prompts-per-step` and `--group-size` for rollout size. Audit experiments use
`--correction-strength`, `--max-correction-ratio`, and `--audit-probability`
(coverage only). Match the training seed across baseline and audit runs.

`python SFT.py --checks-only` checks task setup without downloading a model.
`--smoke` reduces training and evaluation sizes; model-based smoke runs still
need model weights and compatible prior smoke runs.
