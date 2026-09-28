# Gaussian bandit

Exact population flows for acceptance, hacking, and audit control.
Run from this folder:

```bash
python -m pip install -r requirements.txt
python bandit_growth.py --output runs/growth
python bandit_information.py --output runs/information
python bandit_control.py --n-seeds 10 --gain-mode both --output runs/control
python bandit_growth_appendix.py --experiment mechanisms --output runs/growth_appendix
python bandit_control_appendix.py --experiment control --output runs/control_appendix
python bandit_audit_appendix.py --experiment control --output runs/audit_appendix
```

| Script | Purpose / experiment selectors |
| --- | --- |
| `bandit_growth.py` | Acceptance and hacking trajectories |
| `bandit_information.py` | Shared verifier trajectory under hidden correctness rules |
| `bandit_control.py` | Verifier, regularization, raw audit, and projected audit |
| `bandit_growth_appendix.py` | `mechanisms`, `geometry`, `steps` |
| `bandit_control_appendix.py` | `control`, `regularization` |
| `bandit_audit_appendix.py` | `control`, `coverage`, `projection`, `iss` |

Key arguments: `--output` sets the run folder; `--seed` selects the feature seed;
`--n-seeds` sets repetitions where supported; `--p0` and `--q0` set initial
acceptance and hacked share; `--horizon` sets flow duration. Controls use
`--lambda` for audit gain and `--gamma` for gradient regularization. Appendix
coverage and projection experiments use `--coverage`, `--batch-sizes`, and
`--repetitions`. Use a separate output folder for each run.

For a short execution check:

```bash
python bandit_growth.py --horizon 0.02 --n-times 3 --output runs/smoke
```
