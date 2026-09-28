# Neural bandit

Tanh-network policies with exact population gradients and Hessian-vector products.
Run from this folder:

```bash
python -m pip install -r requirements.txt
python neural_bandit_control.py --n-seeds 10 --gain-mode fixed --output runs/control
python neural_bandit_appendix.py --experiment growth --output runs/growth
```

`neural_bandit_control.py` compares verifier, regularization, raw-audit, and
projected-audit flows. `neural_bandit_appendix.py` supports `growth`, `feedback`,
`coverage`, and `projection` through `--experiment`.

Key arguments: `--output`, `--seed`, `--n-seeds`, `--hidden-width`, `--p0`,
`--q0`, and `--horizon`. Audit controls use `--lambda`; gradient regularization
uses `--gamma`. Coverage and projection ablations use `--coverage`,
`--batch-sizes`, and `--repetitions`. Use separate folders for different runs.

For a short execution check:

```bash
python neural_bandit_control.py --n-seeds 1 --hidden-width 2 --n-prompts 1 --n-per-group 2 --horizon 0.02 --n-times 3 --output runs/smoke
```
