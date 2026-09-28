# Verifier errors

Core experiment code for verifier-driven learning and projected audit correction.
Use Python 3.11 or newer. Install the requirements for the experiment family
you want to run; the LLM experiments require a CUDA GPU for normal runs.

- `bandit/`: Gaussian contextual-bandit experiments and appendix ablations.
- `neural_bandit/`: neural-bandit experiments and appendix ablations.
- `llm/`: task generation, SFT initialization, GRPO, and audit experiments.

Each folder has short run instructions. Commands use paths relative to the
current directory. Experiment scripts expose their arguments with `--help`.
`llm/evaluation.py` is an internal dependency of the audit experiments.

## Constant and adaptive audit gain

`bandit/bandit_control.py` and `neural_bandit/neural_bandit_control.py` support
constant and adaptive audit gain $\lambda$. Adaptive gain adjusts to the current
gradients to keep the audit correction strong enough to overcome increases in
the probability of accepted errors caused by verifier optimization. A constant
gain also suffices if it exceeds the required threshold throughout training.
These statements concern the exact gradient flow and require a nonzero
correction direction.

| Option | Behavior |
| --- | --- |
| `--gain-mode fixed --lambda 6` | Hold $\lambda=6$ constant |
| `--gain-mode adaptive --adaptive-delta 1` | Recompute the required nonnegative gain threshold and add a margin of 1 |
| `--gain-mode both` | Run both settings |

`llm/PAC.py` adapts the correction to estimated gradients and the actual
optimizer step, then caps its magnitude. Estimation error, finite steps, and
the cap mean that accepted errors need not decrease at every update. The LLM
audit ablations instead use a shared budget for the correction norm; that
budget does not imply a constant $\lambda$.

No checkpoints, results, collection-only utilities, standalone analysis tools,
test suites, or original archive metadata are included.
