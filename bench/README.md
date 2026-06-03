# Skill Bench

Skill Bench lives in this repository because it evaluates public skill package
changes. WBAF remains the eval runtime: it owns task definitions, agent
profiles, sandbox execution, and the `factory.run_eval` row contract.

Local plan-only check:

```bash
python3 -m skillbench.cli plan \
  --wbaf-root ../WandBAgentFactory \
  --candidate-ref HEAD \
  --skill wandb-primary
```

Live runs require maintainer approval and benchmark secrets. Do not run live
benchmarks from untrusted PR workflow code.
