# Contributing to wandb skills

1. Create `skills/<your-skill>/SKILL.md` with frontmatter and instructions
2. Open a PR. CI validates the public skill package shape.
3. A maintainer can trigger Skill Bench for larger changes. Live benchmark
   runs are not automatic because they use paid model calls and W&B eval
   credentials.
4. Merge to `main` to publish

## Skill Bench

Skill Bench lives in this repository and evaluates public skill changes against
WBAF eval tasks. WBAF remains the runtime for `factory.run_eval`, task corpora,
agent profiles, and sandbox execution.

Plan a benchmark locally without model calls:

```bash
python3 -m skillbench.cli plan \
  --wbaf-root ../WandBAgentFactory \
  --candidate-ref HEAD \
  --skill wandb-primary
```

Live benchmark runs require maintainer approval and benchmark secrets. Do not
wire live Skill Bench to untrusted PR workflow code.

To update an existing installation:

```bash
./install.sh --force
```

## License headers

<!--- REUSE-IgnoreStart -->

Source code should contain an SPDX-style license header, reflecting:

- Year & Copyright owner
- SPDX License identifier `SPDX-License-Identifier: Apache-2.0`
- Package Name: `SPDX-PackageName: skills`

This can be partially automated with [FSFe REUSE](https://reuse.software/dev/#tool)

```shell
reuse annotate --license Apache-2.0 --copyright 'CoreWeave, Inc.'  --year 2026 --template default_template --skip-existing $FILE
```

Blindly adding the headers to every file without review risks assigning the
wrong copyright owner! You should endeavor to understand who owns
contributions!

- The Skills source are licensed under the Apache-2.0 license to protect the
  rights of all parties.

Licensing state & SPDX bill-of-materials (BOM) can be valiated & generated with:

```shell
reuse lint
reuse spdx
```

<!--- REUSE-IgnoreEnd -->
