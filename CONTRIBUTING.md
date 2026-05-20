# Contributing to wandb skills

1. Create `skills/<your-skill>/SKILL.md` with frontmatter and instructions.
2. Open a PR. CI validates public package shape, frontmatter, helper parsing,
   and basic sanitization.
3. Review the skill benchmark plan attached to your PR. Maintainers can run a
   live benchmark when a change needs behavioral evidence.
4. Merge to `main` to publish.

## Skill benchmarks

Skill benchmarks compare the current public skill on `main` against the skill
content in your PR on the W&B Agent Factory task list selected for that skill.

On PRs, the workflow runs in plan-only mode when the private WBAF checkout token
is available. Plan-only mode does not run model calls. It reports:

- The changed skill.
- The WBAF suite and scenarios that would run.
- The harness that would execute the benchmark.
- The secrets required for a trusted live run.

Maintainers can trigger a live benchmark with the `Skill bench` workflow's
manual dispatch. Live benchmarks produce a base-versus-candidate report with
improved, regressed, unchanged, missing, and must-pass regression counts.

The benchmark report is the source of truth for PR review. Badges, if restored
later, should summarize trusted mainline runs only and should not replace PR
reports.

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
