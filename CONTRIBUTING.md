# Contributing to wandb skills

1. Create `skills/<your-skill>/SKILL.md` with frontmatter and instructions
2. Open a PR — CI will run the skill through the eval suite
3. Merge to `main` to publish

To update an existing installation:

```bash
./install.sh --force
```

## Upstream skills mirrored from wandb/core

Some skills in this repository are mirrored from an internal source of truth
in [wandb/core](https://github.com/wandb/core) via an automated publish
pipeline in [wandb/WandBAgentFactory](https://github.com/wandb/WandBAgentFactory).
These skills are identifiable by a `.publish_manifest.json` present in the
publishing PRs — the manifest records the source commit and the internal
names of each mirrored skill.

If you want to contribute changes to a mirrored skill:

- Prefer opening a PR against the upstream file in
  `core/services/wb_agent/src/agent_repository/context_content/skills/<internal-name>/`.
  Once merged, the change flows here automatically on the next publish cycle.
- Drive-by fixes directly in this repo are still welcome for typos, links,
  and similar. The next automated publish PR will surface any divergence
  for human review, so nothing is silently overwritten.
- Never edit `.publish_manifest.json` manually.

A skill opts into being mirrored by adding a `public:` block to its
SKILL.md frontmatter in core. The publish pipeline never reads from
`souls/`, `system_prompts/`, or any `production/` path in core.

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
