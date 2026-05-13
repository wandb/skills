# Contributing to wandb skills

1. Create `skills/<your-skill>/SKILL.md` with frontmatter and instructions.
2. Open a PR. CI validates public package shape, frontmatter, helper parsing,
   and basic sanitization.
3. Attach W&B Agent Factory benchmark evidence for behavioral changes.
4. Merge to `main` to publish.

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
