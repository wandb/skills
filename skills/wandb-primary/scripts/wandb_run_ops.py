# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""CLI entry point for bounded W&B run, artifact, and workspace inspection."""

from wandb_run_ops_lib.common import *  # noqa: F403
from wandb_run_ops_lib.cli import main

if __name__ == "__main__":
    main()
