# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Public wrapper for autoresearch_state.py."""

from autoresearch_state_impl import *  # noqa: F403
from autoresearch_state_impl import _cli

if __name__ == "__main__":
    _cli()
