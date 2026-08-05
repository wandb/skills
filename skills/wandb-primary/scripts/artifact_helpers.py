# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Public wrapper for artifact_helpers.py."""

from artifact_helpers_impl import *  # noqa: F403
from artifact_helpers_impl import main

if __name__ == "__main__":
    raise SystemExit(main())
