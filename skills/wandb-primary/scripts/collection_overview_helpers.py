# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Public wrapper for collection_overview_helpers.py."""

from collection_overview_helpers_impl import *  # noqa: F403
from collection_overview_helpers_impl import main

if __name__ == "__main__":
    raise SystemExit(main())
