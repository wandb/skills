# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Public wrapper for weave_monitor_ops.py."""

import sys

from weave_monitor_ops_impl import *  # noqa: F403
from weave_monitor_ops_impl import main

if __name__ == "__main__":
    sys.exit(main())
