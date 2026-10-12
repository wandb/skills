# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Public wrapper for automation_helpers.py."""

import sys

from automation_helpers_impl import *  # noqa: F403
from automation_helpers_impl import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
