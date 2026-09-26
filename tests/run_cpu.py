# SPDX-License-Identifier: Apache-2.0
"""Run the CPU contracts, failing if discovery accidentally finds no tests."""
from pathlib import Path
import unittest


def main():
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "cpu"))
    if suite.countTestCases() == 0:
        raise SystemExit("No CPU tests discovered")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
