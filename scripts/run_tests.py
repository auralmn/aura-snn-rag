"""
Convenience test runner for peer review.

Runs the repository test suite under repo/tests with PYTHONPATH set to the repo root.
Usage:
    python scripts/run_tests.py
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    cmd = [sys.executable, "-m", "pytest", "repo/tests"]
    print("Running:", " ".join(cmd))
    print("PYTHONPATH:", env["PYTHONPATH"])
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
