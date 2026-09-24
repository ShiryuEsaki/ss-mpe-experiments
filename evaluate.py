#!/usr/bin/env python3
"""Direct command-line entry point for the low-level MPE evaluator."""

from __future__ import annotations

import os
from pathlib import Path
import sys


repository = Path(__file__).resolve().parent
source_root = repository / "src"
upstream = source_root / "ss_nt_mpe_rc"
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(source_root), str(upstream), os.environ.get("PYTHONPATH", "")]
)
sys.path[:0] = [str(source_root), str(upstream)]

from ss_mpe_experiments.evaluator import main


if __name__ == "__main__":
    main()
