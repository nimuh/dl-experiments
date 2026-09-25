"""Make `dna` (and its sibling `transformer`) importable from anywhere."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE):  # dl-experiments/, dl-experiments/dna/tests/
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
