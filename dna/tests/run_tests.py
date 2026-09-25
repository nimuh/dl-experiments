"""Run every test in this directory without pytest: `python3 tests/run_tests.py`."""

import importlib
import sys
import traceback
from pathlib import Path

import conftest  # noqa: F401  (puts the package and this directory on sys.path)
import torch

HERE = Path(__file__).resolve().parent


def main() -> int:
    torch.manual_seed(0)
    failures = []

    for path in sorted(HERE.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        for name, fn in sorted(vars(module).items()):
            if not (name.startswith("test_") and callable(fn)):
                continue
            try:
                fn()
                print(f"pass  {path.stem}.{name}")
            except Exception:
                failures.append(f"{path.stem}.{name}")
                print(f"FAIL  {path.stem}.{name}\n{traceback.format_exc()}")

    print(f"\n{len(failures)} failed" if failures else "\nall tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
