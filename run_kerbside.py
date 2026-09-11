"""Start the console without installing the package.

`pip install -e .` is the normal route, but a machine where the interpreter's
Scripts directory is not writable should not be a machine where this project
cannot be run. This adds backend/ to sys.path and hands over to the real entry
point, so `python run_kerbside.py` works from a clean checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from kerbside.serve import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
