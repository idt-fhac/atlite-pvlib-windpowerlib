"""Copy result artifacts into text/data/ for the manuscript."""

from __future__ import annotations

import shutil
from pathlib import Path

from utils import PAPER_ROOT

TEXT_DATA = PAPER_ROOT / "text" / "data"


def copy_to_manuscript(src: str | Path) -> Path:
    TEXT_DATA.mkdir(parents=True, exist_ok=True)
    dst = TEXT_DATA / Path(src).name
    shutil.copy2(src, dst)
    print(f"Copied → {dst}")
    return dst
