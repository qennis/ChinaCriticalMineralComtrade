from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_RAW = REPO_ROOT / "data_raw"
DATA_WORK = REPO_ROOT / "data_work"
FIGURES = REPO_ROOT / "figures"
OUTPUTS = REPO_ROOT / "outputs"
NOTES = REPO_ROOT / "notes"


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_WORK, FIGURES, OUTPUTS):
        p.mkdir(parents=True, exist_ok=True)
