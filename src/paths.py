"""Repository-local path definitions.

No machine-specific absolute paths are used in the public analysis scripts.
Large upstream inputs belong under data/raw or data/processed and generated
numerical outputs under results/rpe1.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results" / "rpe1"
FIGURES_DIR = ROOT / "figures" / "rpe1"

for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, description: str = "required input") -> Path:
    """Raise a readable error when an external input has not been installed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "See data/README.md for the expected external inputs."
        )
    return path
