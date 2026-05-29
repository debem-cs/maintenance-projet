"""Central configuration: filesystem paths and dataset constants.

Everything that depends on *where* things live on disk is resolved here, so the
rest of the code never hard-codes a path. Paths are derived relative to the
repository root, which keeps the project portable across machines.
"""

from __future__ import annotations

from pathlib import Path

# --- Repository layout -----------------------------------------------------
# config.py lives at <root>/src/robot_maintenance/config.py
ROOT = Path(__file__).resolve().parents[2]

# Raw competition data package (kept exactly as downloaded, never modified).
DATASET_DIR = ROOT / "robot-predictive-maintenance-season-2026"

# The package ships with a doubled folder nesting, e.g.
# training_data/training_data/<test_id>/data_motor_k.csv
TRAINING_DIR = DATASET_DIR / "training_data" / "training_data"
TESTING_DIR = DATASET_DIR / "testing_data" / "testing_data"

SAMPLE_SUBMISSION = DATASET_DIR / "sample_submission.csv"

# --- Generated artefacts ---------------------------------------------------
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

# --- Dataset constants -----------------------------------------------------
N_MOTORS = 6
MOTORS = tuple(range(1, N_MOTORS + 1))

# Channels recorded for every motor (besides the label and time).
SIGNAL_COLUMNS = ("position", "temperature", "voltage")

# Nominal sampling period of the condition-monitoring program (~10 Hz).
SAMPLE_PERIOD_S = 0.1

# Name of the per-test description / failure-flag spreadsheet.
CONDITIONS_FILENAME = "Test conditions.xlsx"


def split_dir(split: str) -> Path:
    """Return the data directory for ``"train"`` or ``"test"``."""
    key = split.lower()
    if key in ("train", "training", "training_data"):
        return TRAINING_DIR
    if key in ("test", "testing", "testing_data"):
        return TESTING_DIR
    raise ValueError(f"Unknown split {split!r}; use 'train' or 'test'.")


def ensure_dirs() -> None:
    """Create the output/log directories if they do not yet exist."""
    for path in (OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
