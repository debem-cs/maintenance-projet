"""Generate exploratory plots of the robot maintenance dataset.

For every training sequence it renders a 6-motor temperature figure with the
injected-failure spans shaded, plus an overview failure matrix. Optionally it
does the same for the testing sequences (which have no ground-truth failures).

Usage (from the repository root)::

    py scripts/plot_data.py                 # training sequences + overview
    py scripts/plot_data.py --split test     # testing sequences too
    py scripts/plot_data.py --split both
    py scripts/plot_data.py --channel voltage
    py scripts/plot_data.py --only 20240503_164675   # a single test id

Figures are written to ``outputs/figures/`` and a run log to ``logs/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# --- Make the src/ package importable without installing -------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_maintenance import config, data, plots  # noqa: E402

log = logging.getLogger("plot_data")


def setup_logging() -> Path:
    """Configure logging to both stdout and a timestamped file in logs/."""
    config.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"plot_data_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return log_file


def plot_split(split: str, channel: str, only: str | None) -> int:
    """Render one figure per test sequence in ``split``. Returns count made."""
    conditions = data.load_test_conditions(split)
    desc_by_id = dict(zip(conditions["Test id"].astype(str), conditions["Description"]))

    test_ids = data.list_test_ids(split)
    if only:
        test_ids = [t for t in test_ids if t == only]
        if not test_ids:
            log.warning("Test id %s not found in split %s", only, split)
            return 0

    out_dir = config.FIGURES_DIR / "sequences" / split
    made = 0
    for test_id in test_ids:
        wide = data.load_wide_test(test_id, split)
        failed = data.failed_motors(conditions, test_id)
        description = str(desc_by_id.get(test_id, "")).strip()
        out_path = out_dir / f"{test_id}_{channel}.png"
        plots.plot_sequence(wide, test_id, description, out_path,
                            channel=channel, failed=failed)
        tag = f"  failed={failed}" if failed else ""
        log.info("[%s] %s -> %s%s", split, test_id, out_path.name, tag)
        made += 1
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "both"], default="train",
                        help="Which data split to plot (default: train).")
    parser.add_argument("--channel", default="temperature",
                        choices=list(config.SIGNAL_COLUMNS),
                        help="Signal to plot per motor (default: temperature).")
    parser.add_argument("--only", default=None,
                        help="Plot only this single test id.")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip the training failure-matrix overview.")
    args = parser.parse_args(argv)

    log_file = setup_logging()
    log.info("Dataset dir: %s", config.DATASET_DIR)
    if not config.DATASET_DIR.exists():
        log.error("Dataset directory not found. Check config.DATASET_DIR.")
        return 1
    log.info("Figures -> %s | log -> %s", config.FIGURES_DIR, log_file)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    total = 0
    for split in splits:
        total += plot_split(split, args.channel, args.only)

    if not args.no_overview and not args.only:
        conditions = data.load_test_conditions("train")
        matrix_path = config.FIGURES_DIR / "overview_failure_matrix.png"
        plots.plot_failure_matrix(conditions, matrix_path)
        log.info("Overview -> %s", matrix_path.name)

    log.info("Done. %d sequence figure(s) written.", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
