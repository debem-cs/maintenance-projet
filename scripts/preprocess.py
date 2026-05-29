"""Apply Tier 1 cleaning, report how many samples were repaired, and render
before/after diagnostic figures.

For every sequence it counts the out-of-range samples repaired per channel and
per motor, logs a summary table, and (by default) saves a before/after figure
for each motor that actually had a repair.

Usage (from the repository root)::

    py scripts/preprocess.py                  # train split, diagnostics for repaired motors
    py scripts/preprocess.py --split both
    py scripts/preprocess.py --only 20240325_155003
    py scripts/preprocess.py --no-figures      # just the repair report
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_maintenance import config, data, plots, preprocessing  # noqa: E402

log = logging.getLogger("preprocess")


def setup_logging() -> Path:
    config.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"preprocess_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return log_file


def process_split(split: str, only: str | None, make_figures: bool) -> dict[str, int]:
    test_ids = data.list_test_ids(split)
    if only:
        test_ids = [t for t in test_ids if t == only]

    diag_dir = config.FIGURES_DIR / "diagnostics" / split
    totals: dict[str, int] = defaultdict(int)
    n_figs = 0

    for test_id in test_ids:
        motors_raw = data.load_motor_test(test_id, split, clean=False)
        seq_repairs: dict[str, int] = defaultdict(int)
        motors_with_repair = []

        for k, raw_df in motors_raw.items():
            repairs = preprocessing.count_repairs(raw_df)
            for ch, n in repairs.items():
                seq_repairs[ch] += n
                totals[ch] += n
            if sum(repairs.values()) > 0:
                motors_with_repair.append(k)

        summary = ", ".join(f"{ch}={n}" for ch, n in seq_repairs.items() if n) or "clean"
        log.info("[%s] %s -> repaired: %s", split, test_id, summary)

        if make_figures and motors_with_repair:
            raw_wide = data.load_wide_test(test_id, split, clean=False)
            clean_wide = preprocessing.clean_wide_df(raw_wide)
            for k in motors_with_repair:
                out = diag_dir / f"{test_id}_motor{k}.png"
                plots.plot_cleaning_diagnostic(raw_wide, clean_wide, test_id, k, out)
                n_figs += 1

    log.info("[%s] TOTAL repaired: %s | diagnostics written: %d",
             split, dict(totals), n_figs)
    return dict(totals)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--only", default=None, help="Process only this test id.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the before/after diagnostic figures.")
    args = parser.parse_args(argv)

    log_file = setup_logging()
    log.info("Tier 1 valid ranges: %s", preprocessing.DEFAULT_VALID_RANGES)
    log.info("Log -> %s", log_file)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        process_split(split, args.only, make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
