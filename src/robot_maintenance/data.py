"""Data loading helpers for the robot predictive-maintenance dataset.

The competition stores one folder per *test sequence*; inside each folder there
are six ``data_motor_k.csv`` files (k = 1..6) sharing the same length. This
module turns those raw files into tidy in-memory structures:

* :func:`read_motor_csv`      -- one motor file -> DataFrame (+ relative time).
* :func:`load_wide_test`      -- merge the six motors of one test into a *wide*
                                 frame with ``data_motor_k_<signal>`` columns,
                                 mirroring the competition's ``utility.py``.
* :func:`load_test_conditions`-- read the ``Test conditions.xlsx`` spreadsheet.
* :func:`failed_motors`       -- which motors were faulted in a given test.
* :func:`list_test_ids`       -- discover the test folders on disk.
* :func:`fault_spans`         -- contiguous label==1 runs, for plotting/shading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Raw columns present in every per-motor CSV.
_RAW_COLUMNS = ("time", "position", "temperature", "voltage", "label")


def list_test_ids(split: str) -> list[str]:
    """Return the sorted test-id folder names for a split ('train'/'test')."""
    base = config.split_dir(split)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _relative_time(time: pd.Series, n: int) -> np.ndarray:
    """Normalise the absolute wall-clock time to start at zero.

    Mirrors ``utility.read_all_csvs_one_test``: subtract the first value, and
    if that fails (or yields nonsense) fall back to a synthetic 0.1 s grid.
    """
    try:
        rel = time.to_numpy(dtype=float) - float(time.iloc[0])
        if not np.all(np.isfinite(rel)):
            raise ValueError
        return rel
    except Exception:  # noqa: BLE001 -- any parsing issue -> synthetic grid
        return np.arange(n, dtype=float) * config.SAMPLE_PERIOD_S


def read_motor_csv(path: str | Path, clean: bool = False) -> pd.DataFrame:
    """Read a single ``data_motor_k.csv`` and add a relative ``time_s`` column.

    The original (absolute) ``time`` column is kept untouched; ``time_s`` is the
    seconds-since-start axis used for plotting. With ``clean=True`` the signal
    channels are repaired via :func:`robot_maintenance.preprocessing.clean_motor_df`
    (Tier 1 outlier repair); the row count and the ``label`` column are preserved.
    """
    df = pd.read_csv(path)
    df["time_s"] = _relative_time(df["time"], len(df))
    if clean:
        from .preprocessing import clean_motor_df  # local import avoids cycle
        df = clean_motor_df(df)
    return df


def load_motor_test(test_id: str, split: str, clean: bool = False) -> dict[int, pd.DataFrame]:
    """Load all six motor frames of one test as ``{motor_index: DataFrame}``.

    ``clean=True`` applies Tier 1 outlier repair to each motor frame.
    """
    base = config.split_dir(split) / test_id
    out: dict[int, pd.DataFrame] = {}
    for k in config.MOTORS:
        path = base / f"data_motor_{k}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing motor file: {path}")
        out[k] = read_motor_csv(path, clean=clean)
    return out


def load_wide_test(test_id: str, split: str, clean: bool = False) -> pd.DataFrame:
    """Merge the six motors of one test into a single wide DataFrame.

    Columns follow the competition convention used in ``utility.py``::

        time_s,
        data_motor_1_position, data_motor_1_temperature,
        data_motor_1_voltage,  data_motor_1_label,
        ... data_motor_6_label,
        test_condition

    All six files share the same length, so a positional concatenation is safe.
    The relative time axis is taken from motor 1. ``clean=True`` applies Tier 1
    outlier repair to the signal channels before merging.
    """
    motors = load_motor_test(test_id, split, clean=clean)

    pieces = []
    for k, df in motors.items():
        keep = [c for c in ("position", "temperature", "voltage", "label") if c in df.columns]
        renamed = df[keep].add_prefix(f"data_motor_{k}_")
        renamed = renamed.reset_index(drop=True)
        pieces.append(renamed)

    wide = pd.concat(pieces, axis=1)
    wide.insert(0, "time_s", motors[1]["time_s"].reset_index(drop=True))
    wide["test_condition"] = test_id
    return wide


def load_test_conditions(split: str) -> pd.DataFrame:
    """Read the ``Test conditions.xlsx`` spreadsheet for a split.

    Training has ``Motor_k_failure`` columns (1.0 / NaN); testing has only
    ``Test id`` and ``Description``.
    """
    path = config.split_dir(split) / config.CONDITIONS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Conditions spreadsheet not found: {path}")
    df = pd.read_excel(path)
    df["Test id"] = df["Test id"].astype(str)
    return df


def failed_motors(conditions: pd.DataFrame, test_id: str) -> list[int]:
    """Return the list of motor indices flagged as failed for ``test_id``.

    Returns ``[]`` when the test has no failure columns (testing split) or no
    motor was faulted.
    """
    row = conditions.loc[conditions["Test id"] == str(test_id)]
    if row.empty:
        return []
    failed = []
    for k in config.MOTORS:
        col = f"Motor_{k}_failure"
        if col in conditions.columns and float(row[col].iloc[0] or 0) == 1.0:
            failed.append(k)
    return failed


def fault_spans(labels: pd.Series | np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs of contiguous label==1 runs.

    ``end`` is inclusive. Useful for shading failure regions on a plot.
    """
    # Compare in float space first so NaN (e.g. the placeholder testing labels)
    # is treated as "not a fault"; NaN == 1 is False, no risky int cast.
    arr = np.asarray(labels, dtype=float)
    if arr.size == 0:
        return []
    mask = (arr == 1).astype(int)
    # Pad with zeros on both sides to catch runs that touch the edges.
    padded = np.concatenate(([0], mask, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))
