"""Tier 1 preprocessing: repair isolated sensor glitches in place.

The raw signals contain isolated, physically-impossible spikes (temperature
jumps to ~255 C, voltage to ~-25000 mV, position to ~-32000) caused by
transmission / integer-overflow errors. They never co-occur across channels, so
each channel is cleaned independently.

Strategy (per channel):
    1. mask values outside the physical range -> NaN;
    2. linear-interpolate the NaNs (glitches are isolated single samples, so
       interpolation reconstructs them almost exactly);
    3. forward/back-fill any NaN left at the very first/last sample.

Crucially this **preserves row count and order** -- a hard requirement because
the competition is scored per row and the submission must stay aligned with the
test sequences. The ``label``, ``time`` and ``time_s`` columns are never touched.

All three channels use a fixed physical band; any sample outside it is treated
as a glitch, masked, and interpolated. The voltage band ``(6000, 9000)`` matches
the competition's own ``utility.remove_outliers`` demo, so the 5777-5999 mV
readings are clipped along with the catastrophic negatives.
"""

from __future__ import annotations

import pandas as pd

from . import config

# (low, high) inclusive physical bounds. Anything outside is treated as a glitch.
DEFAULT_VALID_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (0, 100),     # legit values top out ~75 C; spikes start at 167
    "voltage": (6000, 9000),     # demo band: clips negatives and the low band
    "position": (0, 1000),       # servo range; spikes reach int16 underflow
}


def out_of_range_mask(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Boolean mask of samples outside ``[lo, hi]`` (or non-numeric/NaN)."""
    v = pd.to_numeric(series, errors="coerce")
    return (v < lo) | (v > hi) | v.isna()


def clean_channel(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Repair one channel: out-of-range -> NaN -> linear interpolate -> fill edges.

    Returns a float Series of the same length and index as the input.
    """
    v = pd.to_numeric(series, errors="coerce").astype(float)
    v = v.mask((v < lo) | (v > hi))  # out-of-range -> NaN
    v = v.interpolate(method="linear", limit_direction="both")
    # Linear interpolation can't extrapolate past the first/last valid point;
    # hold the nearest valid value there so no NaN survives.
    v = v.ffill().bfill()
    return v


def clean_motor_df(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Return a copy of a per-motor frame with its signal channels repaired.

    Only the columns present in ``ranges`` (default the three signal channels)
    are cleaned; everything else (``time``, ``time_s``, ``label``) is copied
    through untouched.
    """
    ranges = ranges or DEFAULT_VALID_RANGES
    out = df.copy()
    for channel, (lo, hi) in ranges.items():
        if channel in out.columns:
            out[channel] = clean_channel(out[channel], lo, hi)
    return out


def clean_wide_df(
    wide: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Repair the ``data_motor_k_<channel>`` columns of a wide frame in place.

    Useful when you already hold a merged wide frame; the loaders in
    :mod:`robot_maintenance.data` can instead clean at the per-motor level via
    their ``clean=True`` flag, which yields the same result.
    """
    ranges = ranges or DEFAULT_VALID_RANGES
    out = wide.copy()
    for k in config.MOTORS:
        for channel, (lo, hi) in ranges.items():
            col = f"data_motor_{k}_{channel}"
            if col in out.columns:
                out[col] = clean_channel(out[col], lo, hi)
    return out


def count_repairs(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, int]:
    """Count how many samples would be repaired per channel in a per-motor frame."""
    ranges = ranges or DEFAULT_VALID_RANGES
    return {
        channel: int(out_of_range_mask(df[channel], lo, hi).sum())
        for channel, (lo, hi) in ranges.items()
        if channel in df.columns
    }
