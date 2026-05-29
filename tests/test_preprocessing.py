"""Tests for Tier 1 outlier-repair preprocessing."""

import numpy as np
import pandas as pd
import pytest

from robot_maintenance import config, data, preprocessing


def test_isolated_spike_is_interpolated():
    # A single 255 C spike between two 30s should be repaired to ~30.
    s = pd.Series([30, 30, 255, 30, 30], dtype=float)
    out = preprocessing.clean_channel(s, 0, 100)
    assert out.iloc[2] == pytest.approx(30.0)
    assert len(out) == len(s)  # row count preserved


def test_linear_interpolation_value():
    # Spike between 20 and 40 should interpolate to the midpoint 30.
    s = pd.Series([20, 999, 40], dtype=float)
    out = preprocessing.clean_channel(s, 0, 100)
    assert out.iloc[1] == pytest.approx(30.0)


def test_leading_and_trailing_glitch_filled():
    # Edge glitches can't be interpolated -> held to nearest valid value.
    s = pd.Series([255, 30, 31, 255], dtype=float)
    out = preprocessing.clean_channel(s, 0, 100)
    assert out.notna().all()
    assert out.iloc[0] == pytest.approx(30.0)
    assert out.iloc[-1] == pytest.approx(31.0)


def test_all_outputs_within_range():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.integers(30, 50, size=200).astype(float))
    s.iloc[50] = 255
    s.iloc[120] = -100
    out = preprocessing.clean_channel(s, 0, 100)
    assert ((out >= 0) & (out <= 100)).all()


def test_voltage_band_clips_low_band_and_negatives():
    # Voltage range is (6000, 9000): both 5800 (low band) and -25000 are repaired.
    lo, hi = preprocessing.DEFAULT_VALID_RANGES["voltage"]
    assert (lo, hi) == (6000, 9000)
    s = pd.Series([7000, 5800, -25000, 7200], dtype=float)
    out = preprocessing.clean_channel(s, lo, hi)
    # Two consecutive bad samples between 7000 and 7200 -> linear ramp.
    assert out.iloc[1] == pytest.approx(7000 + (7200 - 7000) / 3)
    assert out.iloc[2] == pytest.approx(7000 + 2 * (7200 - 7000) / 3)
    assert ((out >= lo) & (out <= hi)).all()


def test_clean_motor_df_preserves_label_and_time():
    df = pd.DataFrame({
        "time": [100.0, 100.1, 100.2],
        "position": [10, -32000, 12],
        "temperature": [30, 255, 31],
        "voltage": [7000, -25000, 7001],
        "label": [0, 1, 1],
        "time_s": [0.0, 0.1, 0.2],
    })
    out = preprocessing.clean_motor_df(df)
    # Untouched columns are identical.
    pd.testing.assert_series_equal(out["label"], df["label"])
    pd.testing.assert_series_equal(out["time"], df["time"])
    pd.testing.assert_series_equal(out["time_s"], df["time_s"])
    # Signal glitches repaired.
    assert out["temperature"].iloc[1] == pytest.approx(30.5)
    assert out["position"].iloc[1] == pytest.approx(11.0)


def test_count_repairs():
    df = pd.DataFrame({
        "position": [10, -32000, 12],
        "temperature": [30, 255, 254],
        "voltage": [7000, 7001, 7002],
    })
    counts = preprocessing.count_repairs(df)
    assert counts == {"temperature": 2, "voltage": 0, "position": 1}


# --- Integration with the loader (skips if dataset absent) -----------------

@pytest.mark.skipif(not config.TRAINING_DIR.exists(), reason="dataset not present")
def test_loader_clean_flag_preserves_shape_and_labels():
    raw = data.load_wide_test("20240325_155003", "train", clean=False)
    clean = data.load_wide_test("20240325_155003", "train", clean=True)
    assert raw.shape == clean.shape
    # Labels untouched by cleaning.
    for k in config.MOTORS:
        col = f"data_motor_{k}_label"
        pd.testing.assert_series_equal(raw[col], clean[col])
    # No temperature value above the physical ceiling after cleaning.
    for k in config.MOTORS:
        assert clean[f"data_motor_{k}_temperature"].max() <= 100
