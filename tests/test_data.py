"""Tests for the data-loading helpers.

The tests that touch real files are skipped automatically when the dataset is
not present, so the suite still runs on a checkout without the data package.
"""

import numpy as np
import pandas as pd
import pytest

from robot_maintenance import config, data

DATA_AVAILABLE = config.TRAINING_DIR.exists()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="dataset not present")


# --- Pure helpers (no disk access) -----------------------------------------

def test_fault_spans_basic():
    labels = [0, 0, 1, 1, 1, 0, 1, 0]
    assert data.fault_spans(labels) == [(2, 4), (6, 6)]


def test_fault_spans_edges():
    assert data.fault_spans([1, 1, 0, 0, 1]) == [(0, 1), (4, 4)]
    assert data.fault_spans([0, 0, 0]) == []
    assert data.fault_spans([]) == []


def test_relative_time_fallback_is_synthetic_grid():
    # Non-numeric time -> synthetic 0.1 s grid of the right length.
    bad = pd.Series(["a", "b", "c"])
    rel = data._relative_time(bad, len(bad))
    np.testing.assert_allclose(rel, [0.0, 0.1, 0.2])


def test_relative_time_subtracts_first():
    t = pd.Series([100.0, 100.1, 100.2])
    rel = data._relative_time(t, len(t))
    np.testing.assert_allclose(rel, [0.0, 0.1, 0.2], atol=1e-9)


# --- Disk-backed tests ------------------------------------------------------

@needs_data
def test_list_test_ids_nonempty():
    ids = data.list_test_ids("train")
    assert len(ids) == 23
    assert "20240503_164675" in ids


@needs_data
def test_load_wide_test_schema():
    wide = data.load_wide_test("20240503_164675", "train")
    assert "time_s" in wide.columns
    assert wide["test_condition"].iloc[0] == "20240503_164675"
    for k in config.MOTORS:
        for sig in ("position", "temperature", "voltage", "label"):
            assert f"data_motor_{k}_{sig}" in wide.columns
    # Relative time starts at zero.
    assert wide["time_s"].iloc[0] == pytest.approx(0.0)


@needs_data
def test_failed_motors_matches_briefing():
    conditions = data.load_test_conditions("train")
    # Motor-6-only fault sequence from the briefing.
    assert data.failed_motors(conditions, "20240503_164675") == [6]
    # All-motors fault sequence.
    assert data.failed_motors(conditions, "20240426_140055") == [1, 2, 3, 4, 5, 6]
    # Healthy sequence.
    assert data.failed_motors(conditions, "20240105_164214") == []
