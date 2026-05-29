"""Plotting helpers: motor signals over time with injected failures shaded.

All functions are *pure* in the sense that they take already-loaded data and a
target output path, render with the non-interactive Agg backend, and return the
saved path. Orchestration (which tests to plot, logging) lives in
``scripts/plot_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files, no display needed.

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .data import fault_spans

# Colour used to shade the injected-failure spans (label == 1).
_FAULT_COLOR = "#d62728"
_FAULT_ALPHA = 0.18


def _shade_faults(ax, time_s: np.ndarray, labels, label_once: bool = True) -> bool:
    """Shade every contiguous failure span on ``ax``. Returns True if any."""
    spans = fault_spans(labels)
    first = True
    for start, end in spans:
        # Extend the right edge to the next sample so single-point spans show.
        x0 = time_s[start]
        x1 = time_s[min(end + 1, len(time_s) - 1)]
        ax.axvspan(
            x0,
            x1,
            color=_FAULT_COLOR,
            alpha=_FAULT_ALPHA,
            label="failure (label=1)" if (first and label_once) else None,
        )
        first = False
    return bool(spans)


def plot_sequence(
    wide,
    test_id: str,
    description: str,
    out_path: str | Path,
    channel: str = "temperature",
    failed: list[int] | None = None,
) -> Path:
    """One figure per test: ``channel`` of all six motors, failures shaded.

    Each motor gets its own row sharing the time axis. The injected-failure
    spans (``data_motor_k_label == 1``) are shaded in red on the corresponding
    motor's panel.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed = failed or []

    time_s = wide["time_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        config.N_MOTORS, 1, figsize=(12, 11), sharex=True, constrained_layout=True
    )

    for k, ax in zip(config.MOTORS, axes):
        sig_col = f"data_motor_{k}_{channel}"
        lab_col = f"data_motor_{k}_label"

        ax.plot(time_s, wide[sig_col].to_numpy(), color="#1f77b4", lw=1.0)

        has_fault = False
        if lab_col in wide.columns:
            has_fault = _shade_faults(ax, time_s, wide[lab_col].to_numpy())

        tag = "  [FAULT INJECTED]" if k in failed or has_fault else ""
        ax.set_ylabel(f"Motor {k}\n{channel}", fontsize=9)
        ax.set_title(f"Motor {k}{tag}", loc="left", fontsize=9,
                     color=_FAULT_COLOR if tag else "black")
        ax.grid(True, alpha=0.3)
        if has_fault:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time since start (s)")

    failed_txt = ", ".join(map(str, failed)) if failed else "none"
    fig.suptitle(
        f"{test_id}  —  {description}\nFailed motors: {failed_txt}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_motor_all_channels(
    wide,
    test_id: str,
    motor: int,
    out_path: str | Path,
    description: str = "",
) -> Path:
    """Detailed view of one motor: position, temperature, voltage stacked.

    Failure spans are shaded across all three channels so you can see how the
    injected temperature rise relates to the (unaffected) position/voltage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    time_s = wide["time_s"].to_numpy(dtype=float)
    lab_col = f"data_motor_{motor}_label"
    labels = wide[lab_col].to_numpy() if lab_col in wide.columns else None

    fig, axes = plt.subplots(
        len(config.SIGNAL_COLUMNS), 1, figsize=(12, 7), sharex=True,
        constrained_layout=True,
    )
    for channel, ax in zip(config.SIGNAL_COLUMNS, axes):
        col = f"data_motor_{motor}_{channel}"
        ax.plot(time_s, wide[col].to_numpy(), color="#1f77b4", lw=1.0)
        if labels is not None:
            _shade_faults(ax, time_s, labels)
        ax.set_ylabel(channel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time since start (s)")

    suffix = f"  —  {description}" if description else ""
    fig.suptitle(f"{test_id} · Motor {motor}{suffix}", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cleaning_diagnostic(
    raw_wide,
    clean_wide,
    test_id: str,
    motor: int,
    out_path: str | Path,
    ranges: dict | None = None,
) -> Path:
    """Before/after view of one motor's three channels (Tier 1 cleaning).

    Left column: raw signal with out-of-range samples marked red. Right column:
    the repaired signal. One row per channel (position/temperature/voltage).
    """
    from .preprocessing import DEFAULT_VALID_RANGES, out_of_range_mask

    ranges = ranges or DEFAULT_VALID_RANGES
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    time_s = raw_wide["time_s"].to_numpy(dtype=float)
    channels = list(config.SIGNAL_COLUMNS)

    lab_col = f"data_motor_{motor}_label"
    labels = raw_wide[lab_col].to_numpy() if lab_col in raw_wide.columns else None

    fig, axes = plt.subplots(len(channels), 2, figsize=(15, 9),
                             sharex=True, constrained_layout=True)
    for row, channel in enumerate(channels):
        col = f"data_motor_{motor}_{channel}"
        lo, hi = ranges.get(channel, (float("-inf"), float("inf")))
        raw = raw_wide[col].to_numpy(dtype=float)
        cleaned = clean_wide[col].to_numpy(dtype=float)
        oob = out_of_range_mask(raw_wide[col], lo, hi).to_numpy()

        ax_raw, ax_clean = axes[row]
        ax_raw.plot(time_s, raw, color="#1f77b4", lw=0.8)
        ax_raw.scatter(time_s[oob], raw[oob], color=_FAULT_COLOR, s=18, zorder=5,
                       label=f"out-of-range ({int(oob.sum())})")
        ax_raw.set_ylabel(channel)
        ax_raw.grid(True, alpha=0.3)

        ax_clean.plot(time_s, cleaned, color="#2ca02c", lw=0.8)
        ax_clean.grid(True, alpha=0.3)

        # Shade the injected-failure spans on both panels (skip on test data,
        # where labels are placeholders and fault_spans returns nothing).
        if labels is not None:
            _shade_faults(ax_raw, time_s, labels)
            _shade_faults(ax_clean, time_s, labels)

        # One combined legend per panel when there is anything to label.
        for ax in (ax_raw, ax_clean):
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="upper right", fontsize=8)

        if row == 0:
            ax_raw.set_title("RAW", loc="left", fontsize=11, fontweight="bold")
            ax_clean.set_title("CLEANED (Tier 1)", loc="left", fontsize=11,
                               fontweight="bold")

    for ax in axes[-1]:
        ax.set_xlabel("time since start (s)")
    fig.suptitle(f"Tier 1 cleaning diagnostic — {test_id} · Motor {motor}",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_failure_matrix(conditions, out_path: str | Path) -> Path:
    """Overview heatmap: which motor failed in which training sequence."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_ids = conditions["Test id"].astype(str).tolist()
    matrix = np.zeros((config.N_MOTORS, len(test_ids)))
    for j, _ in enumerate(test_ids):
        for i, k in enumerate(config.MOTORS):
            col = f"Motor_{k}_failure"
            if col in conditions.columns:
                val = conditions[col].iloc[j]
                matrix[i, j] = 1.0 if float(val or 0) == 1.0 else 0.0

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)

    ax.set_xticks(range(len(test_ids)))
    ax.set_xticklabels(test_ids, rotation=90, fontsize=8)
    ax.set_yticks(range(config.N_MOTORS))
    ax.set_yticklabels([f"Motor {k}" for k in config.MOTORS])
    ax.set_title("Injected failures per training sequence (red = failure)",
                 fontweight="bold")

    # Grid lines between cells for readability.
    ax.set_xticks(np.arange(-0.5, len(test_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, config.N_MOTORS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
