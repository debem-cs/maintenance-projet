# Robot Predictive Maintenance — Season 2026

Solution workspace for the Kaggle competition
[*Robot predictive maintenance — Season 2026*](https://kaggle.com/competitions/robot-predictive-maintenance-season-2026).

The task: for each of the six servo motors of an ArmPi FPV robot, predict at
every time step whether the motor is **normal (0)** or undergoing an **abnormal
temperature rise (1)**. Scoring is the **mean F1 across the six motors**. See
[`briefing/`](briefing/) for the full problem description.

## Project layout

```
projet/
├── README.md                 # this file
├── pyproject.toml            # package metadata + pytest config
├── requirements.txt          # pinned-ish dependencies
├── briefing/                 # competition briefing (LaTeX + PDF)
├── robot-predictive-maintenance-season-2026/   # raw data (never modified)
│   ├── training_data/training_data/<test_id>/data_motor_{1..6}.csv
│   ├── testing_data/testing_data/<test_id>/data_motor_{1..6}.csv
│   ├── sample_submission.csv
│   └── utility.py            # competition-provided helper module
├── src/robot_maintenance/    # reusable library code
│   ├── config.py             # paths + dataset constants
│   ├── data.py               # CSV/XLSX loading, wide frames, fault spans
│   ├── preprocessing.py      # Tier 1 outlier repair (clip -> interpolate)
│   └── plots.py              # signal + failure visualisations
├── scripts/                  # runnable entry points (CLI)
│   ├── plot_data.py          # generate all exploratory figures
│   └── preprocess.py         # repair report + before/after diagnostics
├── tests/                    # pytest suite
│   └── test_data.py
├── notebooks/                # exploratory notebooks (kept out of src)
├── outputs/figures/          # generated plots (git-ignored)
└── logs/                     # timestamped run logs (git-ignored)
```

**Why this shape.** Library code in `src/` is import-only and side-effect free;
anything you actually *run* lives in `scripts/`; throwaway exploration goes in
`notebooks/`; generated artefacts (`outputs/`, `logs/`) are kept out of version
control but their folders are preserved with `.gitkeep`.

## Setup

Requires Python 3.10+. On this machine the launcher is `py`:

```powershell
py -m pip install -r requirements.txt
# (optional) editable install so `import robot_maintenance` works anywhere:
py -m pip install -e .
```

The scripts also self-bootstrap `src/` onto the path, so they run without the
editable install.

## Plotting the data and the failures

```powershell
# One temperature figure per training sequence (failures shaded red)
# plus an overview failure matrix:
py scripts/plot_data.py

# Include the testing sequences (no ground-truth failures):
py scripts/plot_data.py --split both

# A single sequence, or a different channel:
py scripts/plot_data.py --only 20240503_164675
py scripts/plot_data.py --channel voltage
```

Outputs land in `outputs/figures/`:

- `sequences/<split>/<test_id>_<channel>.png` — six stacked panels (one per
  motor) of the chosen channel over time. Spans where `label == 1` are shaded
  red and the panel title is flagged `[FAULT INJECTED]`. The injected fault is
  a triangular temperature pulse — a steep rise over the first ~quarter of the
  span followed by a slower decay.
- `overview_failure_matrix.png` — heatmap of which motor failed in which
  training sequence (motor 6 dominates; two sequences fault all six).

### What the data looks like

- **23 training** sequences, **8 testing** sequences. Each sequence is one
  folder of six `data_motor_k.csv` files (`time, position, temperature,
  voltage, label`), sampled at ~10 Hz, all six the same length.
- Only **7 training sequences** contain an injected fault: 5 are motor-6-only,
  2 fault all six motors. Testing labels are placeholders to be predicted.

## Preprocessing (Tier 1 — outlier repair)

The raw signals contain isolated sensor glitches (temperature spikes to ~255 °C,
voltage to ~−25000 mV, position to ~−32000) that never co-occur across channels.
Tier 1 repairs them **in place** per channel — mask out-of-physical-range →
linear-interpolate → fill edges — which preserves row count/order (required for
per-row scoring) and leaves the `label` column untouched. Defaults
([`preprocessing.py`](src/robot_maintenance/preprocessing.py)):

| channel | valid range | note |
|---|---|---|
| temperature | `[0, 100]` | legit values top out ~75 °C; spikes start at 167 |
| voltage | `[6000, 9000]` | demo band: clips negatives and the 5777–5999 mV low band |
| position | `[0, 1000]` | servo range |

Load cleaned data straight from the loader, or apply it to a wide frame:

```python
from robot_maintenance import data, preprocessing
wide = data.load_wide_test("20240325_155003", "train", clean=True)   # cleaned
wide = preprocessing.clean_wide_df(data.load_wide_test(tid, "train")) # equivalent
```

Generate a repair report + before/after diagnostic figures:

```powershell
py scripts/preprocess.py --split both        # report + diagnostics for repaired motors
py scripts/preprocess.py --only 20240325_155003
py scripts/preprocess.py --no-figures         # report only
```

Diagnostics land in `outputs/figures/diagnostics/<split>/<test_id>_motor<k>.png`.

## Tests

```powershell
py -m pytest -q
```

The disk-backed tests skip automatically if the dataset folder is absent, so
the pure-logic tests still run on a bare checkout.
```
