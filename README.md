# STG Cell Classification Project

Feature extraction and classification pipeline for a population of fitted,
conductance-based single-compartment models of stomatogastric ganglion (STG)
neurons. Each cell is characterized by its response to a grid of holding and
injected current levels: tonic/bursting/silent classification, post-inhibitory
rebound, sag depth, spike-frequency adaptation, and firing-rate/current slope.
Cells are then clustered on these features.

This README covers environment setup and the command sequence to go from a
cell's parameter file to its **validation packet** -- a PDF report that reruns
every classification algorithm on real simulated traces and marks what it
found directly on the trace, so each result can be checked by eye rather than
taken on faith (see `src/generate_validation_packet.py`).

## Environment setup

A `.venv` is already present at the project root. On Windows:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt  # if you need to reinstall
```

There is no committed `requirements.txt` at time of writing; the core
dependencies are `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, and
`pytest`. All commands below assume `.venv/Scripts/python.exe` (or the
equivalent `.venv/bin/python` on macOS/Linux) as the interpreter, run from the
project root.

## Pipeline overview

Each stage reads the previous stage's cache and writes its own, so any stage
can be rerun in isolation once its inputs exist. All commands below take
`--cells CELLID [CELLID ...]` to scope to specific cells; omit it to run the
full population (69 cells) already present in `src/models/`.

| Stage | Script | Input | Output |
|---|---|---|---|
| 0 | `generate_steady_state.py` | `src/models/{cell}.txt` | `cell_steady_states.pkl` |
| 1 | `find_silencing_threshold.py` | steady-state cache | `cell_silencing_thresholds.pkl` |
| 2 | `run_held_injected_grid.py` | steady-state + silencing caches | `cell_held_injected_grid.pkl` |
| 3 | `extract_grid_features.py` | grid cache | `cell_grid_features.pkl` |
| 4 | `consolidate_features.py` | grid features + other tracks | `master_features.csv` |
| 5 | `cluster_features.py` | `master_features.csv` | `cell_clusters.csv` / `.pkl` |
| 6 | `generate_validation_packet.py` | grid + grid-features + steady-state caches | `figures/validation_packet/{cell}_validation_packet.pdf` |

All of these caches are already committed at the project root for the current
69-cell population, so **stages 0-5 do not need to be rerun** unless you are
changing the underlying model, the classification logic, or adding cells.
Skip straight to stage 6 to regenerate the validation packet from what's
already on disk.

### Stage 0: steady-state cache

```bash
python src/generate_steady_state.py --cells XB2IQX
```

Simulates each cell at zero injected current, burns in, and finds a
self-consistent limit-cycle state via shooting. Every downstream simulation
warm-starts from this cached state rather than from an arbitrary initial
condition. Rerun a specific cell with a smaller `--dt` if its status comes
back anything other than `ok`.

### Stage 1: silencing threshold

```bash
python src/find_silencing_threshold.py --cells XB2IQX
```

Sweeps hyperpolarizing current inward from each cell's steady state to find
where it stops firing (coarse-to-fine continuation, warm-started from the
previous step at each level).

### Stage 2: held x injected current grid

```bash
python src/run_held_injected_grid.py --cells XB2IQX
```

The core simulation stage: for a grid of holding current levels x injected
current steps, classifies the response during the step (tonic / bursting /
silent), the response during the recovery period after release (rebound
pattern), sag depth, and spike-frequency adaptation. This is the stage that
almost all classification logic lives behind -- see the module docstring for
the full list of tunable thresholds (`--min-ashman-d`,
`--trailing-silence-ratio`, etc.), all exposed as CLI flags with their
literature-grounded or empirically-tuned defaults.

### Stage 3: per-cell grid features

```bash
python src/extract_grid_features.py --cells XB2IQX
```

Reduces each cell's full grid of classified points into a handful of scalar
summary features (burstiness index, firing-rate/current slope, mean rebound
latency, etc.) plus a cross-cell PCA.

### Stage 4: consolidate features

```bash
python src/consolidate_features.py
```

Merges grid features with other measurement tracks (intrinsic properties,
phase-response-curve and entrainment results) into one `master_features.csv`,
one row per cell. Always runs over the full population -- feature dropping
and coverage filtering downstream are population-relative (see stage 5).

### Stage 5: clustering

```bash
python src/cluster_features.py
```

Clusters cells on `master_features.csv` (hierarchical, k-means, and Gaussian
mixture, compared for agreement). Uses a hard correlation threshold to prune
redundant features and a coverage threshold to exclude cells with too much
missing data -- both computed across the whole population, so a change to
even one cell's features can occasionally shift which features are retained
for everyone (see `figures/clustering/`).

### Stage 6: validation packet

```bash
python src/generate_validation_packet.py --cells XB2IQX
```

Writes `figures/validation_packet/XB2IQX_validation_packet.pdf` plus
standalone per-section PNGs under `figures/validation_packet/XB2IQX/`. Defaults
to `XB2IQX` if `--cells` is omitted. This is the report to hand someone who
wants to check the classification pipeline is doing the right thing, not just
trust the summary numbers -- every figure reruns the actual production
classification function on a real simulated trace and marks what it found.

## Running the tests

```bash
python -m pytest tests/ -q
```

Synthetic ground-truth tests for the spike-detection and burst/tonic
classification functions in `find_silencing_threshold.py` and
`run_held_injected_grid.py` -- fed fabricated interval sequences with a known
correct answer, including several real-cell edge cases from this project's
own data.

## Scope note

Day-to-day development and testing of classification changes is scoped to a
curated 6-cell working set (`XB2IQX` plus five cells chosen to span the range
of behaviors in the population -- see `plot_example_traces.py`'s
`DEFAULT_CURATED_CELLS`), not the full 69-cell population. Re-running any
stage above without `--cells` reprocesses everyone, which is a much longer
job -- do this deliberately, not as a default.
