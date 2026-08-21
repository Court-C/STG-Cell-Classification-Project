# STG Cell Classification Project

Feature extraction and classification pipeline for a population of fitted,
conductance-based single-compartment models of stomatogastric ganglion (STG)
neurons. Each cell is characterized by its response to a grid of holding and
injected current levels: tonic/bursting/silent classification, post-inhibitory
rebound, sag depth, spike-frequency adaptation, and firing-rate/current slope.
Cells are then clustered on these features.

This README covers environment setup and the command sequence to go from a
cell's parameter file through feature extraction to clustering.

## Environment setup

On macOS/Linux, from the project root:

```bash
./setup_env.sh
```

Creates `.venv` if it doesn't already exist and installs `requirements.txt`
into it (`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `pytest`,
`joblib`, `numba`). Safe to re-run to pick up new dependencies. Activate with
`source .venv/bin/activate`, or just run scripts directly via
`.venv/bin/python`.

On Windows:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

All commands below assume `.venv/Scripts/python.exe` (or the equivalent
`.venv/bin/python` on macOS/Linux) as the interpreter, run from the project
root.

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

All of these caches are already committed at the project root for the current
69-cell population, so **stages 0-5 do not need to be rerun** unless you are
changing the underlying model, the classification logic, or adding cells.

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

Merges grid features with other measurement tracks (intrinsic properties)
into one `master_features.csv`, one row per cell. Always runs over the full
population -- feature dropping and coverage filtering downstream are
population-relative (see stage 5).

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

## Running the tests

```bash
python -m pytest tests/ -q
```

Synthetic ground-truth tests for the spike-detection and burst/tonic
classification functions in `find_silencing_threshold.py` and
`run_held_injected_grid.py` -- fed fabricated interval sequences with a known
correct answer, including several real-cell edge cases from this project's
own data.

## Diagnostic tools

Stage 3's per-cell grid-features heatmap (`figures/grid_features/{cell}_grid.png`,
14 panels) sometimes shows a parameter that doesn't vary smoothly across the
grid the way you'd expect, and the 2D heatmap format itself isn't always the
most intuitive way to read a result. Three ad hoc tools help make sense of a
heatmap rather than just trusting it:

```bash
python src/plot_example_traces.py --cells XB2IQX
```

Picks one representative (held, injected) point per distinct
(firing pattern, rebound pattern) combination observed in a cell's grid and
plots its resimulated trace -- see the module docstring. Defaults to a
curated 6-cell set (`DEFAULT_CURATED_CELLS`) spanning the range of behaviors
in the population.

```bash
python src/plot_parameter_trace.py --cell XB2IQX --parameter firing_rate --point -0.37,-2.02
```

Point -> trace: resimulates and plots the trace for one specific (held,
injected) coordinate you've spotted on a specific parameter's heatmap, next
to that heatmap with the point marked -- for explaining a single surprising
pixel by eye. `--parameter` accepts any of the 14 panel keys (see
`PARAMETER_PANELS` in `run_held_injected_grid.py`); `--point` is repeatable.

```bash
python src/plot_held_slice.py --cell XB2IQX --parameter firing_rate --held 0.0
```

F/I-style slice curve: plots a parameter against injected current along one
or more fixed held-current rows, next to the heatmap with those rows marked
-- a more familiar 1D supplement to the 2D heatmap. No resimulation, reads
straight from the grid/features caches. `--held` is repeatable to overlay
several rows.

## Scope note

Day-to-day development and testing of classification changes is scoped to a
curated 6-cell working set (`XB2IQX` plus five cells chosen to span the range
of behaviors in the population -- see `plot_example_traces.py`'s
`DEFAULT_CURATED_CELLS`), not the full 69-cell population. Re-running any
stage above without `--cells` reprocesses everyone, which is a much longer
job -- do this deliberately, not as a default.
