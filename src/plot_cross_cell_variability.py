"""Cross-cell uniqueness figure -- research question 2 (see the two-question
framing in run_held_injected_grid.py's PARAMETER_PANELS module comment and
project memory): do different cells show meaningfully DIFFERENT behavior at
the SAME current values, i.e. real population heterogeneity, rather than one
cell's grid being representative of all of them?

Every other tool in this toolset (plot_grid_overview.py,
plot_parameter_trace.py, plot_held_slice.py, run_held_injected_grid.py's own
14/16-panel figure) is single-cell (--cell singular) and serves question 1
(is THIS cell's own behavior non-trivial across its own grid). This script
is the first cross-cell one: it puts every "ok" cell's firing-rate map and
bursting-or-not map onto one common (held_frac, injected_frac) grid --
reusing extract_grid_features.py's normalize_grid_to_common_coords, the same
machinery its cross-cell PCA already relies on to make cells with very
different absolute current ranges comparable -- then computes the cross-cell
MEAN and STANDARD DEVIATION at every matched point. A pixel with a small
std is a point where every cell behaves alike regardless of the current
step; a pixel with a large std is exactly the direct, visual evidence for
question 2: cells genuinely diverge from each other there.

No new simulation, no new cache -- reads only the already-computed
cell_held_injected_grid.pkl (for the raw test_pattern used to build the
bursting-or-not map) and cell_grid_features.pkl (for firing_rate_map).
"""

import pickle
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_held_injected_grid import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH
from run_held_injected_grid import load_output_cache as load_grid_cache
from extract_grid_features import (DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_FEATURES_CACHE_PATH,
                                   normalize_grid_to_common_coords)

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures"
DEFAULT_FIGURE_FORMAT = "png"
DEFAULT_GRID_N = 25


def stack_cross_cell(grid_cache: dict, features_cache: dict, grid_n: int) -> dict:
    """Normalizes every ok cell's firing_rate_map and a derived
    bursting-or-not map onto a common grid_n x grid_n (held_frac,
    injected_frac) grid, and returns the stacked arrays plus their
    cross-cell mean/std -- nan-aware, since not every cell's interpolated
    grid covers every normalized coordinate (e.g. a cell whose real grid
    barely extends past its floor leaves the far corners undefined).
    """
    rate_layers, burst_layers = [], []
    cell_ids = []
    for cid, gres in grid_cache.items():
        feat = features_cache.get(cid)
        if gres.get("status") != "ok" or feat is None or feat.get("status") != "ok":
            continue
        cell_floor_nA = gres["cell_floor_nA"]

        rate_map = {k: float(v) for k, v in (feat.get("firing_rate_map") or {}).items()}
        burst_map = {(p["held_nA"], p["injected_nA"]): (1.0 if p["test_pattern"] == "bursting" else 0.0)
                    for p in gres["grid"].values() if p["test_pattern"] is not None}

        rate_layers.append(normalize_grid_to_common_coords(cell_floor_nA, rate_map, grid_n, "drop_feature"))
        burst_layers.append(normalize_grid_to_common_coords(cell_floor_nA, burst_map, grid_n, "drop_feature"))
        cell_ids.append(cid)

    rate_stack = np.stack(rate_layers)  # (n_cells, grid_n, grid_n)
    burst_stack = np.stack(burst_layers)
    return {
        "cell_ids": cell_ids,
        "rate_mean": np.nanmean(rate_stack, axis=0), "rate_std": np.nanstd(rate_stack, axis=0),
        "rate_n": np.sum(~np.isnan(rate_stack), axis=0),
        "burst_mean": np.nanmean(burst_stack, axis=0), "burst_std": np.nanstd(burst_stack, axis=0),
        "burst_n": np.sum(~np.isnan(burst_stack), axis=0),
    }


def _panel(ax, matrix, n_covering, title, cmap, label, min_n_cells):
    """Blanks out (white, no color) any pixel fewer than min_n_cells actually
    cover -- a "high variance" pixel backed by only 2-3 cells is noise, not
    a real population signal (mirrors this toolset's existing convention,
    see _exact_grid_matrix's docstring in run_held_injected_grid.py, of
    never rendering a value that isn't honestly backed by real data).
    """
    shown = np.where(n_covering >= min_n_cells, matrix, np.nan)
    im = ax.imshow(shown, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=cmap,
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, label=label)
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=15)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("held current (fraction of floor)", fontsize=16)
    ax.set_ylabel("injected current (fraction of floor)", fontsize=16)
    ax.tick_params(labelsize=15)


def build_cross_cell_variability_fig(result: dict, n_cells_total: int, min_n_cells: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    _panel(axes[0, 0], result["rate_mean"], result["rate_n"], "mean firing rate",
          "viridis", "Hz", min_n_cells)
    _panel(axes[0, 1], result["rate_std"], result["rate_n"], "firing rate variability (SD)",
          "inferno", "Hz", min_n_cells)
    _panel(axes[1, 0], result["burst_mean"], result["burst_n"], "fraction of cells bursting",
          "cividis", "fraction", min_n_cells)
    _panel(axes[1, 1], result["burst_std"], result["burst_n"], "bursting variability (SD)",
          "inferno", "SD", min_n_cells)
    fig.suptitle("Cross-cell variability in firing rate and bursting\n"
                f"(n = {len(result['cell_ids'])}/{n_cells_total} cells, matched on held/injected "
                f"as a fraction of each cell's own floor)",
                fontsize=18)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-cell uniqueness figure (research question 2): mean and cross-cell "
                    "standard deviation of firing rate and bursting, on a common (held_frac, "
                    "injected_frac) grid so cells with different current ranges are directly "
                    "comparable at 'the same' current.")
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH)
    parser.add_argument("--features-cache", default=DEFAULT_FEATURES_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--grid-n", type=int, default=DEFAULT_GRID_N,
                        help="Resolution of the common normalized grid each cell is interpolated onto.")
    parser.add_argument("--min-n-cells", type=int, default=10,
                        help="A pixel covered by fewer than this many cells is blanked out "
                             "rather than shown as a (noisy, low-n) mean/std.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.grid_cache, "rb") as f:
        grid_cache = pickle.load(f)
    with open(args.features_cache, "rb") as f:
        features_cache = pickle.load(f)

    result = stack_cross_cell(grid_cache, features_cache, args.grid_n)
    fig = build_cross_cell_variability_fig(result, len(grid_cache), args.min_n_cells)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outpath = figures_dir / f"cross_cell_variability.{args.figure_format}"
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    print(f"{len(result['cell_ids'])} cells stacked -> {outpath}")


if __name__ == "__main__":
    main()
