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
cell_held_injected_grid.pkl (for the raw test_pattern/rebound_pattern used
to build the test- and recovery-phase bursting-or-not maps) and
cell_grid_features.pkl (for the continuous feature maps).

Continuous features (firing rate, sag depth, n bursts, rebound peak, ...)
are z-scored per cell before stacking (each cell's own mean/std over its
own real grid points, not the population's) so a cell that's simply larger
in absolute magnitude everywhere -- e.g. a uniformly higher-firing cell --
doesn't dominate the cross-cell mean at every pixel it covers; what's
plotted is how far above/below ITS OWN typical value a cell runs at each
point, averaged across cells. Bursting fraction is left alone since it's
already a 0/1 indicator, not a magnitude. Disable with --no-zscore.
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

# Registry of map-valued grid features available for cross-cell stacking,
# beyond the always-included derived "bursting" map (built from test_pattern
# in the raw grid cache rather than read directly off features_cache).
# feat_key indexes straight into extract_grid_features.py's per-cell dict.
CROSS_CELL_FEATURES = {
    "firing_rate": dict(feat_key="firing_rate_map", cmap="viridis", label="Hz", title="firing rate"),
    "sag_depth": dict(feat_key="sag_depth_map", cmap="magma", label="mV", title="sag depth"),
    "n_bursts": dict(feat_key="n_bursts_map", cmap="plasma", label="count", title="n bursts"),
    "spikes_per_burst": dict(feat_key="spikes_per_burst_map", cmap="plasma", label="spikes/burst",
                             title="spikes per burst"),
    "adaptation_ratio": dict(feat_key="adaptation_ratio_map", cmap="coolwarm", label="ratio",
                             title="adaptation ratio (tonic)"),
    "rebound_count": dict(feat_key="rebound_count_map", cmap="plasma", label="count",
                          title="rebound spike count"),
    "rebound_latency": dict(feat_key="rebound_latency_map", cmap="viridis_r", label="ms",
                            title="rebound latency"),
    "rebound_peak": dict(feat_key="rebound_peak_mV_map", cmap="magma", label="mV",
                         title="rebound peak"),
}
DEFAULT_FEATURES = ["firing_rate", "sag_depth", "n_bursts", "rebound_peak"]

# Both derived directly from the raw grid cache (not features_cache), always
# included -- test_pattern=="bursting" for the test phase, rebound_pattern==
# "bursting_rebound" for the recovery phase. Points where the corresponding
# phase's classification doesn't apply at all (test_pattern is None /
# rebound_pattern is "not_applicable") are excluded rather than counted as
# 0 -- there's no answer to report there, not a confident "not bursting".
BURST_INDICATORS = {
    "test_burst": dict(title="fraction of cells bursting (test)",
                       point_fn=lambda p: None if p["test_pattern"] is None
                                          else 1.0 if p["test_pattern"] == "bursting" else 0.0),
    "recovery_burst": dict(title="fraction of cells bursting (recovery)",
                           point_fn=lambda p: None if p["rebound_pattern"] in (None, "not_applicable")
                                              else 1.0 if p["rebound_pattern"] == "bursting_rebound" else 0.0),
}


def _zscore_map(feat_map: dict) -> dict:
    """Centers and scales one cell's real (pre-interpolation) feature map by
    ITS OWN mean/std, so the cross-cell stack averages comparable, dimension-
    less "SDs from this cell's own typical value" rather than raw units a
    uniformly larger/smaller cell would otherwise dominate. A cell with <2
    points or zero variance is left centered-only (z=0 everywhere, since
    every value already equals the mean) rather than divided by zero.
    """
    if len(feat_map) < 2:
        return {k: 0.0 for k in feat_map}
    values = np.array(list(feat_map.values()), dtype=float)
    mean, std = float(np.mean(values)), float(np.std(values))
    if std < 1e-9:
        return {k: 0.0 for k in feat_map}
    return {k: (v - mean) / std for k, v in feat_map.items()}


def stack_cross_cell(grid_cache: dict, features_cache: dict, grid_n: int, feature_names: list,
                     cell_ids_filter=None, zscore: bool = True) -> dict:
    """Normalizes every ok cell's requested map-valued features, plus the
    test- and recovery-phase bursting-or-not maps, onto a common
    grid_n x grid_n (held_frac, injected_frac) grid, and returns the stacked
    arrays plus their cross-cell mean/std -- nan-aware, since not every
    cell's interpolated grid covers every normalized coordinate (e.g. a cell
    whose real grid barely extends past its floor leaves the far corners
    undefined).

    cell_ids_filter, when given, restricts stacking to that subset of cell
    IDs (e.g. the curated dev set) rather than every "ok" cell in the cache.
    zscore controls whether the continuous feature_names maps are per-cell
    z-scored before stacking (see _zscore_map); the burst indicators are
    never z-scored, they're already 0/1.
    """
    layers = {name: [] for name in feature_names}
    burst_layers = {name: [] for name in BURST_INDICATORS}
    cell_ids = []
    for cid, gres in grid_cache.items():
        if cell_ids_filter is not None and cid not in cell_ids_filter:
            continue
        feat = features_cache.get(cid)
        if gres.get("status") != "ok" or feat is None or feat.get("status") != "ok":
            continue
        cell_floor_nA = gres["cell_floor_nA"]

        for name in feature_names:
            feat_map = {k: float(v) for k, v in (feat.get(CROSS_CELL_FEATURES[name]["feat_key"]) or {}).items()}
            if zscore:
                feat_map = _zscore_map(feat_map)
            layers[name].append(normalize_grid_to_common_coords(cell_floor_nA, feat_map, grid_n, "drop_feature"))

        for name, spec in BURST_INDICATORS.items():
            point_map = {}
            for p in gres["grid"].values():
                value = spec["point_fn"](p)
                if value is not None:
                    point_map[(p["held_nA"], p["injected_nA"])] = value
            burst_layers[name].append(normalize_grid_to_common_coords(cell_floor_nA, point_map, grid_n,
                                                                       "drop_feature"))
        cell_ids.append(cid)

    result = {"cell_ids": cell_ids}
    for name in feature_names:
        stack = np.stack(layers[name])  # (n_cells, grid_n, grid_n)
        result[f"{name}_mean"] = np.nanmean(stack, axis=0)
        result[f"{name}_std"] = np.nanstd(stack, axis=0)
        result[f"{name}_n"] = np.sum(~np.isnan(stack), axis=0)

    for name in BURST_INDICATORS:
        stack = np.stack(burst_layers[name])
        result[f"{name}_mean"] = np.nanmean(stack, axis=0)
        result[f"{name}_std"] = np.nanstd(stack, axis=0)
        result[f"{name}_n"] = np.sum(~np.isnan(stack), axis=0)
    return result


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


def build_cross_cell_variability_fig(result: dict, n_cells_total: int, min_n_cells: int,
                                     feature_names: list, zscore: bool) -> plt.Figure:
    mean_label = "z-score (SD from cell's own mean)" if zscore else None
    rows = [(name, CROSS_CELL_FEATURES[name]["title"], CROSS_CELL_FEATURES[name]["cmap"],
            mean_label or CROSS_CELL_FEATURES[name]["label"]) for name in feature_names]
    rows += [(name, spec["title"], "cividis", "fraction") for name, spec in BURST_INDICATORS.items()]

    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 7 * len(rows)), squeeze=False)
    for row_idx, (key, title, cmap, label) in enumerate(rows):
        is_burst = key in BURST_INDICATORS
        _panel(axes[row_idx, 0], result[f"{key}_mean"], result[f"{key}_n"], f"mean {title}",
              cmap, label, min_n_cells)
        _panel(axes[row_idx, 1], result[f"{key}_std"], result[f"{key}_n"], f"{title} variability (SD)",
              "inferno", "SD" if is_burst else "SD of z-score" if zscore else label, min_n_cells)

    fig.suptitle("Cross-cell variability" +
                (" (continuous features z-scored per cell)" if zscore else "") +
                f"\n(n = {len(result['cell_ids'])}/{n_cells_total} cells, matched on held/injected "
                f"as a fraction of each cell's own floor)",
                fontsize=18)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.03 * (2.0 / len(rows))))
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
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                        choices=sorted(CROSS_CELL_FEATURES), metavar="FEATURE",
                        help=f"Map-valued grid features to stack, in addition to test- and "
                             f"recovery-phase bursting fraction (always included). One of "
                             f"{sorted(CROSS_CELL_FEATURES)}.")
    parser.add_argument("--cells", nargs="+", default=None, metavar="CELLID",
                        help="Restrict to these cell IDs (e.g. the curated dev set) instead of "
                             "every 'ok' cell in the cache -- use for testing before a full-"
                             "population run. --min-n-cells will usually need lowering to match.")
    parser.add_argument("--no-zscore", dest="zscore", action="store_false",
                        help="Stack continuous features in their raw units instead of per-cell "
                             "z-scoring them first -- lets a uniformly larger/smaller cell "
                             "dominate the cross-cell mean at every pixel it covers; the default "
                             "(z-scored) avoids that.")
    parser.set_defaults(zscore=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.grid_cache, "rb") as f:
        grid_cache = pickle.load(f)
    with open(args.features_cache, "rb") as f:
        features_cache = pickle.load(f)

    cell_ids_filter = set(args.cells) if args.cells else None
    result = stack_cross_cell(grid_cache, features_cache, args.grid_n, args.features, cell_ids_filter,
                              args.zscore)
    fig = build_cross_cell_variability_fig(result, len(grid_cache), args.min_n_cells, args.features,
                                           args.zscore)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(args.features)
    if not args.zscore:
        suffix += "_raw"
    if args.cells is not None:
        suffix += "_test-cells"
    outpath = figures_dir / f"cross_cell_variability_{suffix}.{args.figure_format}"
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    print(f"{len(result['cell_ids'])} cells stacked -> {outpath}")


if __name__ == "__main__":
    main()
