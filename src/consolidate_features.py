"""Master feature table: joins every per-cell scalar already sitting in the
pipeline's caches -- Track A conductances, steady-state baseline (Iapp=0),
silencing-threshold sweep, and Step 2c grid features -- into one table, plus
a PCA over it. No new simulation: everything here was already computed by
an earlier stage and is just being pulled together and, in a few cases,
cheaply re-derived from an already-persisted curve (e.g. decline_steepness
from the silencing sweep's freq_hz_curve).

Deliberately separate from extract_grid_features.py's own PCA
(cell_grid_features_pca.pkl), which runs eigen-response PCA over each
cell's full (held, injected) grid *map* (sag depth, rebound occurrence/
count) interpolated onto a common coordinate system -- a different kind of
input (spatial maps vs. per-cell scalars) that doesn't belong in the same
matrix as this one.
"""

import pickle
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from steady_state_cache import load_cache as load_steady_state_cache
from steady_state_cache import CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH

DEFAULT_TRACK_A_PATH = ROOT_DIR / "track_a_features.csv"
DEFAULT_SILENCING_CACHE_PATH = ROOT_DIR / "cell_silencing_thresholds.pkl"
DEFAULT_GRID_FEATURES_CACHE_PATH = ROOT_DIR / "cell_grid_features.pkl"
DEFAULT_OUTPUT_CSV_PATH = ROOT_DIR / "master_features.csv"
DEFAULT_PCA_OUTPUT_PATH = ROOT_DIR / "master_features_pca.pkl"
DEFAULT_PCA_LOADINGS_CSV_PATH = ROOT_DIR / "master_features_pca_loadings.csv"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "master_features"
DEFAULT_FIGURE_FORMAT = "svg"


# ---------------------------------------------------------------------------
# Per-source scalar extraction
# ---------------------------------------------------------------------------

def load_track_a(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="cell_id")


def extract_steady_state_features(cache_path: Path) -> pd.DataFrame:
    """Baseline (Iapp=0) intrinsic properties. period_ms is omitted -- it's
    exactly 1000/freq_hz, so including both would just be a collinear
    duplicate in the PCA.
    """
    cache = load_steady_state_cache(cache_path)
    rows = {}
    for cell_id, entry in cache.items():
        if entry.get("status") != "ok":
            continue
        rows[cell_id] = {
            "baseline_freq_hz": entry["freq_hz"],
            "baseline_v_min_mV": entry["v_min_mV"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _decline_steepness(entry: dict) -> float:
    """Slope (Hz/nA) of the firing-rate decline as hyperpolarizing current
    approaches the silencing threshold, fit over the already-persisted
    freq_hz_curve/iapp_levels_nA from the silencing sweep -- the same
    linear-fit approach extract_grid_features.py uses for fi_slope, just
    applied to the *silencing* curve instead of the grid's held=0 column.
    Restricted to firing points only (freq_hz > 0): the silent tail carries
    no rate information, and including it would just measure "how far the
    threshold sits from 0" (the same confound the module docstring in
    find_silencing_threshold.py already documents for why an explicit
    clean-vs-graded classification wasn't kept), not decline shape.
    Returns None if fewer than 2 firing points are available.
    """
    iapp = np.asarray(entry.get("iapp_levels_nA"))
    freq = np.asarray(entry.get("freq_hz_curve"))
    if iapp is None or freq is None or len(iapp) < 2:
        return None
    mask = freq > 0
    if mask.sum() < 2 or np.ptp(iapp[mask]) < 1e-9:
        return None
    slope, _ = np.polyfit(iapp[mask], freq[mask], 1)
    return float(slope)


def extract_silencing_features(cache_path: Path) -> pd.DataFrame:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    rows = {}
    for cell_id, entry in cache.items():
        if entry.get("status") not in ("ok", "unsettled"):
            continue
        bracket = entry.get("silencing_threshold_bracket_nA")
        if bracket is not None:
            threshold_nA = 0.5 * (bracket[0] + bracket[1])
        else:
            # "unsettled" cells have no confirmed bracket -- first_silent_level_nA
            # is the best available anchor (see get_cell_floor_nA in
            # run_held_injected_grid.py, which falls back the same way).
            threshold_nA = entry.get("first_silent_level_nA")
        rows[cell_id] = {
            "silencing_threshold_nA": threshold_nA,
            "silencing_status": entry["status"],
            "any_bursting_before_silence": bool(entry.get("any_bursting_detected", False)),
            "n_burst_transitions": len(entry.get("burst_transitions", [])),
            "silencing_decline_steepness_hz_per_nA": _decline_steepness(entry),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _map_summary(feature_map: dict, agg: str) -> float:
    if not feature_map:
        return None
    vals = np.array(list(feature_map.values()), dtype=float)
    if agg == "max":
        return float(np.nanmax(vals))
    if agg == "mean":
        return float(np.nanmean(vals))
    raise ValueError(agg)


def extract_grid_scalar_features(cache_path: Path) -> pd.DataFrame:
    """Scalar grid-derived features already computed by extract_grid_features.py,
    plus simple max/mean reductions of its map features (sag depth, rebound
    latency/peak) down to one number per cell -- the maps themselves are
    spatial and belong in extract_grid_features.py's own map-based PCA, not
    flattened arbitrarily into this scalar table.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    rows = {}
    for cell_id, entry in cache.items():
        if entry.get("status") != "ok":
            continue
        rows[cell_id] = {
            "fi_slope_hz_per_nA": entry["fi_slope_hz_per_nA"],
            "burstiness_index": entry["burstiness_index"],
            "fraction_rebound": entry["fraction_rebound"],
            "peak_test_freq_hz": entry["peak_test_freq_hz"],
            "max_rebound_spike_count": entry["max_rebound_spike_count"],
            "rebound_boundary_slope": entry["rebound_boundary_slope"],
            "burst_boundary_slope": entry["burst_boundary_slope"],
            "max_sag_depth_mV": _map_summary(entry["sag_depth_map"], "max"),
            "mean_rebound_latency_ms": _map_summary(entry["rebound_latency_map"], "mean"),
            "max_rebound_peak_mV": _map_summary(entry["rebound_peak_mV_map"], "max"),
            "max_adaptation_ratio": _map_summary(entry["adaptation_ratio_map"], "max"),
            "max_n_bursts": _map_summary(entry["n_bursts_map"], "max"),
            "mean_spikes_per_burst": _map_summary(entry["spikes_per_burst_map"], "mean"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Assembly + PCA
# ---------------------------------------------------------------------------

def build_master_table(track_a_path: Path, steady_state_cache_path: Path,
                       silencing_cache_path: Path, grid_features_cache_path: Path) -> pd.DataFrame:
    track_a = load_track_a(track_a_path)
    ss = extract_steady_state_features(steady_state_cache_path)
    sil = extract_silencing_features(silencing_cache_path)
    grid = extract_grid_scalar_features(grid_features_cache_path)

    # Outer join on cell_id -- a cell missing from a later stage (e.g. the 3
    # shooting_failed cells with no steady-state/silencing/grid data at all)
    # still keeps its Track A row, just with NaN for everything downstream,
    # rather than being silently dropped from the table entirely.
    table = track_a.join([ss, sil, grid], how="outer")
    table.index.name = "cell_id"
    return table


def run_pca_on_master_table(table: pd.DataFrame, min_feature_coverage: float = 0.85) -> dict:
    """PCA restricted to cells with a fully complete row -- imputing this
    many disparate scalar sources for an incomplete cell would manufacture
    signal that isn't there, rather than honestly reporting which cells the
    current pipeline stage couldn't reach.

    Complete-case selection is applied in two stages, not one: first drop
    any FEATURE covering fewer than min_feature_coverage of cells (e.g.
    burst_boundary_slope, which needs a burst-onset boundary to actually
    exist within a cell's swept grid to be defined at all -- confirmed
    directly this single column alone was excluding 32/69 cells, most of
    which have every OTHER feature and were only being dropped for this one
    genuinely-sparse one). Only THEN drop cells still missing something
    among the remaining, well-covered features. Skipping the first step and
    going straight to complete-case rows would let one sparse feature veto
    the whole table's coverage.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    numeric = table[numeric_cols]

    coverage = numeric.notna().mean()
    kept_cols = coverage[coverage >= min_feature_coverage].index.tolist()
    dropped_cols = coverage[coverage < min_feature_coverage].index.tolist()

    numeric = numeric[kept_cols]
    complete_mask = numeric.notna().all(axis=1)
    complete = numeric[complete_mask]
    excluded = table.index[~complete_mask].tolist()

    if len(complete) < 3:
        return {"status": "insufficient_cells", "n_complete": len(complete), "excluded_cell_ids": excluded,
                "dropped_feature_names": dropped_cols}

    scaler = StandardScaler()
    X_z = scaler.fit_transform(complete.values)
    pca = PCA()
    scores = pca.fit_transform(X_z)
    loadings = pd.DataFrame(pca.components_.T, index=kept_cols,
                            columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])])
    return {
        "status": "ok", "cell_ids": complete.index.tolist(), "excluded_cell_ids": excluded,
        "dropped_feature_names": dropped_cols,
        "feature_names": kept_cols, "scores": scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "loadings": loadings,
    }


def plot_pca(pca_result: dict, outdir: Path, command: str, fig_format: str) -> None:
    if pca_result["status"] != "ok":
        return
    outdir.mkdir(parents=True, exist_ok=True)
    scores = pca_result["scores"]
    var = pca_result["explained_variance_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(scores[:, 0], scores[:, 1], s=25, color="steelblue")
    for i, cid in enumerate(pca_result["cell_ids"]):
        axes[0].annotate(cid, (scores[i, 0], scores[i, 1]), fontsize=5, alpha=0.7)
    axes[0].set_xlabel(f"PC1 ({var[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({var[1]:.1%})")
    axes[0].set_title(f"Master feature PCA ({len(pca_result['cell_ids'])} cells)", fontsize=10)

    loadings = pca_result["loadings"]
    top = loadings["PC1"].abs().sort_values(ascending=False).index
    axes[1].barh(range(len(top)), loadings.loc[top, "PC1"].values, color="steelblue")
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top, fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("PC1 loading")
    axes[1].set_title("Feature loadings on PC1", fontsize=10)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.text(0.5, 0.01, command, ha="center", va="bottom", fontsize=6, color="gray")
    outpath = outdir / f"master_features_pca.{fig_format}"
    fig.savefig(outpath)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-a-csv", default=DEFAULT_TRACK_A_PATH)
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--silencing-cache", default=DEFAULT_SILENCING_CACHE_PATH)
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument("--pca-output", default=DEFAULT_PCA_OUTPUT_PATH)
    parser.add_argument("--pca-loadings-csv", default=DEFAULT_PCA_LOADINGS_CSV_PATH,
                        help="Full feature x PC loadings table (not just the top few printed to "
                             "the console), for inspecting what drives PC2+ directly.")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--min-feature-coverage", type=float, default=0.85,
                        help="Features present for fewer than this fraction of cells are dropped "
                             "from the PCA matrix entirely (rather than letting them veto complete-case "
                             "row selection for every cell missing just that one column).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)

    table = build_master_table(Path(args.track_a_csv), Path(args.steady_state_cache),
                               Path(args.silencing_cache), Path(args.grid_features_cache))
    table.to_csv(args.output_csv)
    print(f"Master feature table: {table.shape[0]} cells x {table.shape[1]} columns "
          f"-> {args.output_csv}")

    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    n_missing = table[numeric_cols].isna().sum().sort_values(ascending=False)
    n_missing = n_missing[n_missing > 0]
    if len(n_missing):
        print("\nMissing-value counts per numeric column (of "
              f"{table.shape[0]} cells):")
        print(n_missing.to_string())

    if args.no_pca:
        return

    pca_result = run_pca_on_master_table(table, min_feature_coverage=args.min_feature_coverage)
    if pca_result["status"] != "ok":
        print(f"\nPCA skipped: {pca_result['status']} "
              f"(n_complete={pca_result.get('n_complete')})")
        return

    if pca_result["dropped_feature_names"]:
        print(f"\nDropped from PCA for coverage < {args.min_feature_coverage:.0%}: "
              f"{pca_result['dropped_feature_names']}")
    print(f"\nPCA over {len(pca_result['cell_ids'])} cells with complete rows "
          f"({len(pca_result['excluded_cell_ids'])} excluded for missing data: "
          f"{pca_result['excluded_cell_ids']})")
    var = pca_result["explained_variance_ratio"]
    cum = np.cumsum(var)
    for i, (v, c) in enumerate(zip(var, cum)):
        print(f"  PC{i + 1}: {v:.1%} (cumulative {c:.1%})")

    # PC1 and PC2 are each ranked by their OWN |loading|, not just PC2's value
    # for PC1's top features -- PC2 can (and often does) be driven by a mostly
    # different set of features than PC1, which the previous PC1-only ranking
    # would have hidden.
    top_pc1 = pca_result["loadings"]["PC1"].abs().sort_values(ascending=False)
    print("\nTop PC1 loadings:")
    print(pca_result["loadings"].loc[top_pc1.index[:8], ["PC1", "PC2"]].to_string())

    top_pc2 = pca_result["loadings"]["PC2"].abs().sort_values(ascending=False)
    print("\nTop PC2 loadings:")
    print(pca_result["loadings"].loc[top_pc2.index[:8], ["PC1", "PC2"]].to_string())

    pca_result["loadings"].to_csv(args.pca_loadings_csv)
    print(f"\nFull PC loadings table (all features x all PCs) written to {args.pca_loadings_csv}")

    with open(args.pca_output, "wb") as f:
        pickle.dump(pca_result, f)
    print(f"PCA result written to {args.pca_output}")

    if not args.no_plot:
        plot_pca(pca_result, Path(args.figures_dir), command, args.figure_format)
        print(f"Figure written to {args.figures_dir}/")


if __name__ == "__main__":
    main()
