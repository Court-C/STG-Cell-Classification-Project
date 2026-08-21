"""Step 2c: feature extraction from run_held_injected_grid.py's per-cell
(held, injected) grid, plus cross-cell PCA.

Phase 1 (implemented here): features computable purely from the grid cache
that Step 2 already produced -- no re-simulation. This covers most of the
2c spec: burst/tonic classification aggregation, burstiness index, sag
depth map, rebound maps (binary/count/latency/peak), firing-rate map, F-I
slope (held=0 column), rebound/burst-onset boundary slope, summary stats,
spike-rate adaptation map (tonic points), burst-structure maps (bursting
points: burst count, average spikes/burst), and cross-cell PCA over a
common normalized (held_frac, injected_frac) grid.

Adaptation ratio and burst structure were originally deferred here (would
have needed re-simulation to see the raw per-spike ISI sequence) but are now
computed inline in run_held_injected_grid.py's run_test_and_recovery, where
that sequence is already in scope before being reduced to summary stats --
see test_adaptation_ratio/test_n_bursts/test_spikes_per_burst there.

Still deliberately NOT attempted here (genuinely need re-simulation -- no
raw voltage trace is persisted per grid point, only derived scalars):
input resistance/tau, spike waveform features (threshold/amplitude/
half-width/AHP), and exact ISI CV/LV.
"""

import os
import pickle
import sys
import time
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import griddata

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_held_injected_grid import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH
from run_held_injected_grid import load_output_cache as load_grid_cache, plot_cell_grid_features

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_grid_features.pkl"
DEFAULT_PCA_OUTPUT_PATH = ROOT_DIR / "cell_grid_features_pca.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "grid_features"
DEFAULT_FIGURE_FORMAT = "png"


# ---------------------------------------------------------------------------
# Cache-only per-cell features
# ---------------------------------------------------------------------------

def compute_fi_slope(grid: dict, held_level: float = 0.0) -> dict:
    """Linear fit of test_freq_hz vs injected_nA along the held=0 column,
    restricted to non-silent, confidently-classified points (a flat run of
    zero-rate points below the effective rheobase-equivalent would bias a
    linear slope estimate without adding information).
    """
    points = [p for p in grid.values()
             if p["held_nA"] == held_level and not p["blew_up"]
             and p["test_pattern"] not in (None, "silent")]
    if len(points) < 2:
        return {"fi_slope_hz_per_nA": None, "fi_slope_r2": None, "fi_slope_n_points": len(points)}
    x = np.array([p["injected_nA"] for p in points])
    y = np.array([p["test_freq_hz"] for p in points])
    if np.ptp(x) < 1e-9:
        return {"fi_slope_hz_per_nA": None, "fi_slope_r2": None, "fi_slope_n_points": len(points)}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"fi_slope_hz_per_nA": float(slope), "fi_slope_r2": float(r2) if r2 is not None else None,
            "fi_slope_n_points": len(points)}


def compute_burstiness(grid: dict) -> dict:
    # test_pattern is only ever None (blew_up), "silent", "tonic", or
    # "bursting" now (see to_stored_pattern in find_silencing_threshold.py),
    # so "confident" here is just "not blew_up".
    confident = [p for p in grid.values() if not p["blew_up"]
                and p["test_pattern"] is not None]
    n_confident = len(confident)
    n_bursting = sum(1 for p in confident if p["test_pattern"] == "bursting")
    n_tonic = sum(1 for p in confident if p["test_pattern"] == "tonic")
    n_silent = sum(1 for p in confident if p["test_pattern"] == "silent")
    burstiness_index = n_bursting / n_confident if n_confident > 0 else None
    return {"burstiness_index": burstiness_index, "n_confident_points": n_confident,
            "n_bursting_points": n_bursting, "n_tonic_points": n_tonic, "n_silent_points": n_silent}


def compute_sag_depth_map(grid: dict, min_pre_spike_window_ms: float = 100.0) -> dict:
    """Sag depth per point (mV): hold_v_trough - test_v_min_pre_spike, i.e.
    how much FURTHER V dips below the held level's own baseline before
    either a fixed early window elapses or the cell's first spike fires,
    whichever comes first (test_v_min_pre_spike_mV, from
    run_held_injected_grid.py's compute_pre_spike_sag_trough). Sag is a
    passive, subthreshold, Ih-mediated relaxation that plays out BEFORE the
    first spike/burst, so it's assessed here for every test window that has
    one -- tonic and bursting points included, not just windows that stay
    silent throughout (sag is not exclusive to fully-quiescent cells).

    Baseline is hold_v_trough_mV (the minimum V already reached during the
    held level's own last settle chunk), not a single endpoint sample. For a
    flatline hold these are identical (a constant trace's min is its
    endpoint) -- but for a held level that's itself tonically/burst firing
    at rest (common: e.g. XB2IQX fires spontaneously even at held=0, and at
    held=-3nA still fires at 19.5Hz), a single endpoint sample lands at
    whatever random phase of the oscillation the chunked settle loop
    happened to stop on (confirmed on a real XB2IQX point: -39.8mV
    end-of-chunk snapshot mid-cycle, vs. a trace that actually spans spike
    peak down to ~-70mV AHP) -- not a usable "resting" reference. This used
    to be handled by skipping every non-flatline-hold point entirely
    (silently -- NOT stored as a real 0, easy to misread as "no sag" in a
    heatmap), which dropped sag everywhere a cell fires spontaneously,
    including a real ~30mV dip at XB2IQX held=-3.00/injected=-5.50 that
    never got measured. Using the hold's own already-simulated trough
    instead needs no extra simulation, reduces to the exact old numeric
    definition when hold_is_flatline, and gives a well-defined value
    ("does the test window dip below where the cell already goes on its
    own?") when it isn't -- so every point with a valid test window now
    gets a sag value, not just the flatline-hold subset.

    Reports an absolute depth (mV), not a normalized 0-1 ratio: a proper
    ratio needs a second "relaxed/recovered" reference voltage after the
    trough, and directly checking that against real traces showed the
    natural candidate (voltage right before the first spike) is usually
    still deep in the spike's own fast Na+ upstroke, not a subthreshold
    value (one VC08B6 point: -30mV eight timesteps before the peak, +19mV
    one timestep before it) -- not usable without dedicated spike-onset
    (threshold-crossing) detection, out of scope for a cache-only Phase-1
    feature.

    Two further gates, both confirmed necessary against real XB2IQX data
    (472/2500 points, up to -17.3mV, before these were added):

    1. min_pre_spike_window_ms: excludes any point whose test_first_spike_ms
       is under this threshold. test_v_min_pre_spike_mV is a windowed min
       over WHATEVER time elapses before the first spike (up to
       sag_window_ms) -- when that window is short (a spike fires quickly,
       e.g. because injected barely perturbs an already-fast-firing held
       level), V simply hasn't had time to move far, no matter how much real
       sag the cell has. That shallow, truncated min then gets compared
       against hold_v_trough_mV -- which IS measured over a full multi-
       second, multi-cycle settle chunk and reliably finds the true AHP
       floor -- so short windows systematically read as spuriously small or
       negative "sag" purely as a window-length artifact, not a real
       measurement. Confirmed directly: mean sag by first_spike_ms bucket
       goes from -3.2mV (0-10ms) up to +18.8mV (no spike within
       sag_window_ms at all, i.e. the full window) -- window length alone
       predicts the sign. 100ms keeps ~41% of a real 2500-point XB2IQX grid
       while cutting the negative-outlier count from 472 to 18 and the worst
       outlier from -17.3mV to -7.6mV (chosen over stricter thresholds like
       300ms, which reach zero negative outliers but only by dropping to 6%
       coverage -- essentially just the fully-silenced points).
    2. A negative result (baseline - trough < 0, i.e. the test window's
       measured trough sits ABOVE the held level's own oscillation floor) is
       physically impossible for a real sag response -- hyperpolarizing
       injected current cannot make V "sag upward" past where the cell
       already goes on its own. Explicitly rejected (excluded, not
       clamped to 0) rather than silently reported as a small negative
       number, even after gate 1: gate 1 alone still leaves a residual few
       (18/2500 at the 100ms threshold) points where the window-truncation
       artifact hasn't fully resolved.
    """
    sag_map = {}
    for key, p in grid.items():
        if p["blew_up"]:
            continue
        trough = p.get("test_v_min_pre_spike_mV")
        baseline = p.get("hold_v_trough_mV")
        if trough is None or baseline is None or not np.isfinite(trough) or not np.isfinite(baseline):
            continue
        first_spike_ms = p.get("test_first_spike_ms")
        if first_spike_ms is not None and first_spike_ms < min_pre_spike_window_ms:
            continue
        sag = float(baseline - trough)
        if sag < 0:
            continue
        sag_map[key] = sag
    return sag_map


def compute_rebound_maps(grid: dict) -> dict:
    occurred, count, latency, peak_mV, pattern = {}, {}, {}, {}, {}
    for key, p in grid.items():
        if p["blew_up"] or not p["rebound_applicable"]:
            continue
        occurred[key] = p["rebound_occurred"]
        count[key] = p["rebound_spike_count"]
        if p["rebound_latency_ms"] is not None:
            latency[key] = p["rebound_latency_ms"]
        if p["rebound_peak_mV"] is not None:
            peak_mV[key] = p["rebound_peak_mV"]
        pattern[key] = p["rebound_pattern"]
    return {"rebound_occurred_map": occurred, "rebound_count_map": count,
            "rebound_latency_map": latency, "rebound_peak_mV_map": peak_mV,
            "rebound_pattern_map": pattern}


def compute_burst_rate_maps(grid: dict) -> dict:
    """Cache-only approximations: intra-burst rate from the short-ISI mode,
    burst frequency approximated as 1/long-ISI (treating the long ISI as
    roughly the inter-burst-onset interval). Exact burst count and
    spikes-per-burst are NOT approximations -- see compute_burst_structure_maps,
    computed inline in run_held_injected_grid.py from the actual per-spike
    ISI sequence (test_n_bursts/test_spikes_per_burst), not derived here.
    """
    intra_burst_rate, burst_freq_approx = {}, {}
    for key, p in grid.items():
        if p["blew_up"] or p["test_pattern"] != "bursting":
            continue
        if p["test_isi_short_ms"]:
            intra_burst_rate[key] = 1000.0 / p["test_isi_short_ms"]
        if p["test_isi_long_ms"]:
            burst_freq_approx[key] = 1000.0 / p["test_isi_long_ms"]
    return {"intra_burst_rate_map": intra_burst_rate, "burst_freq_approx_map": burst_freq_approx}


def compute_burst_structure_maps(grid: dict) -> dict:
    """Burst count and average spikes/burst per point (test_n_bursts,
    test_spikes_per_burst, from run_held_injected_grid.py -- computed there
    from n_long_isis + 1, i.e. the actual count of inter-burst gaps in the
    test window's ISI sequence, not an approximation). Answers "does this
    cell burst continuously through the injected window, or fire a handful
    of discrete bursts before falling silent" per grid point. Restricted to
    test_pattern == "bursting" upstream, same as compute_burst_rate_maps.
    """
    n_bursts, spikes_per_burst = {}, {}
    for key, p in grid.items():
        if p["blew_up"] or p["test_n_bursts"] is None:
            continue
        n_bursts[key] = p["test_n_bursts"]
        if p["test_spikes_per_burst"] is not None:
            spikes_per_burst[key] = p["test_spikes_per_burst"]
    return {"n_bursts_map": n_bursts, "spikes_per_burst_map": spikes_per_burst}


def compute_adaptation_map(grid: dict) -> dict:
    """Spike-rate adaptation ratio per point (test_adaptation_ratio, from
    run_held_injected_grid.py): mean of the last adaptation_edge_n ISIs over
    the first adaptation_edge_n ISIs of the test window. Restricted to
    test_pattern == "tonic" upstream (a bursting train's first-k/last-k ISIs
    would mix intra-/inter-burst intervals -- see that function's
    docstring), so this map is sparser than sag_depth_map by design.
    """
    adaptation_map = {}
    for key, p in grid.items():
        if p["blew_up"] or p["test_adaptation_ratio"] is None:
            continue
        adaptation_map[key] = float(p["test_adaptation_ratio"])
    return adaptation_map


def compute_firing_rate_map(grid: dict) -> dict:
    """test_freq_hz per point -- the grid's core response surface (spike
    rate as a function of held x injected current). Previously computed for
    every point but only ever consumed piecemeal (the held=0 column for
    compute_fi_slope, a grid-wide max for peak_test_freq_hz in
    compute_summary_stats) and never exposed as its own spatial feature or
    included in the cross-cell PCA -- despite being the single most direct
    measurement the whole grid sweep produces.
    """
    rate_map = {}
    for key, p in grid.items():
        if p["blew_up"] or p["test_freq_hz"] is None:
            continue
        rate_map[key] = float(p["test_freq_hz"])
    return rate_map


def compute_recovery_firing_rate_map(grid: dict) -> dict:
    """recovery_freq_hz per point -- the recovery-phase counterpart to
    compute_firing_rate_map's test_freq_hz. Computed over the FULL recovery
    window (run_held_injected_grid.py's run_test_and_recovery), not just the
    "rebound" spikes past rebound_latency_min_ms, so this reads as a
    genuine firing-rate map for the whole released-back-to-held period --
    directly comparable to the test-window firing rate map above it, rather
    than the narrower rebound-specific scalars (rebound_occurred/_count/
    _latency), which stay scoped to the "was this distinguishable from the
    held level's own baseline" question they were designed for.
    """
    rate_map = {}
    for key, p in grid.items():
        # .get(), not p["recovery_freq_hz"] (unlike test_freq_hz, which has
        # always existed): a grid cache point from before this field was
        # added (2026-08-21) has no key at all, not a None value -- skip it
        # rather than KeyError, so a stale cell's other 14 panels still
        # render while it awaits a stage-2 rerun to backfill this one.
        recovery_freq_hz = p.get("recovery_freq_hz")
        if p["blew_up"] or recovery_freq_hz is None:
            continue
        rate_map[key] = float(recovery_freq_hz)
    return rate_map


def _boundary_crossings(grid: dict, predicate) -> list:
    """For each distinct held row, walk points in descending injected order
    and record the (held, injected) midpoint of the first place `predicate`
    flips from False to True. This approximates "the boundary" directly
    from the adaptively-sampled point cloud without needing to reconstruct
    the sweep's original (and not persisted) edge graph.
    """
    by_held: dict = {}
    for p in grid.values():
        if p["blew_up"]:
            continue
        by_held.setdefault(p["held_nA"], []).append(p)

    crossings = []
    for held, points in by_held.items():
        points_sorted = sorted(points, key=lambda p: -p["injected_nA"])  # 0 -> more negative
        prev = None
        for p in points_sorted:
            val = predicate(p)
            if val is None:
                continue
            if prev is not None and val and not prev[1]:
                crossings.append((held, 0.5 * (prev[0]["injected_nA"] + p["injected_nA"])))
            prev = (p, val)
    return crossings


def _fit_boundary_slope(crossings: list) -> dict:
    if len(crossings) < 2:
        return {"slope": None, "r2": None, "n_points": len(crossings)}
    x = np.array([c[0] for c in crossings])
    y = np.array([c[1] for c in crossings])
    if np.ptp(x) < 1e-9:
        return {"slope": None, "r2": None, "n_points": len(crossings)}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope": float(slope), "r2": float(r2) if r2 is not None else None, "n_points": len(crossings)}


def compute_boundary_slopes(grid: dict) -> dict:
    # Deliberately NOT restricted to rebound_applicable points: filtering to
    # "applicable" first (as an earlier version of this function did) drops
    # every point in the "not yet suppressed enough" phase, so there's no
    # False state left to observe a transition FROM -- confirmed empirically
    # this made rebound_boundary_slope None for every single cell, since
    # fraction_rebound is very high (0.7-0.99) among applicable points here:
    # once a cell is even mildly suppressed it almost always rebounds, so
    # the real boundary of interest is "applicable+occurred" vs. everything
    # else (not suppressed enough, or suppressed with no rebound), not a
    # transition hiding within the applicable subset alone.
    rebound_crossings = _boundary_crossings(
        grid, lambda p: bool(p["rebound_applicable"] and p["rebound_occurred"]))
    burst_crossings = _boundary_crossings(
        grid, lambda p: (p["test_pattern"] == "bursting") if p["test_pattern"] is not None else None)
    reb = _fit_boundary_slope(rebound_crossings)
    burst = _fit_boundary_slope(burst_crossings)
    return {
        "rebound_boundary_slope": reb["slope"], "rebound_boundary_r2": reb["r2"],
        "n_rebound_boundary_points": reb["n_points"],
        "burst_boundary_slope": burst["slope"], "burst_boundary_r2": burst["r2"],
        "n_burst_boundary_points": burst["n_points"],
    }


def compute_summary_stats(grid: dict) -> dict:
    confident = [p for p in grid.values() if not p["blew_up"] and p["test_pattern"] is not None]
    applicable = [p for p in confident if p["rebound_applicable"]]
    freqs = [p["test_freq_hz"] for p in confident if p["test_freq_hz"] is not None]
    counts = [p["rebound_spike_count"] for p in applicable]
    return {
        "fraction_rebound": (sum(1 for p in applicable if p["rebound_occurred"]) / len(applicable)
                            if applicable else None),
        "n_rebound_applicable": len(applicable),
        "n_rebound_occurred": sum(1 for p in applicable if p["rebound_occurred"]),
        "peak_test_freq_hz": max(freqs) if freqs else None,
        "max_rebound_spike_count": max(counts) if counts else None,
    }


def extract_cell_features(cell_id: str, cell_result: dict) -> dict:
    base = {"cell_id": cell_id}
    if cell_result["status"] != "ok":
        return {**base, "status": "no_grid_data", "grid_status": cell_result["status"]}

    grid = cell_result["grid"]
    if not grid:
        return {**base, "status": "no_grid_data", "grid_status": "empty"}

    features = {**base, "status": "ok", "cell_floor_nA": cell_result["cell_floor_nA"],
               "injected_floor_nA": cell_result["injected_floor_nA"]}
    features.update(compute_fi_slope(grid))
    features.update(compute_burstiness(grid))
    features.update(compute_summary_stats(grid))
    features.update(compute_boundary_slopes(grid))
    features["sag_depth_map"] = compute_sag_depth_map(grid)
    features["adaptation_ratio_map"] = compute_adaptation_map(grid)
    features["firing_rate_map"] = compute_firing_rate_map(grid)
    features["recovery_firing_rate_map"] = compute_recovery_firing_rate_map(grid)
    features.update(compute_rebound_maps(grid))
    features.update(compute_burst_rate_maps(grid))
    features.update(compute_burst_structure_maps(grid))
    return features


# ---------------------------------------------------------------------------
# Cross-cell PCA over a common normalized (held_frac, injected_frac) grid
# ---------------------------------------------------------------------------

def normalize_grid_to_common_coords(cell_floor_nA: float, feature_map: dict,
                                    pca_grid_n: int, nan_policy: str) -> np.ndarray:
    """Interpolates a scattered {(held,injected): value} map onto a common
    pca_grid_n x pca_grid_n grid in normalized [0,1] coordinates (fraction
    of this cell's own cell_floor_nA), so cells with very different current
    ranges become comparable, fixed-length vectors for PCA.
    """
    if len(feature_map) < 4:
        return np.full((pca_grid_n, pca_grid_n), np.nan)

    points = np.array([[h / cell_floor_nA, i / cell_floor_nA] for (h, i) in feature_map.keys()])
    values = np.array(list(feature_map.values()), dtype=float)
    grid_coords = np.linspace(0, 1, pca_grid_n)
    gh, gi = np.meshgrid(grid_coords, grid_coords)

    # Linear interpolation needs a genuine 2D point cloud. A sparse feature
    # (adaptation_ratio_map is tonic-only, so far sparser than sag/rebound)
    # can easily end up with every surviving point sharing one axis's
    # coordinate -- confirmed directly: 4 real cells have adaptation_ratio_map
    # entries only at held=0 (the only column where they fire tonic rather
    # than bursting/silent elsewhere), which Qhull's triangulation rejects
    # outright ("input is less than 3-dimensional"). Nearest-neighbor doesn't
    # need a triangulation (it's a KDTree lookup), so it still works fine on
    # a collinear point cloud -- skip straight to it rather than crashing.
    collinear = np.ptp(points[:, 0]) < 1e-9 or np.ptp(points[:, 1]) < 1e-9
    if collinear:
        interpolated = (griddata(points, values, (gh, gi), method="nearest")
                        if nan_policy == "impute_nearest"
                        else np.full((pca_grid_n, pca_grid_n), np.nan))
        return interpolated

    interpolated = griddata(points, values, (gh, gi), method="linear")
    if nan_policy == "impute_nearest":
        nan_mask = np.isnan(interpolated)
        if nan_mask.any() and not nan_mask.all():
            nearest = griddata(points, values, (gh, gi), method="nearest")
            interpolated[nan_mask] = nearest[nan_mask]
    return interpolated


PCA_MAP_FEATURES = ["sag_depth_map", "rebound_occurred_map", "rebound_count_map",
                    "firing_rate_map", "adaptation_ratio_map"]


def build_pca_matrix(all_features: dict, pca_grid_n: int, nan_policy: str) -> dict:
    ok_cells = {cid: f for cid, f in all_features.items() if f["status"] == "ok"}
    cell_ids = sorted(ok_cells.keys())
    if len(cell_ids) < 3:
        return {"status": "insufficient_cells", "n_cells": len(cell_ids)}

    per_feature_grids = {feat: [] for feat in PCA_MAP_FEATURES}
    for cid in cell_ids:
        f = ok_cells[cid]
        for feat in PCA_MAP_FEATURES:
            fmap = {k: float(v) for k, v in f[feat].items()}
            g = normalize_grid_to_common_coords(f["cell_floor_nA"], fmap, pca_grid_n, nan_policy)
            per_feature_grids[feat].append(g.flatten())

    vectors = []
    valid_cell_ids = []
    for idx, cid in enumerate(cell_ids):
        row = np.concatenate([per_feature_grids[feat][idx] for feat in PCA_MAP_FEATURES])
        if np.all(np.isnan(row)):
            continue
        if np.any(np.isnan(row)):
            row = np.nan_to_num(row, nan=np.nanmean(row))
        vectors.append(row)
        valid_cell_ids.append(cid)

    if len(vectors) < 3:
        return {"status": "insufficient_cells", "n_cells": len(vectors)}

    X = np.array(vectors)
    return {"status": "ok", "X": X, "cell_ids": valid_cell_ids, "pca_grid_n": pca_grid_n,
            "feature_names": PCA_MAP_FEATURES}


def run_pca(pca_input: dict) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = pca_input["X"]
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    pca = PCA()
    scores = pca.fit_transform(X_z)
    return {
        "status": "ok", "scores": scores, "cell_ids": pca_input["cell_ids"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_, "pca_grid_n": pca_input["pca_grid_n"],
        "feature_names": pca_input["feature_names"],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
#
# The old plot_cell_features (12-panel) lived here; it's retired as of
# 2026-08-11 in favor of plot_cell_grid_features (run_held_injected_grid.py),
# one merged 14-panel figure combining this module's features with the
# grid-only panels (firing pattern, i_H at recovery trough) that plot_
# cell_grid used to render separately. See that function's docstring.


def plot_pca(pca_result: dict, outdir: Path, command: str, fig_format: str) -> None:
    if pca_result["status"] != "ok":
        return
    scores = pca_result["scores"]
    var_ratio = pca_result["explained_variance_ratio"]
    cell_ids = pca_result["cell_ids"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_show = min(10, len(var_ratio))
    axes[0].bar(range(1, n_show + 1), var_ratio[:n_show], color="steelblue")
    axes[0].plot(range(1, n_show + 1), np.cumsum(var_ratio[:n_show]), "o-", color="darkorange")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Scree plot", fontsize=9)

    axes[1].scatter(scores[:, 0], scores[:, 1], s=40, edgecolor="black", alpha=0.8)
    for cid, (pc1, pc2) in zip(cell_ids, scores[:, :2]):
        axes[1].annotate(cid, (pc1, pc2), fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0, color="gray", lw=0.5)
    axes[1].set_title("Cells in PC1-PC2 space", fontsize=9)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(0.5, 0.01, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"grid_features_pca.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_output_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path) -> None:
    # Write-then-rename, with a retry on Windows's transient PermissionError
    # -- see run_held_injected_grid.py's identical helper for why.
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(cache, handle)
    for attempt in range(5):
        try:
            os.replace(tmp_path, cache_path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-cell features from run_held_injected_grid.py's cache "
                    "(cache-only: no re-simulation) plus cross-cell PCA. Rin/tau, spike "
                    "waveform, and exact ISI CV/LV need re-simulation and are not part of "
                    "this pass -- see module docstring.")
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH)
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--pca-output", default=DEFAULT_PCA_OUTPUT_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--pca-grid-n", type=int, default=15,
                        help="Common normalized-grid resolution used to flatten each cell's "
                             "scattered adaptive grid into a fixed-length vector for PCA.")
    parser.add_argument("--pca-nan-policy", choices=["drop_feature", "impute_nearest"], default="impute_nearest")
    parser.add_argument("--no-pca", action="store_true", help="Skip cross-cell PCA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    grid_cache = load_grid_cache(Path(args.grid_cache))

    cell_ids = args.cells or sorted(grid_cache.keys())
    print(f"Extracting features for {len(cell_ids)} of {len(grid_cache)} cell(s) from {args.grid_cache}")

    def process(cell_id: str):
        cell_result = grid_cache.get(cell_id)
        if cell_result is None:
            return cell_id, {"cell_id": cell_id, "status": "no_grid_data", "grid_status": "missing"}
        return cell_id, extract_cell_features(cell_id, cell_result)

    output_cache_path = Path(args.output_cache)
    output_cache = load_output_cache(output_cache_path)

    # Save+plot after every cell -- see run_held_injected_grid.py's main()
    # for the same fix and why.
    if args.jobs == 1:
        result_iter = (process(cid) for cid in cell_ids)
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid) for cid in cell_ids)

    status_counts: dict = {}
    n_done = 0
    for cell_id, features in result_iter:
        output_cache[cell_id] = features
        status_counts[features["status"]] = status_counts.get(features["status"], 0) + 1
        if not args.no_plot:
            cell_result = grid_cache.get(cell_id)
            if cell_result is not None:
                plot_cell_grid_features(cell_result, features, figures_dir, command,
                                        fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cell_ids)}] {cell_id}: {features['status']}")
    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\nCache written to {args.output_cache}")

    if not args.no_pca:
        pca_input = build_pca_matrix(output_cache, args.pca_grid_n, args.pca_nan_policy)
        if pca_input["status"] != "ok":
            print(f"\nPCA skipped: {pca_input['status']} (n_cells={pca_input.get('n_cells')})")
        else:
            pca_result = run_pca(pca_input)
            with open(args.pca_output, "wb") as handle:
                pickle.dump(pca_result, handle)
            print(f"\nPCA over {len(pca_result['cell_ids'])} cells written to {args.pca_output}")
            print(f"PC1 explains {pca_result['explained_variance_ratio'][0]:.1%} of variance, "
                 f"PC1+PC2 {sum(pca_result['explained_variance_ratio'][:2]):.1%}")
            if not args.no_plot:
                plot_pca(pca_result, figures_dir, command, args.figure_format)


if __name__ == "__main__":
    main()
