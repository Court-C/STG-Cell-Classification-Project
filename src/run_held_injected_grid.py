"""Held x injected current grid sweep (Step 2a/2b): for each cell, map how
firing pattern during a current step and post-inhibitory rebound (PIR)
behavior after it depend jointly on two current axes -- a sustained held
current (the operating point/context the cell is adapted to) and an
injected current that is the ABSOLUTE test-window current level (not added
on top of held). Both axes are restricted to <=0 nA (inhibitory-only
project phase).

held and injected are deliberately NOT combined additively: two
independently-≤0 currents stacked (held+injected) would routinely drive the
test window to roughly twice either axis's own meaningful range, which in
practice meant most of the grid just probed "too hyperpolarized to do
anything" territory (confirmed empirically -- this was the initial design
and was corrected after review). Instead, held only does two things: (a)
sets the pre-test adapted state the cell sits in before the test window
(different held levels can leave the cell in different slow-adaptation
states, e.g. IntCa, even for the same absolute test level), and (b) is the
level released back to after the test window, which is what a PIR response
must actually cross spike threshold from.

Per-trial protocol at each (held, injected) grid point:
  1. settle at the held current alone (warm-started from the cell's cached
     Iapp=0 y_ss, or continuation from an already-settled held level)
  2. apply injected (the absolute test-window current level, ignoring held)
     for a test window
  3. release back to the HELD level (not to 0) for a recovery window and
     watch for post-inhibitory rebound spiking

Range anchoring: both axes span [0, cell_floor_nA], where
cell_floor_nA = silencing_threshold_bracket_nA[0] - cell_floor_margin_nA
(1.0 nA past the cell's own confirmed quiescent level by default), read
from cell_silencing_thresholds.pkl (see find_silencing_threshold.py).

Sweep strategy is coarse-to-fine: a coarse NxM grid is swept first, then
edges where classification changes between neighbors (rebound-onset or
burst-onset boundaries) are recursively bisected and independently
re-simulated -- never interpolated -- since find_silencing_threshold.py
found this model shows path-dependent (hysteretic) settling near dynamical
transitions, so a coarse call about a boundary is not trustworthy at fine
resolution without re-confirmation.

This script covers 2a (coarse grid) and 2b (adaptive refinement) only.
Feature extraction (2c: Rin/tau, spike waveform, F-I slope, ISI stats,
burst stats, cross-cell PCA, etc.) is a separate future script that reads
this script's output cache.
"""

import pickle
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate, get_currents, CURRENT_NAMES
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)
from find_silencing_threshold import (constant_iapp_func, count_spikes_and_rate,
                                      compute_isis_ms, classify_burst_pattern,
                                      settle_at_level, _round_level,
                                      PROMINENCE_FRACTION, FLATLINE_MV,
                                      DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_SILENCING_CACHE_PATH)

SCHEMA_VERSION = 5  # bumped: capture test_v_min_pre_spike_mV / test_first_spike_ms so sag can be
                    # assessed from the pre-spike onset trough alone -- sag is a subthreshold Ih
                    # relaxation that can precede the first spike/burst even in a cell that goes on
                    # to fire, so it shouldn't be restricted to fully-silent test windows (previously
                    # test_v_min_mV/test_v_end_mV were whole-window stats that conflate this with
                    # arbitrary spike-phase/AHP values once the cell starts firing). Also fixed a
                    # get_or_settle_hold crash and added adaptive test-window extension (carried over
                    # from schema 4).

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_held_injected_grid.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "held_injected_grid"
DEFAULT_FIGURE_FORMAT = "svg"
DEFAULT_TEMP = 25.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

V_INDEX = 0
H_CURRENT_INDEX = CURRENT_NAMES.index("H")
CAT_CURRENT_INDEX = CURRENT_NAMES.index("CaT")


def _round_point(pt) -> tuple:
    h, i = pt
    return (_round_level(h), _round_level(i))


def get_cell_floor_nA(silencing_entry: dict, margin_nA: float):
    """Returns (cell_floor_nA, anchor_source), or None if this cell must be
    skipped -- no valid Step 1 silencing-threshold result to anchor from.
    """
    status = silencing_entry.get("status")
    if status == "ok":
        first_silent, _last_firing = silencing_entry["silencing_threshold_bracket_nA"]
        return first_silent - margin_nA, "bracket_first_silent"
    if status == "unsettled":
        return silencing_entry["first_silent_level_nA"] - margin_nA, "unsettled_first_silent_fallback"
    return None


# ---------------------------------------------------------------------------
# Per-point simulation
# ---------------------------------------------------------------------------

def settle_hold_level(params, held_nA, warm_start_state, dt, temp, reftemp,
                      chunk_s, max_settle_s, settle_rtol, min_peaks_for_rate) -> dict:
    """Settle at a constant held current, reusing Step 1's settle_at_level
    engine unchanged, plus the end-of-settle voltage (needed by a future
    sag-ratio feature; captured now so it isn't lost).
    """
    r = settle_at_level(params, warm_start_state, held_nA, dt, temp, reftemp,
                        chunk_s, max_settle_s, settle_rtol, min_peaks_for_rate)
    if r["blew_up"]:
        return r
    r["hold_v_end_mV"] = float(r["last_chunk_v"][-1]) if r["last_chunk_v"] is not None else float("nan")
    return r


def compute_pre_spike_sag_trough(v_test, t_test, sag_window_ms: float):
    """Minimum voltage from test-window onset up to whichever comes first:
    sag_window_ms elapsed, or the first spike. Isolates the passive,
    Ih-relaxation-style sag trough from post-spike AHP troughs, so it stays
    meaningful even for a test window that goes on to fire tonically or
    burst -- sag is a subthreshold phenomenon that precedes the first spike,
    not something restricted to windows that stay silent throughout.

    Deliberately does NOT also try to report a "recovered/relaxed" reference
    voltage at the window boundary (which would let a full trough-relative-
    to-relaxation sag *ratio* be computed the same way for firing and silent
    windows alike): confirmed directly that the sample(s) immediately before
    a detected spike peak are already deep in the fast Na+ upstroke, not a
    subthreshold "relaxed" value -- e.g. one VC08B6 point had samples at
    -30mV eight timesteps (dt=0.05ms) before the peak and +19mV one timestep
    before it, so "last sample before the peak" is not a well-defined
    recovery reference without proper spike-onset-boundary detection
    (derivative threshold-crossing or similar), which is out of scope for a
    cache-only Phase-1 feature. The trough itself doesn't have this problem
    since np.min over the pre-spike window is unaffected by how the window's
    *end* happens to land relative to the upstroke.

    Returns (trough_mV, first_spike_ms or None).
    """
    v_range = v_test.max() - v_test.min()
    first_spike_ms = None
    if v_range >= FLATLINE_MV:
        peaks, _ = find_peaks(v_test, prominence=v_range * PROMINENCE_FRACTION)
        if len(peaks) > 0:
            first_spike_ms = float(t_test[peaks[0]])
    window_end_ms = min(sag_window_ms, first_spike_ms) if first_spike_ms is not None else sag_window_ms
    mask = t_test <= window_end_ms
    if not np.any(mask):
        mask = t_test <= t_test[0]  # degenerate (window_end_ms before first sample) -- fall back to first sample
    return float(v_test[mask].min()), first_spike_ms


def run_test_and_recovery(params, hold_state, held_nA, injected_nA, hold_freq_hz, dt, temp, reftemp,
                          test_window_s, recovery_window_s, rebound_latency_min_ms,
                          min_isis_for_burst_test, isi_mode_prominence_frac, min_isi_ratio,
                          sag_window_ms: float = 500.0,
                          max_test_window_s: float = None, test_window_extend_factor: float = 2.0,
                          return_traces: bool = False) -> dict:
    """Test window at the ABSOLUTE injected current level (held is not added
    on top -- see module docstring) followed by a recovery window (released
    back to held) watched for post-inhibitory rebound.

    If the test window comes back "insufficient_data" (too few ISIs to run
    the bimodality test at all), the window is doubled (re-simulated from
    the same hold_state, not extended in place -- simpler, and the retry
    only ever happens for the minority of points that actually need it) up
    to max_test_window_s before giving up and reporting insufficient_data
    for real. This directly targets the dominant cause of insufficient_data:
    firing can be much slower near a boundary than the baseline-period-scaled
    default window assumes, exactly the lesson find_silencing_threshold.py
    already learned (its dedicated isi_window_s exists for the same reason).

    return_traces=True additionally returns the raw (t_ms, v_mV) arrays for
    both windows under "_trace_*" keys (underscore-prefixed since these are
    never meant to reach the persisted grid cache -- only plot_example_traces.py
    uses this, for on-demand illustrative figures; the grid sweep itself
    always calls this with the default False to keep the cache small).
    """
    if max_test_window_s is None:
        max_test_window_s = test_window_s
    Iapp_test = constant_iapp_func(injected_nA)

    window_s = test_window_s
    while True:
        try:
            t_test, states_test = simulate(params, window_s, temp, dt=dt, reftemp=reftemp,
                                           cis=hold_state, Iapp_func=Iapp_test)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            return {"blew_up": True, "error": f"test window: {exc}"}
        if not np.all(np.isfinite(states_test)):
            return {"blew_up": True, "error": "test window: non-finite trajectory"}

        v_test = states_test[:, V_INDEX]
        test_freq_hz, _n_peaks, test_is_flatline = count_spikes_and_rate(
            v_test, window_s * 1000.0, min_peaks_for_rate=2)
        if test_is_flatline:
            test_pattern = "silent"
            test_isi_short_ms = test_isi_long_ms = None
            test_n_isis = 0
            test_bimodality_metric = None
        else:
            isis_test = compute_isis_ms(v_test, t_test, PROMINENCE_FRACTION)
            burst_test = classify_burst_pattern(isis_test, min_isis_for_burst_test,
                                                isi_mode_prominence_frac, min_isi_ratio)
            test_pattern = burst_test["pattern"]
            test_isi_short_ms = burst_test["isi_short_ms"]
            test_isi_long_ms = burst_test["isi_long_ms"]
            test_n_isis = burst_test["n_isis"]
            test_bimodality_metric = burst_test["bimodality_metric"]

        if test_pattern != "insufficient_data" or window_s >= max_test_window_s:
            break
        window_s = min(window_s * test_window_extend_factor, max_test_window_s)

    test_v_min_pre_spike_mV, test_first_spike_ms = compute_pre_spike_sag_trough(
        v_test, t_test, sag_window_ms)

    test_result = {
        "test_pattern": test_pattern, "test_isi_short_ms": test_isi_short_ms,
        "test_isi_long_ms": test_isi_long_ms, "test_n_isis": test_n_isis,
        "test_bimodality_metric": test_bimodality_metric, "test_freq_hz": test_freq_hz,
        # test/recovery windows are fixed adaptive-duration ISI-capture windows
        # (like Step 1's dedicated isi_window_s), not a chunked settle-loop like
        # settle_at_level -- "settled" isn't a meaningful concept for the recovery
        # window in particular, since it's deliberately watching a transient
        # rebound decay, not waiting for a new steady state. Kept as None rather
        # than a fabricated bool.
        "test_settled": None,
        "test_v_min_mV": float(v_test.min()), "test_v_end_mV": float(v_test[-1]),
        "test_v_min_pre_spike_mV": test_v_min_pre_spike_mV,
        "test_first_spike_ms": test_first_spike_ms,
        "test_window_s_used": window_s,
    }

    # --- recovery window: release back to held, watch for rebound ---
    # (note: an early-return here only needs "blew_up"/"error"/test_result --
    # run_trial_point backfills every recovery_result/rebound_* key from
    # _TEST_RECOVERY_DEFAULTS for any point whose dict doesn't fully reach here)
    Iapp_recovery = constant_iapp_func(held_nA)
    try:
        t_rec, states_rec = simulate(params, recovery_window_s, temp, dt=dt, reftemp=reftemp,
                                     cis=states_test[-1], Iapp_func=Iapp_recovery)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {"blew_up": True, "error": f"recovery window: {exc}", **test_result}
    if not np.all(np.isfinite(states_rec)):
        return {"blew_up": True, "error": "recovery window: non-finite trajectory", **test_result}

    v_rec = states_rec[:, V_INDEX]
    recovery_v_min_mV = float(v_rec.min())
    recovery_v_final_mV = float(v_rec[-1])

    trough_idx = int(np.argmin(v_rec))
    currents_at_trough = get_currents(states_rec[trough_idx], params, temp, reftemp)
    rebound_peak_iH_nA = float(currents_at_trough[H_CURRENT_INDEX])
    rebound_peak_iCaT_nA = float(currents_at_trough[CAT_CURRENT_INDEX])

    # "Rebound" is only a meaningful concept relative to firing that was
    # actually suppressed during the test window -- a cell that kept firing
    # throughout (e.g. injected=0, or a mild step that barely slows it) will
    # always have *some* spike somewhere in a multi-second recovery window
    # regardless of any rebound mechanism, so evaluating rebound there just
    # detects "did the cell fire at all" (confirmed empirically: every
    # non-silenced coarse point showed rebound_occurred=True before this
    # gate was added, including the held=0/injected=0 unperturbed control).
    test_suppressed = (test_pattern == "silent"
                       or (hold_freq_hz > 0 and test_freq_hz < 0.5 * hold_freq_hz))

    if not test_suppressed:
        rebound_occurred, rebound_spike_count = False, 0
        rebound_latency_ms, rebound_peak_mV = None, None
        rebound_pattern = "not_applicable"
    else:
        if v_rec.max() - v_rec.min() < FLATLINE_MV:
            peak_times_ms = np.array([])
        else:
            peaks, _ = find_peaks(v_rec, prominence=(v_rec.max() - v_rec.min()) * PROMINENCE_FRACTION)
            peak_times_ms = t_rec[peaks]

        qualifying = peak_times_ms[peak_times_ms >= rebound_latency_min_ms]
        rebound_occurred = len(qualifying) > 0
        rebound_spike_count = int(len(qualifying))
        rebound_latency_ms = float(qualifying[0]) if rebound_occurred else None
        if rebound_occurred:
            first_idx = int(np.argmin(np.abs(t_rec - qualifying[0])))
            rebound_peak_mV = float(v_rec[first_idx])
        else:
            rebound_peak_mV = None

        if not rebound_occurred:
            rebound_pattern = "none"
        elif rebound_spike_count == 1:
            rebound_pattern = "single_spike"
        else:
            rebound_isis = np.diff(qualifying)
            rebound_burst = classify_burst_pattern(rebound_isis, min_isis_for_burst_test,
                                                   isi_mode_prominence_frac, min_isi_ratio)
            rebound_pattern = "bursting_rebound" if rebound_burst["pattern"] == "bursting" else "tonic_rebound"

    recovery_result = {
        "rebound_applicable": test_suppressed,
        "rebound_occurred": rebound_occurred, "rebound_spike_count": rebound_spike_count,
        "rebound_latency_ms": rebound_latency_ms, "rebound_peak_mV": rebound_peak_mV,
        "rebound_peak_iH_nA": rebound_peak_iH_nA, "rebound_peak_iCaT_nA": rebound_peak_iCaT_nA,
        "rebound_pattern": rebound_pattern, "recovery_settled": None,
        "recovery_v_min_mV": recovery_v_min_mV, "recovery_v_final_mV": recovery_v_final_mV,
    }

    result = {"blew_up": False, "error": None, **test_result, **recovery_result}
    if return_traces:
        result["_trace_t_test_ms"] = t_test
        result["_trace_v_test_mV"] = v_test
        result["_trace_t_rec_ms"] = t_rec
        result["_trace_v_rec_mV"] = v_rec
    return result


# Full set of test-/recovery-stage defaults, used both for a hold-stage
# blow-up (make_blew_up_point) and to backfill any keys a mid-trial blow-up
# inside run_test_and_recovery didn't reach -- every GridPoint is guaranteed
# to carry this full key set regardless of where in the trial it failed, so
# downstream code (e.g. plot_cell_grid) never needs to special-case a
# partial dict.
_TEST_RECOVERY_DEFAULTS = {
    "test_pattern": None, "test_isi_short_ms": None, "test_isi_long_ms": None,
    "test_n_isis": 0, "test_bimodality_metric": None, "test_freq_hz": 0.0,
    "test_settled": None, "test_v_min_mV": float("nan"), "test_v_end_mV": float("nan"),
    "test_v_min_pre_spike_mV": float("nan"), "test_first_spike_ms": None,
    "test_window_s_used": float("nan"),
    "rebound_applicable": False,
    "rebound_occurred": False, "rebound_spike_count": 0, "rebound_latency_ms": None,
    "rebound_peak_mV": None, "rebound_peak_iH_nA": None, "rebound_peak_iCaT_nA": None,
    "rebound_pattern": "not_applicable", "recovery_settled": None,
    "recovery_v_min_mV": float("nan"), "recovery_v_final_mV": float("nan"),
}


def make_blew_up_point(held_nA, injected_nA, source, error, hold_blew_up=False) -> dict:
    return {
        "held_nA": held_nA, "injected_nA": injected_nA, "source": source,
        "hold_settled": False, "hold_freq_hz": 0.0, "hold_is_flatline": False,
        "hold_v_end_mV": float("nan"), "hold_blew_up": hold_blew_up,
        **_TEST_RECOVERY_DEFAULTS,
        "blew_up": True, "error": error,
    }


def run_trial_point(params, hold_result, held_nA, injected_nA, dt, temp, reftemp,
                    test_window_s, recovery_window_s, rebound_latency_min_ms,
                    isi_kwargs, source) -> dict:
    if hold_result["blew_up"]:
        return make_blew_up_point(held_nA, injected_nA, source, hold_result.get("error"), hold_blew_up=True)

    tr = run_test_and_recovery(params, hold_result["final_state"], held_nA, injected_nA,
                               hold_result["freq_hz"] or 0.0, dt, temp, reftemp,
                               test_window_s, recovery_window_s,
                               rebound_latency_min_ms, **isi_kwargs)
    point = {
        "held_nA": held_nA, "injected_nA": injected_nA, "source": source,
        "hold_settled": hold_result["settled"], "hold_freq_hz": hold_result["freq_hz"] or 0.0,
        "hold_is_flatline": hold_result["is_flatline"], "hold_v_end_mV": hold_result["hold_v_end_mV"],
        "hold_blew_up": False,
        **_TEST_RECOVERY_DEFAULTS,
    }
    point.update(tr)
    return point


def get_or_settle_hold(params, held_nA, hold_settle_cache, y_ss, baseline_freq_hz,
                       dt, temp, reftemp, settle_kwargs) -> dict:
    """Settle at held_nA, reusing a cached settle if this exact (rounded)
    held level has already been solved, else warm-starting from whichever
    already-settled held level is numerically nearest. held=0 is free --
    reuses the cell's own cached Iapp=0 limit-cycle state.
    """
    key = _round_level(held_nA)
    if key in hold_settle_cache:
        return hold_settle_cache[key]
    if key == 0.0:
        result = {"blew_up": False, "settled": True, "freq_hz": baseline_freq_hz,
                  "is_flatline": False, "final_state": y_ss.copy(),
                  "hold_v_end_mV": float(y_ss[V_INDEX])}
        hold_settle_cache[key] = result
        return result
    # Only warm-start from a PREVIOUSLY SUCCESSFUL settle -- a blown-up cache
    # entry has no "final_state" to continue from (confirmed as a real crash:
    # a later held level whose numerically-nearest neighbor happened to be a
    # blown-up entry raised KeyError deep inside a parallel worker). held=0.0
    # is always seeded first and can never itself blow up (it's the cell's
    # own cached limit-cycle state, not simulated here), so there's always at
    # least one valid entry to fall back to.
    valid_keys = [k for k, r in hold_settle_cache.items() if not r["blew_up"]]
    nearest_key = min(valid_keys, key=lambda k: abs(k - key))
    warm_state = hold_settle_cache[nearest_key]["final_state"]
    result = settle_hold_level(params, key, warm_state, dt, temp, reftemp, **settle_kwargs)
    hold_settle_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# 2a: coarse grid
# ---------------------------------------------------------------------------

def _build_levels(step_nA: float, cell_floor_nA: float) -> list:
    """Steps down from 0 by a FIXED step size (matching Step 1's own
    coarse_step_nA convention) until reaching cell_floor_nA, rather than
    dividing this cell's own (non-round, cell-specific) floor into a fixed
    number of points. The latter was the original design and produced
    non-round, cell-specific step sizes (e.g. 0.68 nA for one cell, 3.2 nA
    for another) that made grid values look arbitrary and hard to compare
    across cells -- fixed step size gives round, physically interpretable
    values (0, -0.5, -1.0, ...) at the cost of point count varying by each
    cell's own range instead of being forced to a constant count.
    cell_floor_nA is always included as the final point even if it isn't an
    exact multiple of step_nA, so the swept range still reaches exactly as
    far as Step 1's threshold anchoring intended.
    """
    levels = [0.0]
    level = _round_level(-step_nA)
    while level > cell_floor_nA:
        levels.append(level)
        level = _round_level(level - step_nA)
    if levels[-1] != _round_level(cell_floor_nA):
        levels.append(_round_level(cell_floor_nA))
    return levels


def build_coarse_grid(params, y_ss, baseline_freq_hz, cell_floor_nA, args):
    held_levels = _build_levels(args.coarse_step_held_nA, cell_floor_nA)
    injected_levels = _build_levels(args.coarse_step_injected_nA, cell_floor_nA)

    settle_kwargs = dict(chunk_s=args.hold_settle_chunk_s, max_settle_s=args.max_hold_settle_s,
                         settle_rtol=args.hold_settle_rtol, min_peaks_for_rate=args.min_peaks_for_rate)
    isi_kwargs = dict(min_isis_for_burst_test=args.min_isis_for_burst_test,
                      isi_mode_prominence_frac=args.isi_mode_prominence_frac,
                      min_isi_ratio=args.min_isi_ratio,
                      sag_window_ms=args.sag_window_ms,
                      max_test_window_s=args.max_test_window_s,
                      test_window_extend_factor=args.test_window_extend_factor)

    baseline_period_s = 1.0 / baseline_freq_hz if baseline_freq_hz else 1.0
    test_window_s = args.fixed_test_window_s or max(args.min_test_window_s,
                                                     args.test_window_periods * baseline_period_s)
    recovery_window_s = args.fixed_recovery_window_s or max(args.min_recovery_window_s,
                                                             args.recovery_window_periods * baseline_period_s)

    grid = {}
    hold_settle_cache = {}
    for held in held_levels:
        hold_result = get_or_settle_hold(params, held, hold_settle_cache, y_ss, baseline_freq_hz,
                                         args.dt, args.temp, args.reftemp, settle_kwargs)
        for injected in injected_levels:
            point = run_trial_point(params, hold_result, held, injected, args.dt, args.temp, args.reftemp,
                                    test_window_s, recovery_window_s, args.rebound_latency_min_ms,
                                    isi_kwargs, source="coarse")
            grid[(held, injected)] = point

    return grid, hold_settle_cache, held_levels, injected_levels, test_window_s, recovery_window_s


# ---------------------------------------------------------------------------
# 2b: adaptive refinement (edge bisection)
# ---------------------------------------------------------------------------

def build_edge_list(held_levels, injected_levels):
    edges = []
    for held in held_levels:
        for j in range(len(injected_levels) - 1):
            edges.append(((held, injected_levels[j]), (held, injected_levels[j + 1])))
    for injected in injected_levels:
        for i in range(len(held_levels) - 1):
            edges.append(((held_levels[i], injected), (held_levels[i + 1], injected)))
    return edges


def _point_status(point) -> str:
    if point["blew_up"]:
        return "blew_up"
    if point["test_pattern"] == "insufficient_data":
        return "insufficient_data"
    return "confident"


def edge_is_boundary(grid, edge) -> bool:
    """An edge is worth bisecting only when BOTH endpoints are confidently
    classified and they disagree. "insufficient_data" (sparse/no firing,
    common across much of the grid since the whole point of this sweep is
    to explore near and beyond each cell's own 1D silencing threshold on
    both axes at once) is deliberately NOT treated as an automatic boundary
    trigger: it can't be resolved by bisecting further (the sparse-firing
    issue is systemic to the fixed test-window length, not a resolution
    problem), and earlier code that gave every "ambiguous" edge one free
    bisection attempt did not actually bound refinement cost -- each
    bisection creates two brand-new, never-before-attempted child edges, so
    with most of the grid legitimately "insufficient_data" the point count
    nearly doubled every depth (confirmed empirically: 49 -> 74 -> 136 -> 260
    -> 497 for one real cell, hitting the truncation safety valve every time).
    """
    a, b = grid[edge[0]], grid[edge[1]]
    if _point_status(a) != "confident" or _point_status(b) != "confident":
        return False
    rebound_differs = a["rebound_occurred"] != b["rebound_occurred"]
    burst_boundary = ("bursting" in (a["test_pattern"], b["test_pattern"])
                      and a["test_pattern"] != b["test_pattern"])
    return rebound_differs or burst_boundary


def refine_grid(params, grid, hold_settle_cache, y_ss, baseline_freq_hz, held_levels, injected_levels,
                test_window_s, recovery_window_s, args):
    settle_kwargs = dict(chunk_s=args.hold_settle_chunk_s, max_settle_s=args.max_hold_settle_s,
                         settle_rtol=args.hold_settle_rtol, min_peaks_for_rate=args.min_peaks_for_rate)
    isi_kwargs = dict(min_isis_for_burst_test=args.min_isis_for_burst_test,
                      isi_mode_prominence_frac=args.isi_mode_prominence_frac,
                      min_isi_ratio=args.min_isi_ratio,
                      sag_window_ms=args.sag_window_ms,
                      max_test_window_s=args.max_test_window_s,
                      test_window_extend_factor=args.test_window_extend_factor)

    depth = 0
    edges = build_edge_list(held_levels, injected_levels)
    refinement_truncated = False
    max_depth_reached = 0

    while depth < args.max_refine_depth:
        boundary_edges = [e for e in edges if edge_is_boundary(grid, e)]
        if not boundary_edges:
            break
        if len(boundary_edges) > args.max_boundary_edges_per_depth:
            refinement_truncated = True
            break

        next_edges = []
        for (p1, p2) in boundary_edges:
            held1, inj1 = p1
            held2, inj2 = p2
            edge_len = max(abs(held1 - held2), abs(inj1 - inj2))
            if edge_len < args.min_edge_nA:
                continue
            mid = _round_point(((held1 + held2) / 2.0, (inj1 + inj2) / 2.0))
            if mid not in grid:
                mid_held, mid_injected = mid
                hold_result = get_or_settle_hold(params, mid_held, hold_settle_cache, y_ss, baseline_freq_hz,
                                                 args.dt, args.temp, args.reftemp, settle_kwargs)
                grid[mid] = run_trial_point(params, hold_result, mid_held, mid_injected,
                                            args.dt, args.temp, args.reftemp,
                                            test_window_s, recovery_window_s, args.rebound_latency_min_ms,
                                            isi_kwargs, source=f"refine_depth_{depth + 1}")
            next_edges.append((p1, mid))
            next_edges.append((mid, p2))
        edges = next_edges
        depth += 1
        max_depth_reached = depth

    return grid, max_depth_reached, refinement_truncated


# ---------------------------------------------------------------------------
# Top-level per-cell orchestrator
# ---------------------------------------------------------------------------

def run_cell_grid(cell_id: str, params: np.ndarray, ss_entry, silencing_entry, args) -> dict:
    base = {"cell_id": cell_id, "params": params, "schema_version": SCHEMA_VERSION,
            "run_args": vars(args)}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}
    if silencing_entry is None:
        return {**base, "status": "no_silencing_threshold"}

    floor_result = get_cell_floor_nA(silencing_entry, args.cell_floor_margin_nA)
    if floor_result is None:
        return {**base, "status": "silencing_not_ok"}
    cell_floor_nA, floor_anchor_source = floor_result

    y_ss = ss_entry["y_ss"]
    baseline_freq_hz = ss_entry["freq_hz"]

    try:
        grid, hold_settle_cache, held_levels, injected_levels, test_window_s, recovery_window_s = \
            build_coarse_grid(params, y_ss, baseline_freq_hz, cell_floor_nA, args)
        grid, max_depth_reached, refinement_truncated = refine_grid(
            params, grid, hold_settle_cache, y_ss, baseline_freq_hz, held_levels, injected_levels,
            test_window_s, recovery_window_s, args)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {**base, "status": "blew_up", "error": str(exc)}

    n_points_by_source: dict = {}
    for point in grid.values():
        n_points_by_source[point["source"]] = n_points_by_source.get(point["source"], 0) + 1

    hold_states = {lvl: r["final_state"] for lvl, r in hold_settle_cache.items() if not r["blew_up"]}

    return {
        **base, "status": "ok",
        "cell_floor_nA": cell_floor_nA, "floor_anchor_source": floor_anchor_source,
        "coarse_held_levels_nA": np.array(held_levels), "coarse_injected_levels_nA": np.array(injected_levels),
        "test_window_s": test_window_s, "recovery_window_s": recovery_window_s,
        "grid": grid, "hold_states": hold_states,
        "refine_max_depth_reached": max_depth_reached, "refinement_truncated": refinement_truncated,
        "n_points_total": len(grid), "n_points_by_source": n_points_by_source,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_output_cache(cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> None:
    with open(cache_path, "wb") as handle:
        pickle.dump(cache, handle)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

PATTERN_COLORS = {"silent": "gray", "tonic": "steelblue", "bursting": "mediumpurple",
                  "insufficient_data": "lightgray"}


def plot_cell_grid(cell_result: dict, outdir: Path, command: str, fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    if cell_result["status"] != "ok" or not cell_result["grid"]:
        return
    cell_id = cell_result["cell_id"]
    grid = cell_result["grid"]
    held = np.array([p["held_nA"] for p in grid.values()])
    injected = np.array([p["injected_nA"] for p in grid.values()])
    is_coarse = np.array([p["source"] == "coarse" for p in grid.values()])
    sizes = np.where(is_coarse, 40, 14)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    patterns = np.array([p["test_pattern"] or "insufficient_data" for p in grid.values()])
    colors = [PATTERN_COLORS.get(pat, "black") for pat in patterns]
    axes[0, 0].scatter(held, injected, c=colors, s=sizes, edgecolor="none")
    axes[0, 0].set_title("test-window firing pattern", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=lbl)
              for lbl, c in PATTERN_COLORS.items()]
    axes[0, 0].legend(handles=handles, loc="best", fontsize=6)

    rebound = np.array([p["rebound_occurred"] for p in grid.values()])
    rebound_counts = np.array([p["rebound_spike_count"] for p in grid.values()])
    rcolors = np.where(rebound, "darkorange", "lightgray")
    rsizes = 14 + 8 * rebound_counts
    axes[0, 1].scatter(held, injected, c=rcolors, s=rsizes, edgecolor="none")
    axes[0, 1].set_title("rebound occurred (size = spike count)", fontsize=9)

    lat = np.array([p["rebound_latency_ms"] if p["rebound_occurred"] else np.nan for p in grid.values()])
    if np.any(~np.isnan(lat)):
        sc = axes[1, 0].scatter(held[~np.isnan(lat)], injected[~np.isnan(lat)],
                                c=lat[~np.isnan(lat)], cmap="viridis", s=30)
        fig.colorbar(sc, ax=axes[1, 0], label="ms")
    axes[1, 0].set_title("rebound latency", fontsize=9)

    iH = np.array([p["rebound_peak_iH_nA"] if p["rebound_peak_iH_nA"] is not None else np.nan
                  for p in grid.values()])
    if np.any(~np.isnan(iH)):
        sc = axes[1, 1].scatter(held[~np.isnan(iH)], injected[~np.isnan(iH)],
                                c=iH[~np.isnan(iH)], cmap="magma", s=30)
        fig.colorbar(sc, ax=axes[1, 1], label="nA")
    axes[1, 1].set_title("i_H at recovery trough", fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("held (nA)")
        ax.set_ylabel("injected (nA)")
        ax.invert_xaxis()
        ax.invert_yaxis()

    title = (f"{cell_id} — status: {cell_result['status']}, "
            f"cell_floor={cell_result['cell_floor_nA']:.2f} nA, "
            f"n_points={cell_result['n_points_total']}, "
            f"refine_depth={cell_result['refine_max_depth_reached']}"
            + (" [TRUNCATED]" if cell_result["refinement_truncated"] else ""))
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    fig.text(0.5, 0.005, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_grid.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep a 2D held x injected hyperpolarizing current grid per cell, "
                    "with adaptive refinement near rebound-onset and burst-onset boundaries. "
                    "Range is anchored per cell from Step 1's silencing threshold. "
                    "Depolarizing current is out of scope for this script.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) to sweep. Default: every cell found in --params-dir.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH,
                        help="Path to the Iapp=0 steady-state cache (input, read-only).")
    parser.add_argument("--silencing-cache", default=DEFAULT_SILENCING_CACHE_PATH,
                        help="Path to Step 1's cell_silencing_thresholds.pkl (input, read-only; "
                             "used to anchor each cell's grid range).")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--jobs", type=int, default=-1)

    parser.add_argument("--coarse-step-held-nA", type=float, default=0.5,
                        help="Fixed held-current step size for the coarse grid (0 down to cell_floor_nA), "
                             "matching Step 1's own coarse_step_nA convention -- gives round, "
                             "cell-comparable values rather than dividing each cell's own floor into a "
                             "fixed point count. Point count therefore varies per cell; a 2D grid at a "
                             "fixed step is quadratically more expensive for deep-threshold cells.")
    parser.add_argument("--coarse-step-injected-nA", type=float, default=0.5,
                        help="Fixed injected-current step size for the coarse grid.")
    parser.add_argument("--cell-floor-margin-nA", type=float, default=1.0,
                        help="Both axes span [0, silencing_threshold - margin]. "
                             "Margin gives headroom past the cell's own confirmed quiescent level.")

    parser.add_argument("--max-refine-depth", type=int, default=5,
                        help="Max edge-bisection depth. 2D refinement cost grows with the number "
                             "of surviving boundary edges per depth, so this is capped more "
                             "conservatively than Step 1's 1D fine sweep.")
    parser.add_argument("--min-edge-nA", type=float, default=0.02,
                        help="Absolute floor on edge length; stop bisecting an edge once it's this short "
                             "regardless of --max-refine-depth.")
    parser.add_argument("--max-boundary-edges-per-depth", type=int, default=500,
                        help="Safety valve: if a depth level would produce more boundary edges than this "
                             "(e.g. a pathologically noisy/checkerboarded cell), stop refining and flag "
                             "refinement_truncated=True rather than exploding compute.")

    parser.add_argument("--hold-settle-chunk-s", type=float, default=2.0)
    parser.add_argument("--max-hold-settle-s", type=float, default=20.0)
    parser.add_argument("--hold-settle-rtol", type=float, default=0.05)
    parser.add_argument("--min-peaks-for-rate", type=int, default=2)

    parser.add_argument("--test-window-periods", type=float, default=8.0,
                        help="Test window length = max(--min-test-window-s, this many baseline periods).")
    parser.add_argument("--min-test-window-s", type=float, default=3.0)
    parser.add_argument("--fixed-test-window-s", type=float, default=None,
                        help="If set, overrides the adaptive test-window formula with a fixed duration.")
    parser.add_argument("--recovery-window-periods", type=float, default=5.0,
                        help="Recovery window length = max(--min-recovery-window-s, this many baseline periods). "
                             "Shorter than the test window by design -- PIR is a fast phenomenon.")
    parser.add_argument("--min-recovery-window-s", type=float, default=2.0)
    parser.add_argument("--fixed-recovery-window-s", type=float, default=None,
                        help="If set, overrides the adaptive recovery-window formula with a fixed duration.")
    parser.add_argument("--rebound-latency-min-ms", type=float, default=5.0,
                        help="Minimum time after release before a spike counts as a rebound spike, "
                             "rather than one already in flight at the release instant.")

    parser.add_argument("--min-isis-for-burst-test", type=int, default=6)
    parser.add_argument("--isi-mode-prominence-frac", type=float, default=0.05)
    parser.add_argument("--min-isi-ratio", type=float, default=1.5)
    parser.add_argument("--sag-window-ms", type=float, default=500.0,
                        help="Window (from test-window onset) over which the pre-spike sag trough "
                             "(test_v_min_pre_spike_mV) is measured, truncated early if a spike occurs "
                             "first. Captures the passive Ih-relaxation sag independent of whether the "
                             "cell goes on to fire tonically/burst -- sag precedes the first spike, so "
                             "it isn't restricted to windows that stay silent throughout.")
    parser.add_argument("--max-test-window-s", type=float, default=20.0,
                        help="If a test window comes back 'insufficient_data' (too few ISIs to "
                             "classify at all), it's re-simulated with a longer window (doubled by "
                             "default, see --test-window-extend-factor) up to this cap before giving "
                             "up -- firing can be much slower near a boundary than the baseline-period "
                             "-scaled default window assumes.")
    parser.add_argument("--test-window-extend-factor", type=float, default=2.0,
                        help="Multiplier applied to the test window on each insufficient_data retry.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    ss_cache_path = Path(args.steady_state_cache)
    silencing_cache_path = Path(args.silencing_cache)
    output_cache_path = Path(args.output_cache)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_cells = load_all_cells(params_dir)
    if args.cells:
        unknown = sorted(set(args.cells) - set(all_cells))
        if unknown:
            raise SystemExit(f"Unknown cell id(s): {unknown}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Sweeping held x injected grid for {len(cells)} of {len(all_cells)} cell(s) from {params_dir}")

    silencing_cache = load_output_cache(silencing_cache_path)
    output_cache = load_output_cache(output_cache_path)

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        silencing_entry = silencing_cache.get(cell_id)
        result = run_cell_grid(cell_id, params, ss_entry, silencing_entry, args)
        return cell_id, result

    if args.jobs == 1:
        results = [process(cid, params) for cid, params in cells.items()]
    else:
        results = Parallel(n_jobs=args.jobs)(delayed(process)(cid, params) for cid, params in cells.items())

    status_counts: dict = {}
    for cell_id, result in results:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if not args.no_plot:
            plot_cell_grid(result, figures_dir, command, fig_format=args.figure_format)

    save_output_cache(output_cache, output_cache_path)

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    truncated = [cid for cid, result in results if result.get("refinement_truncated")]
    if truncated:
        print(f"\n{len(truncated)} cell(s) hit --max-boundary-edges-per-depth and were refinement_truncated: {truncated}")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    flagged = [cid for cid, result in results if result["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'no_silencing_threshold'/'silencing_not_ok': run find_silencing_threshold.py first "
              "(or check its status for that cell). 'blew_up': consider a smaller --dt.")


if __name__ == "__main__":
    main()
