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

Range anchoring: held spans [0, cell_floor_nA], where
cell_floor_nA = silencing_threshold_bracket_nA[0] - cell_floor_margin_nA
(1.0 nA past the cell's own confirmed quiescent level by default), read
from cell_silencing_thresholds.pkl (see find_silencing_threshold.py).
injected spans [0, cell_floor_nA - injected_floor_margin_nA] -- one nA
further than held by default -- so the test window can probe past the
deepest level held itself ever reaches; held stays the "lowest hold
current" reference point unchanged.

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

import os
import pickle
import sys
import time
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from scipy.interpolate import griddata
from scipy.spatial import QhullError

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate, get_currents, CURRENT_NAMES
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)
from find_silencing_threshold import (constant_iapp_func, count_spikes_and_rate,
                                      compute_isis_ms, classify_burst_pattern,
                                      to_stored_pattern,
                                      settle_at_level, _round_level,
                                      PROMINENCE_FRACTION, FLATLINE_MV,
                                      DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_SILENCING_CACHE_PATH)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_held_injected_grid.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "held_injected_grid"
DEFAULT_FIGURE_FORMAT = "svg"
# temp == reftemp: at dtemp=0, every q10^(dtemp/10) factor in
# singlecell_model_v1.py's _derivatives_core collapses to 1 regardless of a
# cell's own q10 parameters, so every cell's conductances/kinetics run at
# their literal reference-temperature values -- previously this and
# find_silencing_threshold.py ran at temp=25 (a 15C offset from reftemp),
# which q10-scaled conductances by up to ~2x between cells with identical
# reftemp values but different q10 profiles (confirmed directly on
# PWLDD3/QVVQF5's gH). generate_steady_state.py's baseline cache already
# used temp=reftemp=10; this brings Step 1/2 into agreement with it.
DEFAULT_TEMP = 10.0
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
    engine unchanged, plus the end-of-settle voltage (the pre-test baseline
    compute_pre_spike_sag_trough's sag depth is measured against).
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


def compute_adaptation_ratio(isis_ms: np.ndarray, edge_n: int):
    """Spike-rate adaptation within a single tonic firing episode: mean of
    the last edge_n ISIs over mean of the first edge_n ISIs (isis_ms is
    chronologically ordered -- see compute_isis_ms). >1 means firing slows
    over the course of the test window (e.g. 4QSWXH held=-3/inj=-2: ISIs go
    from ~22-27ms at onset to ~250-350ms by test end, ~8.3x).

    Requires n_isis >= 2*edge_n so the first-k/last-k windows never overlap;
    below that, returns None rather than a degenerate/fabricated ratio. Only
    ever called for test_pattern == "tonic" (see run_test_and_recovery) --
    a bursting train's ISIs mix intra-burst and inter-burst intervals, so a
    first-k-vs-last-k ratio there would measure burst structure, not smooth
    rate adaptation.

    An adaptation TIME CONSTANT (exponential fit of ISI vs spike index) was
    considered and deliberately not attempted: the 4QSWXH ISI sequence
    itself (22, 27, 47, 50, 49, ..., 156, 177, 206, 249, 348 ms) isn't
    obviously mono-exponential, and fitting one without validating the
    functional form against real traces first would repeat exactly what
    compute_pre_spike_sag_trough's own docstring above warns against.
    """
    if len(isis_ms) < 2 * edge_n:
        return None
    return float(np.mean(isis_ms[-edge_n:])) / float(np.mean(isis_ms[:edge_n]))


def run_test_and_recovery(params, hold_state, held_nA, injected_nA, hold_freq_hz, dt, temp, reftemp,
                          test_window_s, recovery_window_s, rebound_latency_min_ms,
                          min_isis_for_burst_test, isi_mode_prominence_frac, min_isi_ratio,
                          sag_window_ms: float = 500.0, adaptation_edge_n: int = 3,
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
    # Only ever assigned in the non-flatline branch below -- a window that's
    # flatline on every retry attempt would otherwise leave these undefined
    # for the post-loop adaptation/burst-structure computation.
    isis_test = None
    burst_test = None
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
            test_n_spikes = _n_peaks
            test_bimodality_metric = None
        else:
            isis_test, test_n_spikes = compute_isis_ms(v_test, t_test, PROMINENCE_FRACTION)
            burst_test = classify_burst_pattern(isis_test, min_isis_for_burst_test,
                                                isi_mode_prominence_frac, min_isi_ratio,
                                                n_peaks=test_n_spikes)
            test_pattern = burst_test["pattern"]
            test_isi_short_ms = burst_test["isi_short_ms"]
            test_isi_long_ms = burst_test["isi_long_ms"]
            test_n_isis = burst_test["n_isis"]
            test_bimodality_metric = burst_test["bimodality_metric"]

        # "sparse" (real but under-sampled firing, see classify_burst_pattern)
        # gets the same window-extension retry as "insufficient_data" -- more
        # window time is exactly what a too-few-spikes case needs, regardless
        # of which of the two labels it currently carries.
        if test_pattern not in ("insufficient_data", "sparse") or window_s >= max_test_window_s:
            break
        window_s = min(window_s * test_window_extend_factor, max_test_window_s)

    # Retry decision above is done -- collapse "insufficient_data"/"sparse"
    # (whichever it still is) down to "silent" for the reported label. See
    # to_stored_pattern's docstring for why this happens here, not earlier.
    test_pattern = to_stored_pattern(test_pattern)

    test_v_min_pre_spike_mV, test_first_spike_ms = compute_pre_spike_sag_trough(
        v_test, t_test, sag_window_ms)

    # Restricted to "tonic": a bursting train's first-k/last-k ISIs would mix
    # intra-/inter-burst intervals (see compute_adaptation_ratio docstring).
    test_adaptation_ratio = (compute_adaptation_ratio(isis_test, adaptation_edge_n)
                             if test_pattern == "tonic" else None)
    # n_bursts = n_long_isis + 1 (a "burst" is a maximal run of consecutive
    # short ISIs, so each inter-burst gap marks one more burst boundary);
    # spikes_per_burst is the average over the whole test window, not a
    # per-burst array -- see classify_burst_pattern's n_long_isis docstring.
    test_n_bursts = (burst_test["n_long_isis"] + 1
                     if test_pattern == "bursting" and burst_test is not None else None)
    test_spikes_per_burst = ((test_n_isis + 1) / test_n_bursts
                             if test_n_bursts else None)

    test_result = {
        "test_pattern": test_pattern, "test_isi_short_ms": test_isi_short_ms,
        "test_isi_long_ms": test_isi_long_ms, "test_n_isis": test_n_isis,
        "test_n_spikes": test_n_spikes,
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
        "test_adaptation_ratio": test_adaptation_ratio,
        "test_n_bursts": test_n_bursts, "test_spikes_per_burst": test_spikes_per_burst,
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
                                                   isi_mode_prominence_frac, min_isi_ratio,
                                                   n_peaks=rebound_spike_count)
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
    "test_n_isis": 0, "test_n_spikes": 0, "test_bimodality_metric": None, "test_freq_hz": 0.0,
    "test_settled": None, "test_v_min_mV": float("nan"), "test_v_end_mV": float("nan"),
    "test_v_min_pre_spike_mV": float("nan"), "test_first_spike_ms": None,
    "test_adaptation_ratio": None, "test_n_bursts": None, "test_spikes_per_burst": None,
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

SHALLOW_FLOOR_THRESHOLD_NA = 3.0
SHALLOW_FLOOR_STEP_NA = 0.25


def _adaptive_step(default_step_nA: float, floor_nA: float, min_points: int) -> float:
    """Shrinks the coarse-grid step for a shallow-floor cell so it still
    gets a reasonable number of levels on this axis, without ever
    coarsening a cell that already clears that bar at the default step.
    Confirmed directly this matters: at the 0.5nA default, a cell with
    floor=-3nA (e.g. 4QSWXH) gets only 7 held levels, while a cell with
    floor=-17nA (e.g. VC08B6) gets 36 -- a >5x density gap driven purely by
    how deep each cell's own floor happens to be, not by anything about the
    sweep itself.

    Two regimes:
    - |floor_nA| < SHALLOW_FLOOR_THRESHOLD_NA (3.0): flat SHALLOW_FLOOR_
      STEP_NA (0.25) -- a simple, round, predictable step for genuinely
      narrow-range cells, rather than the min-points formula's non-round
      value (e.g. 0.179nA for floor=-2.5), which made grid values harder to
      compare across shallow cells. Low added compute cost either way since
      the range itself is small.
    - |floor_nA| >= SHALLOW_FLOOR_THRESHOLD_NA: original min-points-based
      shrinking, unchanged. _build_levels produces roughly |floor_nA| /
      step + 1 levels (0 plus however many steps reach the floor, floor
      itself always included), so solving for the step that yields exactly
      min_points gives |floor_nA| / (min_points - 1); taking the min with
      default_step_nA means this can only ever shrink the step, never grow
      it past the default.
    """
    if floor_nA == 0:
        return default_step_nA
    if abs(floor_nA) < SHALLOW_FLOOR_THRESHOLD_NA:
        return min(default_step_nA, SHALLOW_FLOOR_STEP_NA)
    if min_points < 2:
        return default_step_nA
    guaranteed_step = abs(floor_nA) / (min_points - 1)
    return min(default_step_nA, guaranteed_step)


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


def build_coarse_grid(params, y_ss, baseline_freq_hz, cell_floor_nA, injected_floor_nA, args):
    held_step = _adaptive_step(args.coarse_step_held_nA, cell_floor_nA, args.min_coarse_points)
    injected_step = _adaptive_step(args.coarse_step_injected_nA, injected_floor_nA, args.min_coarse_points)
    held_levels = _build_levels(held_step, cell_floor_nA)
    injected_levels = _build_levels(injected_step, injected_floor_nA)

    settle_kwargs = dict(chunk_s=args.hold_settle_chunk_s, max_settle_s=args.max_hold_settle_s,
                         settle_rtol=args.hold_settle_rtol, min_peaks_for_rate=args.min_peaks_for_rate)
    isi_kwargs = dict(min_isis_for_burst_test=args.min_isis_for_burst_test,
                      isi_mode_prominence_frac=args.isi_mode_prominence_frac,
                      min_isi_ratio=args.min_isi_ratio,
                      sag_window_ms=args.sag_window_ms,
                      adaptation_edge_n=args.adaptation_edge_n,
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
    # test_pattern is only ever None (blew_up), "silent", "tonic", or
    # "bursting" now -- to_stored_pattern (see find_silencing_threshold.py)
    # already resolved "insufficient_data"/"sparse" into "silent" before a
    # point ever reaches here, after run_test_and_recovery's own window-
    # extension retry already gave it every chance to resolve into a real
    # tonic/bursting call. So every non-blew_up point is "confident" now --
    # there's no longer a separate "we genuinely don't know" bucket to
    # distinguish (see edge_is_boundary for why that distinction used to
    # matter for refinement cost).
    return "blew_up" if point["blew_up"] else "confident"


def edge_is_boundary(grid, edge) -> bool:
    """An edge is worth bisecting only when BOTH endpoints are confidently
    classified and they disagree. Only "blew_up" points are excluded from
    "confident" now (see _point_status) -- a "silent" label already means
    the point was given every chance to resolve into tonic/bursting via
    run_test_and_recovery's window-extension retry, so a "silent"-vs-
    "bursting" (or "silent"-vs-"tonic"/rebound-differs) edge is a genuine
    boundary worth resolving, not the kind of unresolvable ambiguity earlier
    code had to specifically guard against (an earlier version of this
    function treated "insufficient_data"/"sparse" points as automatically
    non-confident, since bisecting them couldn't help -- with those labels
    gone, that guard is no longer needed).
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
                      adaptation_edge_n=args.adaptation_edge_n,
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
    base = {"cell_id": cell_id, "params": params, "run_args": vars(args)}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}
    if silencing_entry is None:
        return {**base, "status": "no_silencing_threshold"}

    floor_result = get_cell_floor_nA(silencing_entry, args.cell_floor_margin_nA)
    if floor_result is None:
        return {**base, "status": "silencing_not_ok"}
    cell_floor_nA, floor_anchor_source = floor_result
    # Injected probes deeper than held: held stays the "lowest hold current"
    # reference (cell_floor_nA, unchanged), injected extends one more margin
    # step past it.
    injected_floor_nA = cell_floor_nA - args.injected_floor_margin_nA

    y_ss = ss_entry["y_ss"]
    baseline_freq_hz = ss_entry["freq_hz"]

    try:
        grid, hold_settle_cache, held_levels, injected_levels, test_window_s, recovery_window_s = \
            build_coarse_grid(params, y_ss, baseline_freq_hz, cell_floor_nA, injected_floor_nA, args)
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
        "cell_floor_nA": cell_floor_nA, "injected_floor_nA": injected_floor_nA,
        "floor_anchor_source": floor_anchor_source,
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
    # Write-then-rename rather than writing cache_path directly: this gets
    # called after every cell now (see main()'s incremental save), not just
    # once at the end, so a mid-write crash/kill or a concurrent reader
    # (e.g. checking progress on a still-running sweep) must never be able
    # to observe a truncated/corrupt pickle. os.replace is atomic on both
    # POSIX and Windows.
    #
    # os.replace can still transiently fail on Windows with PermissionError
    # (WinError 5) if something else briefly has cache_path open -- e.g. a
    # concurrent read of the cache while checking progress on a still-
    # running sweep (confirmed directly: this crashed a real 69-cell run).
    # POSIX rename doesn't have this problem (an open file handle doesn't
    # block a rename there), so retrying with a short backoff is enough --
    # it's a transient OS-level lock, not a real conflict.
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


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# test_pattern is exactly one of these three now -- to_stored_pattern (see
# find_silencing_threshold.py) collapses "insufficient_data"/"sparse" into
# "silent" before it's ever stored, so there's no fourth/fifth category to
# render here anymore.
PATTERN_COLORS = {"silent": "gray", "tonic": "steelblue", "bursting": "mediumpurple"}
# 150->200: more render pixels for the bilinear pixel-blending step (see
# _render_heatmap) to work with, so tile-boundary gradients look smoother
# without changing the underlying nearest-neighbor VALUE at any pixel --
# purely a rendering-resolution bump, not a data-fabrication concern.
HEATMAP_RESOLUTION = 200


def _fine_grid_coords(cell_floor_nA: float, injected_floor_nA: float, resolution: int = HEATMAP_RESOLUTION):
    """Regular (held, injected) grid in nA, spanning [floor, 0] on each axis
    at this cell's own resolution -- the render target every scattered
    (adaptively-sampled) point below gets interpolated onto, so the figure
    reads as a smooth heat map regardless of how sparse the underlying
    simulated points actually are (a coarse 7x9 sweep still fills the whole
    canvas; it just interpolates over more area per real data point).
    """
    # Ascending order (floor -> 0), matplotlib's natural default -- panels
    # call ax.invert_xaxis()/invert_yaxis() afterward (same as the previous
    # scatter-based version) to display 0 at the left/bottom edge. Building
    # this descending instead and skipping the invert calls would look
    # identical for a single panel, but would silently double-flip if
    # anything ever draws overlay artists (e.g. a marker at a specific
    # (held, injected)) using the un-inverted axes convention.
    held_coords = np.linspace(cell_floor_nA, 0, resolution)
    injected_coords = np.linspace(injected_floor_nA, 0, resolution)
    return np.meshgrid(held_coords, injected_coords)


def _merge_close_points(held, injected, values, tolerance_nA, is_categorical=False):
    """Groups real points whose (held, injected) coordinates fall within
    tolerance_nA of each other (snapped to a tolerance_nA grid) and averages
    their values into one representative point per group -- mean for a
    continuous quantity, mode (most common code) for a categorical one.

    This is where genuine "averaging between data points" happens, targeted
    specifically at near-duplicate point clusters: confirmed directly on
    real data (4QSWXH) that nearest-neighbor distances between real grid
    points span from 4e-6 nA to 0.214 nA -- a >53,000x spread -- because
    refine_grid's edge bisection keeps halving the gap between two points
    right up to --min-edge-nA, while coarse regions stay at the full coarse
    step. That extreme non-uniformity, not sparse resolution, is what made
    cubic interpolation ring/overshoot (Clough-Tocher triangulation is
    numerically ill-conditioned on near-zero-area triangles). tolerance_nA
    defaults to --min-edge-nA itself: refinement never intentionally places
    two distinct points closer than that, so anything closer than it is a
    near-duplicate artifact of the bisection process, not independent
    signal -- merging it isn't discarding real information.

    Returns (merged_held, merged_injected, merged_values), each 1D arrays,
    one entry per group.
    """
    finite = np.isfinite(values) if not is_categorical else np.ones(len(values), dtype=bool)
    held, injected, values = held[finite], injected[finite], values[finite]
    if len(held) == 0:
        return held, injected, values

    snapped_h = np.round(held / tolerance_nA).astype(np.int64)
    snapped_i = np.round(injected / tolerance_nA).astype(np.int64)
    groups: dict = {}
    for h, i, v, sh, si in zip(held, injected, values, snapped_h, snapped_i):
        groups.setdefault((sh, si), {"h": [], "i": [], "v": []})
        g = groups[(sh, si)]
        g["h"].append(h)
        g["i"].append(i)
        g["v"].append(v)

    merged_h = np.empty(len(groups))
    merged_i = np.empty(len(groups))
    merged_v = np.empty(len(groups))
    for idx, g in enumerate(groups.values()):
        merged_h[idx] = np.mean(g["h"])
        merged_i[idx] = np.mean(g["i"])
        if is_categorical:
            vals, counts = np.unique(g["v"], return_counts=True)
            merged_v[idx] = vals[np.argmax(counts)]
        else:
            merged_v[idx] = np.mean(g["v"])
    return merged_h, merged_i, merged_v


def _mask_long_triangles(pts, hh, ii, max_edge_multiple: float = 4.0):
    """Marks render pixels that fall inside a Delaunay triangle (over the
    real, merged point cloud) whose longest edge exceeds max_edge_multiple
    times the median real-point spacing.

    Linear interpolation is bounded by its triangle's own corner values, so
    it never fabricates a value outside what's real -- but stretched across
    an unusually long/thin triangle (common where an adaptively-refined
    point cloud goes from densely-sampled near a boundary to sparse
    elsewhere), the result reads visually as a "ray" implying smooth local
    structure over a distance where no nearby real data actually supports
    that reading. This flags exactly those triangles so the caller can fall
    back to honest nearest-neighbor there instead, keeping linear's smooth
    blending only where real local point density actually earns it.
    """
    from scipy.spatial import Delaunay, cKDTree
    if len(pts) < 4:
        return np.zeros(hh.shape, dtype=bool)
    tree = cKDTree(pts)
    nn_dist, _ = tree.query(pts, k=2)
    median_spacing = np.median(nn_dist[:, 1])
    if not np.isfinite(median_spacing) or median_spacing <= 0:
        return np.zeros(hh.shape, dtype=bool)
    max_edge = max_edge_multiple * median_spacing

    try:
        tri = Delaunay(pts)
    except QhullError:
        # Same degenerate/near-collinear point configurations _render_heatmap's
        # own linear-interpolation call can hit -- no masking is the safe
        # default (falls through to whatever _render_heatmap does, which
        # already has its own nearest-neighbor fallback for this).
        return np.zeros(hh.shape, dtype=bool)
    simplex_pts = pts[tri.simplices]
    edge_lengths = np.stack([
        np.linalg.norm(simplex_pts[:, 0] - simplex_pts[:, 1], axis=1),
        np.linalg.norm(simplex_pts[:, 1] - simplex_pts[:, 2], axis=1),
        np.linalg.norm(simplex_pts[:, 2] - simplex_pts[:, 0], axis=1),
    ], axis=1)
    long_simplex = edge_lengths.max(axis=1) > max_edge

    query_pts = np.column_stack([hh.ravel(), ii.ravel()])
    simplex_idx = tri.find_simplex(query_pts)
    mask = simplex_idx < 0  # outside the triangulation entirely -> also fall back
    inside = ~mask
    mask[inside] = long_simplex[simplex_idx[inside]]
    return mask.reshape(hh.shape)


def _render_heatmap(held, injected, values, hh, ii, tolerance_nA, is_categorical=False, mode="tile"):
    """Merge-then-render: _merge_close_points collapses near-duplicate real
    points into local averages (see its docstring, and the module's
    SCHEMA_VERSION-era investigation confirming that near-zero-distance
    point pairs -- not resolution -- caused the original cubic rendering to
    ring/overshoot). What happens after merging is controlled by `mode`:

    - "tile" (default): nearest-neighbor only, for both categorical and
      continuous quantities -- each real (merged) point owns a flat,
      delineated Voronoi-style tile out to its nearest neighbors, with no
      blending between tiles. Reverted to this as the default after the
      "linear" mode below (despite its two accuracy safeguards) still
      wasn't giving the reading the data actually supports -- tiling is the
      more honest baseline (every pixel's value traces to exactly one real
      point, never a fabricated blend), revisit smoothing separately later.
    - "linear": LINEAR interpolation for continuous quantities (still
      nearest-neighbor for categorical -- blending two category codes has
      no meaning, e.g. "tonic"=1 blended with "bursting"=2 could land near
      1.5, an intermediate category that doesn't exist). Linear is
      barycentric -- each rendered value is a weighted average of its
      surrounding triangle's three real (merged) corner values, so it's
      mathematically bounded by them and cannot ring/overshoot the way
      cubic did -- kept available (not deleted) for revisiting later, just
      not the active default.

    Returns an all-NaN grid if fewer than 1 real (merged) point survives.
    """
    m_held, m_injected, m_values = _merge_close_points(held, injected, values, tolerance_nA, is_categorical)
    if len(m_held) == 0:
        return np.full(hh.shape, np.nan)
    pts = np.column_stack([m_held, m_injected])
    if is_categorical or mode == "tile":
        return griddata(pts, m_values, (hh, ii), method="nearest")
    # mode == "linear" from here down.
    # Linear needs a genuine 2D point cloud to triangulate -- a sparse
    # feature (e.g. adaptation_ratio_map, tonic-only, or iH-at-trough now
    # restricted to fully-silenced points) can end up with too few merged
    # points, or points that are collinear/near-collinear along some
    # non-axis-aligned direction, either of which makes Qhull raise a hard
    # error instead of returning NaN. The obvious axis-aligned check
    # (np.ptp along held/injected separately) doesn't catch every
    # degenerate case -- confirmed directly (a 4-point QhullError on a real
    # cell whose points passed that check but were still coplanar/degenerate
    # for Qhull's lifted-paraboloid Delaunay construction) -- so this
    # catches the error directly rather than trying to predict it.
    if len(m_held) < 3:
        return griddata(pts, m_values, (hh, ii), method="nearest")
    try:
        rendered = griddata(pts, m_values, (hh, ii), method="linear")
    except QhullError:
        return griddata(pts, m_values, (hh, ii), method="nearest")
    # Reject pixels whose triangle stretches too far relative to local real
    # point density (see _mask_long_triangles) -- these render as visually
    # misleading "rays", still bounded by real values but implying smooth
    # structure the nearby data doesn't actually support. Route them (and
    # the pre-existing outside-convex-hull NaNs) to nearest-neighbor.
    nan_mask = np.isnan(rendered) | _mask_long_triangles(pts, hh, ii)
    if nan_mask.any() and not nan_mask.all():
        nearest = griddata(pts, m_values, (hh, ii), method="nearest")
        rendered[nan_mask] = nearest[nan_mask]
    return rendered


def plot_cell_grid(cell_result: dict, outdir: Path, command: str, fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    if cell_result["status"] != "ok" or not cell_result["grid"]:
        return
    cell_id = cell_result["cell_id"]
    grid = cell_result["grid"]
    held = np.array([p["held_nA"] for p in grid.values()])
    injected = np.array([p["injected_nA"] for p in grid.values()])
    hh, ii = _fine_grid_coords(cell_result["cell_floor_nA"], cell_result["injected_floor_nA"])
    extent = (cell_result["cell_floor_nA"], 0, cell_result["injected_floor_nA"], 0)
    # Same tolerance this cell's own sweep used to decide "two points are
    # meaningfully distinct" (see _merge_close_points) -- not a separate,
    # possibly-mismatched constant.
    merge_tolerance_nA = cell_result["run_args"]["min_edge_nA"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))

    pattern_names = list(PATTERN_COLORS.keys())
    # test_pattern is None for a point whose test window never actually ran
    # (e.g. hold_blew_up -- see _TEST_RECOVERY_DEFAULTS) -- "silent" is the
    # nearest available bucket now that "insufficient_data" is gone (there's
    # no meaningful data to distinguish it from either).
    pattern_codes = np.array([pattern_names.index(p["test_pattern"] or "silent")
                              for p in grid.values()], dtype=float)
    pattern_grid = _render_heatmap(held, injected, pattern_codes, hh, ii, merge_tolerance_nA,
                                   is_categorical=True)
    pattern_cmap = ListedColormap(list(PATTERN_COLORS.values()))
    pattern_norm = BoundaryNorm(np.arange(len(pattern_names) + 1) - 0.5, pattern_cmap.N)
    # Colormap the category codes to RGBA ourselves, THEN let imshow blend
    # that RGBA image with bilinear interpolation -- blending already-mapped
    # colors pixel-to-pixel is a safe, purely visual antialiasing of the
    # boundary (softens the jagged nearest-neighbor staircase). Letting
    # imshow interpolate the raw integer CODES instead (interpolation=
    # "bilinear" on pattern_grid directly) would average category indices
    # (e.g. "silent"=0 blended with "tonic"=1 could land near 0.5) and get
    # reassigned to whichever bucket BoundaryNorm puts that fractional value
    # in -- a fabricated intermediate category, not a real one.
    pattern_rgba = pattern_cmap(pattern_norm(pattern_grid))
    axes[0, 0].imshow(pattern_rgba, origin="lower", extent=extent, aspect="auto",
                      interpolation="bilinear")
    axes[0, 0].set_title("test-window firing pattern", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=lbl)
              for lbl, c in PATTERN_COLORS.items()]
    axes[0, 0].legend(handles=handles, loc="best", fontsize=6)

    # Firing rate is otherwise invisible in the categorical panel above (a
    # near-rheobase 2 Hz tonic point and a fast 40 Hz one both just read
    # "tonic") -- shown for every non-silent point regardless of pattern
    # (bursting included: test_freq_hz is the whole-window spike count over
    # window duration, still a meaningful rate even when spikes cluster into
    # bursts) since it's already computed for every point via
    # count_spikes_and_rate, not just the confidently-classified ones.
    freq = np.array([p["test_freq_hz"] if p["test_pattern"] != "silent" else np.nan
                     for p in grid.values()])
    freq_grid = _render_heatmap(held, injected, freq, hh, ii, merge_tolerance_nA)
    im = axes[0, 1].imshow(freq_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis",
                           interpolation="bicubic")
    fig.colorbar(im, ax=axes[0, 1], label="Hz")
    axes[0, 1].set_title("test-window firing rate (blank = silent)", fontsize=9)

    # spikes_per_burst is only meaningful where test_pattern == "bursting"
    # (see run_test_and_recovery's test_n_bursts/test_spikes_per_burst,
    # computed only in that branch) -- exactly the quantity today's
    # avg_spikes_per_burst >= 1.5 fix (see classify_burst_pattern) now
    # gates on, so this panel is the direct visual check that the fix
    # behaves sensibly across a whole cell's grid, not just the handful of
    # points spot-checked during calibration.
    spb = np.array([p["test_spikes_per_burst"] if p["test_pattern"] == "bursting" else np.nan
                    for p in grid.values()])
    spb_grid = _render_heatmap(held, injected, spb, hh, ii, merge_tolerance_nA)
    im = axes[0, 2].imshow(spb_grid, origin="lower", extent=extent, aspect="auto", cmap="plasma",
                           interpolation="bicubic")
    fig.colorbar(im, ax=axes[0, 2], label="spikes/burst")
    axes[0, 2].set_title("spikes per burst (bursting points only)", fontsize=9)

    # rebound_spike_count where applicable+occurred (0 where applicable but
    # didn't occur); NaN (excluded from interpolation) where rebound wasn't
    # even applicable -- those points' recovery windows were never
    # meaningfully evaluated for rebound onset (see run_test_and_recovery's
    # test_suppressed gate), so a count of 0 there would be a different,
    # misleading claim from "checked and found none".
    rebound_counts = np.array([float(p["rebound_spike_count"]) if p["rebound_applicable"] else np.nan
                               for p in grid.values()])
    rebound_grid = _render_heatmap(held, injected, rebound_counts, hh, ii, merge_tolerance_nA)
    im = axes[1, 0].imshow(rebound_grid, origin="lower", extent=extent, aspect="auto", cmap="Oranges",
                           interpolation="bicubic")
    fig.colorbar(im, ax=axes[1, 0], label="rebound spike count")
    axes[1, 0].set_title("rebound spike count (blank = not applicable)", fontsize=9)

    lat = np.array([p["rebound_latency_ms"] if p["rebound_occurred"] else np.nan for p in grid.values()])
    lat_grid = _render_heatmap(held, injected, lat, hh, ii, merge_tolerance_nA)
    im = axes[1, 1].imshow(lat_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis",
                           interpolation="bicubic")
    fig.colorbar(im, ax=axes[1, 1], label="ms")
    axes[1, 1].set_title("rebound latency", fontsize=9)

    # trough_idx = argmin(v_rec) (see run_test_and_recovery) is only a
    # genuine post-inhibitory-rebound trough when the cell was actually
    # silenced during the test window -- rebound_applicable/test_suppressed
    # alone isn't strict enough (it allows a still-firing "tonic" test
    # window as long as its rate dropped enough), so for a point that kept
    # firing, argmin(v_rec) can land on an arbitrary ongoing spike's AHP
    # trough instead, contaminating the value with spike-phase noise. Same
    # category of issue already found and fixed for sag
    # (test_v_min_pre_spike_mV) -- confirmed directly here too: one 9GBDEX
    # point (held=-0.8, inj=-4.71) reports iH=-6.1 nA against a neighbor
    # median of -2.0 nA, while test_pattern there is "tonic" not "silent".
    iH = np.array([p["rebound_peak_iH_nA"] if (p["rebound_peak_iH_nA"] is not None
                                                and p["test_pattern"] == "silent") else np.nan
                  for p in grid.values()])
    iH_grid = _render_heatmap(held, injected, iH, hh, ii, merge_tolerance_nA)
    im = axes[1, 2].imshow(iH_grid, origin="lower", extent=extent, aspect="auto", cmap="magma",
                           interpolation="bicubic")
    fig.colorbar(im, ax=axes[1, 2], label="nA")
    axes[1, 2].set_title("i_H at recovery trough (test window fully silenced only)", fontsize=9)

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
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    # Sharp bands in rebound latency/spike-count are usually real, not an
    # interpolation limit: confirmed directly (4QSWXH) that latency's
    # nearest-neighbor point-to-point jump is 10x larger relative to its
    # range at the 90th percentile than at the median -- a bimodal split
    # between two rebound regimes a few tenths of a nA apart, consistent
    # with this model's already-documented path-dependent (hysteretic)
    # settling near dynamical transitions (see module docstring). i_H is
    # smooth because it's a state-space value at a fixed instant, not a
    # spike-timing quantity that can flip discretely between regimes.
    fig.text(0.5, 0.025,
             "Sharp bands in rebound latency/spike-count typically reflect real hysteretic "
             "transitions between rebound regimes, not interpolation resolution.",
             ha="center", va="bottom", fontsize=6.5, style="italic", color="dimgray")
    fig.text(0.5, 0.005, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_grid.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=180)
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
                             "fixed step is quadratically more expensive for deep-threshold cells. Only "
                             "used as-is for cells deep enough to already clear --min-coarse-points at "
                             "this step -- shallower cells get an automatically finer, cell-specific step.")
    parser.add_argument("--coarse-step-injected-nA", type=float, default=0.5,
                        help="Fixed injected-current step size for the coarse grid.")
    parser.add_argument("--min-coarse-points", type=int, default=15,
                        help="Minimum coarse-grid levels guaranteed per axis: the fixed step above "
                             "is shrunk (never grown) for a cell whose own floor is shallow enough "
                             "that it would otherwise fall below this count -- confirmed directly a "
                             "0.5nA step gives a deep cell like ZE23IV (floor=-20.3) 42 held levels "
                             "but a shallow cell like 4QSWXH (floor=-3.0) only 7, a >5x density gap "
                             "driven purely by floor depth. Set to 0 or 1 to disable (always use the "
                             "fixed step).")
    parser.add_argument("--cell-floor-margin-nA", type=float, default=1.0,
                        help="Held spans [0, silencing_threshold - margin] (cell_floor_nA). "
                             "Margin gives headroom past the cell's own confirmed quiescent level. "
                             "Injected spans further still -- see --injected-floor-margin-nA.")
    parser.add_argument("--injected-floor-margin-nA", type=float, default=1.0,
                        help="Injected spans [0, cell_floor_nA - this], i.e. this much further past "
                             "the cell's confirmed quiescent level than held's own floor (cell_floor_nA) "
                             "-- held stays anchored at cell_floor_nA.")

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
    parser.add_argument("--adaptation-edge-n", type=int, default=3,
                        help="Number of ISIs averaged at each end of a tonic test window to compute "
                             "test_adaptation_ratio (mean of the last N ISIs / mean of the first N "
                             "ISIs). Requires test_n_isis >= 2x this value or the point reports None. "
                             "Coupled to --min-isis-for-burst-test in practice: since test_pattern == "
                             "'tonic' already requires >= --min-isis-for-burst-test ISIs (default 6), "
                             "the default of 3 here guarantees every tonic point qualifies -- raising "
                             "this without also raising --min-isis-for-burst-test will silently make "
                             "many tonic points report None.")
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

    # Save+plot after EVERY cell, not once at the end -- a 69-cell sweep can
    # run 30+ minutes, and the old batch-at-the-end behavior meant nothing
    # was visible on disk (cache or figures) until the entire run finished,
    # even though most cells complete far earlier. return_as="generator_
    # unordered" (vs. the default "list") is what makes this possible with
    # joblib: results stream back as each worker finishes, in completion
    # order, rather than Parallel() blocking until every job is done before
    # returning anything. save_output_cache's write-then-rename means a
    # reader checking progress mid-run never sees a torn/partial file.
    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items())

    status_counts: dict = {}
    n_done = 0
    for cell_id, result in result_iter:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if not args.no_plot:
            plot_cell_grid(result, figures_dir, command, fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cells)}] {cell_id}: {result['status']}")

    # Derived from output_cache (not a `results` list -- the generator
    # consumed above is exhausted, and per-cell entries already live in
    # output_cache via the incremental-save loop) filtered to just the
    # cells this run actually processed.
    this_run_results = [(cid, output_cache[cid]) for cid in cells]

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    truncated = [cid for cid, result in this_run_results if result.get("refinement_truncated")]
    if truncated:
        print(f"\n{len(truncated)} cell(s) hit --max-boundary-edges-per-depth and were refinement_truncated: {truncated}")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    flagged = [cid for cid, result in this_run_results if result["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'no_silencing_threshold'/'silencing_not_ok': run find_silencing_threshold.py first "
              "(or check its status for that cell). 'blew_up': consider a smaller --dt.")


if __name__ == "__main__":
    main()
