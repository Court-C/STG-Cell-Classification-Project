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

Sweep strategy (2026-08-11): a simple uniform NxM grid -- args.n_points_held
x args.n_points_injected evenly-spaced levels from 0 to each axis's own
floor, the SAME point count for every cell regardless of floor depth.
Replaces an earlier two-stage coarse-grid-then-boundary-driven-adaptive-
refinement design (fixed-nA-step coarse grid, edges where classification
changed between neighbors recursively bisected and re-simulated) that
produced wildly uneven per-cell point counts -- deliberately dropped in
favor of standardizing simulation count across the population, at the cost
of no longer concentrating extra samples right at classification
boundaries.

This script covers Step 2 (the grid sweep) only. Feature extraction (2c:
Rin/tau, spike waveform, F-I slope, ISI stats, burst stats, cross-cell PCA,
etc.) lives in extract_grid_features.py, which reads this script's output
cache.
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
                                      PROMINENCE_FRACTION, FLATLINE_MV, _spike_prominence,
                                      DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_SILENCING_CACHE_PATH)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_held_injected_grid.pkl"
# This script no longer renders a figure itself (2026-08-11 merge) -- the
# merged panel figure needs both this cache's grid AND extract_grid_
# features.py's features dict, so it's built and called from there. Kept
# here (not moved) since plot_cell_grid_features itself still lives in
# this module, alongside _fine_grid_coords/_render_heatmap.
DEFAULT_FIGURE_FORMAT = "png"
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

def classify_hold_pattern(chunk_v, chunk_t, min_isis_for_burst_test, isi_mode_prominence_frac,
                          min_isi_ratio, min_ashman_d: float = 2.0,
                          ratio_override_ashman_d: float = 5.0,
                          ratio_override_min_ratio: float = 1.15) -> str:
    """tonic/bursting/silent for a held level's OWN steady-state firing,
    from whichever settle chunk is already being simulated anyway (the
    last hold-settle chunk for held!=0, the cached Iapp=0 burn-in trace for
    held=0) -- zero extra simulation cost, same compute_isis_ms +
    classify_burst_pattern + to_stored_pattern pipeline find_silencing_
    threshold.py's own baseline (Iapp=0) probe already uses. Added
    2026-08-13 so test_suppressed can compare test_pattern against what
    this held level ACTUALLY produces, not just its average rate (see
    run_test_and_recovery's test_suppressed docstring).
    """
    if chunk_v is None:
        return "silent"
    isis_ms, n_peaks = compute_isis_ms(chunk_v, chunk_t, PROMINENCE_FRACTION)
    burst = classify_burst_pattern(isis_ms, min_isis_for_burst_test, isi_mode_prominence_frac,
                                   min_isi_ratio, min_ashman_d=min_ashman_d,
                                   ratio_override_ashman_d=ratio_override_ashman_d,
                                   ratio_override_min_ratio=ratio_override_min_ratio, n_peaks=n_peaks)
    return to_stored_pattern(burst["pattern"])


def settle_hold_level(params, held_nA, warm_start_state, dt, temp, reftemp,
                      chunk_s, max_settle_s, settle_rtol, min_peaks_for_rate,
                      min_isis_for_burst_test, isi_mode_prominence_frac, min_isi_ratio,
                      min_ashman_d: float = 2.0, ratio_override_ashman_d: float = 5.0,
                      ratio_override_min_ratio: float = 1.15) -> dict:
    """Settle at a constant held current, reusing Step 1's settle_at_level
    engine unchanged, plus the end-of-settle voltage (the pre-test baseline
    compute_pre_spike_sag_trough's sag depth is measured against) and this
    held level's own tonic/bursting/silent classification (see
    classify_hold_pattern).
    """
    r = settle_at_level(params, warm_start_state, held_nA, dt, temp, reftemp,
                        chunk_s, max_settle_s, settle_rtol, min_peaks_for_rate)
    if r["blew_up"]:
        return r
    r["hold_v_end_mV"] = float(r["last_chunk_v"][-1]) if r["last_chunk_v"] is not None else float("nan")
    r["hold_pattern"] = classify_hold_pattern(r["last_chunk_v"], r["last_chunk_t"],
                                              min_isis_for_burst_test, isi_mode_prominence_frac,
                                              min_isi_ratio, min_ashman_d, ratio_override_ashman_d,
                                              ratio_override_min_ratio)
    # Trough of the held level's OWN last settle chunk -- for a flatline hold
    # this equals hold_v_end_mV (a constant trace's min is its endpoint), but
    # for a hold that's itself tonically/burst firing at rest (e.g. XB2IQX at
    # held=-3nA, 19.5Hz), hold_v_end_mV is just whatever single sample the
    # chunked settle loop happened to stop on -- effectively a random phase
    # of the oscillation (confirmed: -39.8mV, mid-cycle, for a trace that
    # spans spike peak down to ~-70mV AHP). That single sample is not a valid
    # "resting baseline" a sag depth can be measured against, which is why
    # compute_sag_depth_map used to gate on hold_is_flatline and simply skip
    # every oscillating-hold point -- silently, not as a real 0. Using the
    # already-fully-simulated last_chunk_v's own minimum instead gives a
    # well-defined, reproducible reference (how far the cell already dips on
    # its own) that sag can be measured beyond, with no extra simulation and
    # no is_flatline branching needed downstream.
    r["hold_v_trough_mV"] = float(np.min(r["last_chunk_v"])) if r["last_chunk_v"] is not None else float("nan")
    return r


def compute_pre_spike_sag_trough(v_test, t_test, sag_window_ms: float,
                                 spike_dead_zone_ms: float = 5.0):
    """Minimum voltage from test-window onset up to whichever comes first:
    sag_window_ms elapsed, or the first spike. Isolates the passive,
    Ih-relaxation-style sag trough from post-spike AHP troughs, so it stays
    meaningful even for a test window that goes on to fire tonically or
    burst -- sag is a subthreshold phenomenon that precedes the first spike,
    not something restricted to windows that stay silent throughout.

    spike_dead_zone_ms excludes any "spike" within this many ms of test
    onset from counting as the first spike, mirroring the exact same
    already-in-flight-at-release guard run_test_and_recovery's recovery-
    window rebound detection uses (rebound_latency_min_ms). Needed because
    the test window continues directly from the held level's own final
    state -- if the held level fires spontaneously (common: e.g. XB2IQX
    fires even at held=0), that state can land mid-upstroke of a spike
    already committed before the injected step ever took effect. Without
    this guard, that leftover spike fires ~1ms into the test window
    regardless of injected depth, truncating the sag window down to ~1ms
    and understating (or corrupting) the sag reading -- confirmed directly:
    22/120 points on a real XB2IQX 0.5nA-resolution grid showed a
    first_spike_ms under 5ms, concentrated at held=0 and weak injected
    steps, each reporting the exact same ~1ms first-spike time regardless
    of how deep injected was.

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
        peaks, _ = find_peaks(v_test, prominence=_spike_prominence(v_range, PROMINENCE_FRACTION))
        if len(peaks) > 0:
            peak_times_ms = t_test[peaks]
            qualifying = peak_times_ms[peak_times_ms >= spike_dead_zone_ms]
            if len(qualifying) > 0:
                first_spike_ms = float(qualifying[0])
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


def detect_onset_burst(isis_ms: np.ndarray, min_isi_ratio: float, min_onset_isis: int = 2):
    """A leading run of consecutive short ISIs at the very START of the test
    window, immediately followed by a jump of at least min_isi_ratio to the
    next ISI. This is classify_burst_pattern's own short/long timescale
    separation (same min_isi_ratio, same "meaningfully different timescale,
    not just noise" reasoning), applied locally to the first few ISIs
    instead of globally via KDE across the whole train -- so it can catch a
    genuine onset burst even in a window whose whole-window classification
    comes back "tonic" because what follows the burst isn't a second clean
    KDE mode.

    Confirmed case this exists for: XB2IQX held=-2.48/inj=-4.04, ISIs
    13.1, 13.5, 16.8 | 41.0, 60.0, 30.0, 61.0, 42.2, 72.2, 53.7, 111.7 (ms).
    The whole-window KDE test calls this "tonic" (bimodality_metric=0.0) --
    the post-burst ISIs span 30-112ms, too wide/noisy to form their own
    tight KDE peak -- so classify_burst_pattern never sees the burst at
    all, and test_n_bursts/test_spikes_per_burst are never computed for a
    window that plainly opens with one. Scanning locally for the first big
    jump sidesteps needing the tail to look like anything in particular.

    min_onset_isis=2 (i.e. >=3 spikes) mirrors classify_burst_pattern's own
    min_spikes_per_burst=1.5 in spirit: a single short ISI followed by a
    jump is indistinguishable from ordinary tonic jitter, but two
    consecutive short ISIs both followed by the same jump is a real
    burst-shaped feature.

    Returns (onset_n_spikes, onset_isi_mean_ms), or (None, None) if no
    qualifying leading run is found (including: the whole train is one
    consistent cluster, with no jump at all -- already handled by the
    whole-window classifier in that case, nothing extra to report here).
    """
    n = len(isis_ms)
    if n < min_onset_isis + 1:
        return None, None
    for j in range(min_onset_isis, n):
        if isis_ms[j] / isis_ms[j - 1] >= min_isi_ratio:
            onset_isis = isis_ms[:j]
            return int(j + 1), float(np.mean(onset_isis))
    return None, None


def cessation_reference_isi_ms(isis_ms: np.ndarray, recent_n: int = 5) -> float:
    """Reference ISI scale for judging whether trailing silence after the
    last spike is genuine cessation vs. ordinary jitter -- used everywhere
    a "trailing_silence >= trailing_silence_ratio * <reference>" check is
    made. min(last ISI, median of the last recent_n ISIs) rather than the
    last ISI alone: when a train is winding down, the last ISI itself can
    already be an outlier gap (a symptom of stopping, not a "normal"
    interval), which inflates the cessation bar and can mask a real stop.
    Taking the min never LOOSENS an already-triggered cessation call (it
    only ever shrinks the reference), so this adds sensitivity without
    causing any prior true "ceased" case to flip false. Confirmed real case
    (user-flagged, 2026-08-14), XB2IQX held=-3.489796/inj=-2.132653: a
    smooth 12.9ms->96.3ms deceleration (19 ISIs), then one 442.9ms outlier
    gap, one final isolated spike, then 673ms of genuine trailing silence --
    673ms clears 3x the ~60-90ms recent trend but not 3x the 442.9ms outlier
    gap itself, so the last-ISI-only version missed it. User's framing: the
    final spike is "an extra spike not part of the burst" but the episode
    as a whole should still read as bursting, i.e. the same "genuine
    cessation implies burst" principle, just applied with a reference scale
    that isn't itself corrupted by the anomaly it's supposed to detect.
    """
    last = float(isis_ms[-1])
    recent = isis_ms[-recent_n:] if len(isis_ms) > 0 else isis_ms
    return min(last, float(np.median(recent)))


def classify_rebound_pattern(rebound_isis: np.ndarray, rebound_trailing_silence_ms: float,
                             min_isis_for_burst_test: int, isi_mode_prominence_frac: float,
                             min_isi_ratio: float, n_peaks: int, min_onset_isis: int = 2,
                             rebound_min_onset_isis: int = 1, trailing_silence_ratio: float = 3.0,
                             min_ashman_d: float = 2.0, ratio_override_ashman_d: float = 5.0,
                             ratio_override_min_ratio: float = 1.15,
                             onset_rescue_max_isis: int = 15) -> str:
    """"bursting_rebound" or "tonic_rebound" for a >=2-spike recovery-window
    train (rebound_spike_count==1/0 are handled by the caller before this is
    reached). classify_burst_pattern's whole-train KDE bimodality test is
    right to be authoritative for LONG trains (it won't call a long,
    gradually-adapting tonic rebound "bursting" just because it eventually
    tapers off) -- but rebound trains are typically SHORT, bounded events
    (a fixed recovery window, not indefinite firing), and the KDE test has
    two confirmed failure modes there (XB2IQX, 2026-08-13): (1) enough ISIs
    to attempt the test, but a short irregular onset-burst-then-decelerating
    -tail sequence doesn't form two clean KDE clusters (held=-4.22/inj=
    -3.37: 7 ISIs, onset 13.2/13.7/17.9ms then an irregular 31-88ms tail,
    KDE says "tonic"); (2) too few ISIs (<6) to attempt the test at all,
    silently defaulting to "tonic_rebound" regardless of structure
    (held=-4.50/inj=-3.93: 2 ISIs, a tight 16.3/21.6ms triplet).

    Three tiers, each reusing already-validated machinery:
    1. Enough ISIs for the KDE test (n >= min_isis_for_burst_test): trust
       it, OR detect_onset_burst's local leading-run finding -- either
       "bursting" wins. Gated to this tier only (not applied to shorter
       trains) so a confident KDE "tonic" verdict on a long train is never
       second-guessed by the shorter-train fallbacks below. Uses the
       STANDARD min_onset_isis (2, same as test-window onset detection),
       not the relaxed rebound_min_onset_isis -- confirmed as a real
       regression (2026-08-13, held=-2.94/inj=-3.03): a long (44-ISI),
       genuinely unimodal 40-48ms alternating train (KDE correctly says
       "tonic", ratio ~1.15-1.2x, nowhere near the 1.5x bar) got forced to
       "bursting_rebound" by detect_onset_burst(min_onset_isis=1) firing
       on a single anomalously-short LEADING isi[0] (likely a release
       transient, not a real burst) -- exactly the single-ISI-vs-noise
       ambiguity min_onset_isis=2 exists to rule out. The 4 confirmed
       genuine onset-burst rebounds validated earlier all had >=3
       consecutive short ISIs and still pass at 2; only tier 2/3's
       genuinely-short trains need the relaxed threshold.

       The onset-detector OR-clause is further restricted to trains no
       longer than onset_rescue_max_isis (default 15) -- confirmed as a
       second real bug (2026-08-14, held=-3.49/inj=-5.5): a smoothly,
       monotonically decelerating 49-ISI tonic rebound (11ms -> 170ms,
       ordinary spike-frequency adaptation, KDE correctly says "tonic")
       crosses the 1.5x ratio threshold once, ~1/8 of the way through,
       purely because that's what a continuously-slowing sequence
       eventually does -- detect_onset_burst can't tell that apart from a
       genuine burst-then-different-rhythm jump. This module's own
       docstring already says the KDE test is right to be authoritative
       for LONG trains; the onset-rescue was only ever meant for the
       SHORT, borderline-length trains in item (1) above (the confirmed
       rescue case was 7 ISIs) where the KDE genuinely doesn't have enough
       data to resolve a real bimodal structure it should have found.

       Also rescued (2026-08-14) when the train genuinely CEASES firing
       well before the recovery window ends (same check as tier 3 below,
       now also available to long trains) -- a "tonic" KDE verdict implies
       ongoing/sustained firing, which a train that then definitively
       stops is not, regardless of its internal ISI shape.
    2. Too few for the KDE test but enough for detect_onset_burst
       (rebound_min_onset_isis defaults to 1, not test-window onset
       detection's 2, so even a 2-ISI train gets a real jump check):
       use detect_onset_burst alone.
    3. No internal jump found anywhere (uniform ISIs, or too short to
       compare at all): if the train then definitively ENDED within the
       window (trailing silence >= trailing_silence_ratio times
       cessation_reference_isi_ms's reference scale, the same reasoning
       test_likely_ceased_firing already uses),
       treat the whole short train as one completed burst -- a handful of
       spikes that fire in a consistent rhythm and then genuinely stop is
       closer to a bounded burst than to "tonic" firing, which implies
       ongoing/sustained activity. Deliberately NOT anchored to an
       absolute ms threshold or a same-level tonic-rate comparison: a
       short, internally-consistent, now-over train reads as a burst on
       its own terms, arbitrary as that is at very small spike counts.
    """
    n = len(rebound_isis)
    if n >= min_isis_for_burst_test:
        kde_result = classify_burst_pattern(rebound_isis, min_isis_for_burst_test,
                                            isi_mode_prominence_frac, min_isi_ratio,
                                            min_ashman_d=min_ashman_d,
                                            ratio_override_ashman_d=ratio_override_ashman_d,
                                            ratio_override_min_ratio=ratio_override_min_ratio, n_peaks=n_peaks)
        onset_n, _ = detect_onset_burst(rebound_isis, min_isi_ratio, min_onset_isis)
        onset_rescues = onset_n is not None and n <= onset_rescue_max_isis
        # Cessation rescue (2026-08-14, same principle as run_test_and_
        # recovery's test_ceased_burst, applied here too -- this tier had
        # no equivalent, tier 3 below is restricted to n < min_isis_for_
        # burst_test): a KDE-"tonic" long recovery train that then
        # genuinely stops well before the recovery window ends is a
        # bounded, self-terminated episode, not ongoing/sustained firing,
        # regardless of whether its ISIs alternate or just smoothly
        # decelerate -- confirmed real case, held=-3.67/inj=-3.14, 20-ISI
        # recovery train (12ms onset -> 144ms), genuinely stops (trailing
        # silence far exceeds 3x its last ISI), onset_n=5 found but ignored
        # since n=20 > onset_rescue_max_isis=15, and until this fix tier 1
        # had no fallback for that case at all.
        ceased_rescue = (rebound_trailing_silence_ms
                         >= trailing_silence_ratio * cessation_reference_isi_ms(rebound_isis))
        return "bursting_rebound" if (kde_result["pattern"] == "bursting" or onset_rescues
                                      or ceased_rescue) else "tonic_rebound"

    onset_n, _ = detect_onset_burst(rebound_isis, min_isi_ratio, rebound_min_onset_isis)
    if onset_n is not None:
        return "bursting_rebound"

    ceased = (rebound_trailing_silence_ms
             >= trailing_silence_ratio * cessation_reference_isi_ms(rebound_isis))
    return "bursting_rebound" if ceased else "tonic_rebound"


def run_test_and_recovery(params, hold_state, held_nA, injected_nA, hold_freq_hz, hold_pattern,
                          dt, temp, reftemp,
                          test_window_s, recovery_window_s, rebound_latency_min_ms,
                          min_isis_for_burst_test, isi_mode_prominence_frac, min_isi_ratio,
                          sag_window_ms: float = 500.0, sag_dead_zone_ms: float = 5.0,
                          adaptation_edge_n: int = 3, min_onset_isis: int = 2,
                          trailing_silence_ratio: float = 3.0, rebound_min_onset_isis: int = 1,
                          max_test_window_s: float = None, test_window_extend_factor: float = 2.0,
                          min_ashman_d: float = 2.0, ratio_override_ashman_d: float = 5.0,
                          ratio_override_min_ratio: float = 1.15,
                          adaptation_ratio_override_threshold: float = 5.0,
                          isi_alternation_min_rate: float = 0.3,
                          onset_rescue_max_isis: int = 15,
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
                                                min_ashman_d=min_ashman_d,
                                                ratio_override_ashman_d=ratio_override_ashman_d,
                                                ratio_override_min_ratio=ratio_override_min_ratio,
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
    # _raw_test_pattern is kept (pre-collapse) so the onset-dominates-sparse
    # override below can tell "genuinely too few spikes to classify" apart
    # from a confidently-computed "tonic"/"bursting" that happens to also
    # have a small onset run.
    _raw_test_pattern = test_pattern
    test_pattern = to_stored_pattern(test_pattern)

    test_v_min_pre_spike_mV, test_first_spike_ms = compute_pre_spike_sag_trough(
        v_test, t_test, sag_window_ms, spike_dead_zone_ms=sag_dead_zone_ms)

    # test_onset_n_spikes: a burst-shaped leading run at the very start of
    # the window, detected locally (see detect_onset_burst) independent of
    # test_pattern -- catches an onset burst that the whole-window KDE test
    # misses when the rest of the train doesn't form its own clean second
    # mode (e.g. a decaying, irregular trickle of single spikes rather than
    # a repeating burst rhythm). Computed unconditionally: harmless/
    # redundant when test_pattern == "bursting" already reports the same
    # opening burst via test_n_bursts, useful specifically when it's
    # "tonic" or "silent" and would otherwise be invisible.
    test_onset_n_spikes, test_onset_isi_mean_ms = (
        detect_onset_burst(isis_test, min_isi_ratio, min_onset_isis)
        if isis_test is not None and len(isis_test) > 0 else (None, None))

    # test_trailing_silence_ms: gap between the last detected spike and the
    # end of the (possibly window-extended) test window -- invisible to any
    # ISI-based statistic, since ISIs only exist between spikes. Needed to
    # tell "still firing, window just cut off" apart from "genuinely
    # stopped partway through" -- compute_adaptation_ratio's first-k/last-k
    # ratio silently assumes the former. test_likely_ceased_firing flags the
    # latter: trailing silence at least trailing_silence_ratio times the
    # most recent ISI is no longer ordinary inter-spike jitter. Left None
    # (rather than False) when there isn't at least one ISI to set a scale
    # from, e.g. 0-1 total spikes -- "ceased" isn't a meaningful question
    # for a window that barely started firing in the first place. Computed
    # here (moved ahead of the onset-override block below, 2026-08-14) so
    # the tiny-train override can use it.
    test_trailing_silence_ms = None
    test_likely_ceased_firing = None
    if test_n_spikes >= 1 and test_first_spike_ms is not None:
        last_spike_ms = test_first_spike_ms + (float(np.sum(isis_test))
                                                if isis_test is not None and len(isis_test) > 0 else 0.0)
        test_trailing_silence_ms = window_s * 1000.0 - last_spike_ms
        if isis_test is not None and len(isis_test) > 0:
            test_likely_ceased_firing = bool(
                test_trailing_silence_ms
                >= trailing_silence_ratio * cessation_reference_isi_ms(isis_test))

    # test_pattern override, 2026-08-13: too few total ISIs for the whole-
    # window KDE test to even attempt classification (_raw_test_pattern was
    # "insufficient_data"/"sparse", collapsed to "silent" above) -- but
    # detect_onset_burst still found a genuine leading burst. At this low a
    # spike count (classify_burst_pattern needs >= min_isis_for_burst_test+1
    # spikes just to TRY the KDE test; detect_onset_burst needs only
    # min_onset_isis+1), an onset burst it can still find necessarily
    # accounts for most/all of what little activity there is -- confirmed
    # case: XB2IQX held=-4.50/inj=-3.93, 5 total spikes, 3 of them the
    # onset burst, then genuine silence, reported "silent" even though the
    # cell plainly burst. Deliberately narrower than "onset found ->
    # bursting" unconditionally: a LONG train with a small onset
    # (confirmed contrast: held=-2.39/inj=-3.37) legitimately runs its KDE
    # test successfully (_raw_test_pattern == "tonic" there, not "sparse"/
    # "insufficient_data"), so this override never triggers for it -- see
    # the separate adaptation-ratio override below for that case instead.
    # test_n_bursts/test_spikes_per_burst are set directly from the onset
    # run below (burst_test's own n_long_isis doesn't apply -- classify_
    # burst_pattern never reached its "bursting" branch here).
    test_onset_dominates_sparse_window = (
        _raw_test_pattern in ("insufficient_data", "sparse") and test_onset_n_spikes is not None)

    # test_pattern override, 2026-08-14: a train too short even for
    # detect_onset_burst to attempt (needs >= min_onset_isis+1 ISIs; this
    # has fewer), but with at least 1 ISI (2 spikes) that then definitively
    # ended (test_likely_ceased_firing) -- the tiny-train analogue of
    # classify_rebound_pattern's tier-3 "doublet burst" rule, extended here
    # to the test window for the same reason: at this low a spike count, a
    # burst-then-silence shape IS the entire observed train, and there's no
    # meaningful "wait for more ISIs" option. Confirmed case: XB2IQX
    # held=-4.50/inj=-4.04, ISIs [20.1, 36.6] (3 spikes) then 19.85s of
    # silence in a 20s window -- reported "silent" even though the cell
    # plainly fired a short burst and stopped. Does not overlap with
    # test_onset_dominates_sparse_window above (that one requires
    # test_onset_n_spikes is not None, only possible with >= min_onset_isis+1
    # ISIs -- this covers what's too short even for that).
    test_tiny_burst_then_silence = (
        _raw_test_pattern in ("insufficient_data", "sparse") and test_onset_n_spikes is None
        and isis_test is not None and len(isis_test) >= 1 and bool(test_likely_ceased_firing))

    if test_onset_dominates_sparse_window or test_tiny_burst_then_silence:
        test_pattern = "bursting"

    # Restricted to "tonic" AND not likely-ceased: a bursting train's
    # first-k/last-k ISIs would mix intra-/inter-burst intervals (see
    # compute_adaptation_ratio docstring), and a train that stops partway
    # through the window isn't smoothly rate-adapting either -- a first-k/
    # last-k ratio over a decaying-then-silent train (confirmed case:
    # XB2IQX held=-2.48/inj=-4.04, ISIs 13-17ms onset then an irregular
    # 30-112ms trickle that stops with ~2.3s of the 3s window left silent)
    # reports a large, "clean-looking" number that isn't measuring the same
    # thing it measures for a train that keeps firing steadily to the end.
    test_adaptation_ratio = (compute_adaptation_ratio(isis_test, adaptation_edge_n)
                             if test_pattern == "tonic" and not test_likely_ceased_firing else None)

    # test_pattern override, 2026-08-14: classify_burst_pattern's whole-
    # window KDE test can resolve only 1 mode (-> "tonic") for a train whose
    # "long" ISIs are themselves spread across a wide range -- the spread
    # smears what would otherwise be a second peak into a low plateau
    # instead of a resolvable mode, which is a KDE resolution failure, not
    # evidence the train is actually tonic. Confirmed case: XB2IQX
    # held=-2.39/inj=-3.37, 19 ISIs ranging 15ms to 617ms (~41x spread),
    # KDE finds only 1 mode.
    #
    # Deliberately NOT gated on test_adaptation_ratio (first-3/last-3 ISI
    # ratio) despite that being the more obviously "adaptation-shaped"
    # signal: test_adaptation_ratio is itself only computed when NOT
    # test_likely_ceased_firing, and this exact confirmed case is already
    # documented (see generate_validation_packet.py's adapt_exemplars comment)
    # to be unstable across resimulation runs on that specific flag --
    # sometimes ceased=True (adaptation_ratio never computed, None), other
    # times ceased=False (adaptation_ratio=17.5). A max/min spread ratio on
    # the raw ISIs measures the same underlying non-stationarity but doesn't
    # depend on where the window happened to cut off, so it stays available
    # either way. test_n_bursts/test_spikes_per_burst are deliberately left
    # None below for this case: unlike the onset-dominated overrides, there
    # is no well-defined "one burst" or clean n_long_isis count for a train
    # shaped like this, and reporting a fabricated number would be worse
    # than reporting none.
    test_isi_spread_ratio = (float(isis_test.max() / isis_test.min())
                             if isis_test is not None and len(isis_test) > 0 and isis_test.min() > 0
                             else None)

    # A wide spread alone doesn't distinguish a genuinely OSCILLATING (burst-
    # like) train from one that just smoothly, monotonically decelerates to
    # a new steady rate (ordinary spike-frequency adaptation, e.g. after an
    # onset transient) -- both can post a large max/min ratio, but only the
    # first is actually bursting. Confirmed directly (2026-08-14): a KDE-
    # "tonic" train with isis_test spanning 14.2ms->296ms (21x spread, well
    # over this override's threshold) was being called "bursting" purely
    # from spread, despite being a brief onset transient (~30 ISIs, tiny
    # monotonic growth) then no real oscillation until a genuine burst
    # phase only emerges in the last ~14 ISIs. test_isi_alternation_rate
    # (the fraction of consecutive ISI-to-ISI steps that flip direction --
    # up-then-down-then-up..., as a genuine short/long/short/long
    # alternation would) separates these: a purely monotonic adapting tail
    # scores far below (sampled real case: 0.15), a genuinely alternating
    # train scores far above (the original confirmed panel-11 case: 0.88;
    # the above mixed-phase case, which DOES eventually show real
    # alternation, scores 0.55 -- correctly in between, not incorrectly
    # rejected). A broader sample of 60 currently-spread-overridden points
    # shows the same split: median rate 0.68, only ~12% below 0.3.
    isi_diff_signs = np.sign(np.diff(isis_test)) if isis_test is not None and len(isis_test) > 2 else None
    if isi_diff_signs is not None:
        isi_diff_signs = isi_diff_signs[isi_diff_signs != 0]
    test_isi_alternation_rate = (
        float(np.mean(isi_diff_signs[1:] != isi_diff_signs[:-1]))
        if isi_diff_signs is not None and len(isi_diff_signs) > 1 else None)

    # Second, independent signature (2026-08-14, user-proposed): a "tonic"
    # verdict implies ONGOING/sustained firing -- a train that instead
    # genuinely CEASES firing well before the window ends (test_likely_
    # ceased_firing, already computed above) is a bounded, self-terminated
    # episode, which reads as a burst on its own terms regardless of
    # whether its internal ISIs alternate or just smoothly decelerate
    # ("even if the ISI begins to increase after each spike"). Confirmed
    # directly: EVERY one of the 4 reference cases examined (the original
    # panel-11 case, both flagged false-positive/borderline cases, and a
    # real "bursting via the plain KDE test" case used as a control) shows
    # ceased=True -- and separately, sampling 10 of the 46 total XB2IQX
    # points that are "tonic" AND ceased (all with n_isis 10-42, out of
    # 1433 tonic points total) shows the identical shape: a brief fast
    # onset, smooth deceleration, then 1.6-2s of genuine silence in a ~3s
    # window -- not "still firing, window just cut off." Tonic points as a
    # whole show ceased=True only 3.2% of the time vs. 86.6% for points
    # already called bursting, so this is a clean, selective signal, not
    # something that would fire indiscriminately.
    test_oscillating_burst = (
        test_isi_spread_ratio is not None
        and test_isi_spread_ratio >= adaptation_ratio_override_threshold
        and test_isi_alternation_rate is not None
        and test_isi_alternation_rate >= isi_alternation_min_rate)
    test_ceased_burst = bool(test_likely_ceased_firing)
    test_adaptation_ratio_extreme = (
        test_pattern == "tonic" and (test_oscillating_burst or test_ceased_burst))
    if test_adaptation_ratio_extreme:
        test_pattern = "bursting"
        # test_adaptation_ratio was computed above while test_pattern was
        # still "tonic" (pre-override) -- clear it now that the window has
        # been reclassified "bursting", matching the same invariant already
        # enforced for test_n_bursts/test_spikes_per_burst below (a first-k/
        # last-k ISI ratio isn't meaningful for a burst window either).
        test_adaptation_ratio = None

    # n_bursts = n_long_isis + 1 (a "burst" is a maximal run of consecutive
    # short ISIs, so each inter-burst gap marks one more burst boundary);
    # spikes_per_burst is the average over the whole test window, not a
    # per-burst array -- see classify_burst_pattern's n_long_isis docstring.
    # Checked in order: the two onset-based overrides supply their own
    # count (the onset run IS the whole observed train in both cases); the
    # adaptation-ratio override supplies no count (see its comment above);
    # otherwise fall back to classify_burst_pattern's own n_long_isis.
    if test_onset_dominates_sparse_window:
        test_n_bursts = 1
        test_spikes_per_burst = float(test_onset_n_spikes)
    elif test_tiny_burst_then_silence:
        test_n_bursts = 1
        test_spikes_per_burst = float(test_n_spikes)
    elif test_adaptation_ratio_extreme:
        test_n_bursts = None
        test_spikes_per_burst = None
    else:
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
        "test_onset_n_spikes": test_onset_n_spikes, "test_onset_isi_mean_ms": test_onset_isi_mean_ms,
        "test_trailing_silence_ms": test_trailing_silence_ms,
        "test_likely_ceased_firing": test_likely_ceased_firing,
        "test_isi_alternation_rate": test_isi_alternation_rate,
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
    #
    # hold_freq_hz == 0, not just the ratio clause (2026-08-13): a held
    # level the cell doesn't sustain ANY firing at on its own is exactly
    # the kind of baseline a genuine rebound burst should be evaluated
    # against, but the ratio clause requires hold_freq_hz > 0 to even run
    # -- confirmed as a real, missed case on XB2IQX held=-4.32/inj=-3.26
    # (hold_freq_hz=0.0, test_pattern="tonic" so not caught by the silent
    # clause either): releasing back to held produces a genuine 8-spike
    # decelerating burst, previously discarded entirely as
    # rebound_applicable=False/"not_applicable" without ever running peak
    # detection on the already-simulated recovery trace. Checked across
    # XB2IQX's full grid: 438/2500 points (17.5%) share this exact gap
    # (hold_freq_hz==0, test_pattern != "silent").
    #
    # test_pattern != hold_pattern, not just a rate comparison (2026-08-13):
    # the rate-ratio clause is only a PROXY for "did something qualitative
    # change" -- it misses a pattern change (tonic hold -> bursting test)
    # that isn't also a rate drop. hold_pattern (classify_hold_pattern on
    # the hold level's own settle chunk, zero extra simulation cost) is the
    # actual thing being approximated, so compare it directly. Confirmed
    # missed cases on XB2IQX: held=-3.12/inj=-3.03 (hold tonic @ 13.5Hz,
    # test bursting @ 18.7Hz -- rate went UP, never suppressed by the ratio
    # clause) and held=-3.31/inj=-2.92 (hold tonic @ 5.0Hz, test tonic @
    # 24.7Hz) both show a genuine multi-spike recovery burst that was
    # entirely discarded.
    test_suppressed = (test_pattern == "silent"
                       or hold_freq_hz == 0
                       or test_pattern != hold_pattern
                       or (hold_freq_hz > 0 and test_freq_hz < 0.5 * hold_freq_hz))

    # Option B, 2026-08-14: test_suppressed (stored as rebound_applicable)
    # no longer GATES classification -- it's now a purely descriptive "is
    # this distinguishable from what this held level does on its own" flag,
    # kept for exactly the filtering every current feature-extraction
    # consumer already does with it (extract_grid_features.py's
    # compute_summary_stats/compute_boundary_slopes/compute_rebound_maps
    # all independently filter on rebound_applicable -- none of them
    # depend on rebound_pattern being "not_applicable" as a signal, so this
    # doesn't change any existing numeric feature). What changes: the
    # recovery window's peaks are now ALWAYS examined and rebound_pattern
    # always reports a real classification ("none"/"single_spike"/
    # "tonic_rebound"/"bursting_rebound"), never the placeholder
    # "not_applicable" for a genuinely-simulated point (still used for
    # actually-blown-up points, see make_blew_up_point/_TEST_RECOVERY_
    # DEFAULTS -- an unrelated case, no data exists there at all).
    #
    # The old all-or-nothing suppression was hiding real, describable
    # recovery-window behavior any time it happened to match what the held
    # level already does on its own -- confirmed real cases: XB2IQX
    # held=-3.12/inj=-3.03 genuinely bursts in recovery (a triplet, then
    # settling into doublets matching hold's own rhythm) and held=-2.85/
    # inj=-1.23 genuinely fires tonically in recovery, both previously
    # reported as "not_applicable" with peak detection never even run.
    # rebound_applicable=False on either now correctly means "matches what
    # this held level does anyway, not a distinguishable rebound" while
    # rebound_pattern still tells you what the recovery window actually did.
    if v_rec.max() - v_rec.min() < FLATLINE_MV:
        peak_times_ms = np.array([])
    else:
        peaks, _ = find_peaks(v_rec, prominence=_spike_prominence(v_rec.max() - v_rec.min(),
                                                                   PROMINENCE_FRACTION))
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
        rebound_trailing_silence_ms = recovery_window_s * 1000.0 - qualifying[-1]
        rebound_pattern = classify_rebound_pattern(
            rebound_isis, rebound_trailing_silence_ms, min_isis_for_burst_test,
            isi_mode_prominence_frac, min_isi_ratio, rebound_spike_count,
            min_onset_isis=min_onset_isis, rebound_min_onset_isis=rebound_min_onset_isis,
            trailing_silence_ratio=trailing_silence_ratio, min_ashman_d=min_ashman_d,
            ratio_override_ashman_d=ratio_override_ashman_d,
            ratio_override_min_ratio=ratio_override_min_ratio,
            onset_rescue_max_isis=onset_rescue_max_isis)

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
    "test_onset_n_spikes": None, "test_onset_isi_mean_ms": None,
    "test_trailing_silence_ms": None, "test_likely_ceased_firing": None,
    "test_isi_alternation_rate": None,
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
        "hold_v_end_mV": float("nan"), "hold_v_trough_mV": float("nan"), "hold_pattern": None,
        "hold_blew_up": hold_blew_up,
        **_TEST_RECOVERY_DEFAULTS,
        "blew_up": True, "error": error,
    }


def run_trial_point(params, hold_result, held_nA, injected_nA, dt, temp, reftemp,
                    test_window_s, recovery_window_s, rebound_latency_min_ms,
                    isi_kwargs, source) -> dict:
    if hold_result["blew_up"]:
        return make_blew_up_point(held_nA, injected_nA, source, hold_result.get("error"), hold_blew_up=True)

    tr = run_test_and_recovery(params, hold_result["final_state"], held_nA, injected_nA,
                               hold_result["freq_hz"] or 0.0, hold_result["hold_pattern"],
                               dt, temp, reftemp,
                               test_window_s, recovery_window_s,
                               rebound_latency_min_ms, **isi_kwargs)
    point = {
        "held_nA": held_nA, "injected_nA": injected_nA, "source": source,
        "hold_settled": hold_result["settled"], "hold_freq_hz": hold_result["freq_hz"] or 0.0,
        "hold_is_flatline": hold_result["is_flatline"], "hold_v_end_mV": hold_result["hold_v_end_mV"],
        "hold_v_trough_mV": hold_result["hold_v_trough_mV"], "hold_pattern": hold_result["hold_pattern"],
        "hold_blew_up": False,
        **_TEST_RECOVERY_DEFAULTS,
    }
    point.update(tr)
    return point


def get_or_settle_hold(params, held_nA, hold_settle_cache, y_ss, baseline_freq_hz,
                       dt, temp, reftemp, settle_kwargs, y_ss_trough_mV=None,
                       y_ss_hold_pattern=None) -> dict:
    """Settle at held_nA, reusing a cached settle if this exact (rounded)
    held level has already been solved, else warm-starting from whichever
    already-settled held level is numerically nearest. held=0 is free --
    reuses the cell's own cached Iapp=0 limit-cycle state.

    y_ss_trough_mV (min V over the cached Iapp=0 burn-in trace, see
    run_cell_grid) plays the same role for held=0 as hold_v_trough_mV plays
    for every other held level -- held=0.0 also fires spontaneously for many
    cells (XB2IQX included), so it has the identical single-sample-baseline
    problem settle_hold_level's hold_v_trough_mV addition fixes. Falls back
    to the single y_ss sample only if no burn-in trace was cached.
    y_ss_hold_pattern (classify_hold_pattern on that same burn-in trace)
    plays the identical role for hold_pattern.
    """
    key = _round_level(held_nA)
    if key in hold_settle_cache:
        return hold_settle_cache[key]
    if key == 0.0:
        trough = y_ss_trough_mV if y_ss_trough_mV is not None else float(y_ss[V_INDEX])
        hold_pattern = y_ss_hold_pattern if y_ss_hold_pattern is not None else "silent"
        result = {"blew_up": False, "settled": True, "freq_hz": baseline_freq_hz,
                  "is_flatline": False, "final_state": y_ss.copy(),
                  "hold_v_end_mV": float(y_ss[V_INDEX]), "hold_v_trough_mV": trough,
                  "hold_pattern": hold_pattern}
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
# 2a: uniform grid
# ---------------------------------------------------------------------------
#
# Replaces a previous two-stage design (fixed-step coarse grid + boundary-
# driven adaptive bisection refinement) with a single, simple, non-adaptive
# division of each axis into a fixed point count -- per explicit request
# (2026-08-11) to standardize simulation count across all 69 cells rather
# than let it vary with each cell's own floor depth (the old design ranged
# 7-42 coarse levels/axis depending on floor depth, and refinement itself
# had NO point-count budget at all, only an absolute nA edge-length floor
# and a recursion-depth cap, so total simulated points per cell varied even
# more than the coarse stage alone did). Every cell now gets exactly
# args.n_points_held x args.n_points_injected simulated points, full stop
# -- deeper-floor cells are sampled at coarser absolute nA spacing, not
# simulated more or fewer times.

def build_uniform_grid(params, y_ss, baseline_freq_hz, cell_floor_nA, injected_floor_nA, args,
                       y_ss_trough_mV=None, y_ss_hold_pattern=None):
    held_levels = [_round_level(v) for v in np.linspace(0.0, cell_floor_nA, args.n_points_held)]
    injected_levels = [_round_level(v) for v in np.linspace(0.0, injected_floor_nA, args.n_points_injected)]

    settle_kwargs = dict(chunk_s=args.hold_settle_chunk_s, max_settle_s=args.max_hold_settle_s,
                         settle_rtol=args.hold_settle_rtol, min_peaks_for_rate=args.min_peaks_for_rate,
                         min_isis_for_burst_test=args.min_isis_for_burst_test,
                         isi_mode_prominence_frac=args.isi_mode_prominence_frac,
                         min_isi_ratio=args.min_isi_ratio, min_ashman_d=args.min_ashman_d,
                         ratio_override_ashman_d=args.ratio_override_ashman_d,
                         ratio_override_min_ratio=args.ratio_override_min_ratio)
    isi_kwargs = dict(min_isis_for_burst_test=args.min_isis_for_burst_test,
                      isi_mode_prominence_frac=args.isi_mode_prominence_frac,
                      min_isi_ratio=args.min_isi_ratio,
                      min_ashman_d=args.min_ashman_d,
                      ratio_override_ashman_d=args.ratio_override_ashman_d,
                      ratio_override_min_ratio=args.ratio_override_min_ratio,
                      adaptation_ratio_override_threshold=args.adaptation_ratio_override_threshold,
                      isi_alternation_min_rate=args.isi_alternation_min_rate,
                      onset_rescue_max_isis=args.onset_rescue_max_isis,
                      rebound_min_onset_isis=args.rebound_min_onset_isis,
                      sag_window_ms=args.sag_window_ms,
                      sag_dead_zone_ms=args.sag_dead_zone_ms,
                      adaptation_edge_n=args.adaptation_edge_n,
                      min_onset_isis=args.min_onset_isis,
                      trailing_silence_ratio=args.trailing_silence_ratio,
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
                                         args.dt, args.temp, args.reftemp, settle_kwargs,
                                         y_ss_trough_mV=y_ss_trough_mV,
                                         y_ss_hold_pattern=y_ss_hold_pattern)
        for injected in injected_levels:
            point = run_trial_point(params, hold_result, held, injected, args.dt, args.temp, args.reftemp,
                                    test_window_s, recovery_window_s, args.rebound_latency_min_ms,
                                    isi_kwargs, source="uniform")
            grid[(held, injected)] = point

    return grid, hold_settle_cache, held_levels, injected_levels, test_window_s, recovery_window_s


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
    burnin_v = ss_entry.get("burnin_v")
    burnin_t = ss_entry.get("burnin_t")
    y_ss_trough_mV = float(np.min(burnin_v)) if burnin_v is not None else None
    y_ss_hold_pattern = classify_hold_pattern(burnin_v, burnin_t, args.min_isis_for_burst_test,
                                              args.isi_mode_prominence_frac, args.min_isi_ratio,
                                              args.min_ashman_d, args.ratio_override_ashman_d,
                                              args.ratio_override_min_ratio)

    try:
        grid, hold_settle_cache, held_levels, injected_levels, test_window_s, recovery_window_s = \
            build_uniform_grid(params, y_ss, baseline_freq_hz, cell_floor_nA, injected_floor_nA, args,
                               y_ss_trough_mV=y_ss_trough_mV, y_ss_hold_pattern=y_ss_hold_pattern)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {**base, "status": "blew_up", "error": str(exc)}

    hold_states = {lvl: r["final_state"] for lvl, r in hold_settle_cache.items() if not r["blew_up"]}

    return {
        **base, "status": "ok",
        "cell_floor_nA": cell_floor_nA, "injected_floor_nA": injected_floor_nA,
        "floor_anchor_source": floor_anchor_source,
        "held_levels_nA": np.array(held_levels), "injected_levels_nA": np.array(injected_levels),
        "test_window_s": test_window_s, "recovery_window_s": recovery_window_s,
        "grid": grid, "hold_states": hold_states,
        "n_points_total": len(grid),
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
# Moved here from extract_grid_features.py (2026-08-11 figure merge) --
# plot_cell_grid_features needs both this and PATTERN_COLORS, and the
# established one-directional import convention (extract_grid_features.py
# imports FROM this module, never the reverse) means any constant shared
# by both panels has to live here.
REBOUND_PATTERN_COLORS = {"none": "gray", "single_spike": "steelblue",
                          "tonic_rebound": "seagreen", "bursting_rebound": "mediumpurple",
                          "not_applicable": "lightgray"}
# "none" and "not_applicable" both mean "no rebound to report here" -- they're
# rendered as BLANK_COLOR (see _panel_categorical's blank_labels), not their
# nominal colors above, so this dict's colors for them are unused in practice.
REBOUND_BLANK_LABELS = {"none", "not_applicable"}
NO_DATA_COLOR = (1.0, 1.0, 1.0, 1.0)  # plain white: a pixel with nothing to report,
# whether that's a genuinely missing grid coordinate or a blank_labels category --
# a reader doesn't need three shades of gray individually justified to read
# "nothing happened here" (user-flagged 2026-08-14).


def _test_pattern_map(cell_result: dict, features: dict) -> dict:
    pattern_names = list(PATTERN_COLORS.keys())
    return {(p["held_nA"], p["injected_nA"]): pattern_names.index(p["test_pattern"] or "silent")
            for p in cell_result["grid"].values()}


def _rebound_pattern_map(cell_result: dict, features: dict) -> dict:
    rp_names = list(REBOUND_PATTERN_COLORS.keys())
    return {k: rp_names.index(v) for k, v in (features.get("rebound_pattern_map") or {}).items()}


def _spikes_per_burst_map(cell_result: dict, features: dict) -> dict:
    return {(p["held_nA"], p["injected_nA"]): p["test_spikes_per_burst"]
           for p in cell_result["grid"].values() if p["test_pattern"] == "bursting"}


def _iH_map(cell_result: dict, features: dict) -> dict:
    # See build_cell_grid_features_fig's original inline comment (still
    # applies): restricted to test_pattern == "silent" -- for a point that
    # kept firing during the test window, argmin(v_rec) can land on an
    # ongoing spike's AHP trough instead of a genuine PIR trough, confirmed
    # directly on a real 9GBDEX point (held=-0.8, inj=-4.71).
    return {(p["held_nA"], p["injected_nA"]): p["rebound_peak_iH_nA"]
           for p in cell_result["grid"].values()
           if p["rebound_peak_iH_nA"] is not None and p["test_pattern"] == "silent"}


# One entry per grid-features heatmap panel, in the same order/grouping
# build_cell_grid_features_fig lays them out in (classification overview /
# rate-timing / rebound / passive-other). `key` is the stable CLI-facing name
# used by plot_parameter_trace.py's and plot_held_slice.py's --parameter.
# Shared by build_cell_grid_features_fig (renders all 14 via
# draw_parameter_panel) and both of those scripts (render one, looked up by
# key) so a panel's rendering can never drift between call sites.
PARAMETER_PANELS = [
    dict(key="test_pattern", title="test-window firing pattern", kind="categorical",
        value_map_fn=_test_pattern_map, colors=PATTERN_COLORS, blank_labels=frozenset()),
    dict(key="rebound_pattern", title="rebound pattern", kind="categorical",
        value_map_fn=_rebound_pattern_map, colors=REBOUND_PATTERN_COLORS, blank_labels=REBOUND_BLANK_LABELS),
    dict(key="firing_rate", title="test-window firing rate", kind="map", cmap="viridis", label="Hz",
        value_map_fn=lambda cr, f: f.get("firing_rate_map") or {}),
    dict(key="intra_burst_rate", title="intra-burst rate", kind="map", cmap="cividis", label="Hz",
        value_map_fn=lambda cr, f: f.get("intra_burst_rate_map") or {}),
    dict(key="burst_freq_approx", title="burst freq, approx", kind="map", cmap="cividis", label="Hz",
        value_map_fn=lambda cr, f: f.get("burst_freq_approx_map") or {}),
    dict(key="n_bursts", title="n bursts", kind="map", cmap="plasma", label="count",
        value_map_fn=lambda cr, f: f.get("n_bursts_map") or {}),
    dict(key="rebound_occurred", title="rebound occurred", kind="map", cmap="Oranges", label="fraction",
        value_map_fn=lambda cr, f: f.get("rebound_occurred_map") or {}),
    dict(key="rebound_count", title="rebound spike count", kind="map", cmap="YlOrBr", label="spike count",
        value_map_fn=lambda cr, f: f.get("rebound_count_map") or {}),
    dict(key="rebound_latency", title="rebound latency", kind="map", cmap="viridis", label="ms",
        value_map_fn=lambda cr, f: f.get("rebound_latency_map") or {}),
    dict(key="rebound_peak_mV", title="rebound peak", kind="map", cmap="magma", label="mV",
        value_map_fn=lambda cr, f: f.get("rebound_peak_mV_map") or {}),
    dict(key="sag_depth", title="sag depth", kind="map", cmap="YlGnBu", label="mV",
        value_map_fn=lambda cr, f: f.get("sag_depth_map") or {}),
    dict(key="adaptation_ratio", title="adaptation ratio (tonic)", kind="map", cmap="plasma", label="ratio",
        value_map_fn=lambda cr, f: f.get("adaptation_ratio_map") or {}),
    dict(key="spikes_per_burst", title="spikes per burst (bursting only)", kind="map", cmap="plasma",
        label="spikes/burst", value_map_fn=_spikes_per_burst_map),
    dict(key="iH_recovery_trough", title="i_H at recovery trough (silenced only)", kind="map", cmap="magma",
        label="nA", value_map_fn=_iH_map),
]
PARAMETER_PANELS_BY_KEY = {spec["key"]: spec for spec in PARAMETER_PANELS}


def _exact_grid_matrix(value_map: dict, held_levels, injected_levels) -> np.ndarray:
    """Builds an exact (n_injected, n_held) matrix directly from a
    {(held_nA, injected_nA): value} mapping and this cell's own uniform
    grid axis levels (cell_result["held_levels_nA"]/["injected_levels_nA"])
    -- NO interpolation, merging, or nearest-neighbor lookup of any kind
    (2026-08-11/12, per explicit instruction: every rendered heat-map pixel
    must show exactly what was simulated, never a value blended or
    borrowed from a neighboring point).

    Replaces the old griddata/_merge_close_points/_render_heatmap pipeline
    entirely. That machinery existed to interpolate the OLD adaptively-
    refined, irregularly-spaced grid onto a regular canvas for imshow; the
    uniform grid (2026-08-11) already IS a regular canvas (every (held,
    injected) combination on a fixed n_points_held x n_points_injected grid
    was actually simulated), so no interpolation step is needed, and one
    less step means one less place for rendering artifacts to creep in.

    A coordinate missing from value_map (e.g. a feature only computed for
    bursting points, or "rebound pattern" wherever rebound wasn't even
    applicable) renders as NaN -- the honest "no data here", never a value
    borrowed from a real but different point (which is what the old
    nearest-neighbor-over-an-upsampled-canvas approach did silently: e.g.
    the old "rebound pattern" panel would tile a NOT-applicable cell with
    whichever real applicable neighbor happened to be nearest, misreporting
    it as some real pattern rather than "not applicable").
    """
    matrix = np.full((len(injected_levels), len(held_levels)), np.nan)
    for i, held in enumerate(held_levels):
        for j, injected in enumerate(injected_levels):
            v = value_map.get((held, injected))
            if v is not None:
                matrix[j, i] = v
    return matrix


def _panel_map(ax, value_map, held_levels, injected_levels, extent, cmap, label, title):
    """Shared continuous-panel renderer for plot_cell_grid_features -- an
    exact per-point matrix (see _exact_grid_matrix), imshow'd with
    interpolation="nearest" (meaning: no resampling/blending even when the
    figure's saved pixel density differs from the grid's own N_held x
    N_injected point count -- every displayed pixel still traces to
    exactly one real matrix cell, never an average of neighbors).
    """
    matrix = _exact_grid_matrix(value_map, held_levels, injected_levels)
    if np.all(np.isnan(matrix)):
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        im = ax.imshow(matrix, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title, fontsize=9)


def _panel_categorical(ax, value_map, held_levels, injected_levels, extent, colors, title,
                       blank_labels=frozenset()):
    """Shared categorical-panel renderer -- colormaps category codes to RGBA
    (never blends raw integer codes: "silent"=0 blended with "tonic"=1
    could land near 0.5, a fabricated intermediate category that doesn't
    exist), then imshow's with interpolation="nearest".

    A pixel is painted NO_DATA_COLOR (plain white, no legend entry) when its
    grid coordinate is genuinely missing (see _exact_grid_matrix) OR its
    category is listed in blank_labels -- categories that themselves just
    mean "nothing to report here" (e.g. rebound pattern's "none"/
    "not_applicable"). Both cases read identically to a reader: don't give
    "no data" its own gray shade and legend line distinct from "none"'s,
    when the point of the pixel is that there's nothing there either way
    (user-flagged 2026-08-14 -- three near-identical grays each individually
    explained in the legend, to say "nothing happened").
    """
    names = list(colors.keys())
    matrix = _exact_grid_matrix(value_map, held_levels, injected_levels)
    cmap = ListedColormap(list(colors.values()))
    norm = BoundaryNorm(np.arange(len(names) + 1) - 0.5, cmap.N)
    blank_codes = [names.index(lbl) for lbl in blank_labels if lbl in names]
    blank = np.isnan(matrix) | np.isin(matrix, blank_codes)
    rgba = cmap(norm(np.where(blank, 0, matrix)))
    rgba[blank] = NO_DATA_COLOR
    ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=8, label=lbl)
              for lbl, c in colors.items() if lbl not in blank_labels]
    ax.legend(handles=handles, loc="best", fontsize=6)


def draw_parameter_panel(ax, cell_result: dict, features: dict, parameter_key: str) -> dict:
    """Renders one PARAMETER_PANELS entry into `ax` -- heatmap or
    categorical, exactly matching build_cell_grid_features_fig's own
    per-panel rendering -- and returns its value_map, so a caller can look
    up a specific point's or row's actual value after drawing. Shared by
    build_cell_grid_features_fig (loops over all 14) and
    plot_parameter_trace.py / plot_held_slice.py (draw one, by key).
    """
    spec = PARAMETER_PANELS_BY_KEY[parameter_key]
    held_levels = list(cell_result["held_levels_nA"])
    injected_levels = list(cell_result["injected_levels_nA"])
    extent = (held_levels[0], held_levels[-1], injected_levels[0], injected_levels[-1])
    value_map = spec["value_map_fn"](cell_result, features)
    if spec["kind"] == "categorical":
        _panel_categorical(ax, value_map, held_levels, injected_levels, extent,
                           spec["colors"], spec["title"], blank_labels=spec["blank_labels"])
    else:
        _panel_map(ax, value_map, held_levels, injected_levels, extent,
                  spec["cmap"], spec["label"], spec["title"])
    return value_map


def build_cell_grid_features_fig(cell_result: dict, features: dict, bottom_margin: float = 0.025):
    """Builds the 14-panel grid-features figure and returns it WITHOUT a
    suptitle, footer, or save/close -- split out of plot_cell_grid_features
    (below) so a second caller (generate_validation_packet.py's feature-
    overview page) can reuse the exact same panel-drawing code, including
    the no-interpolation exact-matrix rendering, instead of maintaining a
    second copy that could drift out of sync. Returns None under the same
    conditions plot_cell_grid_features used to skip silently (not ok /
    empty grid / features not ok).

    Merged replacement for the old plot_cell_grid (6 panels, this module)
    + plot_cell_features (12 panels, extract_grid_features.py) -- 4 of the
    6/12 were exact duplicates (firing rate, rebound count, rebound
    latency, spikes/burst); this shows the 14 genuinely distinct panels
    once each, grouped thematically (classification overview / rate-timing
    / rebound / passive-other) rather than in arbitrary reading order.

    Needs BOTH the raw cell_result (grid: test_pattern, rebound_peak_iH_nA
    -- neither is exposed as a features map) and the precomputed features
    dict (everything else, already computed by extract_cell_features).

    Firing-rate definition reconciled (2026-08-11): the old plot_cell_grid
    NaN'd out silent points; features["firing_rate_map"] (compute_
    firing_rate_map) includes them at their real 0.0 Hz -- kept here, since
    a silent-but-not-blown-up point has a well-defined rate, not an
    undefined one.
    """
    if cell_result["status"] != "ok" or not cell_result["grid"] or features.get("status") != "ok":
        return None

    # held_levels/injected_levels (used inside draw_parameter_panel for each
    # panel's extent) run 0 -> floor (descending numerically, since floor <
    # 0) by construction (build_uniform_grid) -- rendered directly as (left,
    # right)/(bottom, top) this gives a "reversed" extent (left > right,
    # bottom > top numerically) that matplotlib renders correctly (0 at the
    # outer edge, floor at the inner edge, matching every other
    # current-injection figure in this project) with no separate
    # invert_xaxis()/invert_yaxis() step needed.
    fig, axes = plt.subplots(4, 4, figsize=(19, 16))
    for ax in (axes[0, 2], axes[0, 3]):
        ax.axis("off")

    # Panel grid positions in PARAMETER_PANELS order: row 0 has only 2
    # (classification overview, categorical), the other 2 slots in that row
    # are blank (turned off above); rows 1-3 are full (rate/timing, rebound,
    # passive/other, 4 panels each). firing_rate_map (row 1, first) is
    # reconciled to the more permissive definition: includes silent points
    # at their real 0.0 Hz rather than NaN-ing them out.
    panel_axes = [axes[0, 0], axes[0, 1],
                 axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3],
                 axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3],
                 axes[3, 0], axes[3, 1], axes[3, 2], axes[3, 3]]
    for ax, spec in zip(panel_axes, PARAMETER_PANELS):
        draw_parameter_panel(ax, cell_result, features, spec["key"])

    for ax in axes.flat:
        if not ax.axison:
            continue
        ax.set_xlabel("held (nA)", fontsize=7)
        ax.set_ylabel("injected (nA)", fontsize=7)
        ax.tick_params(labelsize=7)

    # Left title-less: tight_layout's rect still reserves the same top/
    # bottom margin whether or not a suptitle/footer is added yet, so each
    # caller can add its own afterward without triggering a re-layout.
    # bottom_margin is a parameter (not always 0.025) because the
    # validation packet's feature-overview page adds a multi-line prose
    # caption below the panels that the standalone figure doesn't have --
    # it needs much more reserved space than a one-line command footer does.
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.965))
    return fig


def plot_cell_grid_features(cell_result: dict, features: dict, outdir: Path, command: str,
                            fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    """Thin save-to-disk wrapper around build_cell_grid_features_fig -- see
    that function's docstring for what the figure itself contains.
    """
    fig = build_cell_grid_features_fig(cell_result, features)
    if fig is None:
        return
    cell_id = cell_result["cell_id"]
    burstiness = features.get("burstiness_index")
    fi_slope = features.get("fi_slope_hz_per_nA")
    burstiness_str = f"{burstiness:.3f}" if burstiness is not None else "None"
    fi_slope_str = f"{fi_slope:.3f} Hz/nA" if fi_slope is not None else "None"
    title = (f"{cell_id} — status: {cell_result['status']}, "
            f"cell_floor={cell_result['cell_floor_nA']:.2f} nA, "
            f"n_points={cell_result['n_points_total']}, "
            f"burstiness={burstiness_str}, F-I slope={fi_slope_str}")
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.005, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_grid.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=170)
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
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--jobs", type=int, default=-1)

    parser.add_argument("--n-points-held", type=int, default=50,
                        help="Number of evenly-spaced held-current levels from 0 to cell_floor_nA, "
                             "the SAME for every cell regardless of floor depth (2026-08-11: replaced "
                             "the old fixed-nA-step coarse grid + boundary-driven adaptive refinement, "
                             "which produced wildly uneven point counts per cell -- 7-42 coarse levels/"
                             "axis alone depending on floor depth, before even counting refinement's "
                             "own unbounded-by-count additions). Deeper-floor cells are sampled at "
                             "coarser absolute nA spacing, not simulated more or fewer times.")
    parser.add_argument("--n-points-injected", type=int, default=50,
                        help="Same as --n-points-held, for the injected-current axis.")
    parser.add_argument("--cell-floor-margin-nA", type=float, default=1.0,
                        help="Held spans [0, silencing_threshold - margin] (cell_floor_nA). "
                             "Margin gives headroom past the cell's own confirmed quiescent level. "
                             "Injected spans further still -- see --injected-floor-margin-nA.")
    parser.add_argument("--injected-floor-margin-nA", type=float, default=1.0,
                        help="Injected spans [0, cell_floor_nA - this], i.e. this much further past "
                             "the cell's confirmed quiescent level than held's own floor (cell_floor_nA) "
                             "-- held stays anchored at cell_floor_nA.")

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
    parser.add_argument("--min-ashman-d", type=float, default=2.0,
                        help="Minimum Ashman's D (separation of the short/long ISI clusters in "
                             "units of pooled SD) required, on top of the ratio and spikes-per-"
                             "burst gates, before classify_burst_pattern will call a window "
                             "'bursting'. D>=2 is the classical mixture-modeling threshold for "
                             "cleanly separated components (Ashman, Bird & Zepf 1994); exposed as "
                             "a flag (unlike min-spikes-per-burst) so it can be swept during audits.")
    parser.add_argument("--ratio-override-ashman-d", type=float, default=5.0,
                        help="A short/long ISI split that fails the ratio gate (--min-isi-ratio) is "
                             "still accepted as 'bursting' if Ashman's D reaches this threshold -- "
                             "a tight, low-variance alternation can be decisively separated (high D) "
                             "even with a modest absolute ratio, which the ratio gate alone can't "
                             "tell apart from noise. Set comfortably above --min-ashman-d so only "
                             "decisively-separated cases use the override, not ordinary borderline "
                             "ones. Confirmed real case: XB2IQX recovery-window 39.35/51.6ms "
                             "alternation, ratio 1.31x (fails 1.5x) but D=30.2.")
    parser.add_argument("--ratio-override-min-ratio", type=float, default=1.15,
                        help="Minimum ratio still required for --ratio-override-ashman-d to apply at "
                             "all -- without this floor, Ashman's D alone is fooled by two KDE-split "
                             "clusters that are almost numerically IDENTICAL (ratio ~1.0-1.05) but "
                             "happen to also have near-zero internal spread, which inflates D even "
                             "though there's no real second timescale -- confirmed directly on a "
                             "rock-steady XB2IQX hold level (ISIs ~45.7-45.8ms) that the KDE still "
                             "quantization-splits into 45.78/45.90ms (ratio 1.0025) with D=4.4, just "
                             "under the default override bar. 1.15 sits with margin above confirmed "
                             "quantization cases (~1.0025-1.006) and margin below the confirmed "
                             "genuine override case (1.31).")
    parser.add_argument("--adaptation-ratio-override-threshold", type=float, default=5.0,
                        help="A test window classify_burst_pattern calls 'tonic' (KDE resolves only "
                             "1 mode) is reclassified 'bursting' if its raw ISI max/min spread ratio "
                             "reaches this threshold -- ordinary tonic firing has a roughly consistent "
                             "ISI by definition, so a spread this extreme on a 'tonic'-labeled train "
                             "means the KDE failed to resolve real structure (typically: the 'long' "
                             "ISI population's own internal spread smeared what would be a second mode "
                             "into a plateau), not that the train is actually tonic. Uses the raw "
                             "spread rather than test_adaptation_ratio specifically since the latter "
                             "is only computed when the window isn't flagged likely-ceased-firing, "
                             "and is confirmed unstable across resimulation on the exact case this "
                             "was added for. Confirmed real case: XB2IQX held=-2.39/inj=-3.37, ISIs "
                             "spanning 15-617ms (~41x spread).")
    parser.add_argument("--isi-alternation-min-rate", type=float, default=0.3,
                        help="Companion gate for --adaptation-ratio-override-threshold: a wide ISI "
                             "spread alone doesn't distinguish a genuinely OSCILLATING (burst-like) "
                             "train from one that just smoothly, monotonically decelerates to a new "
                             "steady rate (ordinary adaptation), which can post an equally large max/"
                             "min ratio. test_isi_alternation_rate is the fraction of consecutive ISI-"
                             "to-ISI steps that flip direction (up-then-down-then-up..., as genuine "
                             "short/long/short/long alternation would); both this and the spread gate "
                             "must pass. Confirmed real case: XB2IQX isis spanning 14.2-296ms (21x, "
                             "clears the spread gate) but alternation_rate=0.15 (a brief onset then "
                             "~30 ISIs of tiny monotonic growth) -- not bursting. A sample of 60 "
                             "currently-overridden points: median rate 0.68, only ~12 percent below 0.3.")
    parser.add_argument("--onset-rescue-max-isis", type=int, default=15,
                        help="classify_rebound_pattern's tier-1 (n >= --min-isis-for-burst-test) onset-"
                             "detector OR-clause only rescues a whole-window KDE 'tonic' verdict for "
                             "trains no longer than this -- a smoothly, monotonically decelerating long "
                             "train (ordinary spike-frequency adaptation) will eventually cross the "
                             "ratio-jump threshold once just from continuous slowing, which "
                             "detect_onset_burst can't tell apart from a genuine burst-then-different-"
                             "rhythm jump. The KDE test is authoritative for genuinely long trains (see "
                             "this function's own docstring); the onset-rescue exists for short, "
                             "borderline-length trains where the KDE doesn't have enough data yet. "
                             "Confirmed real case: XB2IQX held=-3.49/inj=-5.5 recovery, 49 ISIs "
                             "11ms->170ms, KDE correctly says 'tonic', onset-detector spuriously fires "
                             "at ISI #6 (ratio 1.506, barely over 1.5x) purely from ordinary adaptation.")
    parser.add_argument("--sag-window-ms", type=float, default=500.0,
                        help="Window (from test-window onset) over which the pre-spike sag trough "
                             "(test_v_min_pre_spike_mV) is measured, truncated early if a spike occurs "
                             "first. Captures the passive Ih-relaxation sag independent of whether the "
                             "cell goes on to fire tonically/burst -- sag precedes the first spike, so "
                             "it isn't restricted to windows that stay silent throughout.")
    parser.add_argument("--sag-dead-zone-ms", type=float, default=5.0,
                        help="A 'spike' within this many ms of test-window onset doesn't count as the "
                             "first spike for --sag-window-ms truncation purposes. Needed because the "
                             "test window continues directly from the held level's own state -- if held "
                             "fires spontaneously, that state can be mid-upstroke of a spike already "
                             "committed before injected ever took effect, which would otherwise truncate "
                             "the sag window to ~1ms regardless of injected depth.")
    parser.add_argument("--adaptation-edge-n", type=int, default=3,
                        help="Number of ISIs averaged at each end of a tonic test window to compute "
                             "test_adaptation_ratio (mean of the last N ISIs / mean of the first N "
                             "ISIs). Requires test_n_isis >= 2x this value or the point reports None. "
                             "Coupled to --min-isis-for-burst-test in practice: since test_pattern == "
                             "'tonic' already requires >= --min-isis-for-burst-test ISIs (default 6), "
                             "the default of 3 here guarantees every tonic point qualifies -- raising "
                             "this without also raising --min-isis-for-burst-test will silently make "
                             "many tonic points report None.")
    parser.add_argument("--min-onset-isis", type=int, default=2,
                        help="Minimum consecutive short ISIs (i.e. min_onset_isis+1 spikes) required "
                             "at the very start of the test window, immediately followed by a jump of "
                             "at least --min-isi-ratio, to report test_onset_n_spikes/test_onset_isi_"
                             "mean_ms -- a local burst-onset detector that runs independent of "
                             "test_pattern, for onset bursts the whole-window KDE test misses (see "
                             "detect_onset_burst docstring).")
    parser.add_argument("--trailing-silence-ratio", type=float, default=3.0,
                        help="A test window is flagged test_likely_ceased_firing when the gap between "
                             "its last spike and window end is at least this many times the most recent "
                             "ISI -- distinguishes a train that genuinely stopped partway through the "
                             "window from one still firing (slowly) when the window happened to end. "
                             "test_adaptation_ratio is only computed for tonic windows that are NOT "
                             "flagged this way -- a first-k/last-k ISI ratio measures something "
                             "different for a train that dies out mid-window than for one that keeps "
                             "firing steadily throughout.")
    parser.add_argument("--rebound-min-onset-isis", type=int, default=1,
                        help="Same role as --min-onset-isis, but for classify_rebound_pattern's local "
                             "jump check on a recovery-window's own ISIs, not the test window's. Default "
                             "1 (not --min-onset-isis's 2): rebound trains are short by nature, and a "
                             "2-ISI train (3 spikes) needs this at 1 to get any jump check at all -- see "
                             "classify_rebound_pattern's docstring for the confirmed XB2IQX cases this "
                             "distinguishes.")
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
    print(f"\nCache written to {output_cache_path}")

    flagged = [cid for cid, result in this_run_results if result["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'no_silencing_threshold'/'silencing_not_ok': run find_silencing_threshold.py first "
              "(or check its status for that cell). 'blew_up': consider a smaller --dt.")


if __name__ == "__main__":
    main()
