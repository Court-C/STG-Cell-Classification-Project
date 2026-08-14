"""Silencing (cessation) threshold sweep: for each cell, find how much
hyperpolarizing current (Iapp <= 0) is required to suppress spontaneous
pyloric pacemaker oscillation entirely -- and, along the way, whether the
cell passes through a bursting regime before it stops.

Classical "rheobase" does not apply here -- all 69 cells in src/models
already fire spontaneously at Iapp=0 -- so this script sweeps *inward*
(hyperpolarizing) and looks for cessation instead. Excitatory/depolarizing
current and depolarization block are explicitly out of scope for this
project phase.

Each cell is warm-started from its cached Iapp=0 limit-cycle state in
cell_steady_states.pkl (see generate_steady_state.py / steady_state_cache.py)
-- never from DEFAULT_CIS. A cell with no valid cache entry is skipped with
status "no_cached_steady_state"; run generate_steady_state.py for it first.

Sweep strategy is coarse-to-fine continuation: step down by --coarse-step-nA
until firing stops (bracketing the transition), then re-sweep that bracket
at --fine-step-nA. Each step is warm-started from the *previous* step's own
settled state (not from y_ss every time) so the limit cycle is tracked
smoothly as it deforms toward the bifurcation. At each level the cell is
integrated in chunks (--settle-chunk-s) until its firing frequency stabilizes
or --max-settle-s is exhausted -- a fixed short window is not long enough in
general, since period can lengthen sharply near cessation.

The silencing threshold is reported as a fine-resolution bracket (the last
firing level and the first silent level), not forced into a single scalar.
No attempt is made to further characterize the broader shape of the
approach to cessation ("clean" vs. "graded") -- an earlier version of this
script tried that by comparing frequency to baseline across the whole swept
range, but that mostly just measured how far the threshold sits from 0
(more sweep room gives an ordinary smooth decline more room to cross an
arbitrary cutoff before reaching cessation) rather than a real qualitative
difference between cells. The full frequency-vs-Iapp curve is still
recorded, so the shape is visible in the figure/curve directly.

At every tested level this script also classifies the firing pattern itself
-- "tonic", "bursting", or "silent" -- via a log-ISI bimodality test: a
tonic spike train has one dominant inter-spike interval, a burst has two
(a short intra-burst ISI and a long inter-burst gap between the last spike
of one burst and the first spike of the next). Cells are not assumed to
transition tonic->bursting only once on the way to silence -- a cell can go
tonic -> bursting -> tonic before eventually stopping, and every such
transition is recorded, not just the first.
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
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_silencing_thresholds.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "silencing"
DEFAULT_FIGURE_FORMAT = "svg"  # matches generate_steady_state.py's figures/cache convention
# temp == reftemp -- see run_held_injected_grid.py's DEFAULT_TEMP comment.
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

V_INDEX = 0

# Match generate_steady_state.py's spike-detection conventions so frequency
# estimates from this script are comparable to the cached baseline freq_hz.
PROMINENCE_FRACTION = 0.3
FLATLINE_MV = 1.0

# A peak's find_peaks prominence must clear this ABSOLUTE floor (mV), not
# just PROMINENCE_FRACTION of the trace's own min-max range, before counting
# as a spike. Confirmed necessary (2026-08-13) on a real XB2IQX rebound
# point (held=-4.50/inj=-4.38): the recovery-window trace only spans 2.68mV
# total, so a ~1mV subthreshold wobble cleared PROMINENCE_FRACTION*2.68mV
# =~0.8mV and got counted as a "rebound spike" -- nowhere near this model's
# real spike amplitude. Checked against every rebound-occurred point in
# XB2IQX's grid (rebound_peak_mV - recovery_v_min_mV as an amplitude proxy):
# the distribution is cleanly bimodal, genuine spikes at 48-76mV, false
# positives at 1-10mV, nothing in between -- 15.0 sits in that gap with wide
# margin on both sides, so this can't clip a real spike's prominence while
# still catching the subthreshold false positives.
MIN_SPIKE_AMPLITUDE_MV = 15.0


def _spike_prominence(voltage_range_mV: float, prominence_fraction: float) -> float:
    """The prominence floor every spike-detection find_peaks call in this
    project should use: PROMINENCE_FRACTION of the trace's own range, OR
    MIN_SPIKE_AMPLITUDE_MV, whichever is larger -- so a trace confined to a
    narrow subthreshold band can no longer have a small wobble read as
    "prominent" purely because the trace's own range is even smaller.
    """
    return max(voltage_range_mV * prominence_fraction, MIN_SPIKE_AMPLITUDE_MV)


def constant_iapp_func(level_nA: float):
    return lambda t_ms: level_nA


def _round_level(level_nA: float) -> float:
    """Round an Iapp level to a fixed precision before using it as a dict
    key. Different sub-sweeps (coarse, fine, transition-refinement) reach
    "the same" nominal level via different chains of floating-point
    subtraction (e.g. -0.5*5 vs. repeated -=0.05 from a different start),
    landing on bit-distinct floats a few ulps apart (e.g. -2.5 vs.
    -2.4999999999999982). Left unrounded, these show up as separate,
    near-duplicate entries in the merged curve -- observed directly to
    produce a spuriously tiny/degenerate silencing bracket. 1e-6 nA is far
    finer than the smallest --fine-step-nA anyone would reasonably use, so
    this only collapses float noise, never genuinely distinct levels.
    """
    return round(float(level_nA), 6)


def count_spikes_and_rate(voltage: np.ndarray, duration_ms: float,
                          min_peaks_for_rate: int) -> tuple[float, int, bool]:
    """Spike count / duration over one settling chunk. Unlike
    generate_steady_state.estimate_period (which needs >=4 peaks to average
    inter-peak intervals), this only needs a rate estimate, so the bar for
    "nonzero" is lower -- but 0 or 1 peaks in a chunk is still reported as
    freq_hz=0.0 rather than extrapolated from a single interval, since one
    peak carries no interval information.
    """
    if voltage.max() - voltage.min() < FLATLINE_MV:
        return 0.0, 0, True
    peaks, _ = find_peaks(voltage, prominence=_spike_prominence(voltage.max() - voltage.min(),
                                                                PROMINENCE_FRACTION))
    if len(peaks) < min_peaks_for_rate:
        return 0.0, len(peaks), False
    freq_hz = 1000.0 * len(peaks) / duration_ms
    return freq_hz, len(peaks), False


def compute_isis_ms(voltage: np.ndarray, t_ms: np.ndarray,
                    prominence_fraction: float = PROMINENCE_FRACTION) -> tuple[np.ndarray, int]:
    """Inter-spike intervals (ms) from a voltage trace, for the burst-pattern
    bimodality test, plus the raw spike count. The ISI array alone can't
    distinguish "0 spikes" from "1 spike" (both give an empty array, since
    forming even one interval needs 2 spikes) -- returning n_peaks alongside
    it lets a caller tell a genuinely silent-but-not-flatlined window apart
    from one with real, if too-sparse-to-classify, spiking activity (see
    classify_burst_pattern's n_peaks parameter).
    """
    if voltage.max() - voltage.min() < FLATLINE_MV:
        return np.array([]), 0
    peaks, _ = find_peaks(voltage, prominence=_spike_prominence(voltage.max() - voltage.min(),
                                                                prominence_fraction))
    if len(peaks) < 2:
        return np.array([]), len(peaks)
    return np.diff(t_ms[peaks]), len(peaks)


DEFAULT_DVDT_THRESHOLD_MV_PER_MS = 10.0  # same Bean (2007) convention/value as extract_intrinsic_properties.py
DEFAULT_MIN_PRE_SPIKE_MS = 20.0  # matches extract_intrinsic_properties.py's own default


def detect_spikes_dvdt_confirmed(v: np.ndarray, t_ms: np.ndarray, dt_ms: float,
                                 prominence_fraction: float = PROMINENCE_FRACTION,
                                 flatline_mv: float = FLATLINE_MV,
                                 dvdt_threshold: float = DEFAULT_DVDT_THRESHOLD_MV_PER_MS,
                                 min_pre_spike_ms: float = DEFAULT_MIN_PRE_SPIKE_MS) -> tuple:
    """Prominence-based candidate peaks (identical convention to every other
    detector in this project), each then required to have a genuine fast
    upstroke -- dV/dt >= dvdt_threshold somewhere in its own backward search
    window -- before counting as a confirmed spike. Prominence alone can't
    tell a real fast Na+ spike apart from a large but slow non-spike
    transient (e.g. a strong rebound/plateau depolarization) that happens to
    be a big fraction of a window's range; this adds that missing shape
    check as a GATE on top of prominence, not a replacement for it.

    The backward search is anchored AT the peak and walks backward through
    the contiguous run of dV/dt >= dvdt_threshold samples immediately
    preceding it -- NOT a forward scan from search_start for the first
    qualifying sample anywhere in the window. That distinction matters and
    was caught directly during validation on a real bursting cell (MXIC4S):
    a forward scan can land on a fast (but unrelated) dV/dt swing from the
    TAIL of the previous spike's own repolarization/AHP, which sits right at
    the window's start when consecutive spikes are close together (as in a
    burst) -- producing a bogus "threshold" voltage that's actually higher
    than the current peak, a spurious negative amplitude, and a false
    rejection of a genuine spike. Anchoring at the peak and walking backward
    through the LAST (closest-to-peak) contiguous qualifying run avoids
    this: an earlier, disconnected fast-dV/dt event from a prior spike is
    correctly ignored. (extract_intrinsic_properties.py's analyze_spike_
    waveform has the same forward-scan pattern and likely the same latent
    issue -- out of scope to fix here, flagged separately.)

    Returns (confirmed_peaks, rejected_peaks), both index arrays into v, so
    a caller can see exactly what changed rather than just a final count.
    Purely additive: does not change count_spikes_and_rate/compute_isis_ms/
    classify_burst_pattern or any existing call site.
    """
    if v.max() - v.min() < flatline_mv:
        return np.array([], dtype=int), np.array([], dtype=int)
    candidates, _ = find_peaks(v, prominence=_spike_prominence(v.max() - v.min(), prominence_fraction))
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    dvdt = np.gradient(v, dt_ms)  # mV/ms
    back_samples = int(round(min_pre_spike_ms / dt_ms))
    confirmed, rejected = [], []
    for k, peak_idx in enumerate(candidates):
        prev_bound = int(candidates[k - 1]) if k > 0 else 0
        search_start = max(prev_bound, peak_idx - back_samples, 0)
        if search_start >= peak_idx:
            rejected.append(peak_idx)
            continue
        window = dvdt[search_start:peak_idx + 1]
        hits = np.where(window >= dvdt_threshold)[0]
        if len(hits) == 0:
            rejected.append(peak_idx)
            continue
        run_start = int(hits[-1])  # closest-to-peak qualifying sample
        while run_start > 0 and window[run_start - 1] >= dvdt_threshold:
            run_start -= 1
        threshold_idx = search_start + run_start
        amplitude_mV = float(v[peak_idx] - v[threshold_idx])
        if amplitude_mV <= 0:
            rejected.append(peak_idx)
            continue
        confirmed.append(peak_idx)

    return np.array(confirmed, dtype=int), np.array(rejected, dtype=int)


def classify_burst_pattern(isis_ms: np.ndarray, min_isis_for_burst_test: int,
                          isi_mode_prominence_frac: float, min_isi_ratio: float = 1.5,
                          min_spikes_per_burst: float = 1.5, min_ashman_d: float = 2.0,
                          ratio_override_ashman_d: float = 5.0, ratio_override_min_ratio: float = 1.15,
                          n_peaks: int = None) -> dict:
    """Log-ISI bimodality test. A tonically firing cell has one dominant
    inter-spike interval (log(ISI) density is unimodal); a burster has two
    -- a short mode (intra-burst ISI) and a long mode (the gap from the last
    spike of one burst to the first spike of the next). Requires at least
    min_isis_for_burst_test ISIs to attempt the test at all -- with too few
    spikes the test is unreliable and is reported as "insufficient_data"
    (n_peaks == 0: no real spikes detected, just some subthreshold voltage
    excursion) or "sparse" (n_peaks >= 1: the cell genuinely spiked, just not
    enough to run the bimodality test confidently) rather than guessed.
    "sparse" is real, observed data, not a data gap -- distinguishing it from
    "insufficient_data" avoids conflating "we don't know what this point is
    doing" with "this point rarely fires," which n_peaks alone can tell apart
    (isis_ms can't: 0 or 1 spike both give an empty ISI array, since forming
    even one interval needs 2 spikes). n_peaks=None keeps the old
    "insufficient_data" label for call sites that don't have a spike count
    handy.

    Four gates must all pass before a candidate short/long ISI split is
    accepted as "bursting": enough ISIs to run the test at all, the ratio
    gate (min_isi_ratio) -- overridable when Ashman's D is decisively high
    (ratio_override_ashman_d) -- the spikes-per-burst gate
    (min_spikes_per_burst), and finally the Ashman's D gate (min_ashman_d)
    itself -- see the comments at each gate for why every one of them is
    needed on top of the others.

    Every return also carries a "diagnostics" key -- None wherever no KDE
    was ever evaluated (too few ISIs, degenerate/trivial ISI spread, or a
    failed KDE fit), otherwise a dict of the intermediate log-ISI KDE curve,
    mode locations, and valley split point behind this call, including for
    every "tonic" rejection reached after the KDE ran (ratio-test and
    spikes-per-burst-test rejections included). Purely additive -- existing
    callers that only read "pattern"/"isi_short_ms"/etc. are unaffected --
    and exists so a caller (see src/trace_annotations.py) can plot exactly
    why a borderline point was or wasn't called bursting, including the
    near-miss rejections (e.g. the VC08B6 case documented below) that are
    otherwise invisible outside this function.
    """
    n = len(isis_ms)
    if n < min_isis_for_burst_test:
        pattern = "insufficient_data" if not n_peaks else "sparse"
        return {"pattern": pattern, "isi_short_ms": None,
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": None,
                "n_long_isis": None, "diagnostics": None}

    log_isi = np.log10(isis_ms)
    if np.ptp(log_isi) < 1e-6:
        # All ISIs effectively identical -- trivially unimodal/tonic.
        return {"pattern": "tonic", "isi_short_ms": float(np.mean(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": 0.0,
                "n_long_isis": None, "diagnostics": None}

    # Pad the evaluation grid past [min, max]: a mode sitting at or near a
    # data extreme (very common here -- the short-ISI cluster starts right
    # at log_isi.min(), the long-ISI cluster sits right at log_isi.max())
    # would otherwise have no interior neighbor for find_peaks to compare
    # against and would be silently missed, always yielding "tonic".
    try:
        kde = gaussian_kde(log_isi)
    except np.linalg.LinAlgError:
        return {"pattern": "tonic", "isi_short_ms": float(np.median(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": None,
                "n_long_isis": None, "diagnostics": None}

    pad = 0.15 * np.ptp(log_isi)
    grid = np.linspace(log_isi.min() - pad, log_isi.max() + pad, 200)
    density = kde(grid)
    peaks, _ = find_peaks(density, prominence=density.max() * isi_mode_prominence_frac)
    # "kde" (the fitted callable itself, not just this call's own padded-to-
    # its-own-data grid/density) is kept so a caller comparing multiple
    # panels can re-evaluate the SAME fit over a shared, wider x-range for
    # plotting -- extending the visible window this way redraws the actual
    # already-fitted curve further out, it does not fabricate/smooth new
    # data. See trace_annotations.mark_isi_classification's log_isi_xlim.
    diagnostics = {"log_isi_grid": grid, "density": density, "candidate_mode_idx": peaks,
                  "isi_mode_prominence_frac": isi_mode_prominence_frac, "kde": kde}

    if len(peaks) < 2:
        return {"pattern": "tonic", "isi_short_ms": float(np.median(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": 0.0,
                "n_long_isis": None, "diagnostics": diagnostics}

    # Keep the two tallest candidate modes; split at the valley between them.
    top2 = peaks[np.argsort(density[peaks])[-2:]]
    mode_lo, mode_hi = sorted(top2)
    valley_idx = mode_lo + int(np.argmin(density[mode_lo:mode_hi + 1]))
    split_log_isi = grid[valley_idx]
    diagnostics.update({"mode_lo_log_isi": float(grid[mode_lo]), "mode_hi_log_isi": float(grid[mode_hi]),
                        "split_log_isi": float(split_log_isi)})

    short_mask = log_isi <= split_log_isi
    long_mask = ~short_mask
    if not short_mask.any() or not long_mask.any():
        return {"pattern": "tonic", "isi_short_ms": float(np.median(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": None,
                "n_long_isis": None, "diagnostics": diagnostics}

    isi_short_ms = float(np.mean(isis_ms[short_mask]))
    isi_long_ms = float(np.mean(isis_ms[long_mask]))
    diagnostics.update({"isi_short_ms": isi_short_ms, "isi_long_ms": isi_long_ms,
                        "min_isi_ratio": min_isi_ratio, "short_mask": short_mask})

    # Ashman's D computed HERE (moved ahead of the ratio gate, 2026-08-14) --
    # see its own comment below for the full definition/citation. Needed
    # this early so a very strong separation can rescue a case the ratio
    # gate alone would reject: confirmed real case, XB2IQX held=-2.94/
    # inj=-3.03 recovery window, a clean, low-variance 39.35ms/51.6ms
    # alternation (ratio 1.31x, under the 1.5x bar) that nonetheless scores
    # D=30.2 -- about as cleanly separated as any case in this dataset. The
    # ratio gate's own justification (intra-/inter-burst gaps "differ by
    # 10x or more" in genuine cases) was calibrated against noisy/high-
    # variance examples; it was never meant to veto a tight, low-noise
    # alternation just because the two means happen to be numerically close.
    s_short, s_long = np.std(isis_ms[short_mask]), np.std(isis_ms[long_mask])
    pooled = np.sqrt((s_short ** 2 + s_long ** 2) / 2.0)
    ashman_d = abs(isi_long_ms - isi_short_ms) / pooled if pooled > 0 else float("inf")
    diagnostics["ashman_d"] = float(ashman_d)

    # A KDE peak count of 2 is not, by itself, reliable evidence of two
    # genuinely distinct timescales -- with a modest, discretized (fixed-dt)
    # sample, quantization/sampling noise can split an essentially unimodal
    # tonic ISI cluster into two adjacent "peaks" a fraction of a timestep
    # apart (observed directly: isi_short=17.0ms, isi_long=17.1ms for a
    # clearly tonic spike train). Require the long mode to be a meaningfully
    # different timescale, not just a different KDE bin, before calling it
    # bursting -- intra-burst vs inter-burst gaps in this model differ by
    # 10x or more, so this threshold has a lot of room below genuine cases.
    # Overridable when ashman_d clears ratio_override_ashman_d (default 5.0,
    # well above the normal min_ashman_d=2.0 "cleanly separated" bar, so an
    # ordinary borderline case still can't sneak through a weak ratio just
    # because it's non-noisy -- this is reserved for cases as decisively
    # separated as the 39.35/51.6ms example above) AND ratio clears
    # ratio_override_min_ratio (default 1.15). The min-ratio floor matters
    # a great deal: D is a RATIO of separation to spread, so it explodes
    # for two clusters that are almost numerically identical but happen to
    # also have near-zero internal spread -- confirmed directly on a
    # rock-steady XB2IQX hold level (isis essentially constant at
    # 45.7-45.8ms): the KDE still quantization-splits it into two adjacent
    # pseudo-modes at 45.78/45.90ms (ratio 1.0025, obviously not two real
    # timescales) and D reaches 4.4 purely because each pseudo-mode's own
    # spread is nearly zero -- almost enough to clear a D-only override.
    # 1.15 sits with real margin above confirmed quantization cases (this
    # one and the ratio gate's own original 17.0/17.1ms example, ~1.006-
    # 1.0025) and real margin below the confirmed genuine case (1.31).
    ratio = isi_long_ms / isi_short_ms
    ratio_rescued = ratio >= ratio_override_min_ratio and ashman_d >= ratio_override_ashman_d
    if ratio < min_isi_ratio and not ratio_rescued:
        return {"pattern": "tonic", "isi_short_ms": float(np.median(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": float(ashman_d),
                "n_long_isis": None, "diagnostics": diagnostics}

    # A burst is defined by >=2 spikes in rapid succession, not by ISI
    # timescale alone -- confirmed directly this matters: a real VC08B6
    # point (held=-12.5, inj=-4.4) passed the isi_long/isi_short ratio check
    # (9.6ms vs 14.5ms, 1.5x, right at the old threshold) and got called
    # "bursting" with n_bursts=174, but averaged only 1.29 spikes per burst
    # -- i.e. almost every "burst" was a single isolated spike with mild ISI
    # jitter around it, not a real burst. A near-unimodal but slightly noisy
    # tonic ISI cluster can get KDE-split into two adjacent modes that both
    # individually look "short" (9-15ms, nothing like the 30ms+ genuine
    # inter-burst gaps seen in confirmed real bursters), which the ratio
    # check alone doesn't catch. n = len(isis_ms) is the total ISI count
    # (n+1 spikes); n_long is the count of "long" (inter-burst) ISIs, so
    # there are n_long+1 bursts and (n+1)/(n_long+1) average spikes/burst.
    #
    # Threshold is 1.5, not a literal 2.0, because of a confirmed off-by-one
    # edge effect: a perfectly clean, genuine doublet train (real burst of
    # exactly 2 spikes, repeating) computes to just UNDER 2.0 whenever the
    # sampled window happens to end on a long (inter-burst) ISI -- the
    # trailing spike after that last gap has no partner within the window,
    # so one "burst" out of n_long+1 is an incomplete 1-spike fragment,
    # pulling the average to 2 - 2/n_bursts (e.g. 65 real spikes forming 32
    # clean doublets plus one dangling trailing spike -> 65/33 = 1.97, not
    # 2.0). Confirmed directly on a synthetic clean 31.4ms/60.0ms alternating
    # doublet train (only Gaussian jitter, no other noise) -- it scored 1.97
    # and would have been wrongly rejected by a literal >=2.0 cutoff. 1.5
    # keeps a wide margin below genuine doublets (~1.97-2.0+) while staying
    # well above the confirmed-spurious VC08B6 case (~1.29-1.34).
    n_long = int(long_mask.sum())
    avg_spikes_per_burst = (n + 1) / (n_long + 1)
    diagnostics["avg_spikes_per_burst"] = avg_spikes_per_burst
    if avg_spikes_per_burst < min_spikes_per_burst:
        return {"pattern": "tonic", "isi_short_ms": float(np.median(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": float(ashman_d),
                "n_long_isis": None, "diagnostics": diagnostics}

    # Ashman's D: separation of two Gaussians in units of pooled SD --
    # D = |mean_long - mean_short| / sqrt((sd_short^2 + sd_long^2) / 2).
    # D >= 2 is the classical mixture-modeling threshold for declaring two
    # components "cleanly" separated (Ashman, Bird & Zepf 1994; the log-ISI
    # valley-split approach this whole function implements is itself the
    # field-standard substrate for burst/tonic classification, e.g. Selinger
    # et al. 2007's void parameter / Pasquale et al. 2010's LogISI method,
    # ranked #2 of 8 methods in Cotterill et al. 2016's ground-truth
    # comparison). The other gates catch specific known false-positive
    # SHAPES (quantization-noise pseudo-modes a timestep apart; a minority of
    # stray short ISIs scattered through an otherwise-uniform train, e.g.
    # VC08B6) but none of them measure whether the two candidate clusters
    # are actually well-separated POPULATIONS rather than overlapping ones --
    # a short cluster with heavy internal spread can pass both the ratio and
    # count checks while still bleeding into the long cluster's range. D
    # itself is computed earlier now (see above, 2026-08-14) so it can also
    # rescue a modest-ratio case; this final check is the ordinary "reject
    # if not cleanly separated" gate for everything that already passed.
    if ashman_d < min_ashman_d:
        return {"pattern": "tonic", "isi_short_ms": float(np.median(isis_ms)),
                "isi_long_ms": None, "n_isis": n, "bimodality_metric": float(ashman_d),
                "n_long_isis": None, "diagnostics": diagnostics}

    # n_long_isis (the count of inter-burst gaps) lets a caller segment the
    # spike train into discrete bursts without redoing this KDE split: a
    # "burst" is a maximal run of consecutive short ISIs, so there are
    # exactly n_long_isis + 1 bursts, and (n_isis + 1) / (n_long_isis + 1)
    # is the average spikes-per-burst (see run_held_injected_grid.py).
    return {"pattern": "bursting", "isi_short_ms": isi_short_ms,
            "isi_long_ms": isi_long_ms, "n_isis": n, "bimodality_metric": float(ashman_d),
            "n_long_isis": int(long_mask.sum()), "diagnostics": diagnostics}


def to_stored_pattern(pattern: str) -> str:
    """Collapses classify_burst_pattern's two "not enough data" intermediate
    labels ("insufficient_data": zero spikes, "sparse": some spikes but too
    few to run the bimodality test) into a single reported "silent" -- a
    handful of scattered spikes that can't sustain a classifiable rhythm is
    functionally silent for this project's tonic/bursting/silent
    classification, not a fourth ambiguous category to report. The two
    labels are kept distinct ONLY as an internal signal for callers that
    retry with a longer window when they see one (see
    run_held_injected_grid.py's run_test_and_recovery) -- this collapse is
    meant to be applied to whatever pattern comes out AFTER that retry
    decision is made (succeeded, or gave up and exhausted its window
    budget), not before.
    """
    return "silent" if pattern in ("insufficient_data", "sparse") else pattern


def settle_at_level(params: np.ndarray, y0: np.ndarray, level_nA: float,
                    dt: float, temp: float, reftemp: float,
                    chunk_s: float, max_settle_s: float, settle_rtol: float,
                    min_peaks_for_rate: int) -> dict:
    """Integrate at a constant Iapp level in chunks, carrying state forward,
    until the firing frequency stabilizes across two consecutive chunks or
    max_settle_s is exhausted. Returns settled=False (not raised) if it never
    stabilizes -- the caller decides whether that's decision-critical.
    """
    state = y0.copy()
    total_s = 0.0
    prev_freq = None
    prev_flat = None
    consecutive_stable = 0
    first_chunk_v = None
    first_chunk_t = None
    last_chunk_v = None
    last_chunk_t = None
    Iapp_func = constant_iapp_func(level_nA)

    while total_s < max_settle_s:
        try:
            t_ms, states = simulate(params, chunk_s, temp, dt=dt, reftemp=reftemp,
                                    cis=state, Iapp_func=Iapp_func)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            return {"blew_up": True, "settled": False, "error": str(exc)}

        if not np.all(np.isfinite(states)):
            return {"blew_up": True, "settled": False, "error": "non-finite trajectory"}

        v = states[:, V_INDEX]
        if first_chunk_v is None:
            first_chunk_v, first_chunk_t = v, t_ms
        last_chunk_v, last_chunk_t = v, t_ms

        freq, n_peaks, flat = count_spikes_and_rate(v, chunk_s * 1000.0, min_peaks_for_rate)
        state = states[-1]
        total_s += chunk_s

        if prev_freq is not None:
            rel_diff = abs(freq - prev_freq) / max(prev_freq, 1e-9)
            if (rel_diff < settle_rtol) or (flat and prev_flat):
                consecutive_stable += 1
            else:
                consecutive_stable = 0
        prev_freq, prev_flat = freq, flat

        if consecutive_stable >= 2:
            return {"blew_up": False, "settled": True, "freq_hz": freq, "is_flatline": flat,
                    "final_state": state, "total_settle_s_used": total_s,
                    "first_chunk_v": first_chunk_v, "first_chunk_t": first_chunk_t,
                    "last_chunk_v": last_chunk_v, "last_chunk_t": last_chunk_t}

    return {"blew_up": False, "settled": False, "freq_hz": prev_freq, "is_flatline": prev_flat,
            "final_state": state, "total_settle_s_used": total_s,
            "first_chunk_v": first_chunk_v, "first_chunk_t": first_chunk_t,
            "last_chunk_v": last_chunk_v, "last_chunk_t": last_chunk_t}


def run_isi_window(params: np.ndarray, state: np.ndarray, level_nA: float,
                   dt: float, temp: float, reftemp: float, isi_window_s: float) -> dict:
    """One dedicated continuation chunk purely to collect enough ISIs for the
    burst-pattern classification. Only used during fine-resolution passes
    (see probe_level) -- near a tonic/burst transition, burst period can be
    much longer than the settling chunk length, so a short reused chunk would
    under-sample right where the classification matters most.
    """
    Iapp_func = constant_iapp_func(level_nA)
    try:
        t_ms, states = simulate(params, isi_window_s, temp, dt=dt, reftemp=reftemp,
                                cis=state, Iapp_func=Iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {"blew_up": True, "error": str(exc)}
    if not np.all(np.isfinite(states)):
        return {"blew_up": True, "error": "non-finite trajectory"}
    return {"blew_up": False, "v": states[:, V_INDEX], "t_ms": t_ms}


def probe_level(params: np.ndarray, state: np.ndarray, level_nA: float,
                dt: float, temp: float, reftemp: float,
                chunk_s: float, max_settle_s: float, settle_rtol: float, min_peaks_for_rate: int,
                use_isi_window: bool, isi_window_s: float, min_isis_for_burst_test: int,
                isi_mode_prominence_frac: float, min_isi_ratio: float) -> dict:
    """Settle at a level, then classify its firing pattern (tonic / bursting /
    silent / insufficient_data). use_isi_window controls whether a dedicated
    extra simulate() call is spent to gather enough ISIs (fine-resolution
    passes) or whether the already-settled chunk is reused (coarse passes --
    cheap, but can under-sample a slow-bursting regime). Silent levels skip
    the ISI test entirely, since there's nothing to classify and no point
    paying for the extra window.
    """
    r = settle_at_level(params, state, level_nA, dt, temp, reftemp,
                        chunk_s, max_settle_s, settle_rtol, min_peaks_for_rate)
    if r["blew_up"]:
        return r

    if r["is_flatline"]:
        r["burst_pattern"] = "silent"
        r["isi_short_ms"] = None
        r["isi_long_ms"] = None
        r["n_isis"] = 0
        r["n_spikes"] = 0
        return r

    if use_isi_window:
        isi_result = run_isi_window(params, r["final_state"], level_nA, dt, temp, reftemp, isi_window_s)
        if isi_result["blew_up"]:
            r["blew_up"] = True
            r["error"] = isi_result.get("error")
            return r
        isis_ms, n_peaks = compute_isis_ms(isi_result["v"], isi_result["t_ms"])
    elif r["last_chunk_v"] is not None:
        isis_ms, n_peaks = compute_isis_ms(r["last_chunk_v"], r["last_chunk_t"])
    else:
        isis_ms, n_peaks = np.array([]), 0

    burst_info = classify_burst_pattern(isis_ms, min_isis_for_burst_test, isi_mode_prominence_frac, min_isi_ratio,
                                        n_peaks=n_peaks)
    # Kept as the RAW (uncollapsed) label here -- find_pattern_transitions
    # needs to tell "insufficient_data"/"sparse" (bridge over, keep
    # scanning) apart from genuine "silent" (stop scanning). Collapsed to
    # the reported 3-category label only in _finalize_curve, once all
    # scanning/transition-detection that needs the raw distinction is done.
    r["burst_pattern"] = burst_info["pattern"]
    r["isi_short_ms"] = burst_info["isi_short_ms"]
    r["isi_long_ms"] = burst_info["isi_long_ms"]
    r["n_isis"] = burst_info["n_isis"]
    r["n_spikes"] = n_peaks
    return r


def find_pattern_transitions(levels: list, patterns: list) -> list:
    """Scan a (levels, patterns) curve in sweep order (0 -> increasingly
    hyperpolarizing) for tonic<->bursting transitions, bridging over
    "insufficient_data" gaps by carrying the last confidently-classified
    label forward. Stops considering further transitions once "silent" is
    reached, since silencing is tracked separately by the threshold sweep.
    Cells are not assumed to transition only once -- a cell can go
    tonic -> bursting -> tonic before eventually silencing, and every such
    transition is reported here, not just the first.
    """
    transitions = []
    last_confident_level = None
    last_confident_label = None
    for level, pattern in zip(levels, patterns):
        if pattern == "silent":
            break
        if pattern == "insufficient_data":
            continue
        if last_confident_label is not None and pattern != last_confident_label:
            transitions.append({"direction": f"{last_confident_label}_to_{pattern}",
                                "before_level_nA": last_confident_level,
                                "after_level_nA": level})
        last_confident_label = pattern
        last_confident_level = level
    return transitions


def refine_transition(params: np.ndarray, start_state: np.ndarray,
                      before_level_nA: float, after_level_nA: float, fine_step_nA: float,
                      target_pattern: str, max_extra_steps: int, probe_kwargs: dict) -> dict:
    """Given a coarse-resolution bracket [before_level_nA (old pattern),
    after_level_nA (new/target pattern)], step through it at fine resolution
    with a dedicated ISI window per level, independently re-confirming
    target_pattern rather than trusting the coarse call -- the silencing
    threshold sweep found this model shows path-dependent (hysteretic)
    settling near transitions, so a single large coarse jump can land on a
    different regime than a sequence of small fine steps does. Extends past
    the original bracket by up to max_extra_steps if needed before giving up
    and reporting confirmed=False (the coarse-resolution bracket is still
    returned to the caller in that case, just flagged as unconfirmed).
    """
    levels, probes = [], []
    state = start_state
    level = _round_level(before_level_nA - fine_step_nA)
    consecutive_target = 0
    confirmed = False
    n_bracket_steps = max(1, int(round(abs(before_level_nA - after_level_nA) / fine_step_nA)))
    step_budget = n_bracket_steps + max_extra_steps

    for _ in range(step_budget):
        r = probe_level(params, state, level, **probe_kwargs)
        levels.append(level)
        probes.append(r)
        if r["blew_up"]:
            break
        state = r["final_state"]
        if r["burst_pattern"] == target_pattern:
            consecutive_target += 1
            if consecutive_target >= 2:
                confirmed = True
                break
        else:
            consecutive_target = 0
        level = _round_level(level - fine_step_nA)

    return {"levels": levels, "probes": probes, "confirmed": confirmed}


def _finalize_curve(merged: dict) -> dict:
    """Turn a level -> probe-result dict into sorted parallel arrays (0 ->
    most hyperpolarizing) plus a level -> (t_ms, v) trace lookup used for
    figure plotting. Single source of truth for every array/derived field
    that depends on "all levels tested so far", so coarse-scan, fine-scan,
    and transition-refinement levels are always merged consistently.
    """
    ordered_levels = sorted(merged.keys(), reverse=True)
    freqs = [merged[lv].get("freq_hz") or 0.0 for lv in ordered_levels]
    settled = [bool(merged[lv].get("settled")) for lv in ordered_levels]
    # Collapsed to the reported 3-category label (tonic/bursting/silent)
    # here, at the very end -- find_pattern_transitions/refine_transition
    # already ran on the raw per-level "burst_pattern" values upstream
    # (via merged directly, not through this function) and need the raw
    # "insufficient_data"/"sparse" distinction to bridge over ambiguous
    # gaps correctly; only the final persisted/reported curve collapses.
    patterns = [to_stored_pattern(merged[lv].get("burst_pattern", "insufficient_data"))
               for lv in ordered_levels]
    isi_short = [merged[lv].get("isi_short_ms") for lv in ordered_levels]
    isi_long = [merged[lv].get("isi_long_ms") for lv in ordered_levels]
    fine_traces = {lv: (merged[lv]["last_chunk_t"], merged[lv]["last_chunk_v"])
                  for lv in ordered_levels if merged[lv].get("last_chunk_v") is not None}
    return {
        "iapp_levels_nA": np.array(ordered_levels),
        "freq_hz_curve": np.array(freqs),
        "settled_flags": np.array(settled),
        "burst_pattern_curve": np.array(patterns),
        "isi_short_ms_curve": np.array([v if v is not None else np.nan for v in isi_short], dtype=float),
        "isi_long_ms_curve": np.array([v if v is not None else np.nan for v in isi_long], dtype=float),
        "fine_traces": fine_traces,
    }


def sweep_cell(params: np.ndarray, ss_entry: dict, args: argparse.Namespace) -> dict:
    """Coarse-to-fine hyperpolarizing sweep for one cell. Tracks the
    silencing threshold (as before) and, simultaneously, the firing-pattern
    (tonic/bursting/silent) curve, from which every tonic<->bursting
    transition is detected and independently fine-refined.
    """
    y_ss = ss_entry["y_ss"]
    settle_kwargs = dict(dt=args.dt, temp=args.temp, reftemp=args.reftemp,
                         chunk_s=args.settle_chunk_s, max_settle_s=args.max_settle_s,
                         settle_rtol=args.settle_rtol, min_peaks_for_rate=args.min_peaks_for_rate)
    isi_kwargs = dict(isi_window_s=args.isi_window_s,
                      min_isis_for_burst_test=args.min_isis_for_burst_test,
                      isi_mode_prominence_frac=args.isi_mode_prominence_frac,
                      min_isi_ratio=args.min_isi_ratio)
    coarse_probe_kwargs = {**settle_kwargs, **isi_kwargs, "use_isi_window": False}
    fine_probe_kwargs = {**settle_kwargs, **isi_kwargs, "use_isi_window": True}

    merged: dict = {}

    # Baseline (Iapp=0) probe, reusing the burn-in trace already cached by
    # generate_steady_state.py -- zero extra simulation cost. Without this,
    # a cell that's already bursting at the very first coarse step (observed
    # directly: 2EXYPV) would silently miss its tonic->bursting transition,
    # since it may have happened anywhere between 0 and -coarse_step_nA,
    # a range the coarse scan itself never probes.
    baseline_isis, baseline_n_peaks = compute_isis_ms(ss_entry["burnin_v"], ss_entry["burnin_t"])
    baseline_burst = classify_burst_pattern(baseline_isis, args.min_isis_for_burst_test,
                                            args.isi_mode_prominence_frac, args.min_isi_ratio,
                                            n_peaks=baseline_n_peaks)
    merged[0.0] = {
        "blew_up": False, "settled": True, "freq_hz": ss_entry["freq_hz"], "is_flatline": False,
        "final_state": y_ss.copy(), "total_settle_s_used": 0.0,
        "first_chunk_v": ss_entry["burnin_v"], "first_chunk_t": ss_entry["burnin_t"],
        "last_chunk_v": ss_entry["burnin_v"], "last_chunk_t": ss_entry["burnin_t"],
        "burst_pattern": baseline_burst["pattern"], "isi_short_ms": baseline_burst["isi_short_ms"],
        "isi_long_ms": baseline_burst["isi_long_ms"], "n_isis": baseline_burst["n_isis"],
        "n_spikes": baseline_n_peaks,
    }

    # --- coarse scan ---
    state = y_ss.copy()
    level = _round_level(-args.coarse_step_nA)
    consecutive_flat = 0
    coarse_levels_order = []
    coarse_status = None
    while level >= args.max_hyperpolarizing_nA:
        r = probe_level(params, state, level, **coarse_probe_kwargs)
        if r["blew_up"]:
            curve = _finalize_curve(merged)
            return {"status": "blew_up", "error": r.get("error"), **curve}

        merged[level] = r
        coarse_levels_order.append(level)
        state = r["final_state"]

        if r["is_flatline"]:
            consecutive_flat += 1
            if consecutive_flat >= 2:
                coarse_status = "bracketed"
                break
        else:
            consecutive_flat = 0
        level = _round_level(level - args.coarse_step_nA)
    else:
        coarse_status = "did_not_silence_within_range"

    if coarse_status == "did_not_silence_within_range":
        curve = _finalize_curve(merged)
        return {"status": "did_not_silence_within_range", **curve}

    # bracket: last firing level (2 flat levels ago) -> first-of-the-two-flat level
    first_silent_idx = len(coarse_levels_order) - 2
    last_firing_idx = first_silent_idx - 1
    while (last_firing_idx >= 0
          and merged[coarse_levels_order[last_firing_idx]]["settled"] is False
          and not merged[coarse_levels_order[last_firing_idx]]["freq_hz"]):
        last_firing_idx -= 1

    if last_firing_idx < 0:
        # Silenced on the very first coarse step -- bracket is [0, -coarse_step]
        bracket_hi = 0.0
        bracket_lo = coarse_levels_order[first_silent_idx]
        fine_start_state = y_ss.copy()
    else:
        bracket_hi = coarse_levels_order[last_firing_idx]
        bracket_lo = coarse_levels_order[first_silent_idx]
        fine_start_state = merged[bracket_hi]["final_state"]

    # --- fine scan for silencing, continuation from the last-firing state ---
    # Deliberately does NOT stop at bracket_lo: this model shows path-dependent
    # (hysteretic) settling near the transition, so the coarse pass's own
    # "2 consecutive flat" call about bracket_lo is not trustworthy at fine
    # resolution. Re-applies the same stopping rule independently, continuing
    # past bracket_lo if needed, all the way out to max_hyperpolarizing_nA.
    state = fine_start_state
    level = _round_level(bracket_hi - args.fine_step_nA)
    consecutive_flat = 0
    fine_bracketed = False
    while level >= args.max_hyperpolarizing_nA:
        r = probe_level(params, state, level, **fine_probe_kwargs)
        if r["blew_up"]:
            curve = _finalize_curve(merged)
            return {"status": "blew_up", "error": r.get("error"), **curve}

        merged[level] = r
        state = r["final_state"]

        if r["is_flatline"]:
            consecutive_flat += 1
            if consecutive_flat >= 2:
                fine_bracketed = True
                break
        else:
            consecutive_flat = 0
        level = _round_level(level - args.fine_step_nA)

    # --- tonic<->bursting transitions: detect on the baseline point plus the
    # coarse curve up to the last firing coarse level, then independently
    # fine-refine each one ---
    coarse_curve_levels = [0.0] + (coarse_levels_order[:last_firing_idx + 1] if last_firing_idx >= 0 else [])
    coarse_curve_patterns = [merged[lv]["burst_pattern"] for lv in coarse_curve_levels]
    coarse_transitions = find_pattern_transitions(coarse_curve_levels, coarse_curve_patterns)

    max_extra_steps = max(4, int(round(4 * args.coarse_step_nA / args.fine_step_nA)))
    burst_transition_records = []
    for trans in coarse_transitions:
        before_level = trans["before_level_nA"]
        after_level = trans["after_level_nA"]
        target_pattern = trans["direction"].split("_to_")[-1]
        start_state = merged[before_level]["final_state"]
        refine = refine_transition(params, start_state, before_level, after_level,
                                   args.fine_step_nA, target_pattern, max_extra_steps,
                                   fine_probe_kwargs)
        for lv, r in zip(refine["levels"], refine["probes"]):
            merged.setdefault(lv, r)

        if refine["confirmed"]:
            tail_levels = refine["levels"][-2:]
            bracket_nA = (min(tail_levels), max(tail_levels))
            resolution = "fine"
        else:
            bracket_nA = (min(before_level, after_level), max(before_level, after_level))
            resolution = "coarse_unconfirmed"
        burst_transition_records.append({"direction": trans["direction"],
                                         "bracket_nA": bracket_nA, "resolution": resolution})

    curve = _finalize_curve(merged)
    curve["burst_transitions"] = burst_transition_records
    curve["bracket_hi"] = bracket_hi
    curve["bracket_lo"] = bracket_lo
    curve["status"] = "swept" if fine_bracketed else "did_not_silence_within_range"
    return curve


def find_silencing_bracket(sweep: dict) -> dict:
    """Identify the fine-resolution bracket [first_silent_level,
    last_firing_level] around cessation from the swept frequency curve,
    escalating to "unsettled" if the decision-critical (near-bracket) levels
    never settled -- i.e. settle_at_level ran through the full
    --max-settle-s budget without two consecutive chunks agreeing on firing
    rate there. This is not a root-finding/fixed-point convergence failure
    like generate_steady_state.py's shooting method (there's no equation
    being solved here, no target being converged toward) -- it just means
    the firing-rate estimate didn't stabilize in the time given, so the
    reported bracket at that level isn't trustworthy.

    Earlier versions of this function also tried to classify the broader
    approach to cessation as "clean" (frequency holds near baseline then
    cliffs) vs. "graded" (frequency declines gradually), by comparing
    frequency against baseline across the *entire* swept range. That test
    was dropped: across a batch of real cells it came back "graded" for
    essentially every cell, because it was really just measuring how far
    the threshold sits from 0 (a cell with more hyperpolarizing range has
    more room for an ordinary smooth decline to cross an arbitrary
    fraction-of-baseline cutoff before reaching cessation) rather than any
    real qualitative difference in cessation behavior between cells. The
    full frequency-vs-Iapp curve is still recorded in full, so the shape is
    visible in the figure/curve -- it's just not collapsed into a label.
    """
    # Copy and drop "status" -- sweep_cell's own status ("swept") is only a
    # marker that classification hasn't run yet; every return below sets its
    # own status, and **sweep must not be allowed to clobber it back.
    sweep = {k: v for k, v in sweep.items() if k != "status"}
    levels = sweep["iapp_levels_nA"]
    freqs = sweep["freq_hz_curve"]
    settled = sweep["settled_flags"]

    nonzero = np.nonzero(freqs > 0.0)[0]
    if nonzero.size == 0:
        # Silenced before any fine level could register a rate -- treat as
        # bracketed at 0 with the first tested level.
        k = -1
    else:
        k = nonzero[-1]
    silent_idx = k + 1
    if silent_idx >= len(levels):
        return {"status": "did_not_silence_within_range", **sweep}

    # decision-critical window: last firing level, first silent level, and
    # one level on either side if present
    critical = settled[max(0, k):min(len(settled), silent_idx + 1)]
    if not np.all(critical):
        return {"status": "unsettled",
                "last_firing_level_nA": levels[k] if k >= 0 else 0.0,
                "first_silent_level_nA": levels[silent_idx], **sweep}

    last_firing_level_nA = levels[k] if k >= 0 else 0.0
    first_silent_level_nA = levels[silent_idx]
    return {"status": "ok",
            "silencing_threshold_bracket_nA": (first_silent_level_nA, last_firing_level_nA),
            "last_firing_level_nA": last_firing_level_nA,
            "first_silent_level_nA": first_silent_level_nA, **sweep}


def summarize_burst_transitions(burst_pattern_curve: np.ndarray, burst_transitions: list) -> dict:
    """Post-hoc summary of the whole firing-pattern curve: a collapsed
    sequence of distinct patterns in sweep order (e.g. ["tonic", "bursting",
    "tonic"] for a cell that bursts transiently before returning to tonic
    firing and eventually silencing), whether bursting appears anywhere, and
    the bracket of the *first* tonic->bursting transition (the "does it
    burst before stopping, and where" answer) -- None if the cell never
    bursts on the way to silence.
    """
    # burst_pattern_curve is always the post-_finalize_curve collapsed
    # curve (tonic/bursting/silent only, see to_stored_pattern) -- no
    # "insufficient_data"/"sparse" entries ever reach here.
    sequence = []
    for pattern in burst_pattern_curve:
        if not sequence or sequence[-1] != pattern:
            sequence.append(str(pattern))
    tonic_to_burst = [t for t in burst_transitions if t["direction"] == "tonic_to_bursting"]
    any_bursting = bool(np.any(burst_pattern_curve == "bursting")) if len(burst_pattern_curve) else False
    return {
        "overall_pattern_sequence": sequence,
        "any_bursting_detected": any_bursting,
        "first_tonic_to_burst_bracket_nA": tonic_to_burst[0]["bracket_nA"] if tonic_to_burst else None,
    }


def _trim_curve_beyond(result: dict, keep_down_to_nA) -> None:
    """Drop any swept levels more hyperpolarizing than keep_down_to_nA.
    The coarse scan's own "2 consecutive flat" call about a level is
    sometimes superseded once the fine scan re-visits that neighborhood
    (the same path-dependent/hysteretic settling that motivated the fine
    scan re-confirming silencing independently -- see sweep_cell) -- when
    that happens the coarse scan's un-revisited, more-hyperpolarizing
    confirmation point (e.g. a coarse level at -3 nA when the fine scan
    pinned cessation at -2.55 nA) is left dangling in the merged curve.
    There's no value in plotting or recording current levels tested well
    past where the cell was already confirmed quiescent.
    """
    if keep_down_to_nA is None:
        return
    levels = result.get("iapp_levels_nA")
    if levels is None:
        return
    keep = levels >= keep_down_to_nA - 1e-9
    for key in ("iapp_levels_nA", "freq_hz_curve", "settled_flags",
               "burst_pattern_curve", "isi_short_ms_curve", "isi_long_ms_curve"):
        if result.get(key) is not None:
            result[key] = result[key][keep]


def find_cell_silencing_threshold(cell_id: str, params: np.ndarray,
                                  ss_entry: dict, args: argparse.Namespace) -> tuple:
    """Top-level per-cell orchestrator. Never raises -- any failure is
    captured in the returned dict's "status" field.

    Returns (result, fine_traces): fine_traces (Iapp level -> (t_ms, v)) is
    kept out of the pickled result to avoid bloating the cache across 69
    cells with full voltage traces -- it's only used transiently to draw the
    bracketing-trace panel of this cell's diagnostic figure.
    """
    base = {"cell_id": cell_id, "params": params,
            "dt": args.dt, "temp": args.temp, "reftemp": args.reftemp,
            "coarse_step_nA": args.coarse_step_nA, "fine_step_nA": args.fine_step_nA,
            "max_hyperpolarizing_nA": args.max_hyperpolarizing_nA,
            "settle_chunk_s": args.settle_chunk_s, "max_settle_s": args.max_settle_s,
            "settle_rtol": args.settle_rtol, "min_peaks_for_rate": args.min_peaks_for_rate,
            "isi_window_s": args.isi_window_s,
            "min_isis_for_burst_test": args.min_isis_for_burst_test,
            "isi_mode_prominence_frac": args.isi_mode_prominence_frac,
            "min_isi_ratio": args.min_isi_ratio}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}, {}

    y_ss = ss_entry["y_ss"]
    baseline_freq_hz = 1000.0 / ss_entry["period_ms"]
    base["source_steady_state_params"] = ss_entry["params"]
    base["y_ss_used"] = y_ss
    base["baseline_freq_hz"] = baseline_freq_hz

    sweep = sweep_cell(params, ss_entry, args)
    fine_traces = sweep.get("fine_traces", {})
    burst_summary = summarize_burst_transitions(sweep.get("burst_pattern_curve", np.array([])),
                                                sweep.get("burst_transitions", []))

    if sweep["status"] in ("blew_up", "did_not_silence_within_range"):
        return {**base, "status": sweep["status"],
                "iapp_levels_nA": sweep["iapp_levels_nA"],
                "freq_hz_curve": sweep["freq_hz_curve"],
                "settled_flags": sweep["settled_flags"],
                "burst_pattern_curve": sweep.get("burst_pattern_curve"),
                "isi_short_ms_curve": sweep.get("isi_short_ms_curve"),
                "isi_long_ms_curve": sweep.get("isi_long_ms_curve"),
                "burst_transitions": sweep.get("burst_transitions", []),
                **burst_summary,
                "error": sweep.get("error")}, fine_traces

    result = find_silencing_bracket(sweep)
    result.pop("bracket_hi", None)
    result.pop("bracket_lo", None)
    result.pop("fine_traces", None)
    _trim_curve_beyond(result, result.get("first_silent_level_nA"))
    result.update(burst_summary)
    return {**base, **result}, fine_traces


def load_output_cache(cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> None:
    # Write-then-rename -- see run_held_injected_grid.py's identical helper
    # for why (called after every cell now, not just once at the end).
    # os.replace can transiently fail on Windows with PermissionError
    # (WinError 5) if something else briefly has cache_path open -- e.g. a
    # concurrent read of the cache while checking progress on a still-
    # running sweep (confirmed directly: this crashed a real 69-cell run at
    # cell 36). POSIX rename doesn't have this problem (an open file handle
    # doesn't block a rename there), so retrying with a short backoff is
    # enough -- it's a transient OS-level lock, not a real conflict.
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


def _contiguous_label_spans(levels: np.ndarray, labels: np.ndarray, target_label: str) -> list:
    """Contiguous runs of target_label along the (sorted) levels curve, as a
    list of (lo_nA, hi_nA) spans -- there can be more than one, e.g. a cell
    that bursts, returns to tonic, then bursts again before silencing.
    """
    spans = []
    start = None
    for i, label in enumerate(labels):
        if label == target_label and start is None:
            start = i
        elif label != target_label and start is not None:
            spans.append((float(min(levels[start], levels[i - 1])), float(max(levels[start], levels[i - 1]))))
            start = None
    if start is not None:
        spans.append((float(min(levels[start], levels[-1])), float(max(levels[start], levels[-1]))))
    return spans


def plot_cell_silencing(result: dict, fine_traces: dict, outdir: Path, command: str,
                        fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    cell_id = result["cell_id"]
    levels = result.get("iapp_levels_nA")
    freqs = result.get("freq_hz_curve")
    if levels is None or len(levels) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Bursting-phase indicator -- a thin strip along the very top of the
    # axes (not a full-height span) so it reads as its own distinct signal
    # rather than blending into the frequency curve/markers below; there
    # can be more than one span if the cell goes tonic -> bursting -> tonic
    # before eventually silencing.
    patterns = result.get("burst_pattern_curve")
    if patterns is not None and len(patterns) == len(levels):
        bursting_spans = _contiguous_label_spans(levels, patterns, "bursting")
        for j, (lo, hi) in enumerate(bursting_spans):
            axes[0].axvspan(lo, hi, ymin=0.94, ymax=1.0, color="mediumpurple", alpha=0.9,
                            zorder=5, label="bursting phase" if j == 0 else None)

    settled = result.get("settled_flags")
    colors = np.where(settled, "steelblue", "firebrick") if settled is not None else "steelblue"
    axes[0].scatter(levels, freqs, c=colors, s=18, zorder=3)
    axes[0].plot(levels, freqs, color="steelblue", lw=0.8, alpha=0.5, zorder=2)
    baseline = result.get("baseline_freq_hz")
    if baseline is not None:
        axes[0].axhline(baseline, color="gray", ls=":", lw=1, label="baseline (Iapp=0)")
    if result.get("silencing_threshold_bracket_nA") is not None:
        lo, hi = result["silencing_threshold_bracket_nA"]
        mid = 0.5 * (lo + hi)
        axes[0].axvline(mid, color="darkorange", ls="--", lw=1.5,
                        label=f"silencing threshold ≈ {mid:.3f} nA [{lo:.2f}, {hi:.2f}]")
    axes[0].set_xlabel("Iapp (nA)")
    axes[0].set_ylabel("firing rate (Hz)")
    title = f"{cell_id} — status: {result['status']}"
    if result.get("any_bursting_detected"):
        title += ", bursts before stopping"
    axes[0].set_title(title, fontsize=9)
    axes[0].legend(loc="best", fontsize=7)
    axes[0].invert_xaxis()

    axes[1].set_title("bracketing traces (settled tail)", fontsize=9)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("V (mV)")
    hi_level = result.get("last_firing_level_nA")
    lo_level = result.get("first_silent_level_nA")
    pattern_by_level = dict(zip(levels, patterns)) if patterns is not None else {}
    plotted_any = False
    if hi_level is not None and hi_level in fine_traces:
        t_hi, v_hi = fine_traces[hi_level]
        tag = " [bursting]" if pattern_by_level.get(hi_level) == "bursting" else ""
        axes[1].plot(t_hi, v_hi, color="steelblue", lw=0.8,
                    label=f"last firing ({hi_level:.3g} nA){tag}")
        plotted_any = True
    if lo_level is not None and lo_level in fine_traces:
        t_lo, v_lo = fine_traces[lo_level]
        axes[1].plot(t_lo, v_lo, color="firebrick", lw=0.8,
                    label=f"first silent ({lo_level:.3g} nA)")
        plotted_any = True
    if plotted_any:
        axes[1].legend(loc="best", fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "no bracketing trace available\n(e.g. cell silenced on the "
                              "very first coarse step)",
                    ha="center", va="center", fontsize=7, transform=axes[1].transAxes, color="dimgray")

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(0.5, 0.01, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_silencing.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep hyperpolarizing Iapp per cell to find the silencing "
                    "(cessation) threshold and any tonic<->bursting transitions along "
                    "the way. Depolarizing current and depolarization block are out of "
                    "scope for this script.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR,
                        help="Directory containing 40-value cell parameter files.")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) to sweep. Default: every cell found "
                             "in --params-dir.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH,
                        help="Path to the Iapp=0 steady-state cache (input, read-only; "
                             "see generate_steady_state.py).")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH,
                        help="Path to this script's own output pickle cache.")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR,
                        help="Directory for diagnostic frequency-vs-Iapp figures.")
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT,
                        choices=["png", "svg", "pdf"])
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip diagnostic figure generation.")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS,
                        help="Integration timestep in ms, used for every sweep level.")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel workers across cells (-1 uses all cores).")

    parser.add_argument("--coarse-step-nA", type=float, default=0.5,
                        help="Coarse hyperpolarizing step size in nA.")
    parser.add_argument("--fine-step-nA", type=float, default=0.25,
                        help="Fine step size (within a bracket) in nA. 0.25 (not a finer value) is "
                             "deliberate: the resulting threshold precision is still well within "
                             "run_held_injected_grid.py's --cell-floor-margin-nA (1.0 nA default) "
                             "margin past it, so finer precision here isn't load-bearing downstream "
                             "-- confirmed directly this cuts point count ~3.6x on the densest cells "
                             "(61->17 for one real cell) with no loss of downstream precision. Also a "
                             "clean divisor of run_held_injected_grid.py's own default grid step "
                             "(0.5 nA), so Step 1's sampled Iapp levels align with Step 2's grid nodes.")
    parser.add_argument("--max-hyperpolarizing-nA", type=float, default=-25.0,
                        help="Hard stop for the coarse scan; beyond this a cell is "
                             "flagged did_not_silence_within_range rather than swept "
                             "indefinitely. Some cells in this parameter set are "
                             "confirmed to need hyperpolarization approaching -20 nA "
                             "before silencing, so this leaves margin past that.")

    parser.add_argument("--settle-chunk-s", type=float, default=2.0,
                        help="Integration chunk length (s) used while waiting for "
                             "firing rate to stabilize at each Iapp level.")
    parser.add_argument("--max-settle-s", type=float, default=20.0,
                        help="Max total integration time (s) per level before giving "
                             "up and recording that level as unsettled.")
    parser.add_argument("--settle-rtol", type=float, default=0.05,
                        help="Relative frequency change between consecutive chunks "
                             "below which a level is considered settled.")
    parser.add_argument("--min-peaks-for-rate", type=int, default=2,
                        help="Minimum peaks in a chunk to report a nonzero firing "
                             "rate (lower than generate_steady_state's 4-peak period "
                             "estimate, since near-threshold chunks fire sparsely).")

    parser.add_argument("--isi-window-s", type=float, default=10.0,
                        help="Dedicated extra integration window (s), run at every "
                             "fine-resolution level (silencing fine-scan and any "
                             "burst-transition refinement), purely to collect enough "
                             "ISIs for the log-ISI bimodality test -- burst period can "
                             "be much longer than --settle-chunk-s near a transition.")
    parser.add_argument("--min-isis-for-burst-test", type=int, default=6,
                        help="Minimum number of inter-spike intervals required to "
                             "attempt the tonic/bursting bimodality test; below this "
                             "a level is classified 'insufficient_data' rather than "
                             "guessed.")
    parser.add_argument("--isi-mode-prominence-frac", type=float, default=0.05,
                        help="Minimum peak prominence (as a fraction of the tallest "
                             "peak) required for a mode in the log-ISI density to "
                             "count as a distinct candidate cluster.")
    parser.add_argument("--min-isi-ratio", type=float, default=1.5,
                        help="Minimum isi_long_ms / isi_short_ms ratio required to "
                             "accept a bimodal KDE split as genuine bursting rather "
                             "than quantization/sampling noise splitting an "
                             "essentially unimodal (tonic) ISI cluster into two "
                             "adjacent peaks. Intra- vs inter-burst gaps in this "
                             "model differ by 10x or more, so this has a lot of "
                             "headroom below genuine bursting cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    ss_cache_path = Path(args.steady_state_cache)
    output_cache_path = Path(args.output_cache)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_cells = load_all_cells(params_dir)
    if args.cells:
        unknown = sorted(set(args.cells) - set(all_cells))
        if unknown:
            raise SystemExit(f"Unknown cell id(s): {unknown}\n"
                              f"Available cells in {params_dir}: {sorted(all_cells)}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Sweeping silencing threshold for {len(cells)} of {len(all_cells)} cell(s) "
          f"from {params_dir}")

    output_cache = load_output_cache(output_cache_path)

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        result, fine_traces = find_cell_silencing_threshold(cell_id, params, ss_entry, args)
        return cell_id, result, fine_traces

    # Save+plot after EVERY cell -- see run_held_injected_grid.py's main()
    # for the same fix and why (streams results via generator_unordered
    # instead of blocking on the whole batch, write-then-rename keeps a
    # mid-run read of the cache safe).
    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items()
        )

    status_counts: dict = {}
    bursting_cells = []
    n_done = 0
    for cell_id, result, fine_traces in result_iter:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if result.get("any_bursting_detected"):
            bursting_cells.append(cell_id)
        if not args.no_plot:
            plot_cell_silencing(result, fine_traces, figures_dir, command, fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cells)}] {cell_id}: {result['status']}")

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\n{len(bursting_cells)} cell(s) show a bursting phase before silencing.")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    # results (the generator) is exhausted -- derive from output_cache,
    # which the incremental-save loop above already populated.
    flagged = [cid for cid in cells if output_cache[cid]["status"] not in ("ok",)]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("Check their figures, and for 'no_cached_steady_state' run "
              "generate_steady_state.py first; for 'unsettled' or 'blew_up', "
              "consider a smaller --dt or --settle-chunk-s for just that cell.")


if __name__ == "__main__":
    main()
