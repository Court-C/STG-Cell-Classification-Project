"""Phase response curve (PRC) protocol (Step 5): for each cell, characterize
how a brief current pulse delivered at different phases of the cell's own
ongoing spontaneous oscillation shifts the timing of the next spike (or, for
a cell that bursts at Iapp=0, the next burst onset).

Starting rhythm: this protocol needs a stable ongoing oscillation to perturb.
Two candidates were available -- searching cell_held_injected_grid.pkl for a
held==injected point classified tonic/bursting, or simply using the cell's
own Iapp=0 limit cycle from cell_steady_states.pkl. The Iapp=0 baseline is
used here: every one of the 69 cells fires spontaneously at Iapp=0 by
construction (see find_silencing_threshold.py's module docstring), so it
needs no cell-specific search, is already warm-startable from a cached
settled state (y_ss), and gives every cell the exact same, simplest starting
point for a PRC. It also avoids touching cell_held_injected_grid.pkl and
cell_silencing_thresholds.pkl entirely, which matters while a concurrent
pipeline job may be writing to both.

Phase convention (fixed for this whole script -- do not reinterpret
elsewhere): phase phi in [0, 1) is the fraction of the cell's own natural
inter-event period elapsed, since a reference event, at which a stimulus
pulse's ONSET is delivered. An "event" is a spike peak for a tonic baseline,
or the first spike of a burst (burst onset) for a bursting baseline -- see
characterize_baseline. phi=0 means the pulse coincides with the reference
event itself; phi -> 1 means the pulse is delivered just before the next
unperturbed event would have occurred.

Sign convention for phase shift (fixed for this whole script): for a trial
at phase phi,
    shift_ms = t_next_event_perturbed - t_next_event_control
where both times are measured from the SAME origin (the reference event).
shift_ms < 0 means the perturbed next event arrives EARLIER than it would
have unperturbed -> phase ADVANCE. shift_ms > 0 means it arrives LATER ->
phase DELAY. phase_shift_norm = shift_ms / natural_period_ms is the
dimensionless version used for the PRC curve and its classification.

Pulse polarity: hyperpolarizing (negative nA) by default, not depolarizing.
This matches two things at once -- this project's established inhibitory-
only scope for every current-injection protocol so far (see
find_silencing_threshold.py / run_held_injected_grid.py module docstrings),
and the physiologically relevant question for a pyloric neuron, which
receives inhibitory (IPSP-like) synaptic input from its network neighbors,
not depolarizing pulses. --pulse-amplitude-nA is a plain CLI float, so a
positive value works too if a depolarizing PRC is wanted later.

PRC shape classification: monophasic (Type I, phase shift keeps one sign
essentially everywhere) vs. biphasic (Type II, phase shift changes sign
somewhere across the cycle) is computed, not eyeballed -- see
classify_prc_type. It counts sign changes across the (circularly ordered,
since phase is periodic) sequence of valid phase-shift measurements, using a
small deadzone around zero to avoid calling numerical noise a sign flip.

F-I cross-check (Step 2): cell_grid_features.pkl (read-only) has a linear
F-I slope/R^2 (fi_slope_hz_per_nA / fi_slope_r2, from extract_grid_features.
py's compute_fi_slope) but NO continuous-vs-discontinuous onset
classification -- nothing in any current cache distinguishes Class I from
Class II excitability by F-I shape. That cross-check is therefore reported
here only as the raw slope/R^2 value plus an explicit note that no onset-
shape label exists yet, rather than inventing a proxy for one.
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)
from find_silencing_threshold import (constant_iapp_func, classify_burst_pattern,
                                      to_stored_pattern, PROMINENCE_FRACTION, FLATLINE_MV)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_prc.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "prc"
DEFAULT_FIGURE_FORMAT = "svg"
# Read-only cross-check input -- see module docstring. Never written by this
# script.
DEFAULT_GRID_FEATURES_CACHE_PATH = ROOT_DIR / "cell_grid_features.pkl"
# temp == reftemp -- same reasoning as find_silencing_threshold.py /
# run_held_injected_grid.py: at dtemp=0 every q10 factor collapses to 1, so
# every cell runs at its literal reference-temperature parameters, matching
# Steps 1-2's already-corrected convention.
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

V_INDEX = 0

# A normalized phase shift smaller than this is treated as "no measurable
# shift" (sign 0) for classify_prc_type's purposes, rather than a genuine
# advance or delay -- keeps floating-point/integration noise around a true
# zero-crossing from being read as a spurious extra sign flip.
PRC_SIGN_DEADZONE = 1e-3


# ---------------------------------------------------------------------------
# Spike / event detection
# ---------------------------------------------------------------------------

def detect_spike_times_ms(v: np.ndarray, t_ms: np.ndarray,
                          prominence_fraction: float = PROMINENCE_FRACTION) -> np.ndarray:
    """Spike-peak times (ms), using the same prominence-relative-to-range
    convention as the rest of this project (see find_silencing_threshold.
    py's compute_isis_ms). Returns the peak times themselves, not just
    interval statistics -- this protocol needs actual event timestamps to
    measure phase shifts against.
    """
    if v.max() - v.min() < FLATLINE_MV:
        return np.array([])
    peaks, _ = find_peaks(v, prominence=(v.max() - v.min()) * prominence_fraction)
    return t_ms[peaks]


def characterize_baseline(params: np.ndarray, y_ss: np.ndarray, dt: float, temp: float, reftemp: float,
                          initial_period_ms: float, min_isis_for_burst_test: int,
                          isi_mode_prominence_frac: float, min_isi_ratio: float,
                          n_cycles: float, max_baseline_s: float, extend_factor: float) -> dict:
    """Integrate the cell's own Iapp=0 limit cycle to (a) measure its natural
    period directly, with the SAME spike-detection convention this script
    uses for phase-shift timing elsewhere (not just trust the cached
    estimate from generate_steady_state.py, which used its own independent
    period estimator), (b) classify tonic vs. bursting, and (c) for a
    bursting baseline, derive the ISI threshold that separates "still the
    same burst" from "start of the next burst" -- reused for every
    perturbed-trial event detection later (find_next_event_ms), so the
    event definition is identical in the reference recording and every
    trial measured against it.

    Extends the recording window (restart from y_ss at a longer duration,
    not a continued chunk -- simpler, and only the minority of slow/sparse
    cells ever need the retry) if too few spikes come back to run the
    bimodality test with any confidence, up to max_baseline_s -- same
    escape-hatch pattern find_silencing_threshold.py's isi_window_s logic
    uses.
    """
    window_s = max(n_cycles * initial_period_ms / 1000.0, 0.5)
    Iapp_func = constant_iapp_func(0.0)
    while True:
        try:
            t_ms, states = simulate(params, window_s, temp, dt=dt, reftemp=reftemp,
                                    cis=y_ss, Iapp_func=Iapp_func)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            return {"status": "blew_up", "error": str(exc)}
        if not np.all(np.isfinite(states)):
            return {"status": "blew_up", "error": "baseline recording: non-finite trajectory"}

        v = states[:, V_INDEX]
        spike_times = detect_spike_times_ms(v, t_ms)
        if len(spike_times) >= 2 * min_isis_for_burst_test or window_s >= max_baseline_s:
            break
        window_s = min(window_s * extend_factor, max_baseline_s)

    if len(spike_times) < 3:
        # Should not happen for an Iapp=0 baseline per this project's own
        # established convention (every cell fires spontaneously at Iapp=0)
        # -- reported honestly rather than assumed away, in case a specific
        # cell's cached y_ss turns out not to hold up under this script's
        # own re-integration.
        return {"status": "insufficient_baseline_spikes", "n_spikes": len(spike_times)}

    isis_ms = np.diff(spike_times)
    burst_info = classify_burst_pattern(isis_ms, min_isis_for_burst_test,
                                        isi_mode_prominence_frac, min_isi_ratio,
                                        n_peaks=len(spike_times))
    pattern = to_stored_pattern(burst_info["pattern"])
    if pattern == "silent":
        return {"status": "baseline_silent_unexpected", "n_spikes": len(spike_times)}

    if pattern == "bursting":
        # Geometric mean of the fitted short/long ISI modes -- a direct,
        # computed split point in log-ISI space (the natural midpoint
        # between two roughly log-normal clusters), not a fabricated
        # threshold. Reused verbatim by find_next_event_ms for every
        # perturbed trial.
        burst_isi_threshold_ms = float(np.sqrt(burst_info["isi_short_ms"] * burst_info["isi_long_ms"]))
        event_times = [spike_times[0]]
        for i in range(1, len(spike_times)):
            if spike_times[i] - spike_times[i - 1] > burst_isi_threshold_ms:
                event_times.append(spike_times[i])
        event_times = np.array(event_times)
    else:
        burst_isi_threshold_ms = None
        event_times = spike_times

    if len(event_times) < 3:
        return {"status": "insufficient_baseline_events", "n_events": len(event_times)}

    natural_period_ms = float(np.median(np.diff(event_times)))
    return {"status": "ok", "pattern": pattern, "natural_period_ms": natural_period_ms,
            "event_times_ms": event_times, "burst_isi_threshold_ms": burst_isi_threshold_ms,
            "t_ms": t_ms, "states": states, "window_s_used": window_s}


def pick_reference_anchor(baseline: dict, min_index_frac: float = 0.3) -> tuple:
    """Picks one baseline event as phase-0 (t_ref) and returns the model
    state AT that event (nearest sample), plus the unperturbed "next event"
    time relative to it -- this becomes the single shared control reference
    every phase's perturbed trial is compared against.

    Skips the earliest ~min_index_frac of events before choosing t_ref
    (rather than always using event 0): y_ss is already a settled limit-
    cycle state, so there's no burn-in transient to avoid here, but this
    still keeps the anchor away from the very first recorded event in case
    of any edge sampling artifact at t=0, at negligible cost since the
    baseline recording covers many cycles.
    """
    event_times = baseline["event_times_ms"]
    n = len(event_times)
    k = max(1, int(n * min_index_frac))
    k = min(k, n - 2)
    t_ref = event_times[k]
    idx = int(np.argmin(np.abs(baseline["t_ms"] - t_ref)))
    y_ref = baseline["states"][idx]
    control_next_event_ms = float(event_times[k + 1] - t_ref)
    return y_ref, float(t_ref), control_next_event_ms


def find_next_event_ms(v: np.ndarray, t_ms: np.ndarray, pattern: str,
                       burst_isi_threshold_ms) -> "float | None":
    """First event at local time > 0 in a trial trace that starts exactly at
    the reference-anchor state (t=0 == the reference event itself). scipy's
    find_peaks can never mark a boundary sample as a peak, so the anchor
    sample at t=0 can't spuriously register as "the next event" here --
    every detected peak is genuinely later than the reference.

    For a bursting baseline, skips spikes that are still part of the SAME
    burst as the reference event (ISI from the previous marker <= the
    burst_isi_threshold_ms computed once in characterize_baseline) and
    returns the first spike that starts the NEXT burst instead.
    """
    spike_times = detect_spike_times_ms(v, t_ms)
    spike_times = spike_times[spike_times > 0]
    if len(spike_times) == 0:
        return None
    if pattern == "tonic":
        return float(spike_times[0])

    prev = 0.0
    for st in spike_times:
        if st - prev > burst_isi_threshold_ms:
            return float(st)
        prev = st
    return None


# ---------------------------------------------------------------------------
# Per-phase trial
# ---------------------------------------------------------------------------

def make_pulse_iapp_func(pulse_start_ms: float, pulse_end_ms: float, amplitude_nA: float):
    def f(t):
        return amplitude_nA if pulse_start_ms <= t < pulse_end_ms else 0.0
    return f


def run_phase_trial(params: np.ndarray, y_ref: np.ndarray, phase: float, natural_period_ms: float,
                    pulse_duration_ms: float, pulse_amplitude_nA: float, dt: float, temp: float, reftemp: float,
                    horizon_ms: float, pattern: str, burst_isi_threshold_ms, return_trace: bool = False) -> dict:
    pulse_start_ms = phase * natural_period_ms
    pulse_end_ms = pulse_start_ms + pulse_duration_ms
    Iapp_func = make_pulse_iapp_func(pulse_start_ms, pulse_end_ms, pulse_amplitude_nA)
    try:
        t_ms, states = simulate(params, horizon_ms / 1000.0, temp, dt=dt, reftemp=reftemp,
                                cis=y_ref, Iapp_func=Iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {"blew_up": True, "error": str(exc)}
    if not np.all(np.isfinite(states)):
        return {"blew_up": True, "error": "phase trial: non-finite trajectory"}

    v = states[:, V_INDEX]
    next_event_ms = find_next_event_ms(v, t_ms, pattern, burst_isi_threshold_ms)
    result = {"blew_up": False, "error": None, "pulse_start_ms": pulse_start_ms,
             "pulse_end_ms": pulse_end_ms, "next_event_ms": next_event_ms}
    if return_trace:
        result["t_ms"] = t_ms
        result["v"] = v
    return result


def classify_prc_type(phases: np.ndarray, shift_norm: np.ndarray, event_found: np.ndarray) -> dict:
    """Monophasic (Type I) vs. biphasic (Type II), computed from actual
    sign changes rather than eyeballed. Phase is circular (phi=1 wraps back
    to phi=0), so the sign sequence is checked for a wraparound flip too,
    not just adjacent-pair flips within [0, 1).
    """
    valid = event_found & ~np.isnan(shift_norm)
    if valid.sum() < 2:
        return {"type": "undetermined", "n_zero_crossings": None}

    order = np.argsort(phases[valid])
    vals = shift_norm[valid][order]
    signs = np.sign(vals)
    signs[np.abs(vals) < PRC_SIGN_DEADZONE] = 0
    nonzero_signs = signs[signs != 0]

    if len(nonzero_signs) == 0:
        # No phase produced a shift big enough to clear the deadzone --
        # a flat, effectively unresponsive PRC. Reported as monophasic
        # (trivially single-signed: no sign at all), not biphasic.
        return {"type": "monophasic", "n_zero_crossings": 0}

    crossings = 0
    prev = None
    for s in nonzero_signs:
        if prev is not None and s != prev:
            crossings += 1
        prev = s
    if nonzero_signs[0] != nonzero_signs[-1]:
        crossings += 1  # wraparound: last tested phase back to the first
    return {"type": "monophasic" if crossings == 0 else "biphasic", "n_zero_crossings": int(crossings)}


# ---------------------------------------------------------------------------
# Top-level per-cell orchestrator
# ---------------------------------------------------------------------------

def run_cell_prc(cell_id: str, params: np.ndarray, ss_entry, fi_features_entry, args) -> tuple:
    """Never raises -- every failure is captured in the returned dict's
    "status" field, matching find_silencing_threshold.py / run_held_
    injected_grid.py's convention.

    Returns (result, example_trace): example_trace (a dict of raw arrays
    for one representative phase, plus the matching unperturbed control
    trace) is kept OUT of the pickled result -- same reasoning as run_held_
    injected_grid.py's fine_traces -- it's only used transiently to draw
    this cell's diagnostic figure, not to bloat the cache with full voltage
    traces for every cell.
    """
    cell_floor_nA = fi_features_entry.get("cell_floor_nA") if fi_features_entry else None
    if args.pulse_amplitude_frac_of_floor is not None and cell_floor_nA is not None:
        pulse_amplitude_nA = args.pulse_amplitude_frac_of_floor * cell_floor_nA
        pulse_amplitude_source = "frac_of_floor"
    else:
        pulse_amplitude_nA = args.pulse_amplitude_nA
        pulse_amplitude_source = "fixed_nA"

    base = {"cell_id": cell_id, "params": params, "dt": args.dt, "temp": args.temp, "reftemp": args.reftemp,
            "pulse_amplitude_nA": pulse_amplitude_nA, "pulse_amplitude_source": pulse_amplitude_source,
            "cell_floor_nA": cell_floor_nA, "n_phases": args.n_phases,
            "pulse_duration_frac_of_period": args.pulse_duration_frac_of_period,
            "horizon_cycles": args.horizon_cycles}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}, None

    y_ss = ss_entry["y_ss"]
    initial_period_ms = ss_entry["period_ms"]

    baseline = characterize_baseline(params, y_ss, args.dt, args.temp, args.reftemp, initial_period_ms,
                                     args.min_isis_for_burst_test, args.isi_mode_prominence_frac,
                                     args.min_isi_ratio, args.n_baseline_cycles, args.max_baseline_s,
                                     args.baseline_extend_factor)
    if baseline["status"] != "ok":
        return {**base, "status": baseline["status"], "error": baseline.get("error")}, None

    y_ref, t_ref, control_next_event_ms = pick_reference_anchor(baseline)
    natural_period_ms = baseline["natural_period_ms"]
    pattern = baseline["pattern"]
    burst_isi_threshold_ms = baseline["burst_isi_threshold_ms"]

    pulse_duration_ms = args.pulse_duration_ms_fixed or (args.pulse_duration_frac_of_period * natural_period_ms)
    horizon_ms = args.horizon_cycles * natural_period_ms

    phases = np.linspace(0.0, 1.0, args.n_phases, endpoint=False)
    shift_ms = np.full(args.n_phases, np.nan)
    shift_norm = np.full(args.n_phases, np.nan)
    event_found = np.zeros(args.n_phases, dtype=bool)
    trial_blew_up = np.zeros(args.n_phases, dtype=bool)

    for i, phase in enumerate(phases):
        r = run_phase_trial(params, y_ref, phase, natural_period_ms, pulse_duration_ms,
                            pulse_amplitude_nA, args.dt, args.temp, args.reftemp,
                            horizon_ms, pattern, burst_isi_threshold_ms)
        if r["blew_up"]:
            trial_blew_up[i] = True
            continue
        if r["next_event_ms"] is not None:
            event_found[i] = True
            shift_ms[i] = r["next_event_ms"] - control_next_event_ms
            shift_norm[i] = shift_ms[i] / natural_period_ms

    prc_class = classify_prc_type(phases, shift_norm, event_found)

    valid = event_found & ~np.isnan(shift_norm)
    example_trace = None
    if valid.any():
        idx_valid = np.flatnonzero(valid)
        max_advance_norm = float(np.min(shift_norm[valid]))
        phase_of_max_advance = float(phases[idx_valid[np.argmin(shift_norm[valid])]])
        max_delay_norm = float(np.max(shift_norm[valid]))
        phase_of_max_delay = float(phases[idx_valid[np.argmax(shift_norm[valid])]])
        idx_sens = idx_valid[np.argmax(np.abs(shift_norm[valid]))]
        phase_of_max_sensitivity = float(phases[idx_sens])

        if not args.no_plot:
            # Rerun just the most-sensitive phase (cheap: one extra
            # simulate() call) plus an unperturbed control over the same
            # horizon, purely for the diagnostic figure's trace overlay --
            # the shift itself was already measured above without needing
            # full traces for every phase.
            example = run_phase_trial(params, y_ref, phase_of_max_sensitivity, natural_period_ms,
                                      pulse_duration_ms, pulse_amplitude_nA, args.dt, args.temp,
                                      args.reftemp, horizon_ms, pattern, burst_isi_threshold_ms,
                                      return_trace=True)
            control = run_phase_trial(params, y_ref, phase_of_max_sensitivity, natural_period_ms,
                                      pulse_duration_ms, 0.0, args.dt, args.temp, args.reftemp,
                                      horizon_ms, pattern, burst_isi_threshold_ms, return_trace=True)
            if not example["blew_up"] and not control["blew_up"]:
                example_trace = {"phase": phase_of_max_sensitivity, "perturbed": example, "control": control}
    else:
        max_advance_norm = phase_of_max_advance = None
        max_delay_norm = phase_of_max_delay = None
        phase_of_max_sensitivity = None

    fi_slope_hz_per_nA = fi_features_entry.get("fi_slope_hz_per_nA") if fi_features_entry else None
    fi_slope_r2 = fi_features_entry.get("fi_slope_r2") if fi_features_entry else None
    fi_cross_check_note = ("cell_grid_features.pkl provides only a linear F-I slope/R^2 "
                           "(compute_fi_slope in extract_grid_features.py), not a continuous-vs-"
                           "discontinuous onset (Class I/II excitability) classification -- no such "
                           "label exists anywhere in the current caches, so it is not fabricated here.")

    result = {
        **base, "status": "ok",
        "baseline_pattern": pattern, "natural_period_ms": natural_period_ms,
        "baseline_freq_hz": 1000.0 / natural_period_ms,
        "pulse_duration_ms": pulse_duration_ms,
        "burst_isi_threshold_ms": burst_isi_threshold_ms,
        "reference_event_time_ms": t_ref, "control_next_event_ms": control_next_event_ms,
        "phases": phases, "phase_shift_ms": shift_ms, "phase_shift_norm": shift_norm,
        "event_found": event_found, "trial_blew_up": trial_blew_up,
        "n_events_missing": int(np.sum(~event_found & ~trial_blew_up)),
        "n_trials_blew_up": int(np.sum(trial_blew_up)),
        "prc_type": prc_class["type"], "n_zero_crossings": prc_class["n_zero_crossings"],
        "max_advance_norm": max_advance_norm, "phase_of_max_advance": phase_of_max_advance,
        "max_delay_norm": max_delay_norm, "phase_of_max_delay": phase_of_max_delay,
        "phase_of_max_sensitivity": phase_of_max_sensitivity,
        "fi_slope_hz_per_nA": fi_slope_hz_per_nA, "fi_slope_r2": fi_slope_r2,
        "fi_cross_check_note": fi_cross_check_note,
        "error": None,
    }
    return result, example_trace


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_output_cache(cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> None:
    # Write-then-rename + PermissionError retry -- copied verbatim from
    # run_held_injected_grid.py's save_output_cache: os.replace can
    # transiently fail on Windows (WinError 5) if something else briefly
    # has cache_path open (confirmed there as a real crash mid-sweep), and
    # retrying with a short backoff is enough since it's a transient OS-
    # level lock, not a real conflict.
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


def load_grid_features_cache(cache_path: Path) -> dict:
    """Read-only cross-check input (see module docstring) -- degrades to
    "no cross-check available" rather than failing the whole run if the
    file is missing or (in principle) transiently unreadable, since this
    script never depends on it for the PRC measurement itself.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_cell_prc(result: dict, example_trace, outdir: Path, command: str,
                  fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    if result["status"] != "ok":
        return
    cell_id = result["cell_id"]
    phases = result["phases"]
    shift_norm = result["phase_shift_norm"]
    event_found = result["event_found"]
    valid = event_found & ~np.isnan(shift_norm)

    ncols = 2 if example_trace is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    if valid.any():
        order = np.argsort(phases[valid])
        ax.plot(phases[valid][order], shift_norm[valid][order], color="steelblue", lw=1.2,
               marker="o", ms=4, zorder=3)
    if (~valid & ~result["trial_blew_up"]).any():
        miss = ~valid & ~result["trial_blew_up"]
        ax.scatter(phases[miss], np.zeros(int(miss.sum())), marker="x", color="firebrick", s=30,
                  zorder=4, label="no next event detected")
    if result["max_advance_norm"] is not None:
        ax.axvline(result["phase_of_max_advance"], color="mediumseagreen", ls="--", lw=1,
                  label=f"max advance {result['max_advance_norm']:.3f} @ φ={result['phase_of_max_advance']:.2f}")
        ax.axvline(result["phase_of_max_delay"], color="darkorange", ls="--", lw=1,
                  label=f"max delay {result['max_delay_norm']:.3f} @ φ={result['phase_of_max_delay']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("stimulus phase φ (fraction of natural period)")
    ax.set_ylabel("normalized phase shift  Δt / T\n(negative = advance, positive = delay)")
    ax.set_title(f"{cell_id} — PRC ({result['baseline_pattern']} baseline), type: {result['prc_type']} "
                f"({result['n_zero_crossings']} zero-crossing(s))", fontsize=9)
    ax.legend(loc="best", fontsize=6.5)

    if example_trace is not None:
        ax2 = axes[1]
        ctrl, pert = example_trace["control"], example_trace["perturbed"]
        ax2.plot(ctrl["t_ms"], ctrl["v"], color="gray", lw=0.8, label="unperturbed")
        ax2.plot(pert["t_ms"], pert["v"], color="steelblue", lw=0.8,
                label=f"perturbed (φ={example_trace['phase']:.2f})")
        ax2.axvspan(pert["pulse_start_ms"], pert["pulse_end_ms"], color="firebrick", alpha=0.3,
                   label="pulse")

        # natural_period_ms trace annotation: t=0 IS the reference event
        # itself (phase phi=0 convention, see module docstring), and
        # control_next_event_ms is the very next unperturbed event measured
        # in THIS SAME baseline recording -- bracketing [0, control_next_
        # event_ms] shows one real, concrete period instance directly on
        # the trace, not just the printed natural_period_ms number (the
        # median ISI over the whole baseline recording, of which this is
        # one instance -- close but not necessarily bit-identical, which
        # the label states explicitly rather than implying exact equality).
        v_span = ctrl["v"].max() - ctrl["v"].min()
        y_bracket = ctrl["v"].max() + 0.08 * v_span
        ax2.annotate("", xy=(result["control_next_event_ms"], y_bracket), xytext=(0, y_bracket),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
        ax2.text(result["control_next_event_ms"] / 2.0, y_bracket, f"T ≈ {result['natural_period_ms']:.1f} ms",
                 ha="center", va="bottom", fontsize=7)
        # autoscale only accounts for plotted line/patch data, not
        # annotate()/text() extents -- without this, a short-period cell
        # (small y-range) can place the bracket+label right at or above the
        # axes' autoscaled top edge, overlapping the panel title above it.
        ylo, yhi = ax2.get_ylim()
        ax2.set_ylim(ylo, max(yhi, y_bracket + 0.15 * v_span))

        ax2.set_xlabel("time since reference event (ms)")
        ax2.set_ylabel("V (mV)")
        ax2.set_title("most phase-sensitive trial: perturbed vs. unperturbed", fontsize=9)
        ax2.legend(loc="best", fontsize=6.5)

    if result.get("pulse_amplitude_source") == "frac_of_floor" and result.get("cell_floor_nA"):
        pulse_source_note = f"{result['pulse_amplitude_nA'] / result['cell_floor_nA'] * 100:.0f}% of cell_floor_nA={result['cell_floor_nA']:.2f} nA"
    else:
        pulse_source_note = "fixed nA"
    fi_slope_str = f"{result['fi_slope_hz_per_nA']:.3f}" if result["fi_slope_hz_per_nA"] is not None else "None"
    fi_r2_str = f"{result['fi_slope_r2']:.3f}" if result["fi_slope_r2"] is not None else "None"
    # explicit "\n" line breaks, not wrap=True -- letting matplotlib's own
    # wrap algorithm choose break points on two fig.text calls placed close
    # together was landing lines on top of each other for cells whose
    # numbers made either line longer than usual (e.g. large cell_floor_nA).
    info_text = (
        f"pulse: {result['pulse_amplitude_nA']:.2f} nA ({pulse_source_note}) x "
        f"{result['pulse_duration_ms']:.2f} ms "
        f"({result['pulse_duration_frac_of_period'] * 100:.1f}% of T={result['natural_period_ms']:.2f} ms)\n"
        f"F-I cross-check: slope={fi_slope_str}, R²={fi_r2_str} "
        "(no onset-shape class available -- see fi_cross_check_note)"
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.text(0.5, 0.045, info_text, ha="center", va="bottom", fontsize=6.5, style="italic", color="dimgray")
    fig.text(0.5, 0.005, command, ha="center", va="bottom", fontsize=6, family="monospace",
             color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_prc.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase response curve (PRC) protocol: brief current pulses at multiple "
                    "phases of each cell's own Iapp=0 spontaneous oscillation, measuring the "
                    "resulting shift in next-spike/next-burst timing. See module docstring for "
                    "the exact phase and sign conventions used throughout.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) to run. Default: every cell found in --params-dir.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH,
                        help="Path to the Iapp=0 steady-state cache (input, read-only; "
                             "see generate_steady_state.py).")
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH,
                        help="Path to Step 2c's cell_grid_features.pkl (input, read-only; used "
                             "only for the F-I slope cross-check -- see module docstring).")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--jobs", type=int, default=-1)

    parser.add_argument("--pulse-amplitude-nA", type=float, default=-1.0,
                        help="Constant current during the pulse (nA). Negative (hyperpolarizing) "
                             "by default -- see module docstring for why. Used directly unless "
                             "--pulse-amplitude-frac-of-floor is set, in which case it's only the "
                             "fallback for cells with no grid_features_cache entry.")
    parser.add_argument("--pulse-amplitude-frac-of-floor", type=float, default=None,
                        help="If set, overrides --pulse-amplitude-nA PER CELL as this fraction of "
                             "the cell's own cell_floor_nA (from --grid-features-cache, same "
                             "silencing-threshold-anchored quantity run_held_injected_grid.py scales "
                             "its grid with) -- e.g. 0.5 = 50%% of this cell's own silencing "
                             "threshold, negative*positive=negative so the pulse stays hyperpolarizing. "
                             "A fixed absolute nA (the old default) is an arbitrary fraction of each "
                             "cell's dynamic range -- as small as 5%% of floor for the deepest-floor "
                             "cells, over 100%% of floor for the shallowest (see project memory/"
                             "2026-08-11 investigation) -- this makes the pulse size comparable "
                             "across cells instead. Falls back to --pulse-amplitude-nA for any cell "
                             "missing a grid_features_cache entry.")
    parser.add_argument("--pulse-duration-frac-of-period", type=float, default=0.05,
                        help="Pulse duration as a fraction of the cell's own measured natural "
                             "period, so the pulse stays 'brief' relative to each cell's own "
                             "timescale rather than using one fixed duration across very "
                             "different firing rates. Overridden by --pulse-duration-ms-fixed "
                             "if given.")
    parser.add_argument("--pulse-duration-ms-fixed", type=float, default=None,
                        help="If set, use this fixed pulse duration (ms) for every cell instead "
                             "of the period-scaled default.")
    parser.add_argument("--n-phases", type=int, default=16,
                        help="Number of phases sampled across one full cycle [0, 1). 12-20 is a "
                             "reasonable range for a real curve shape; smoke-test runs may need "
                             "fewer (see this script's own module-level notes / run report).")
    parser.add_argument("--horizon-cycles", type=float, default=2.5,
                        help="Per-phase trial length, in multiples of the natural period, over "
                             "which to search for the next event after the pulse. Needs enough "
                             "margin for a strong delaying pulse to push the next event close to "
                             "a full extra cycle later; a phase with no event found within this "
                             "horizon is reported honestly (event_found=False), not extrapolated.")

    parser.add_argument("--n-baseline-cycles", type=float, default=8.0,
                        help="Initial baseline-recording length, in multiples of the cached "
                             "period estimate -- automatically extended (see characterize_baseline) "
                             "if this doesn't yield enough spikes to classify tonic/bursting or "
                             "measure the period confidently.")
    parser.add_argument("--max-baseline-s", type=float, default=10.0,
                        help="Cap on the baseline-recording extension loop.")
    parser.add_argument("--baseline-extend-factor", type=float, default=2.0)

    parser.add_argument("--min-isis-for-burst-test", type=int, default=6)
    parser.add_argument("--isi-mode-prominence-frac", type=float, default=0.05)
    parser.add_argument("--min-isi-ratio", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    ss_cache_path = Path(args.steady_state_cache)
    grid_features_cache_path = Path(args.grid_features_cache)
    output_cache_path = Path(args.output_cache)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_cells = load_all_cells(params_dir)
    if args.cells:
        unknown = sorted(set(args.cells) - set(all_cells))
        if unknown:
            raise SystemExit(f"Unknown cell id(s): {unknown}\nAvailable cells in {params_dir}: {sorted(all_cells)}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Running PRC protocol for {len(cells)} of {len(all_cells)} cell(s) from {params_dir}")

    grid_features_cache = load_grid_features_cache(grid_features_cache_path)
    output_cache = load_output_cache(output_cache_path)

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        fi_features_entry = grid_features_cache.get(cell_id)
        result, example_trace = run_cell_prc(cell_id, params, ss_entry, fi_features_entry, args)
        return cell_id, result, example_trace

    # Save+plot after EVERY cell -- see run_held_injected_grid.py's main()
    # for the same fix and why (streams results via generator_unordered
    # instead of blocking on the whole batch; write-then-rename keeps a
    # mid-run read of the cache safe).
    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items())

    status_counts: dict = {}
    prc_type_counts: dict = {}
    n_done = 0
    for cell_id, result, example_trace in result_iter:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if result["status"] == "ok":
            prc_type_counts[result["prc_type"]] = prc_type_counts.get(result["prc_type"], 0) + 1
        if not args.no_plot:
            plot_cell_prc(result, example_trace, figures_dir, command, fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cells)}] {cell_id}: {result['status']}"
             + (f" ({result['prc_type']})" if result["status"] == "ok" else ""))

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    if prc_type_counts:
        print("\nPRC type summary:")
        for prc_type, count in sorted(prc_type_counts.items()):
            print(f"  {prc_type}: {count}")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    flagged = [cid for cid in cells if output_cache[cid]["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'insufficient_baseline_spikes'/'insufficient_baseline_events': try a larger "
              "--max-baseline-s. 'blew_up': consider a smaller --dt.")


if __name__ == "__main__":
    main()
