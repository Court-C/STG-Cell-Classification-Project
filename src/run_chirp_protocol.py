"""Chirp / impedance protocol (Step 4): a swept-frequency sinusoidal ("ZAP")
current injection, used to estimate each cell's subthreshold membrane
impedance |Z(f)| and phase, and to detect Ih/IA-mediated subthreshold
resonance -- a standard technique in the resonance literature (Hutcheon &
Yarom 2000).

--- Design decision 1: additive, not Step-2-style replacement ---
Step 2's held/injected convention (run_held_injected_grid.py) is deliberately
NOT additive: held sets a pre-test adapted context and is released back to
after a separate, discrete test-window current level. That design exists
because Step 2 studies a state CHANGE -- a step from one regime to another,
and recovery afterward.

The chirp/ZAP protocol studies something different: the LINEAR response of
the system around one fixed, unchanging operating point. Impedance is only a
well-defined concept relative to a single bias point that the system sits at
throughout the measurement -- there is no separate "test" state to step into
and no "recovery" phase to release back to, because the operating point
never changes. So here, "held" is not a pre/post context around a discrete
test window the way it is in Step 2; it IS the DC bias applied continuously,
and the chirp rides on top of it additively for the entire simulated
duration: Iapp(t) = held_nA + amplitude_nA * sin(phase(t)). This matches the
standard ZAP-protocol convention (small-amplitude oscillation around an
operating point) and is the only construction under which "impedance" as a
single-valued function of frequency is even meaningful.

--- Design decision 2: what "held" and "amplitude" are, per cell ---
These are endogenous pacemaker cells -- at Iapp=0 essentially all of them
fire spontaneously, so Iapp=0 is not a subthreshold operating point and a
chirp riding on top of it would just perturb an ongoing limit cycle, not
probe passive/subthreshold impedance. Step 1 (find_silencing_threshold.py)
already found, per cell, a fine-resolution bracket
(first_silent_level_nA, last_firing_level_nA) bracketing the cessation
threshold. This script anchors:
  held_nA = first_silent_level_nA -- the least-hyperpolarized CONFIRMED
    quiescent level from Step 1, i.e. already known subthreshold.
  amplitude_nA = amplitude_fraction * (last_firing_level_nA - first_silent_level_nA)
    -- a fraction (default 0.4) of the bracket's own width, clipped to
    [--min-amplitude-nA, --max-amplitude-nA]. Since amplitude_fraction < 1,
    held + amplitude is always strictly less hyperpolarized than
    first_silent but strictly more hyperpolarized than last_firing -- the
    depolarizing swing of the chirp can never reach a level Step 1 actually
    confirmed firing at, by construction. This uses Step 1's own bracket as
    the subthreshold ceiling reference the task asked for, rather than a
    fixed global amplitude that would be wildly wrong for some cells (the
    bracket width varies enormously across this population).
  If Step 1's status is "unsettled" and last_firing_level_nA isn't
    available/trustworthy, or the bracket has non-positive width, a small
    fixed --fallback-amplitude-nA (default 0.1 nA) is used instead --
    documented via amplitude_anchor_source in the result.

--- Design decision 3: frequency range and sweep law ---
Linearly-swept chirp, 0.2-20 Hz over 20 s by default. Chosen by inspecting
_derivatives_core's own time constants in singlecell_model_v1.py at
subthreshold voltages (roughly -70 to -50 mV):
  - tauHM (Ih, the classic resonance-generating current) spans roughly
    60-300+ ms in this range -- corresponds to a resonance candidate well
    under 10 Hz, comfortably covered by the low end of this range.
  - tauCaTM (T-type Ca, the other classic resonance/rebound current) spans
    roughly 5-40 ms subthreshold -- corresponds to timescales up to ~a few
    tens of Hz, covered by the range's upper end.
  - tauNaM (fast Na activation) is sub-millisecond to ~1 ms even at rest --
    corresponds to >100s of Hz, well above anything a subthreshold
    impedance measurement needs to probe (Na is barely activated below
    threshold in the first place, and is not a resonance-generating current
    here -- it's the spike trigger). Sweeping up to it would just waste
    spectral bandwidth on a regime this protocol has no business probing.
  20 Hz is therefore a deliberately conservative upper bound that covers
  every plausible resonance-generating current in this model without
  chasing the (irrelevant, spike-generation-only) fast Na timescale.
  LINEAR (not logarithmic) frequency sweep: this range spans a modest 2
  decades, and the linear sweep is literature-standard for ZAP protocols
  (Hutcheon & Yarom's original design). A log sweep would concentrate more
  dwell time/spectral energy at the low-frequency end at the cost of the
  high end -- worth revisiting only if a cell's resonance turns out to sit
  right at the low edge of this range and needs more spectral resolution
  there than a linear sweep affords.
  20 s duration gives 1/20 = 0.05 Hz frequency resolution (FFT bin width),
  fine relative to the 0.2-20 Hz range being probed.

--- Design decision 4: impedance estimation ---
Raw FFT ratio: |Z(f)| = |FFT(V_ac(t))| / |FFT(Iapp_ac(t))|, phase(f) =
angle(FFT(V_ac)/FFT(Iapp_ac)), both AC-coupled (each signal's own window
mean subtracted) and identically Tukey-tapered (alpha=0.1) before the FFT to
suppress edge-discontinuity spectral leakage (the chirp trace has a hard
start/end and doesn't complete a whole number of cycles at the sweep
boundaries). This is the standard/simplest approach for a genuine
continuously-swept chirp waveform, per the task's own framing, and was not
replaced with a sliding-window instantaneous-ratio estimate: at this
sweep rate (df/dt = 0.99 Hz/s), any window short enough to have a
well-defined instantaneous frequency would span too few cycles at the low
end (0.2 Hz -> 5 s/cycle) to estimate an amplitude ratio reliably, while a
long enough window would smear across enough of the sweep to lose frequency
resolution -- exactly the tradeoff a single whole-trace FFT sidesteps.

Resonance frequency is only reported when scipy.signal.find_peaks finds a
genuine INTERIOR local maximum of |Z(f)| within the swept range (prominence
>= --resonance-prominence-frac of the |Z(f)| range) -- find_peaks never
reports the first/last sample as a peak, so a monotonically decaying
(simple low-pass) or monotonically rising |Z(f)| correctly yields
has_resonance=False / resonance_frequency_hz=None rather than a forced,
meaningless "peak" at a range edge.

--- What gets persisted ---
Matching the codebase's established convention of never caching full raw
voltage traces (see find_silencing_threshold.py's fine_traces and
run_held_injected_grid.py's module docstring): the full V(t)/Iapp(t) chirp
trace is used to compute the spectrum and draw the diagnostic figure, then
discarded -- only the (much smaller) frequency/impedance/phase arrays and
scalar summary fields are written to cell_chirp_protocol.pkl.
"""

import math
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
from scipy.signal.windows import tukey

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)
from find_silencing_threshold import (settle_at_level, _round_level,
                                      load_output_cache as load_silencing_cache,
                                      DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_SILENCING_CACHE_PATH,
                                      PROMINENCE_FRACTION, FLATLINE_MV)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_chirp_protocol.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "chirp_protocol"
DEFAULT_FIGURE_FORMAT = "svg"
# temp == reftemp -- see run_held_injected_grid.py's DEFAULT_TEMP comment
# (at dtemp=0 every q10 factor collapses to 1, so every cell runs at its
# literal reference-temperature values; this keeps Step 4 consistent with
# Steps 0-2, all of which were corrected to this convention).
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

DEFAULT_F_START_HZ = 0.2
DEFAULT_F_END_HZ = 20.0
DEFAULT_DURATION_S = 20.0
DEFAULT_AMPLITUDE_FRACTION = 0.4
DEFAULT_MIN_AMPLITUDE_NA = 0.02
DEFAULT_MAX_AMPLITUDE_NA = 2.0
DEFAULT_FALLBACK_AMPLITUDE_NA = 0.1
DEFAULT_RESONANCE_PROMINENCE_FRAC = 0.02
DEFAULT_TUKEY_ALPHA = 0.1

V_INDEX = 0


# ---------------------------------------------------------------------------
# Chirp waveform
# ---------------------------------------------------------------------------

def _make_chirp_iapp_func(held_nA: float, amplitude_nA: float, f_start_hz: float, k_hz_per_s: float):
    """Scalar t_ms -> nA closure matching this codebase's Iapp_func
    convention (see constant_iapp_func in find_silencing_threshold.py:
    scalar in, scalar out, t in ms). simulate() calls this once per
    integration step via a plain Python list comprehension (not
    vectorized) -- at dt=0.1ms and a 20s chirp that's 200,000 scalar calls,
    so plain `math` is used instead of numpy here to avoid numpy's
    per-call array-creation overhead on a single scalar, multiplied
    200,000x.
    """
    two_pi = 2.0 * math.pi

    def iapp_func(t_ms):
        t_s = t_ms / 1000.0
        phase = two_pi * (f_start_hz * t_s + 0.5 * k_hz_per_s * t_s * t_s)
        return held_nA + amplitude_nA * math.sin(phase)

    return iapp_func


def chirp_waveform_array(t_ms: np.ndarray, held_nA: float, amplitude_nA: float,
                         f_start_hz: float, f_end_hz: float, duration_s: float) -> np.ndarray:
    """Vectorized reconstruction of the exact waveform _make_chirp_iapp_func
    produced inside simulate() -- simulate() returns V(t) only, not the
    Iapp(t) it actually applied, so this recomputes it from the identical
    formula (same phase law, same held/amplitude) rather than an
    independently-derived approximation, guaranteeing the array used for
    the FFT/plot matches what actually drove the integration.
    """
    t_s = np.asarray(t_ms, dtype=np.float64) / 1000.0
    k = (f_end_hz - f_start_hz) / duration_s
    phase = 2.0 * np.pi * (f_start_hz * t_s + 0.5 * k * t_s ** 2)
    return held_nA + amplitude_nA * np.sin(phase)


# ---------------------------------------------------------------------------
# Amplitude/baseline anchoring from Step 1's silencing-threshold cache
# ---------------------------------------------------------------------------

def get_chirp_baseline(silencing_entry: dict, args: argparse.Namespace):
    """Returns (held_nA, amplitude_nA, anchor_source), or None if this cell
    must be skipped -- no valid Step 1 result to anchor a subthreshold
    operating point from. See module docstring, design decision 2.
    """
    status = silencing_entry.get("status")
    if status not in ("ok", "unsettled"):
        return None

    first_silent = silencing_entry.get("first_silent_level_nA")
    last_firing = silencing_entry.get("last_firing_level_nA")
    if first_silent is None and status == "ok":
        first_silent, last_firing = silencing_entry["silencing_threshold_bracket_nA"]
    if first_silent is None:
        return None

    held_nA = _round_level(first_silent)
    bracket_width = (last_firing - first_silent) if last_firing is not None else None

    if bracket_width is not None and bracket_width > 0:
        amplitude_nA = args.amplitude_fraction * bracket_width
        amplitude_nA = float(np.clip(amplitude_nA, args.min_amplitude_nA, args.max_amplitude_nA))
        anchor_source = "bracket_width"
    else:
        amplitude_nA = args.fallback_amplitude_nA
        anchor_source = "fallback_fixed"

    return held_nA, _round_level(amplitude_nA), anchor_source


# ---------------------------------------------------------------------------
# Impedance spectrum
# ---------------------------------------------------------------------------

def compute_impedance_spectrum(v_mV: np.ndarray, iapp_nA: np.ndarray, dt_ms: float,
                                f_start_hz: float, f_end_hz: float, tukey_alpha: float) -> dict:
    """FFT-ratio impedance estimate over the swept band only -- see module
    docstring, design decision 4. Returns freqs/z_mag/z_phase restricted to
    [f_start_hz, f_end_hz] AND to bins where the injected current itself
    has non-negligible spectral power (a bin with near-zero Iapp power would
    otherwise produce a spuriously huge |Z| from dividing by ~0, even though
    a linear chirp's dwell time -- and therefore expected power -- is
    roughly uniform across the swept band, so this is mostly a noise-floor
    guard rather than something expected to discard much).
    """
    n = len(v_mV)
    window = tukey(n, alpha=tukey_alpha)
    v_ac = (v_mV - v_mV.mean()) * window
    i_ac = (iapp_nA - iapp_nA.mean()) * window

    dt_s = dt_ms / 1000.0
    v_fft = np.fft.rfft(v_ac)
    i_fft = np.fft.rfft(i_ac)
    freqs = np.fft.rfftfreq(n, d=dt_s)

    in_band = (freqs >= f_start_hz) & (freqs <= f_end_hz)
    i_mag = np.abs(i_fft)
    band_peak = i_mag[in_band].max() if in_band.any() else 0.0
    valid = in_band & (i_mag > 1e-6 * band_peak) if band_peak > 0 else np.zeros_like(in_band)

    freqs_valid = freqs[valid]
    z = v_fft[valid] / i_fft[valid]
    return {
        "freqs_hz": freqs_valid,
        "z_mag_MOhm": np.abs(z),  # mV/nA == MOhm
        "z_phase_rad": np.angle(z),
    }


def find_resonance(freqs_hz: np.ndarray, z_mag_MOhm: np.ndarray, prominence_frac: float) -> dict:
    """Only reports a resonance frequency for a genuine interior local
    maximum of |Z(f)| -- some cells are expected to show simple low-pass
    behavior (monotonically decaying |Z(f)|) with no resonance at all, and
    this must say so explicitly rather than forcing a peak-finder to return
    something. find_peaks never treats a first/last-sample maximum as a
    peak, so a monotonic curve correctly yields has_resonance=False.
    """
    if len(freqs_hz) < 3:
        return {"has_resonance": False, "resonance_frequency_hz": None, "resonance_impedance_MOhm": None}
    z_range = z_mag_MOhm.max() - z_mag_MOhm.min()
    if z_range <= 0:
        return {"has_resonance": False, "resonance_frequency_hz": None, "resonance_impedance_MOhm": None}
    peaks, props = find_peaks(z_mag_MOhm, prominence=prominence_frac * z_range)
    if len(peaks) == 0:
        return {"has_resonance": False, "resonance_frequency_hz": None, "resonance_impedance_MOhm": None}
    best = peaks[int(np.argmax(props["prominences"]))]
    return {"has_resonance": True,
            "resonance_frequency_hz": float(freqs_hz[best]),
            "resonance_impedance_MOhm": float(z_mag_MOhm[best])}


# ---------------------------------------------------------------------------
# Per-cell orchestrator
# ---------------------------------------------------------------------------

def run_cell_chirp(cell_id: str, params: np.ndarray, ss_entry, silencing_entry, args: argparse.Namespace):
    """Never raises -- any failure is captured in the returned dict's
    "status" field. Returns (result, trace): trace (t_ms/v_mV/iapp_nA, or
    None) is kept OUT of the persisted cache (see module docstring) and is
    only used transiently by plot_cell_chirp.
    """
    base = {"cell_id": cell_id, "params": params, "dt": args.dt, "temp": args.temp, "reftemp": args.reftemp,
            "f_start_hz": args.f_start_hz, "f_end_hz": args.f_end_hz, "duration_s": args.duration_s}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}, None
    if silencing_entry is None:
        return {**base, "status": "no_silencing_threshold"}, None

    baseline = get_chirp_baseline(silencing_entry, args)
    if baseline is None:
        return {**base, "status": "silencing_not_ok"}, None
    held_nA, amplitude_nA, anchor_source = baseline
    base.update(held_nA=held_nA, amplitude_nA=amplitude_nA, amplitude_anchor_source=anchor_source)

    y_ss = ss_entry["y_ss"]
    hold_result = settle_at_level(params, y_ss.copy(), held_nA, args.dt, args.temp, args.reftemp,
                                  args.hold_settle_chunk_s, args.max_hold_settle_s,
                                  args.hold_settle_rtol, args.min_peaks_for_rate)
    if hold_result["blew_up"]:
        return {**base, "status": "blew_up", "stage": "hold", "error": hold_result.get("error")}, None

    base.update(hold_settled=hold_result["settled"], hold_freq_hz=hold_result["freq_hz"] or 0.0,
               hold_is_flatline=hold_result["is_flatline"])
    if not hold_result["is_flatline"]:
        # held_nA is Step 1's own confirmed-silent level, so this should
        # never actually fire -- if it does, the silencing cache entry is
        # likely stale/from different params than the ones just loaded.
        # Not fatal (the chirp still runs), but the "subthreshold" premise
        # this protocol depends on is violated, so it's flagged rather than
        # silently trusted.
        base["held_baseline_warning"] = ("hold level did not settle to flatline -- impedance "
                                         "estimate may be contaminated by ongoing spiking")

    k_hz_per_s = (args.f_end_hz - args.f_start_hz) / args.duration_s
    iapp_func = _make_chirp_iapp_func(held_nA, amplitude_nA, args.f_start_hz, k_hz_per_s)

    try:
        t_ms, states = simulate(params, args.duration_s, args.temp, dt=args.dt, reftemp=args.reftemp,
                                cis=hold_result["final_state"], Iapp_func=iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {**base, "status": "blew_up", "stage": "chirp", "error": str(exc)}, None
    if not np.all(np.isfinite(states)):
        return {**base, "status": "blew_up", "stage": "chirp", "error": "non-finite trajectory"}, None

    v_mV = states[:, V_INDEX]
    iapp_nA = chirp_waveform_array(t_ms, held_nA, amplitude_nA, args.f_start_hz, args.f_end_hz, args.duration_s)

    # Spike-detection guard: held_nA is Step 1's own confirmed-quiescent DC
    # level, but that's a STATIC threshold -- it doesn't bound a slowly-
    # varying AC signal. At the chirp's low-frequency end especially, a long
    # dwell near the depolarized swing's peak behaves almost like a sustained
    # step, and can trigger real spikes even though no *static* level this
    # chirp ever reaches was confirmed firing. Confirmed directly this
    # happens on real cells (e.g. 4QSWXH spikes for the first ~40% of the
    # sweep, ceasing once frequency rises enough that the membrane can no
    # longer track the oscillation). A spike is a large, brief, broadband
    # event -- its energy contaminates the FFT-ratio impedance estimate
    # across the ENTIRE spectrum, not just the frequencies where it
    # happened, so this can't be fixed by trimming the affected time window
    # after the fact; it has to be flagged, not silently trusted. The raw
    # spectrum is still returned (never silently discarded), but
    # impedance_valid=False tells any consumer this run's |Z(f)|/resonance
    # numbers are not trustworthy and must not be used without re-deriving
    # a smaller/safer amplitude for this specific cell.
    if v_mV.max() - v_mV.min() >= FLATLINE_MV:
        chirp_peaks, _ = find_peaks(v_mV, prominence=(v_mV.max() - v_mV.min()) * PROMINENCE_FRACTION)
        n_spikes_during_chirp = int(len(chirp_peaks))
    else:
        n_spikes_during_chirp = 0
    impedance_valid = n_spikes_during_chirp == 0

    spectrum = compute_impedance_spectrum(v_mV, iapp_nA, args.dt, args.f_start_hz, args.f_end_hz, args.tukey_alpha)
    resonance = find_resonance(spectrum["freqs_hz"], spectrum["z_mag_MOhm"], args.resonance_prominence_frac)

    result = {
        **base, "status": "ok", "error": None,
        "n_spikes_during_chirp": n_spikes_during_chirp, "impedance_valid": impedance_valid,
        "freqs_hz": spectrum["freqs_hz"], "z_mag_MOhm": spectrum["z_mag_MOhm"], "z_phase_rad": spectrum["z_phase_rad"],
        **resonance,
        "v_min_mV": float(v_mV.min()), "v_max_mV": float(v_mV.max()),
    }
    trace = {"t_ms": t_ms, "v_mV": v_mV, "iapp_nA": iapp_nA}
    return result, trace


# ---------------------------------------------------------------------------
# Cache I/O (this script's own output cache -- separate from, and never
# writes to, cell_silencing_thresholds.pkl or cell_held_injected_grid.pkl)
# ---------------------------------------------------------------------------

def load_output_cache(cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> None:
    # Write-then-rename + retry -- identical pattern to find_silencing_threshold.py
    # and run_held_injected_grid.py's save_output_cache (see either for the
    # confirmed Windows os.replace/PermissionError rationale). Called after
    # EVERY cell, not just once at the end.
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

def plot_cell_chirp(result: dict, trace, outdir: Path, command: str, fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    if result["status"] != "ok" or trace is None:
        return
    cell_id = result["cell_id"]
    t_ms, v_mV, iapp_nA = trace["t_ms"], trace["v_mV"], trace["iapp_nA"]
    freqs, z_mag, z_phase = result["freqs_hz"], result["z_mag_MOhm"], result["z_phase_rad"]

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 3, 1], hspace=0.5)
    ax_v = fig.add_subplot(gs[0])
    ax_i = fig.add_subplot(gs[1], sharex=ax_v)
    ax_z = fig.add_subplot(gs[2])
    ax_ph = fig.add_subplot(gs[3], sharex=ax_z)

    ax_v.plot(t_ms, v_mV, color="black", lw=0.5)
    ax_v.set_ylabel("V (mV)")
    title = (f"{cell_id} -- chirp/ZAP: held={result['held_nA']:.3f} nA, "
            f"amp=±{result['amplitude_nA']:.3f} nA ({result['amplitude_anchor_source']}), "
            f"{result['f_start_hz']:.2g}-{result['f_end_hz']:.2g} Hz over {result['duration_s']:.0f}s")
    if not result.get("impedance_valid", True):
        title += f"  [INVALID: {result['n_spikes_during_chirp']} spikes during chirp]"
        ax_v.set_title(title, fontsize=9, color="firebrick")
    else:
        ax_v.set_title(title, fontsize=9)

    ax_i.plot(t_ms, iapp_nA, color="firebrick", lw=0.5)
    ax_i.set_ylabel("Iapp (nA)")
    ax_i.set_xlabel("time (ms)")

    # A spike is a large, brief, broadband event -- if one occurred anywhere
    # in the trace it contaminates the FFT-ratio |Z(f)| estimate at every
    # frequency, not just where it happened (see run_cell_chirp's spike-
    # detection guard), so the whole panel gets flagged, not trimmed.
    z_color = "lightcoral" if not result.get("impedance_valid", True) else "steelblue"
    ax_z.plot(freqs, z_mag, color=z_color, lw=1.0)
    if not result.get("impedance_valid", True):
        ax_z.text(0.5, 0.5, f"INVALID -- {result['n_spikes_during_chirp']} spike(s) occurred during\n"
                            "the chirp; impedance estimate is contaminated\n"
                            "across the whole spectrum, not just near the spikes",
                 transform=ax_z.transAxes, ha="center", va="center", fontsize=8,
                 color="firebrick", style="italic",
                 bbox=dict(boxstyle="round", facecolor="mistyrose", edgecolor="firebrick", alpha=0.9))
    ax_z.set_xscale("log")
    ax_z.set_ylabel("|Z| (MΩ)")
    if result["has_resonance"]:
        ax_z.axvline(result["resonance_frequency_hz"], color="darkorange", ls="--", lw=1.2,
                    label=f"resonance ≈ {result['resonance_frequency_hz']:.2f} Hz")
        ax_z.legend(loc="best", fontsize=7)
    else:
        ax_z.text(0.02, 0.9, "no resonance peak (low-pass-like)", transform=ax_z.transAxes,
                 fontsize=7, color="dimgray", va="top")
    if result.get("held_baseline_warning"):
        ax_z.text(0.02, 0.05, result["held_baseline_warning"], transform=ax_z.transAxes,
                 fontsize=6, color="firebrick", va="bottom", wrap=True)

    ax_ph.plot(freqs, z_phase, color="seagreen", lw=1.0)
    ax_ph.set_xscale("log")
    ax_ph.axhline(0, color="gray", lw=0.5, ls=":")
    ax_ph.set_ylabel("phase (rad)")
    ax_ph.set_xlabel("frequency (Hz)")

    fig.text(0.5, 0.005, command, ha="center", va="bottom", fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_chirp.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chirp/ZAP protocol: swept-frequency sinusoidal current injection "
                    "(additive on top of a per-cell subthreshold held baseline, anchored "
                    "from Step 1's silencing threshold) to estimate subthreshold membrane "
                    "impedance |Z(f)|, phase, and resonance frequency.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR,
                        help="Directory containing 40-value cell parameter files.")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) to run. Default: every cell found in --params-dir.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH,
                        help="Path to the Iapp=0 steady-state cache (input, read-only; see generate_steady_state.py).")
    parser.add_argument("--silencing-cache", default=DEFAULT_SILENCING_CACHE_PATH,
                        help="Path to Step 1's silencing-threshold cache (input, read-only; "
                             "see find_silencing_threshold.py). Used to anchor the subthreshold "
                             "held baseline and chirp amplitude per cell.")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH,
                        help="Path to this script's own output pickle cache.")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR,
                        help="Directory for diagnostic impedance-spectrum figures.")
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["png", "svg", "pdf"])
    parser.add_argument("--no-plot", action="store_true", help="Skip diagnostic figure generation.")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS, help="Integration timestep in ms.")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel workers across cells (-1 uses all cores).")

    parser.add_argument("--f-start-hz", type=float, default=DEFAULT_F_START_HZ,
                        help="Chirp start frequency (Hz). See module docstring for the "
                             "time-constant-based reasoning behind the default range.")
    parser.add_argument("--f-end-hz", type=float, default=DEFAULT_F_END_HZ,
                        help="Chirp end frequency (Hz).")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S,
                        help="Total chirp duration (s); frequency is swept LINEARLY from "
                             "--f-start-hz to --f-end-hz over this duration.")

    parser.add_argument("--amplitude-fraction", type=float, default=DEFAULT_AMPLITUDE_FRACTION,
                        help="Chirp amplitude, as a fraction of this cell's Step-1 silencing "
                             "bracket width (last_firing_level_nA - first_silent_level_nA). "
                             "Kept < 1 so the depolarizing swing of the chirp never reaches a "
                             "level Step 1 actually confirmed firing at.")
    parser.add_argument("--min-amplitude-nA", type=float, default=DEFAULT_MIN_AMPLITUDE_NA,
                        help="Floor on chirp amplitude after applying --amplitude-fraction.")
    parser.add_argument("--max-amplitude-nA", type=float, default=DEFAULT_MAX_AMPLITUDE_NA,
                        help="Ceiling on chirp amplitude after applying --amplitude-fraction.")
    parser.add_argument("--fallback-amplitude-nA", type=float, default=DEFAULT_FALLBACK_AMPLITUDE_NA,
                        help="Fixed amplitude used when this cell's bracket width isn't usable "
                             "(e.g. Step 1 status 'unsettled' with no last_firing_level_nA).")

    parser.add_argument("--hold-settle-chunk-s", type=float, default=2.0,
                        help="Integration chunk length (s) while settling at the held baseline "
                             "before the chirp starts (reuses find_silencing_threshold.py's "
                             "settle_at_level engine).")
    parser.add_argument("--max-hold-settle-s", type=float, default=20.0)
    parser.add_argument("--hold-settle-rtol", type=float, default=0.05)
    parser.add_argument("--min-peaks-for-rate", type=int, default=2)

    parser.add_argument("--tukey-alpha", type=float, default=DEFAULT_TUKEY_ALPHA,
                        help="Tukey window taper fraction applied to both V(t) and Iapp(t) "
                             "before the FFT (see module docstring, design decision 4).")
    parser.add_argument("--resonance-prominence-frac", type=float, default=DEFAULT_RESONANCE_PROMINENCE_FRAC,
                        help="Minimum |Z(f)| peak prominence, as a fraction of the swept-band "
                             "|Z(f)| range, required to report a resonance frequency.")
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
            raise SystemExit(f"Unknown cell id(s): {unknown}\nAvailable cells in {params_dir}: {sorted(all_cells)}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Running chirp/impedance protocol for {len(cells)} of {len(all_cells)} cell(s) from {params_dir}")

    silencing_cache = load_silencing_cache(silencing_cache_path)
    output_cache = load_output_cache(output_cache_path)

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        silencing_entry = silencing_cache.get(cell_id)
        result, trace = run_cell_chirp(cell_id, params, ss_entry, silencing_entry, args)
        return cell_id, result, trace

    # Save+plot after EVERY cell via generator_unordered -- see
    # find_silencing_threshold.py/run_held_injected_grid.py's main() for the
    # same pattern and rationale (streams results as each cell finishes,
    # write-then-rename keeps a concurrent reader safe).
    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items())

    status_counts: dict = {}
    n_done = 0
    for cell_id, result, trace in result_iter:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if not args.no_plot:
            plot_cell_chirp(result, trace, figures_dir, command, fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cells)}] {cell_id}: {result['status']}")

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    invalid = [cid for cid in cells if output_cache[cid].get("status") == "ok"
              and not output_cache[cid].get("impedance_valid", True)]
    if invalid:
        print(f"\n{len(invalid)} cell(s) reached 'ok' but spiked during the chirp -- their "
             f"impedance estimate is contaminated and must not be used as-is: {invalid}")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    flagged = [cid for cid in cells if output_cache[cid]["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'no_silencing_threshold'/'silencing_not_ok': run find_silencing_threshold.py first "
              "(or check its status for that cell). 'blew_up': consider a smaller --dt.")


if __name__ == "__main__":
    main()
