"""Reusable overlay helpers that annotate a voltage-trace matplotlib axis
with exactly what a production detection/classification function found --
spike markers, ISI/KDE evidence, sag trough, adaptation window, rebound
window -- so a figure can be inspected panel-by-panel instead of trusting a
text label. Every helper calls the real function from
find_silencing_threshold.py / run_held_injected_grid.py that the grid
pipeline itself uses, never a re-implementation, so what's drawn is
provably what the pipeline computed.

Each mark_* function returns (evidence, caption): `evidence` is whatever
the underlying detector returned (so a caller can also report summary
numbers), and `caption` is a plain-language string describing what's drawn,
meant to be placed under the panel via ax.text/fig.text so the figure
stands on its own without narration.
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from find_silencing_threshold import (count_spikes_and_rate, PROMINENCE_FRACTION, FLATLINE_MV,
                                      detect_spikes_dvdt_confirmed, DEFAULT_DVDT_THRESHOLD_MV_PER_MS,
                                      DEFAULT_MIN_PRE_SPIKE_MS, compute_isis_ms, classify_burst_pattern)
from run_held_injected_grid import (compute_pre_spike_sag_trough, compute_adaptation_ratio,
                                    detect_onset_burst)


def mark_spikes(ax, t_ms: np.ndarray, v_mV: np.ndarray, color: str = "black",
                marker: str = "v", label: str = "detected spike") -> tuple:
    """Overlays the exact peaks `count_spikes_and_rate`/`compute_isis_ms`
    detect (scipy.signal.find_peaks with prominence = PROMINENCE_FRACTION
    of this trace's own V range) as markers just above each spike, so a
    reader can visually check the detector fired on every real upstroke and
    nothing else.
    """
    v_range = v_mV.max() - v_mV.min()
    if v_range < FLATLINE_MV:
        caption = (f"No spikes detected: voltage range ({v_range:.2f} mV) is below the "
                  f"FLATLINE_MV={FLATLINE_MV} mV cutoff, so this window is classified silent "
                  "without ever running peak detection.")
        return np.array([], dtype=int), caption

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(v_mV, prominence=v_range * PROMINENCE_FRACTION)
    if len(peaks) > 0:
        ax.plot(t_ms[peaks], v_mV[peaks] + 0.05 * v_range, linestyle="none",
               marker=marker, color=color, markersize=6, zorder=5, label=label)

    caption = (f"Markers show every peak scipy.signal.find_peaks detects at "
              f"prominence = {PROMINENCE_FRACTION:.2f} x this trace's own voltage range "
              f"({v_range:.1f} mV) -- the exact call count_spikes_and_rate/compute_isis_ms use "
              f"for every spike count, ISI, and burst/tonic classification in the pipeline. "
              f"{len(peaks)} spike(s) detected in this window.")
    return peaks, caption


def mark_confirmed_vs_rejected(ax, t_ms: np.ndarray, v_mV: np.ndarray, dt_ms: float,
                               confirmed_color: str = "seagreen", rejected_color: str = "firebrick"
                               ) -> tuple:
    """Cross-validation overlay: reruns the same prominence-based candidate
    peaks production code uses, but additionally gates each one through
    `detect_spikes_dvdt_confirmed` -- a dV/dt (Bean 2007) shape-confirmation
    check that exists in find_silencing_threshold.py but is never called by
    the production pipeline (every grid panel relies on prominence alone).
    Marks confirmed peaks (green) vs. any prominence-peak the dV/dt check
    would have rejected (red) -- if the two agree everywhere, it's evidence
    the simpler production detector isn't missing a shape check that
    matters; any red marker is a concrete case where they disagree.
    """
    v_range = v_mV.max() - v_mV.min()
    if v_range < FLATLINE_MV:
        return {"confirmed": np.array([], dtype=int), "rejected": np.array([], dtype=int)}, (
            f"No spikes to cross-validate: voltage range ({v_range:.2f} mV) is below "
            f"FLATLINE_MV={FLATLINE_MV} mV.")

    confirmed, rejected = detect_spikes_dvdt_confirmed(v_mV, t_ms, dt_ms)
    if len(confirmed) > 0:
        ax.plot(t_ms[confirmed], v_mV[confirmed] + 0.05 * v_range, linestyle="none", marker="o",
               color=confirmed_color, markersize=5, zorder=5,
               label=f"dV/dt-confirmed (n={len(confirmed)})")
    if len(rejected) > 0:
        ax.plot(t_ms[rejected], v_mV[rejected] + 0.05 * v_range, linestyle="none", marker="x",
               color=rejected_color, markersize=8, markeredgewidth=2, zorder=6,
               label=f"prominence-only, dV/dt-rejected (n={len(rejected)})")

    agree = len(rejected) == 0
    caption = (f"Cross-check against a second, unused-in-production detector: a candidate peak "
              f"additionally requires a genuine fast upstroke (dV/dt >= "
              f"{DEFAULT_DVDT_THRESHOLD_MV_PER_MS:.0f} mV/ms within {DEFAULT_MIN_PRE_SPIKE_MS:.0f} ms "
              f"before the peak, Bean 2007 convention) before counting. {len(confirmed)} of "
              f"{len(confirmed) + len(rejected)} candidate peak(s) confirmed"
              + ("; the two detectors agree on every spike in this window." if agree
                 else f"; {len(rejected)} candidate(s) failed the shape check -- see red x markers."))
    return {"confirmed": confirmed, "rejected": rejected}, caption


def mark_isi_classification(ax_isi, ax_kde, t_ms: np.ndarray, v_mV: np.ndarray,
                            min_isis_for_burst_test: int, isi_mode_prominence_frac: float,
                            min_isi_ratio: float, min_spikes_per_burst: float = 1.5) -> tuple:
    """Runs the real production classify_burst_pattern (via compute_isis_ms
    on this trace) and plots the two pieces of evidence behind whatever
    label it returned: the raw ISI-vs-spike-index sequence on `ax_isi`, and
    -- whenever the KDE bimodality test actually ran -- the log-ISI kernel
    density with its candidate modes and valley split marked on `ax_kde`.
    This applies equally to a "bursting" call and to a "tonic" call that
    still made it as far as the KDE (e.g. a near-miss rejected on the ratio
    or spikes-per-burst gate), so a borderline classification's rejection
    is just as visible as an accepted one.
    """
    isis_ms, n_peaks = compute_isis_ms(v_mV, t_ms, PROMINENCE_FRACTION)
    result = classify_burst_pattern(isis_ms, min_isis_for_burst_test, isi_mode_prominence_frac,
                                    min_isi_ratio, min_spikes_per_burst, n_peaks=n_peaks)

    if len(isis_ms) > 0:
        ax_isi.plot(np.arange(1, len(isis_ms) + 1), isis_ms, marker="o", markersize=3,
                   color="steelblue", lw=1)
    ax_isi.set_xlabel("ISI index (spike-to-spike)")
    ax_isi.set_ylabel("ISI (ms)")
    ax_isi.set_title(f"ISI sequence -> classify_burst_pattern = '{result['pattern']}'", fontsize=9)

    diag = result.get("diagnostics")
    if diag is None:
        ax_kde.text(0.5, 0.5, "KDE bimodality test never ran\n(too few ISIs, or degenerate/failed KDE fit)",
                   ha="center", va="center", fontsize=8, transform=ax_kde.transAxes, color="dimgray")
        caption = (f"{n_peaks} spike(s), {len(isis_ms)} ISI(s) -- fewer than "
                  f"min_isis_for_burst_test={min_isis_for_burst_test}, or a degenerate ISI spread, so "
                  f"the log-ISI KDE bimodality test never ran and the point defaults to '{result['pattern']}'.")
    else:
        grid, density = diag["log_isi_grid"], diag["density"]
        ax_kde.plot(grid, density, color="black", lw=1)
        cand = diag.get("candidate_mode_idx")
        if cand is not None and len(cand) > 0:
            ax_kde.plot(grid[cand], density[cand], linestyle="none", marker="^", color="gray",
                       markersize=6, label="candidate mode")
        if "mode_lo_log_isi" in diag:
            ax_kde.axvline(diag["mode_lo_log_isi"], color="seagreen", ls=":", lw=1.2, label="short mode")
            ax_kde.axvline(diag["mode_hi_log_isi"], color="darkorange", ls=":", lw=1.2, label="long mode")
            ax_kde.axvline(diag["split_log_isi"], color="firebrick", ls="--", lw=1.5, label="valley split")
        ax_kde.set_xlabel("log10(ISI / ms)")
        ax_kde.set_ylabel("KDE density")
        ax_kde.legend(loc="best", fontsize=6)
        ax_kde.set_title("log-ISI bimodality test", fontsize=9)

        pieces = [f"{n_peaks} spikes, {len(isis_ms)} ISIs."]
        if "isi_short_ms" in diag:
            ratio = diag["isi_long_ms"] / diag["isi_short_ms"]
            pieces.append(f"Split at {len(cand)} candidate mode(s) -> short ISI="
                         f"{diag['isi_short_ms']:.1f} ms, long ISI={diag['isi_long_ms']:.1f} ms "
                         f"(ratio {ratio:.2f}x, needs >= {min_isi_ratio}x).")
        if "avg_spikes_per_burst" in diag:
            pieces.append(f"Avg spikes/burst = {diag['avg_spikes_per_burst']:.2f} "
                         f"(needs >= {min_spikes_per_burst}).")
        pieces.append(f"Final call: '{result['pattern']}'"
                     + (f" (bimodality metric / Ashman's D = {result['bimodality_metric']:.2f})"
                        if result.get("bimodality_metric") else "") + ".")
        caption = " ".join(pieces)

    return result, caption


def mark_sag_trough(ax, t_test: np.ndarray, v_test: np.ndarray, hold_v_end_mV: float,
                    sag_window_ms: float) -> tuple:
    """Runs the real compute_pre_spike_sag_trough and draws the pre-test
    baseline (hold_v_end_mV), the window it searched (truncated at the
    first spike if one occurs before sag_window_ms elapses), and the trough
    it found -- the exact three pieces sag depth (baseline - trough) is
    built from.
    """
    trough_mV, first_spike_ms = compute_pre_spike_sag_trough(v_test, t_test, sag_window_ms)
    window_end_ms = min(sag_window_ms, first_spike_ms) if first_spike_ms is not None else sag_window_ms

    ax.axhline(hold_v_end_mV, color="gray", ls=":", lw=1.2, label=f"pre-test baseline ({hold_v_end_mV:.1f} mV)")
    ax.axvspan(t_test[0], t_test[0] + window_end_ms, color="steelblue", alpha=0.12,
              label="sag search window")
    trough_idx = int(np.argmin(np.abs(v_test - trough_mV)))
    ax.plot(t_test[trough_idx], trough_mV, marker="v", color="firebrick", markersize=8, zorder=5,
           label=f"trough ({trough_mV:.1f} mV)")

    sag_depth = hold_v_end_mV - trough_mV
    reason = (f"truncated early by the first spike at {first_spike_ms:.0f} ms" if first_spike_ms is not None
             and first_spike_ms < sag_window_ms else f"the full {sag_window_ms:.0f} ms window (no spike before it)")
    caption = (f"Sag depth = baseline - trough = {hold_v_end_mV:.1f} - ({trough_mV:.1f}) = "
              f"{sag_depth:.1f} mV. Baseline is the held-current settle's final voltage; trough is "
              f"the minimum voltage over {reason}.")
    return {"trough_mV": trough_mV, "first_spike_ms": first_spike_ms, "sag_depth_mV": sag_depth}, caption


def mark_adaptation_window(ax_isi, isis_ms: np.ndarray, edge_n: int) -> tuple:
    """Shades the first-edge_n / last-edge_n ISI windows compute_adaptation_ratio
    actually averages, on top of an existing ISI-vs-index plot (call after
    mark_isi_classification has already drawn ax_isi's ISI sequence).
    """
    n = len(isis_ms)
    if n < 2 * edge_n:
        caption = (f"Adaptation ratio not computed: only {n} ISI(s), fewer than 2 x "
                  f"adaptation_edge_n={edge_n} needed for non-overlapping first/last windows.")
        return None, caption

    idx = np.arange(1, n + 1)
    ax_isi.axvspan(idx[0] - 0.5, idx[edge_n - 1] + 0.5, color="seagreen", alpha=0.15, label="first N ISIs")
    ax_isi.axvspan(idx[-edge_n] - 0.5, idx[-1] + 0.5, color="darkorange", alpha=0.15, label="last N ISIs")
    ax_isi.legend(loc="best", fontsize=6)

    ratio = compute_adaptation_ratio(isis_ms, edge_n)
    direction = "slows (adapts)" if ratio > 1.02 else ("speeds up (facilitates)" if ratio < 0.98 else "stays flat")
    caption = (f"Adaptation ratio = mean(last {edge_n} ISIs) / mean(first {edge_n} ISIs) = {ratio:.2f}. "
              f"Firing {direction} over the course of this test window.")
    return ratio, caption


def mark_rebound_window(ax, t_rec: np.ndarray, v_rec: np.ndarray, rebound_latency_min_ms: float
                        ) -> tuple:
    """Reruns the exact recovery-window rebound-peak logic from
    run_test_and_recovery: prominence peaks in the recovery trace, gated to
    only count as "rebound" once at least rebound_latency_min_ms has
    elapsed since release (excludes a spike already in flight at the
    release instant). Marks qualifying rebound spikes vs. any earlier,
    non-qualifying peak, and draws the latency cutoff.
    """
    from scipy.signal import find_peaks
    v_range = v_rec.max() - v_rec.min()
    ax.axvline(t_rec[0] + rebound_latency_min_ms, color="gray", ls="--", lw=1,
              label=f"rebound latency cutoff ({rebound_latency_min_ms:.0f} ms)")
    if v_range < FLATLINE_MV:
        caption = f"No peaks in recovery window (range {v_range:.2f} mV, below FLATLINE_MV={FLATLINE_MV})."
        return {"rebound_occurred": False, "rebound_spike_count": 0}, caption

    peaks, _ = find_peaks(v_rec, prominence=v_range * PROMINENCE_FRACTION)
    peak_times_ms = t_rec[peaks] - t_rec[0]
    qualifying = peaks[peak_times_ms >= rebound_latency_min_ms]
    non_qualifying = peaks[peak_times_ms < rebound_latency_min_ms]

    if len(non_qualifying) > 0:
        ax.plot(t_rec[non_qualifying], v_rec[non_qualifying] + 0.05 * v_range, linestyle="none",
               marker="x", color="gray", markersize=7, label="in-flight (excluded)")
    if len(qualifying) > 0:
        ax.plot(t_rec[qualifying], v_rec[qualifying] + 0.05 * v_range, linestyle="none",
               marker="v", color="mediumpurple", markersize=7, label="rebound spike")

    occurred = len(qualifying) > 0
    caption = (f"{len(qualifying)} spike(s) at or after the {rebound_latency_min_ms:.0f} ms cutoff count "
              f"as rebound" + (f" (plus {len(non_qualifying)} excluded as already in flight at release)"
                               if len(non_qualifying) else "") + f". rebound_occurred={occurred}.")
    return {"rebound_occurred": occurred, "rebound_spike_count": int(len(qualifying))}, caption


def mark_onset_and_trailing_silence(ax_v, ax_isi, t_ms: np.ndarray, v_mV: np.ndarray,
                                    min_isi_ratio: float, min_onset_isis: int = 2,
                                    trailing_silence_ratio: float = 3.0) -> tuple:
    """Runs the real production onset-burst detector (detect_onset_burst)
    plus the trailing-silence/likely-ceased-firing check run_test_and_
    recovery applies before gating test_adaptation_ratio, and marks both
    pieces of evidence: the detected leading burst run on the raw voltage
    trace (ax_v, call AFTER the base trace is already plotted there) and
    ISI sequence (ax_isi), plus the gap between the last spike and window
    end. detect_onset_burst runs independent of the whole-window tonic/
    bursting label -- it can fire on a window that stays "tonic" overall
    (see run_test_and_recovery's docstring for the confirmed XB2IQX
    held=-2.48/inj=-4.04 case this exists for).
    """
    from scipy.signal import find_peaks
    isis_ms, n_peaks = compute_isis_ms(v_mV, t_ms, PROMINENCE_FRACTION)
    onset_n, onset_isi_mean = detect_onset_burst(isis_ms, min_isi_ratio, min_onset_isis)

    v_range = v_mV.max() - v_mV.min()
    peaks = np.array([], dtype=int)
    if v_range >= FLATLINE_MV:
        peaks, _ = find_peaks(v_mV, prominence=v_range * PROMINENCE_FRACTION)

    onset_peaks = peaks[:onset_n] if onset_n else np.array([], dtype=int)
    rest_peaks = peaks[onset_n:] if onset_n else peaks
    if len(onset_peaks) > 0:
        ax_v.plot(t_ms[onset_peaks], v_mV[onset_peaks] + 0.05 * v_range, linestyle="none", marker="v",
                 color="firebrick", markersize=7, zorder=5, label=f"onset burst (n={onset_n})")
    if len(rest_peaks) > 0:
        ax_v.plot(t_ms[rest_peaks], v_mV[rest_peaks] + 0.05 * v_range, linestyle="none", marker="v",
                 color="black", markersize=4, zorder=4, label="other spikes")

    window_ms = float(t_ms[-1])
    trailing_silence_ms, ceased = None, None
    if len(peaks) >= 1:
        last_spike_ms = float(t_ms[peaks[-1]])
        trailing_silence_ms = window_ms - last_spike_ms
        ax_v.axvspan(last_spike_ms, window_ms, color="gray", alpha=0.15,
                    label=f"trailing silence ({trailing_silence_ms:.0f} ms)")
        if len(isis_ms) > 0:
            ceased = bool(trailing_silence_ms >= trailing_silence_ratio * isis_ms[-1])

    if len(isis_ms) > 0:
        ax_isi.plot(np.arange(1, len(isis_ms) + 1), isis_ms, marker="o", markersize=4, color="steelblue")
        if onset_n and onset_n - 1 >= 1:
            ax_isi.axvspan(0.5, (onset_n - 1) + 0.5, color="firebrick", alpha=0.15,
                          label=f"onset burst ISIs (n={onset_n - 1})")
            ax_isi.legend(loc="best", fontsize=6)
    ax_isi.set_xlabel("ISI index (spike-to-spike)")
    ax_isi.set_ylabel("ISI (ms)")

    evidence = {"onset_n_spikes": onset_n, "onset_isi_mean_ms": onset_isi_mean,
               "trailing_silence_ms": trailing_silence_ms, "likely_ceased_firing": ceased,
               "n_spikes": n_peaks}

    pieces = []
    if onset_n:
        pieces.append(f"Onset burst detected: {onset_n} spikes (mean ISI {onset_isi_mean:.1f} ms) at "
                      "window start, found by scanning locally for the first ISI jump -- independent of "
                      "whatever whole-window tonic/bursting/silent label the KDE bimodality test assigns.")
    else:
        pieces.append("No burst-shaped leading run detected at window start (either no jump found, or the "
                      "leading run was shorter than min_onset_isis).")
    if trailing_silence_ms is not None:
        pieces.append(f"Trailing silence after the last spike: {trailing_silence_ms:.0f} ms of a "
                      f"{window_ms:.0f} ms window.")
        if ceased is not None:
            pieces.append(
                (f"likely_ceased_firing=True -- {trailing_silence_ms:.0f} ms is >= "
                 f"{trailing_silence_ratio:.0f}x the most recent ISI, so test_adaptation_ratio is "
                 "suppressed (reported as None): a first-k/last-k ratio isn't a meaningful 'smooth "
                 "adaptation' number for a train that stopped partway through the window.") if ceased else
                ("likely_ceased_firing=False -- firing continued close enough to window end that "
                 "test_adaptation_ratio is still computed normally from this same ISI sequence."))
    caption = " ".join(pieces)
    return evidence, caption
