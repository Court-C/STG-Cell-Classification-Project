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
                                      DEFAULT_MIN_PRE_SPIKE_MS, compute_isis_ms, classify_burst_pattern,
                                      MIN_SPIKE_AMPLITUDE_MV, _spike_prominence)
from run_held_injected_grid import (compute_pre_spike_sag_trough, compute_adaptation_ratio,
                                    detect_onset_burst)


def mark_spikes(ax, t_ms: np.ndarray, v_mV: np.ndarray, color: str = "black",
                marker: str = "v", label: str = "detected spike") -> tuple:
    """Overlays the exact peaks `count_spikes_and_rate`/`compute_isis_ms`
    detect (scipy.signal.find_peaks with prominence = max(PROMINENCE_FRACTION
    of this trace's own V range, MIN_SPIKE_AMPLITUDE_MV) -- see
    _spike_prominence) as markers just above each spike, so a reader can
    visually check the detector fired on every real upstroke and nothing
    else.
    """
    v_range = v_mV.max() - v_mV.min()
    if v_range < FLATLINE_MV:
        caption = (f"No spikes were detected: the voltage stayed within a {v_range:.2f} mV range "
                  f"for the whole window, below the {FLATLINE_MV} mV range required even to attempt "
                  "peak detection, so this window is classified as electrically silent.")
        return np.array([], dtype=int), caption

    from scipy.signal import find_peaks
    prominence = _spike_prominence(v_range, PROMINENCE_FRACTION)
    peaks, _ = find_peaks(v_mV, prominence=prominence)
    if len(peaks) > 0:
        ax.plot(t_ms[peaks], v_mV[peaks] + 0.05 * v_range, linestyle="none",
               marker=marker, color=color, markersize=6, zorder=5, label=label)

    floor_active = prominence > v_range * PROMINENCE_FRACTION
    caption = (f"Markers show every spike this trace's automated peak detector found, requiring "
              f"each peak to rise at least {prominence:.1f} mV above its surroundings ("
              + (f"a fixed {MIN_SPIKE_AMPLITUDE_MV:.0f} mV floor, since this trace's own "
                 f"{v_range:.1f} mV voltage range is small enough that a simple {PROMINENCE_FRACTION:.0%} "
                 "threshold would let subthreshold noise through"
                 if floor_active else
                 f"{PROMINENCE_FRACTION:.0%} of this trace's own {v_range:.1f} mV voltage range")
              + f") -- this is the same spike count used for every interspike-interval and "
              f"burst/tonic classification elsewhere in this report. {len(peaks)} spike(s) were "
              "detected in this window.")
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
            f"There are no spikes to cross-validate: the voltage stayed within a {v_range:.2f} mV "
            f"range for the whole window, below the {FLATLINE_MV} mV threshold for silence.")

    confirmed, rejected = detect_spikes_dvdt_confirmed(v_mV, t_ms, dt_ms)
    if len(confirmed) > 0:
        ax.plot(t_ms[confirmed], v_mV[confirmed] + 0.05 * v_range, linestyle="none", marker="o",
               color=confirmed_color, markersize=5, zorder=5,
               label=f"confirmed by upstroke shape (n={len(confirmed)})")
    if len(rejected) > 0:
        ax.plot(t_ms[rejected], v_mV[rejected] + 0.05 * v_range, linestyle="none", marker="x",
               color=rejected_color, markersize=8, markeredgewidth=2, zorder=6,
               label=f"rejected on upstroke shape (n={len(rejected)})")

    agree = len(rejected) == 0
    caption = (f"As an independent check, each candidate spike was also required to show a "
              f"fast upstroke -- a rate of voltage rise of at least {DEFAULT_DVDT_THRESHOLD_MV_PER_MS:.0f} "
              f"mV/ms within {DEFAULT_MIN_PRE_SPIKE_MS:.0f} ms before the peak (the convention used by "
              f"Bean, 2007) -- rather than amplitude alone. {len(confirmed)} of "
              f"{len(confirmed) + len(rejected)} candidate spike(s) were confirmed by this stricter test"
              + ("; the two methods agree on every spike in this window." if agree
                 else f"; {len(rejected)} candidate(s) failed the upstroke-shape test -- see the "
                      "red x markers."))
    return {"confirmed": confirmed, "rejected": rejected}, caption


def mark_isi_classification(ax_isi, ax_kde, t_ms: np.ndarray, v_mV: np.ndarray,
                            min_isis_for_burst_test: int, isi_mode_prominence_frac: float,
                            min_isi_ratio: float, min_spikes_per_burst: float = 1.5,
                            min_ashman_d: float = 2.0, isis_override=None, n_peaks_override=None,
                            log_isi_xlim: tuple = None) -> tuple:
    """Runs the real production classify_burst_pattern (via compute_isis_ms
    on this trace) and plots the two pieces of evidence behind whatever
    label it returned: the raw ISI-vs-spike-index sequence on `ax_isi`, and
    -- whenever the KDE bimodality test actually ran -- the log-ISI kernel
    density with its candidate modes and valley split marked on `ax_kde`.
    This applies equally to a "bursting" call and to a "tonic" call that
    still made it as far as the KDE (e.g. a near-miss rejected on the ratio
    or spikes-per-burst gate), so a borderline classification's rejection
    is just as visible as an accepted one.

    isis_override/n_peaks_override: pass a pre-computed ISI array directly
    (t_ms/v_mV are then ignored) for panels illustrating a CONSTRUCTED/
    synthetic ISI sequence with no underlying simulated trace to derive one
    from -- e.g. a hand-built example of a specific gate's decision
    boundary. Both must be given together.

    log_isi_xlim: when a caller is placing several of these panels side by
    side for direct comparison, each panel's own KDE curve is otherwise
    plotted over whatever [min, max] its own data happens to span -- which
    makes the panels visually incomparable (a confidently-tonic point's ISI
    might span a fraction of a log10-unit while a confidently-bursting
    point's spans two full units). Pass a shared (lo, hi) range -- the union
    across every panel being compared, not this panel's own range -- and
    the ALREADY-FITTED kde (diag["kde"]) is re-evaluated over that wider
    window before plotting, so the curve is a real continuation of the same
    fit, not a cropped/rescaled copy of the original narrower one.
    """
    if isis_override is not None:
        isis_ms, n_peaks = isis_override, n_peaks_override
    else:
        isis_ms, n_peaks = compute_isis_ms(v_mV, t_ms, PROMINENCE_FRACTION)
    result = classify_burst_pattern(isis_ms, min_isis_for_burst_test, isi_mode_prominence_frac,
                                    min_isi_ratio, min_spikes_per_burst, min_ashman_d, n_peaks=n_peaks)

    if len(isis_ms) > 0:
        ax_isi.plot(np.arange(1, len(isis_ms) + 1), isis_ms, marker="o", markersize=3,
                   color="steelblue", lw=1)
        # Always include zero: auto-scaling tightly around the data (e.g. a
        # clean tonic train whose intervals span only 12.9-13.0ms) blows a
        # sub-millisecond wobble up to fill the whole panel height, making
        # negligible jitter look like a dramatic zigzag -- user-flagged
        # 2026-08-14, page 12 rows 2-3. A real axis anchored at zero keeps
        # the panel's visual scale honest about how small that variation
        # actually is relative to the interval itself.
        ax_isi.set_ylim(bottom=0)
    ax_isi.set_xlabel("interval index (spike-to-spike)")
    ax_isi.set_ylabel("interspike interval (ms)")
    ax_isi.set_title(f"Interspike intervals -- classified as '{result['pattern']}'", fontsize=9)

    diag = result.get("diagnostics")
    if diag is None:
        ax_kde.text(0.5, 0.5, "Too few spikes to test for two interval populations",
                   ha="center", va="center", fontsize=8, transform=ax_kde.transAxes, color="dimgray")
        caption = (f"This window had {n_peaks} spike(s), giving {len(isis_ms)} interspike interval(s) -- "
                  f"fewer than the {min_isis_for_burst_test} needed to statistically test whether "
                  "intervals fall into two separate populations (short, within-burst gaps vs. long, "
                  f"between-burst gaps), so the window defaults to '{result['pattern']}'.")
    else:
        if log_isi_xlim is not None:
            grid = np.linspace(log_isi_xlim[0], log_isi_xlim[1], 400)
            density = diag["kde"](grid)
        else:
            grid, density = diag["log_isi_grid"], diag["density"]
        ax_kde.plot(grid, density, color="black", lw=1)
        # candidate_mode_idx indexes diag's OWN original grid/density arrays
        # (200 points over this panel's own data range), never the
        # log_isi_xlim-rescaled arrays above (400 points, different range/
        # spacing) -- index into diag directly rather than the local
        # grid/density names to stay correct in both branches.
        cand = diag.get("candidate_mode_idx")
        if cand is not None and len(cand) > 0:
            ax_kde.plot(diag["log_isi_grid"][cand], diag["density"][cand], linestyle="none", marker="^",
                       color="gray", markersize=6, label="candidate interval population")
        if "mode_lo_log_isi" in diag:
            ax_kde.axvline(diag["mode_lo_log_isi"], color="seagreen", ls=":", lw=1.2,
                           label="short (within-burst) intervals")
            ax_kde.axvline(diag["mode_hi_log_isi"], color="darkorange", ls=":", lw=1.2,
                           label="long (between-burst) intervals")
            ax_kde.axvline(diag["split_log_isi"], color="firebrick", ls="--", lw=1.5,
                           label="dividing line between the two")
        if log_isi_xlim is not None:
            ax_kde.set_xlim(log_isi_xlim)
        ax_kde.set_xlabel("interspike interval (log scale, ms)")
        ax_kde.set_ylabel("relative frequency")
        ax_kde.legend(loc="best", fontsize=6)
        ax_kde.set_title("Do intervals form one population or two?", fontsize=9)
        # Ashman's D isn't a position on this axis (it's a scalar summarizing
        # how separated the two mode clusters are), so it can't be drawn as
        # a line the way the modes/valley are -- shown as an on-plot
        # annotation instead of only in the caption below, so the pass/fail
        # margin against the threshold is visible without reading text.
        if "ashman_d" in diag:
            # Bottom-center, not a corner: the mode/valley legend's loc="best"
            # consistently lands top-right on this shape of curve (two peaks
            # with a dip between them), so anchoring there collided with it.
            # The area below the valley dip is reliably open regardless of
            # where the two modes themselves happen to sit.
            d, passed = diag["ashman_d"], diag["ashman_d"] >= min_ashman_d
            ax_kde.annotate(f"separation score = {d:.2f} (>= {min_ashman_d} counts as two separate "
                            "populations)",
                            xy=(0.5, 0.04), xycoords="axes fraction", ha="center", va="bottom", fontsize=7,
                            color="darkgreen" if passed else "firebrick",
                            bbox=dict(boxstyle="round", fc="white", ec="darkgreen" if passed else "firebrick",
                                     alpha=0.9))

        pieces = [f"This window contained {n_peaks} spikes, giving {len(isis_ms)} interspike intervals."]
        if "isi_short_ms" in diag:
            ratio = diag["isi_long_ms"] / diag["isi_short_ms"]
            pieces.append(f"Splitting the intervals at their {len(cand)} candidate population(s) gives a "
                         f"short (within-burst) interval of {diag['isi_short_ms']:.1f} ms and a long "
                         f"(between-burst) interval of {diag['isi_long_ms']:.1f} ms -- a "
                         f"{ratio:.2f}-fold difference, above the {min_isi_ratio}-fold difference "
                         "required to treat them as distinct.")
        if "avg_spikes_per_burst" in diag:
            pieces.append(f"On average, {diag['avg_spikes_per_burst']:.2f} spikes fall within each burst "
                         f"(a real burst requires at least {min_spikes_per_burst}, ruling out what would "
                         "otherwise just be occasional missed spikes in an otherwise tonic train).")
        if "ashman_d" in diag:
            pieces.append(f"A statistical separation score (Ashman's D) of {diag['ashman_d']:.2f} "
                         f"quantifies how cleanly the short and long intervals separate into two "
                         f"distinct populations (a score of {min_ashman_d} or higher is required to "
                         "call them separate).")
        pieces.append(f"Taken together, this window is classified as '{result['pattern']}'"
                     + (f", with a separation score of {result['bimodality_metric']:.2f}"
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

    ax.axhline(hold_v_end_mV, color="gray", ls=":", lw=1.2, label=f"pre-step baseline ({hold_v_end_mV:.1f} mV)")
    ax.axvspan(t_test[0], t_test[0] + window_end_ms, color="steelblue", alpha=0.12,
              label="window searched for the trough")
    trough_idx = int(np.argmin(np.abs(v_test - trough_mV)))
    ax.plot(t_test[trough_idx], trough_mV, marker="v", color="firebrick", markersize=8, zorder=5,
           label=f"most hyperpolarized point ({trough_mV:.1f} mV)")

    sag_depth = hold_v_end_mV - trough_mV
    reason = (f"stopping early at the first spike, {first_spike_ms:.0f} ms in" if first_spike_ms is not None
             and first_spike_ms < sag_window_ms else f"the full {sag_window_ms:.0f} ms window (no spike "
             "occurred before it ended)")
    caption = (f"Sag depth is the difference between the voltage just before the current step "
              f"({hold_v_end_mV:.1f} mV) and the most hyperpolarized point reached afterward "
              f"({trough_mV:.1f} mV), giving {sag_depth:.1f} mV of sag -- a hallmark of the "
              f"hyperpolarization-activated current (Ih) partially repolarizing the cell during the "
              f"step. The trough was found by searching {reason}.")
    return {"trough_mV": trough_mV, "first_spike_ms": first_spike_ms, "sag_depth_mV": sag_depth}, caption


def mark_adaptation_window(ax_isi, isis_ms: np.ndarray, edge_n: int) -> tuple:
    """Shades the first-edge_n / last-edge_n ISI windows compute_adaptation_ratio
    actually averages, on top of an existing ISI-vs-index plot (call after
    mark_isi_classification has already drawn ax_isi's ISI sequence).
    """
    n = len(isis_ms)
    if n < 2 * edge_n:
        caption = (f"Spike-frequency adaptation was not assessed: this window contained only {n} "
                  f"interspike interval(s), fewer than the {2 * edge_n} needed to compare non-overlapping "
                  "groups of intervals at the start and end of firing.")
        return None, caption

    idx = np.arange(1, n + 1)
    ax_isi.axvspan(idx[0] - 0.5, idx[edge_n - 1] + 0.5, color="seagreen", alpha=0.15, label="first intervals")
    ax_isi.axvspan(idx[-edge_n] - 0.5, idx[-1] + 0.5, color="darkorange", alpha=0.15, label="last intervals")
    ax_isi.legend(loc="best", fontsize=6)

    ratio = compute_adaptation_ratio(isis_ms, edge_n)
    direction = ("slows over the course of firing (spike-frequency adaptation)" if ratio > 1.02
                else ("speeds up over the course of firing (facilitation)" if ratio < 0.98
                      else "stays roughly constant"))
    caption = (f"The adaptation ratio -- the mean of the last {edge_n} interspike intervals divided by "
              f"the mean of the first {edge_n} -- is {ratio:.2f}. Firing {direction} over this window.")
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
              label=f"minimum latency to count as rebound ({rebound_latency_min_ms:.0f} ms)")
    if v_range < FLATLINE_MV:
        caption = (f"No spikes occurred during the recovery window (voltage range "
                  f"{v_range:.2f} mV, below the {FLATLINE_MV} mV threshold for silence): this cell "
                  "showed no post-inhibitory rebound.")
        return {"rebound_occurred": False, "rebound_spike_count": 0}, caption

    peaks, _ = find_peaks(v_rec, prominence=_spike_prominence(v_range, PROMINENCE_FRACTION))
    peak_times_ms = t_rec[peaks] - t_rec[0]
    qualifying = peaks[peak_times_ms >= rebound_latency_min_ms]
    non_qualifying = peaks[peak_times_ms < rebound_latency_min_ms]

    if len(non_qualifying) > 0:
        ax.plot(t_rec[non_qualifying], v_rec[non_qualifying] + 0.05 * v_range, linestyle="none",
               marker="x", color="gray", markersize=7, label="already firing at release (excluded)")
    if len(qualifying) > 0:
        ax.plot(t_rec[qualifying], v_rec[qualifying] + 0.05 * v_range, linestyle="none",
               marker="v", color="mediumpurple", markersize=7, label="rebound spike")

    occurred = len(qualifying) > 0
    caption = (f"{len(qualifying)} spike(s) occurred at or after {rebound_latency_min_ms:.0f} ms "
              "post-release and so count as post-inhibitory rebound"
              + (f" ({len(non_qualifying)} additional spike(s) were already underway at the moment of "
                 "release and are excluded)" if len(non_qualifying) else "")
              + f". This cell {'did' if occurred else 'did not'} show a rebound response.")
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
                    label=f"silence after the last spike ({trailing_silence_ms:.0f} ms)")
        if len(isis_ms) > 0:
            ceased = bool(trailing_silence_ms >= trailing_silence_ratio * isis_ms[-1])

    if len(isis_ms) > 0:
        ax_isi.plot(np.arange(1, len(isis_ms) + 1), isis_ms, marker="o", markersize=4, color="steelblue")
        # See mark_isi_classification's identical fix: anchoring at zero
        # keeps the panel from blowing a sub-millisecond wobble up into an
        # apparently dramatic zigzag.
        ax_isi.set_ylim(bottom=0)
        if onset_n and onset_n - 1 >= 1:
            ax_isi.axvspan(0.5, (onset_n - 1) + 0.5, color="firebrick", alpha=0.15,
                          label=f"onset burst (n={onset_n - 1})")
            ax_isi.legend(loc="best", fontsize=6)
    ax_isi.set_xlabel("interval index (spike-to-spike)")
    ax_isi.set_ylabel("interspike interval (ms)")

    evidence = {"onset_n_spikes": onset_n, "onset_isi_mean_ms": onset_isi_mean,
               "trailing_silence_ms": trailing_silence_ms, "likely_ceased_firing": ceased,
               "n_spikes": n_peaks}

    pieces = []
    if onset_n:
        pieces.append(f"A leading onset burst of {onset_n} spikes (mean interval {onset_isi_mean:.1f} ms) "
                      "occurred at the start of the window. This local test is independent of whichever "
                      "tonic/bursting/silent label the whole-window statistical test assigns, so it can "
                      "detect a real leading burst even when the window as a whole reads as tonic.")
    else:
        pieces.append("No burst-shaped leading run of spikes occurred at the start of this window.")
    if trailing_silence_ms is not None:
        pieces.append(f"Firing was followed by {trailing_silence_ms:.0f} ms of silence before the end of "
                      f"this {window_ms:.0f} ms window.")
        if ceased is not None:
            pieces.append(
                (f"This silence is at least {trailing_silence_ratio:.0f} times the most recent interspike "
                 "interval, indicating firing had ended rather than being cut off by the window boundary; "
                 "an adaptation ratio is not reported for this window, since comparing early and late "
                 "intervals is not meaningful once firing has stopped partway through.")
                if ceased else
                ("Firing continued close enough to the end of the window that it was not cut off "
                 "partway through, so an adaptation ratio is still computed normally from this interval "
                 "sequence."))
    caption = " ".join(pieces)
    return evidence, caption
