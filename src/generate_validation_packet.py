"""Consolidated validation packet: one PDF (plus standalone per-section PNGs
for attaching individually) that walks through every grid-feature panel's
underlying detection/classification evidence on real traces, so the reader
can check that the feature-extraction pipeline is doing the right thing
without taking any classification on faith.

Every page/section reuses the actual production functions from
find_silencing_threshold.py / run_held_injected_grid.py / extract_grid_features.py
via src/trace_annotations.py's overlay helpers -- nothing here re-implements
detection logic, so what's drawn is provably what the pipeline computed.
"""

import argparse
import pickle
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from steady_state_cache import get_cached_state, CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH
from plot_example_traces import (resimulate_point, select_exemplar_points, build_example_trace_figure,
                                 describe_pattern as _describe_pattern)
from find_silencing_threshold import (compute_isis_ms, classify_burst_pattern, count_spikes_and_rate,
                                      PROMINENCE_FRACTION, detect_spikes_dvdt_confirmed, FLATLINE_MV,
                                      MIN_SPIKE_AMPLITUDE_MV)
from extract_grid_features import (DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_FEATURES_CACHE_PATH,
                                   compute_fi_slope)
from run_held_injected_grid import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH
from run_held_injected_grid import build_cell_grid_features_fig
from trace_annotations import (mark_spikes, mark_confirmed_vs_rejected, mark_isi_classification,
                               mark_sag_trough, mark_adaptation_window, mark_rebound_window,
                               mark_onset_and_trailing_silence)

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "validation_packet"

# A train with more spikes than this renders its raw trace/ISI panels as a
# solid block of color / an unreadable dense line rather than distinguishable
# spikes, so the DISPLAYED window is cropped to its first DISPLAY_WINDOW_MS --
# shared by every page that plots a raw voltage trace or ISI-vs-index
# sequence next to a classification (sections 5 and 6, user-flagged
# 2026-08-14). A trace with fewer spikes than this is left uncropped: a slow,
# low-spike-count bursting train (e.g. 11 spikes over 3s) is already readable
# at its full, natural window, and a fixed time crop would instead cut off
# most of its inter-burst rhythm for no reason. The classification itself is
# always computed from the full train regardless.
CROP_TRIGGER_N_SPIKES = 25

# A fixed TIME window, not a fixed spike COUNT (superseded 2026-08-26 for
# whatever trace DOES get cropped, per CROP_TRIGGER_N_SPIKES above): the
# first N spikes of a fast train and the first N spikes of a slow train span
# very different real durations, so a spike-count-derived crop width doesn't
# give a consistent, actually-readable window across different firing rates
# -- confirmed still "too saturated to make out individual spikes" on a
# ~60Hz tonic train even cropped to its first 25 spikes (~420ms). 500ms
# keeps a fast tonic train's individual spikes visually distinct.
DISPLAY_WINDOW_MS = 500.0


def _crop_display_window(ax_v, ax_isi, t: np.ndarray, peaks: np.ndarray,
                         window_ms: float = DISPLAY_WINDOW_MS,
                         trigger_n_spikes: int = CROP_TRIGGER_N_SPIKES) -> None:
    """Crops the DISPLAYED voltage trace (`ax_v`) and ISI-vs-index panel
    (`ax_isi`) to this trace's first `window_ms`, but only when it's actually
    dense enough to need it (more than `trigger_n_spikes` spikes) -- see
    CROP_TRIGGER_N_SPIKES/DISPLAY_WINDOW_MS's docstrings for why each of
    those is a separate decision (whether to crop vs. how wide the crop is).
    No-op if the trace doesn't meet the trigger, or is already shorter than
    the window. The classification itself (caption text, KDE curve,
    separation score) is always computed from the full train; this only
    narrows what's drawn.
    """
    if len(peaks) <= trigger_n_spikes:
        return
    if len(t) == 0 or t[-1] - t[0] <= window_ms:
        return
    crop_end_ms = t[0] + window_ms
    ax_v.set_xlim(t[0], crop_end_ms)
    n_isis_in_window = int(np.sum(t[peaks] <= crop_end_ms)) - 1
    if n_isis_in_window >= 1:
        ax_isi.set_xlim(0.5, n_isis_in_window + 0.5)


def _wrap(text: str, width: int = 100) -> str:
    return textwrap.fill(text, width=width)




def _place_caption_below(fig, ax, caption: str, width: int = None, fontsize: float = None, gap: float = 0.10,
                         right_ax=None) -> None:
    """Places a caption under `ax` using its REAL post-layout bounding box
    (fig.transFigure, computed from ax.get_position() after tight_layout has
    already run) rather than a fixed ax.transAxes fraction. A fraction-based
    offset (e.g. ax.text(0.5, -0.3, ...)) interacts badly with the xlabel's
    own fixed-point padding -- confirmed directly this caused a caption's
    first line to render on top of "time (ms)" for a short single-row
    figure, since a -0.3 axes-fraction offset is a different absolute
    distance depending on how tall the axes happens to be, while the
    xlabel's gap is a fixed point size regardless. Anchoring to the actual
    figure-coordinate bbox bottom instead sidesteps that entirely -- must be
    called AFTER fig.tight_layout()/subplots_adjust(), not before.

    right_ax: when the caption is only really "about" one column (e.g. the
    ISI panel) but there are neighboring columns with nothing of their own
    printed underneath, pass the rightmost axes of the span to center the
    caption under -- gives a much wider wrap width for the same font size
    instead of cramming into one column while the rest of the row sits
    empty (user-flagged 2026-08-14, page 5).

    width/fontsize: left as None, both scale automatically with the span's
    real width in inches, so a caption under a single narrow column doesn't
    default to the same small size as one spanning a whole multi-column row
    (user-flagged 2026-08-14 -- "make the captions larger depending on the
    page size" -- previously every page hand-picked its own fixed values).
    The two coefficients below were fit to the fontsize/width pairs already
    hand-tuned and visually confirmed not to overflow on pages 2, 5, 6, 7,
    and 8 (spans from ~6in single columns to ~19in full-row spans) -- pass
    either explicitly to override the derived value for a specific page.
    """
    bbox = ax.get_position()
    x0, x1 = bbox.x0, bbox.x1
    if right_ax is not None:
        x1 = right_ax.get_position().x1
    span_in = (x1 - x0) * fig.get_size_inches()[0]
    if fontsize is None:
        fontsize = float(np.clip(6.35 + 0.192 * span_in, 6.5, 11.0))
    if width is None:
        width = max(30, int(65 * span_in / fontsize))
    fig.text((x0 + x1) / 2, bbox.y0 - gap, _wrap(caption, width),
             ha="center", va="top", fontsize=fontsize, transform=fig.transFigure)


def _save_page(fig, pdf: PdfPages, png_dir: Path, name: str, command: str, source: str) -> None:
    """Stamps two provenance lines on every page before saving, matching
    every other figure-generating script in this repo (run_held_injected_
    grid.py, extract_grid_features.py, consolidate_features.py, cluster_
    features.py, plot_example_traces.py all embed the exact command that
    produced the figure) -- generate_validation_packet.py was the one script
    in the pipeline that didn't, which matters more here than anywhere
    else: a validation packet whose own figures can't be traced back to
    the command and cache that produced them undercuts its own point.
    Bottom-left, not the bottom-center every caption already occupies, so
    the two never collide regardless of a given page's own layout.
    """
    fig.text(0.005, 0.005, f"{command}\nsource: {source}", ha="left", va="bottom",
             fontsize=5, family="monospace", color="dimgray")
    pdf.savefig(fig)
    # name may include a subfolder (e.g. "representative_traces_detail/foo")
    # -- create it if this is the first page landing there.
    png_path = png_dir / f"{name}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _resim(cell_id, params, y_ss, baseline_freq_hz, cell_result, held_nA, injected_nA, hold_tail_s=0.5,
          ss_entry=None):
    return resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA, cell_result, hold_tail_s,
                            ss_entry=ss_entry)


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def make_cover_page(cell_id: str, cell_result: dict, features: dict, sections: list) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.93, f"Grid Features Validation Packet -- {cell_id}", fontsize=18, ha="center", weight="bold")

    intro = _wrap(
        "Every trace in this packet is simulated output from a fitted, multi-conductance "
        "single-compartment model of this cell, integrated deterministically -- not a raw "
        "current-clamp recording. What is being evaluated here is the feature-extraction "
        "methodology applied on top of that model: spike detection, burst/tonic classification, "
        "post-inhibitory rebound detection, sag depth, and spike-frequency adaptation. Each figure "
        "below reruns the actual classification algorithm on a real simulated trace and marks what "
        "it found directly on the trace.", width=95)
    fig.text(0.08, 0.87, intro, fontsize=10, va="top", wrap=True)

    burstiness = features.get("burstiness_index")
    fi_slope = features.get("fi_slope_hz_per_nA")
    fi_slope_r2 = features.get("fi_slope_r2")
    summary_line1 = (f"Silencing threshold: {cell_result['cell_floor_nA']:.2f} nA   |   "
                     f"{cell_result['n_points_total']} points sampled across the current grid")
    summary_line2 = f"Burstiness index: {burstiness:.3f}" if burstiness is not None else "Burstiness index: None"
    summary_line3 = (f"Firing-rate/current slope: {fi_slope:.2f} Hz/nA (R^2={fi_slope_r2:.3f})"
                     if fi_slope is not None and fi_slope_r2 is not None
                     else "Firing-rate/current slope: not available (no tonic points to fit)")
    fig.text(0.08, 0.60, summary_line1, fontsize=9, va="top", family="monospace")
    fig.text(0.08, 0.575, summary_line2, fontsize=9, va="top", family="monospace")
    fig.text(0.08, 0.55, summary_line3, fontsize=9, va="top", family="monospace")

    fig.text(0.08, 0.50, "Contents:", fontsize=11, va="top", weight="bold")
    y = 0.46
    # 0.41, not the full 0.44 down to y=0.02: reserves room above the
    # provenance footer _save_page stamps at y=0.005 on every page
    # (including this one) -- confirmed directly that the previous 0.47
    # span let an 11-section list's last entry visually collide with it.
    # Each entry budgets for a title line plus up to 2 description lines
    # (0.014 each) -- descriptions are kept short enough to usually need
    # only 1, but the budget must cover the worst case or a long
    # description collides with the next entry's title.
    step = 0.41 / len(sections)
    for i, (title, desc) in enumerate(sections, start=1):
        fig.text(0.10, y, f"{i}. {title}", fontsize=9.5, va="top", weight="bold")
        fig.text(0.14, y - 0.016, _wrap(desc, width=105), fontsize=7, va="top", color="dimgray")
        y -= step

    return fig


def make_feature_overview_page(cell_id, cell_result, features) -> plt.Figure:
    """The actual result being validated: every panel of extracted features
    across this cell's full current grid, laid out exactly as extract_grid_
    features.py's own standalone figure (same helper, same no-interpolation
    exact-matrix rendering -- see build_cell_grid_features_fig). Every page
    after this one drills into one piece of evidence behind one of these
    panels; without this page first, a reader has no way to see what the
    rest of the packet is actually building a case for (user-flagged
    2026-08-14: "hard to show what we've proved something if we don't know
    what's being proved").

    No resimulation -- built entirely from the cached grid and features
    dicts already on disk, the same cached data every other panel in this
    row of pipeline scripts reads.
    """
    fig = build_cell_grid_features_fig(cell_result, features, bottom_margin=0.16)
    burstiness = features.get("burstiness_index")
    fi_slope = features.get("fi_slope_hz_per_nA")
    fi_r2 = features.get("fi_slope_r2")
    fig.suptitle(f"1. Feature overview: {cell_id}'s full current grid", fontsize=13, weight="bold", y=0.995)
    burstiness_str = f"{burstiness:.3f}" if burstiness is not None else "None"
    fi_slope_str = (f"of {fi_slope:.2f} Hz/nA (R^2={fi_r2:.3f})" if fi_slope is not None and fi_r2 is not None
                    else "that is not available (no tonic points to fit)")
    caption = _wrap(
       f"Every panel below is computed from the same {cell_result['n_points_total']}-point grid of held "
       "and injected current levels: which pattern each point fires in and whether it rebounds after "
       "release from hyperpolarization (top row), firing rate and burst structure (second row), rebound "
       "spike count, latency, and peak voltage (third row), and sag depth, adaptation, and burst size "
       f"(bottom row). Two single-number summaries condense the whole grid: a burstiness index of "
       f"{burstiness_str} (the fraction of points classified bursting) and a firing-rate-vs-current slope "
       f"{fi_slope_str}. The rest of this packet does not add new results -- it "
       "re-derives individual pieces of this same grid from real simulated traces so each one can be "
       "checked by eye against the panel it feeds.",
       width=155)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8.5)
    return fig


# ---------------------------------------------------------------------------
# Representative traces
# ---------------------------------------------------------------------------

def make_representative_traces_page(cell_id, cell_result, resim) -> plt.Figure:
    """One compact trace per distinct (test_pattern, rebound_pattern)
    combination actually present in this cell's grid -- the same selection
    plot_example_traces.py uses for its own standalone per-combination
    figures (select_exemplar_points, most-prominent point per combination
    via _prominence_score), just laid out here as a single overview page
    instead of one file per combination. "Representative" means "actually
    observed across the full grid," not hand-picked to look illustrative.

    Every panel shares the same V and time axis limits (computed from the
    actual min/max/duration across all panels, not a fixed guess) --
    matplotlib's default is to auto-scale each subplot to its own data,
    which independently zooms a near-flat subthreshold trace to fill the
    same panel height as a real 90mV action potential. That made every
    combination look equally "spiky" regardless of real amplitude --
    exactly backwards for a page whose point is to show what actually
    differs between combinations. Confirmed necessary directly: before this,
    the "silent/none" panel (a ~0.35mV wobble) was visually indistinguishable
    in prominence from "tonic/tonic_rebound" (a real ~90mV spike train) at
    each panel's own auto-scaled zoom.
    """
    exemplars = select_exemplar_points(cell_result)
    combos = sorted(exemplars.keys())
    n = len(combos)
    # Balanced/square-ish grid (e.g. 3x3 for n=9), not a fixed 5-wide strip
    # that leaves an oddly short, visually lopsided last row (5-then-4 for
    # n=9, user-flagged 2026-08-14) -- ncols grows only as fast as needed to
    # keep the grid roughly as wide as it is tall.
    ncols = max(1, int(np.ceil(np.sqrt(n)))) if n else 1
    nrows = -(-n // ncols) if n else 1

    # First pass: resimulate every exemplar and collect its offset traces,
    # so shared axis limits can be computed BEFORE anything is drawn.
    panels = []
    v_min, v_max = np.inf, -np.inf
    duration_max = 0.0
    for combo in combos:
        test_pattern, rebound_pattern = combo
        held_nA, injected_nA = exemplars[combo][0]
        tr = resim(held_nA, injected_nA)
        if tr.get("blew_up"):
            panels.append((combo, held_nA, injected_nA, None))
            continue

        t_hold, v_hold = tr["_trace_t_hold_ms"], tr["_trace_v_hold_mV"]
        t_test, v_test = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        t_rec, v_rec = tr["_trace_t_rec_ms"], tr["_trace_v_rec_mV"]
        dt_ms = t_hold[1] - t_hold[0] if len(t_hold) > 1 else 0.1
        hold_end = t_hold[-1] + dt_ms if len(t_hold) else 0.0
        t_test_off = t_test + hold_end
        test_end = t_test_off[-1] + dt_ms if len(t_test_off) else hold_end
        t_rec_off = t_rec + test_end
        rec_end = t_rec_off[-1] if len(t_rec_off) else test_end

        v_min = min(v_min, v_hold.min(), v_test.min(), v_rec.min())
        v_max = max(v_max, v_hold.max(), v_test.max(), v_rec.max())
        duration_max = max(duration_max, rec_end)
        panels.append((combo, held_nA, injected_nA,
                       (t_hold, v_hold, t_test_off, v_test, t_rec_off, v_rec, hold_end, test_end)))

    v_pad = 0.05 * (v_max - v_min) if v_max > v_min else 1.0
    ylim = (v_min - v_pad, v_max + v_pad)
    xlim = (0.0, duration_max)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = list(axes.flat)

    for ax, (combo, held_nA, injected_nA, trace_data) in zip(axes_flat, panels):
        test_pattern, rebound_pattern = combo
        if trace_data is None:
            ax.text(0.5, 0.5, "simulation failed", ha="center", va="center", transform=ax.transAxes,
                   fontsize=8)
            ax.set_title(f"{_describe_pattern(test_pattern)}, {_describe_pattern(rebound_pattern)}",
                        fontsize=7.5)
            continue
        t_hold, v_hold, t_test_off, v_test, t_rec_off, v_rec, hold_end, test_end = trace_data

        ax.plot(t_hold, v_hold, color="gray", lw=0.6)
        ax.plot(t_test_off, v_test, color="firebrick", lw=0.6)
        ax.plot(t_rec_off, v_rec, color="steelblue", lw=0.6)
        ax.axvline(hold_end, color="black", ls=":", lw=0.6)
        ax.axvline(test_end, color="black", ls=":", lw=0.6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{_describe_pattern(test_pattern)}, then {_describe_pattern(rebound_pattern)}\n"
                    f"held={held_nA:.2f} nA, injected={injected_nA:.2f} nA", fontsize=7.5)
        ax.tick_params(labelsize=6)

    for ax in axes_flat[n:]:
        ax.axis("off")

    caption = (
       "One representative trace for each distinct combination of response-to-current-step pattern "
       f"and rebound pattern observed across this cell's full {cell_result['n_points_total']}-"
       "point grid of held and injected current levels: gray is the held baseline before the current "
       "step, red is the response during the current step, and blue is the recovery period after "
       "release back to the held level. Each combination is represented by its most clearly-expressed "
       "example among the grid points that produced it. Every panel "
       f"shares the same voltage ({ylim[0]:.0f} to {ylim[1]:.0f} mV) and time (0-{xlim[1]:.0f} ms) axes "
       "so amplitude and duration are directly comparable by eye -- without this, a small subthreshold "
       "wobble and a full-amplitude spike train would each be independently rescaled to fill the same "
       "panel and look equally prominent.")
    fig.suptitle("2. Representative traces", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    bottom_left = axes_flat[(nrows - 1) * ncols]
    bottom_right = axes_flat[-1]
    _place_caption_below(fig, bottom_left, caption, gap=0.06, right_ax=bottom_right)
    return fig


def make_example_trace_detail_pages(cell_id, cell_result, resim) -> list:
    """One full-detail page per (test_pattern, rebound_pattern) combination
    -- same exemplar selection as make_representative_traces_page's compact
    overview grid, same underlying figure plot_example_traces.py used to
    write as a standalone file per combination (build_example_trace_figure).
    Folded into the packet itself instead of a separate figures/
    example_traces/{cell_id}/ directory of loose files: everything a PI
    needs to see lives in the one PDF, not scattered across the repo, and a
    standalone copy risks going stale exactly the way one did (last
    regenerated 2026-08-10, silently out of date through several
    classification fixes landed after that).
    """
    exemplars = select_exemplar_points(cell_result)
    pages = []
    for (test_pattern, rebound_pattern), (key, _point) in sorted(exemplars.items()):
        held_nA, injected_nA = key
        tr = resim(held_nA, injected_nA)
        if tr.get("blew_up"):
            continue
        fig = build_example_trace_figure(cell_id, held_nA, injected_nA, test_pattern, rebound_pattern, tr)
        pages.append((f"{test_pattern}/{rebound_pattern}", fig))
    return pages


# ---------------------------------------------------------------------------
# Spike detection + prominence sensitivity
# ---------------------------------------------------------------------------

def make_spike_detection_page(cell_id, held_inj_pairs, resim) -> plt.Figure:
    fig, axes = plt.subplots(len(held_inj_pairs), 1, figsize=(11, 4.2 * len(held_inj_pairs)))
    if len(held_inj_pairs) == 1:
        axes = [axes]
    captions = []
    for ax, (held_nA, injected_nA, tag) in zip(axes, held_inj_pairs):
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        ax.plot(t, v, color="firebrick", lw=0.7)
        _, caption = mark_spikes(ax, t, v)
        ax.set_title(f"{cell_id} held={held_nA:.2f} inj={injected_nA:.2f} ({tag})", fontsize=9)
        ax.set_ylabel("V (mV)")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("time (ms)")
        captions.append((ax, caption))
    fig.suptitle("3. Spike detection", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.subplots_adjust(hspace=0.9)
    for ax, caption in captions:
        _place_caption_below(fig, ax, caption, width=95)
    return fig


def make_prominence_sensitivity_page(cell_id, held_inj_pairs, resim) -> plt.Figure:
    from scipy.signal import find_peaks
    fractions = np.linspace(0.10, 0.55, 10)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for held_nA, injected_nA, tag in held_inj_pairs:
        tr = resim(held_nA, injected_nA)
        v = tr["_trace_v_test_mV"]
        v_range = v.max() - v.min()
        counts = []
        for frac in fractions:
            if v_range < FLATLINE_MV:
                counts.append(0)
                continue
            # max(..., MIN_SPIKE_AMPLITUDE_MV), matching _spike_prominence --
            # this sweep has to apply the same absolute floor production
            # does, or it would show the pre-fix, floor-less sensitivity
            # curve instead of what the pipeline actually computes now.
            peaks, _ = find_peaks(v, prominence=max(v_range * frac, MIN_SPIKE_AMPLITUDE_MV))
            counts.append(len(peaks))
        line, = ax.plot(fractions, counts, marker="o", markersize=4, label=f"{tag} (held={held_nA:.2f} nA)")
        # State the finding directly on the plot -- a flat line of dots
        # reads as "nothing happened" at a glance unless the reader is told
        # what to look for; label each trace with its own constant count so
        # the plateau's meaning doesn't depend on reading the caption first
        # (user-flagged 2026-08-14).
        if counts:
            ax.annotate(f"count stays at {counts[0]} across the whole sweep", xy=(fractions[-1], counts[-1]),
                       xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                       fontsize=8, color=line.get_color())
    ax.axvline(PROMINENCE_FRACTION, color="gray", ls="--", lw=1.2,
              label=f"threshold used throughout this report ({PROMINENCE_FRACTION:.0%})")
    ax.set_xlim(fractions[0], fractions[-1] + 0.14)
    ax.set_xlabel("peak-detection threshold (fraction of each trace's own voltage range)")
    ax.set_ylabel("detected spike count")
    ax.legend(loc="center right", fontsize=8)
    ax.set_title(f"{cell_id} -- spike count is unchanged across a 5x range of detection thresholds",
                fontsize=10)
    caption = _wrap(
       "Every spike count, interspike interval, and burst/tonic classification in this report depends "
       f"on one detection threshold: a candidate peak must rise at least {PROMINENCE_FRACTION:.0%} of "
       f"the trace's own voltage range above its surroundings (or {MIN_SPIKE_AMPLITUDE_MV:.0f} mV, "
       "whichever is larger, so a tiny subthreshold wobble in a low-amplitude trace cannot be counted "
       "as a spike). This sweeps that threshold from 10% to 55% on real, large-amplitude spiking traces: "
       "a flat plateau around the value used throughout this report shows the reported spike counts are "
       "insensitive to the exact threshold chosen.",
       width=105)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.20, 1, 0.90))
    fig.suptitle("4. Spike-detection threshold sensitivity", fontsize=13, weight="bold", y=0.99)
    return fig


def make_dvdt_crossvalidation_page(cell_id, grid, resim, dt_ms, n_sample=20, seed=7) -> plt.Figure:
    import random
    rng = random.Random(seed)
    keys = [k for k, p in grid.items() if not p["blew_up"] and p["test_pattern"] in ("tonic", "bursting")]
    sample = rng.sample(keys, min(n_sample, len(keys)))

    total_confirmed, total_rejected = 0, 0
    per_point = []
    for held_nA, injected_nA in sample:
        tr = resim(held_nA, injected_nA)
        if tr.get("blew_up"):
            continue
        v = tr["_trace_v_test_mV"]
        if v.max() - v.min() < FLATLINE_MV:
            continue
        confirmed, rejected = detect_spikes_dvdt_confirmed(v, tr["_trace_t_test_ms"], dt_ms)
        total_confirmed += len(confirmed)
        total_rejected += len(rejected)
        per_point.append((held_nA, injected_nA, len(confirmed), len(rejected)))

    fig, (ax_ex, ax_bar) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    if per_point:
        held_nA, injected_nA, *_ = max(per_point, key=lambda r: r[3])  # example with most disagreement, if any
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        ax_ex.plot(t, v, color="gray", lw=0.6)
        # show a shorter time window for the exemplary trace (0-300 ms)
        try:
            ax_ex.set_xlim(0, 300)
        except Exception:
            pass
        _, ex_caption = mark_confirmed_vs_rejected(ax_ex, t, v, dt_ms)
        ax_ex.set_title(f"example: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax_ex.legend(loc="upper right", fontsize=7)
        ax_ex.text(0.5, -0.18, _wrap(ex_caption, 75), transform=ax_ex.transAxes, ha="center", va="top",
                  fontsize=7)

    labels = ["confirmed", "rejected"]
    values = [total_confirmed, total_rejected]
    ax_bar.bar(labels, values, color=["seagreen", "firebrick"])
    for i, v in enumerate(values):
        ax_bar.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    agree_pct = 100.0 * total_confirmed / max(total_confirmed + total_rejected, 1)
    ax_bar.set_title(f"{len(per_point)} sampled points, {agree_pct:.1f}% of candidate spikes confirmed",
                    fontsize=9)
    ax_bar.set_ylabel("spike count (summed across sample)")

    caption = _wrap(
       f"As an independent check on a random sample of {len(per_point)} grid points from this cell, "
       "every spike detected by amplitude alone was additionally required to show a fast "
       "upstroke in voltage (rather than amplitude alone) before being counted, an independent "
       "detection criterion not used for classification elsewhere in this report. Agreement close to "
       "100% shows that the simpler amplitude-based detector used throughout is not missing any spikes "
       "that a shape-based check would catch.", width=115)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("5. Independent cross-check of spike detection by upstroke shape",
               fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Burst/tonic classification
# ---------------------------------------------------------------------------

def make_burst_classification_page(cell_id, exemplars, resim, run_args) -> plt.Figure:
    """exemplars: list of (held_nA, injected_nA, tag) -- typically one clean
    bursting example and one boundary/near-miss "correctly rejected" example.

    Includes the raw voltage trace (with detected spikes marked) alongside
    the ISI/KDE evidence, not just the derived ISI sequence -- a reader
    shouldn't have to flip back to section 1 to see what trace a given
    classification call is even about.

    Each row's log-ISI panel is scaled to its OWN data range, not a range
    shared across rows: the near-miss/tonic example's intervals cluster in a
    ~0.7ms band, which a range wide enough to also fit the bursting
    example's 500ms+ between-burst gaps compresses into an unreadable
    sliver (user-flagged 2026-08-14; the same fix is applied in
    make_bimodality_teaching_page for the same reason).
    """
    fig, axes = plt.subplots(len(exemplars), 3, figsize=(19, 6.6 * len(exemplars)))
    if len(exemplars) == 1:
        axes = axes[np.newaxis, :]
    traces = [resim(held_nA, injected_nA) for held_nA, injected_nA, _ in exemplars]
    captions = []
    for row, (held_nA, injected_nA, tag), tr in zip(axes, exemplars, traces):
        ax_v, ax_isi, ax_kde = row
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        ax_v.plot(t, v, color="firebrick", lw=0.8, zorder=1)
        peaks, _ = mark_spikes(ax_v, t, v)
        ax_v.set_xlabel("time (ms)")
        ax_v.set_ylabel("V (mV)")
        ax_v.legend(loc="upper right", fontsize=7)
        result, caption = mark_isi_classification(
            ax_isi, ax_kde, t, v, run_args["min_isis_for_burst_test"],
            run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"],
            min_ashman_d=run_args.get("min_ashman_d", 2.0))
        # A train with far more spikes than this renders as a solid block
        # of color rather than distinguishable spikes -- crop the DISPLAYED
        # window to its first DISPLAY_WINDOW_MS only (user-flagged
        # 2026-08-14: the near-miss/tonic example fires 179 times in the
        # same window the bursting example fires 11, so the two rows won't
        # share a time axis here even though they happen to share a window
        # length). The classification itself -- caption text, KDE curve,
        # separation score -- is still computed from the full train; this
        # only narrows what's drawn, not what's measured.
        _crop_display_window(ax_v, ax_isi, t, peaks)
        ax_v.set_title(f"{tag} (held={held_nA:.2f} nA, injected={injected_nA:.2f} nA): "
                      f"classified as {_describe_pattern(result['pattern'])}", fontsize=9, wrap=True)
        captions.append((ax_v, ax_kde, caption))
    fig.suptitle("6. Tonic / bursting / silent classification", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.subplots_adjust(hspace=0.85)
    for ax_v, ax_kde, caption in captions:
        _place_caption_below(fig, ax_v, caption, width=140, fontsize=10, gap=0.05, right_ax=ax_kde)
    return fig


def _synthetic_ashman_d_near_miss():
    """Same construction, same seed, same parameters as tests/
    test_burst_detection.py's test_high_variance_cluster_rejected_by_
    ashman_d_alone -- a 3-spike-burst train where a minority of inter-burst
    ISIs are drawn from a much wider, farther-out distribution. The ratio
    and spikes-per-burst gates both pass (short/long cluster means are
    still >=1.5x apart, still >=1.5 spikes/burst), but the long cluster's
    tail-inflated spread pulls Ashman's D to ~1.75 -- below the min_ashman_d
    =2.0 default -- so classify_burst_pattern correctly rejects it as
    tonic. No real cell in this project's current grid data has ever
    exercised this specific gate (see the 2026-08 audit: 0/2513 points
    across the 6-cell working set), so this panel is a CONSTRUCTED example
    illustrating what the gate is for, not an observed case.
    """
    rng = np.random.default_rng(11)
    mean_short, sd_short, mean_long, sd_long = 10.0, 0.5, 20.0, 2.0
    tail_frac, tail_mean, tail_sd = 0.1, 55.0, 6.0
    n_bursts, spikes_per_burst = 60, 3
    seq = []
    for _ in range(n_bursts):
        for _ in range(spikes_per_burst - 1):
            seq.append(max(1.0, rng.normal(mean_short, sd_short)))
        if rng.random() < tail_frac:
            seq.append(max(1.0, rng.normal(tail_mean, tail_sd)))
        else:
            seq.append(max(1.0, rng.normal(mean_long, sd_long)))
    isis_ms = np.array(seq[:-1])
    return isis_ms, len(isis_ms) + 1


def make_bimodality_teaching_page(cell_id, real_exemplars, resim, run_args) -> plt.Figure:
    """Dedicated teaching figure for the log-ISI bimodality test underlying
    ALL burst/tonic classification in this pipeline: for each row, the raw
    voltage trace with detected spikes marked (so within-burst vs. between-
    burst gaps are visible as literal time gaps, not just abstract ISI
    numbers), the ISI-vs-spike-index sequence, and the log-ISI KDE panel
    with the short (within-burst) and long (between-burst) ISI populations
    explicitly labeled and Ashman's D annotated against its threshold.

    real_exemplars: list of (held_nA, injected_nA, tag) for real, resimulated
    XB2IQX points. A fourth, CONSTRUCTED row (see
    _synthetic_ashman_d_near_miss) is always appended, since no real point
    in this project's data has ever needed the Ashman's D gate specifically
    to reject it (confirmed by direct audit, 2026-08) -- shown clearly
    labeled as synthetic rather than omitted, so the gate's purpose is still
    demonstrated even though it has been inert on real data so far.

    Each row's log-ISI panel scales to its OWN data range, and each real
    row's raw trace/ISI-index panels crop to at most DISPLAY_WINDOW_MS, for
    the same reasons as make_burst_classification_page (user-flagged
    2026-08-14 for section 5, extended here to section 6): a shared x-axis
    across rows
    this different in scale and firing rate hides the tight examples'
    structure rather than showing it.
    """
    n_rows = len(real_exemplars) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(19, 6.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    traces = [resim(held_nA, injected_nA) for held_nA, injected_nA, _ in real_exemplars]
    synthetic_isis, synthetic_n_peaks = _synthetic_ashman_d_near_miss()

    captions = []
    for row, (held_nA, injected_nA, tag), tr in zip(axes, real_exemplars, traces):
        ax_v, ax_isi, ax_kde = row
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        peaks, _ = mark_spikes(ax_v, t, v)
        ax_v.plot(t, v, color="black", lw=0.6)
        ax_v.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax_v.set_xlabel("time (ms)")
        ax_v.set_ylabel("V (mV)")
        result, caption = mark_isi_classification(
            ax_isi, ax_kde, t, v, run_args["min_isis_for_burst_test"],
            run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"],
            min_ashman_d=run_args.get("min_ashman_d", 2.0))
        _crop_display_window(ax_v, ax_isi, t, peaks)
        if tag.startswith("separation-score near-miss"):
            # This row's whole point is a real ~0.1ms separation between the
            # short and long interval populations -- the default zero-
            # anchored range (see mark_isi_classification) is right for a
            # flat/uniform row but hides this row's actual structure by
            # compressing it against the top of a 0-17ms axis. User-set
            # range, 2026-08-14.
            ax_isi.set_ylim(14, 18)
        ax_isi.set_title(f"Interspike intervals -- classified as {_describe_pattern(result['pattern'])}",
                        fontsize=9)
        captions.append((ax_v, ax_kde, f"Simulated {cell_id} trace. {caption}"))

    ax_v, ax_isi, ax_kde = axes[-1]
    ax_v.axis("off")
    ax_v.text(0.5, 0.5, "Constructed example\n(a hand-built interval sequence,\nnot a simulated trace)",
             ha="center", va="center", fontsize=9, style="italic", color="dimgray",
             transform=ax_v.transAxes, wrap=True)
    result, caption = mark_isi_classification(
        ax_isi, ax_kde, None, None, run_args["min_isis_for_burst_test"],
        run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"],
        min_ashman_d=run_args.get("min_ashman_d", 2.0), isis_override=synthetic_isis,
        n_peaks_override=synthetic_n_peaks)
    # This row has no simulated trace/time axis (a hand-built ISI sequence),
    # so DISPLAY_WINDOW_MS's time-based crop doesn't apply -- kept as the
    # original fixed-spike-count crop instead, matching every other row's
    # displayed span closely enough (a ~13ms-ISI train's first 25 spikes is
    # ~325ms, in the same ballpark as DISPLAY_WINDOW_MS) without needing a
    # fabricated time axis just to reuse _crop_display_window.
    SYNTHETIC_DISPLAY_SPIKES = 25
    if synthetic_n_peaks > SYNTHETIC_DISPLAY_SPIKES:
        ax_isi.set_xlim(0.5, SYNTHETIC_DISPLAY_SPIKES - 0.5)
    ax_isi.set_title(f"Interspike intervals -- classified as {_describe_pattern(result['pattern'])}",
                    fontsize=9)
    captions.append((ax_v, ax_kde, f"This example is constructed, not an observed point from {cell_id}: it "
                    "illustrates a case where the short and long interval populations pass the simpler "
                    "gates but remain too statistically overlapping to count as separate, a "
                    "case this cell's own data has not been observed to produce. "
                    f"{caption}"))

    fig.suptitle("7. Bimodality test: within-burst vs. between-burst ISIs", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    fig.subplots_adjust(hspace=1.3)
    for ax_v, ax_kde, caption in captions:
        _place_caption_below(fig, ax_v, caption, width=140, fontsize=9, gap=0.028, right_ax=ax_kde)
    return fig


# ---------------------------------------------------------------------------
# Rebound detection
# ---------------------------------------------------------------------------

def make_rebound_page(cell_id, exemplars, resim, rebound_latency_min_ms) -> plt.Figure:
    fig, axes = plt.subplots(len(exemplars), 1, figsize=(11, 3.8 * len(exemplars)))
    if len(exemplars) == 1:
        axes = [axes]
    captions = []
    for ax, (held_nA, injected_nA, tag) in zip(axes, exemplars):
        tr = resim(held_nA, injected_nA)
        t_rec, v_rec = tr["_trace_t_rec_ms"], tr["_trace_v_rec_mV"]
        ax.plot(t_rec, v_rec, color="steelblue", lw=0.6)
        evidence, caption = mark_rebound_window(ax, t_rec, v_rec, rebound_latency_min_ms)
        ax.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("time since release (ms)")
        captions.append((ax, caption))
    fig.suptitle("9. Post-inhibitory rebound detection", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.subplots_adjust(hspace=0.9)
    for ax, caption in captions:
        _place_caption_below(fig, ax, caption, width=95)
    return fig


# ---------------------------------------------------------------------------
# Sag depth
# ---------------------------------------------------------------------------

def make_sag_page(cell_id, exemplars, resim, sag_window_ms) -> plt.Figure:
    fig, axes = plt.subplots(1, len(exemplars), figsize=(6.5 * len(exemplars), 5.5))
    if len(exemplars) == 1:
        axes = [axes]
    captions = []
    for ax, (held_nA, injected_nA, tag) in zip(axes, exemplars):
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        hold_v_end = tr["_trace_v_hold_mV"][-1]
        ax.plot(t, v, color="firebrick", lw=0.7)
        _, caption = mark_sag_trough(ax, t, v, hold_v_end, sag_window_ms)
        ax.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("V (mV)")
        ax.legend(loc="upper right", fontsize=6.5)
        captions.append((ax, caption))
    fig.suptitle("10. Sag depth", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.90))
    for ax, caption in captions:
        _place_caption_below(fig, ax, caption, width=55)
    return fig


# ---------------------------------------------------------------------------
# Adaptation ratio
# ---------------------------------------------------------------------------

def make_adaptation_page(cell_id, exemplars, resim, adaptation_edge_n) -> plt.Figure:
    """Includes the raw voltage trace alongside the derived ISI-vs-index
    plot -- the ratio is computed from ISIs, but a reader still needs to
    see the actual spikes it came from, not just the abstract sequence.
    """
    fig, axes = plt.subplots(2, len(exemplars), figsize=(6.5 * len(exemplars), 8.5), squeeze=False)
    captions = []
    for col, (held_nA, injected_nA, tag) in zip(axes.T, exemplars):
        ax_v, ax_isi = col
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        ax_v.plot(t, v, color="firebrick", lw=0.6, zorder=1)
        mark_spikes(ax_v, t, v)
        ax_v.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax_v.set_xlabel("time (ms)")
        ax_v.set_ylabel("V (mV)")
        ax_v.legend(loc="upper right", fontsize=6)
        isis_ms, _ = compute_isis_ms(v, t, PROMINENCE_FRACTION)
        ax_isi.plot(np.arange(1, len(isis_ms) + 1), isis_ms, marker="o", markersize=3, color="steelblue")
        # Anchor at zero so a tightly-clustered interval sequence doesn't get
        # auto-scaled into a visually exaggerated zigzag (see trace_
        # annotations.mark_isi_classification's identical fix).
        ax_isi.set_ylim(bottom=0)
        ratio, caption = mark_adaptation_window(ax_isi, isis_ms, adaptation_edge_n)
        ax_isi.set_xlabel("interval index (spike-to-spike)")
        ax_isi.set_ylabel("interspike interval (ms)")
        captions.append((ax_isi, caption))
    fig.suptitle("11. Spike-frequency adaptation", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    fig.subplots_adjust(hspace=0.6)
    for ax_isi, caption in captions:
        _place_caption_below(fig, ax_isi, caption, width=55)
    return fig


# ---------------------------------------------------------------------------
# F-I slope
# ---------------------------------------------------------------------------

def make_fi_slope_page(cell_id, grid, features) -> plt.Figure:
    points = [p for p in grid.values()
             if p["held_nA"] == 0.0 and not p["blew_up"] and p["test_pattern"] not in (None, "silent")]
    x = np.array([p["injected_nA"] for p in points])
    y = np.array([p["test_freq_hz"] for p in points])
    order = np.argsort(x)
    x, y = x[order], y[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=25, color="steelblue", zorder=3, label="grid points (held=0)")
    slope = features.get("fi_slope_hz_per_nA")
    r2 = features.get("fi_slope_r2")
    if slope is not None and len(x) > 1:
        x_fit = np.linspace(x.min(), x.max(), 50)
        intercept = np.mean(y) - slope * np.mean(x)
        ax.plot(x_fit, slope * x_fit + intercept, color="darkorange", lw=1.5,
               label=f"linear fit: {slope:.2f} Hz/nA (R^2={r2:.3f})")
    ax.set_xlabel("injected current (nA)")
    ax.set_ylabel("firing rate (Hz)")
    ax.invert_xaxis()
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{cell_id} -- firing rate vs. injected current, no holding current", fontsize=10)
    caption = _wrap(
       f"{len(points)} non-silent points with a clearly-classified firing pattern, sampled across "
       "injected current with no holding current applied (points still below firing threshold, which "
       "would otherwise bias the linear fit toward zero, are excluded). This firing-rate/current slope "
       "is one of the features used to characterize this cell in the cross-cell comparison; here it is "
       "plotted directly against the data it was fit to.", width=110)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("12. Firing rate vs. injected current", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------

def make_self_consistency_page(cell_id, cell_result, resim, n_sample=40, seed=11) -> plt.Figure:
    """Re-simulates a random sample of cached grid points and compares the
    resulting test_pattern/rebound_pattern against what's stored in
    cell_held_injected_grid.pkl. No coarse/refined split (2026-08-13): the
    uniform-grid sweep protocol has no adaptive-refinement tier anymore --
    every held level (except held=0, which reuses the cell's own cached
    Iapp=0 state exactly) is settled via get_or_settle_hold's warm-start
    from whichever already-settled held level is numerically nearest, so
    the same path-dependent-settling caveat from the old coarse/refined
    split still applies uniformly to the whole grid rather than
    concentrating in one bucket -- there's no longer a "coarse, settled
    directly" subset to contrast it against.
    """
    import random
    rng = random.Random(seed)
    grid = cell_result["grid"]
    keys = [k for k, p in grid.items() if not p["blew_up"]]
    sample = rng.sample(keys, min(n_sample, len(keys)))

    matches = []
    for held_nA, injected_nA in sample:
        p = grid[(held_nA, injected_nA)]
        tr = resim(held_nA, injected_nA)
        match = (not tr.get("blew_up")) and (tr["test_pattern"] == p["test_pattern"]) and \
               (tr["rebound_pattern"] == p["rebound_pattern"])
        matches.append(match)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    rate = 100.0 * sum(matches) / len(matches)
    bar = ax.bar(["all sampled points"], [rate], color="steelblue")
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, rate, f"{rate:.0f}%\n(n={len(matches)})",
           ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("match rate (%)")
    ax.set_title(f"{cell_id} -- reproducibility of stored classifications under re-simulation", fontsize=10)

    caption = _wrap(
       f"{len(sample)} random points from this cell's grid were re-simulated independently and "
       f"compared against the classification stored from the original sweep. Match rate: {rate:.0f}%. "
       "Except at zero holding current, where the cell's own steady-state limit cycle is reused "
       "exactly, each holding-current level is reached by integrating forward from whichever "
       "already-settled level is numerically nearest -- this model is known to show path-dependent "
       "(hysteretic) settling near dynamical transitions, so a re-simulation that approaches a "
       "borderline point along a different path than the original sweep can occasionally land on a "
       "different classification right at that boundary.",
       width=115)
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("13. Reproducibility check", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.88))
    return fig


# ---------------------------------------------------------------------------
# Burst-onset-then-silence detection (2026-08-13)
# ---------------------------------------------------------------------------

def make_onset_burst_page(cell_id, exemplars, resim, run_args) -> plt.Figure:
    """exemplars: list of (held_nA, injected_nA, tag). Added alongside the
    fix this section verifies: adaptation_ratio and spikes_per_burst used to
    have no way to represent a window that opens with a real burst and then
    trails off into silence before the window ends -- confirmed case:
    XB2IQX held=-2.48/inj=-4.04, ISIs 13-17ms onset then an irregular
    30-112ms trickle that stops with ~2.3s of a 3s window left silent,
    which used to report test_adaptation_ratio=5.47 as if it were smooth
    tonic adaptation. detect_onset_burst (local leading-run detection) plus
    a trailing-silence/likely-ceased-firing check now catch this
    independent of the whole-window tonic/bursting label.
    """
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(13, 4.6 * len(exemplars)))
    if len(exemplars) == 1:
        axes = axes[np.newaxis, :]
    captions = []
    for row, (held_nA, injected_nA, tag) in zip(axes, exemplars):
        ax_v, ax_isi = row
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        ax_v.plot(t, v, color="firebrick", lw=0.6, zorder=1)
        evidence, caption = mark_onset_and_trailing_silence(
            ax_v, ax_isi, t, v, run_args["min_isi_ratio"],
            run_args.get("min_onset_isis", 2), run_args.get("trailing_silence_ratio", 3.0))
        ratio = tr.get("test_adaptation_ratio")
        ratio_str = "not reported (see caption)" if ratio is None else f"{ratio:.2f}"
        ax_v.set_title(f"{tag} (held={held_nA:.2f} nA, injected={injected_nA:.2f} nA): "
                      f"classified as {_describe_pattern(tr['test_pattern'])}, "
                      f"adaptation ratio {ratio_str}", fontsize=8, wrap=True)
        ax_v.set_xlabel("time (ms)")
        ax_v.set_ylabel("V (mV)")
        ax_v.legend(loc="upper right", fontsize=6)
        captions.append((ax_v, caption))
    fig.suptitle("8. Burst-onset-then-silence detection", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.subplots_adjust(hspace=1.0)
    for ax_v, caption in captions:
        _place_caption_below(fig, ax_v, caption, width=110)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_packet(cell_id: str, args) -> None:
    with open(args.grid_cache, "rb") as f:
        grid_cache = pickle.load(f)
    with open(args.grid_features_cache, "rb") as f:
        features_cache = pickle.load(f)

    cell_result = grid_cache[cell_id]
    features = features_cache[cell_id]
    grid = cell_result["grid"]
    run_args = cell_result["run_args"]
    params = cell_result["params"]

    ss_entry = get_cached_state(cell_id, params, cache_path=Path(args.steady_state_cache))
    y_ss, baseline_freq_hz = ss_entry["y_ss"], ss_entry["freq_hz"]

    def resim(held_nA, injected_nA):
        return _resim(cell_id, params, y_ss, baseline_freq_hz, cell_result, held_nA, injected_nA,
                      ss_entry=ss_entry)

    dt_ms = run_args["dt"]

    spike_pts = [(-2.892861, -2.958704, "near-boundary bursting"), (-3.093754, -3.535713, "tonic")]
    # -1.377551/-1.234694 replaces the packet's original near-miss point
    # (-2.892861, -2.958704): confirmed 2026-08-14 that the original point
    # is no longer a near-miss at all -- it now scores Ashman's D=7.27 and
    # classifies confidently "bursting" (rescued by the ratio-override gate
    # added earlier this session), so a caption calling it a "correctly
    # rejected" boundary case had gone stale. The replacement genuinely sits
    # near the D=2.0 gate (D=1.76 resimulated, matches cached D=1.75 --
    # confirmed stable, unlike several other candidates whose cached D
    # didn't survive resimulation) and is still classified "tonic".
    burst_exemplars = [(-3.21429, -3.363838, "clean bursting"),
                       (-1.377551, -1.234694, "near-miss boundary")]
    # -4.040816/-4.377551 replaces the packet's original single_spike
    # exemplar (-3.857148, -3.92857): confirmed 2026-08-13 that the original
    # point was ITSELF one of the subthreshold false positives the
    # MIN_SPIKE_AMPLITUDE_MV fix this packet's section 1/6 now verifies was
    # added to reject -- resimulating it post-fix shows rebound_occurred=
    # False, not "single_spike". After the fix XB2IQX has exactly one
    # genuine single_spike rebound point left in its whole grid; this is it
    # (48.6mV amplitude, confirmed stable under resimulation).
    rebound_exemplars = [(-4.040816, -4.377551, "single-spike rebound"),
                        (-0.321429, -3.535713, "sustained (tonic) rebound")]
    sag_exemplars = [(-3.535719, -5.5, "silent, deep sag")]
    # -4.316327/-3.255102: this cell's classifier logic changed enough times
    # this session that TWO earlier adaptation-ratio exemplars went stale in
    # turn (-3.093754/-3.535713 -> flagged test_likely_ceased_firing=True;
    # its replacement -3.765306/-3.367347 -> later tripped
    # test_oscillating_burst, a genuine 617/43/537/43/599/37ms alternation,
    # not smooth adaptation, reclassifying the whole window "bursting").
    # That second flip also surfaced a real code bug: test_adaptation_ratio
    # was computed BEFORE the oscillating-burst override ran, so it stayed
    # populated (9.51) even after test_pattern flipped -- fixed in
    # run_held_injected_grid.py's run_test_and_recovery to clear it when the
    # override fires, matching the invariant already enforced for
    # test_n_bursts/test_spikes_per_burst. This point is confirmed stable
    # under resim() (adaptation_ratio=9.64 resimulated vs. 9.89 cached,
    # test_pattern stays "tonic", not ceased, not oscillating) and shows a
    # real 5-spike onset (mean ISI 15ms) before settling, so it doubles as
    # the onset_exemplars "burst then sustained" case below.
    adapt_exemplars = [(-4.316327, -3.255102, "tonic, sustained")]
    onset_exemplars = [(-2.479592, -4.040816, "burst then ceased"),
                       (-4.316327, -3.255102, "burst then sustained")]
    # Reuses burst_exemplars' two real points (confidently bursting, D=6.33;
    # and the D-gate near-miss, correctly rejected as tonic just below the
    # D=2.0 threshold) plus a real confidently-tonic point -- picked 2026-08
    # for a clean, non-adapting, high-spike-count train (n_isis=230,
    # adaptation_ratio=1.00) so its single dominant ISI mode is unambiguous.
    bimodality_exemplars = [(-3.21429, -3.363838, "confidently bursting"),
                            (-0.091837, -0.112245, "confidently tonic"),
                            (-1.377551, -1.234694, "separation-score near-miss (rejected as tonic)")]

    # Ordered as a narrative, not by whatever order each page was originally
    # implemented in (user-flagged 2026-08-14: the old order interleaved
    # unrelated topics and split the tonic/bursting classification test
    # (section 5) from its own deep-dive and edge-case follow-ups (sections
    # 11-12), separating them with five pages on unrelated features):
    # the actual result first (every extracted-feature panel across the
    # whole grid, user-flagged 2026-08-14: "hard to show what we've proved
    # something if we don't know what's being proved") -> foundation (spike
    # detection, validated three ways) -> the classification built on top of
    # it (whole-window test, its statistical basis, then its short-train
    # edge case) -> other independent per-point features -> population-level
    # summary -> end-to-end reproducibility as a capstone check.
    sections = [
        ("Feature overview", "Every extracted-feature panel across the full current grid -- what the rest "
         "of this packet verifies."),
        ("Representative traces", "Every distinct response pattern this cell produces, then one "
         "full-detail trace per pattern."),
        ("Spike detection", "Every spike the automated detector found, marked on real traces."),
        ("Spike-detection threshold sensitivity", "How stable spike counts are across detection "
         "thresholds."),
        ("Independent cross-check of spike detection", "A second detection method compared against the "
         "primary one."),
        ("Tonic / bursting / silent classification", "Interspike intervals and the statistical test for "
         "two separate interval populations."),
        ("Bimodality test", "The full statistical-separation evidence behind the tonic/bursting "
         "distinction."),
        ("Burst-onset-then-silence detection", "A leading onset burst followed by cessation, vs. "
         "one that sustains."),
        ("Post-inhibitory rebound detection", "Rebound spikes after release from hyperpolarization, and "
         "the latency cutoff used."),
        ("Sag depth", "The baseline voltage, search window, and trough behind the sag measurement."),
        ("Spike-frequency adaptation", "Early vs. late interspike intervals on a real, sustained tonic "
         "trace."),
        ("Firing rate vs. injected current", "Firing rate vs. current, with the linear fit used to "
         "summarize it."),
        ("Reproducibility check", "How often re-simulation reproduces the original classification."),
    ]

    outdir = Path(args.figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    png_dir = outdir / cell_id
    # Cleared, not just overwritten by name: page numbering/filenames have
    # changed across sessions as sections were added, removed, or reordered
    # (most recently 2026-08-14, regrouping into a logical narrative order --
    # see the `sections` comment above), and a stale leftover from an old
    # numbering would otherwise sit next to the new set indefinitely. Same
    # reasoning already applied to just the representative_traces_detail
    # subfolder below; extended to the whole directory since this reordering
    # is exactly the scenario that logic didn't cover.
    if png_dir.exists():
        shutil.rmtree(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{cell_id}_validation_packet.pdf"

    # Every page's footer names exactly which cache(s) and which production
    # function(s) its evidence traces back to -- see _save_page's docstring
    # for why this packet specifically needs it spelled out per page, not
    # just once in the cover page's prose.
    command = "python " + " ".join(sys.argv)
    grid_cache_name = Path(args.grid_cache).name
    grid_features_cache_name = Path(args.grid_features_cache).name

    # Tracks the real number of pages saved -- sections is a fixed top-level
    # list, but section 1 now writes a variable number of detail pages (one
    # per exemplar combination actually present in the grid), so len(sections)
    # no longer equals the actual page count.
    page_count = 0

    skipped = []

    def save(fig, name, source):
        nonlocal page_count
        _save_page(fig, pdf, png_dir, name, command, f"{grid_cache_name} -> {source}")
        page_count += 1

    def try_save(build_fn, name, source):
        # Several sections (03-11) use (held, injected) exemplar coordinates
        # hand-picked and confirmed stable specifically for XB2IQX (see the
        # inline comments above where spike_pts/burst_exemplars/etc. are
        # defined) -- they are not guaranteed to land on a valid, non-blown-
        # up point in another cell's grid (confirmed directly: 8Z3ESD's
        # historical packet silently stopped at section 2 for exactly this
        # reason, since an uncaught exception here still lets `with
        # PdfPages(...)` save whatever pages were already written before
        # re-raising). Catching per-section and skipping with a logged
        # reason -- rather than either crashing the whole packet or silently
        # truncating it -- means every cell gets as much of the packet as is
        # actually valid for it, and it's visible in the printed summary
        # (and the cover page's own section list) which sections that was.
        try:
            fig = build_fn()
        except Exception as exc:
            skipped.append((name, str(exc)))
            print(f"  {cell_id}: skipped {name} ({exc})")
            return
        save(fig, name, source)

    with PdfPages(pdf_path) as pdf:
        _save_page(make_cover_page(cell_id, cell_result, features, sections), pdf, png_dir, "00_cover",
                  command, f"{grid_cache_name} + {grid_features_cache_name}")
        page_count += 1
        try_save(lambda: make_feature_overview_page(cell_id, cell_result, features), "01_feature_overview",
                f"{grid_features_cache_name} (run_held_injected_grid.build_cell_grid_features_fig(), no "
                "resimulation)")
        try_save(lambda: make_representative_traces_page(cell_id, cell_result, resim), "02_representative_traces",
                "plot_example_traces.select_exemplar_points() + resimulate_point()")
        # One full-detail page per combination, folded into the packet
        # instead of a standalone figures/example_traces/{cell_id}/
        # directory -- see make_example_trace_detail_pages' docstring for
        # why (a standalone copy of this exact content went stale for over
        # a week through several classification fixes before being caught).
        # The SET of (test_pattern, rebound_pattern) combinations actually
        # present changes as classification fixes land (confirmed directly
        # -- Option B alone dropped 11 combos to 9), which is exactly why
        # png_dir as a whole is cleared above rather than just this
        # subfolder -- a stale leftover combination would otherwise sit next
        # to the new set indefinitely.
        for i, (combo_label, detail_fig) in enumerate(
                make_example_trace_detail_pages(cell_id, cell_result, resim), start=1):
            slug = combo_label.replace("/", "-").replace(" ", "_")
            save(detail_fig, f"representative_traces_detail/{i:02d}__{slug}",
                "plot_example_traces.select_exemplar_points() + resimulate_point() -> "
                "plot_example_traces.build_example_trace_figure()")
        try_save(lambda: make_spike_detection_page(cell_id, spike_pts, resim), "03_spike_detection",
                "resimulate_point() -> trace_annotations.mark_spikes() (scipy.signal.find_peaks, same call "
                "find_silencing_threshold.compute_isis_ms uses)")
        try_save(lambda: make_prominence_sensitivity_page(cell_id, spike_pts, resim), "04_prominence_sensitivity",
                "resimulate_point() -> scipy.signal.find_peaks swept over PROMINENCE_FRACTION")
        try_save(lambda: make_dvdt_crossvalidation_page(cell_id, grid, resim, dt_ms), "05_dvdt_crossvalidation",
                "resimulate_point() -> find_silencing_threshold.detect_spikes_dvdt_confirmed()")
        try_save(lambda: make_burst_classification_page(cell_id, burst_exemplars, resim, run_args),
                "06_burst_classification", "resimulate_point() -> find_silencing_threshold.classify_burst_pattern()")
        try_save(lambda: make_bimodality_teaching_page(cell_id, bimodality_exemplars, resim, run_args),
                "07_bimodality_test", "resimulate_point() -> find_silencing_threshold.classify_burst_pattern() "
                "(4th row: constructed synthetic ISIs, no resimulation)")
        try_save(lambda: make_onset_burst_page(cell_id, onset_exemplars, resim, run_args), "08_onset_burst_detection",
                "resimulate_point() -> run_held_injected_grid.detect_onset_burst()")
        try_save(lambda: make_rebound_page(cell_id, rebound_exemplars, resim, run_args["rebound_latency_min_ms"]),
                "09_rebound_detection", "resimulate_point() -> trace_annotations.mark_rebound_window() "
                "(reproduces run_held_injected_grid.run_test_and_recovery's rebound-peak logic)")
        try_save(lambda: make_sag_page(cell_id, sag_exemplars, resim, run_args["sag_window_ms"]), "10_sag_depth",
                "resimulate_point() -> run_held_injected_grid.compute_pre_spike_sag_trough()")
        try_save(lambda: make_adaptation_page(cell_id, adapt_exemplars, resim, run_args["adaptation_edge_n"]),
                "11_adaptation_ratio", "resimulate_point() -> run_held_injected_grid.compute_adaptation_ratio()")
        try_save(lambda: make_fi_slope_page(cell_id, grid, features), "12_fi_slope",
                f"{grid_features_cache_name} (extract_grid_features.compute_fi_slope())")
        try_save(lambda: make_self_consistency_page(cell_id, cell_result, resim), "13_self_consistency",
                "cached test_pattern/rebound_pattern vs. fresh resimulate_point() results")

    skip_note = f", {len(skipped)} section(s) skipped ({', '.join(n for n, _ in skipped)})" if skipped else ""
    print(f"{cell_id}: wrote {pdf_path} and {page_count} section PNGs to {png_dir}/{skip_note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a consolidated validation packet (PDF + "
                                    "standalone section PNGs) walking through the trace-level evidence "
                                    "behind every grid-feature panel, to verify the feature-extraction "
                                    "and classification pipeline is accurate.")
    parser.add_argument("--cells", nargs="+", default=["XB2IQX"])
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH)
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH)
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for cell_id in args.cells:
        build_packet(cell_id, args)


if __name__ == "__main__":
    main()
