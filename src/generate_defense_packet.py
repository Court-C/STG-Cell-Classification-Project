"""Consolidated PI defense packet: one PDF (plus standalone per-section PNGs
for attaching individually) that walks through every grid-feature panel's
underlying detection/classification evidence on real traces, so the packet
stands on its own without narration.

Every page/section reuses the actual production functions from
find_silencing_threshold.py / run_held_injected_grid.py / extract_grid_features.py
via src/trace_annotations.py's overlay helpers -- nothing here re-implements
detection logic, so what's drawn is provably what the pipeline computed.
"""

import argparse
import pickle
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
from plot_example_traces import resimulate_point, select_exemplar_points, build_example_trace_figure
from find_silencing_threshold import (compute_isis_ms, classify_burst_pattern, count_spikes_and_rate,
                                      PROMINENCE_FRACTION, detect_spikes_dvdt_confirmed, FLATLINE_MV,
                                      MIN_SPIKE_AMPLITUDE_MV)
from extract_grid_features import (DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_FEATURES_CACHE_PATH,
                                   compute_fi_slope)
from run_held_injected_grid import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH
from trace_annotations import (mark_spikes, mark_confirmed_vs_rejected, mark_isi_classification,
                               mark_sag_trough, mark_adaptation_window, mark_rebound_window,
                               mark_onset_and_trailing_silence)

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "defense_packet"


def _wrap(text: str, width: int = 100) -> str:
    return textwrap.fill(text, width=width)


def _place_caption_below(fig, ax, caption: str, width: int = 60, fontsize: float = 7, gap: float = 0.10) -> None:
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
    """
    bbox = ax.get_position()
    fig.text(bbox.x0 + bbox.width / 2, bbox.y0 - gap, _wrap(caption, width),
             ha="center", va="top", fontsize=fontsize, transform=fig.transFigure)


def _save_page(fig, pdf: PdfPages, png_dir: Path, name: str, command: str, source: str) -> None:
    """Stamps two provenance lines on every page before saving, matching
    every other figure-generating script in this repo (run_held_injected_
    grid.py, extract_grid_features.py, consolidate_features.py, cluster_
    features.py, plot_example_traces.py all embed the exact command that
    produced the figure) -- generate_defense_packet.py was the one script
    in the pipeline that didn't, which matters more here than anywhere
    else: a PI defense packet whose own figures can't be traced back to
    the command and cache that produced them undercuts its own point.
    Bottom-left, not the bottom-center every caption already occupies, so
    the two never collide regardless of a given page's own layout.
    """
    fig.text(0.005, 0.005, f"{command}\nsource: {source}", ha="left", va="bottom",
             fontsize=5, family="monospace", color="dimgray")
    pdf.savefig(fig)
    fig.savefig(png_dir / f"{name}.png", dpi=150)
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
    fig.text(0.5, 0.93, f"Grid Features Defense Packet -- {cell_id}", fontsize=18, ha="center", weight="bold")

    intro = _wrap(
        "Every trace in this packet is SIMULATED output from a fitted conductance-based single-"
        "compartment cell model (src/singlecell_model_v1.py), integrated deterministically at a fixed "
        "timestep -- not a raw current-clamp recording. What's being defended here is the feature-"
        "extraction methodology applied on top of that model (spike detection, burst/tonic "
        "classification, rebound detection, sag depth, adaptation ratio) -- i.e. do these algorithms "
        "correctly characterize the dynamics the model actually produces. Every panel below reruns the "
        "exact production function on a real simulated trace and marks what it found, so the "
        "classification isn't just a text label to take on faith.", width=95)
    fig.text(0.08, 0.87, intro, fontsize=10, va="top", wrap=True)

    summary = (f"cell_floor = {cell_result['cell_floor_nA']:.2f} nA   |   "
              f"n grid points = {cell_result['n_points_total']}   |   "
              f"burstiness index = {features.get('burstiness_index'):.3f}   |   "
              f"F-I slope = {features.get('fi_slope_hz_per_nA'):.2f} Hz/nA "
              f"(R^2={features.get('fi_slope_r2'):.3f})")
    fig.text(0.08, 0.60, summary, fontsize=9, va="top", family="monospace")

    fig.text(0.08, 0.53, "Contents:", fontsize=11, va="top", weight="bold")
    y = 0.49
    # 0.41, not the full 0.47 down to y=0.02: reserves room above the
    # provenance footer _save_page stamps at y=0.005 on every page
    # (including this one) -- confirmed directly that the previous 0.47
    # span let an 11-section list's last entry visually collide with it.
    step = 0.41 / len(sections)
    for i, (title, desc) in enumerate(sections, start=1):
        fig.text(0.10, y, f"{i}. {title}", fontsize=9.5, va="top", weight="bold")
        fig.text(0.14, y - 0.018, _wrap(desc, width=95), fontsize=7.5, va="top", color="dimgray")
        y -= step

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
    exemplars = select_exemplar_points(cell_result["grid"])
    combos = sorted(exemplars.keys())
    n = len(combos)
    ncols = min(5, n) or 1
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
            ax.text(0.5, 0.5, "resim failed", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{test_pattern} / {rebound_pattern}", fontsize=7.5)
            continue
        t_hold, v_hold, t_test_off, v_test, t_rec_off, v_rec, hold_end, test_end = trace_data

        ax.plot(t_hold, v_hold, color="gray", lw=0.6)
        ax.plot(t_test_off, v_test, color="firebrick", lw=0.6)
        ax.plot(t_rec_off, v_rec, color="steelblue", lw=0.6)
        ax.axvline(hold_end, color="black", ls=":", lw=0.6)
        ax.axvline(test_end, color="black", ls=":", lw=0.6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{test_pattern} / {rebound_pattern}\nheld={held_nA:.2f} inj={injected_nA:.2f}",
                    fontsize=7.5)
        ax.tick_params(labelsize=6)

    for ax in axes_flat[n:]:
        ax.axis("off")

    caption = _wrap(
       "One representative trace per distinct (test-window pattern, rebound pattern) combination "
       f"actually present across this cell's full {cell_result['n_points_total']}-point held x injected "
       "grid: gray = held baseline, red = test window (injected current), blue = recovery window "
       "(released back to held). Chosen by the same selection plot_example_traces.py uses for its own "
       "per-combination figures (most-prominent point, see that script's _prominence_score) -- this is "
       "what the cell actually does at each combination, not a hand-picked illustration. Every panel "
       f"shares the same V ({ylim[0]:.0f} to {ylim[1]:.0f} mV) and time (0-{xlim[1]:.0f} ms) axes, computed "
       "from the actual data, so amplitude and duration are directly comparable across panels by eye -- "
       "not independently auto-scaled, which would make a subthreshold wobble look as prominent as a real "
       "spike train.", width=175)
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("1. Representative traces", fontsize=13, weight="bold")
    # rect bottom raised to 0.11 (was 0.05): the added axis-standardization
    # sentence makes this a 4-5 line caption now, not 3 -- confirmed
    # directly that 0.05 let it overlap the bottom row's own x-axis tick
    # labels.
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
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
    exemplars = select_exemplar_points(cell_result["grid"])
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
    fig.suptitle("2. Spike detection", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.subplots_adjust(hspace=0.9)
    for ax, caption in captions:
        _place_caption_below(fig, ax, caption, width=95)
    return fig


def make_prominence_sensitivity_page(cell_id, held_inj_pairs, resim) -> plt.Figure:
    from scipy.signal import find_peaks
    fractions = np.linspace(0.10, 0.55, 10)
    fig, ax = plt.subplots(figsize=(9, 5))
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
        ax.plot(fractions, counts, marker="o", markersize=4, label=f"{tag} (held={held_nA:.2f})")
    ax.axvline(PROMINENCE_FRACTION, color="gray", ls="--", lw=1.2,
              label=f"production value ({PROMINENCE_FRACTION})")
    ax.set_xlabel("PROMINENCE_FRACTION")
    ax.set_ylabel("detected spike count")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{cell_id} -- spike count vs. prominence threshold", fontsize=10)
    caption = _wrap(
       "PROMINENCE_FRACTION=0.3 is the single most load-bearing, least-independently-calibrated "
       "constant in the pipeline -- every spike count, ISI, and burst/tonic call across the codebase "
       f"traces back to one find_peaks(..., prominence=max(range*0.3, {MIN_SPIKE_AMPLITUDE_MV:.0f}mV)) "
       "call (the absolute floor added 2026-08-13, see section 1's representative traces for why). This "
       "sweeps the fraction from 0.10 to 0.55 on real, large-amplitude spiking traces (the floor rarely "
       "binds here, since these traces' own range x0.3 already exceeds it) -- a flat plateau around the "
       "production value means the spike count this pipeline reports is not sensitive to the exact "
       "fraction chosen, not an arbitrary cliff-edge pick.", width=105)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.suptitle("3. Prominence-threshold sensitivity", fontsize=13, weight="bold", y=1.0)
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
       f"Cross-check across a random sample of {len(per_point)} XB2IQX grid points: every candidate "
       "prominence-based peak is additionally required to pass a dV/dt shape-confirmation gate "
       "(detect_spikes_dvdt_confirmed -- present in the codebase but not called by the production "
       "pipeline). Agreement close to 100% means the simpler production detector isn't missing a shape "
       "check that changes the answer.", width=115)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("4. dV/dt cross-validation (unused-in-production detector as a sanity check)",
               fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Burst/tonic classification
# ---------------------------------------------------------------------------

def _shared_log_isi_xlim(isis_list) -> tuple:
    """Union, not intersection: when several log-ISI KDE panels are shown
    side by side for direct comparison, scaling each to its OWN data range
    makes them visually incomparable -- a confidently-tonic point's ISIs
    might span a fraction of a log10-unit while a confidently-bursting
    point's span two full units, and both would render as similarly-sized
    curves if each panel optimizes its own axis. Computes the same padded
    range classify_burst_pattern's own grid uses (0.15x the data's own
    span) per array, then takes the union across every array being
    compared, so the shared range is exactly what the union of the panels'
    own individual ranges would have been, not a fixed/arbitrary window.
    """
    los, his = [], []
    for isis_ms in isis_list:
        if len(isis_ms) < 2:
            continue
        log_isi = np.log10(isis_ms)
        if np.ptp(log_isi) < 1e-6:
            los.append(log_isi.min() - 0.5)
            his.append(log_isi.max() + 0.5)
            continue
        pad = 0.15 * np.ptp(log_isi)
        los.append(log_isi.min() - pad)
        his.append(log_isi.max() + pad)
    return (min(los), max(his))


def make_burst_classification_page(cell_id, exemplars, resim, run_args) -> plt.Figure:
    """exemplars: list of (held_nA, injected_nA, tag) -- typically one clean
    bursting example and one boundary/near-miss "correctly rejected" example.
    """
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(13, 5.8 * len(exemplars)))
    if len(exemplars) == 1:
        axes = axes[np.newaxis, :]
    traces = [resim(held_nA, injected_nA) for held_nA, injected_nA, _ in exemplars]
    isis_list = [compute_isis_ms(tr["_trace_v_test_mV"], tr["_trace_t_test_ms"], PROMINENCE_FRACTION)[0]
                for tr in traces]
    shared_xlim = _shared_log_isi_xlim(isis_list)
    captions = []
    for row, (held_nA, injected_nA, tag), tr in zip(axes, exemplars, traces):
        ax_isi, ax_kde = row
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        result, caption = mark_isi_classification(
            ax_isi, ax_kde, t, v, run_args["min_isis_for_burst_test"],
            run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"],
            min_ashman_d=run_args.get("min_ashman_d", 2.0), log_isi_xlim=shared_xlim)
        ax_isi.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f} -> '{result['pattern']}'",
                        fontsize=9)
        captions.append((ax_isi, caption))
    fig.suptitle("5. Tonic / bursting / silent classification", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.subplots_adjust(hspace=0.9)
    for ax_isi, caption in captions:
        _place_caption_below(fig, ax_isi, caption, width=58)
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
    """
    n_rows = len(real_exemplars) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # First pass: gather every row's ISIs (resimulating the real points, no
    # plotting yet) so the shared log-ISI range can be computed as the union
    # across ALL rows before any panel is drawn -- see _shared_log_isi_xlim.
    traces = [resim(held_nA, injected_nA) for held_nA, injected_nA, _ in real_exemplars]
    real_isis = [compute_isis_ms(tr["_trace_v_test_mV"], tr["_trace_t_test_ms"], PROMINENCE_FRACTION)[0]
                for tr in traces]
    synthetic_isis, synthetic_n_peaks = _synthetic_ashman_d_near_miss()
    shared_xlim = _shared_log_isi_xlim(real_isis + [synthetic_isis])

    captions = []
    for row, (held_nA, injected_nA, tag), tr in zip(axes, real_exemplars, traces):
        ax_v, ax_isi, ax_kde = row
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        mark_spikes(ax_v, t, v)
        ax_v.plot(t, v, color="black", lw=0.6)
        ax_v.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax_v.set_xlabel("time (ms)")
        ax_v.set_ylabel("V (mV)")
        result, caption = mark_isi_classification(
            ax_isi, ax_kde, t, v, run_args["min_isis_for_burst_test"],
            run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"],
            min_ashman_d=run_args.get("min_ashman_d", 2.0), log_isi_xlim=shared_xlim)
        ax_isi.set_title(f"ISI sequence -> '{result['pattern']}'", fontsize=9)
        captions.append((ax_isi, f"[REAL, resimulated] {caption}"))

    ax_v, ax_isi, ax_kde = axes[-1]
    ax_v.axis("off")
    ax_v.text(0.5, 0.5, "CONSTRUCTED example\n(no simulated voltage trace --\nhand-built ISI sequence)",
             ha="center", va="center", fontsize=9, style="italic", color="dimgray",
             transform=ax_v.transAxes, wrap=True)
    result, caption = mark_isi_classification(
        ax_isi, ax_kde, None, None, run_args["min_isis_for_burst_test"],
        run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"],
        min_ashman_d=run_args.get("min_ashman_d", 2.0), isis_override=synthetic_isis,
        n_peaks_override=synthetic_n_peaks, log_isi_xlim=shared_xlim)
    ax_isi.set_title(f"ISI sequence -> '{result['pattern']}'", fontsize=9)
    captions.append((ax_isi, f"[CONSTRUCTED, not an observed {cell_id} point -- same generator as "
                    f"tests/test_burst_detection.py's Ashman's D boundary test] {caption}"))

    fig.suptitle("12. Bimodality test: within-burst vs. between-burst ISIs", fontsize=13, weight="bold")
    # Both hspace and _place_caption_below's gap are figure-fraction
    # quantities, not per-row -- a gap tuned for a 2-row page (see
    # make_burst_classification_page) eats a much bigger slice of a single
    # row's vertical budget here with 4 rows, which is what caused captions
    # to collide with the NEXT row's title before this was tuned down.
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.subplots_adjust(hspace=1.5)
    for ax_isi, caption in captions:
        _place_caption_below(fig, ax_isi, caption, width=95, gap=0.045)
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
    fig.suptitle("6. Rebound (post-inhibitory) detection", fontsize=13, weight="bold")
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
    fig.suptitle("7. Sag depth", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.90))
    for ax, caption in captions:
        _place_caption_below(fig, ax, caption, width=55)
    return fig


# ---------------------------------------------------------------------------
# Adaptation ratio
# ---------------------------------------------------------------------------

def make_adaptation_page(cell_id, exemplars, resim, adaptation_edge_n) -> plt.Figure:
    fig, axes = plt.subplots(1, len(exemplars), figsize=(6.5 * len(exemplars), 5.5))
    if len(exemplars) == 1:
        axes = [axes]
    captions = []
    for ax, (held_nA, injected_nA, tag) in zip(axes, exemplars):
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        isis_ms, _ = compute_isis_ms(v, t, PROMINENCE_FRACTION)
        ax.plot(np.arange(1, len(isis_ms) + 1), isis_ms, marker="o", markersize=3, color="steelblue")
        ratio, caption = mark_adaptation_window(ax, isis_ms, adaptation_edge_n)
        ax.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f}", fontsize=9)
        ax.set_xlabel("ISI index")
        ax.set_ylabel("ISI (ms)")
        captions.append((ax, caption))
    fig.suptitle("8. Adaptation ratio", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.90))
    for ax, caption in captions:
        _place_caption_below(fig, ax, caption, width=55)
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
    ax.set_xlabel("injected (nA)")
    ax.set_ylabel("firing rate (Hz)")
    ax.invert_xaxis()
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{cell_id} -- F-I relationship along held=0", fontsize=10)
    caption = _wrap(
       f"{len(points)} non-silent, confidently-classified points along the held=0 column "
       "(compute_fi_slope's own filter -- flat zero-rate points below rheobase-equivalent excluded so "
       "they don't bias the linear fit). This scalar (fi_slope_hz_per_nA) already feeds the cross-cell "
       "feature table; this is simply the first time it's been plotted against the data it was fit to.",
       width=110)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("9. Firing rate / F-I slope", fontsize=13, weight="bold")
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rate = 100.0 * sum(matches) / len(matches)
    bar = ax.bar(["all sampled points"], [rate], color="steelblue")
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, rate, f"{rate:.0f}%\n(n={len(matches)})",
           ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("match rate (%)")
    ax.set_title(f"{cell_id} -- resimulated vs. cached classification match rate", fontsize=10)

    caption = _wrap(
       f"Re-simulates {len(sample)} random cached grid points end-to-end and checks whether the "
       f"resulting test_pattern/rebound_pattern still matches what's stored in "
       f"cell_held_injected_grid.pkl. Match rate: {rate:.0f}%. Every held level in the current "
       "uniform-grid sweep (except held=0, which reuses the cell's own cached limit-cycle state "
       "exactly) is settled via a warm-start from whichever already-settled held level is "
       "numerically nearest -- the module docstring in run_held_injected_grid.py already documents "
       "that this model shows path-dependent (hysteretic) settling near dynamical transitions, so a "
       "resimulation that warm-starts from a different already-settled state than the original sweep "
       "used can land on a different classification near a boundary. Any mismatches here are "
       "consistent with that known property, not a new bug.", width=115)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("10. Self-consistency check", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Burst-onset-then-silence detection (2026-08-13)
# ---------------------------------------------------------------------------

def make_onset_burst_page(cell_id, exemplars, resim, run_args) -> plt.Figure:
    """exemplars: list of (held_nA, injected_nA, tag). Added alongside the
    fix this section defends: adaptation_ratio and spikes_per_burst used to
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
        ratio_str = "None (suppressed)" if ratio is None else f"{ratio:.2f}"
        ax_v.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f} -> "
                      f"pattern='{tr['test_pattern']}', adaptation_ratio={ratio_str}",
                      fontsize=8.5)
        ax_v.set_xlabel("time (ms)")
        ax_v.set_ylabel("V (mV)")
        ax_v.legend(loc="upper right", fontsize=6)
        captions.append((ax_v, caption))
    fig.suptitle("11. Burst-onset-then-silence detection", fontsize=13, weight="bold")
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
    burst_exemplars = [(-3.21429, -3.363838, "clean bursting"), (-2.892861, -2.958704, "near-miss boundary")]
    # -4.040816/-4.377551 replaces the packet's original single_spike
    # exemplar (-3.857148, -3.92857): confirmed 2026-08-13 that the original
    # point was ITSELF one of the subthreshold false positives the
    # MIN_SPIKE_AMPLITUDE_MV fix this packet's section 1/6 now defends was
    # added to reject -- resimulating it post-fix shows rebound_occurred=
    # False, not "single_spike". After the fix XB2IQX has exactly one
    # genuine single_spike rebound point left in its whole grid; this is it
    # (48.6mV amplitude, confirmed stable under resimulation).
    rebound_exemplars = [(-4.040816, -4.377551, "single_spike"), (-0.321429, -3.535713, "tonic_rebound")]
    sag_exemplars = [(-3.535719, -5.5, "silent, deep sag")]
    # -3.765306/-3.367347 replaces the packet's original adaptation-ratio
    # exemplar (-3.093754, -3.535713): confirmed 2026-08-13 that the
    # original point is itself flagged test_likely_ceased_firing=True under
    # the current pipeline, so production actually reports
    # test_adaptation_ratio=None there now -- keeping it would have shown
    # this section computing a ratio (1.22) the pipeline no longer reports
    # for that exact point. mark_adaptation_window (unlike production) has
    # no ceased-firing gate of its own -- it always reports whatever
    # compute_adaptation_ratio returns given enough ISIs -- so the exemplar
    # itself has to already be a genuinely sustained point, and confirmed
    # stable specifically under resim()'s own single-hop warm-start (not
    # just the cached grid value -- see the onset_exemplars comment below
    # for a point where those two disagree).
    adapt_exemplars = [(-3.765306, -3.367347, "tonic, sustained")]
    # -3.765306/-3.367347, not -2.387755/-3.367347: the latter's CACHED grid
    # entry shows likely_ceased_firing=False (adaptation_ratio=17.51), but
    # confirmed 2026-08-13 that resimulate_point's own single-hop warm-start
    # doesn't reproduce that -- it lands on trailing_silence=571ms and
    # ceased=True instead, exactly the path-dependent-settling mismatch
    # section 9 documents. Since this page plots resim()'s own trace/values
    # (not the cached ones), the exemplar needs to be stable under
    # resimulation too -- confirmed directly this one is (ratio 9.51
    # resimulated vs. 9.51 cached).
    onset_exemplars = [(-2.479592, -4.040816, "burst then ceased"),
                       (-3.765306, -3.367347, "burst then sustained")]
    # Reuses burst_exemplars' two real points (confidently bursting, D=6.33;
    # and the ratio-gate near-miss, correctly rejected as tonic before D is
    # ever computed) plus a real confidently-tonic point -- picked 2026-08
    # for a clean, non-adapting, high-spike-count train (n_isis=230,
    # adaptation_ratio=1.00) so its single dominant ISI mode is unambiguous.
    bimodality_exemplars = [(-3.21429, -3.363838, "confidently bursting"),
                            (-0.091837, -0.112245, "confidently tonic"),
                            (-2.892861, -2.958704, "ratio-gate near-miss (rejected as tonic)")]

    sections = [
        ("Representative traces", "One overview page (compact panels), then one full-detail page per "
         "pattern/rebound combination actually observed in this cell's grid -- the same figures "
         "plot_example_traces.py would otherwise write as separate standalone files per cell."),
        ("Spike detection", "Every detected peak overlaid on real traces."),
        ("Prominence-threshold sensitivity", "How stable spike counts are across a range of thresholds."),
        ("dV/dt cross-validation", "Independent detector cross-check across a random sample."),
        ("Tonic/bursting/silent classification", "ISI sequence + log-ISI KDE evidence."),
        ("Rebound detection", "Recovery-window rebound spikes and the latency cutoff."),
        ("Sag depth", "Baseline, search window, and trough on a real trace."),
        ("Adaptation ratio", "First/last ISI windows on a real, sustained tonic trace."),
        ("Firing rate / F-I slope", "Rate vs. injected current with the fit line."),
        ("Self-consistency check", "Resimulated vs. cached classification match rate."),
        ("Burst-onset-then-silence detection", "Onset-burst detection and the adaptation-ratio "
         "ceased-firing gate, contrasted against a burst-onset that keeps firing."),
        ("Bimodality test", "Voltage trace, ISI sequence, and log-ISI KDE evidence for the within-burst "
         "vs. between-burst ISI split underlying every burst/tonic call, including the Ashman's D "
         "separation gate."),
    ]

    outdir = Path(args.figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    png_dir = outdir / cell_id
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{cell_id}_defense_packet.pdf"

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

    def save(fig, name, source):
        nonlocal page_count
        _save_page(fig, pdf, png_dir, name, command, f"{grid_cache_name} -> {source}")
        page_count += 1

    with PdfPages(pdf_path) as pdf:
        _save_page(make_cover_page(cell_id, cell_result, features, sections), pdf, png_dir, "00_cover",
                  command, f"{grid_cache_name} + {grid_features_cache_name}")
        page_count += 1
        save(make_representative_traces_page(cell_id, cell_result, resim), "01_representative_traces",
            "plot_example_traces.select_exemplar_points() + resimulate_point()")
        # One full-detail page per combination, folded into the packet
        # instead of a standalone figures/example_traces/{cell_id}/
        # directory -- see make_example_trace_detail_pages' docstring for
        # why (a standalone copy of this exact content went stale for over
        # a week through several classification fixes before being caught).
        for combo_label, detail_fig in make_example_trace_detail_pages(cell_id, cell_result, resim):
            slug = combo_label.replace("/", "-").replace(" ", "_")
            save(detail_fig, f"01_representative_traces_detail__{slug}",
                "plot_example_traces.select_exemplar_points() + resimulate_point() -> "
                "plot_example_traces.build_example_trace_figure()")
        save(make_spike_detection_page(cell_id, spike_pts, resim), "02_spike_detection",
            "resimulate_point() -> trace_annotations.mark_spikes() (scipy.signal.find_peaks, same call "
            "find_silencing_threshold.compute_isis_ms uses)")
        save(make_prominence_sensitivity_page(cell_id, spike_pts, resim), "03_prominence_sensitivity",
            "resimulate_point() -> scipy.signal.find_peaks swept over PROMINENCE_FRACTION")
        save(make_dvdt_crossvalidation_page(cell_id, grid, resim, dt_ms), "04_dvdt_crossvalidation",
            "resimulate_point() -> find_silencing_threshold.detect_spikes_dvdt_confirmed()")
        save(make_burst_classification_page(cell_id, burst_exemplars, resim, run_args),
            "05_burst_classification", "resimulate_point() -> find_silencing_threshold.classify_burst_pattern()")
        save(make_rebound_page(cell_id, rebound_exemplars, resim, run_args["rebound_latency_min_ms"]),
            "06_rebound_detection", "resimulate_point() -> trace_annotations.mark_rebound_window() "
            "(reproduces run_held_injected_grid.run_test_and_recovery's rebound-peak logic)")
        save(make_sag_page(cell_id, sag_exemplars, resim, run_args["sag_window_ms"]), "07_sag_depth",
            "resimulate_point() -> run_held_injected_grid.compute_pre_spike_sag_trough()")
        save(make_adaptation_page(cell_id, adapt_exemplars, resim, run_args["adaptation_edge_n"]),
            "08_adaptation_ratio", "resimulate_point() -> run_held_injected_grid.compute_adaptation_ratio()")
        save(make_fi_slope_page(cell_id, grid, features), "09_fi_slope",
            f"{grid_features_cache_name} (extract_grid_features.compute_fi_slope())")
        save(make_self_consistency_page(cell_id, cell_result, resim), "10_self_consistency",
            "cached test_pattern/rebound_pattern vs. fresh resimulate_point() results")
        save(make_onset_burst_page(cell_id, onset_exemplars, resim, run_args), "11_onset_burst_detection",
            "resimulate_point() -> run_held_injected_grid.detect_onset_burst()")
        save(make_bimodality_teaching_page(cell_id, bimodality_exemplars, resim, run_args),
            "12_bimodality_test", "resimulate_point() -> find_silencing_threshold.classify_burst_pattern() "
            "(4th row: constructed synthetic ISIs, no resimulation)")

    print(f"{cell_id}: wrote {pdf_path} and {page_count} section PNGs to {png_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a consolidated PI defense packet (PDF + "
                                    "standalone section PNGs) walking through the trace-level evidence "
                                    "behind every grid-feature panel.")
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
