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
from plot_example_traces import resimulate_point
from find_silencing_threshold import (compute_isis_ms, classify_burst_pattern, count_spikes_and_rate,
                                      PROMINENCE_FRACTION, detect_spikes_dvdt_confirmed, FLATLINE_MV)
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


def _save_page(fig, pdf: PdfPages, png_dir: Path, name: str) -> None:
    pdf.savefig(fig)
    fig.savefig(png_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def _resim(cell_id, params, y_ss, baseline_freq_hz, cell_result, held_nA, injected_nA, hold_tail_s=0.5):
    return resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA, cell_result, hold_tail_s)


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
    step = 0.47 / len(sections)
    for i, (title, desc) in enumerate(sections, start=1):
        fig.text(0.10, y, f"{i}. {title}", fontsize=9.5, va="top", weight="bold")
        fig.text(0.14, y - 0.018, _wrap(desc, width=95), fontsize=7.5, va="top", color="dimgray")
        y -= step

    return fig


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
    fig.suptitle("1. Spike detection", fontsize=13, weight="bold")
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
            peaks, _ = find_peaks(v, prominence=v_range * frac)
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
       "traces back to one find_peaks(..., prominence=range*0.3) call. This sweeps that threshold from "
       "0.10 to 0.55 on real traces: a flat plateau around the production value means the spike count "
       "this pipeline reports is not sensitive to the exact threshold chosen, not an arbitrary cliff-edge "
       "pick.", width=105)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.suptitle("2. Prominence-threshold sensitivity", fontsize=13, weight="bold", y=1.0)
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
    fig.suptitle("3. dV/dt cross-validation (unused-in-production detector as a sanity check)",
               fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Burst/tonic classification
# ---------------------------------------------------------------------------

def make_burst_classification_page(cell_id, exemplars, resim, run_args) -> plt.Figure:
    """exemplars: list of (held_nA, injected_nA, tag) -- typically one clean
    bursting example and one boundary/near-miss "correctly rejected" example.
    """
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(13, 5.8 * len(exemplars)))
    if len(exemplars) == 1:
        axes = axes[np.newaxis, :]
    captions = []
    for row, (held_nA, injected_nA, tag) in zip(axes, exemplars):
        ax_isi, ax_kde = row
        tr = resim(held_nA, injected_nA)
        t, v = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
        result, caption = mark_isi_classification(
            ax_isi, ax_kde, t, v, run_args["min_isis_for_burst_test"],
            run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"])
        ax_isi.set_title(f"{tag}: held={held_nA:.2f} inj={injected_nA:.2f} -> '{result['pattern']}'",
                        fontsize=9)
        captions.append((ax_isi, caption))
    fig.suptitle("4. Tonic / bursting / silent classification", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.subplots_adjust(hspace=0.9)
    for ax_isi, caption in captions:
        _place_caption_below(fig, ax_isi, caption, width=58)
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
    fig.suptitle("5. Rebound (post-inhibitory) detection", fontsize=13, weight="bold")
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
    fig.suptitle("6. Sag depth", fontsize=13, weight="bold")
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
    fig.suptitle("7. Adaptation ratio", fontsize=13, weight="bold")
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
    fig.suptitle("8. Firing rate / F-I slope", fontsize=13, weight="bold")
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
    fig.suptitle("9. Self-consistency check", fontsize=13, weight="bold")
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
    fig.suptitle("10. Burst-onset-then-silence detection", fontsize=13, weight="bold")
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
        return _resim(cell_id, params, y_ss, baseline_freq_hz, cell_result, held_nA, injected_nA)

    dt_ms = run_args["dt"]

    spike_pts = [(-2.892861, -2.958704, "near-boundary bursting"), (-3.093754, -3.535713, "tonic")]
    burst_exemplars = [(-3.21429, -3.363838, "clean bursting"), (-2.892861, -2.958704, "near-miss boundary")]
    rebound_exemplars = [(-3.857148, -3.92857, "single_spike"), (-0.321429, -3.535713, "tonic_rebound")]
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

    sections = [
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
    ]

    outdir = Path(args.figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    png_dir = outdir / cell_id
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{cell_id}_defense_packet.pdf"

    with PdfPages(pdf_path) as pdf:
        _save_page(make_cover_page(cell_id, cell_result, features, sections), pdf, png_dir, "00_cover")
        _save_page(make_spike_detection_page(cell_id, spike_pts, resim), pdf, png_dir, "01_spike_detection")
        _save_page(make_prominence_sensitivity_page(cell_id, spike_pts, resim), pdf, png_dir,
                  "02_prominence_sensitivity")
        _save_page(make_dvdt_crossvalidation_page(cell_id, grid, resim, dt_ms), pdf, png_dir,
                  "03_dvdt_crossvalidation")
        _save_page(make_burst_classification_page(cell_id, burst_exemplars, resim, run_args), pdf, png_dir,
                  "04_burst_classification")
        _save_page(make_rebound_page(cell_id, rebound_exemplars, resim, run_args["rebound_latency_min_ms"]),
                  pdf, png_dir, "05_rebound_detection")
        _save_page(make_sag_page(cell_id, sag_exemplars, resim, run_args["sag_window_ms"]), pdf, png_dir,
                  "06_sag_depth")
        _save_page(make_adaptation_page(cell_id, adapt_exemplars, resim, run_args["adaptation_edge_n"]),
                  pdf, png_dir, "07_adaptation_ratio")
        _save_page(make_fi_slope_page(cell_id, grid, features), pdf, png_dir, "08_fi_slope")
        _save_page(make_self_consistency_page(cell_id, cell_result, resim), pdf, png_dir,
                  "09_self_consistency")
        _save_page(make_onset_burst_page(cell_id, onset_exemplars, resim, run_args), pdf, png_dir,
                  "10_onset_burst_detection")

    print(f"{cell_id}: wrote {pdf_path} and {len(sections) + 1} section PNGs to {png_dir}/")


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
