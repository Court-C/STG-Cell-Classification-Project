"""Multi-page per-cell PDF: page 1 is the full 14-panel grid-features
heatmap (run_held_injected_grid.py's build_cell_grid_features_fig,
unchanged); pages 2-3 auto-showcase every mode actually present in the
test-window firing pattern and rebound pattern heatmaps (silent/tonic/
bursting; single_spike/tonic_rebound/bursting_rebound), one representative
point + resimulated trace per mode; next are one page per
--held-slice-parameter (default: firing_rate, sag_depth) -- that
parameter's heatmap with three held-current rows marked
(plot_held_slice.py's build_held_slice_figure) next to their slice curves,
so the heatmap's smooth-looking gradient can be read as an actual curve --
defaults to control (held=0), the deepest swept level, and roughly the
midpoint between them, overridable via --held-slice; then one page per
--trace-parameter (default: n_bursts, spikes_per_burst -- the burst-
structure pair) -- low/mid/high REPRESENTATIVE TRACES instead of a summary
curve, for a parameter (like a burst count) whose value alone doesn't show
what the underlying trace looks like; optional further pages hold any
hand-picked (parameter, point) examples passed via --example. Each page
lets a reader flip from "here's everything" to "here's what a specific
mode/pixel/row actually looks like."

Unlike every other stage script, this takes a single --cell, not --cells --
same reasoning as plot_parameter_trace.py/plot_held_slice.py (an example
point is only meaningful relative to one cell's own grid).
"""

import pickle
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from steady_state_cache import CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import get_cached_state
from run_held_injected_grid import (PARAMETER_PANELS_BY_KEY, build_cell_grid_features_fig,
                                    grid_features_title,
                                    DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH)
from extract_grid_features import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_FEATURES_CACHE_PATH
from plot_example_traces import (resimulate_point, select_exemplar_by_category, _prominence_score,
                                 rebound_prominence_score, trace_time_offsets, _pick_best)
from plot_parameter_trace import resolve_grid_point, draw_parameter_trace_panel, _round_level
from plot_held_slice import build_held_slice_figure, resolve_held_level

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "grid_overview"
# One held-slice CURVE page per parameter here, in order -- firing_rate is
# the "easily provable" starting point; sag_depth added 2026-08-21 as the
# first harder-to-conceptualize parameter to get the same treatment. Extend
# this list (or pass --held-slice-parameter) as more parameters come up.
DEFAULT_HELD_SLICE_PARAMETERS = ["firing_rate", "sag_depth"]
# One representative-TRACES page per parameter here (build_value_trace_page:
# low/mid/high example traces instead of a summary curve) -- n_bursts and
# spikes_per_burst added 2026-08-21 (user-specified: "two pages on burst
# parameters," then "two-three representative traces... so it can be
# visibly seen" instead of a held-slice curve -- a count-like value doesn't
# show what the underlying burst structure actually looks like the way a
# real trace does). intra_burst_rate/burst_freq_approx deliberately
# excluded from either treatment: both are already-approximate rate
# readings of the same phenomenon n_bursts/spikes_per_burst cover more
# directly (see compute_burst_rate_maps's own docstring).
DEFAULT_TRACE_PARAMETERS = ["n_bursts", "spikes_per_burst"]

# The two categorical panels this script auto-showcases a mode page for, and
# how to pick a representative point for each of their categories -- see
# select_exemplar_by_category. Every other PARAMETER_PANELS entry is
# continuous (a "mode" doesn't apply), so only these two get this treatment.
_MODE_PAGES = [
    dict(parameter="test_pattern", category_of=lambda p: p["test_pattern"] or "silent",
        score_of=_prominence_score),
    # rebound_pattern is only meaningful when rebound_applicable -- matching
    # compute_rebound_maps's own gating (extract_grid_features.py), which is
    # what the rebound_pattern heatmap panel's value_map is actually built
    # from. Without this gate, a point could be selected here whose
    # rebound_pattern the heatmap itself has no value for at all (confirmed
    # directly: produced a "= None" panel title, 2026-08-21).
    dict(parameter="rebound_pattern",
        category_of=lambda p: p["rebound_pattern"] if p.get("rebound_applicable") else None,
        score_of=rebound_prominence_score),
]


def _parse_example(text: str) -> tuple:
    try:
        parameter, point_str = text.split(":")
        held_str, injected_str = point_str.split(",")
        if parameter not in PARAMETER_PANELS_BY_KEY:
            raise ValueError
        return parameter, _round_level(float(held_str)), _round_level(float(injected_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--example must be 'PARAMETER:HELD,INJECTED' (e.g. firing_rate:-0.37,-2.02), got {text!r}")


def build_points_page(cell_id: str, cell_result: dict, features: dict, params, y_ss, baseline_freq_hz,
                      ss_entry, rows: list, hold_tail_s: float, title: str) -> tuple:
    """Renders one page: `rows` is a list of (parameter, held_nA, injected_nA)
    triples, one row per panel (mini heatmap with the point marked + its
    resimulated trace). Shared by the auto mode pages (test_pattern/
    rebound_pattern, one row per mode actually present) and the optional
    hand-picked --example page.
    """
    fig = plt.figure(figsize=(15, 6 * len(rows)))
    outer = fig.add_gridspec(len(rows), 1)
    n_failed = 0
    for i, (parameter, held_nA, injected_nA) in enumerate(rows):
        resolve_grid_point(cell_result, held_nA, injected_nA)  # fail fast with a clear message
        tr = resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA,
                              cell_result, hold_tail_s, ss_entry=ss_entry)
        if tr["blew_up"]:
            print(f"  {parameter} ({held_nA:+.2f}, {injected_nA:+.2f}): blew up re-simulating "
                 f"({tr.get('stage')}: {tr.get('error')})")
            n_failed += 1
            continue
        draw_parameter_trace_panel(fig, outer[i, 0], cell_id, cell_result, features, parameter,
                                   held_nA, injected_nA, tr)
    fig.suptitle(f"{cell_id} — {title}", fontsize=19)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig, n_failed


def build_mode_page(cell_id: str, cell_result: dict, features: dict, params, y_ss, baseline_freq_hz,
                    ss_entry, parameter: str, category_of, score_of, hold_tail_s: float):
    """One row per mode of `parameter` actually present in this cell's grid
    (e.g. every test_pattern value found: silent/tonic/bursting, or every
    rebound_pattern value found: single_spike/tonic_rebound/bursting_rebound
    -- exactly however many are actually present, not a fixed count).
    Returns (None, 0) if the grid has no non-blank category for this
    parameter at all (e.g. a cell where rebound never occurs), so the caller
    can skip the page entirely rather than emit an empty one.
    """
    spec = PARAMETER_PANELS_BY_KEY[parameter]
    exemplars = select_exemplar_by_category(cell_result, category_of, score_of,
                                            exclude_categories=spec["blank_labels"])
    if not exemplars:
        print(f"  {parameter}: no category present in this cell's grid -- page skipped.")
        return None, 0

    # Canonical order (matches the heatmap's own legend order), restricted
    # to categories actually present -- "however many are present," not
    # padded out to every category this cell type could theoretically show.
    ordered_categories = [c for c in spec["colors"].keys() if c in exemplars]
    rows = [(parameter, key[0], key[1]) for c in ordered_categories for key, _point in [exemplars[c]]]
    return build_points_page(cell_id, cell_result, features, params, y_ss, baseline_freq_hz, ss_entry,
                             rows, hold_tail_s, f"{spec['title']} — every mode present")


def select_sag_exemplars(cell_result: dict, features: dict) -> list:
    """Two representative (held, injected) points for demonstrating how
    sag_depth is calculated -- one near the deep/high end of the range
    actually present, one near the shallow/low end -- so the annotation
    page shows the calculation across contrasting cases rather than a
    single example (user-specified 2026-08-21).

    For each end, the single MOST extreme point is deliberately excluded
    first (could be an outlier/edge artifact) and the next-most-extreme
    picked instead; within what's left, restricted to points whose HELD
    level itself is quiescent (hold_freq_hz == 0, so the annotated baseline
    lands on a flat resting trace rather than an ongoing spike's AHP
    trough) with the usual interior/held<injected preferences (_pick_best)
    applied on top. Falls back to the full sag_depth_map if no
    quiescent-hold point has a sag value at all. Returns 1 key if the two
    ends coincide (a grid with very few sag points), 0 if none exist.
    """
    sag_map = features.get("sag_depth_map") or {}
    grid = cell_result["grid"]
    quiescent = [(k, grid[k]) for k in sag_map if k in grid and grid[k]["hold_freq_hz"] == 0]
    pool = quiescent if quiescent else [(k, grid[k]) for k in sag_map if k in grid]
    if not pool:
        return []
    held_levels = list(cell_result["held_levels_nA"])
    injected_levels = list(cell_result["injected_levels_nA"])

    def pick(reverse: bool) -> tuple:
        value_of = lambda kp: sag_map[(kp[1]["held_nA"], kp[1]["injected_nA"])]
        ranked = sorted(pool, key=value_of, reverse=reverse)
        remaining = ranked[1:] if len(ranked) > 1 else ranked
        sign = -1.0 if reverse else 1.0  # score_fn must rank "closer to this end" highest
        key, _point = _pick_best(remaining, lambda p: sign * sag_map[(p["held_nA"], p["injected_nA"])],
                                 True, held_levels, injected_levels)
        return key

    high_key, low_key = pick(reverse=True), pick(reverse=False)
    return [high_key, low_key] if high_key != low_key else [high_key]


def select_value_range_exemplars(cell_result: dict, value_map: dict, n: int = 3) -> list:
    """n representative (held, injected) points spanning value_map's range
    (evenly spaced by percentile -- low/mid/high for n=3), each snapped to
    the nearest point actually near that target value with the usual
    interior/held<injected preferences (_pick_best) applied -- for a page
    of real traces across a parameter's range instead of a single summary
    curve (user-specified 2026-08-21: a count-like value such as n_bursts
    doesn't show what the underlying burst structure actually looks like
    the way a real V(t) trace does). Deduplicates in case two target
    percentiles land on the same nearest point. Returns [] if value_map is
    empty.
    """
    grid = cell_result["grid"]
    pts = [(k, grid[k]) for k in value_map if k in grid and value_map[k] is not None]
    if not pts:
        return []
    held_levels = list(cell_result["held_levels_nA"])
    injected_levels = list(cell_result["injected_levels_nA"])
    sorted_values = sorted(value_map[k] for k, _p in pts)

    keys = []
    for frac in np.linspace(0.0, 1.0, n):
        target = sorted_values[int(round(frac * (len(sorted_values) - 1)))]
        key, _point = _pick_best(pts, lambda p, t=target: -abs(value_map[(p["held_nA"], p["injected_nA"])] - t),
                                 True, held_levels, injected_levels)
        if key not in keys:
            keys.append(key)
    return keys


def build_value_trace_page(cell_id: str, cell_result: dict, features: dict, params, y_ss, baseline_freq_hz,
                           ss_entry, parameter: str, hold_tail_s: float, n: int = 3):
    """One row per select_value_range_exemplars pick for `parameter` (a
    continuous "map" panel) -- low/mid/high representative traces instead
    of a held-slice curve. Returns None if the parameter has no points at
    all or every resimulation blows up.
    """
    spec = PARAMETER_PANELS_BY_KEY[parameter]
    value_map = spec["value_map_fn"](cell_result, features)
    keys = select_value_range_exemplars(cell_result, value_map, n=n)
    if not keys:
        print(f"  {parameter}: no points available -- trace page skipped.")
        return None
    rows = [(parameter, key[0], key[1]) for key in keys]
    fig, n_failed = build_points_page(cell_id, cell_result, features, params, y_ss, baseline_freq_hz, ss_entry,
                                      rows, hold_tail_s, f"{spec['title']} — representative traces")
    if fig is None or n_failed == len(rows):
        return None
    return fig


def _annotate_sag_calculation(ax_v, tr: dict, point: dict, features: dict,
                              held_nA: float, injected_nA: float) -> None:
    """Draws the sag-depth calculation onto an already-built trace panel: a
    dashed line at the held level's own baseline trough (hold_v_trough_mV --
    what the cell was doing right before the test step), and an arrow down
    to the deepest point reached before the first spike
    (test_v_min_pre_spike_mV), labeled with the delta between them -- so a
    reader can see sag is a DIFFERENCE between two measured points, not an
    absolute reading off the trace (user-flagged 2026-08-21).
    """
    hold_v_trough = point["hold_v_trough_mV"]
    test_v_min_pre_spike = point["test_v_min_pre_spike_mV"]
    sag_value = (features.get("sag_depth_map") or {}).get((held_nA, injected_nA))
    if sag_value is None:
        sag_value = hold_v_trough - test_v_min_pre_spike

    # Locate the trough's time in the same stitched coordinate the trace is
    # plotted in (trace_time_offsets), by matching the actual sample closest
    # to the already-computed test_v_min_pre_spike_mV value -- both come
    # from the same v_test array, so this recovers the real sample rather
    # than guessing a time from the value alone.
    offsets = trace_time_offsets(tr)
    v_test = tr["_trace_v_test_mV"]
    trough_idx = int(np.argmin(np.abs(v_test - test_v_min_pre_spike)))
    trough_t = offsets["t_test_off"][trough_idx]

    ax_v.axhline(hold_v_trough, color="black", ls="--", lw=1.2, zorder=5,
                label=f"held baseline trough ({hold_v_trough:.1f} mV)")
    ax_v.annotate("", xy=(trough_t, test_v_min_pre_spike), xytext=(trough_t, hold_v_trough),
                 arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.5), zorder=6)

    # Offset the label off to the side of the arrow (not centered ON it) --
    # sitting right on top of the trace/dip obscured the very thing it's
    # meant to point out (user-flagged 2026-08-21). Offset is a fraction of
    # the axes' own x-range so it scales with however wide this particular
    # trace's hold/test/recovery span happens to be, rather than a fixed
    # ms offset that could land differently panel to panel.
    x0, x1 = ax_v.get_xlim()
    label_x = trough_t + 0.03 * (x1 - x0)
    ax_v.text(label_x, 0.5 * (hold_v_trough + test_v_min_pre_spike), f"sag = {sag_value:.1f} mV",
             color="crimson", fontsize=18, va="center", ha="left", zorder=6)

    # Widen the y-limits if needed so the baseline/arrow are never clipped
    # by the 5 mV autoscale floor _draw_trace_axes already applied (that
    # floor is sized to the raw trace alone, before this annotation).
    y0, y1 = ax_v.get_ylim()
    y0 = min(y0, hold_v_trough, test_v_min_pre_spike) - 1.0
    y1 = max(y1, hold_v_trough, test_v_min_pre_spike) + 1.0
    ax_v.set_ylim(y0, y1)
    ax_v.legend(loc="upper right", fontsize=16)


def build_sag_calculation_page(cell_id: str, cell_result: dict, features: dict, params, y_ss,
                               baseline_freq_hz, ss_entry, hold_tail_s: float):
    """One row per exemplar from select_sag_exemplars (near-max and
    near-min sag, one row each), each a heatmap(+marked point)/trace panel
    with the calculation annotated -- see _annotate_sag_calculation.
    Returns None if no sag_depth points exist or every resimulation blows
    up, so the caller can skip the page entirely.
    """
    keys = select_sag_exemplars(cell_result, features)
    if not keys:
        print("  sag calculation: no sag_depth points available -- page skipped.")
        return None

    fig = plt.figure(figsize=(15, 6 * len(keys)))
    outer = fig.add_gridspec(len(keys), 1)
    n_added = 0
    for i, (held_nA, injected_nA) in enumerate(keys):
        point = cell_result["grid"][(held_nA, injected_nA)]
        tr = resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA,
                              cell_result, hold_tail_s, ss_entry=ss_entry)
        if tr["blew_up"]:
            print(f"  sag calculation ({held_nA:+.2f}, {injected_nA:+.2f}): blew up re-simulating "
                 f"({tr.get('stage')}: {tr.get('error')})")
            continue
        _ax_heat, ax_v, _ax_i = draw_parameter_trace_panel(fig, outer[i, 0], cell_id, cell_result, features,
                                                            "sag_depth", held_nA, injected_nA, tr)
        _annotate_sag_calculation(ax_v, tr, point, features, held_nA, injected_nA)
        n_added += 1

    if n_added == 0:
        plt.close(fig)
        return None
    fig.suptitle(f"{cell_id} — how sag depth is calculated (near-max and near-min examples)", fontsize=19)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-page per-cell PDF: page 1 = the full 14-panel grid-features heatmap; "
                    "pages 2-3 = every mode actually present in the test-window firing pattern and "
                    "rebound pattern heatmaps, one representative point + trace per mode; one page per "
                    "--held-slice-parameter, that parameter's heatmap with three held-current rows marked "
                    "next to their slice curves; one page per --trace-parameter, low/mid/high "
                    "representative traces instead of a curve; further pages (optional) = hand-picked "
                    "(parameter, point) examples via --example.")
    parser.add_argument("--cell", required=True, help="Single cell ID (an example point is cell-specific).")
    parser.add_argument("--example", dest="examples", action="append", default=[], type=_parse_example,
                        help="PARAMETER:HELD,INJECTED, e.g. firing_rate:-0.37,-2.02. Repeatable, optional.")
    parser.add_argument("--held-slice", dest="held_slice_values", action="append", type=float, default=None,
                        help="Held current level (nA) to mark/plot on each held-slice page. Repeatable. "
                            "Default: control (0.0), this cell's deepest swept held level, and roughly "
                            "the midpoint between them.")
    parser.add_argument("--held-slice-parameter", dest="held_slice_parameters", action="append",
                        choices=list(PARAMETER_PANELS_BY_KEY.keys()), default=None,
                        help="Parameter to render a held-slice CURVE page for. Repeatable, one page each, "
                            f"in order given. Default: {DEFAULT_HELD_SLICE_PARAMETERS}.")
    parser.add_argument("--trace-parameter", dest="trace_parameters", action="append",
                        choices=list(PARAMETER_PANELS_BY_KEY.keys()), default=None,
                        help="Parameter to render a representative-TRACES page for (low/mid/high example "
                            "traces instead of a summary curve -- for a parameter where the underlying "
                            f"trace is more informative than a value-vs-current curve). Default: "
                            f"{DEFAULT_TRACE_PARAMETERS}.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH)
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--hold-tail-s", type=float, default=2.0,
                        help="Duration of hold-only context shown before the test window (s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_id = args.cell

    with open(args.grid_cache, "rb") as f:
        grid_cache = pickle.load(f)
    with open(args.grid_features_cache, "rb") as f:
        features_cache = pickle.load(f)

    cell_result = grid_cache.get(cell_id)
    if cell_result is None or cell_result["status"] != "ok":
        raise SystemExit(f"{cell_id}: not present or not status 'ok' in {args.grid_cache}")
    features = features_cache.get(cell_id)
    if features is None or features.get("status") != "ok":
        raise SystemExit(f"{cell_id}: not present or not status 'ok' in {args.grid_features_cache}")

    params = cell_result["params"]
    ss_entry = get_cached_state(cell_id, params, cache_path=Path(args.steady_state_cache))
    if ss_entry is None:
        raise SystemExit(f"{cell_id}: no valid steady-state cache entry")
    y_ss, baseline_freq_hz = ss_entry["y_ss"], ss_entry["freq_hz"]

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outpath = figures_dir / f"{cell_id}_overview.pdf"

    page1 = build_cell_grid_features_fig(cell_result, features)
    if page1 is None:
        raise SystemExit(f"{cell_id}: grid/features not usable (status not 'ok' or empty grid)")
    page1.suptitle(grid_features_title(cell_result, features), fontsize=19)
    pages = [page1]

    for mode_page in _MODE_PAGES:
        fig, n_failed = build_mode_page(cell_id, cell_result, features, params, y_ss, baseline_freq_hz,
                                        ss_entry, mode_page["parameter"], mode_page["category_of"],
                                        mode_page["score_of"], args.hold_tail_s)
        if fig is not None:
            pages.append(fig)
            print(f"  {mode_page['parameter']}: mode page added ({n_failed} failed to re-simulate).")

    # Default to three guaranteed-valid grid levels spanning the swept
    # range -- control (held=0, the reference condition every trace on
    # pages 2-3 is compared against), the deepest swept level (the other
    # extreme), and roughly the midpoint between them -- so the F/I-slice
    # page has a sensible spread out of the box without requiring
    # --held-slice (user-specified default, 2026-08-21: two extremes alone
    # can land almost on top of each other for a cell whose firing rate is
    # only weakly held-dependent).
    held_levels_nA = list(cell_result["held_levels_nA"])
    held_slice_values = args.held_slice_values or [held_levels_nA[0], held_levels_nA[len(held_levels_nA) // 2],
                                                    held_levels_nA[-1]]
    held_slice_values = [resolve_held_level(cell_result, h) for h in held_slice_values]
    held_slice_parameters = args.held_slice_parameters or DEFAULT_HELD_SLICE_PARAMETERS
    for parameter in held_slice_parameters:
        held_slice_page = build_held_slice_figure(cell_id, cell_result, features, parameter, held_slice_values)
        pages.append(held_slice_page)
        print(f"  {parameter} slice at held={held_slice_values} added.")

    if "sag_depth" in held_slice_parameters:
        sag_page = build_sag_calculation_page(cell_id, cell_result, features, params, y_ss,
                                              baseline_freq_hz, ss_entry, args.hold_tail_s)
        if sag_page is not None:
            pages.append(sag_page)
            print("  sag calculation page added.")

    trace_parameters = args.trace_parameters or DEFAULT_TRACE_PARAMETERS
    for parameter in trace_parameters:
        trace_page = build_value_trace_page(cell_id, cell_result, features, params, y_ss, baseline_freq_hz,
                                            ss_entry, parameter, args.hold_tail_s)
        if trace_page is not None:
            pages.append(trace_page)
            print(f"  {parameter}: representative-traces page added.")

    if args.examples:
        examples_page, n_failed = build_points_page(cell_id, cell_result, features, params, y_ss,
                                                     baseline_freq_hz, ss_entry, args.examples,
                                                     args.hold_tail_s, "curated examples")
        pages.append(examples_page)
        print(f"  curated examples: {len(args.examples)} example(s), {n_failed} failed.")

    with PdfPages(outpath) as pdf:
        for page in pages:
            pdf.savefig(page)
            plt.close(page)

    print(f"{cell_id}: {len(pages)} page(s) -> {outpath}")


if __name__ == "__main__":
    main()
