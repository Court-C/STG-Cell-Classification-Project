"""Multi-page per-cell PDF: page 1 is the full 14-panel grid-features
heatmap (run_held_injected_grid.py's build_cell_grid_features_fig,
unchanged); pages 2-3 auto-showcase every mode actually present in the
test-window firing pattern and rebound pattern heatmaps (silent/tonic/
bursting; single_spike/tonic_rebound/bursting_rebound), one representative
point + resimulated trace per mode; page 4 is the firing-rate heatmap with
three held-current rows marked (plot_held_slice.py's build_held_slice_figure)
next to their F/I curves, so the heatmap's smooth-looking gradient can be
read as an actual curve -- defaults to control (held=0), the deepest swept
level, and roughly the midpoint between them, overridable via
--held-slice; optional further pages hold
any hand-picked (parameter, point) examples passed via --example. Each page
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
                                 rebound_prominence_score)
from plot_parameter_trace import resolve_grid_point, draw_parameter_trace_panel, _round_level
from plot_held_slice import build_held_slice_figure, resolve_held_level

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "grid_overview"
DEFAULT_HELD_SLICE_PARAMETER = "firing_rate"

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
    fig.suptitle(f"{cell_id} — {title}", fontsize=10)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-page per-cell PDF: page 1 = the full 14-panel grid-features heatmap; "
                    "pages 2-3 = every mode actually present in the test-window firing pattern and "
                    "rebound pattern heatmaps, one representative point + trace per mode; page 4 = the "
                    "firing-rate heatmap with three held-current rows marked next to their F/I curves; "
                    "further pages (optional) = hand-picked (parameter, point) examples via --example.")
    parser.add_argument("--cell", required=True, help="Single cell ID (an example point is cell-specific).")
    parser.add_argument("--example", dest="examples", action="append", default=[], type=_parse_example,
                        help="PARAMETER:HELD,INJECTED, e.g. firing_rate:-0.37,-2.02. Repeatable, optional.")
    parser.add_argument("--held-slice", dest="held_slice_values", action="append", type=float, default=None,
                        help="Held current level (nA) to mark/plot on page 4's firing-rate F/I-slice. "
                            "Repeatable. Default: control (0.0), this cell's deepest swept held level, "
                            "and roughly the midpoint between them.")
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
    page1.suptitle(grid_features_title(cell_result, features), fontsize=10)
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
    held_slice_page = build_held_slice_figure(cell_id, cell_result, features, DEFAULT_HELD_SLICE_PARAMETER,
                                              held_slice_values)
    pages.append(held_slice_page)
    print(f"  {DEFAULT_HELD_SLICE_PARAMETER} F/I slice at held={held_slice_values} added.")

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
