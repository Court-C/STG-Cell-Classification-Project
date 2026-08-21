"""Two-page per-cell PDF: page 1 is the full 14-panel grid-features heatmap
(run_held_injected_grid.py's build_cell_grid_features_fig, unchanged), page 2
is a handful of hand-picked (parameter, point) examples -- each a mini
heatmap with that point marked next to its resimulated trace
(plot_parameter_trace.py's draw_parameter_trace_panel) -- so a reader can
flip straight from "here's everything" to "here's what a few specific,
possibly-non-obvious pixels actually look like."

Which examples appear on page 2 is entirely up to --example: this script
doesn't guess at what's interesting, it just assembles what you point it at.

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
from plot_example_traces import resimulate_point
from plot_parameter_trace import resolve_grid_point, draw_parameter_trace_panel, _round_level

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "grid_overview"


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


def build_examples_page(cell_id: str, cell_result: dict, features: dict, params, y_ss,
                        baseline_freq_hz, ss_entry, examples: list, hold_tail_s: float) -> plt.Figure:
    fig = plt.figure(figsize=(15, 6 * len(examples)))
    outer = fig.add_gridspec(len(examples), 1)
    n_failed = 0
    for i, (parameter, held_nA, injected_nA) in enumerate(examples):
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
    fig.suptitle(f"{cell_id} — curated examples", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig, n_failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-page per-cell PDF: page 1 = the full 14-panel grid-features heatmap, "
                    "page 2 = a handful of hand-picked (parameter, point) examples with their "
                    "resimulated traces.")
    parser.add_argument("--cell", required=True, help="Single cell ID (an example point is cell-specific).")
    parser.add_argument("--example", dest="examples", action="append", required=True, type=_parse_example,
                        help="PARAMETER:HELD,INJECTED, e.g. firing_rate:-0.37,-2.02. Repeatable.")
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

    page2, n_failed = build_examples_page(cell_id, cell_result, features, params, y_ss, baseline_freq_hz,
                                          ss_entry, args.examples, args.hold_tail_s)

    with PdfPages(outpath) as pdf:
        pdf.savefig(page1)
        pdf.savefig(page2)
    plt.close(page1)
    plt.close(page2)

    print(f"{cell_id}: page 1 (14-panel heatmap) + page 2 ({len(args.examples)} example(s), "
         f"{n_failed} failed) -> {outpath}")


if __name__ == "__main__":
    main()
