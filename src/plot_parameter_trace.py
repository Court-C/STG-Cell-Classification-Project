"""Point -> trace inspection tool: given a cell, one of the 14 grid-features
heatmap parameters (src/run_held_injected_grid.py's PARAMETER_PANELS), and a
specific (held, injected) coordinate read off that parameter's heatmap,
resimulates and plots the actual trace for that exact point next to the
heatmap with the point marked -- so a heatmap pixel that looks surprising
(non-monotonic, doesn't follow the gradient you'd expect) can be explained
by eye rather than guessed at.

Unlike every other stage script, this takes a single --cell, not --cells --
a (held, injected) coordinate is only meaningful relative to one cell's own
grid levels (held/injected floors vary per cell), so batching cells doesn't
make sense here the way it does for e.g. generate_steady_state.py.

Reuses run_held_injected_grid.py's PARAMETER_PANELS/draw_parameter_panel for
the heatmap panel and plot_example_traces.py's resimulate_point/
_draw_trace_axes for the trace -- nothing here reimplements either.
"""

import pickle
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from steady_state_cache import CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import get_cached_state
from run_held_injected_grid import (PARAMETER_PANELS_BY_KEY, draw_parameter_panel, _round_level,
                                    DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH)
from extract_grid_features import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_FEATURES_CACHE_PATH
from plot_example_traces import resimulate_point, _draw_trace_axes, describe_pattern

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "parameter_traces"
DEFAULT_FIGURE_FORMAT = "png"


def _parse_point(text: str) -> tuple:
    try:
        held_str, injected_str = text.split(",")
        return _round_level(float(held_str)), _round_level(float(injected_str))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--point must be 'HELD,INJECTED' (e.g. -0.40,-2.10), got {text!r}")


def resolve_grid_point(cell_result: dict, held_nA: float, injected_nA: float) -> dict:
    """Exact-match lookup against this cell's own grid -- no snapping to a
    nearest neighbor, matching the same "exact matrix, no interpolation"
    convention _exact_grid_matrix documents in run_held_injected_grid.py:
    a coordinate that was never actually simulated has no honest trace to
    show. Raises with the cell's real levels spelled out if the point isn't
    on the grid, rather than silently resimulating a nearby-but-different
    point.
    """
    point = cell_result["grid"].get((held_nA, injected_nA))
    if point is None:
        held_levels = list(cell_result["held_levels_nA"])
        injected_levels = list(cell_result["injected_levels_nA"])
        raise SystemExit(
            f"({held_nA}, {injected_nA}) is not an exact grid point for this cell.\n"
            f"  held_levels_nA:     {held_levels}\n"
            f"  injected_levels_nA: {injected_levels}")
    return point


def draw_parameter_trace_panel(fig: plt.Figure, subplot_spec, cell_id: str, cell_result: dict,
                               features: dict, parameter: str, held_nA: float, injected_nA: float,
                               tr: dict) -> None:
    """Draws one heatmap(+marked point)/trace panel into `subplot_spec` (a
    matplotlib SubplotSpec -- e.g. one cell of an outer GridSpec) -- split
    out of build_parameter_trace_figure so a caller building a page with
    SEVERAL examples (e.g. a curated-examples summary page) can lay out
    multiple panels on one figure via nested GridSpecs instead of gluing
    together several independent Figures.
    """
    point = resolve_grid_point(cell_result, held_nA, injected_nA)
    spec = PARAMETER_PANELS_BY_KEY[parameter]

    inner = subplot_spec.subgridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[3, 1])
    ax_heat = fig.add_subplot(inner[:, 0])
    ax_v = fig.add_subplot(inner[0, 1])
    ax_i = fig.add_subplot(inner[1, 1], sharex=ax_v)

    value_map = draw_parameter_panel(ax_heat, cell_result, features, parameter)
    ax_heat.plot(held_nA, injected_nA, marker="x", color="red", markersize=12, markeredgewidth=2)

    value = value_map.get((held_nA, injected_nA))
    value_str = f"{value:.3g}" if isinstance(value, (int, float)) else str(value)
    _draw_trace_axes(ax_v, ax_i, cell_id, held_nA, injected_nA,
                     point["test_pattern"], point["rebound_pattern"], tr)
    ax_v.set_title(f"{spec['title']} = {value_str} at held={held_nA:+.2f} nA, injected={injected_nA:+.2f} nA\n"
                   f"{describe_pattern(point['test_pattern'])} during the current step, then "
                   f"{describe_pattern(point['rebound_pattern'])}", fontsize=9)


def build_parameter_trace_figure(cell_id: str, cell_result: dict, features: dict, parameter: str,
                                 held_nA: float, injected_nA: float, tr: dict) -> plt.Figure:
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 1)
    draw_parameter_trace_panel(fig, gs[0, 0], cell_id, cell_result, features, parameter,
                               held_nA, injected_nA, tr)
    fig.suptitle(cell_id, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resimulate and plot the trace for a specific (held, injected) grid point, "
                    "next to the heatmap for one parameter with that point marked -- for explaining "
                    "a heatmap pixel that doesn't follow the gradient you'd expect.")
    parser.add_argument("--cell", required=True, help="Single cell ID (a grid point is cell-specific).")
    parser.add_argument("--parameter", required=True, choices=list(PARAMETER_PANELS_BY_KEY.keys()))
    parser.add_argument("--point", dest="points", action="append", required=True, type=_parse_point,
                        help="HELD,INJECTED (nA), e.g. -0.40,-2.10. Repeatable.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH)
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
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

    figures_dir = Path(args.figures_dir) / cell_id
    figures_dir.mkdir(parents=True, exist_ok=True)

    total_written, total_failed = 0, 0
    for held_nA, injected_nA in args.points:
        resolve_grid_point(cell_result, held_nA, injected_nA)  # fail fast with a clear message
        tr = resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA,
                              cell_result, args.hold_tail_s, ss_entry=ss_entry)
        if tr["blew_up"]:
            print(f"  ({held_nA:+.2f}, {injected_nA:+.2f}): blew up re-simulating "
                 f"({tr.get('stage')}: {tr.get('error')})")
            total_failed += 1
            continue
        fig = build_parameter_trace_figure(cell_id, cell_result, features, args.parameter,
                                           held_nA, injected_nA, tr)
        outpath = figures_dir / f"{args.parameter}__held{held_nA:+.2f}__inj{injected_nA:+.2f}.{args.figure_format}"
        fig.savefig(outpath, format=args.figure_format, dpi=150)
        plt.close(fig)
        print(f"  ({held_nA:+.2f}, {injected_nA:+.2f}) [{args.parameter}] -> {outpath}")
        total_written += 1

    print(f"\n{total_written} figure(s) written to {figures_dir}/, {total_failed} failed to re-simulate.")


if __name__ == "__main__":
    main()
