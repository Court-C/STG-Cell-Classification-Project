"""F/I-style slice-curve tool: given a cell, one of the 14 grid-features
heatmap parameters (src/run_held_injected_grid.py's PARAMETER_PANELS), and
one or more fixed held-current values, plots that parameter against injected
current along each held "row" -- the classic F/I-curve view -- next to the
heatmap with those rows marked as vertical lines.

The 2D heatmap isn't an immediately intuitive format on its own; this 1D
slice supplements it with the more familiar curve, read directly off the
same held row a reader can see marked on the heatmap.

No resimulation: reads only the already-computed grid + grid-features
caches (cell_held_injected_grid.pkl / cell_grid_features.pkl), so this is
fast and needs no steady-state cache or model params. See
plot_parameter_trace.py for the point-level (resimulated trace) companion
tool.

Unlike every other stage script, this takes a single --cell, not --cells --
held-current levels are only meaningful relative to one cell's own grid
axis (see plot_parameter_trace.py's docstring for the same reasoning).
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

from run_held_injected_grid import (PARAMETER_PANELS_BY_KEY, draw_parameter_panel, _round_level,
                                    DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH)
from extract_grid_features import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_FEATURES_CACHE_PATH

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "held_slices"
DEFAULT_FIGURE_FORMAT = "png"
DEFAULT_PARAMETER = "firing_rate"


def resolve_held_level(cell_result: dict, held_nA: float) -> float:
    """Exact-match against this cell's own held_levels_nA -- same
    no-nearest-neighbor discipline as plot_parameter_trace.py's
    resolve_grid_point, for the same reason: a held value that was never
    actually swept has no honest slice to show.
    """
    held_levels = list(cell_result["held_levels_nA"])
    rounded = _round_level(held_nA)
    if rounded not in held_levels:
        raise SystemExit(f"{held_nA} is not an exact held level for this cell.\n"
                         f"  held_levels_nA: {held_levels}")
    return rounded


def build_held_slice_figure(cell_id: str, cell_result: dict, features: dict, parameter: str,
                            held_values: list) -> plt.Figure:
    spec = PARAMETER_PANELS_BY_KEY[parameter]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_heat, ax_curve) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.3]})
    value_map = draw_parameter_panel(ax_heat, cell_result, features, parameter)

    # draw_parameter_panel's imshow sets xlim to exactly the swept held
    # range, so a reference line at either extreme (e.g. held=0, the
    # control condition, which IS the range's own edge by construction)
    # lands exactly on the axes frame and gets clipped away entirely rather
    # than just being hard to see (confirmed directly: a held=0 axvline
    # was invisible, not just faint, 2026-08-21). A small symmetric margin
    # gives every reference line room to render fully inside the frame.
    x0, x1 = ax_heat.get_xlim()
    margin = 0.03 * abs(x1 - x0)
    ax_heat.set_xlim(min(x0, x1) - margin, max(x0, x1) + margin)

    if spec["kind"] == "categorical":
        category_names = list(spec["colors"].keys())
        ax_curve.set_yticks(range(len(category_names)))
        ax_curve.set_yticklabels(category_names)
        ax_curve.set_ylabel(spec["title"])
    else:
        ax_curve.set_ylabel(f"{spec['title']} ({spec['label']})")

    for i, held_nA in enumerate(held_values):
        color = colors[i % len(colors)]
        ax_heat.axvline(held_nA, color=color, ls="--", lw=1.5)

        rows = sorted((injected_nA, v) for (h, injected_nA), v in value_map.items()
                      if h == held_nA and v is not None)
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax_curve.plot(xs, ys, marker="o", ms=4, color=color, label=f"held={held_nA:+.2f} nA")

    ax_curve.set_xlabel("injected current (nA)")
    ax_curve.legend(loc="best", fontsize=8)
    ax_curve.set_title(f"{spec['title']} vs. injected current", fontsize=9)

    fig.suptitle(f"{cell_id} — {spec['title']}", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a parameter's value vs. injected current along one or more fixed held-current "
                    "rows (F/I-curve style), next to the heatmap with those rows marked -- a more "
                    "familiar 1D supplement to the 2D grid-features heatmap.")
    parser.add_argument("--cell", required=True, help="Single cell ID (held levels are cell-specific).")
    parser.add_argument("--parameter", default=DEFAULT_PARAMETER, choices=list(PARAMETER_PANELS_BY_KEY.keys()))
    parser.add_argument("--held", dest="held_values", action="append", required=True, type=float,
                        help="Held current level (nA), e.g. 0.0. Repeatable to overlay several rows.")
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH)
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
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

    held_values = [resolve_held_level(cell_result, h) for h in args.held_values]

    fig = build_held_slice_figure(cell_id, cell_result, features, args.parameter, held_values)

    figures_dir = Path(args.figures_dir) / cell_id
    figures_dir.mkdir(parents=True, exist_ok=True)
    held_slug = "_".join(f"{h:+.2f}" for h in held_values)
    outpath = figures_dir / f"{args.parameter}__held{held_slug}.{args.figure_format}"
    fig.savefig(outpath, format=args.figure_format, dpi=150)
    plt.close(fig)

    print(f"{cell_id}: {args.parameter} slice at held={held_values} -> {outpath}")


if __name__ == "__main__":
    main()
