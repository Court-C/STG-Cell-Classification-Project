"""One-off: generate a trace figure for EVERY (held, injected) point in a
grid cache, instead of plot_example_traces.py's curated-exemplar-per-
combination selection. Reuses the same resimulate_point/plot_example_trace
machinery so figures are identical in style/content.
"""

import pickle
import sys
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from steady_state_cache import CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH
from steady_state_cache import get_cached_state
from plot_example_traces import resimulate_point, plot_example_trace, DEFAULT_FIGURE_FORMAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--grid-cache", required=True)
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--hold-tail-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)

    with open(args.grid_cache, "rb") as handle:
        grid_cache = pickle.load(handle)
    cell_result = grid_cache[args.cell]
    params = cell_result["params"]

    ss_entry = get_cached_state(args.cell, params, cache_path=Path(args.steady_state_cache))
    y_ss, baseline_freq_hz = ss_entry["y_ss"], ss_entry["freq_hz"]

    figures_dir = Path(args.figures_dir)
    grid = cell_result["grid"]
    keys = sorted(grid.keys())
    print(f"{args.cell}: {len(keys)} grid point(s) to trace")

    total_written, total_failed = 0, 0
    for held_nA, injected_nA in keys:
        point = grid[(held_nA, injected_nA)]
        test_pattern = point["test_pattern"] or "silent"
        rebound_pattern = point["rebound_pattern"]
        tr = resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA,
                              cell_result, args.hold_tail_s, ss_entry=ss_entry)
        if tr["blew_up"]:
            print(f"  ({held_nA:+.2f}, {injected_nA:+.2f}): blew up re-simulating "
                 f"({tr.get('stage')}: {tr.get('error')})")
            total_failed += 1
            continue
        outpath = plot_example_trace(args.cell, held_nA, injected_nA, test_pattern, rebound_pattern,
                                     tr, figures_dir, command, args.figure_format)
        print(f"  ({held_nA:+.2f}, {injected_nA:+.2f}) [{test_pattern}/{rebound_pattern}] -> {outpath}")
        total_written += 1

    print(f"\n{total_written} trace figure(s) written to {figures_dir}/, {total_failed} failed to re-simulate.")


if __name__ == "__main__":
    main()
