"""Generate illustrative full-trace figures for representative (cell, held,
injected) grid points from run_held_injected_grid.py's output.

run_held_injected_grid.py deliberately never persists full voltage traces
(only scalar/derived summaries per grid point) to keep its cache small
across 69 cells x up to ~150 points each. This script re-simulates specific
points on demand -- reusing the exact same settle/test/recovery machinery
(settle_hold_level, run_test_and_recovery) so the reproduced trace matches
what actually produced the classification stored in the grid cache -- and
plots them.

For each selected cell, one exemplar point is chosen per distinct
(test_pattern, rebound_pattern) combination actually observed in that
cell's grid (preferring coarse-grid points, which sit further from a
bisected boundary and are more "typical" of their region). Output is
organized as one subfolder per cell, one file per exemplar, with the
current levels and classification baked into the filename so the figure
set is self-documenting without needing an index:

    figures/example_traces/{cell_id}/held{H:+.2f}_inj{I:+.2f}_test-{pattern}_rebound-{pattern}.svg
"""

import pickle
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import get_cached_state
from run_held_injected_grid import (constant_iapp_func, settle_hold_level, run_test_and_recovery,
                                    classify_hold_pattern, _round_level, V_INDEX,
                                    DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH)

DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "example_traces"
DEFAULT_FIGURE_FORMAT = "png"

# Curated to span the range of behaviors seen in the real 69-cell grid
# (see the session that built this: richest classification diversity,
# deepest silencing threshold, bursting-at-rest, tonic-only, deep+diverse).
DEFAULT_CURATED_CELLS = ["W0E22J", "WX7CJ9", "2EXYPV", "4QSWXH", "5A6WBD"]

# Font sizes for _draw_trace_axes/plot_example_trace -- sized for projector/
# screen-share legibility (user-flagged 2026-08-21: the previous defaults,
# tuned for a standalone reader zoomed into the PNG, were too small to read
# in a live presentation).
TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11
FOOTER_FONTSIZE = 8

# The held=0/injected=0 control trace has no perturbation to resolve at any
# particular timescale, so instead of cell_result's test_window_s/
# recovery_window_s (sized for ISI-classification accuracy at arbitrary
# grid points, routinely several seconds) it's shown over this many of the
# cell's own baseline periods -- long enough to read as a rhythm, short
# enough that individual spikes stay resolvable rather than smearing into a
# saturated block (user-flagged 2026-08-21: a full-length control trace at
# a healthy firing rate crams hundreds of spikes into one figure).
DEFAULT_CONTROL_CYCLES = 15.0
# Fallback window when the cell doesn't fire at all at baseline (baseline_
# freq_hz == 0) -- there's no period to scale a cycle count from, so this
# is a plain fixed duration instead.
DEFAULT_CONTROL_FALLBACK_WINDOW_S = 1.0

_PATTERN_LABELS = {
    "tonic": "tonic firing", "bursting": "bursting", "silent": "silent (no firing)",
    "tonic_rebound": "a sustained (tonic) rebound", "bursting_rebound": "a burst-like rebound",
    "single_spike": "a single-spike rebound", "none": "no rebound",
    "control": "its baseline rhythm (no injected current)",
}


def describe_pattern(pattern) -> str:
    """Plain-language description of a classification label, for figure
    titles and captions meant to be read by a person rather than the code
    that produced them.
    """
    return _PATTERN_LABELS.get(pattern, str(pattern))


def _prominence_score(point: dict) -> float:
    """How clearly a point demonstrates its own test_pattern classification
    -- used so an example trace actually backs up the label it's plotted
    under, rather than an arbitrary or borderline point that only just
    cleared the classification threshold. Each pattern uses whichever
    already-computed scalar most directly reflects "how obviously is this
    happening":
      - bursting: test_bimodality_metric (how separated the short/long ISI
        modes are -- a clean burst, not a marginal one right at
        classify_burst_pattern's threshold).
      - tonic: |test_adaptation_ratio - 1| (bigger deviation from
        non-adapting is more visually interesting/informative, whichever
        direction it goes).
      - silent: depth below rest (test_v_min_mV) -- decisively silenced,
        not just barely quiet. (test_pattern is only ever "tonic",
        "bursting", or "silent" now -- to_stored_pattern in
        find_silencing_threshold.py already collapses ambiguous/sparse
        windows into "silent" before a point is ever stored.)
    Rebound spike count is folded in as a secondary tiebreak whenever
    rebound_applicable, since more rebound spikes makes a rebound_pattern
    label more visibly demonstrated too.
    """
    pattern = point["test_pattern"]
    if pattern == "bursting":
        score = point.get("test_bimodality_metric") or 0.0
    elif pattern == "tonic":
        ratio = point.get("test_adaptation_ratio")
        score = abs(ratio - 1.0) if ratio is not None else 0.0
    elif pattern == "silent":
        v = point.get("test_v_min_mV")
        score = -v if v is not None else 0.0
    else:
        score = 0.0
    if point.get("rebound_applicable"):
        score += 0.01 * (point.get("rebound_spike_count") or 0)
    return score


def _prefer(pts: list, predicate) -> list:
    """Restricts `pts` to those satisfying `predicate`, falling back to the
    full pool if none do -- shared soft-preference pattern: apply a
    preference, but never let it eliminate a category that has no point
    satisfying it at all.
    """
    preferred = [(k, p) for k, p in pts if predicate(k, p)]
    return preferred if preferred else pts


def _pick_best(pts: list, score_fn, require_held_lt_injected: bool,
               held_levels: list = None, injected_levels: list = None) -> tuple:
    """Shared by select_exemplar_points and select_exemplar_by_category.
    Applies two soft preferences (each falling back to the full pool if it
    would eliminate every candidate), then picks the single (key, point)
    scoring highest under score_fn among what's left:

    1. Interior points (neither held_nA nor injected_nA at the swept grid's
       own outer edge) over border points -- a marked point sitting exactly
       on the heatmap's edge is easy to miss/misread (user-flagged
       2026-08-21). Only applied when held_levels/injected_levels are given.
    2. held_nA < injected_nA (the test window releases somewhat from the
       held baseline, rather than deepening it) -- confirmed directly this
       is where illustrative/interesting dynamics concentrate across a
       randomly-sampled population; the opposite direction rarely shows
       anything worth spotlighting.
    """
    pool = pts
    if held_levels is not None and injected_levels is not None:
        held_min, held_max = held_levels[0], held_levels[-1]
        inj_min, inj_max = injected_levels[0], injected_levels[-1]
        pool = _prefer(pool, lambda k, p: k[0] not in (held_min, held_max)
                       and k[1] not in (inj_min, inj_max))
    if require_held_lt_injected:
        pool = _prefer(pool, lambda k, p: k[0] < k[1])
    return sorted(pool, key=lambda kp: (-score_fn(kp[1]), kp[0]))[0]


def select_exemplar_points(cell_result: dict, require_held_lt_injected: bool = True) -> dict:
    """One representative point per distinct (test_pattern, rebound_pattern)
    combination actually present in this cell's grid, chosen by
    _prominence_score to most clearly demonstrate that combination rather
    than an arbitrary or borderline point -- so the resulting trace
    actually backs up its own label. See select_exemplar_by_category for the
    single-dimension version (one representative per test_pattern value
    alone, or per rebound_pattern value alone, ignoring the other axis).
    """
    grid = cell_result["grid"]
    held_levels = list(cell_result["held_levels_nA"])
    injected_levels = list(cell_result["injected_levels_nA"])
    candidates: dict = {}
    for key, point in grid.items():
        if point["blew_up"] or point["test_pattern"] is None:
            continue
        combo = (point["test_pattern"], point["rebound_pattern"])
        candidates.setdefault(combo, []).append((key, point))
    return {combo: _pick_best(pts, _prominence_score, require_held_lt_injected, held_levels, injected_levels)
           for combo, pts in candidates.items()}


def select_exemplar_by_category(cell_result: dict, category_of, score_of, exclude_categories: frozenset = frozenset(),
                                require_held_lt_injected: bool = True) -> dict:
    """One representative point per distinct value of category_of(point)
    actually present in this cell's grid -- unlike select_exemplar_points
    (which groups by the JOINT (test_pattern, rebound_pattern) combination),
    this groups by a single dimension alone, for a page that showcases every
    mode of ONE heatmap panel (e.g. every test_pattern value, or every
    rebound_pattern value) rather than every combination of two.

    exclude_categories drops values that mean "nothing to report" (e.g.
    rebound_pattern's "none"/"not_applicable" -- matching
    REBOUND_BLANK_LABELS, the same categories run_held_injected_grid.py's
    _panel_categorical renders as blank rather than a real color) --
    there's no trace that meaningfully demonstrates "no rebound happened."
    category_of is responsible for its own applicability gating (e.g.
    returning None for a point where rebound_pattern isn't meaningful) so
    the category selected here always matches what the corresponding
    heatmap panel's own value_map would show for that point.
    """
    grid = cell_result["grid"]
    held_levels = list(cell_result["held_levels_nA"])
    injected_levels = list(cell_result["injected_levels_nA"])
    candidates: dict = {}
    for key, point in grid.items():
        if point["blew_up"]:
            continue
        category = category_of(point)
        if category is None or category in exclude_categories:
            continue
        candidates.setdefault(category, []).append((key, point))
    return {category: _pick_best(pts, score_of, require_held_lt_injected, held_levels, injected_levels)
           for category, pts in candidates.items()}


def rebound_prominence_score(point: dict) -> float:
    """How clearly a point demonstrates its own rebound_pattern -- more
    rebound spikes is more visibly demonstrative, mirroring
    _prominence_score's logic for test_pattern but keyed on the rebound side
    instead (test_pattern's own bimodality/adaptation-ratio criteria don't
    apply here: a point's rebound_pattern is independent of what it did
    during the test window itself).
    """
    return point.get("rebound_spike_count") or 0


def resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA, cell_result, hold_tail_s,
                     ss_entry=None) -> dict:
    """Re-run hold-settle -> (short plotting-context hold tail) -> test ->
    recovery for one specific grid point, using the exact same run_args the
    original grid sweep used for this cell, and capturing full traces.

    ss_entry (the steady-state cache entry y_ss/baseline_freq_hz were
    themselves read from) is optional only for backward compatibility with
    existing call sites that don't need hold_pattern accuracy for held=0 --
    every current call site already has it at hand and passes it. Without
    it, held=0 falls back to hold_pattern=None, which would make
    run_test_and_recovery's test_pattern != hold_pattern check spuriously
    fire for every held=0 point (None never equals a real pattern string).
    """
    run_args = cell_result["run_args"]
    dt, temp, reftemp = run_args["dt"], run_args["temp"], run_args["reftemp"]
    settle_kwargs = dict(chunk_s=run_args["hold_settle_chunk_s"], max_settle_s=run_args["max_hold_settle_s"],
                         settle_rtol=run_args["hold_settle_rtol"], min_peaks_for_rate=run_args["min_peaks_for_rate"],
                         min_isis_for_burst_test=run_args["min_isis_for_burst_test"],
                         isi_mode_prominence_frac=run_args["isi_mode_prominence_frac"],
                         min_isi_ratio=run_args["min_isi_ratio"])
    isi_kwargs = dict(min_isis_for_burst_test=run_args["min_isis_for_burst_test"],
                      isi_mode_prominence_frac=run_args["isi_mode_prominence_frac"],
                      min_isi_ratio=run_args["min_isi_ratio"])

    if _round_level(held_nA) == 0.0:
        hold_final_state = y_ss.copy()
        hold_freq_hz = baseline_freq_hz
        burnin_v = ss_entry.get("burnin_v") if ss_entry is not None else None
        burnin_t = ss_entry.get("burnin_t") if ss_entry is not None else None
        hold_pattern = classify_hold_pattern(burnin_v, burnin_t, run_args["min_isis_for_burst_test"],
                                             run_args["isi_mode_prominence_frac"], run_args["min_isi_ratio"])
    else:
        hold_result = settle_hold_level(params, held_nA, y_ss.copy(), dt, temp, reftemp, **settle_kwargs)
        if hold_result["blew_up"]:
            return {"blew_up": True, "error": hold_result.get("error"), "stage": "hold"}
        hold_final_state = hold_result["final_state"]
        hold_freq_hz = hold_result["freq_hz"] or 0.0
        hold_pattern = hold_result["hold_pattern"]

    # Short dedicated hold-only segment, purely for plotting context before
    # the test window -- continues at the same held level from the settled
    # state, so it's just a further stretch of the same trajectory.
    Iapp_hold = constant_iapp_func(held_nA)
    try:
        t_hold, states_hold = simulate(params, hold_tail_s, temp, dt=dt, reftemp=reftemp,
                                       cis=hold_final_state, Iapp_func=Iapp_hold)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {"blew_up": True, "error": f"hold tail: {exc}", "stage": "hold_tail"}
    if not np.all(np.isfinite(states_hold)):
        return {"blew_up": True, "error": "hold tail: non-finite trajectory", "stage": "hold_tail"}

    tr = run_test_and_recovery(params, states_hold[-1], held_nA, injected_nA, hold_freq_hz, hold_pattern,
                               dt, temp, reftemp, cell_result["test_window_s"], cell_result["recovery_window_s"],
                               run_args["rebound_latency_min_ms"], return_traces=True, **isi_kwargs)
    if tr["blew_up"]:
        tr.setdefault("stage", "test_or_recovery")
        return tr

    tr["_trace_t_hold_ms"] = t_hold
    tr["_trace_v_hold_mV"] = states_hold[:, V_INDEX]
    return tr


def control_window_s(baseline_freq_hz: float, n_cycles: float = DEFAULT_CONTROL_CYCLES) -> float:
    """Duration of each phase (hold tail, current step, recovery) in the
    zero-current control trace -- see DEFAULT_CONTROL_CYCLES's docstring for
    why this is scaled to the cell's own baseline period rather than reusing
    cell_result's test/recovery windows. Falls back to a fixed duration for
    a cell that's silent at its own baseline (no period to scale from).
    """
    if not baseline_freq_hz:
        return DEFAULT_CONTROL_FALLBACK_WINDOW_S
    return n_cycles / baseline_freq_hz


def resimulate_control_point(params, y_ss, baseline_freq_hz, cell_result, ss_entry,
                             n_cycles: float = DEFAULT_CONTROL_CYCLES) -> dict:
    """Re-simulates the held=0/injected=0 control point (no perturbation at
    all) over a short window scaled to the cell's own baseline period,
    instead of cell_result's test_window_s/recovery_window_s -- those are
    sized for ISI-classification accuracy at arbitrary grid points and would
    cram hundreds of baseline-rate spikes into one figure at a healthy
    firing rate, saturating the trace into an unreadable solid block rather
    than a legible rhythm (user-flagged 2026-08-21).
    """
    window_s = control_window_s(baseline_freq_hz, n_cycles)
    control_cell_result = dict(cell_result, test_window_s=window_s, recovery_window_s=window_s)
    return resimulate_point(params, y_ss, baseline_freq_hz, 0.0, 0.0, control_cell_result,
                            hold_tail_s=window_s, ss_entry=ss_entry)


def _draw_trace_axes(ax_v, ax_i, cell_id: str, held_nA: float, injected_nA: float,
                     test_pattern: str, rebound_pattern: str, tr: dict) -> None:
    """Draws the V(t)/I(t) trace into a caller-supplied pair of axes --
    split out of build_example_trace_figure so a second caller that builds
    its own combined figure (e.g. plot_parameter_trace.py, which puts this
    next to a heatmap panel rather than alone) can reuse the exact same
    drawing code instead of a second copy that could drift out of sync.
    """
    t_hold, v_hold = tr["_trace_t_hold_ms"], tr["_trace_v_hold_mV"]
    t_test, v_test = tr["_trace_t_test_ms"], tr["_trace_v_test_mV"]
    t_rec, v_rec = tr["_trace_t_rec_ms"], tr["_trace_v_rec_mV"]

    dt_ms = t_hold[1] - t_hold[0] if len(t_hold) > 1 else 0.1
    hold_end = t_hold[-1] + dt_ms if len(t_hold) else 0.0
    t_test_off = t_test + hold_end
    test_end = t_test_off[-1] + dt_ms if len(t_test_off) else hold_end
    t_rec_off = t_rec + test_end

    ax_v.plot(t_hold, v_hold, color="gray", lw=0.8, label=f"holding current ({held_nA:.2f} nA)")
    ax_v.plot(t_test_off, v_test, color="firebrick", lw=0.8, label=f"current step ({injected_nA:.2f} nA)")
    ax_v.plot(t_rec_off, v_rec, color="steelblue", lw=0.8,
             label=f"recovery, released to {held_nA:.2f} nA")
    ax_v.axvline(hold_end, color="black", ls=":", lw=1)
    ax_v.axvline(test_end, color="black", ls=":", lw=1)
    ax_v.set_ylabel("membrane potential (mV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_v.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax_v.tick_params(labelsize=TICK_FONTSIZE)
    if test_pattern == "control":
        # No perturbation happens in a control trace -- "during the current
        # step, then rebound" doesn't apply, so this gets its own plain title
        # rather than describe_pattern's phrasing built for a real test/
        # rebound pair.
        ax_v.set_title(f"{cell_id}: {describe_pattern(test_pattern)}", fontsize=TITLE_FONTSIZE)
    else:
        ax_v.set_title(f"{cell_id}: {describe_pattern(test_pattern)} during the current step, then "
                       f"{describe_pattern(rebound_pattern)}", fontsize=TITLE_FONTSIZE)

    t_current = np.concatenate([t_hold, t_test_off, t_rec_off])
    i_current = np.concatenate([np.full_like(t_hold, held_nA),
                                np.full_like(t_test_off, injected_nA),
                                np.full_like(t_rec_off, held_nA)])
    ax_i.plot(t_current, i_current, color="black", lw=1.2)
    ax_i.set_ylabel("applied current (nA)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_i.set_xlabel("time (ms)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_i.tick_params(labelsize=TICK_FONTSIZE)


def build_example_trace_figure(cell_id: str, held_nA: float, injected_nA: float,
                               test_pattern: str, rebound_pattern: str, tr: dict) -> plt.Figure:
    """The actual figure-building logic, split out from plot_example_trace
    so a caller that wants the Figure object itself doesn't have to
    save-then-reopen a file to get it.
    """
    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    _draw_trace_axes(ax_v, ax_i, cell_id, held_nA, injected_nA, test_pattern, rebound_pattern, tr)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return fig


def plot_example_trace(cell_id: str, held_nA: float, injected_nA: float,
                       test_pattern: str, rebound_pattern: str, tr: dict,
                       outdir: Path, command: str, fig_format: str) -> Path:
    fig = build_example_trace_figure(cell_id, held_nA, injected_nA, test_pattern, rebound_pattern, tr)
    fig.text(0.5, 0.01, command, ha="center", va="bottom",
             fontsize=FOOTER_FONTSIZE, family="monospace", color="dimgray", wrap=True)

    cell_dir = outdir / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    # Double-underscore section separators so e.g. test_pattern="bursting" +
    # rebound_pattern="bursting_rebound" can't visually run together into an
    # ambiguous single field (confirmed this was a real readability problem
    # with single-underscore separators before this fix).
    slug = f"held{held_nA:+.2f}__inj{injected_nA:+.2f}__test-{test_pattern}__rebound-{rebound_pattern}"
    outpath = cell_dir / f"{slug}.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)
    return outpath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate illustrative full-trace figures for representative (cell, held, "
                    "injected) grid points, re-simulated on demand since run_held_injected_grid.py "
                    "deliberately does not persist full traces in its cache.")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Cell ID(s) to generate example traces for. Default: a curated set of "
                             f"{len(DEFAULT_CURATED_CELLS)} cells spanning the range of behaviors "
                             "observed in the full grid sweep.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH,
                        help="Path to run_held_injected_grid.py's output cache (input, read-only).")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--hold-tail-s", type=float, default=2.0,
                        help="Duration of hold-only context shown before the test window (s).")
    parser.add_argument("--allow-held-gt-injected", action="store_true",
                        help="Don't restrict exemplar selection to held_nA < injected_nA (test window "
                             "releasing from the held baseline) -- by default this is where illustrative "
                             "dynamics concentrate across a randomly-sampled population, so the opposite "
                             "direction is excluded unless a (pattern, rebound) combination has no point "
                             "at all on the preferred side. Pass this to consider the full grid instead.")
    parser.add_argument("--include-control", action="store_true",
                        help="Also generate a held=0/injected=0 control trace per cell (no perturbation "
                             "at all), shown over a short window scaled to the cell's own baseline period "
                             "(see --control-cycles) rather than the cell's full test/recovery windows -- "
                             "those are sized for ISI-classification accuracy and would cram hundreds of "
                             "baseline-rate spikes into one figure, reading as a saturated block instead "
                             "of a legible rhythm.")
    parser.add_argument("--control-cycles", type=float, default=DEFAULT_CONTROL_CYCLES,
                        help="Number of the cell's own baseline periods shown per phase (hold tail, "
                             "current step, recovery) in the --include-control trace.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    cells_to_run = args.cells or DEFAULT_CURATED_CELLS

    with open(args.grid_cache, "rb") as handle:
        grid_cache = pickle.load(handle)

    figures_dir = Path(args.figures_dir)
    ss_cache_path = Path(args.steady_state_cache)

    total_written, total_failed = 0, 0
    for cell_id in cells_to_run:
        cell_result = grid_cache.get(cell_id)
        if cell_result is None or cell_result["status"] != "ok":
            print(f"{cell_id}: skipped -- not present or not status 'ok' in {args.grid_cache}")
            continue

        params = cell_result["params"]
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        if ss_entry is None:
            print(f"{cell_id}: skipped -- no valid steady-state cache entry")
            continue
        y_ss, baseline_freq_hz = ss_entry["y_ss"], ss_entry["freq_hz"]

        if args.include_control:
            tr = resimulate_control_point(params, y_ss, baseline_freq_hz, cell_result, ss_entry,
                                          n_cycles=args.control_cycles)
            if tr["blew_up"]:
                print(f"  (+0.00, +0.00) [control]: blew up re-simulating "
                     f"({tr.get('stage')}: {tr.get('error')})")
                total_failed += 1
            else:
                outpath = plot_example_trace(cell_id, 0.0, 0.0, "control", "none",
                                             tr, figures_dir, command, args.figure_format)
                print(f"  (+0.00, +0.00) [control] -> {outpath}")
                total_written += 1

        exemplars = select_exemplar_points(cell_result,
                                           require_held_lt_injected=not args.allow_held_gt_injected)
        print(f"{cell_id}: {len(exemplars)} exemplar point(s)")
        for (test_pattern, rebound_pattern), (key, _point) in sorted(exemplars.items()):
            held_nA, injected_nA = key
            tr = resimulate_point(params, y_ss, baseline_freq_hz, held_nA, injected_nA,
                                  cell_result, args.hold_tail_s, ss_entry=ss_entry)
            if tr["blew_up"]:
                print(f"  ({held_nA:+.2f}, {injected_nA:+.2f}) [{test_pattern}/{rebound_pattern}]: "
                     f"blew up re-simulating ({tr.get('stage')}: {tr.get('error')})")
                total_failed += 1
                continue
            outpath = plot_example_trace(cell_id, held_nA, injected_nA, test_pattern, rebound_pattern,
                                         tr, figures_dir, command, args.figure_format)
            print(f"  ({held_nA:+.2f}, {injected_nA:+.2f}) [{test_pattern}/{rebound_pattern}] -> {outpath}")
            total_written += 1

    print(f"\n{total_written} trace figure(s) written to {figures_dir}/, {total_failed} failed to re-simulate.")


if __name__ == "__main__":
    main()
