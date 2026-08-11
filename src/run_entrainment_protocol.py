"""Step 6b: validate predict_entrainment.py's (Step 6a) phase-locking
predictions against real simulation, on a small subset of cells only --
NOT the full population. See predict_entrainment.py's module docstring and
project_simulation_checklist_snapshot memory for why (predict-first
methodology, brute-force reserved for a spot-check subset).

Delivers a periodic train of the SAME brief hyperpolarizing pulse
run_prc_protocol.py already uses (same amplitude, same duration-as-
fraction-of-period convention), once per the cell's own natural period, for
many cycles -- a clean, apples-to-apples test of whether the PRC-based
iterated-map prediction actually holds up under real periodic forcing at
that same pulse character, before trying anything more elaborate (a more
biophysically realistic IPSC-shaped waveform is a natural follow-up, not
attempted here).

Natural period, baseline pattern, and the phase-0 reference state are all
RE-DERIVED here via the same characterize_baseline/pick_reference_anchor
functions run_prc_protocol.py uses (imported directly, not reimplemented)
rather than trusting cell_prc.pkl's cached values blindly -- matching this
project's established convention of each script re-measuring anything
load-bearing with its own consistent spike-detection convention (see
run_prc_protocol.py's own module docstring re: not trusting generate_
steady_state.py's independent period estimate for its own phase-shift
timing).

Per-cycle analysis: after the pulse in cycle i releases (at
i*period_ms + pulse_duration_ms), find the first qualifying response event
(first spike for a tonic baseline, first spike of the next burst for a
bursting baseline -- using the SAME burst_isi_threshold_ms threshold
computed once from the baseline recording, exactly as run_prc_protocol.py's
find_next_event_ms does) and record its phase within that forcing cycle:
    response_phase_i = (event_time_i - i*period_ms) / period_ms
After discarding an initial transient (--n-transient-cycles), the sequence
of response_phase_i across the remaining cycles is checked for convergence
to a stable value (observed 1:1 lock) vs. continued drift/variation (no
observed lock) -- see classify_observed_lock.
"""

import os
import pickle
import sys
import time
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH, get_cached_state
from run_prc_protocol import (characterize_baseline, pick_reference_anchor, detect_spike_times_ms,
                              DEFAULT_GRID_FEATURES_CACHE_PATH, load_grid_features_cache)
from predict_entrainment import DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_PREDICTION_CACHE_PATH

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_entrainment_validation.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "entrainment_validation"
DEFAULT_FIGURE_FORMAT = "svg"
DEFAULT_PRC_CACHE_PATH = ROOT_DIR / "cell_prc.pkl"

# Same Q10-inert convention as every other protocol in this project.
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

V_INDEX = 0

# A response-phase sequence (after the transient) is called "locked" if its
# circular spread (max-min, handling phi=1/phi=0 wraparound the same way
# every phase quantity in this project's PRC work is handled) stays below
# this tolerance across all analyzed cycles.
#
# RECALIBRATED (2026-08-08) from an initial, deliberately conservative 0.02
# after looking at real spread values across the first 22 validated cells:
# the distribution is sharply bimodal, NOT a smooth gradient -- 19/22 cells
# cluster with spread < 0.15 (a real, repeatable phase relationship, just
# noisier than an idealized numerical fixed point), while genuinely
# unlocked/drifting cells (DFJ1K3, MXIC4S, QUXTOD) land at spread > 1.0
# (multiple full cycles of drift) with nothing in between. 0.02 was
# splitting hairs inside the "genuinely consistent" cluster rather than
# separating real signal from real noise. 0.15 sits well inside the gap
# between the two clusters on the real data seen so far -- reported
# honestly as a post-hoc, data-informed choice, not a value fixed before
# ever seeing results.
LOCK_TOLERANCE = 0.15


def make_periodic_pulse_iapp_func(period_ms: float, pulse_duration_ms: float, amplitude_nA: float):
    def f(t):
        phase_in_cycle_ms = t % period_ms
        return amplitude_nA if phase_in_cycle_ms < pulse_duration_ms else 0.0
    return f


def find_cycle_response(spike_times_ms: np.ndarray, cycle_start_ms: float, pulse_duration_ms: float,
                        period_ms: float, pattern: str, burst_isi_threshold_ms,
                        prev_event_ms) -> "float | None":
    """First qualifying response event (see module docstring) after this
    cycle's pulse releases. Searches a generous window (up to the START of
    the NEXT cycle's pulse, i.e. almost the full remaining cycle) rather
    than stopping at the nominal cycle boundary, since a delayed response
    can legitimately spill past where the next pulse would nominally begin
    -- if it does, that overlap is itself informative (the cell hasn't
    settled to a clean one-response-per-cycle pattern) and is captured by
    returning a phase >= 1.0, not silently dropped.
    """
    release_ms = cycle_start_ms + pulse_duration_ms
    window = spike_times_ms[spike_times_ms >= release_ms]
    if len(window) == 0:
        return None
    if pattern == "tonic":
        return float(window[0])
    # Bursting: skip spikes that are still part of a burst already underway
    # at release (ISI from prev_event_ms below threshold), same convention
    # as run_prc_protocol.py's find_next_event_ms.
    prev = prev_event_ms if prev_event_ms is not None else -np.inf
    for st in window:
        if st - prev > burst_isi_threshold_ms:
            return float(st)
        prev = st
    return None


def classify_observed_lock(response_phases: np.ndarray, tolerance: float = LOCK_TOLERANCE) -> dict:
    """Circular spread of the (already phase[0,~) valued, possibly >1 for
    spillover cycles -- see find_cycle_response) response-phase sequence.
    Cycles with no detected response (NaN) are excluded from the spread
    calculation but counted and reported, since a cell that often fails to
    respond at all is a distinct, meaningful outcome from one that responds
    but at a drifting phase.
    """
    valid = response_phases[~np.isnan(response_phases)]
    n_missing = int(np.isnan(response_phases).sum())
    if len(valid) < 2:
        return {"locked": False, "reason": "insufficient_responses", "n_missing": n_missing,
               "mean_phase": None, "spread": None}
    spread = float(valid.max() - valid.min())
    locked = spread <= tolerance
    return {"locked": locked, "reason": "ok", "n_missing": n_missing,
           "mean_phase": float(valid.mean()), "spread": spread}


# ---------------------------------------------------------------------------
# Top-level per-cell orchestrator
# ---------------------------------------------------------------------------

def run_cell_entrainment(cell_id: str, params: np.ndarray, ss_entry, fi_features_entry, args) -> tuple:
    cell_floor_nA = fi_features_entry.get("cell_floor_nA") if fi_features_entry else None
    if args.pulse_amplitude_frac_of_floor is not None and cell_floor_nA is not None:
        pulse_amplitude_nA = args.pulse_amplitude_frac_of_floor * cell_floor_nA
        pulse_amplitude_source = "frac_of_floor"
    else:
        pulse_amplitude_nA = args.pulse_amplitude_nA
        pulse_amplitude_source = "fixed_nA"

    base = {"cell_id": cell_id, "dt": args.dt, "temp": args.temp, "reftemp": args.reftemp,
           "pulse_amplitude_nA": pulse_amplitude_nA, "pulse_amplitude_source": pulse_amplitude_source,
           "cell_floor_nA": cell_floor_nA,
           "pulse_duration_frac_of_period": args.pulse_duration_frac_of_period,
           "n_cycles": args.n_cycles, "n_transient_cycles": args.n_transient_cycles}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}, None

    y_ss = ss_entry["y_ss"]
    initial_period_ms = ss_entry["period_ms"]

    baseline = characterize_baseline(params, y_ss, args.dt, args.temp, args.reftemp, initial_period_ms,
                                     args.min_isis_for_burst_test, args.isi_mode_prominence_frac,
                                     args.min_isi_ratio, args.n_baseline_cycles, args.max_baseline_s,
                                     args.baseline_extend_factor)
    if baseline["status"] != "ok":
        return {**base, "status": baseline["status"], "error": baseline.get("error")}, None

    y_ref, _t_ref, _control_next = pick_reference_anchor(baseline)
    period_ms = baseline["natural_period_ms"]
    pattern = baseline["pattern"]
    burst_isi_threshold_ms = baseline["burst_isi_threshold_ms"]
    pulse_duration_ms = args.pulse_duration_ms_fixed or (args.pulse_duration_frac_of_period * period_ms)

    Iapp_func = make_periodic_pulse_iapp_func(period_ms, pulse_duration_ms, pulse_amplitude_nA)
    horizon_s = args.n_cycles * period_ms / 1000.0
    try:
        t_ms, states = simulate(params, horizon_s, args.temp, dt=args.dt, reftemp=args.reftemp,
                                cis=y_ref, Iapp_func=Iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {**base, "status": "blew_up", "error": str(exc)}, None
    if not np.all(np.isfinite(states)):
        return {**base, "status": "blew_up", "error": "entrainment trial: non-finite trajectory"}, None

    v = states[:, V_INDEX]
    spike_times_ms = detect_spike_times_ms(v, t_ms)

    response_phases = np.full(args.n_cycles, np.nan)
    prev_event_ms = None
    for i in range(args.n_cycles):
        cycle_start_ms = i * period_ms
        event_ms = find_cycle_response(spike_times_ms, cycle_start_ms, pulse_duration_ms, period_ms,
                                       pattern, burst_isi_threshold_ms, prev_event_ms)
        if event_ms is not None:
            response_phases[i] = (event_ms - cycle_start_ms) / period_ms
            prev_event_ms = event_ms

    analyzed_phases = response_phases[args.n_transient_cycles:]
    lock_result = classify_observed_lock(analyzed_phases, args.lock_tolerance)

    result = {
        **base, "status": "ok", "baseline_pattern": pattern, "natural_period_ms": period_ms,
        "pulse_duration_ms": pulse_duration_ms,
        "response_phases": response_phases, "analyzed_response_phases": analyzed_phases,
        "observed_lock": lock_result["locked"], "observed_lock_reason": lock_result["reason"],
        "observed_locked_phase": lock_result["mean_phase"], "observed_phase_spread": lock_result["spread"],
        "n_missing_responses": lock_result["n_missing"],
    }
    trace = {"t_ms": t_ms, "v": v, "period_ms": period_ms, "pulse_duration_ms": pulse_duration_ms,
            "n_cycles": args.n_cycles} if args.return_traces else None
    return result, trace


# ---------------------------------------------------------------------------
# Cache I/O -- write-then-rename + retry, this project's standard pattern
# ---------------------------------------------------------------------------

def load_output_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(cache, handle)
    for attempt in range(5):
        try:
            os.replace(tmp_path, cache_path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def load_prediction_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_cell_entrainment(result: dict, trace, prediction: dict, outdir: Path, command: str,
                          fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    if result["status"] != "ok":
        return
    cell_id = result["cell_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    ncols = 2 if trace is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    cycles = np.arange(result["n_cycles"])
    ax.plot(cycles, result["response_phases"], "o-", color="steelblue", ms=4, lw=1)
    ax.axvspan(-0.5, result["n_transient_cycles"] - 0.5, color="gray", alpha=0.15, label="transient (excluded)")

    # observed_phase_spread trace annotation: shade [min, max] of the same
    # post-transient response phases classify_observed_lock computed the
    # spread from, so the printed number is directly traceable to the
    # scatter it summarizes rather than only asserted in a cache field.
    spread = result.get("observed_phase_spread")
    if spread is not None:
        analyzed = result["response_phases"][result["n_transient_cycles"]:]
        valid = analyzed[~np.isnan(analyzed)]
        lo, hi = float(valid.min()), float(valid.max())
        ax.axhspan(lo, hi, color="mediumpurple", alpha=0.15,
                  label=f"post-transient spread = {spread:.3f}")

    if result["observed_lock"]:
        ax.axhline(result["observed_locked_phase"], color="mediumseagreen", ls="--", lw=1,
                  label=f"observed lock @ φ={result['observed_locked_phase']:.3f}")
    pred_phase = prediction.get("predicted_locked_phase") if prediction else None
    if pred_phase is not None:
        ax.axhline(pred_phase, color="darkorange", ls=":", lw=1.3,
                  label=f"predicted lock @ φ={pred_phase:.3f}")
    ax.set_xlabel("forcing cycle index")
    ax.set_ylabel("response phase within cycle (φ)")
    ax.set_title(f"{cell_id} — entrainment ({result['baseline_pattern']} baseline); "
                f"observed: {'LOCKED' if result['observed_lock'] else 'not locked'} "
                f"({result['observed_lock_reason']})", fontsize=9)
    ax.legend(loc="best", fontsize=6.5)

    if trace is not None:
        ax2 = axes[1]
        show_n = min(6, trace["n_cycles"])
        mask = trace["t_ms"] <= show_n * trace["period_ms"]
        ax2.plot(trace["t_ms"][mask], trace["v"][mask], color="steelblue", lw=0.6)
        for i in range(show_n):
            ax2.axvspan(i * trace["period_ms"], i * trace["period_ms"] + trace["pulse_duration_ms"],
                       color="firebrick", alpha=0.25)
        ax2.set_xlabel("time (ms)")
        ax2.set_ylabel("V (mV)")
        ax2.set_title(f"first {show_n} forcing cycles (shaded = pulse)", fontsize=9)

    if result.get("pulse_amplitude_source") == "frac_of_floor" and result.get("cell_floor_nA"):
        pulse_source_note = f"{result['pulse_amplitude_nA'] / result['cell_floor_nA'] * 100:.0f}% of cell_floor_nA={result['cell_floor_nA']:.2f} nA"
    else:
        pulse_source_note = "fixed nA"
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.text(0.5, 0.045, f"pulse: {result['pulse_amplitude_nA']:.2f} nA ({pulse_source_note}) x "
             f"{result['pulse_duration_ms']:.2f} ms/cycle, period={result['natural_period_ms']:.2f} ms",
             ha="center", va="bottom", fontsize=6, style="italic", color="dimgray", wrap=True)
    fig.text(0.5, 0.01, command, ha="center", va="bottom", fontsize=6, family="monospace",
             color="dimgray", wrap=True)
    fig.savefig(outdir / f"{cell_id}_entrainment.{fig_format}", format=fig_format, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--cells", nargs="+", required=True,
                        help="Cell ID(s) to validate. Required (not defaulted to 'all') -- this "
                             "script is for the small spot-check subset only, see module docstring.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH)
    parser.add_argument("--prediction-cache", default=DEFAULT_PREDICTION_CACHE_PATH,
                        help="Step 6a's predictions (read-only, used only to overlay the predicted "
                             "phase on each figure and in the comparison summary).")
    parser.add_argument("--grid-features-cache", default=DEFAULT_GRID_FEATURES_CACHE_PATH,
                        help="Read-only source of each cell's cell_floor_nA, only consulted when "
                             "--pulse-amplitude-frac-of-floor is set.")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--jobs", type=int, default=-1)

    parser.add_argument("--pulse-amplitude-nA", type=float, default=-1.0,
                        help="Same default as run_prc_protocol.py -- see module docstring for why "
                             "this script deliberately matches that convention. Used directly unless "
                             "--pulse-amplitude-frac-of-floor is set, in which case it's only the "
                             "fallback for cells with no grid_features_cache entry.")
    parser.add_argument("--pulse-amplitude-frac-of-floor", type=float, default=None,
                        help="Same option/reasoning as run_prc_protocol.py's flag of the same name -- "
                             "see there for why a fixed absolute nA pulse is an arbitrary fraction of "
                             "different cells' own dynamic range (5%%-400%%+ of cell_floor_nA across "
                             "the population). Falls back to --pulse-amplitude-nA for any cell "
                             "missing a grid_features_cache entry.")
    parser.add_argument("--pulse-duration-frac-of-period", type=float, default=0.05)
    parser.add_argument("--pulse-duration-ms-fixed", type=float, default=None)
    parser.add_argument("--n-cycles", type=int, default=20,
                        help="Total forcing cycles to simulate, including the transient.")
    parser.add_argument("--n-transient-cycles", type=int, default=3,
                        help="Initial cycles excluded from the lock/no-lock analysis, to let any "
                             "startup transient settle before judging convergence.")
    parser.add_argument("--lock-tolerance", type=float, default=LOCK_TOLERANCE,
                        help="Max response-phase spread (across analyzed cycles) to call the "
                             "sequence 'locked' -- see LOCK_TOLERANCE's module-level comment for "
                             "the 2026-08-08 recalibration from an initial 0.02 to 0.15.")

    parser.add_argument("--n-baseline-cycles", type=float, default=8.0)
    parser.add_argument("--max-baseline-s", type=float, default=10.0)
    parser.add_argument("--baseline-extend-factor", type=float, default=2.0)
    parser.add_argument("--min-isis-for-burst-test", type=int, default=6)
    parser.add_argument("--isi-mode-prominence-frac", type=float, default=0.05)
    parser.add_argument("--min-isi-ratio", type=float, default=1.5)
    parser.add_argument("--return-traces", action="store_true", default=True,
                        help="Keep the raw trace for plotting (default on -- this script only ever "
                             "runs on a handful of cells, unlike the population-scale protocols "
                             "where full traces are deliberately discarded to bound cache size).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    figures_dir = Path(args.figures_dir)

    from steady_state_cache import load_all_cells
    all_cells = load_all_cells(params_dir)
    unknown = sorted(set(args.cells) - set(all_cells))
    if unknown:
        raise SystemExit(f"Unknown cell id(s): {unknown}")
    cells = {cid: all_cells[cid] for cid in args.cells}

    print(f"Running entrainment validation for {len(cells)} cell(s): {sorted(cells)}")

    prediction_cache = load_prediction_cache(Path(args.prediction_cache))
    grid_features_cache = load_grid_features_cache(Path(args.grid_features_cache))
    output_cache = load_output_cache(Path(args.output_cache))

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=Path(args.steady_state_cache))
        fi_features_entry = grid_features_cache.get(cell_id)
        result, trace = run_cell_entrainment(cell_id, params, ss_entry, fi_features_entry, args)
        return cell_id, result, trace

    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items())

    comparison_rows = []
    for cell_id, result, trace in result_iter:
        output_cache[cell_id] = result
        prediction = prediction_cache.get(cell_id, {})
        if not args.no_plot:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plot_cell_entrainment(result, trace, prediction, figures_dir, command, args.figure_format)
        save_output_cache(output_cache, Path(args.output_cache))
        print(f"  {cell_id}: {result['status']}"
             + (f" (observed {'LOCK' if result.get('observed_lock') else 'no lock'})"
                if result["status"] == "ok" else ""))
        comparison_rows.append((cell_id, result, prediction))

    print(f"\nCache written to {args.output_cache}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    # Phase D: predicted vs observed comparison, printed directly here
    # rather than as a separate script -- see module docstring.
    print("\nPrediction vs. observation:")
    print(f"{'cell_id':<10} {'predicted':<14} {'pred_phase':>10}   {'observed':<10} {'obs_phase':>10}   agree?")
    n_agree = 0
    n_comparable = 0
    for cell_id, result, prediction in comparison_rows:
        if result["status"] != "ok":
            print(f"{cell_id:<10} status={result['status']}")
            continue
        pred_label = prediction.get("prediction", "no_prediction")
        pred_phase = prediction.get("predicted_locked_phase")
        obs_label = "stable_lock" if result["observed_lock"] else "no_lock"
        obs_phase = result["observed_locked_phase"]
        pred_says_lock = pred_label == "stable_lock"
        lock_agree = pred_says_lock == result["observed_lock"]
        n_comparable += 1
        n_agree += int(lock_agree)
        pred_phase_str = f"{pred_phase:.3f}" if pred_phase is not None else "--"
        obs_phase_str = f"{obs_phase:.3f}" if obs_phase is not None else "--"
        print(f"{cell_id:<10} {pred_label:<14} {pred_phase_str:>10}   {obs_label:<10} {obs_phase_str:>10}   "
             f"{'YES' if lock_agree else 'no'}")
    if n_comparable:
        print(f"\nLock/no-lock agreement: {n_agree}/{n_comparable} ({100.0 * n_agree / n_comparable:.0f}%)")


if __name__ == "__main__":
    main()
