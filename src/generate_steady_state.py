"""Generate a steady-state cache for the cell parameter files in src/models.

Each parameter file must contain exactly 40 whitespace-separated numeric values
in the order expected by src/singlecell_model_v1.py.

This script runs each cell autonomously with Iapp=0, burns in for a few
seconds, estimates the firing period from the end of the trajectory, and
then uses shooting to find a state that returns to itself after one period.
The final state is cached in a project-root pickle file for reuse.

Every cell is simulated exactly ONCE at the given --dt. If a cell blows up,
flatlines, or otherwise fails to reach a converged limit cycle, that's
recorded as its status and the script moves on -- it does NOT retry
automatically at a smaller dt. All cells in src/models are known to fire
spontaneously at Iapp=0, so a non-"ok" status means this script's numerics
need help for that specific cell, not that the cell is really silent.
Rerun just that cell with a smaller dt once you've looked at its figure:

    python generate_steady_state.py --cells CELLID --dt 0.01

By default this runs every cell found in --params-dir; pass --cells to
target one or more specific cells instead (e.g. after fixing up a cell
that failed, to regenerate just its cache entry).
"""

import re
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import DEFAULT_CIS, simulate
from steady_state_cache import load_cache, save_cache

# Canonical set of keys every valid, "ok" steady-state cache entry must
# have -- the single source of truth for what "the current schema" means.
# is_cache_entry_valid checks this directly against actual cache contents at
# load time, instead of via a separately maintained version integer that has
# to be remembered and manually bumped whenever compute_cell_steady_state's
# result shape changes (a real cache entry crashing downstream code, e.g.
# plot_cell_trace reading result["freq_hz"], is the failure this prevents).
# burnin_t is included even though it's not itself persisted to disk --
# steady_state_cache.load_cache reconstructs it on load (see that module),
# so any entry reaching this check already has it back.
RESULT_SCHEMA_KEYS = frozenset({
    "cell_id", "params", "dt", "shoot_dt", "period_ms", "freq_hz", "y_ss",
    "shoot_resid_norm", "v_min_mV", "burnin_t", "burnin_v", "status",
})

DEFAULT_PARAMS_DIR = ROOT_DIR / "src" / "models"
DEFAULT_CACHE_PATH = ROOT_DIR / "cell_steady_states.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "cache"
DEFAULT_FIGURE_FORMAT = "png"
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1
DEFAULT_BURNIN_SECS = 5.0
MIN_PEAKS_FOR_PERIOD = 4
PROMINENCE_FRACTION = 0.3
SPIKE_ZOOM_MS = 50.0

# Cells that need a finer --dt for numerical stability (some require 0.01ms
# vs. the 0.1ms default) were storing their burn-in trace at that same fine
# resolution -- 10x more points than any downstream consumer (the diagnostic
# plot, or find_silencing_threshold.py's ISI-based baseline classification)
# actually needs, since 0.1ms is already this pipeline's standard spike-
# detection resolution everywhere else. Across 69 cells this alone bloated
# cell_steady_states.pkl to 552MB. Downsample the *stored* copy to this
# resolution regardless of the integration dt used; analysis (period
# estimation, shooting) still runs on the full-resolution trace.
MAX_STORED_TRACE_DT_MS = 0.1


def _downsample_trace(t_ms: np.ndarray, voltage: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, round(MAX_STORED_TRACE_DT_MS / dt))
    if stride == 1:
        return t_ms, voltage
    return t_ms[::stride], voltage[::stride]

# Per-variable scale used both to size shooting/least_squares steps and to
# turn a raw 14-vector residual into a single dimensionless number
# comparable against RESID_TOL (state order: V, 12 gating vars, IntCa --
# see singlecell_model_v1's module docstring).
STATE_SCALE = np.array([50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=float)
RESID_TOL = 1e-2
V_INDEX = 0

# Physical bounds on the state vector, used to keep the shooting search
# (see shoot_to_limit_cycle) inside biologically valid territory. Gating
# variables are convex combinations that relax toward a value in [0, 1],
# so they can never leave that range under the real dynamics; IntCa must
# stay strictly positive (CaNernst takes log(CaOut / IntCa)). An
# unconstrained solver is free to probe states outside these bounds while
# searching, which reliably blows up the integrator regardless of whether
# the cell's actual limit cycle is stable -- that was the single biggest
# cause of spurious "blew_up" statuses on cells with a perfectly clean,
# periodic burn-in trace.
STATE_LOWER_BOUNDS = np.array([-100.0] + [0.0] * 12 + [1e-3])
STATE_UPPER_BOUNDS = np.array([60.0] + [1.0] * 12 + [1000.0])

# Shooting needs a tighter per-period residual than burn-in needs for just
# getting onto the attractor, so it integrates at a finer dt than --dt.
# Without this, RESID_TOL sits right at the numerical noise floor of a
# dt=0.1ms RK4 step, and cells with a perfectly good limit cycle get
# flagged as "shooting_failed" by integration error, not real
# non-convergence.
SHOOT_DT_FACTOR = 0.2


def load_cell_params(filepath: Path) -> np.ndarray:
    text = filepath.read_text()
    tokens = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
    params = np.array([float(tok) for tok in tokens], dtype=float)
    if params.size != 40:
        raise ValueError(f"{filepath}: expected 40 parameters, got {params.size}")
    return params


def load_all_cells(params_dir: Path = DEFAULT_PARAMS_DIR) -> dict[str, np.ndarray]:
    params_dir = Path(params_dir)
    cells: dict[str, np.ndarray] = {}
    for filepath in sorted(params_dir.glob("*.txt")):
        cells[filepath.stem] = load_cell_params(filepath)
    return cells


def estimate_period(t_ms: np.ndarray, voltage: np.ndarray) -> float | None:
    peaks, _ = find_peaks(voltage, prominence=(voltage.max() - voltage.min()) * PROMINENCE_FRACTION)
    if len(peaks) < MIN_PEAKS_FOR_PERIOD:
        return None
    peak_times = t_ms[peaks[-MIN_PEAKS_FOR_PERIOD:]]
    return float(np.mean(np.diff(peak_times)))


def integrate_ms(params: np.ndarray,
                 y0: np.ndarray,
                 ms_duration: float,
                 temp: float = DEFAULT_TEMP,
                 reftemp: float = DEFAULT_REFTEMP,
                 dt: float = DEFAULT_DT_MS,
                 iapp_func=None):
    """Run one integration and verify the trajectory stayed finite.

    Raises RuntimeError on a blown-up (non-finite) trajectory or any
    exception from the integrator itself. compute_cell_steady_state catches
    this and records the cell's status as "blew_up" -- it does not retry.
    """
    try:
        nsecs = ms_duration / 1000.0
        t_ms, states = simulate(params, nsecs, temp, dt=dt,
                                 reftemp=reftemp, cis=y0,
                                 Iapp_func=iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise RuntimeError(f"dt={dt:.6f}: {exc}") from exc

    if not np.all(np.isfinite(states)):
        raise RuntimeError(f"dt={dt:.6f}: non-finite trajectory")

    return t_ms, states


def shoot_to_limit_cycle(params: np.ndarray,
                          y0: np.ndarray,
                          period_ms: float,
                          temp: float = DEFAULT_TEMP,
                          reftemp: float = DEFAULT_REFTEMP,
                          dt: float = DEFAULT_DT_MS,
                          iapp_func=None,
                          xtol: float = 1e-6,
                          max_nfev: int = 5000) -> tuple[np.ndarray, bool, float]:
    """Find y such that integrating for one period returns to y.

    Bounded to STATE_LOWER_BOUNDS/STATE_UPPER_BOUNDS via least_squares --
    unlike fsolve, this keeps the search from ever proposing a state that
    doesn't exist biologically (negative calcium, out-of-range gating
    variables), which used to blow up the integrator independent of
    whether the cell's real limit cycle is stable. We deliberately do NOT
    catch a RuntimeError from a bounded probe here -- it still propagates
    out to compute_cell_steady_state as "blew_up" for this cell, but bounds
    should make that the rare case rather than the common one.
    """
    def residual(y: np.ndarray) -> np.ndarray:
        _, states = integrate_ms(params, y, period_ms, temp, reftemp, dt, iapp_func)
        return states[-1] - y

    y0_clipped = np.clip(y0, STATE_LOWER_BOUNDS, STATE_UPPER_BOUNDS)
    result = least_squares(residual, y0_clipped,
                            bounds=(STATE_LOWER_BOUNDS, STATE_UPPER_BOUNDS),
                            x_scale=STATE_SCALE, xtol=xtol, max_nfev=max_nfev)
    y_ss = result.x
    scaled_norm = np.linalg.norm(result.fun / STATE_SCALE)
    converged = scaled_norm < RESID_TOL
    return y_ss, converged, scaled_norm


def compute_cell_steady_state(cell_id: str,
                              params: np.ndarray,
                              temp: float = DEFAULT_TEMP,
                              reftemp: float = DEFAULT_REFTEMP,
                              dt: float = DEFAULT_DT_MS,
                              burnin_secs: float = DEFAULT_BURNIN_SECS) -> dict:
    """One attempt, at the given dt: burn in, estimate the firing period,
    shoot for the limit cycle, and verify it holds over two periods. Any
    failure (a blow-up, a flatlined trace, no detectable period, or a
    non-converged shot) is reported as a status string rather than raised
    or retried -- rerun this cell by name with a smaller --dt if needed.
    """
    shoot_dt = dt * SHOOT_DT_FACTOR
    base = {"cell_id": cell_id, "params": params, "dt": dt, "shoot_dt": shoot_dt,
            "period_ms": None, "freq_hz": None, "y_ss": None,
            "shoot_resid_norm": None, "v_min_mV": None,
            "burnin_t": None, "burnin_v": None}

    try:
        t_ms, states = integrate_ms(params, DEFAULT_CIS.copy(),
                                     burnin_secs * 1000.0,
                                     temp, reftemp, dt, iapp_func=None)
    except RuntimeError as exc:
        return {**base, "status": "blew_up", "error": str(exc)}

    voltage = states[:, V_INDEX]
    base["burnin_t"], base["burnin_v"] = _downsample_trace(t_ms, voltage, dt)

    if voltage.max() - voltage.min() < 1.0:
        # All cells in src/models are known to spike at Iapp=0, so a
        # flatlined trace here means the integration damped out the spikes
        # (dt too coarse) -- not a genuinely quiescent cell.
        return {**base, "status": "silent"}

    period_ms = estimate_period(t_ms, voltage)
    if period_ms is None:
        return {**base, "status": "non_periodic"}

    base["period_ms"] = period_ms
    base["freq_hz"] = 1000.0 / period_ms
    # Trough of the last full oscillation cycle -- unlike a single point on
    # the shot limit cycle, this is a stable, physiologically meaningful
    # reference (useful later e.g. for measuring I_h-mediated sag against).
    tail_mask = t_ms >= (t_ms[-1] - period_ms)
    base["v_min_mV"] = float(voltage[tail_mask].min())

    y_start = states[-1]
    try:
        y_ss, converged, resid_norm = shoot_to_limit_cycle(params, y_start,
                                                           period_ms, temp,
                                                           reftemp, shoot_dt,
                                                           iapp_func=None)
    except RuntimeError as exc:
        return {**base, "status": "blew_up", "error": str(exc)}

    base["shoot_resid_norm"] = resid_norm
    if not converged:
        return {**base, "status": "shooting_failed"}

    # NOTE: we deliberately don't re-verify by integrating further periods
    # and checking for drift back to y_ss. period_ms is only precise to
    # about the burn-in dt's grid resolution (it comes from averaging
    # peak-to-peak timing), so a state that satisfies the periodicity
    # equation for that slightly-imprecise period will appear to "drift"
    # after N periods purely from accumulated phase error -- and because V
    # moves fast during the spike upstroke, even a tiny phase error looks
    # like a large state-space distance there. shoot_resid_norm is already
    # the correct, direct measure of whether y_ss solves the periodicity
    # equation; a second check built on the same imprecise period doesn't
    # add signal, it adds phase-drift noise.
    return {**base, "status": "ok", "y_ss": y_ss}


def is_cache_entry_valid(entry: dict,
                         params: np.ndarray) -> bool:
    cached_params = entry.get("params")
    return (entry.get("status") == "ok"
            and RESULT_SCHEMA_KEYS.issubset(entry.keys())
            and cached_params is not None
            and np.array_equal(cached_params, params))


def plot_cell_trace(result: dict, outdir: Path, command: str,
                    fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    cell_id = result["cell_id"]
    t_ms = result["burnin_t"]
    voltage = result["burnin_v"]
    if t_ms is None or voltage is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(t_ms / 1000.0, voltage, color="steelblue", lw=0.7)
    title = (f"{cell_id} — status: {result['status']} "
            f"(dt={result['dt']:.4g} ms, shoot_dt={result['shoot_dt']:.4g} ms)")
    if result["period_ms"] is not None:
        title += f", period={result['period_ms']:.1f} ms ({result['freq_hz']:.2f} Hz)"
    axes[0].set_title(title, fontsize=9)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("V (mV)")
    if result["v_min_mV"] is not None:
        axes[0].axhline(result["v_min_mV"], color="firebrick", ls="--", lw=1,
                       label="min V (last cycle)")
        axes[0].legend(loc="upper right", fontsize=7)

    # Fixed zoom on the last SPIKE_ZOOM_MS of the trace, regardless of
    # whether a period was detected -- this is what lets you see (and,
    # for failed cells, debug) individual spikes rather than a burst
    # envelope, and it's what the reported freq_hz is meant to correspond to.
    mask = t_ms >= (t_ms[-1] - SPIKE_ZOOM_MS)
    axes[1].plot(t_ms[mask], voltage[mask], color="steelblue", lw=0.9)
    zoom_title = f"Last {SPIKE_ZOOM_MS:.0f} ms"
    if result["freq_hz"] is not None:
        zoom_title += f" — {result['freq_hz']:.2f} Hz"
    axes[1].set_title(zoom_title, fontsize=9)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("V (mV)")
    if result["v_min_mV"] is not None:
        axes[1].axhline(result["v_min_mV"], color="firebrick", ls="--", lw=1,
                       label="min V (last cycle)")
        axes[1].legend(loc="upper right", fontsize=7)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(0.5, 0.01, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_burnin_trace.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a steady-state cache for the cell models in src/models."
    )
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR,
                        help="Directory containing 40-value cell parameter files.")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) (parameter filename stem, e.g. "
                             "0EFZ6A) to simulate. Default: every cell found in "
                             "--params-dir.")
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_PATH,
                        help="Path to the output pickle cache file.")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR,
                        help="Directory for diagnostic burn-in figures.")
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT,
                        choices=["svg", "pdf"],
                        help="Output format for diagnostic figures. Vector formats preserve zoom quality.")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Simulation temperature in deg C.")
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP,
                        help="Reference temperature for q10 scaling.")
    parser.add_argument("--burnin-secs", type=float, default=DEFAULT_BURNIN_SECS,
                        help="Burn-in duration in seconds before period estimation.")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS,
                        help="Integration timestep in ms. Each cell is simulated once "
                             "at this dt -- if a cell blows up, rerun just that cell "
                             "(via --cells) with a smaller --dt.")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel workers (-1 uses all cores).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip diagnostic figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    cache_path = Path(args.cache_file)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_cells = load_all_cells(params_dir)
    if args.cells:
        unknown = sorted(set(args.cells) - set(all_cells))
        if unknown:
            raise SystemExit(f"Unknown cell id(s): {unknown}\n"
                              f"Available cells in {params_dir}: {sorted(all_cells)}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Simulating {len(cells)} of {len(all_cells)} cell parameter file(s) "
          f"from {params_dir} at dt={args.dt} ms")

    cache = load_cache(cache_path)

    def process(cell_id: str, params: np.ndarray):
        existing = cache.get(cell_id)
        if existing is not None and is_cache_entry_valid(existing, params):
            return cell_id, existing, False
        result = compute_cell_steady_state(cell_id, params,
                                          temp=args.temp,
                                          reftemp=args.reftemp,
                                          dt=args.dt,
                                          burnin_secs=args.burnin_secs)
        return cell_id, result, True

    if args.jobs == 1:
        results = [process(cid, params) for cid, params in cells.items()]
    else:
        results = Parallel(n_jobs=args.jobs)(
            delayed(process)(cid, params) for cid, params in cells.items()
        )

    status_counts: dict[str, int] = {}
    for _, result, is_new in results:
        if is_new:
            cache[result["cell_id"]] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if not args.no_plot:
            plot_cell_trace(result, figures_dir, command, fig_format=args.figure_format)

    save_cache(cache, cache_path)

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\nCache written to {cache_path}")
    if not args.no_plot:
        print(f"Trace figures written to {figures_dir}/")

    flagged = [result["cell_id"] for _, result, _ in results if result["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok' at dt={args.dt} ms: {flagged}")
        print("All cells in src/models are known to fire spontaneously at Iapp=0, so this "
              "is a simulation issue for these cells, not real biology. Check their figures "
              f"in {figures_dir}/, then rerun just that cell with a smaller --dt, e.g.:\n"
              f"    python generate_steady_state.py --cells {flagged[0]} --dt {args.dt / 10}")


if __name__ == "__main__":
    main()
