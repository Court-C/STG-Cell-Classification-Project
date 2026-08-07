"""Triangular ramp protocol (Step 3): slow hyperpolarizing down-sweep then
up-sweep, to characterize path-dependence (hysteresis) that a fixed-level
step protocol (run_held_injected_grid.py) structurally cannot reveal -- a
step probes one Iapp level at a time from a reset starting condition, so it
has no way to ask "does the cell's response at a given current level depend
on which direction it got there from." Inhibitory (Iapp <= 0) direction
only, matching the whole project's current scope.

Protocol per cell: hold at 0 (warm-started from the cell's own cached
Iapp=0 limit-cycle state, see steady_state_cache.py) for a short baseline
segment, ramp DOWN linearly to this cell's own floor (anchored the same way
Step 2 anchors its grid -- silencing_threshold_bracket_nA[0] - margin, see
get_cell_floor_nA, reused directly from run_held_injected_grid.py), ramp
back UP to 0, then hold again. One continuous simulate() call with a
time-varying Iapp_func -- never resets/re-warm-starts mid-ramp, since the
entire point is to track one unbroken trajectory through the down/up cycle.

Ramp rate default (0.1 nA/s): this project's own kinetics are shared across
all 69 cells (only max conductances vary, confirmed in Step 0 -- see the
simulation checklist), so the slowest single-channel relaxation time
constant is a fixed, computable quantity, not something that varies per
cell. tauHM = tauX(V, 272, -1499, 42.2, -8.73) ranges from 272 ms
(hyperpolarized) up to 272+1499 = 1771 ms (depolarized) -- see
singlecell_model_v1.py's tauX/tauHM. At 0.1 nA/s, crossing a full 1 nA
takes 10 s, and crossing Step 1's own --fine-step-nA (0.25 nA, the finest
current resolution already established as meaningful in this project) takes
2.5 s -- both comfortably longer than the slowest gating relaxation in the
model, so the ramp stays quasi-adiabatic relative to channel kinetics rather
than effectively behaving like a fast step. This is a judgment call, not a
literature-calibrated value -- --ramp-rate-nA-per-s is exposed precisely so
it can be revisited (e.g. run at 2-3 rates per cell to check whether a
hysteresis effect is rate-INdependent, which is what would make it a
genuine intrinsic hysteresis rather than an artifact of insufficiently slow
sweeping).

Trace persistence: unlike run_held_injected_grid.py's grid (which
deliberately keeps only derived scalars, because any single point can be
losslessly re-simulated on demand from its stored run_args + (held,
injected) key -- see plot_example_traces.py), a ramp's hysteresis IS the
continuous trajectory; there's no discrete "grid" of independently
reproducible points to fall back on, and re-deriving the loop shape after
the fact would mean re-simulating the whole ramp. So this script persists a
downsampled voltage trace (default: every 10th sample of the integration,
i.e. dt_stored = dt * 10 = 1 ms at the default dt=0.1ms -- fine enough to
see the overall envelope/hysteresis-loop shape and individual spikes'
approximate timing, not fine enough to trust for precise re-derived spike
counts) alongside the small set of parameters (hold/ramp durations, floor)
needed to reconstruct Iapp(t) and t(t) exactly, since both are fully
determined by those parameters and never need their own storage (same
"don't store what's reconstructible" principle steady_state_cache.py uses
for burnin_t). All of the actual hysteresis SCALARS (loop width/area, sign,
accommodation index, per-bin firing rates) are computed from the
FULL-resolution trajectory before downsampling and stored directly -- they
do not depend on the stored (lossy) trace and never need re-simulation to
recover.

Figure choice: a V-vs-Iapp phase-plane loop (down-sweep and up-sweep in
different colors), not the two-stacked-panel V(t)/Iapp(t) time trace
plot_example_traces.py uses for step-protocol points. Reasoning: hysteresis
is fundamentally a statement about "the response differs at the SAME Iapp
depending on sweep direction" -- a time-domain panel only shows this
indirectly (the viewer has to mentally compare two different times that
happen to share an Iapp value); the phase-plane loop shows it directly, as
a literal loop that would collapse to a single curve if there were no
path-dependence at all. A small time-domain context panel (matching
plot_example_trace's stacked-thin-line convention) is kept alongside it
purely as a sanity-check/debugging aid, not as the primary hysteresis
evidence.
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
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)
from find_silencing_threshold import (DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_SILENCING_CACHE_PATH,
                                      _round_level, PROMINENCE_FRACTION, FLATLINE_MV)
from run_held_injected_grid import (get_cell_floor_nA,
                                    DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_GRID_CACHE_PATH)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_ramp_protocol.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "ramp_protocol"
DEFAULT_FIGURE_FORMAT = "svg"
# temp == reftemp -- matches find_silencing_threshold.py / run_held_injected_grid.py's
# already-corrected convention (q10 scaling collapses to 1 at dtemp=0).
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

V_INDEX = 0

# See module docstring's "Ramp rate default" section for the tauHM-based
# reasoning behind this value.
DEFAULT_RAMP_RATE_NA_PER_S = 0.1


# ---------------------------------------------------------------------------
# Ramp schedule: hold(0) -> down -> up -> hold(1), a single monotonic
# piecewise-linear Iapp(t). Kept as pure functions of (t1, t2, t3, t4,
# floor_nA) so the schedule can be reconstructed from a handful of stored
# scalars without ever needing to store Iapp(t) or t(t) themselves (see
# module docstring's "Trace persistence" section).
# ---------------------------------------------------------------------------

def build_ramp_schedule_ms(hold0_s: float, down_s: float, up_s: float, hold1_s: float) -> tuple:
    hold0_ms, down_ms, up_ms, hold1_ms = (x * 1000.0 for x in (hold0_s, down_s, up_s, hold1_s))
    t1 = hold0_ms
    t2 = t1 + down_ms
    t3 = t2 + up_ms
    t4 = t3 + hold1_ms
    return t1, t2, t3, t4


def make_ramp_iapp_func(schedule_ms: tuple, floor_nA: float):
    """Scalar t_ms -> Iapp(nA) callable, for simulate()'s Iapp_func (which
    calls this once per integration step -- must stay scalar-in/scalar-out).
    """
    t1, t2, t3, t4 = schedule_ms

    def f(t_ms):
        if t_ms <= t1:
            return 0.0
        if t_ms <= t2:
            return floor_nA * (t_ms - t1) / (t2 - t1)
        if t_ms <= t3:
            return floor_nA * (1.0 - (t_ms - t2) / (t3 - t2))
        return 0.0

    return f


def ramp_iapp_array(t_ms: np.ndarray, schedule_ms: tuple, floor_nA: float) -> np.ndarray:
    """Vectorized counterpart of make_ramp_iapp_func, for post-hoc analysis
    over an already-simulated t_ms array (simulate() itself needs the scalar
    closure above; recomputing Iapp(t) here afterward is far cheaper than a
    million-plus Python-level calls, since numpy does the whole array at once).
    """
    t1, t2, t3, t4 = schedule_ms
    iapp = np.zeros_like(t_ms)
    down_mask = (t_ms > t1) & (t_ms <= t2)
    up_mask = (t_ms > t2) & (t_ms <= t3)
    iapp[down_mask] = floor_nA * (t_ms[down_mask] - t1) / (t2 - t1)
    iapp[up_mask] = floor_nA * (1.0 - (t_ms[up_mask] - t2) / (t3 - t2))
    return iapp


def t_from_iapp_down(iapp_nA: float, schedule_ms: tuple, floor_nA: float) -> float:
    t1, t2, _t3, _t4 = schedule_ms
    return t1 + (iapp_nA / floor_nA) * (t2 - t1)


def t_from_iapp_up(iapp_nA: float, schedule_ms: tuple, floor_nA: float) -> float:
    _t1, t2, t3, _t4 = schedule_ms
    return t2 + (1.0 - iapp_nA / floor_nA) * (t3 - t2)


def build_bin_edges(bin_nA: float, floor_nA: float) -> list:
    """Same fixed-step, round-value convention as run_held_injected_grid.py's
    _build_levels -- reimplemented locally (a few lines) rather than importing
    a module-private helper across scripts. floor_nA is always included as
    the final edge even if not an exact multiple of bin_nA.
    """
    edges = [0.0]
    level = _round_level(-bin_nA)
    while level > floor_nA:
        edges.append(level)
        level = _round_level(level - bin_nA)
    if edges[-1] != _round_level(floor_nA):
        edges.append(_round_level(floor_nA))
    return edges


# ---------------------------------------------------------------------------
# Hysteresis metrics
# ---------------------------------------------------------------------------

def compute_hysteresis_loop(spike_t_ms: np.ndarray, schedule_ms: tuple, floor_nA: float,
                            bin_nA: float) -> dict:
    """For matching Iapp bins spanning [floor_nA, 0], compute the local
    firing rate on the down-sweep vs the up-sweep -- i.e. the actual
    frequency-vs-Iapp "loop" whose two branches would coincide if there were
    no path-dependence. Each bin's time window is derived analytically from
    the (invertible, piecewise-linear) ramp schedule, not by re-detecting
    spikes in a re-simulated sub-window -- spike_t_ms is detected once, over
    the full-resolution trajectory, and reused here.
    """
    edges = build_bin_edges(bin_nA, floor_nA)
    centers, freq_down, freq_up = [], [], []
    for i in range(len(edges) - 1):
        hi_nA, lo_nA = edges[i], edges[i + 1]  # hi closer to 0, lo more hyperpolarizing
        centers.append(0.5 * (hi_nA + lo_nA))

        t_hi_d = t_from_iapp_down(hi_nA, schedule_ms, floor_nA)
        t_lo_d = t_from_iapp_down(lo_nA, schedule_ms, floor_nA)
        lo_t, hi_t = min(t_hi_d, t_lo_d), max(t_hi_d, t_lo_d)
        dur_s = (hi_t - lo_t) / 1000.0
        n = int(np.sum((spike_t_ms >= lo_t) & (spike_t_ms < hi_t)))
        freq_down.append(n / dur_s if dur_s > 0 else 0.0)

        t_hi_u = t_from_iapp_up(hi_nA, schedule_ms, floor_nA)
        t_lo_u = t_from_iapp_up(lo_nA, schedule_ms, floor_nA)
        lo_t, hi_t = min(t_hi_u, t_lo_u), max(t_hi_u, t_lo_u)
        dur_s = (hi_t - lo_t) / 1000.0
        n = int(np.sum((spike_t_ms >= lo_t) & (spike_t_ms < hi_t)))
        freq_up.append(n / dur_s if dur_s > 0 else 0.0)

    return {"centers_nA": np.array(centers), "freq_down_hz": np.array(freq_down),
            "freq_up_hz": np.array(freq_up)}


def summarize_hysteresis(loop: dict, min_diff_hz: float) -> dict:
    centers, freq_down, freq_up = loop["centers_nA"], loop["freq_down_hz"], loop["freq_up_hz"]
    order = np.argsort(centers)
    area_hz_nA = float(np.trapezoid(np.abs(freq_up[order] - freq_down[order]), centers[order]))
    mean_diff_hz = float(np.mean(freq_up - freq_down))
    if mean_diff_hz > min_diff_hz:
        sign = "up_sweep_more_excited"
    elif mean_diff_hz < -min_diff_hz:
        sign = "down_sweep_more_excited"
    else:
        sign = "no_clear_hysteresis"
    return {"hysteresis_area_hz_nA": area_hz_nA, "hysteresis_mean_diff_hz": mean_diff_hz,
            "hysteresis_sign": sign}


def find_ramp_thresholds(spike_t_ms: np.ndarray, schedule_ms: tuple, floor_nA: float) -> dict:
    """The ramp's own operational cessation/resumption levels: the Iapp at
    the LAST spike seen during the whole down-sweep window (below that, in
    time, nothing else fires for the rest of the down-sweep -- a
    well-defined quantity regardless of tonic/bursting structure, unlike
    trying to classify "clean vs. graded" cessation, which
    find_silencing_threshold.py's own docstring found wasn't a real
    qualitative distinction across real cells and deliberately dropped) and
    the Iapp at the FIRST spike seen during the up-sweep (resumption).
    """
    t1, t2, t3, _t4 = schedule_ms
    down_spikes = spike_t_ms[(spike_t_ms > t1) & (spike_t_ms <= t2)]
    up_spikes = spike_t_ms[(spike_t_ms > t2) & (spike_t_ms <= t3)]

    if len(down_spikes) > 0:
        last_down_t = float(down_spikes.max())
        down_threshold_nA = float(floor_nA * (last_down_t - t1) / (t2 - t1))
        down_reason = "ok"
    else:
        down_threshold_nA, down_reason = None, "no_spikes_in_down_sweep"

    if len(up_spikes) > 0:
        first_up_t = float(up_spikes.min())
        up_threshold_nA = float(floor_nA * (1.0 - (first_up_t - t2) / (t3 - t2)))
        up_reason = "ok"
    else:
        # A real, reportable finding, not a failure -- means firing never
        # resumed anywhere on the way back up to 0 within this ramp's own
        # floor/rate, e.g. genuine long-lasting depolarization block or
        # slow-recovering inactivation.
        up_threshold_nA, up_reason = None, "never_resumed_firing"

    return {"ramp_down_threshold_nA": down_threshold_nA, "ramp_down_threshold_reason": down_reason,
            "ramp_up_threshold_nA": up_threshold_nA, "ramp_up_threshold_reason": up_reason}


def get_fast_step_threshold_at_held0(grid_entry) -> tuple:
    """Step 2's own held=0 column, read straight from cell_held_injected_grid.pkl
    (no new fast-step simulation -- the checklist explicitly says this isn't
    needed). Returns (fast_step_threshold_nA or None, reason). The bracket
    midpoint (last firing injected level, first silent injected level) is
    reported, matching find_silencing_threshold.py's own bracket-midpoint
    convention for its silencing threshold.
    """
    if grid_entry is None:
        return None, "no_grid_data"
    if grid_entry.get("status") != "ok":
        return None, f"grid_status_{grid_entry.get('status')}"
    grid = grid_entry.get("grid", {})
    held0_points = [(inj, pt) for (held, inj), pt in grid.items() if abs(held) < 1e-9]
    if not held0_points:
        return None, "no_held0_column"
    held0_points.sort(key=lambda x: -x[0])  # 0.0, -0.5, -1.0, ... (descending injected)

    firing_patterns = {"tonic", "bursting"}
    last_firing_inj = None
    for inj, pt in held0_points:
        pattern = pt.get("test_pattern")
        if pattern in firing_patterns:
            last_firing_inj = inj
        elif pattern == "silent":
            if last_firing_inj is not None:
                return 0.5 * (last_firing_inj + inj), "ok"
            # Silent already at the very first (least hyperpolarizing) level
            # tried at held=0 -- unexpected for a cell confirmed to fire at
            # Iapp=0 in Step 0, but reported rather than assumed away.
            return 0.5 * (0.0 + inj), "ok_silenced_immediately"
    return None, "did_not_silence_in_grid_range"


# ---------------------------------------------------------------------------
# Per-cell orchestrator
# ---------------------------------------------------------------------------

def run_cell_ramp(cell_id: str, params: np.ndarray, ss_entry, silencing_entry, grid_entry,
                  args: argparse.Namespace) -> dict:
    base = {"cell_id": cell_id, "params": params, "run_args": vars(args)}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}
    if silencing_entry is None:
        return {**base, "status": "no_silencing_threshold"}

    floor_result = get_cell_floor_nA(silencing_entry, args.cell_floor_margin_nA)
    if floor_result is None:
        return {**base, "status": "silencing_not_ok"}
    floor_nA, floor_anchor_source = floor_result
    if floor_nA >= 0:
        # Should be structurally impossible (see get_fast_step_threshold_at_held0's
        # sibling docstring reasoning: first_silent_level_nA is always
        # <= -coarse_step_nA in find_silencing_threshold.py), but the ramp
        # schedule's down/up inversion divides by floor_nA -- guard rather
        # than risk a silent div-by-zero/sign-flip downstream.
        return {**base, "status": "invalid_floor", "ramp_floor_nA": floor_nA}

    y_ss = ss_entry["y_ss"]
    baseline_freq_hz = ss_entry["freq_hz"]

    down_s = abs(floor_nA) / args.ramp_rate_nA_per_s
    up_s = down_s  # symmetric by construction -- same total nA excursion, same rate
    if down_s > args.max_ramp_half_duration_s:
        return {**base, "status": "ramp_duration_exceeds_cap", "ramp_floor_nA": floor_nA,
                "down_s": down_s, "up_s": up_s}

    hold_s = max(args.min_baseline_hold_s, args.baseline_hold_periods / baseline_freq_hz) \
        if baseline_freq_hz else args.min_baseline_hold_s
    schedule_ms = build_ramp_schedule_ms(hold_s, down_s, up_s, hold_s)
    Iapp_func = make_ramp_iapp_func(schedule_ms, floor_nA)
    nsecs_total = schedule_ms[3] / 1000.0

    try:
        t_ms, states = simulate(params, nsecs_total, args.temp, dt=args.dt, reftemp=args.reftemp,
                                cis=y_ss, Iapp_func=Iapp_func)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {**base, "status": "blew_up", "error": str(exc), "ramp_floor_nA": floor_nA}
    if not np.all(np.isfinite(states)):
        return {**base, "status": "blew_up", "error": "non-finite trajectory", "ramp_floor_nA": floor_nA}

    V = states[:, V_INDEX]
    t1, t2, t3, t4 = schedule_ms

    result = {**base, "ramp_floor_nA": floor_nA, "floor_anchor_source": floor_anchor_source,
              "baseline_freq_hz": baseline_freq_hz, "hold_s": hold_s, "down_s": down_s, "up_s": up_s,
              "schedule_ms": schedule_ms, "dt_ms": args.dt}

    v_range = float(V.max() - V.min())
    if v_range < FLATLINE_MV:
        result["status"] = "no_spikes_detected"
    else:
        peaks, _ = find_peaks(V, prominence=v_range * PROMINENCE_FRACTION)
        spike_t_ms = t_ms[peaks]

        hold0_spikes = spike_t_ms[spike_t_ms <= t1]
        hold1_spikes = spike_t_ms[spike_t_ms > t3]
        result["hold0_freq_hz"] = len(hold0_spikes) / (t1 / 1000.0) if t1 > 0 else None
        result["hold1_freq_hz"] = len(hold1_spikes) / ((t4 - t3) / 1000.0) if (t4 - t3) > 0 else None
        result["n_spikes_total"] = int(len(spike_t_ms))

        thresholds = find_ramp_thresholds(spike_t_ms, schedule_ms, floor_nA)
        result.update(thresholds)
        if thresholds["ramp_down_threshold_nA"] is not None and thresholds["ramp_up_threshold_nA"] is not None:
            result["hysteresis_width_nA"] = (thresholds["ramp_down_threshold_nA"]
                                             - thresholds["ramp_up_threshold_nA"])
        else:
            result["hysteresis_width_nA"] = None

        loop = compute_hysteresis_loop(spike_t_ms, schedule_ms, floor_nA, args.hysteresis_bin_nA)
        result["hysteresis_bin_centers_nA"] = loop["centers_nA"]
        result["hysteresis_freq_down_hz"] = loop["freq_down_hz"]
        result["hysteresis_freq_up_hz"] = loop["freq_up_hz"]
        result.update(summarize_hysteresis(loop, args.min_hysteresis_diff_hz))

        fast_step_threshold_nA, fast_step_reason = get_fast_step_threshold_at_held0(grid_entry)
        result["fast_step_threshold_nA"] = fast_step_threshold_nA
        result["fast_step_threshold_reason"] = fast_step_reason
        if fast_step_threshold_nA is not None and thresholds["ramp_down_threshold_nA"] is not None:
            # Positive: this ramp's slow-approach down-sweep silences at a
            # LESS hyperpolarizing (weaker) current than an abrupt step does
            # at held=0 -- i.e. the slow approach makes the cell reach
            # cessation "easier"/sooner. Negative: the ramp resists
            # silencing longer than a step would (e.g. because the very
            # slow approach lets some adapting/deinactivating current keep
            # propping firing up further than an abrupt step ever samples).
            # Sign convention documented here since it's a judgment call,
            # not a standard defined elsewhere in this codebase.
            result["accommodation_index_nA"] = (thresholds["ramp_down_threshold_nA"]
                                                - fast_step_threshold_nA)
        else:
            result["accommodation_index_nA"] = None

        result["status"] = "ok"

    stride = max(1, args.trace_downsample_factor)
    result["ramp_v_stored_mV"] = V[::stride].astype(np.float32)
    result["trace_dt_stored_ms"] = args.dt * stride

    return result


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_pickle_with_retry(path: Path, attempts: int = 5) -> dict:
    """Read-only load with retry. cell_held_injected_grid.pkl in particular
    may be actively rewritten by a concurrently running background sweep
    (write-then-rename via os.replace, see run_held_injected_grid.py's
    save_output_cache) while this script reads it -- os.replace is atomic,
    but the exact same transient-PermissionError-on-Windows issue that
    motivated the write-side retry there can in principle show up reading
    mid-rename too, so the same defensive retry is applied on the read side.
    """
    if not Path(path).exists():
        return {}
    for attempt in range(attempts):
        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def load_output_cache(cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> dict:
    return _load_pickle_with_retry(cache_path)


def save_output_cache(cache: dict, cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> None:
    # Write-then-rename, retried on Windows' transient PermissionError --
    # identical pattern to find_silencing_threshold.py / run_held_injected_grid.py's
    # own save_output_cache (each script keeps its own copy rather than
    # importing one another's, matching this codebase's existing convention).
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


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_cell_ramp(result: dict, outdir: Path, command: str, fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    if result["status"] not in ("ok", "no_spikes_detected"):
        return
    v = result.get("ramp_v_stored_mV")
    if v is None or len(v) == 0:
        return
    cell_id = result["cell_id"]
    dt_stored_ms = result["trace_dt_stored_ms"]
    t = np.arange(len(v), dtype=np.float64) * dt_stored_ms
    schedule_ms = result["schedule_ms"]
    t1, t2, t3, t4 = schedule_ms
    floor_nA = result["ramp_floor_nA"]
    iapp = ramp_iapp_array(t, schedule_ms, floor_nA)

    down_mask = (t > t1) & (t <= t2)
    up_mask = (t > t2) & (t <= t3)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4], height_ratios=[3, 1])
    ax_loop = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[1, 1], sharex=ax_v)

    ax_loop.plot(iapp[down_mask], v[down_mask], color="firebrick", lw=0.6, label="down-sweep")
    ax_loop.plot(iapp[up_mask], v[up_mask], color="steelblue", lw=0.6, label="up-sweep")
    if result.get("ramp_down_threshold_nA") is not None:
        ax_loop.axvline(result["ramp_down_threshold_nA"], color="firebrick", ls="--", lw=1, alpha=0.7,
                        label=f"down threshold ({result['ramp_down_threshold_nA']:.2f} nA)")
    if result.get("ramp_up_threshold_nA") is not None:
        ax_loop.axvline(result["ramp_up_threshold_nA"], color="steelblue", ls="--", lw=1, alpha=0.7,
                        label=f"up threshold ({result['ramp_up_threshold_nA']:.2f} nA)")
    ax_loop.set_xlabel("Iapp (nA)")
    ax_loop.set_ylabel("V (mV)")
    ax_loop.invert_xaxis()
    ax_loop.legend(loc="best", fontsize=6)
    if result["status"] == "ok":
        title = (f"{cell_id} — hysteresis area={result['hysteresis_area_hz_nA']:.2f} Hz·nA, "
                 f"sign={result['hysteresis_sign']}")
    else:
        title = f"{cell_id} — no spikes detected across the whole ramp"
    ax_loop.set_title(title, fontsize=8)

    ax_v.plot(t, v, color="black", lw=0.5)
    ax_v.set_ylabel("V (mV)")
    ax_v.axvline(t1, color="gray", ls=":", lw=0.8)
    ax_v.axvline(t2, color="gray", ls=":", lw=0.8)
    ax_v.axvline(t3, color="gray", ls=":", lw=0.8)
    ax_i.plot(t, iapp, color="dimgray", lw=1.0)
    ax_i.set_ylabel("Iapp (nA)")
    ax_i.set_xlabel("time (ms)")

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(0.5, 0.01, command, ha="center", va="bottom", fontsize=6, family="monospace",
             color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_ramp.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slow triangular hyperpolarizing ramp (down-sweep then up-sweep) per cell, "
                    "to measure hysteresis -- structurally distinct from the fixed-level held x "
                    "injected grid (run_held_injected_grid.py), which cannot reveal path-dependence. "
                    "Depolarizing current is out of scope for this script.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) to ramp. Default: every cell found in --params-dir.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH,
                        help="Path to the Iapp=0 steady-state cache (input, read-only).")
    parser.add_argument("--silencing-cache", default=DEFAULT_SILENCING_CACHE_PATH,
                        help="Path to Step 1's cell_silencing_thresholds.pkl (input, read-only; "
                             "used to anchor this cell's ramp floor).")
    parser.add_argument("--grid-cache", default=DEFAULT_GRID_CACHE_PATH,
                        help="Path to Step 2's cell_held_injected_grid.pkl (input, READ-ONLY -- "
                             "may be actively written by a concurrently running background sweep; "
                             "used only to read the held=0 fast-step threshold for the accommodation "
                             "index). Missing/not-yet-swept cells simply get accommodation_index_nA=None.")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["svg", "png", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--jobs", type=int, default=-1)

    parser.add_argument("--ramp-rate-nA-per-s", type=float, default=DEFAULT_RAMP_RATE_NA_PER_S,
                        help="Down-sweep and up-sweep speed, in nA/s (symmetric). See module "
                             "docstring for the tauHM-based reasoning behind the 0.1 nA/s default.")
    parser.add_argument("--cell-floor-margin-nA", type=float, default=1.0,
                        help="Ramp floor = silencing_threshold_bracket_nA[0] - margin, the exact "
                             "same anchoring convention run_held_injected_grid.py uses for its "
                             "held axis (--cell-floor-margin-nA there).")
    parser.add_argument("--baseline-hold-periods", type=float, default=5.0,
                        help="Baseline hold (before ramp-down and after ramp-up) = max("
                             "--min-baseline-hold-s, this many baseline oscillation periods).")
    parser.add_argument("--min-baseline-hold-s", type=float, default=2.0)
    parser.add_argument("--max-ramp-half-duration-s", type=float, default=600.0,
                        help="Safety cap on down_s (== up_s): if this cell's floor/rate combination "
                             "would need longer than this just for the down-sweep, skip simulating it "
                             "(status='ramp_duration_exceeds_cap') rather than run an unbounded "
                             "integration. Matches this project's existing guardrail convention "
                             "(--max-hyperpolarizing-nA, --max-boundary-edges-per-depth).")
    parser.add_argument("--hysteresis-bin-nA", type=float, default=0.25,
                        help="Iapp bin width for the matched down-vs-up firing-rate comparison. "
                             "0.25 nA matches find_silencing_threshold.py's own --fine-step-nA default.")
    parser.add_argument("--min-hysteresis-diff-hz", type=float, default=0.5,
                        help="Minimum |mean(freq_up - freq_down)| (Hz) required to call a sign "
                             "('up_sweep_more_excited' / 'down_sweep_more_excited') rather than "
                             "'no_clear_hysteresis'.")
    parser.add_argument("--trace-downsample-factor", type=int, default=10,
                        help="Store every Nth simulated sample (default 10: dt=0.1ms -> stored "
                             "dt=1ms). See module docstring's 'Trace persistence' section -- the "
                             "hysteresis scalars are computed from the full-resolution trajectory "
                             "before this downsampling, so this only affects the figure/visual "
                             "re-inspection, never the persisted metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    ss_cache_path = Path(args.steady_state_cache)
    silencing_cache_path = Path(args.silencing_cache)
    grid_cache_path = Path(args.grid_cache)
    output_cache_path = Path(args.output_cache)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_cells = load_all_cells(params_dir)
    if args.cells:
        unknown = sorted(set(args.cells) - set(all_cells))
        if unknown:
            raise SystemExit(f"Unknown cell id(s): {unknown}")
        cells = {cid: all_cells[cid] for cid in args.cells}
    else:
        cells = all_cells
    print(f"Running ramp protocol for {len(cells)} of {len(all_cells)} cell(s) from {params_dir}")

    silencing_cache = _load_pickle_with_retry(silencing_cache_path)
    grid_cache = _load_pickle_with_retry(grid_cache_path)  # read-only -- see --grid-cache help
    output_cache = load_output_cache(output_cache_path)

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        silencing_entry = silencing_cache.get(cell_id)
        grid_entry = grid_cache.get(cell_id)
        result = run_cell_ramp(cell_id, params, ss_entry, silencing_entry, grid_entry, args)
        return cell_id, result

    # Save+plot after EVERY cell, streamed via generator_unordered -- same
    # rationale as find_silencing_threshold.py / run_held_injected_grid.py:
    # a long ramp sweep should be visible on disk as it progresses, not only
    # once the whole batch finishes.
    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items())

    status_counts: dict = {}
    n_done = 0
    for cell_id, result in result_iter:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if not args.no_plot:
            plot_cell_ramp(result, figures_dir, command, fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cells)}] {cell_id}: {result['status']}")

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    flagged = [cid for cid in cells if output_cache[cid]["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'no_silencing_threshold'/'silencing_not_ok': run find_silencing_threshold.py first. "
              "'ramp_duration_exceeds_cap': raise --max-ramp-half-duration-s or --ramp-rate-nA-per-s "
              "for that cell. 'no_spikes_detected': the ramp floor may not be deep enough, or this "
              "cell's own silencing bracket is unusually shallow. 'blew_up': consider a smaller --dt.")


if __name__ == "__main__":
    main()
