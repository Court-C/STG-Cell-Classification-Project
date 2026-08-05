"""
Simulate every cell model in src/models for a few seconds, cache each
cell's equilibrium state, and save a diagnostic figure per cell into
figures/cache/ as proof of convergence.

Built against the actual model.py (RK4 fixed-step `simulate`, 14-value
state vector, `temp`/`reftemp` required args) -- not solve_ivp.

WHAT "EQUILIBRIUM" MEANS HERE: these are self-oscillating (pacemaker)
cells, not point-attractor cells, so there's no single fixed voltage to
settle to. "Equilibrium" = the stable limit cycle each cell's own
parameters produce. The script (1) burns in for T_BURNIN_SECS to let the
initial transient die out, (2) estimates that cell's firing period from
the tail of the burn-in trace, then (3) refines onto the exact periodic
fixed point with a shooting method (fsolve), so the cached state sits
precisely on the limit cycle rather than just "close to it" after a fixed
burn-in window.

ASSUMPTIONS -- confirm/edit these before trusting the output:

1. Parameter files: src/models/*.txt, one file per cell, 40 numeric values
   (whitespace/newline separated), in the order documented in model.py.

2. TEMP / REFTEMP: the model's derivatives() need a simulation temperature
   and a q10 reference temperature. Nothing in our prior conversations
   pinned these down, so:
       TEMP    = 25.0   (matches model.py's own CLI default)
       REFTEMP = 10.0   (model.py's documented single-cell default)
   If your actual experiments run at a different temp, change TEMP below
   -- steady states are temperature-dependent, so this matters.

3. Resting/steady state = free-run at Iapp = 0 nA (established in an
   earlier session: cells fire autonomously in isolation at 0 nA). This
   overrides whatever Iapp value (p[39]) is baked into each cell's file.

4. T_BURNIN_SECS = 5s ("a few seconds", per request) and DT_MS = 0.05ms.
   Given IntCa's own slow time constant (tauIntCa, p[38]), some cells may
   need longer to settle -- check the "silent"/"non_periodic" counts
   printed at the end and raise T_BURNIN_SECS if too many cells fail.

5. fsolve scaling (`FSOLVE_DIAG` below): the state vector mixes wildly
   different scales (V ~ -60..40, gating vars 0..1, IntCa ~0..5).
   FSOLVE_DIAG is a rough order-of-magnitude guess to help convergence.
   I couldn't tune this against your full 69-cell set -- if many cells
   report status "shooting_failed", this is the first thing to adjust.
"""

import re
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from joblib import Parallel, delayed

# ---- fix here if your model module lives elsewhere ----
import singlecell_model_v1
# ---------------------------------------------------------

PARAMS_DIR = Path("src/models")
CACHE_PATH = Path("cell_steady_states.pkl")
FIGURES_DIR = Path("figures/cache")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

V_INDEX = 0
N_STATES = 14

TEMP = 25.0        # deg C -- see assumption 2 above
REFTEMP = 10.0     # deg C -- see assumption 2 above
DT_MS = 0.05       # integration step, ms

T_BURNIN_SECS = 5.0   # "a few seconds" burn-in before period estimation
MIN_PEAKS_FOR_PERIOD = 4
PROMINENCE_FRACTION = 0.3

FSOLVE_DIAG = np.array([50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5], dtype=float)
# Residual tolerance for accepting a shot fixed point, in FSOLVE_DIAG-scaled
# units (see shoot_to_limit_cycle). Checked directly rather than trusting
# fsolve's own `ier` flag, which reports "not converged" (ier=5, stalled
# progress) even when the residual is already well within a useful range --
# common for this stiff a 14-D system. Tightening this means "closer to the
# true limit cycle" at the cost of possibly never satisfying it for some
# cells; loosen if too many cells report shooting_failed despite a small
# residual norm (check the printed diagnostics if that happens).
RESID_TOL = 1e-2


# ---------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------

def load_cell_params(filepath):
    text = filepath.read_text()
    values = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)]
    arr = np.array(values, dtype=np.float64)
    if arr.size != 40:
        raise ValueError(f"{filepath}: expected 40 params, got {arr.size}")
    return arr


def load_all_cells(params_dir=PARAMS_DIR):
    """Returns dict: cell_id -> full 40-value parameter array."""
    cells = {}
    for f in sorted(params_dir.glob("*.txt")):
        cell_id = f.stem
        cells[cell_id] = load_cell_params(f)
    return cells


# ---------------------------------------------------------------------
# Steady-state computation (burn-in -> period estimate -> shooting)
# ---------------------------------------------------------------------

def zero_iapp(t):
    return 0.0


def estimate_period(t_ms, v):
    peaks, _ = find_peaks(v, prominence=(v.max() - v.min()) * PROMINENCE_FRACTION)
    if len(peaks) < MIN_PEAKS_FOR_PERIOD:
        return None
    peak_times = t_ms[peaks[-4:]]
    return float(np.mean(np.diff(peak_times)))  # ms


def integrate_ms(params, y0, ms_duration, temp=TEMP, reftemp=REFTEMP, dt=DT_MS):
    """Integrate forward from y0 for ms_duration milliseconds at Iapp=0.
    Returns the full (t_ms, states) trajectory."""
    nsecs = ms_duration / 1000.0
    return simulate(params, nsecs, temp, dt=dt, reftemp=reftemp,
                     cis=y0, Iapp_func=zero_iapp)


def shoot_to_limit_cycle(params, y0, T_period_ms, temp=TEMP, reftemp=REFTEMP,
                          dt=DT_MS, xtol=1e-6, maxfev=5000):
    def F(y):
        _, states = integrate_ms(params, y, T_period_ms, temp, reftemp, dt)
        return states[-1] - y

    y_ss, info, ier, msg = fsolve(F, y0, xtol=xtol, diag=FSOLVE_DIAG,
                                   full_output=True, maxfev=maxfev)

    # Judge convergence by the actual scaled residual, not fsolve's ier flag
    # (ier often reports "stalled" on this stiff a system even when the
    # residual is already small -- see RESID_TOL comment above).
    final_residual = F(y_ss)
    scaled_resid_norm = np.linalg.norm(final_residual / FSOLVE_DIAG)
    converged = scaled_resid_norm < RESID_TOL
    return y_ss, converged, scaled_resid_norm


def compute_cell_steady_state(cell_id, params, temp=TEMP, reftemp=REFTEMP, dt=DT_MS):
    """
    Returns a dict with y_ss, period_ms, status, and the burn-in trace
    (for plotting). y_ss/period_ms are None if the cell didn't converge
    to a clean limit cycle.
    """
    try:
        y0 = DEFAULT_CIS.copy()
        t_ms, states = simulate(params, T_BURNIN_SECS, temp, dt=dt,
                                 reftemp=reftemp, cis=y0, Iapp_func=zero_iapp)
    except (FloatingPointError, OverflowError, ValueError) as e:
        return dict(cell_id=cell_id, y_ss=None, period_ms=None,
                     status=f"integration_error: {e}", burnin_t=None,
                     burnin_v=None, params=params)

    v = states[:, V_INDEX]
    if not np.all(np.isfinite(v)):
        return dict(cell_id=cell_id, y_ss=None, period_ms=None,
                     status="blew_up", burnin_t=t_ms, burnin_v=v, params=params)

    v_range = v.max() - v.min()
    if v_range < 1.0:
        return dict(cell_id=cell_id, y_ss=None, period_ms=None,
                     status="silent", burnin_t=t_ms, burnin_v=v, params=params)

    T_period_ms = estimate_period(t_ms, v)
    if T_period_ms is None:
        return dict(cell_id=cell_id, y_ss=None, period_ms=None,
                     status="non_periodic", burnin_t=t_ms, burnin_v=v, params=params)

    y_after_burnin = states[-1]
    y_ss, converged, resid_norm = shoot_to_limit_cycle(params, y_after_burnin,
                                                         T_period_ms, temp, reftemp, dt)

    if not converged:
        return dict(cell_id=cell_id, y_ss=None, period_ms=T_period_ms,
                     status="shooting_failed", burnin_t=t_ms, burnin_v=v,
                     params=params, shoot_resid_norm=resid_norm)

    # sanity check: integrate 2 more periods past y_ss, confirm no drift.
    # Scaled the same way as the shooting residual (raw norm is dominated by
    # V and IntCa's larger magnitude and was flagging well-converged cells).
    _, states_check = integrate_ms(params, y_ss, 2 * T_period_ms, temp, reftemp, dt)
    drift = np.linalg.norm((states_check[-1] - states_check[0]) / FSOLVE_DIAG)
    status = "ok" if drift < RESID_TOL else "drift_warning"

    return dict(cell_id=cell_id, y_ss=y_ss, period_ms=T_period_ms,
                 status=status, burnin_t=t_ms, burnin_v=v, params=params,
                 shoot_resid_norm=resid_norm)


# ---------------------------------------------------------------------
# Cache I/O (cell_id-keyed, with params stored for staleness check)
# ---------------------------------------------------------------------

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


# ---------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------

def plot_cell_trace(result, outdir=FIGURES_DIR):
    """Two-panel diagnostic: full burn-in trace (with steady-state V marked),
    and a zoomed final-cycle view for visual confirmation of convergence."""
    cell_id = result["cell_id"]
    t_ms, v = result["burnin_t"], result["burnin_v"]
    if t_ms is None:
        return  # integration errored out before producing a trace

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].plot(t_ms / 1000.0, v, lw=0.6, color="steelblue")
    title = f"{cell_id} — status: {result['status']}"
    if result["period_ms"] is not None:
        title += f", period={result['period_ms']:.1f} ms"
    axes[0].set_title(title, fontsize=9)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("V (mV)")
    if result["y_ss"] is not None:
        axes[0].axhline(result["y_ss"][V_INDEX], color="firebrick", ls="--",
                         lw=1, label="steady-state V")
        axes[0].legend(loc="upper right", fontsize=7)

    # zoomed final ~2 periods of the burn-in trace
    if result["period_ms"] is not None:
        window_ms = 2.5 * result["period_ms"]
        mask = t_ms >= (t_ms[-1] - window_ms)
        axes[1].plot(t_ms[mask], v[mask], lw=0.8, color="steelblue",
                     label="end of burn-in")
        if result["y_ss"] is not None:
            axes[1].axhline(result["y_ss"][V_INDEX], color="firebrick",
                             ls="--", lw=1, label="shot steady state")
        axes[1].set_xlabel("time (ms)")
        axes[1].set_ylabel("V (mV)")
        axes[1].set_title("final cycles (zoom)", fontsize=9)
        axes[1].legend(loc="upper right", fontsize=7)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(outdir / f"{cell_id}_burnin_trace.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    cells = load_all_cells()
    print(f"Loaded {len(cells)} cell parameter files from {PARAMS_DIR}")

    cache = load_cache()

    def process_one(cell_id, params):
        entry = cache.get(cell_id)
        if entry is not None and np.array_equal(entry["params"], params):
            return cell_id, entry, False  # already cached, unchanged
        result = compute_cell_steady_state(cell_id, params)
        return cell_id, result, True

    results = Parallel(n_jobs=-1)(
        delayed(process_one)(cid, params) for cid, params in cells.items()
    )

    status_counts = {}
    for cell_id, result, is_new in results:
        if is_new:
            cache[cell_id] = result
        plot_cell_trace(result)
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    save_cache(cache)

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\nCache written to {CACHE_PATH}")
    print(f"Trace figures written to {FIGURES_DIR}/")

    flagged = [cid for cid, r, _ in results if r["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) need manual review (not 'ok'): {flagged}")


if __name__ == "__main__":
    main()