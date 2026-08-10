"""Step 6a: predict each cell's phase-locking behavior to periodic
inhibitory forcing at ITS OWN natural period, from its already-measured PRC
(cell_prc.pkl / Step 5) -- no new simulation, cache-only, cheap.

This is the "prediction" half of the user's checklist's Step 6
("Entrainment") methodology: predict first via a weakly-coupled-oscillator
phase map, then validate the prediction against real brute-force
simulation on a small subset (run_entrainment_protocol.py) before deciding
whether a full-population brute-force run is warranted. See
`project_simulation_checklist_snapshot` / `project_clustering_feature_
search_result` memory for why this predict-first path was chosen over
going straight to brute force.

Forcing period == the cell's own natural period (user's explicit choice,
not the checklist's literal "pyloric cycle frequency" wording -- confirmed
twice, 2026-08-08). This simplifies the standard iterated phase-resetting
(stroboscopic) map for periodic forcing: absent any resetting, one forcing
interval always returns to the same phase, so the map is just

    phi_{n+1} = (phi_n + shift_norm(phi_n)) mod 1

where shift_norm(phi) is the cell's own measured normalized phase-shift
curve (Δt/T, sign convention: negative=advance, positive=delay -- see
run_prc_protocol.py's module docstring, reused verbatim, not
reinterpreted). A fixed point of this map (phi* with phi* == f(phi*) mod 1)
is exactly a zero-crossing of shift_norm(phi) -- a phase at which the
periodic pulse train's resetting effect exactly cancels the one-period
return, which is the definition of a stable rhythm locked to the forcing.
Stability (attracting vs. repelling) is the map's local slope,
1 + d(shift_norm)/d(phi) at phi* -- |slope| < 1 means small perturbations
away from phi* shrink back toward it each cycle (attracting/stable);
|slope| > 1 means they grow (repelling/unstable).

Only 1:1 locking (one response event per one forcing cycle) is predicted
here, matching the "burst once per cycle" framing this whole investigation
is motivated by. Higher-order lockings (1:2, 2:1, ...) are out of scope.

Important limitation, reported alongside every prediction, not just in
this docstring: the PRC itself was measured with a brief (5% of period),
fixed-amplitude hyperpolarizing pulse (run_prc_protocol.py's own
convention) -- not necessarily representative of every possible periodic
forcing waveform. This prediction is only as good as how well that brief-
pulse PRC generalizes, which is exactly why run_entrainment_protocol.py
validates it against real simulation (using the SAME pulse convention, for
a clean apples-to-apples test) rather than trusting it standalone.
"""

import pickle
import sys
from pathlib import Path
import argparse

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_PRC_CACHE_PATH = ROOT_DIR / "cell_prc.pkl"
DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_entrainment_prediction.pkl"

# A cell needs at least this fraction of its 16 sampled phases to have a
# valid (non-NaN) measurement before a prediction is attempted -- too many
# missing phases means the circular interpolant would be guessing across
# large gaps rather than genuinely tracing the PRC shape.
MIN_VALID_PHASE_FRACTION = 0.75

# Step size (in phi) for the numerical derivative used in the stability
# (slope) calculation -- small relative to the ~1/16 native sample spacing
# so it probes local curvature of the interpolant, not the coarse sampling.
DERIVATIVE_STEP = 1e-4

# Number of points used to scan for zero-crossings of shift_norm(phi)
# before refining each crossing by linear interpolation between the
# bracketing scan points -- fine enough to not miss a real crossing given
# only 16 underlying data points (16x oversampling would be the bare
# minimum; this is far finer to be safe).
PHI_SCAN_POINTS = 4000


def circular_interp(phi_query: np.ndarray, phases: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Periodic linear interpolation of `values` sampled at `phases`
    (sorted, spanning [0, 1) with equal spacing, per run_prc_protocol.py's
    np.linspace(0, 1, n_phases, endpoint=False) convention). Wraps one
    point before 0 and one after 1 so plain np.interp (which has no native
    concept of periodicity) handles the phi=0/phi=1 boundary correctly.
    """
    ext_phases = np.concatenate([[phases[-1] - 1.0], phases, [phases[0] + 1.0]])
    ext_values = np.concatenate([[values[-1]], values, [values[0]]])
    return np.interp(np.mod(phi_query, 1.0), ext_phases, ext_values)


def fill_nan_circular(phases: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fills NaN entries in `values` via circular linear interpolation
    from the nearest valid neighbors on either side (wrapping through
    phi=1/phi=0), so a handful of missing phases (a trial where no next
    event was found) don't block building the interpolant entirely.
    """
    valid = ~np.isnan(values)
    if valid.all():
        return values.copy()
    filled = values.copy()
    missing_idx = np.flatnonzero(~valid)
    for i in missing_idx:
        filled[i] = circular_interp(np.array([phases[i]]), phases[valid], values[valid])[0]
    return filled


def find_fixed_points(phases: np.ndarray, shift_norm: np.ndarray) -> list:
    """Scans shift_norm(phi) on a fine grid for zero-crossings (circular,
    including the phi=1 -> phi=0 wraparound), refines each crossing by
    linear interpolation between the bracketing scan points, and reports
    stability via the map's local slope at each fixed point.
    """
    phi_grid = np.linspace(0.0, 1.0, PHI_SCAN_POINTS, endpoint=False)
    vals = circular_interp(phi_grid, phases, shift_norm)

    fixed_points = []
    for i in range(PHI_SCAN_POINTS):
        j = (i + 1) % PHI_SCAN_POINTS
        v0, v1 = vals[i], vals[j]
        if v0 == 0.0:
            phi_star = phi_grid[i]
        elif v0 * v1 < 0.0:
            phi0 = phi_grid[i]
            phi1 = phi_grid[j] if j != 0 else 1.0
            phi_star = phi0 + (0.0 - v0) * (phi1 - phi0) / (v1 - v0)
            phi_star = float(np.mod(phi_star, 1.0))
        else:
            continue

        deriv = (circular_interp(np.array([phi_star + DERIVATIVE_STEP]), phases, shift_norm)[0]
                -circular_interp(np.array([phi_star - DERIVATIVE_STEP]), phases, shift_norm)[0]
                ) / (2 * DERIVATIVE_STEP)
        slope = 1.0 + deriv
        fixed_points.append({"phi_star": float(phi_star), "slope": float(slope),
                             "stable": bool(abs(slope) < 1.0)})

    # De-duplicate near-identical crossings the fine scan might double-count
    # right at a grid boundary (kept simple: round to the scan's own
    # resolution and drop repeats).
    seen = set()
    deduped = []
    for fp in fixed_points:
        key = round(fp["phi_star"], 4)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fp)
    return deduped


def predict_cell_entrainment(prc_result: dict, min_valid_phase_fraction: float = MIN_VALID_PHASE_FRACTION) -> dict:
    cell_id = prc_result.get("cell_id")
    if prc_result.get("status") != "ok":
        return {"cell_id": cell_id, "status": "prc_not_ok", "prc_status": prc_result.get("status")}

    phases = np.asarray(prc_result["phases"])
    shift_norm = np.asarray(prc_result["phase_shift_norm"])
    event_found = np.asarray(prc_result["event_found"])
    valid = event_found & ~np.isnan(shift_norm)

    if valid.mean() < min_valid_phase_fraction:
        return {"cell_id": cell_id, "status": "insufficient_prc_data",
               "n_valid_phases": int(valid.sum()), "n_phases": len(phases)}

    filled = fill_nan_circular(phases, np.where(valid, shift_norm, np.nan))
    fixed_points = find_fixed_points(phases, filled)
    stable = [fp for fp in fixed_points if fp["stable"]]

    if not fixed_points:
        prediction = "no_fixed_point"
        predicted_locked_phase = None
    elif not stable:
        prediction = "unstable_only"
        predicted_locked_phase = None
    else:
        prediction = "stable_lock"
        # If multiple stable fixed points exist, report the one with the
        # smallest |slope| (most strongly attracting) as the primary
        # prediction, but keep all of them -- which basin a real cell
        # lands in depends on initial phase, not captured by this map alone.
        predicted_locked_phase = min(stable, key=lambda fp: abs(fp["slope"]))["phi_star"]

    return {
        "cell_id": cell_id, "status": "ok",
        "baseline_pattern": prc_result["baseline_pattern"],
        "prc_type": prc_result["prc_type"],
        "natural_period_ms": prc_result["natural_period_ms"],
        "n_valid_phases": int(valid.sum()), "n_phases": len(phases),
        "fixed_points": fixed_points, "n_stable_fixed_points": len(stable),
        "prediction": prediction, "predicted_locked_phase": predicted_locked_phase,
    }


def load_prc_cache(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_output_cache(cache: dict, cache_path: Path) -> None:
    # Write-then-rename + retry, matching every other protocol script in
    # this project (see run_prc_protocol.py's save_output_cache for the
    # original rationale -- transient Windows PermissionError from a
    # concurrent reader).
    import os
    import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prc-cache", default=DEFAULT_PRC_CACHE_PATH)
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s) to predict for. Default: every cell in --prc-cache.")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--min-valid-phase-fraction", type=float, default=MIN_VALID_PHASE_FRACTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prc_cache = load_prc_cache(Path(args.prc_cache))
    cell_ids = args.cells if args.cells else sorted(prc_cache.keys())

    print(f"Predicting entrainment for {len(cell_ids)} of {len(prc_cache)} cell(s) from {args.prc_cache}")

    output_cache = {}
    status_counts: dict = {}
    prediction_counts: dict = {}
    for cell_id in cell_ids:
        prc_result = prc_cache.get(cell_id)
        if prc_result is None:
            result = {"cell_id": cell_id, "status": "no_prc_entry"}
        else:
            result = predict_cell_entrainment(prc_result, args.min_valid_phase_fraction)
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if result["status"] == "ok":
            prediction_counts[result["prediction"]] = prediction_counts.get(result["prediction"], 0) + 1
        tag = result.get("prediction", "") if result["status"] == "ok" else result["status"]
        print(f"  {cell_id}: {tag}")

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    if prediction_counts:
        print("\nPrediction summary (of cells with status 'ok'):")
        for pred, count in sorted(prediction_counts.items()):
            print(f"  {pred}: {count}")

    save_output_cache(output_cache, Path(args.output_cache))
    print(f"\nCache written to {args.output_cache}")


if __name__ == "__main__":
    main()
