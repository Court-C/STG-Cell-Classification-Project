"""Step 2c intrinsic-property extraction that genuinely needs re-simulation:
input resistance (R_in) / membrane time constant (tau) from a small
subthreshold hyperpolarizing step, spike waveform features from the cell's
own first baseline spike, and ISI statistics (mean/SD/CV/local variation)
from the baseline spike train.

run_held_injected_grid.py's own module docstring flags these as a separate
future script, precisely because that grid only persists derived scalars per
point -- never raw voltage traces -- so none of the three feature groups
below can be recovered from its cache after the fact. Everything here is a
fresh, dedicated resimulation, warm-started from the cell's cached Iapp=0
limit-cycle state (cell_steady_states.pkl), matching the settle/warm-start
conventions of find_silencing_threshold.py and run_held_injected_grid.py.

Inputs read (never modified): cell_steady_states.pkl (steady_state_cache.py)
and cell_silencing_thresholds.pkl (find_silencing_threshold.py, used only to
anchor the subthreshold step amplitude to each cell's own current scale).

Feature 1 -- R_in / tau: a hyperpolarizing current step is applied on top of
the cell's own already-settled Step-1 silencing threshold (first_silent_
level_nA), not at held=0. Because every cell here fires spontaneously at
Iapp=0 (see find_silencing_threshold.py's module docstring), held=0 has no
genuine passive resting state to step from at all -- even the AHP trough
right after a spike is still actively recovering toward the cell's own next
spike under intrinsic (not passive RC) currents. Confirmed directly this
matters: an earlier version of this script tried exactly that (step from
the Iapp=0 AHP trough) and got a physically impossible NEGATIVE R_in on a
real cell, because the membrane was trending back up toward its next spike
faster than the small hyperpolarizing step pulled it down -- the fit was
capturing pacemaker recovery dynamics, not membrane RC charging. Settling
at the confirmed-silent Step-1 threshold first (matching run_chirp_
protocol.py's identical anchoring for its own subthreshold operating point)
gives this model's only genuine analogue of a resting potential; the test
step is then a further small increment past that already-quiescent
baseline (see settle_quiescent_baseline).

Feature 2 -- spike waveform: from the first spike in a baseline (held=0,
Iapp=0) trace. Cells that don't spike at baseline (none are expected to,
per the module docstring above, but this is checked rather than assumed)
simply get None for every spike/ISI field, with a note, not a fabricated
value.

Feature 3 -- ISI statistics: mean/SD/CV (global) plus local variation
LV/CV2 (Shinomoto, Shima & Tanji, 2003, "Differences in spiking patterns
among cortical neurons", Neural Computation 15(12):2823-2842) from the same
baseline spike train.

Explicitly OUT of scope (per the project's execution checklist): a fitted
adaptation time constant. An earlier attempt at this (see
run_held_injected_grid.py's compute_adaptation_ratio docstring) found a real
adapting ISI sequence checked directly wasn't obviously mono-exponential, so
only the (already-computed elsewhere) first/last-ISI adaptation ratio
exists project-wide -- no adaptation tau is attempted here either.
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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from singlecell_model_v1 import simulate
from steady_state_cache import PARAMS_DIR as DEFAULT_PARAMS_DIR
from steady_state_cache import (CACHE_PATH as DEFAULT_STEADY_STATE_CACHE_PATH,
                                load_all_cells, get_cached_state)
from find_silencing_threshold import (constant_iapp_func, settle_at_level, PROMINENCE_FRACTION, FLATLINE_MV,
                                      DEFAULT_OUTPUT_CACHE_PATH as DEFAULT_SILENCING_CACHE_PATH)

DEFAULT_OUTPUT_CACHE_PATH = ROOT_DIR / "cell_intrinsic_properties.pkl"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures" / "intrinsic_properties"
DEFAULT_FIGURE_FORMAT = "png"
# Matches find_silencing_threshold.py / run_held_injected_grid.py: temp ==
# reftemp so every cell's q10 factors collapse to 1 and conductances run at
# their literal reference-temperature values (both of those scripts were
# corrected from temp=25 to this for exactly this reason).
DEFAULT_TEMP = 10.0
DEFAULT_REFTEMP = 10.0
DEFAULT_DT_MS = 0.1

V_INDEX = 0

# First-derivative spike-onset criterion (mV/ms). 10 mV/ms is a commonly
# used convention in intracellular AP analysis (in the spirit of Bean, 2007,
# "The action potential in mammalian central neurons", Nat Rev Neurosci
# 8:451-465, which discusses dV/dt-based threshold criteria) -- but the
# literature is not fully standardized here (values from ~5 to ~20 mV/ms are
# all used by different labs/pipelines), so this is a judgment call, exposed
# via --dvdt-threshold-mv-per-ms for a reviewer to override.
DEFAULT_DVDT_THRESHOLD_MV_PER_MS = 10.0

# Samples backed off from a detected in-step spike's own peak before fitting
# the exponential relaxation, so the fit segment stays out of the fast Na+
# upstroke itself (which is not RC-relaxation behavior).
SPIKE_BACKOFF_MS = 2.0
MIN_FIT_SAMPLES = 20
# Bi-exponential fit has 5 free parameters (vs 3 for single-exponential) --
# needs more samples to be identifiable at all, not just numerically stable.
MIN_FIT_SAMPLES_BIEXP = 40
# Sag-detection thresholds (see compute_rin_tau) -- a rebound from the step
# response's own trough must clear BOTH an absolute noise floor (0.2 mV) AND
# a fraction of the dip's own depth (15%) to count as genuine Ih-mediated
# sag rather than numerical noise on an otherwise-flat/small response.
SAG_REBOUND_MIN_MV = 0.2
SAG_REBOUND_MIN_FRACTION = 0.15
# Thresholds for adopting a two-exponential fit over the single-exponential
# one (see compute_rin_tau) -- tau2 must be at least this many times tau1 to
# count as a genuinely separated fast/slow pair (not a collapsed/degenerate
# fit), and r2 must improve by at least this much to justify the 2 extra
# free parameters (not just numerically higher from added flexibility).
MIN_BIEXP_TAU_SEPARATION = 2.0
MIN_BIEXP_R2_GAIN = 0.05
# Absolute floor, not just relative to the (possibly also-bad) single-exp
# r2 -- confirmed directly a relative-only bar let a visibly bad, boundary-
# pinned fit through (5A6WBD: r2=0.478 improved on single-exp's 0.134, but
# the fit line didn't track the real trace at all).
MIN_BIEXP_R2_ABSOLUTE = 0.9


# ---------------------------------------------------------------------------
# R_in / tau: subthreshold step at held=0
# ---------------------------------------------------------------------------

def get_step_reference_nA(silencing_entry: dict):
    """The most-hyperpolarized CONFIRMED-silent bracket edge
    (silencing_threshold_bracket_nA[0], i.e. first_silent_level_nA) is used
    as this cell's own current scale to size a "small" step against -- reeds
    directly off Step 1's own per-cell result rather than an arbitrary fixed
    nA value that could be comically large for a shallow cell (e.g. 4QSWXH,
    floor ~-2 nA) or negligible for a deep one (e.g. VC08B6, floor ~-16 nA).
    Accepts "unsettled" the same way run_held_injected_grid.py's
    get_cell_floor_nA does -- first_silent_level_nA is present on both.
    """
    status = silencing_entry.get("status")
    if status == "ok":
        first_silent, _last_firing = silencing_entry["silencing_threshold_bracket_nA"]
        return float(first_silent)
    if status == "unsettled":
        return float(silencing_entry["first_silent_level_nA"])
    return None


def settle_quiescent_baseline(params, y_ss, step_reference_nA, dt, temp, reftemp, settle_kwargs):
    """Settle the cell at its own Step-1-confirmed silencing level
    (step_reference_nA, i.e. first_silent_level_nA) and return the settled
    state, along with the settle result dict for diagnostics.

    Earlier version of this function instead did a short Iapp=0 continuation
    from y_ss and launched the step from that window's voltage MINIMUM (the
    AHP trough) -- meant to avoid starting mid-spike, but didn't actually
    solve the underlying problem: every one of these cells is a spontaneous
    pacemaker at Iapp=0 (see module docstring), so even right after the AHP
    trough the membrane is already mid-recovery, actively trending back
    toward its OWN next spike under intrinsic currents (not sitting at any
    kind of passive rest). A small step superimposed on top of that ongoing
    recovery isn't measuring passive RC charging at all -- it's riding on
    top of genuine pacemaker dynamics, and confirmed directly this produces
    nonsense: a real 4QSWXH point fit an exponential asymptote ABOVE the
    starting voltage during a HYPERPOLARIZING step (the membrane was still
    trending up toward its next spike faster than the small step pulled it
    down), giving a negative R_in -- a physically impossible sign for a
    quantity that's simple Ohm's law (dV/dI) on a hyperpolarizing pulse.

    Settling at the CONFIRMED-quiescent Step-1 threshold instead gives this
    model's only genuine analogue of a passive resting potential -- exactly
    the same anchoring run_chirp_protocol.py already uses for its own
    subthreshold operating point (see that script's "Design decision 2").
    """
    hold_result = settle_at_level(params, y_ss.copy(), step_reference_nA, dt, temp, reftemp, **settle_kwargs)
    if hold_result["blew_up"]:
        return None, hold_result
    return hold_result["final_state"], hold_result


def _exp_relaxation_model(t_rel_ms, C, A, tau_ms):
    return C + A * np.exp(-t_rel_ms / tau_ms)


def fit_exponential_relaxation(t_ms: np.ndarray, v_mV: np.ndarray):
    """Single-exponential fit to a step's initial charging transient -- the
    standard method for estimating a membrane time constant from a current
    step response (RC-circuit charging equation V(t) = V_inf + (V0 -
    V_inf)*exp(-t/tau)). Returns None (not a guessed value) if the fit
    doesn't converge or lands on a non-physical (non-finite or <=0) tau.
    """
    if len(t_ms) < MIN_FIT_SAMPLES:
        return None
    t_rel = t_ms - t_ms[0]
    A0 = float(v_mV[0] - v_mV[-1])
    C0 = float(v_mV[-1])
    tau0 = max(t_rel[-1] / 5.0, 1.0)
    try:
        popt, _pcov = curve_fit(
            _exp_relaxation_model, t_rel, v_mV, p0=[C0, A0, tau0],
            bounds=([-200.0, -300.0, 0.01], [200.0, 300.0, 10.0 * t_rel[-1] + 1.0]),
            maxfev=5000)
    except (RuntimeError, ValueError):
        return None
    C, A, tau = popt
    if not np.isfinite(tau) or tau <= 0:
        return None
    v_fit = _exp_relaxation_model(t_rel, *popt)
    ss_res = float(np.sum((v_mV - v_fit) ** 2))
    ss_tot = float(np.sum((v_mV - np.mean(v_mV)) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"C_mV": float(C), "A_mV": float(A), "tau_ms": float(tau), "r2": r2,
            "t_rel_ms": t_rel, "v_fit_mV": v_fit}


def _biexp_relaxation_model(t_rel_ms, C, A1, tau1_ms, A2, tau2_ms):
    return C + A1 * np.exp(-t_rel_ms / tau1_ms) + A2 * np.exp(-t_rel_ms / tau2_ms)


def fit_biexponential_relaxation(t_ms: np.ndarray, v_mV: np.ndarray):
    """Two-exponential fit: fast RC-charging component (tau1) plus a slower
    Ih-mediated sag/rebound component (tau2) -- V(t) = C + A1*exp(-t/tau1) +
    A2*exp(-t/tau2). Tried as a follow-up to the single-exponential fit
    specifically for has_sag cases, where a single exponential is the wrong
    model by construction (see compute_rin_tau).

    tau1 is bounded to [0.01, 100] ms and tau2 to [20, 10*window+1] ms --
    a deliberate ordering constraint (not a hard non-overlap, there's a
    20-50ms zone either could land in) so the optimizer can't "label-switch"
    which component it calls fast vs slow between fit attempts, which would
    make tau1/tau2 meaningless to compare across cells. The bound values
    themselves come from this model's own documented current time constants
    (see run_chirp_protocol.py's module docstring: tauCaTM ~5-40ms, tauHM/Ih
    ~60-300+ms subthreshold) -- Ih is the sag-generating current here, so
    tau2's lower bound is set comfortably below its fast end, not at it.

    Needs more samples than the single-exponential fit (5 free parameters
    vs 3) to be identifiable at all -- returns None below MIN_FIT_SAMPLES_
    BIEXP rather than attempting an underdetermined fit.
    """
    if len(t_ms) < MIN_FIT_SAMPLES_BIEXP:
        return None
    t_rel = t_ms - t_ms[0]

    min_idx = int(np.argmin(v_mV))
    A1_0 = float(v_mV[0] - v_mV[min_idx])
    tau1_0 = max(float(t_rel[min_idx]) / 3.0, 1.0) if min_idx > 0 else 5.0
    A2_0 = float(v_mV[min_idx] - v_mV[-1])
    tau2_0 = max(float(t_rel[-1] - t_rel[min_idx]) / 2.0, 30.0)
    C0 = float(v_mV[-1])

    lower = [-200.0, -300.0, 0.01, -300.0, 20.0]
    upper = [200.0, 300.0, 100.0, 300.0, 10.0 * t_rel[-1] + 1.0]
    p0 = [C0, A1_0, tau1_0, A2_0, tau2_0]
    # p0 must lie strictly inside bounds -- clip rather than let curve_fit
    # raise on a boundary-violating initial guess (a genuinely tiny/short
    # window can push tau2_0 outside its own upper bound).
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, lower, upper)]
    try:
        popt, _pcov = curve_fit(
            _biexp_relaxation_model, t_rel, v_mV, p0=p0, bounds=(lower, upper), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    C, A1, tau1, A2, tau2 = popt
    if not (np.isfinite(tau1) and np.isfinite(tau2)) or tau1 <= 0 or tau2 <= 0:
        return None
    v_fit = _biexp_relaxation_model(t_rel, *popt)
    ss_res = float(np.sum((v_mV - v_fit) ** 2))
    ss_tot = float(np.sum((v_mV - np.mean(v_mV)) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    # Pinned-at-bound is the standard tell for a non-converged/underdetermined
    # curve_fit result: the optimizer ran out of room to explore rather than
    # settling on an interior optimum -- confirmed directly this happens here
    # (a real 5A6WBD fit landed with tau1 EXACTLY at 50.00ms and tau2 EXACTLY
    # at 3000.00ms, its two bounds, while visibly not tracking the real
    # trace despite r2=0.478 clearing the old relative-improvement-only
    # bar). 0.1% of each bound's own span is the tolerance for "at" a bound.
    at_bound = (tau1 >= upper[2] * 0.999 or tau1 <= lower[2] + 0.001 * (upper[2] - lower[2])
               or tau2 >= upper[4] * 0.999)
    return {"C_mV": float(C), "A1_mV": float(A1), "tau1_ms": float(tau1),
           "A2_mV": float(A2), "tau2_ms": float(tau2), "r2": r2, "at_bound": bool(at_bound),
           "t_rel_ms": t_rel, "v_fit_mV": v_fit}


def compute_rin_tau(params, onset_state, step_nA, step_duration_ms, dt, temp, reftemp) -> dict:
    """Apply step_nA from onset_state for step_duration_ms and fit R_in/tau
    from the response. R_in is computed from the fitted asymptote (C_mV),
    not the raw final sample -- more robust to not-fully-settled traces, and
    it ties R_in's own trustworthiness directly to the fit quality (r_in_r2)
    reported alongside it, since both come from the same fit.

    If the step itself evokes a spike (a real risk -- the cell is a
    spontaneous oscillator, and "small" here is deliberately scaled to this
    cell's own range rather than guaranteed strictly subthreshold in the
    classical rheobase sense), the fit is restricted to the pre-spike
    segment only, backed off SPIKE_BACKOFF_MS from the spike's own peak to
    stay out of the non-RC upstroke -- and this is recorded as a note, not
    silently dropped.
    """
    Iapp_step = constant_iapp_func(step_nA)
    try:
        t_ms, states = simulate(params, step_duration_ms / 1000.0, temp, dt=dt, reftemp=reftemp,
                                cis=onset_state, Iapp_func=Iapp_step)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {"blew_up": True, "error": f"step: {exc}"}
    if not np.all(np.isfinite(states)):
        return {"blew_up": True, "error": "step: non-finite trajectory"}

    v = states[:, V_INDEX]
    v_baseline = float(v[0])
    dt_ms_actual = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else dt

    v_range = float(v.max() - v.min())
    spike_idx = None
    if v_range >= FLATLINE_MV:
        peaks, _ = find_peaks(v, prominence=v_range * PROMINENCE_FRACTION)
        if len(peaks) > 0:
            spike_idx = int(peaks[0])

    notes = []
    if spike_idx is not None:
        margin_samples = max(1, int(round(SPIKE_BACKOFF_MS / dt_ms_actual)))
        end_idx = max(0, spike_idx - margin_samples)
        notes.append(f"step evoked a spike at t={t_ms[spike_idx]:.1f} ms; "
                     f"R_in/tau fit restricted to the pre-spike segment only")
    else:
        end_idx = len(v)

    base_out = {"blew_up": False, "step_t_ms": t_ms, "step_v_mV": v,
               "v_baseline_mV": v_baseline, "spike_idx": spike_idx, "fit_end_idx": end_idx}

    if end_idx < MIN_FIT_SAMPLES:
        notes.append("step onset landed too close to a spike -- no usable subthreshold segment")
        return {**base_out, "r_in_MOhm": None, "tau_ms": None, "r_in_r2": None, "has_sag": None, "notes": notes}

    # Sag detection: a single exponential is only a valid model for a
    # MONOTONIC relaxation toward one asymptote. Ih-mediated sag (fast
    # hyperpolarizing dip, then a slower partial depolarizing rebound) is a
    # real, common feature in this model (see run_held_injected_grid.py's
    # own sag_depth_map) that makes the response genuinely non-monotonic --
    # confirmed directly this happens (a real VC08B6 step) and, when it
    # does, curve_fit doesn't fail loudly: it just quietly fits the SLOW
    # rebound phase as if it were the whole relaxation, producing a
    # plausible-looking but physically wrong R_in sign/magnitude and a tau
    # pinned at its own upper bound. Detected geometrically (does the trace
    # dip to a minimum well before the fit window ends, then recover by a
    # non-trivial amount), not from fit quality (r2 alone doesn't reliably
    # distinguish this -- a moderate r2 can still hide a wrong-shaped fit).
    v_fit_window = v[:end_idx]
    min_idx = int(np.argmin(v_fit_window))
    dip_depth = v_baseline - float(v_fit_window[min_idx])
    rebound = float(v_fit_window[-1]) - float(v_fit_window[min_idx])
    has_sag = bool(min_idx < 0.7 * end_idx and dip_depth > 0
                  and rebound > max(SAG_REBOUND_MIN_MV, SAG_REBOUND_MIN_FRACTION * dip_depth))
    if has_sag:
        notes.append(f"non-monotonic (sag-like) step response detected (dip {dip_depth:.2f} mV, "
                     f"rebound {rebound:.2f} mV) -- single-exponential R_in/tau fit is not a valid "
                     f"model for this response shape and must not be trusted")

    fit = fit_exponential_relaxation(t_ms[:end_idx], v[:end_idx])
    if fit is None:
        notes.append("exponential relaxation fit did not converge")
        return {**base_out, "r_in_MOhm": None, "tau_ms": None, "r_in_r2": None, "has_sag": has_sag, "notes": notes}

    fit_model = "single_exp"
    biexp_fit = None
    if has_sag:
        # Try the two-exponential model as a follow-up specifically for the
        # sag case -- a single exponential is the wrong model by
        # construction here (see the sag-detection comment above), but that
        # doesn't mean R_in/tau are unrecoverable, only that a second
        # (slower, Ih-mediated) component needs to be fit alongside the fast
        # RC one. Adoption requires ALL of: convergence; a genuine fast/slow
        # split (tau2 meaningfully > tau1, not collapsed together); an
        # absolute (not just relative-to-single-exp) r2 floor, since a bad
        # fit can still "improve" on an even-worse baseline; neither tau
        # pinned at its own optimizer bound (the standard non-convergence
        # tell); and a physically valid R_in SIGN -- a hyperpolarizing step
        # must settle net-hyperpolarized relative to baseline no matter how
        # slow the Ih rebound is, so a positive C-v_baseline (with step_nA
        # negative) is not just unlikely, it's a contradiction, regardless
        # of how good r2 looks. Confirmed directly why all of these are
        # needed, not just some: on a 4-cell test batch, requiring only
        # convergence + tau separation + relative r2 gain "adopted" a fit
        # for every cell, but visual inspection showed 3 of 4 were
        # boundary-pinned garbage that happened to have a high or improved
        # r2 anyway (5A6WBD: tau1/tau2 both exactly at their bounds, r2=
        # 0.478, visibly not tracking the real trace; 4QSWXH and W0E22J:
        # tau2 pinned at 3000ms/its own bound despite r2 > 0.97).
        biexp_fit = fit_biexponential_relaxation(t_ms[:end_idx], v[:end_idx])
        candidate_r_in = ((biexp_fit["C_mV"] - v_baseline) / step_nA) if biexp_fit is not None else None
        if (biexp_fit is not None and biexp_fit["r2"] is not None and fit["r2"] is not None
                and not biexp_fit["at_bound"]
                and biexp_fit["tau2_ms"] > MIN_BIEXP_TAU_SEPARATION * biexp_fit["tau1_ms"]
                and biexp_fit["r2"] >= MIN_BIEXP_R2_ABSOLUTE
                and biexp_fit["r2"] - fit["r2"] > MIN_BIEXP_R2_GAIN
                and candidate_r_in is not None and candidate_r_in > 0):
            fit_model = "biexp"
            notes.append(f"single-exponential fit rejected (sag); two-exponential fit adopted "
                        f"instead (tau1={biexp_fit['tau1_ms']:.2f} ms fast + "
                        f"tau2={biexp_fit['tau2_ms']:.2f} ms slow/Ih, "
                        f"r2 {fit['r2']:.3f} -> {biexp_fit['r2']:.3f})")
        else:
            if biexp_fit is None:
                reason = "did not converge"
            elif biexp_fit["at_bound"]:
                reason = (f"tau pinned at its own optimizer bound (tau1={biexp_fit['tau1_ms']:.2f}, "
                         f"tau2={biexp_fit['tau2_ms']:.2f} ms) -- not a genuine convergence")
            elif candidate_r_in is not None and candidate_r_in <= 0:
                reason = f"resulting R_in sign is still non-physical ({candidate_r_in:.2f} MOhm)"
            else:
                reason = (f"r2 {fit['r2']:.3f} -> {biexp_fit['r2']:.3f} "
                         f"(needs >= {MIN_BIEXP_R2_ABSOLUTE} absolute)")
            notes.append(f"two-exponential fit attempted but not adopted ({reason}) -- "
                        f"R_in/tau from the single-exponential fit remains flagged invalid (has_sag)")

    chosen = biexp_fit if fit_model == "biexp" else fit
    # mV / nA = MOhm directly (1e-3 V / 1e-9 A = 1e6 Ohm) -- no extra
    # conversion factor needed. Still computed and returned even when
    # has_sag and no biexp adoption -- never silently discarded -- but the
    # caller/consumer must check has_sag (and fit_model) before trusting it
    # (same "flag, don't discard" pattern as run_chirp_protocol.py's
    # impedance_valid). tau_ms reports tau2 (the slow/Ih component) when
    # biexp is adopted, since that's the one comparable in meaning to a
    # single-exponential tau (the fast component is closer to true passive
    # membrane tau -- both are exposed via tau1_ms/tau2_ms below).
    r_in_MOhm = (chosen["C_mV"] - v_baseline) / step_nA
    reported_tau_ms = chosen["tau2_ms"] if fit_model == "biexp" else chosen["tau_ms"]
    return {**base_out, "r_in_MOhm": float(r_in_MOhm), "tau_ms": reported_tau_ms,
           "r_in_r2": chosen["r2"], "has_sag": has_sag, "fit_model": fit_model,
           "tau1_ms": biexp_fit["tau1_ms"] if biexp_fit else None,
           "tau2_ms": biexp_fit["tau2_ms"] if biexp_fit else None,
           "single_exp_r2": fit["r2"], "biexp_r2": biexp_fit["r2"] if biexp_fit else None,
           "notes": notes, "fit_t_rel_ms": chosen["t_rel_ms"], "fit_v_fit_mV": chosen["v_fit_mV"]}


# ---------------------------------------------------------------------------
# Baseline spike waveform + ISI statistics
# ---------------------------------------------------------------------------

def simulate_baseline(params, y_ss, window_s, dt, temp, reftemp) -> dict:
    Iapp0 = constant_iapp_func(0.0)
    try:
        t_ms, states = simulate(params, window_s, temp, dt=dt, reftemp=reftemp, cis=y_ss, Iapp_func=Iapp0)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return {"blew_up": True, "error": f"baseline: {exc}"}
    if not np.all(np.isfinite(states)):
        return {"blew_up": True, "error": "baseline: non-finite trajectory"}
    return {"blew_up": False, "t_ms": t_ms, "v_mV": states[:, V_INDEX]}


def detect_spikes(t_ms: np.ndarray, v_mV: np.ndarray) -> np.ndarray:
    """Same flatline/prominence convention used throughout the rest of this
    project (find_silencing_threshold.py, run_held_injected_grid.py), so
    baseline spike counts here are directly comparable to their peak counts.
    """
    v_range = float(v_mV.max() - v_mV.min())
    if v_range < FLATLINE_MV:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(v_mV, prominence=v_range * PROMINENCE_FRACTION)
    return peaks


def analyze_spike_waveform(t_ms, v_mV, dt_ms, peaks, dvdt_threshold, min_pre_spike_ms, ahp_search_ms):
    """First spike (in chronological order) with a usable dV/dt threshold
    crossing in its own pre-spike window -- not necessarily peaks[0] itself,
    since an early spike right at trace onset may not have enough preceding
    baseline to detect threshold from.

    Amplitude is defined as peak-minus-threshold (not peak-minus-resting-
    baseline) -- an explicit choice: "resting baseline" isn't well-defined
    for a spontaneously oscillating cell the way it is for a quiescent one,
    while threshold (the point dV/dt takes off) is directly measured on this
    same spike. Half-width is measured at the half-amplitude level relative
    to that same threshold-referenced amplitude. AHP depth is likewise
    measured relative to threshold (not pre-spike baseline), for the same
    reason -- positive means the post-spike trough dips below threshold.
    AHP duration is peak-to-trough time (not trough-to-recovery), the
    simpler and less ambiguous of the two conventions since "recovery" would
    need its own return-to-baseline criterion.
    """
    if len(peaks) == 0:
        return None, ["no spikes detected in the baseline trace"]

    dvdt = np.gradient(v_mV, dt_ms)  # mV/ms
    for k, peak_idx in enumerate(peaks):
        prev_bound = int(peaks[k - 1]) if k > 0 else 0
        back_samples = int(round(min_pre_spike_ms / dt_ms))
        search_start = max(prev_bound, peak_idx - back_samples, 0)
        if search_start >= peak_idx:
            continue

        window = dvdt[search_start:peak_idx + 1]
        crossing = np.where(window >= dvdt_threshold)[0]
        if len(crossing) == 0:
            continue
        threshold_idx = search_start + int(crossing[0])
        threshold_mV = float(v_mV[threshold_idx])
        amplitude_mV = float(v_mV[peak_idx] - threshold_mV)
        if amplitude_mV <= 0:
            continue
        half_v = threshold_mV + amplitude_mV / 2.0

        rise_seg = np.where(v_mV[threshold_idx:peak_idx + 1] >= half_v)[0]
        if len(rise_seg) == 0:
            continue
        rise_idx = threshold_idx + int(rise_seg[0])
        t_rise = _interp_crossing_time(t_ms, v_mV, rise_idx, half_v)

        next_bound = int(peaks[k + 1]) if k + 1 < len(peaks) else len(v_mV) - 1
        search_end = min(next_bound, peak_idx + int(round(ahp_search_ms / dt_ms)), len(v_mV) - 1)

        fall_seg = np.where(v_mV[peak_idx:search_end + 1] <= half_v)[0]
        if len(fall_seg) == 0:
            t_fall = None
            half_width_ms = None
        else:
            fall_idx = peak_idx + int(fall_seg[0])
            t_fall = _interp_crossing_time(t_ms, v_mV, fall_idx, half_v)
            half_width_ms = float(t_fall - t_rise)

        max_dvdt = float(np.max(dvdt[threshold_idx:peak_idx + 1]))

        if search_end <= peak_idx:
            trough_idx = None
        else:
            trough_idx = peak_idx + int(np.argmin(v_mV[peak_idx:search_end + 1]))

        if trough_idx is not None:
            ahp_depth_mV = float(threshold_mV - v_mV[trough_idx])
            ahp_duration_ms = float(t_ms[trough_idx] - t_ms[peak_idx])
        else:
            ahp_depth_mV = None
            ahp_duration_ms = None

        result = {
            "spike_threshold_mV": threshold_mV, "spike_amplitude_mV": amplitude_mV,
            "spike_half_width_ms": half_width_ms, "spike_max_dvdt": max_dvdt,
            "ahp_depth_mV": ahp_depth_mV, "ahp_duration_ms": ahp_duration_ms,
            "peak_idx": int(peak_idx), "threshold_idx": int(threshold_idx),
            "trough_idx": int(trough_idx) if trough_idx is not None else None,
            "peak_mV": float(v_mV[peak_idx]),
        }
        return result, []

    return None, ["no baseline spike had a usable dV/dt threshold crossing within its own "
                 "pre-spike search window"]


def _interp_crossing_time(t_ms, v_mV, idx, target_v):
    """Linear interpolation between sample idx-1 and idx for a sub-timestep
    estimate of when v_mV crosses target_v -- half-width/AHP-duration
    otherwise inherit the raw dt (0.1ms default) as their effective
    resolution floor, which is coarse relative to a typical spike's
    half-width.
    """
    if idx <= 0 or v_mV[idx] == v_mV[idx - 1]:
        return float(t_ms[idx])
    frac = (target_v - v_mV[idx - 1]) / (v_mV[idx] - v_mV[idx - 1])
    frac = min(max(frac, 0.0), 1.0)
    return float(t_ms[idx - 1] + frac * (t_ms[idx] - t_ms[idx - 1]))


def compute_isi_stats(peak_times_ms: np.ndarray) -> dict:
    """Mean/SD/CV plus local variation (LV/CV2) of the baseline ISI
    sequence.

    SD/CV use ddof=1 (sample standard deviation) -- a judgment call flagged
    here rather than presented as the one true convention: some spike-train
    analysis code instead uses the population (ddof=0) estimator, and with
    the ISI counts typical here (tens, not hundreds) the difference is
    modest either way.

    LV (local variation): Shinomoto, Shima & Tanji (2003), "Differences in
    spiking patterns among cortical neurons", Neural Computation
    15(12):2823-2842 --
        LV = 3/(n-1) * sum_{i=1}^{n-1} ((ISI_i - ISI_{i+1}) / (ISI_i + ISI_{i+1}))^2
    where n = number of ISIs (so n-1 consecutive pairs). Requires n >= 2
    (i.e. >=3 spikes); n_isis < 2 reports None rather than a degenerate
    single-term value.
    """
    if len(peak_times_ms) < 2:
        return {"isi_mean_ms": None, "isi_sd_ms": None, "isi_cv": None, "isi_lv": None, "n_isis": 0}

    isis = np.diff(peak_times_ms)
    n = len(isis)
    isi_mean_ms = float(np.mean(isis))

    if n >= 2:
        isi_sd_ms = float(np.std(isis, ddof=1))
        isi_cv = (isi_sd_ms / isi_mean_ms) if isi_mean_ms > 0 else None
    else:
        isi_sd_ms = None
        isi_cv = None

    if n >= 2:
        num = isis[:-1] - isis[1:]
        den = isis[:-1] + isis[1:]
        terms = (num / den) ** 2
        isi_lv = float((3.0 / (n - 1)) * np.sum(terms))
    else:
        isi_lv = None

    return {"isi_mean_ms": isi_mean_ms, "isi_sd_ms": isi_sd_ms, "isi_cv": isi_cv,
            "isi_lv": isi_lv, "n_isis": n}


# ---------------------------------------------------------------------------
# Top-level per-cell orchestrator
# ---------------------------------------------------------------------------

def extract_cell_intrinsic_properties(cell_id: str, params: np.ndarray, ss_entry, silencing_entry,
                                      args: argparse.Namespace) -> tuple:
    """Never raises -- any failure is captured in the returned dict's
    "status" field, matching find_silencing_threshold.py's convention.
    Returns (result, trace_data): trace_data (full arrays for the diagnostic
    figure) is kept out of the persisted cache, same reasoning as
    find_silencing_threshold.py's fine_traces / run_held_injected_grid.py's
    "_trace_*" convention.
    """
    base = {"cell_id": cell_id, "params": params, "dt": args.dt, "temp": args.temp, "reftemp": args.reftemp}

    if ss_entry is None:
        return {**base, "status": "no_cached_steady_state"}, {}
    if silencing_entry is None:
        return {**base, "status": "no_silencing_threshold"}, {}

    step_reference_nA = get_step_reference_nA(silencing_entry)
    if step_reference_nA is None:
        return {**base, "status": "silencing_not_ok"}, {}

    y_ss = ss_entry["y_ss"]
    period_ms = ss_entry.get("period_ms")
    # Additional increment PAST the already-quiescent reference level, not a
    # fraction of it from scratch -- see settle_quiescent_baseline's
    # docstring for why anchoring to a still-oscillating state doesn't work.
    # Always strictly MORE hyperpolarizing (abs() then negated) so the test
    # step can only move further from threshold than the settled baseline
    # already is, never back toward it.
    step_delta_nA = -abs(args.step_fraction * step_reference_nA)
    step_nA = step_reference_nA + step_delta_nA

    settle_kwargs = dict(chunk_s=args.hold_settle_chunk_s, max_settle_s=args.max_hold_settle_s,
                         settle_rtol=args.hold_settle_rtol, min_peaks_for_rate=args.min_peaks_for_rate)
    onset_state, hold_result = settle_quiescent_baseline(
        params, y_ss, step_reference_nA, args.dt, args.temp, args.reftemp, settle_kwargs)
    if onset_state is None:
        return {**base, "status": "blew_up", "error": hold_result.get("error"), "stage": "hold"}, {}
    notes = []
    if not hold_result["is_flatline"]:
        # step_reference_nA is Step 1's own confirmed-silent level, so this
        # should never actually still be oscillating -- if it is, the
        # silencing cache entry is likely stale/from different params than
        # the ones just loaded (same check run_chirp_protocol.py's
        # get_chirp_baseline does for its own held baseline).
        notes.append("settle at step_reference_nA did not reach flatline -- R_in/tau baseline "
                    "may still be oscillating, not a genuine passive rest state")

    step_result = compute_rin_tau(params, onset_state, step_nA, args.step_duration_ms,
                                  args.dt, args.temp, args.reftemp)
    if step_result.get("blew_up"):
        return {**base, "status": "blew_up", "error": step_result.get("error"), "stage": "step"}, {}
    notes += step_result.get("notes", [])

    baseline_window_s = args.min_baseline_window_s
    if period_ms:
        baseline_window_s = max(baseline_window_s, args.baseline_periods * period_ms / 1000.0)
    baseline_window_s = min(baseline_window_s, args.max_baseline_window_s)

    baseline = simulate_baseline(params, y_ss, baseline_window_s, args.dt, args.temp, args.reftemp)
    if baseline.get("blew_up"):
        return {**base, "status": "blew_up", "error": baseline.get("error")}, {}

    t_b, v_b = baseline["t_ms"], baseline["v_mV"]
    dt_ms_actual = float(t_b[1] - t_b[0]) if len(t_b) > 1 else args.dt
    peaks = detect_spikes(t_b, v_b)
    waveform, wf_notes = analyze_spike_waveform(t_b, v_b, dt_ms_actual, peaks,
                                                args.dvdt_threshold_mv_per_ms,
                                                args.min_pre_spike_ms, args.ahp_search_ms)
    notes += wf_notes
    isi_stats = compute_isi_stats(t_b[peaks] if len(peaks) else np.array([]))

    result = {
        **base, "status": "ok", "error": None,
        "r_in_MOhm": step_result.get("r_in_MOhm"), "tau_ms": step_result.get("tau_ms"),
        "r_in_r2": step_result.get("r_in_r2"), "has_sag": step_result.get("has_sag"),
        "fit_model": step_result.get("fit_model"), "tau1_ms": step_result.get("tau1_ms"),
        "tau2_ms": step_result.get("tau2_ms"), "single_exp_r2": step_result.get("single_exp_r2"),
        "biexp_r2": step_result.get("biexp_r2"),
        "step_nA_used": float(step_nA), "step_reference_nA": step_reference_nA,
        "step_delta_nA": float(step_delta_nA), "hold_settled": hold_result["settled"],
        "hold_is_flatline": hold_result["is_flatline"],
        "step_duration_ms": args.step_duration_ms,
        "spike_threshold_mV": waveform["spike_threshold_mV"] if waveform else None,
        "spike_amplitude_mV": waveform["spike_amplitude_mV"] if waveform else None,
        "spike_half_width_ms": waveform["spike_half_width_ms"] if waveform else None,
        "spike_max_dvdt": waveform["spike_max_dvdt"] if waveform else None,
        "ahp_depth_mV": waveform["ahp_depth_mV"] if waveform else None,
        "ahp_duration_ms": waveform["ahp_duration_ms"] if waveform else None,
        "isi_mean_ms": isi_stats["isi_mean_ms"], "isi_sd_ms": isi_stats["isi_sd_ms"],
        "isi_cv": isi_stats["isi_cv"], "isi_lv": isi_stats["isi_lv"], "n_isis": isi_stats["n_isis"],
        "n_baseline_spikes": int(len(peaks)), "baseline_window_s_used": baseline_window_s,
        "notes": notes,
    }

    trace_data = {
        "step_t_ms": step_result.get("step_t_ms"), "step_v_mV": step_result.get("step_v_mV"),
        "step_v_baseline_mV": step_result.get("v_baseline_mV"),
        "step_fit_t_rel_ms": step_result.get("fit_t_rel_ms"), "step_fit_v_fit_mV": step_result.get("fit_v_fit_mV"),
        "step_nA": step_nA,
        "baseline_t_ms": t_b, "baseline_v_mV": v_b, "waveform": waveform,
    }
    return result, trace_data


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_pickle_cache(cache_path: Path) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    return {}


def save_output_cache(cache: dict, cache_path: Path = DEFAULT_OUTPUT_CACHE_PATH) -> None:
    # Write-then-rename + PermissionError retry -- copied exactly from
    # run_held_injected_grid.py's save_output_cache (see its comment for
    # why: os.replace can transiently fail on Windows with WinError 5 if a
    # concurrent reader briefly has cache_path open, e.g. checking progress
    # on this same run).
    cache_path = Path(cache_path)
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

def _fmt(x, spec="{:.3f}"):
    return spec.format(x) if x is not None else "n/a"


def plot_cell_intrinsic(result: dict, trace_data: dict, outdir: Path, command: str,
                        fig_format: str = DEFAULT_FIGURE_FORMAT) -> None:
    cell_id = result["cell_id"]
    fig, (ax_step, ax_spike) = plt.subplots(1, 2, figsize=(12, 4.5))

    t_step, v_step = trace_data.get("step_t_ms"), trace_data.get("step_v_mV")
    if t_step is not None:
        ax_step.plot(t_step, v_step, color="steelblue", lw=0.8, label="V(t)")
        v_base = trace_data.get("step_v_baseline_mV")
        if v_base is not None:
            ax_step.axhline(v_base, color="gray", ls=":", lw=1, label="pre-step baseline")
        fit_t_rel = trace_data.get("step_fit_t_rel_ms")
        fit_v = trace_data.get("step_fit_v_fit_mV")
        fit_model = result.get("fit_model")
        if fit_t_rel is not None:
            if fit_model == "biexp":
                fit_label = (f"2-exp fit (tau1={_fmt(result.get('tau1_ms'), '{:.2f}')}ms + "
                            f"tau2={_fmt(result.get('tau2_ms'), '{:.2f}')}ms)")
            else:
                fit_label = f"exp fit (tau={_fmt(result.get('tau_ms'), '{:.2f}')} ms)"
            ax_step.plot(t_step[0] + fit_t_rel, fit_v, color="firebrick", lw=1.2, ls="--", label=fit_label)
        has_sag = result.get("has_sag")
        title = (f"step response ({trace_data.get('step_nA', float('nan')):.3f} nA)\n"
                f"R_in={_fmt(result.get('r_in_MOhm'), '{:.1f}')} MOhm, "
                f"r2={_fmt(result.get('r_in_r2'), '{:.3f}')}")
        if has_sag and fit_model == "biexp":
            ax_step.set_title(title + "  [sag present, resolved via 2-exp fit]", fontsize=8, color="darkorange")
            ax_step.text(0.5, 0.15, "sag detected, but a two-exponential fit\n"
                                   "converged with a genuine fast+slow split", transform=ax_step.transAxes,
                        ha="center", va="center", fontsize=7, color="darkorange", style="italic",
                        bbox=dict(boxstyle="round", facecolor="#fdf0dc", edgecolor="darkorange", alpha=0.9))
        elif has_sag:
            ax_step.set_title(title + "  [INVALID: sag detected]", fontsize=8, color="firebrick")
            ax_step.text(0.5, 0.5, "INVALID -- non-monotonic (sag-like) response;\n"
                                  "single-exponential fit does not apply\n"
                                  "(2-exp fit attempted, not adopted -- see notes)",
                        transform=ax_step.transAxes, ha="center", va="center", fontsize=7,
                        color="firebrick", style="italic",
                        bbox=dict(boxstyle="round", facecolor="mistyrose", edgecolor="firebrick", alpha=0.9))
        else:
            ax_step.set_title(title, fontsize=8)
        ax_step.legend(fontsize=6, loc="best")
    else:
        ax_step.text(0.5, 0.5, "no step trace available", ha="center", va="center",
                    transform=ax_step.transAxes, fontsize=8, color="dimgray")
    ax_step.set_xlabel("time (ms)")
    ax_step.set_ylabel("V (mV)")

    waveform = trace_data.get("waveform")
    t_b, v_b = trace_data.get("baseline_t_ms"), trace_data.get("baseline_v_mV")
    if waveform is not None and t_b is not None:
        dt_ms_actual = float(t_b[1] - t_b[0]) if len(t_b) > 1 else 0.1
        pk = waveform["peak_idx"]
        pre_samples = int(round(15.0 / dt_ms_actual))
        post_samples = int(round(80.0 / dt_ms_actual))
        lo = max(0, waveform["threshold_idx"] - pre_samples)
        hi = min(len(t_b), pk + post_samples)
        ax_spike.plot(t_b[lo:hi], v_b[lo:hi], color="black", lw=0.8)
        ax_spike.scatter([t_b[waveform["threshold_idx"]]], [waveform["spike_threshold_mV"]],
                         color="darkorange", zorder=5, label="threshold", s=25)
        ax_spike.scatter([t_b[pk]], [waveform["peak_mV"]], color="firebrick", zorder=5,
                         label="peak", s=25)
        if waveform.get("trough_idx") is not None:
            ax_spike.scatter([t_b[waveform["trough_idx"]]], [v_b[waveform["trough_idx"]]],
                             color="steelblue", zorder=5, label="AHP trough", s=25)
        ax_spike.legend(fontsize=6, loc="best")
        ax_spike.set_title(
            f"first baseline spike\namp={_fmt(waveform['spike_amplitude_mV'], '{:.1f}')} mV, "
            f"hw={_fmt(waveform['spike_half_width_ms'], '{:.2f}')} ms, "
            f"max dV/dt={_fmt(waveform['spike_max_dvdt'], '{:.0f}')} mV/ms", fontsize=8)
    else:
        ax_spike.text(0.5, 0.5, "no spike detected at baseline", ha="center", va="center",
                     transform=ax_spike.transAxes, fontsize=8, color="dimgray")
    ax_spike.set_xlabel("time (ms)")
    ax_spike.set_ylabel("V (mV)")

    fig.suptitle(f"{cell_id} — status: {result['status']}", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
    fig.text(0.5, 0.01, command, ha="center", va="bottom",
             fontsize=6, family="monospace", color="dimgray", wrap=True)
    outpath = outdir / f"{cell_id}_intrinsic.{fig_format}"
    fig.savefig(outpath, format=fig_format, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2c intrinsic-property extraction that needs real re-simulation: "
                    "R_in/tau from a small subthreshold hyperpolarizing step at held=0, "
                    "spike waveform features from the first baseline spike, and ISI "
                    "statistics (mean/SD/CV/LV) from the baseline spike train.")
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cell ID(s). Default: every cell found in --params-dir.")
    parser.add_argument("--steady-state-cache", default=DEFAULT_STEADY_STATE_CACHE_PATH,
                        help="Path to the Iapp=0 steady-state cache (input, read-only).")
    parser.add_argument("--silencing-cache", default=DEFAULT_SILENCING_CACHE_PATH,
                        help="Path to Step 1's cell_silencing_thresholds.pkl (input, read-only; "
                             "used both as the settle target for R_in/tau's quiescent baseline "
                             "and to scale the test-step delta per cell).")
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE_PATH)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--figure-format", default=DEFAULT_FIGURE_FORMAT, choices=["png", "svg", "pdf"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--reftemp", type=float, default=DEFAULT_REFTEMP)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--jobs", type=int, default=-1)

    parser.add_argument("--step-fraction", type=float, default=0.2,
                        help="Subthreshold test-step DELTA, applied on top of the cell's own "
                             "already-settled silencing_threshold_bracket_nA[0] (the confirmed-"
                             "silent bracket edge, used as this model's only genuine analogue of "
                             "a passive resting potential -- see settle_quiescent_baseline). "
                             "Delta = this fraction of that reference level, always applied as "
                             "FURTHER hyperpolarization (never back toward threshold). Scaled per "
                             "cell, not a fixed nA value that could be negligible for a deep cell "
                             "or too large for a shallow one.")
    parser.add_argument("--step-duration-ms", type=float, default=1000.0,
                        help="Step duration. 1000ms (not the naive 'tens of ms is enough for a "
                             "membrane tau' guess) because sag-affected cells need real time for "
                             "the slow Ih-mediated rebound component to develop and separate from "
                             "the fast RC component in a two-exponential fit -- confirmed directly "
                             "300ms only let 1 of 4 sag-affected test cells resolve a valid "
                             "two-exponential fit, 1000ms let 2 of 4 (see compute_rin_tau). Longer "
                             "than this risks the cell firing its own spontaneous next spike before "
                             "the step ends (these are all pacemakers) rather than helping further.")
    parser.add_argument("--hold-settle-chunk-s", type=float, default=2.0,
                        help="Integration chunk length (s) while settling at step_reference_nA "
                             "before the test step -- see find_silencing_threshold.py's "
                             "settle_at_level.")
    parser.add_argument("--max-hold-settle-s", type=float, default=20.0)
    parser.add_argument("--hold-settle-rtol", type=float, default=0.05)
    parser.add_argument("--min-peaks-for-rate", type=int, default=2)

    parser.add_argument("--baseline-periods", type=float, default=25.0,
                        help="Baseline (held=0) window for spike-waveform/ISI extraction = "
                             "max(--min-baseline-window-s, this many baseline periods), capped "
                             "at --max-baseline-window-s.")
    parser.add_argument("--min-baseline-window-s", type=float, default=5.0)
    parser.add_argument("--max-baseline-window-s", type=float, default=30.0)

    parser.add_argument("--dvdt-threshold-mv-per-ms", type=float, default=DEFAULT_DVDT_THRESHOLD_MV_PER_MS,
                        help="dV/dt threshold (mV/ms) used for spike-onset (threshold voltage) "
                             "detection. See module docstring -- this is a standard-ish but not "
                             "fully universal convention.")
    parser.add_argument("--min-pre-spike-ms", type=float, default=20.0,
                        help="How far back before a candidate spike's peak to search for its "
                             "own dV/dt threshold crossing.")
    parser.add_argument("--ahp-search-ms", type=float, default=150.0,
                        help="How far after a spike's peak to search for its AHP trough / "
                             "half-width falling crossing, bounded by the next spike if closer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = "python " + " ".join(sys.argv)
    params_dir = Path(args.params_dir)
    ss_cache_path = Path(args.steady_state_cache)
    silencing_cache_path = Path(args.silencing_cache)
    output_cache_path = Path(args.output_cache)
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
    print(f"Extracting intrinsic properties for {len(cells)} of {len(all_cells)} cell(s) "
          f"from {params_dir}")

    # Read-only: cell_silencing_thresholds.pkl is never written by this script.
    silencing_cache = load_pickle_cache(silencing_cache_path)
    output_cache = load_pickle_cache(output_cache_path)

    def process(cell_id: str, params: np.ndarray):
        ss_entry = get_cached_state(cell_id, params, cache_path=ss_cache_path)
        silencing_entry = silencing_cache.get(cell_id)
        result, trace_data = extract_cell_intrinsic_properties(cell_id, params, ss_entry, silencing_entry, args)
        return cell_id, result, trace_data

    # Save+plot after EVERY cell -- same incremental-save pattern as
    # find_silencing_threshold.py / run_held_injected_grid.py (see their
    # comments): streams results via generator_unordered, write-then-rename
    # keeps a concurrent reader safe.
    if args.jobs == 1:
        result_iter = (process(cid, params) for cid, params in cells.items())
    else:
        result_iter = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(process)(cid, params) for cid, params in cells.items())

    status_counts: dict = {}
    n_done = 0
    for cell_id, result, trace_data in result_iter:
        output_cache[cell_id] = result
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        if not args.no_plot and result["status"] == "ok":
            plot_cell_intrinsic(result, trace_data, figures_dir, command, fig_format=args.figure_format)
        save_output_cache(output_cache, output_cache_path)
        n_done += 1
        print(f"  [{n_done}/{len(cells)}] {cell_id}: {result['status']}")

    print("\nStatus summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    sagged = [cid for cid in cells if output_cache[cid].get("status") == "ok"
             and output_cache[cid].get("has_sag")]
    sagged_resolved = [cid for cid in sagged if output_cache[cid].get("fit_model") == "biexp"]
    sagged_unresolved = [cid for cid in sagged if cid not in sagged_resolved]
    if sagged_resolved:
        print(f"\n{len(sagged_resolved)} cell(s) showed sag but a two-exponential fit resolved "
             f"R_in/tau anyway: {sagged_resolved}")
    if sagged_unresolved:
        print(f"\n{len(sagged_unresolved)} cell(s) showed sag with no valid fit (single- or "
             f"two-exponential) -- R_in/tau are not available for these: {sagged_unresolved}")
    print(f"\nCache written to {output_cache_path}")
    if not args.no_plot:
        print(f"Figures written to {figures_dir}/")

    flagged = [cid for cid in cells if output_cache[cid]["status"] != "ok"]
    if flagged:
        print(f"\n{len(flagged)} cell(s) did not reach status 'ok': {flagged}")
        print("'no_cached_steady_state': run generate_steady_state.py first. "
              "'no_silencing_threshold'/'silencing_not_ok': run find_silencing_threshold.py "
              "first. 'blew_up': consider a smaller --dt for just that cell.")


if __name__ == "__main__":
    main()
