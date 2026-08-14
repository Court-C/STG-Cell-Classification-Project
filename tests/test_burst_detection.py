"""Synthetic ground-truth tests for the spike-detection and burst/tonic
classification functions in find_silencing_threshold.py. These are fed
fabricated arrays with a KNOWN correct answer, so they're executable proof
rather than eyeballed traces -- the repo has zero automated tests otherwise
(confirmed by grep -r pytest|unittest src/ -> no hits), and several of these
cases replicate specific real-cell edge cases already narrated in
classify_burst_pattern's own docstring (the synthetic doublet-train case,
the VC08B6 near-miss), turning that prose justification into a re-checkable
assertion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from find_silencing_threshold import (classify_burst_pattern, count_spikes_and_rate,
                                      compute_isis_ms, PROMINENCE_FRACTION, FLATLINE_MV)
from run_held_injected_grid import compute_pre_spike_sag_trough, compute_adaptation_ratio

DEFAULT_KWARGS = dict(min_isis_for_burst_test=6, isi_mode_prominence_frac=0.05, min_isi_ratio=1.5)


def test_clean_tonic_train_classified_tonic():
    rng = np.random.default_rng(0)
    isis_ms = np.full(20, 25.0) + rng.normal(0, 0.1, 20)
    result = classify_burst_pattern(isis_ms, n_peaks=21, **DEFAULT_KWARGS)
    assert result["pattern"] == "tonic"


def test_synthetic_doublet_train_classified_bursting():
    """Replicates the exact synthetic case narrated in classify_burst_pattern's
    own docstring: a clean 31.4ms/60.0ms alternating doublet train (only
    Gaussian jitter) scores ~1.97 spikes/burst, just under 2.0, because the
    sampled window's trailing spike after the last long ISI has no partner
    -- this is exactly why min_spikes_per_burst is 1.5, not a literal 2.0.
    """
    # 32 complete doublets (short ISI within each doublet, long ISI between
    # doublets), the window ending right on a long ISI -- i.e. a 65th
    # "dangling" spike whose own ISI to the next doublet was never sampled.
    # 64 ISIs (32 short + 32 long) -> 65 spikes, 33 bursts -> 65/33 = 1.97,
    # matching the docstring's exact numbers.
    rng = np.random.default_rng(1)
    n_doublets = 32
    seq = []
    for _ in range(n_doublets):
        seq.append(31.4 + rng.normal(0, 0.3))
        seq.append(60.0 + rng.normal(0, 0.3))
    isis_ms = np.array(seq)
    result = classify_burst_pattern(isis_ms, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    assert result["pattern"] == "bursting"
    assert result["diagnostics"]["avg_spikes_per_burst"] == pytest.approx(65 / 33, abs=0.01)
    assert result["diagnostics"]["avg_spikes_per_burst"] < 2.0
    # Non-regression: this known-good case must also clear the new Ashman's D
    # gate -- a clean 31.4/60.0ms separation should score far above threshold.
    assert result["diagnostics"]["ashman_d"] >= 2.0


def test_vc08b6_near_miss_correctly_rejected_as_tonic():
    """Replicates the real VC08B6 (held=-12.5, inj=-4.4) false-positive case
    documented in classify_burst_pattern: ISIs mostly ~14.5ms (long) with a
    MINORITY of isolated ~9.6ms (short) ISIs scattered in -- not a clean
    50/50 alternation -- averaging only ~1.2-1.3 spikes/burst (n_bursts=174,
    1.29 spikes/burst in the real case: almost every "burst" was a single
    isolated spike with mild ISI jitter around it, not a real burst). The
    isi_long/isi_short ratio (~1.51x) alone would pass the old ratio-only
    check; min_spikes_per_burst=1.5 must catch what the ratio check misses
    and reject this as tonic, not bursting.
    """
    rng = np.random.default_rng(2)
    n_total, n_short = 100, 15
    short_positions = set(rng.choice(np.arange(1, n_total - 1, 2), size=n_short, replace=False))
    isis_ms = np.array([
        (9.6 if i in short_positions else 14.5) + rng.normal(0, 0.2) for i in range(n_total)
    ])
    result = classify_burst_pattern(isis_ms, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    assert result["pattern"] == "tonic"


def test_insufficient_data_below_minimum_isi_count():
    isis_ms = np.array([20.0, 22.0])
    result = classify_burst_pattern(isis_ms, n_peaks=3, **DEFAULT_KWARGS)
    assert result["pattern"] == "sparse"
    assert result["diagnostics"] is None


def _make_burst_train_with_long_tail(rng, n_bursts, mean_short=10.0, sd_short=0.5,
                                     mean_long=20.0, sd_long=2.0, tail_frac=0.1,
                                     tail_mean=55.0, tail_sd=6.0, spikes_per_burst=3):
    """3-spike bursts (2 short intra-burst ISIs + 1 inter-burst ISI), where a
    minority (tail_frac) of inter-burst ISIs are drawn from a much wider,
    farther-out distribution instead of the main inter-burst mode. The KDE
    still resolves two clean peaks (the tail is a minority, not its own
    third mode) and the resulting short/long cluster means still pass the
    ratio and spikes-per-burst gates comfortably -- but the tail inflates
    the long cluster's standard deviation enough to pull Ashman's D below
    the well-separated threshold, exactly the "two clusters that pass shape
    checks but aren't actually well-separated populations" case the D-gate
    exists to catch.
    """
    seq = []
    for _ in range(n_bursts):
        for _ in range(spikes_per_burst - 1):
            seq.append(max(1.0, rng.normal(mean_short, sd_short)))
        if rng.random() < tail_frac:
            seq.append(max(1.0, rng.normal(tail_mean, tail_sd)))
        else:
            seq.append(max(1.0, rng.normal(mean_long, sd_long)))
    return np.array(seq[:-1])


def test_high_variance_cluster_rejected_by_ashman_d_alone():
    """Ratio and spikes-per-burst gates both pass on this train (confirmed
    below), so without the Ashman's D gate this would be called "bursting"
    -- but the long cluster's tail-inflated spread pulls D to ~1.75, under
    the default min_ashman_d=2.0 threshold, so it must be rejected as
    "tonic". This isolates the new gate: it's the only one of the four that
    catches this specific case.
    """
    isis_ms = _make_burst_train_with_long_tail(np.random.default_rng(11), n_bursts=60, tail_frac=0.1)
    result = classify_burst_pattern(isis_ms, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    diag = result["diagnostics"]
    ratio = diag["isi_long_ms"] / diag["isi_short_ms"]
    assert ratio >= DEFAULT_KWARGS["min_isi_ratio"], "ratio gate should pass on this data"
    assert diag["avg_spikes_per_burst"] >= 1.5, "spikes-per-burst gate should pass on this data"
    assert diag["ashman_d"] < 2.0
    assert result["pattern"] == "tonic"

    # Same data, looser D threshold -> confirms only the D gate is doing the
    # rejecting above, not some other side effect of this construction.
    looser = classify_burst_pattern(isis_ms, min_ashman_d=1.5, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    assert looser["pattern"] == "bursting"


def test_clean_bimodal_train_passes_ashman_d_gate():
    """A well-separated, low-variance alternating train (10ms/100ms, only
    small Gaussian jitter) must still classify as "bursting" with Ashman's D
    comfortably above threshold -- the new gate must not reject genuine,
    cleanly-separated bursters.
    """
    rng = np.random.default_rng(9)
    seq = []
    for _ in range(20):
        seq.append(10.0 + rng.normal(0, 0.3))
        seq.append(100.0 + rng.normal(0, 0.5))
    isis_ms = np.array(seq)
    result = classify_burst_pattern(isis_ms, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    assert result["pattern"] == "bursting"
    assert result["diagnostics"]["ashman_d"] >= 2.0


def test_ashman_d_gate_flips_pattern_near_the_boundary():
    """Two near-identical long-tail trains (see
    _make_burst_train_with_long_tail), differing only in how often the tail
    fires, straddle the default min_ashman_d=2.0 threshold from opposite
    sides -- isolating the gate's actual decision boundary on real
    (non-degenerate) data rather than a single one-sided example.
    """
    above = _make_burst_train_with_long_tail(np.random.default_rng(11), n_bursts=60, tail_frac=0.07)
    below = _make_burst_train_with_long_tail(np.random.default_rng(11), n_bursts=60, tail_frac=0.08)
    r_above = classify_burst_pattern(above, n_peaks=len(above) + 1, **DEFAULT_KWARGS)
    r_below = classify_burst_pattern(below, n_peaks=len(below) + 1, **DEFAULT_KWARGS)
    assert r_above["diagnostics"]["ashman_d"] >= 2.0
    assert r_above["pattern"] == "bursting"
    assert r_below["diagnostics"]["ashman_d"] < 2.0
    assert r_below["pattern"] == "tonic"


def test_spike_detection_ignores_subthreshold_noise():
    """A fabricated trace with 5 real spikes (large, fast, prominent) plus
    3 small subthreshold noise bumps -- count_spikes_and_rate/compute_isis_ms
    must find exactly the 5 real spikes at PROMINENCE_FRACTION, none of the
    noise.
    """
    dt_ms = 0.1
    t_ms = np.arange(0, 500, dt_ms)
    v = np.full_like(t_ms, -50.0)
    real_spike_times_ms = [50, 130, 210, 300, 400]
    for st in real_spike_times_ms:
        idx = int(st / dt_ms)
        width = 20
        window = np.arange(-width, width)
        v[idx - width:idx + width] += 60.0 * np.exp(-0.5 * (window / 4.0) ** 2)
    noise_times_ms = [80, 170, 350]
    for nt in noise_times_ms:
        idx = int(nt / dt_ms)
        width = 30
        window = np.arange(-width, width)
        v[idx - width:idx + width] += 5.0 * np.exp(-0.5 * (window / 8.0) ** 2)

    freq_hz, n_peaks, is_flatline = count_spikes_and_rate(v, t_ms[-1], min_peaks_for_rate=2)
    assert not is_flatline
    assert n_peaks == len(real_spike_times_ms)

    isis_ms, n_peaks2 = compute_isis_ms(v, t_ms, PROMINENCE_FRACTION)
    assert n_peaks2 == len(real_spike_times_ms)
    assert len(isis_ms) == len(real_spike_times_ms) - 1


def test_flatline_trace_reports_silent_no_spikes():
    t_ms = np.arange(0, 1000, 0.1)
    v = np.full_like(t_ms, -60.0) + np.random.default_rng(3).normal(0, 0.01, len(t_ms))
    freq_hz, n_peaks, is_flatline = count_spikes_and_rate(v, t_ms[-1], min_peaks_for_rate=2)
    assert is_flatline
    assert freq_hz == 0.0
    assert n_peaks == 0


def test_sag_trough_finds_known_minimum_before_spike():
    """A trace that dips from baseline (-40mV) to a known trough (-60mV) at
    a known time, then fires a spike -- compute_pre_spike_sag_trough must
    recover the exact trough value and stop the search at the spike.
    """
    dt_ms = 0.1
    t_ms = np.arange(0, 1000, dt_ms)
    baseline, trough_v, trough_ms = -40.0, -60.0, 150.0
    tau_ms = 60.0
    v = baseline - (baseline - trough_v) * np.exp(-t_ms / tau_ms)
    # Recovery back toward baseline until the spike.
    recover_mask = t_ms > trough_ms
    v[recover_mask] = trough_v + (baseline - trough_v) * (1 - np.exp(-(t_ms[recover_mask] - trough_ms) / tau_ms))
    spike_ms = 400.0
    spike_idx = int(spike_ms / dt_ms)
    width = 20
    window = np.arange(-width, width)
    v[spike_idx - width:spike_idx + width] += 80.0 * np.exp(-0.5 * (window / 4.0) ** 2)

    found_trough, first_spike_ms = compute_pre_spike_sag_trough(v, t_ms, sag_window_ms=500.0)
    assert found_trough == pytest.approx(v[:int(spike_ms / dt_ms)].min(), abs=0.5)
    assert first_spike_ms is not None
    assert first_spike_ms == pytest.approx(spike_ms, abs=2.0)


def test_adaptation_ratio_recovers_known_slowdown():
    """First-3 ISIs at 20ms, last-3 ISIs at 60ms -- ratio must be exactly 3.0."""
    isis_ms = np.array([20.0, 20.0, 20.0, 30.0, 40.0, 50.0, 60.0, 60.0, 60.0])
    ratio = compute_adaptation_ratio(isis_ms, edge_n=3)
    assert ratio == pytest.approx(3.0, rel=1e-6)


def test_adaptation_ratio_none_below_minimum_isis():
    isis_ms = np.array([20.0, 25.0, 30.0])
    ratio = compute_adaptation_ratio(isis_ms, edge_n=3)
    assert ratio is None


def test_tight_low_ratio_alternation_rescued_by_ashman_d():
    """Real case, 2026-08-14: XB2IQX held=-2.94/inj=-3.03 recovery window,
    a clean, low-variance 39.35ms/51.6ms alternation -- ratio 1.31x, under
    the 1.5x min_isi_ratio bar, but Ashman's D=30.2 (real value, computed
    from this exact data). The ratio gate alone would reject this as
    "tonic"; ratio_override_ashman_d (default 5.0) must rescue it, since a
    tight, low-noise separation this decisive isn't the quantization-noise
    scenario the ratio gate exists to catch.
    """
    isis_ms = np.array([
        40.9, 52.0, 38.2, 51.8, 38.6, 51.5, 39.0, 51.3, 39.3, 51.2, 39.3, 51.2, 39.5, 51.1,
        39.6, 51.2, 39.5, 51.3, 39.6, 51.3, 39.5, 51.4, 39.5, 51.5, 39.5, 51.5, 39.5, 51.6,
        39.5, 51.7, 39.4, 51.8, 39.3, 51.9, 39.3, 51.9, 39.3, 52.0, 39.2, 52.1, 39.1, 52.2, 39.1])
    result = classify_burst_pattern(isis_ms, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    diag = result["diagnostics"]
    ratio = diag["isi_long_ms"] / diag["isi_short_ms"]
    assert ratio < 1.5, "this case should fail the plain ratio gate"
    assert diag["ashman_d"] > 20.0
    assert result["pattern"] == "bursting"


def test_low_ashman_d_ratio_failure_stays_tonic():
    """Non-regression companion to the D-override test above: a modest-
    ratio split with genuinely high variance (the existing high-variance
    synthetic case, D~1.75) must NOT be rescued just because it also fails
    the ratio gate -- the override is reserved for decisively-separated (D
    far above min_ashman_d) cases, not an escape hatch for every ratio
    failure.
    """
    isis_ms = _make_burst_train_with_long_tail(np.random.default_rng(11), n_bursts=60, tail_frac=0.1)
    result = classify_burst_pattern(isis_ms, n_peaks=len(isis_ms) + 1, **DEFAULT_KWARGS)
    diag = result["diagnostics"]
    assert diag["ashman_d"] < 5.0, "sanity: this case's D should be well under the override bar"
    assert result["pattern"] == "tonic"
