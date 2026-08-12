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
