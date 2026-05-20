"""T11.3: geometric-shaping factories (APSK).

Verifies the apsk(...) factory produces valid ConstellationConfig instances
and that the resulting constellations round-trip through Transmitter/Receiver
noiselessly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import ConstellationConfig, apsk
from hpcom.receiver import SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_apsk_size_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        apsk(ring_sizes=(4, 8, 3), ring_radii=(1.0, 2.0, 3.0))


def test_apsk_mismatched_ring_lengths():
    with pytest.raises(ValueError, match="same length"):
        apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.0, 3.0))


def test_apsk_rejects_zero_radius():
    with pytest.raises(ValueError, match="ring_radii"):
        apsk(ring_sizes=(4, 12), ring_radii=(0.0, 2.0))


def test_apsk_phase_length_mismatch():
    with pytest.raises(ValueError, match="ring_phases"):
        apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.0), ring_phases=(0.0,))


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ring_sizes,ring_radii,expected_m", [
    ((4, 12), (1.0, 2.85), 16),  # DVB-S2 16-APSK
    ((4, 12, 16), (1.0, 2.6, 4.3), 32),  # 32-APSK
    ((1, 7), (0.0 + 1e-12, 1.0), 8),  # degenerate inner 1-PSK + 7-PSK ring (still 8)
])
def test_apsk_total_size(ring_sizes, ring_radii, expected_m):
    cfg = apsk(ring_sizes=ring_sizes, ring_radii=ring_radii)
    assert isinstance(cfg, ConstellationConfig)
    assert cfg.modulation_order == expected_m


def test_apsk_unit_expected_power():
    cfg = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    power = float(np.mean(np.abs(cfg.points) ** 2))
    assert power == pytest.approx(1.0, abs=1e-12)


def test_apsk_inner_ring_points_have_smaller_magnitude():
    """Layout invariant: ring 0 comes first in `points`, ring 1 after."""
    cfg = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    inner = cfg.points[:4]
    outer = cfg.points[4:]
    # All inner points equidistant from origin; same for outer.
    assert np.allclose(np.abs(inner), np.abs(inner[0]))
    assert np.allclose(np.abs(outer), np.abs(outer[0]))
    assert np.abs(inner[0]) < np.abs(outer[0])


def test_apsk_ring_phase_rotates_points():
    cfg_a = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    cfg_b = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85),
                 ring_phases=(np.pi / 4, 0.0))
    # Inner ring of cfg_b is cfg_a's inner ring rotated by π/4.
    rot = np.exp(1j * np.pi / 4)
    inner_a = cfg_a.points[:4]
    inner_b = cfg_b.points[:4]
    # Both should be normalised by the same total-power factor (same radii).
    np.testing.assert_allclose(inner_b, inner_a * rot, atol=1e-12)


def test_apsk_minimum_distance_positive():
    cfg = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    diffs = cfg.points[:, None] - cfg.points[None, :]
    dists = np.abs(diffs)
    np.fill_diagonal(dists, np.inf)
    assert dists.min() > 0.1  # well-separated for unit-power 16-APSK


# ---------------------------------------------------------------------------
# B2B round-trip
# ---------------------------------------------------------------------------

def test_apsk_16_b2b_noiseless_recovers_bits():
    cfg = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    span = 16
    tx = SingleCarrierTransmitter(constellation=cfg, symbol_rate_hz=32e9, n_symbols=1024,
                     samples_per_symbol=8, pulse_span_symbols=span, seed=17)
    rx = SingleCarrierReceiver(constellation=cfg, samples_per_symbol=8, pulse_span_symbols=span)

    sent = tx.generate()
    received = rx.receive(sent.field)

    k = cfg.bits_per_symbol
    sent_trim = sent.bits.reshape(-1, k)[span:-span].ravel()
    rx_trim = received.bits.reshape(-1, k)[span:-span].ravel()
    np.testing.assert_array_equal(rx_trim, sent_trim)


def test_apsk_32_b2b_noiseless_recovers_bits():
    cfg = apsk(ring_sizes=(4, 12, 16), ring_radii=(1.0, 2.6, 4.3))
    span = 16
    tx = SingleCarrierTransmitter(constellation=cfg, symbol_rate_hz=32e9, n_symbols=2048,
                     samples_per_symbol=8, pulse_span_symbols=span, seed=19)
    rx = SingleCarrierReceiver(constellation=cfg, samples_per_symbol=8, pulse_span_symbols=span)

    sent = tx.generate()
    received = rx.receive(sent.field)

    k = cfg.bits_per_symbol
    sent_trim = sent.bits.reshape(-1, k)[span:-span].ravel()
    rx_trim = received.bits.reshape(-1, k)[span:-span].ravel()
    np.testing.assert_array_equal(rx_trim, sent_trim)


# ---------------------------------------------------------------------------
# Single-ring (pure PSK)
# ---------------------------------------------------------------------------

def test_apsk_single_ring_is_psk():
    """apsk with one ring is equivalent to M-PSK with equi-spaced points."""
    cfg = apsk(ring_sizes=(8,), ring_radii=(1.0,))
    assert cfg.modulation_order == 8
    # All points equidistant from origin (unit-power 8-PSK has |s|=1).
    np.testing.assert_allclose(np.abs(cfg.points), 1.0, atol=1e-12)
    # Consecutive points are equally spaced in angle.
    angles = np.angle(cfg.points)
    diffs = np.diff(np.sort(angles))
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-12)


def test_apsk_single_ring_b2b_noiseless():
    """Single-ring 8-PSK B2B round-trip."""
    cfg = apsk(ring_sizes=(8,), ring_radii=(1.0,))
    span = 8
    tx = SingleCarrierTransmitter(constellation=cfg, symbol_rate_hz=32e9, n_symbols=512,
                     samples_per_symbol=8, pulse_span_symbols=span, seed=21)
    rx = SingleCarrierReceiver(constellation=cfg, samples_per_symbol=8, pulse_span_symbols=span)

    sent = tx.generate()
    received = rx.receive(sent.field)

    k = cfg.bits_per_symbol
    sent_trim = sent.bits.reshape(-1, k)[span:-span].ravel()
    rx_trim = received.bits.reshape(-1, k)[span:-span].ravel()
    np.testing.assert_array_equal(rx_trim, sent_trim)


# ---------------------------------------------------------------------------
# Bit-labeling contract
# ---------------------------------------------------------------------------

def test_apsk_default_labels_are_binary_indexed():
    """Default labels are MSB-first binary (index i → binary(i)).

    This is NOT Gray coding. BER experiments with APSK require custom Gray
    labels passed via ConstellationConfig.from_points for optimal performance.
    The test documents the actual contract so callers are not surprised.
    """
    cfg = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    m = cfg.modulation_order
    k = cfg.bits_per_symbol
    expected = np.zeros((m, k), dtype=np.uint8)
    for i in range(m):
        for b in range(k):
            expected[i, k - 1 - b] = (i >> b) & 1
    np.testing.assert_array_equal(cfg.bit_labels, expected)
