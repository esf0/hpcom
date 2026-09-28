"""WDM composite spectrum: channel placement, band limits, cross-channel isolation."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import PULSE_MODES, trim, wdm_pair


def _psd(sig):
    return np.abs(np.fft.fft(sig.field)) ** 2, np.fft.fftfreq(sig.field.shape[-1], sig.dt_s)


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_energy_sits_on_the_channel_grid(pulse):
    tx, _ = wdm_pair(modulation_order=16, n_symbols=2048, samples_per_symbol=16, seed=0, **pulse)
    psd, f = _psd(tx.generate())
    rs, spacing = tx.symbol_rate_hz, tx.wdm.spacing_hz
    for fc in tx.wdm.channel_frequencies_hz():
        in_band = psd[np.abs(f - fc) <= rs / 2].sum()
        guard = psd[np.abs(f - (fc + spacing / 2)) <= (spacing - rs * (1 + tx.roll_off)) / 2].sum()
        assert in_band > 1e3 * guard
    outer = abs(tx.wdm.channel_frequencies_hz()[0]) + rs * (1 + tx.roll_off) / 2
    assert psd[np.abs(f) > outer * 1.01].sum() / psd.sum() < 1e-3


def test_exact_rrc_channels_have_no_energy_outside_their_band():
    tx, _ = wdm_pair(modulation_order=16, n_symbols=2048, samples_per_symbol=16, seed=0)
    psd, f = _psd(tx.generate())
    half_band = tx.symbol_rate_hz * (1 + tx.roll_off) / 2 * (1 + 1e-9)
    in_some_band = np.zeros_like(f, dtype=bool)
    for fc in tx.wdm.channel_frequencies_hz():
        in_some_band |= np.abs(f - fc) <= half_band
    assert psd[~in_some_band].max() < 1e-24 * psd.max()


def test_constant_symbols_put_a_tone_exactly_on_each_carrier_bin():
    tx, _ = wdm_pair(modulation_order=4, n_symbols=512, samples_per_symbol=16)
    tones = np.array([1 + 0j, 1j, -1 + 0j])
    sig = tx.generate(symbols=np.tile(tones[:, None], (1, 512)))
    psd, f = _psd(sig)
    peaks = np.sort(f[np.argsort(psd)[-3:]])
    np.testing.assert_allclose(peaks, tx.wdm.channel_frequencies_hz(), rtol=0, atol=1e-3)
    # Circular + on-grid carriers: all energy is in exactly three bins.
    assert np.sort(psd)[-4] < 1e-20 * psd.max()


@pytest.mark.parametrize(
    ("pulse", "min_rejection_db"),
    [
        pytest.param({}, 250.0, id="exact-circular"),
        pytest.param({"pulse_span_symbols": 32}, 30.0, id="fir-circular"),
        pytest.param({"pulse_span_symbols": 32, "convolution": "linear"}, 30.0, id="fir-linear"),
    ],
)
@pytest.mark.parametrize("demux", [0, 1, 2])
def test_demultiplexer_rejects_neighbouring_channels(demux, pulse, min_rejection_db):
    """Only a neighbour is modulated; the demuxed channel must see (almost) nothing.

    The exact RRC is strictly band-limited, so with 50 GHz spacing ≥ R_s(1+β)
    the rejection is limited only by rounding; the truncated FIR leaks.
    """
    tx, rx = wdm_pair(
        modulation_order=4,
        n_symbols=512,
        samples_per_symbol=16,
        rx_kwargs={"equalize": "none"},
        **pulse,
    )
    skip = tx.transient_symbols

    def response(driven: int) -> float:
        syms = np.zeros((3, 512), dtype=complex)
        syms[driven] = 1.0
        sig = tx.generate(symbols=syms)
        out = rx.receive(sig.field, sig.dt_s).symbols[demux]
        return float(np.mean(np.abs(trim(out, 1, skip)) ** 2))

    own = response(demux)
    for neighbour in {0, 1, 2} - {demux}:
        rejection = 10 * np.log10(own / max(response(neighbour), 1e-300))
        assert rejection > min_rejection_db, f"neighbour {neighbour}: {rejection:.1f} dB"


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_per_channel_bit_patterns_survive_mux_and_demux(pulse):
    tx, rx = wdm_pair(modulation_order=4, n_symbols=2048, samples_per_symbol=16, **pulse)
    bits = np.random.default_rng(123).integers(0, 2, (3, tx.n_bits_per_stream), dtype=np.uint8)
    sent = tx.generate(bits=bits)
    received = rx.receive(sent.field, sent.dt_s)
    skip = tx.transient_symbols
    np.testing.assert_array_equal(trim(received.bits, 2, skip), trim(bits, 2, skip))
