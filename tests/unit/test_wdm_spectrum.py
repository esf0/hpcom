"""WDM spectrum & isolation (Test D): does the comb actually look like a comb?

Spectral checks for WDM Tx output and Rx demux:

- channel i's energy is centered at f_i with the expected per-channel bandwidth
- a single-tone-per-channel (constant symbols) lands within one FFT bin of f_i
- after demuxing channel i, the adjacent channels are suppressed by a large
  margin (matched-filter selectivity)
- per-channel deterministic-symbol round-trip survives the WDM mux/demux
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.transmitter import SingleCarrierTransmitter
from hpcom.wdm import WDMConfig, WDMReceiver, WDMTransmitter


def _wdm_tx(*, n_channels: int = 3, **overrides: object) -> WDMTransmitter:
    """Reasonable defaults: 3 channels at 50 GHz, Rs=10 GHz, sps high enough."""
    base = dict(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=2048,
        samples_per_symbol=16,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=0,
        wdm=WDMConfig(n_channels=n_channels, spacing_hz=50e9),
    )
    base.update(overrides)
    return WDMTransmitter(**base)  # type: ignore[arg-type]


def test_wdm_spectrum_peaks_at_expected_channel_frequencies():
    """Average |FFT|² of the composite WDM field shows clear peaks at each f_i."""
    tx = _wdm_tx()
    sent = tx.generate()
    fs = 1.0 / sent.dt_s
    n = sent.field.size
    spectrum = np.fft.fftshift(np.fft.fft(sent.field))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=sent.dt_s))
    psd = np.abs(spectrum) ** 2

    assert tx.wdm is not None
    expected_centers = tx.wdm.channel_frequencies_hz()
    rs = tx.symbol_rate_hz
    for f_expected in expected_centers:
        # Energy in the ±Rs/2 window around the channel center must dominate
        # the energy in a same-width window equidistant *between* two channels
        # (a "guard band").
        in_band = psd[np.abs(freqs - f_expected) <= rs / 2].sum()
        guard_band = psd[np.abs(freqs - (f_expected + tx.wdm.spacing_hz / 2)) <= rs / 2].sum()
        assert in_band > 50 * guard_band, (
            f"channel at {f_expected:.2e} Hz: in-band {in_band:.3e} vs "
            f"guard {guard_band:.3e} (ratio {in_band / guard_band:.1f}, "
            f"expected >50)"
        )
    # And no significant energy outside the outer channel + Rs/2 margin.
    outer = abs(float(expected_centers[-1])) + rs
    out_of_band = psd[np.abs(freqs) > outer].sum()
    total = psd.sum()
    assert out_of_band / total < 0.02, (
        f"{out_of_band / total:.3%} of energy outside [-{outer:.2e}, +{outer:.2e}] Hz "
        f"(fs={fs:.2e} Hz)"
    )


def test_single_tone_per_channel_lands_at_correct_frequency():
    """Drive each channel with a constant symbol → that channel becomes a CW
    tone at exactly f_i in the composite field."""
    n_ch = 3
    cfg = WDMConfig(n_channels=n_ch, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=16,
        pulse_span_symbols=32,
        wdm=cfg,
    )
    # Constant symbols per channel: ch0 = 1+0j, ch1 = 0+1j, ch2 = -1+0j.
    constants = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128)
    symbols = np.tile(constants[:, None], (1, tx.n_symbols))
    sent = tx.generate(symbols=symbols)
    fs = 1.0 / sent.dt_s
    n = sent.field.size
    spectrum = np.fft.fftshift(np.fft.fft(sent.field))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=sent.dt_s))
    psd = np.abs(spectrum) ** 2

    # Pick the largest peak in a narrow window around each expected f_i and
    # verify it sits within ±2 FFT bins of f_i.
    df = fs / n
    for f_expected in cfg.channel_frequencies_hz():
        window = np.abs(freqs - f_expected) <= 2 * tx.symbol_rate_hz
        peak_idx_local = int(np.argmax(psd[window]))
        peak_freq = freqs[window][peak_idx_local]
        assert abs(peak_freq - f_expected) <= 2 * df, (
            f"channel peak landed at {peak_freq:.3e} Hz, expected {f_expected:.3e} "
            f"(within ±2 bins = {2 * df:.2e} Hz)"
        )


@pytest.mark.parametrize("ch_demux", [0, 1, 2])
def test_demux_channel_suppresses_neighbours_by_at_least_30_db(ch_demux: int):
    """Drive *only* a neighbour channel with a CW tone; demux a different
    channel. The post-matched-filter output should be ~30 dB below the
    same channel's response when driven with the same tone."""
    n_ch = 3
    cfg = WDMConfig(n_channels=n_ch, spacing_hz=50e9)
    base_kwargs = dict(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=16,
        pulse_span_symbols=32,
        wdm=cfg,
    )
    tx = WDMTransmitter(**base_kwargs)  # type: ignore[arg-type]

    # Reference: drive only the demux'd channel with the tone.
    ref_symbols = np.zeros((n_ch, tx.n_symbols), dtype=np.complex128)
    ref_symbols[ch_demux] = 1.0 + 0.0j
    sent_ref = tx.generate(symbols=ref_symbols)
    rx = WDMReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        wdm=cfg,
        wdm_channel=ch_demux,
    )
    rx_ref = rx.receive(sent_ref.field, dt_s=sent_ref.dt_s)
    skip = tx.pulse_span_symbols
    ref_power = float(np.mean(np.abs(rx_ref.symbols[skip:-skip]) ** 2))

    # Cross-channel: drive *only* a neighbour, demux ch_demux → should be tiny.
    for neighbour in range(n_ch):
        if neighbour == ch_demux:
            continue
        cross_symbols = np.zeros((n_ch, tx.n_symbols), dtype=np.complex128)
        cross_symbols[neighbour] = 1.0 + 0.0j
        sent_cross = WDMTransmitter(**base_kwargs).generate(symbols=cross_symbols)  # type: ignore[arg-type]
        rx_cross = rx.receive(sent_cross.field, dt_s=sent_cross.dt_s)
        cross_power = float(np.mean(np.abs(rx_cross.symbols[skip:-skip]) ** 2))
        rejection_db = 10.0 * np.log10(ref_power / max(cross_power, 1e-30))
        assert rejection_db > 30, (
            f"demux ch={ch_demux} sees neighbour ch={neighbour} at only "
            f"{rejection_db:.1f} dB below the same-channel response (need >30 dB)"
        )


def test_per_channel_deterministic_bits_round_trip_in_wdm():
    """Feed channel-specific bit patterns into all channels of a WDM Tx, then
    demux each channel at the Rx and verify exact bit-level recovery on the
    steady-state region. Catches per-channel cross-talk and seed leakage."""
    n_ch = 3
    cfg = WDMConfig(n_channels=n_ch, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=2048,
        samples_per_symbol=16,
        roll_off=0.1,
        pulse_span_symbols=32,
        wdm=cfg,
    )
    k = 2
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=(n_ch, tx.n_symbols * k), dtype=np.uint8)
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)

    for ch in range(n_ch):
        rx = WDMReceiver(
            modulation_order=4,
            samples_per_symbol=tx.samples_per_symbol,
            roll_off=tx.roll_off,
            pulse_span_symbols=tx.pulse_span_symbols,
            wdm=cfg,
            wdm_channel=ch,
        )
        received = rx.receive(sent.field, dt_s=sent.dt_s)
        skip = tx.pulse_span_symbols
        sent_bits = bits[ch].reshape(-1, k)[skip:-skip]
        rx_bits = received.bits.reshape(-1, k)[skip:-skip]
        n_err = int(np.sum(sent_bits != rx_bits))
        assert n_err == 0, f"ch={ch}: {n_err} bit errors after WDM round-trip"


def test_wdm_tx_total_power_equals_sum_of_per_channel_powers():
    """Composite power should equal the sum of per-channel baseband powers
    (channels are orthogonal in frequency, so cross terms time-average to 0)."""
    n_ch = 3
    cfg = WDMConfig(n_channels=n_ch, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=4096,
        samples_per_symbol=16,
        pulse_span_symbols=32,
        seed=0,
        wdm=cfg,
    )
    sent = tx.generate()
    composite_power = float(np.mean(np.abs(sent.field) ** 2))
    # Expected: n_channels times per-channel mean power, since channels are
    # orthogonal in frequency and the composite is their sum.
    single_tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=tx.symbol_rate_hz,
        n_symbols=tx.n_symbols,
        samples_per_symbol=tx.samples_per_symbol,
        pulse_span_symbols=tx.pulse_span_symbols,
        seed=0,
    )
    per_ch_power = float(np.mean(np.abs(single_tx.generate().field) ** 2))
    np.testing.assert_allclose(composite_power, n_ch * per_ch_power, rtol=0.05)
