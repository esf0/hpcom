"""Numerical-physics value tests (TDD).

"Set X, measure X, verify X" invariants — every test configures a specific
physical quantity, computes the corresponding observable directly from the
output, and asserts equality (to numerical precision or a documented tolerance).
These are the load-bearing physics tests; if they fail, downstream simulations
silently produce wrong numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.amplifier import EDFA
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.receiver import SingleCarrierReceiver
from hpcom.simulation import Simulation
from hpcom.transmitter import SingleCarrierTransmitter
from hpcom.wdm import WDMConfig, WDMTransmitter


# ---------------------------------------------------------------------------
# Launch power
# ---------------------------------------------------------------------------
def _passthrough_link() -> Link:
    """Lossless 1-span fiber + unity-gain amplifier — exactly preserves power."""
    fiber = Fiber(
        length_km=1.0,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=0.0,
        gamma_per_w_per_km=0.0,
        n_steps=1,
        backend="numpy-linear",
    )
    amp = EDFA(gain_db=0.0, noise_figure_db=0.0)
    return Link.from_spans(n_spans=1, fiber=fiber, amplifier=amp, strict=True)


@pytest.mark.parametrize("dbm", [-10.0, -3.0, 0.0, 3.0])
def test_simulation_launch_power_dbm_yields_target_power(dbm: float):
    """`launch_power_dbm=X` → received_field mean power = 10^(X/10) mW exactly."""
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=4096,
        samples_per_symbol=8, pulse_span_symbols=32, seed=0,
    )
    rx = SingleCarrierReceiver(
        modulation_order=16, samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off, pulse_span_symbols=tx.pulse_span_symbols,
    )
    sim = Simulation(
        transmitter=tx, link=_passthrough_link(), receiver=rx,
        launch_power_dbm=dbm,
    )
    result = sim.run()
    measured_w = float(np.mean(np.abs(result.received_field) ** 2))
    target_w = 10.0 ** (dbm / 10.0) * 1e-3
    np.testing.assert_allclose(measured_w, target_w, rtol=1e-12)


def test_launch_power_dbm_none_leaves_field_unscaled():
    """Default behaviour: no rescale; field power = constellation unit / sps."""
    tx = SingleCarrierTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=4096,
        samples_per_symbol=8, pulse_span_symbols=32, seed=0,
    )
    rx = SingleCarrierReceiver(
        modulation_order=4, samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off, pulse_span_symbols=tx.pulse_span_symbols,
    )
    sim = Simulation(transmitter=tx, link=_passthrough_link(), receiver=rx)
    result = sim.run()
    expected = 1.0 / tx.samples_per_symbol  # unit-power constellation / sps
    measured = float(np.mean(np.abs(result.received_field) ** 2))
    np.testing.assert_allclose(measured, expected, rtol=0.05)


# ---------------------------------------------------------------------------
# Fiber loss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,length", [(0.2, 80.0), (0.3, 50.0), (1.0, 10.0)])
def test_fiber_loss_matches_beer_lambert(alpha: float, length: float):
    """Pure-loss fiber: output mean power = input × 10^(-α·L/10) exactly."""
    fiber = Fiber(
        length_km=length, alpha_db_per_km=alpha,
        beta2_s2_per_m=0.0, gamma_per_w_per_km=0.0,
        n_steps=10, backend="numpy-linear",
    )
    rng = np.random.default_rng(0)
    n = 2048
    field = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    p_in = float(np.mean(np.abs(field) ** 2))
    out = fiber.propagate(field, dt_s=1e-11)
    p_out = float(np.mean(np.abs(out) ** 2))
    expected = p_in * 10.0 ** (-alpha * length / 10.0)
    np.testing.assert_allclose(p_out, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# EDFA gain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gain_db", [0.0, 3.0, 10.0, 20.0])
def test_edfa_zero_nf_gain_exact_in_db(gain_db: float):
    """EDFA with NF=0 dB → output power = input × 10^(G/10) (to ASE-leak precision).

    The noise figure floor in dB is 0, which corresponds to NF_linear = 1
    (not 0). ASE PSD is then ``h·ν·(G-1)·1``, which is small but nonzero;
    the test uses a large enough input power that the ASE contribution is
    well below 1e-4 of the deterministic gain.
    """
    edfa = EDFA(gain_db=gain_db, noise_figure_db=0.0)
    rng = np.random.default_rng(1)
    n = 4096
    sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    p_in = float(np.mean(np.abs(sig) ** 2))
    out = edfa.amplify(sig, dt_s=1e-11, rng=np.random.default_rng(2))
    p_out = float(np.mean(np.abs(out) ** 2))
    expected = p_in * 10.0 ** (gain_db / 10.0)
    # ASE leak at NF_lin=1 contributes ~h·ν·(G-1)·fs ≈ 1e-7 W per sample;
    # relative to the unit-power signal that's ~1e-7. The sample-mean
    # statistical noise dominates: ~sqrt(2·gain·ASE/n) ≈ a few ×1e-7.
    np.testing.assert_allclose(p_out, expected, rtol=5e-6)


def test_edfa_deterministic_amplitude_when_seed_pinned():
    """A tighter gain test: subtract the noise contribution by running the
    EDFA twice with the SAME rng (once on the signal, once on zero), then
    verify the deterministic part exactly matches sqrt(G) · signal."""
    edfa = EDFA(gain_db=10.0, noise_figure_db=5.0)
    rng_signal = np.random.default_rng(42)
    rng_zero = np.random.default_rng(42)
    n = 2048
    sig = (np.random.default_rng(0).standard_normal(n)
           + 1j * np.random.default_rng(1).standard_normal(n)) / np.sqrt(2)
    out_signal = edfa.amplify(sig, dt_s=1e-11, rng=rng_signal)
    out_noise = edfa.amplify(
        np.zeros_like(sig), dt_s=1e-11, rng=rng_zero,
    )
    deterministic = out_signal - out_noise
    expected_deterministic = np.sqrt(10.0 ** (edfa.gain_db / 10.0)) * sig
    np.testing.assert_allclose(deterministic, expected_deterministic, rtol=1e-12)


# ---------------------------------------------------------------------------
# WDM carrier frequencies
# ---------------------------------------------------------------------------
def test_wdm_channel_centers_match_grid_to_subbin_precision():
    """Drive each channel with a single-tone constant symbol → the spectral peak
    sits at the configured frequency to better than half an FFT bin."""
    cfg = WDMConfig(n_channels=5, spacing_hz=37.5e9)
    # 5 channels × 37.5 GHz spacing → outer center ±75 GHz; with Rs=10 GHz
    # and roll-off 0.1, need fs > 2·(75e9 + 5.5e9) = 161 GHz → sps ≥ 17.
    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=4096,
        samples_per_symbol=32, pulse_span_symbols=32, wdm=cfg,
    )
    # Constant 1+0j on every channel → each is a pure tone at f_i.
    symbols = np.ones((cfg.n_channels, tx.n_symbols), dtype=np.complex128)
    sent = tx.generate(symbols=symbols)
    fs = 1.0 / sent.dt_s
    n = sent.field.size
    df = fs / n
    spectrum = np.fft.fftshift(np.fft.fft(sent.field))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=sent.dt_s))
    psd = np.abs(spectrum) ** 2

    for f_expected in cfg.channel_frequencies_hz():
        # Quadratic peak interpolation: with bins x_{-1}, x_0, x_{+1} around
        # the peak, sub-bin offset δ ≈ (x_{+1} - x_{-1}) / (2(2 x_0 - x_{-1} - x_{+1}))
        window = np.abs(freqs - f_expected) <= 2 * tx.symbol_rate_hz
        widx = np.flatnonzero(window)
        local = psd[widx]
        peak = int(np.argmax(local))
        if 0 < peak < local.size - 1:
            ym1, y0, yp1 = local[peak - 1 : peak + 2]
            delta = (yp1 - ym1) / (2.0 * (2.0 * y0 - ym1 - yp1))
        else:
            delta = 0.0
        peak_freq = freqs[widx[peak]] + delta * df
        # Sub-bin tolerance: 0.1 bin is plenty (quadratic-interp gives much
        # better in practice).
        assert abs(peak_freq - f_expected) < 0.1 * df, (
            f"channel center landed at {peak_freq:.4e}, expected {f_expected:.4e} "
            f"(error {abs(peak_freq - f_expected):.2e} Hz, bin = {df:.2e})"
        )


# ---------------------------------------------------------------------------
# Polarisation power split
# ---------------------------------------------------------------------------
def test_dual_pol_total_power_equals_sum_of_per_pol_powers():
    """Trivial sanity: |E_x|² + |E_y|² across all samples = total intensity."""
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=4096,
        samples_per_symbol=8, pulse_span_symbols=32, seed=0,
        polarization="dual",
    )
    sent = tx.generate()
    p_x = float(np.mean(np.abs(sent.field[0]) ** 2))
    p_y = float(np.mean(np.abs(sent.field[1]) ** 2))
    p_total_samples = float(np.mean(np.abs(sent.field) ** 2))
    # Mean over both polarisations equals average of per-pol means.
    np.testing.assert_allclose(p_total_samples, 0.5 * (p_x + p_y), rtol=1e-12)


def test_dual_pol_per_pol_power_equal_for_uniform_qam():
    """Identical bit-rate / constellation on X and Y → equal average power."""
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=16384,
        samples_per_symbol=4, pulse_span_symbols=32, seed=0,
        polarization="dual",
    )
    sent = tx.generate()
    p_x = float(np.mean(np.abs(sent.field[0]) ** 2))
    p_y = float(np.mean(np.abs(sent.field[1]) ** 2))
    # Stat tolerance: for n_symbols=16384, fractional std of |s|² ~ 1/√16384 ~ 1%.
    np.testing.assert_allclose(p_x, p_y, rtol=0.02)


# ---------------------------------------------------------------------------
# WDM channel orthogonality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_ch", [2, 3, 5])
def test_wdm_composite_power_equals_n_times_single_channel(n_ch: int):
    """Channels are orthogonal in frequency: cross-terms time-average to 0
    over many symbols. Total composite power = N × per-channel baseband power.
    """
    cfg = WDMConfig(n_channels=n_ch, spacing_hz=50e9)
    # sps must satisfy fs > 2·((n_ch-1)/2·spacing + Rs·(1+ro)/2). For n_ch=5
    # and 50 GHz spacing that's 211 GHz → sps ≥ 22 at Rs=10 GHz. Use 32.
    tx = WDMTransmitter(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=4096,
        samples_per_symbol=32, pulse_span_symbols=32, seed=0, wdm=cfg,
    )
    sent = tx.generate()
    composite_p = float(np.mean(np.abs(sent.field) ** 2))

    # Reference: same-config single-channel Tx (no WDM).
    single = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=tx.n_symbols,
        samples_per_symbol=tx.samples_per_symbol,
        pulse_span_symbols=tx.pulse_span_symbols, seed=0,
    )
    single_p = float(np.mean(np.abs(single.generate().field) ** 2))
    np.testing.assert_allclose(composite_p, n_ch * single_p, rtol=0.02)


# ---------------------------------------------------------------------------
# CD compensation
# ---------------------------------------------------------------------------
def test_cd_compensation_undoes_linear_fiber_exactly():
    """SingleCarrierReceiver(cd_compensation_s2=β₂·L) applied after a lossless gamma=0 fiber
    must recover the input pulse to numerical precision."""
    rng = np.random.default_rng(0)
    n = 2048
    dt_s = 1e-11
    sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    fiber = Fiber(
        length_km=80.0, alpha_db_per_km=0.0,
        beta2_s2_per_m=-2.16e-26, gamma_per_w_per_km=0.0,
        n_steps=100, backend="numpy-linear",
    )
    out = fiber.propagate(sig, dt_s=dt_s)

    rx = SingleCarrierReceiver(
        modulation_order=4, samples_per_symbol=2,
        cd_compensation_s2=fiber.beta2_s2_per_m * fiber.length_km * 1e3,
    )
    # Use private _undo_dispersion directly — receive() does matched filtering
    # we don't want here.
    recovered = rx._undo_dispersion(out, dt_s)
    np.testing.assert_allclose(recovered, sig, atol=1e-10)
