"""OFDMReceiver: demodulation, equalisation, shapes, validation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import Fiber, OFDMConfig, OFDMReceiver, OFDMTransmitter, Receiver
from hpcom.modulation import ConstantCompositionDM, ConstellationConfig, maxwell_boltzmann

PILOTS = (2, 10, 20, 26, 40, 50, 58)


def make_pair(
    *,
    n_subcarriers: int = 64,
    cp_len: int = 16,
    n_guard: int = 4,
    pilots: tuple[int, ...] = (),
    oversampling: int = 1,
    n_frames: int = 16,
    rx_kwargs: dict | None = None,
    **tx_kw,
) -> tuple[OFDMTransmitter, OFDMReceiver]:
    ofdm = OFDMConfig(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        n_guard=n_guard,
        pilot_subcarrier_indices=pilots,
        oversampling=oversampling,
    )
    tx_kw.setdefault("seed", 0)
    if "constellation" not in tx_kw:
        tx_kw.setdefault("modulation_order", 16)
    tx = OFDMTransmitter(subcarrier_spacing_hz=500e6, n_frames=n_frames, ofdm=ofdm, **tx_kw)
    rx_kw = {k: getattr(tx, k) for k in ("modulation_order", "constellation", "dm", "polarization")}
    rx = OFDMReceiver(
        ofdm=ofdm, **{k: v for k, v in rx_kw.items() if v is not None}, **(rx_kwargs or {})
    )
    return tx, rx


def apply_per_subcarrier_channel(
    tx: OFDMTransmitter, field: np.ndarray, h: np.ndarray
) -> np.ndarray:
    """Multiply every frame's L·N-point spectrum by ``h`` (a channel within the CP)."""
    cfg = tx.ofdm
    cp = cfg.cp_len * cfg.oversampling
    frames = field.reshape(*field.shape[:-1], tx.n_frames, cfg.samples_per_frame)[..., cp:]
    out = np.fft.ifft(np.fft.fft(frames, axis=-1) * h, axis=-1)
    return (
        np.concatenate([out[..., -cp:], out], axis=-1).reshape(field.shape)
        if cp
        else out.reshape(field.shape)
    )


# ---- structure and validation --------------------------------------------------------


def test_is_a_receiver_without_pulse_fields():
    _, rx = make_pair()
    assert isinstance(rx, Receiver)
    assert "samples_per_symbol" not in type(rx).model_fields


def test_output_carries_its_receiver():
    tx, rx = make_pair()
    sent = tx.generate()
    assert rx.receive(sent.field, sent.dt_s).receiver.ofdm == tx.ofdm


def test_time_domain_pilots_keyword_is_rejected():
    with pytest.raises(ValidationError, match="pilots"):
        OFDMReceiver(modulation_order=4, ofdm=OFDMConfig(n_subcarriers=16), pilots=None)


def test_pilot_equaliser_needs_pilot_subcarriers():
    with pytest.raises(ValidationError, match="pilot_subcarrier_indices"):
        OFDMReceiver(modulation_order=4, ofdm=OFDMConfig(n_subcarriers=16), equalize="pilot")


def test_field_length_must_be_whole_frames():
    tx, rx = make_pair()
    with pytest.raises(ValueError, match="not a multiple of 80 samples/frame"):
        rx.receive(np.zeros(81, complex), tx.dt_s)


@pytest.mark.parametrize(
    ("polarization", "shape"), [("single", (2, 160)), ("dual", (160,)), ("dual", (3, 160))]
)
def test_polarisation_shape_is_checked(polarization, shape):
    _, rx = make_pair(polarization=polarization)
    with pytest.raises(ValueError, match="expects"):
        rx.receive(np.zeros(shape, complex), 1e-11)


# ---- noiseless back-to-back ------------------------------------------------------


@pytest.mark.parametrize("oversampling", [1, 2])
@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_b2b_is_exact(order, oversampling):
    """Raw subcarrier values equal √P·symbol (absolute calibration); bits are error-free."""
    tx, rx = make_pair(modulation_order=order, oversampling=oversampling)
    sent = tx.generate()
    np.testing.assert_array_equal(rx.receive(sent.field, sent.dt_s).bits, sent.bits)
    raw = rx.replace(equalize="none").receive(sent.field, sent.dt_s).symbols
    np.testing.assert_allclose(
        raw / np.sqrt(tx.launch_power_per_polarization_w), sent.symbols, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(("cp_len", "n_guard"), [(0, 4), (16, 0), (0, 0)])
def test_b2b_without_prefix_or_guard(cp_len, n_guard):
    tx, rx = make_pair(cp_len=cp_len, n_guard=n_guard)
    sent = tx.generate()
    np.testing.assert_array_equal(rx.receive(sent.field, sent.dt_s).bits, sent.bits)


@pytest.mark.parametrize("equalize", ["unit_power", "pilot", "data_aided"])
def test_b2b_dual_polarisation(equalize):
    tx, rx = make_pair(polarization="dual", pilots=PILOTS, rx_kwargs={"equalize": equalize})
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s, reference=sent)
    np.testing.assert_array_equal(received.bits, sent.bits)
    assert received.symbols.shape == sent.symbols.shape == (2, 16 * tx.ofdm.n_data)
    assert received.gain.shape == (2,)


def test_b2b_with_distribution_matcher():
    base = ConstellationConfig.from_square_qam(16)
    pmf = maxwell_boltzmann(base.points, 0.2)
    cfg = base.with_probabilities(pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=56)
    tx, rx = make_pair(constellation=cfg, dm=dm, n_frames=8)
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    np.testing.assert_array_equal(received.bits, sent.bits)
    assert received.dm_block_valid.all()


def test_b2b_with_constellation_config():
    tx, rx = make_pair(constellation=ConstellationConfig.from_square_qam(16), n_frames=8)
    sent = tx.generate()
    np.testing.assert_array_equal(rx.receive(sent.field, sent.dt_s).bits, sent.bits)


# ---- channel equalisation --------------------------------------------------------


@pytest.mark.parametrize("oversampling", [1, 2])
def test_pilot_equaliser_removes_a_flat_complex_gain_exactly(oversampling):
    tx, rx = make_pair(pilots=PILOTS, oversampling=oversampling, rx_kwargs={"equalize": "pilot"})
    sent = tx.generate()
    received = rx.receive(sent.field * (0.3 * np.exp(1j * 2.0)), sent.dt_s)
    np.testing.assert_allclose(received.symbols, sent.symbols, rtol=0, atol=1e-12)


def test_pilot_equaliser_tracks_a_delay_across_dc():
    """A delay τ gives H(f) = e^{−2πifτ}, smooth across DC. Interpolating in
    frequency order keeps the error small everywhere — including the negative-
    frequency subcarriers just below DC, which index-order interpolation
    (the pre-v1-hardening behaviour) extrapolated from the far band edge."""
    tx, rx = make_pair(
        pilots=(0, 8, 16, 24, 40, 48, 56), modulation_order=4, rx_kwargs={"equalize": "pilot"}
    )
    cfg = tx.ofdm
    f = np.fft.fftfreq(cfg.fft_size, tx.dt_s)
    tau = 0.05 / tx.bandwidth_hz
    sent = tx.generate()
    received = rx.receive(
        apply_per_subcarrier_channel(tx, sent.field, np.exp(-2j * np.pi * f * tau)), sent.dt_s
    )
    err = np.abs(received.symbols - sent.symbols).reshape(tx.n_frames, -1).max(axis=0)
    assert err.max() < 0.02, f"worst subcarrier error {err.max():.3f}"
    np.testing.assert_array_equal(received.bits, sent.bits)


def test_channel_interpolation_is_exact_for_a_channel_linear_in_frequency():
    _, rx = make_pair(pilots=(0, 8, 16, 24, 40, 48, 56))
    cfg = rx.ofdm
    f_pil = cfg.subcarrier_frequency_index(cfg.pilot_subcarrier_indices).astype(float)
    f_dat = cfg.subcarrier_frequency_index(cfg.data_subcarrier_indices()).astype(float)
    h_pil = (1 + 0.01 * f_pil) * np.exp(1j * 0.3) + 0j
    h_dat = rx._interpolate_channel(h_pil[None, None, :])[0, 0]
    inside = (f_dat >= f_pil.min()) & (f_dat <= f_pil.max())
    np.testing.assert_allclose(
        h_dat[inside], ((1 + 0.01 * f_dat) * np.exp(1j * 0.3))[inside], atol=1e-14
    )


def test_cd_compensation_restores_an_ofdm_signal_beyond_its_prefix():
    """80 km of SSMF spreads ~10× longer than the CP; receiver CD compensation undoes it."""
    tx, rx0 = make_pair(cp_len=4, oversampling=2, n_frames=32)
    fiber = Fiber.ssmf(80.0, backend="numpy-linear").replace(alpha_db_per_km=0.0)
    rx = rx0.replace(cd_compensation_s2=fiber.accumulated_dispersion_s2, equalize="data_aided")
    sent = tx.generate()
    out = fiber.propagate(sent.field, sent.dt_s)
    assert np.mean(rx0.receive(out, sent.dt_s).bits != sent.bits) > 0.05
    received = rx.receive(out, sent.dt_s, reference=sent)
    np.testing.assert_allclose(received.symbols, sent.symbols, rtol=0, atol=1e-9)


def test_without_equalisation_a_frequency_selective_channel_causes_errors():
    tx, rx = make_pair(modulation_order=4)
    h = 0.8 * np.exp(1j * 0.05 * np.arange(tx.ofdm.fft_size))
    sent = tx.generate()
    received = rx.receive(apply_per_subcarrier_channel(tx, sent.field, h), sent.dt_s)
    assert np.mean(received.bits != sent.bits) > 0.0


# ---- shapes and reproducibility --------------------------------------------------


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_output_shapes(polarization):
    tx, rx = make_pair(
        pilots=PILOTS, polarization=polarization, n_frames=8, rx_kwargs={"equalize": "pilot"}
    )
    sent = tx.generate()
    r = rx.receive(sent.field, sent.dt_s)
    lead = (2,) if polarization == "dual" else ()
    n_data = tx.ofdm.n_data
    assert r.symbols.shape == r.decisions.shape == (*lead, 8 * n_data)
    assert r.bits.shape == (*lead, 8 * n_data * 4)
    assert r.pilot_symbols.shape == (*lead, 8 * len(PILOTS))
    assert r.polarization == polarization


def test_same_seed_same_result():
    (tx1, rx1), (tx2, rx2) = make_pair(seed=77), make_pair(seed=77)
    s1, s2 = tx1.generate(), tx2.generate()
    np.testing.assert_array_equal(
        rx1.receive(s1.field, s1.dt_s).bits, rx2.receive(s2.field, s2.dt_s).bits
    )
