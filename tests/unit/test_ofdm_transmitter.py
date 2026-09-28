"""OFDMTransmitter: structure, spectrum, power normalisation, deterministic input."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import OFDMConfig, OFDMTransmitter, PulseShapedTransmitter, Transmitter
from hpcom.modulation import ConstantCompositionDM, ConstellationConfig, maxwell_boltzmann
from hpcom.units import dbm_to_w

OVERSAMPLING = [pytest.param(1, id="L1"), pytest.param(2, id="L2"), pytest.param(4, id="L4")]


def make_tx(
    *,
    n_subcarriers: int = 64,
    cp_len: int = 16,
    n_guard: int = 4,
    pilots: tuple[int, ...] = (),
    oversampling: int = 1,
    n_frames: int = 16,
    **kw,
) -> OFDMTransmitter:
    ofdm = OFDMConfig(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        n_guard=n_guard,
        pilot_subcarrier_indices=pilots,
        oversampling=oversampling,
    )
    kw.setdefault("modulation_order", 4)
    kw.setdefault("seed", 0)
    return OFDMTransmitter(subcarrier_spacing_hz=500e6, n_frames=n_frames, ofdm=ofdm, **kw)


def frames_spectrum(tx: OFDMTransmitter, field: np.ndarray) -> np.ndarray:
    """(…, n_frames, L·N) spectrum with the transmitter's scaling removed (√P included)."""
    cfg = tx.ofdm
    frames = field.reshape(*field.shape[:-1], tx.n_frames, cfg.samples_per_frame)
    useful = frames[..., cfg.cp_len * cfg.oversampling :]
    scale = np.sqrt(cfg.frame_energy(tx.data_symbol_energy) / tx.launch_power_per_polarization_w)
    return np.fft.fft(useful, axis=-1) * scale / cfg.fft_size


# ---- class structure ----------------------------------------------------------


def test_is_a_transmitter_but_not_pulse_shaped():
    tx = make_tx()
    assert isinstance(tx, Transmitter)
    assert not isinstance(tx, PulseShapedTransmitter)
    for field in (
        "samples_per_symbol",
        "roll_off",
        "pulse_span_symbols",
        "pilots",
        "symbol_rate_hz",
    ):
        assert field not in type(tx).model_fields


@pytest.mark.parametrize("oversampling", OVERSAMPLING)
def test_sample_period_is_inverse_oversampled_bandwidth(oversampling):
    tx = make_tx(oversampling=oversampling)
    assert tx.bandwidth_hz == pytest.approx(64 * 500e6)
    assert tx.dt_s == pytest.approx(1 / (oversampling * 64 * 500e6))


def test_time_domain_pilots_keyword_is_rejected():
    """OFDM pilots are subcarriers (OFDMConfig), not the time-domain PilotConfig."""
    with pytest.raises(ValidationError, match="pilots"):
        OFDMTransmitter(
            modulation_order=4,
            subcarrier_spacing_hz=1e9,
            n_frames=2,
            ofdm=OFDMConfig(n_subcarriers=16),
            pilots=None,
        )


def test_pilot_only_frame_cannot_be_transmitted():
    ofdm = OFDMConfig(n_subcarriers=8, n_guard=1, pilot_subcarrier_indices=(0, 1, 2, 5, 6, 7))
    assert ofdm.n_data == 0
    with pytest.raises(ValidationError, match="no data symbols"):
        OFDMTransmitter(modulation_order=4, subcarrier_spacing_hz=1e9, n_frames=4, ofdm=ofdm)


def test_dm_blocks_must_align_with_the_data_symbols_of_all_frames():
    pts = ConstellationConfig.from_square_qam(16).points
    pmf = maxwell_boltzmann(pts, 0.3)
    cfg = ConstellationConfig.from_points(
        pts, ConstellationConfig.from_square_qam(16).bit_labels, pmf
    )
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=56)
    make_tx(constellation=cfg, dm=dm, modulation_order=None, n_frames=1)  # 56 data subcarriers
    with pytest.raises(ValidationError, match=r"multiple of dm\.block_length"):
        make_tx(
            constellation=cfg, dm=dm.replace(block_length=48), modulation_order=None, n_frames=1
        )


# ---- shapes ---------------------------------------------------------------------


@pytest.mark.parametrize("oversampling", OVERSAMPLING)
@pytest.mark.parametrize(("order", "k"), [(4, 2), (16, 4), (64, 6)])
def test_single_polarisation_shapes(order, k, oversampling):
    tx = make_tx(modulation_order=order, oversampling=oversampling, n_frames=8)
    sent = tx.generate()
    assert sent.field.shape == (8 * oversampling * (64 + 16),) == (tx.n_samples,)
    assert sent.symbols.shape == (8 * 56,)
    assert sent.bits.shape == (8 * 56 * k,)


def test_dual_polarisation_shapes_and_independence():
    tx = make_tx(polarization="dual", n_frames=8)
    sent = tx.generate()
    assert sent.field.shape == (2, 8 * 80)
    assert sent.symbols.shape == (2, 8 * 56)
    assert sent.bits.shape == (2, 8 * 56 * 2)
    assert not np.array_equal(sent.bits[0], sent.bits[1])


# ---- time-domain structure ------------------------------------------------------


@pytest.mark.parametrize("oversampling", OVERSAMPLING)
def test_cyclic_prefix_repeats_the_frame_tail(oversampling):
    tx = make_tx(cp_len=16, oversampling=oversampling, n_frames=4)
    frames = tx.generate().field.reshape(4, tx.ofdm.samples_per_frame)
    cp = 16 * oversampling
    np.testing.assert_array_equal(frames[:, :cp], frames[:, -cp:])


def test_no_cyclic_prefix():
    tx = make_tx(cp_len=0, n_frames=4)
    assert tx.generate().field.shape == (4 * 64,)


# ---- frequency-domain structure -------------------------------------------------


@pytest.mark.parametrize("oversampling", OVERSAMPLING)
def test_every_bin_carries_what_it_should(oversampling):
    tx = make_tx(pilots=(5, 11, 25, 40), oversampling=oversampling, n_frames=4)
    cfg = tx.ofdm
    sent = tx.generate()
    spec = frames_spectrum(tx, sent.field)
    np.testing.assert_allclose(
        spec[:, cfg.fft_bins(cfg.data_subcarrier_indices())],
        sent.symbols.reshape(4, -1),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        spec[:, cfg.fft_bins(cfg.pilot_subcarrier_indices)], cfg.pilot_symbol, atol=1e-12
    )
    used = np.zeros(cfg.fft_size, dtype=bool)
    used[cfg.fft_bins(cfg.data_subcarrier_indices())] = True
    used[cfg.fft_bins(cfg.pilot_subcarrier_indices)] = True
    np.testing.assert_allclose(spec[:, ~used], 0.0, atol=1e-12)  # guard + oversampling bins


@pytest.mark.parametrize("oversampling", [2, 4])
def test_oversampled_spectrum_is_concentrated_in_the_subcarrier_band(oversampling):
    """Each frame is rectangularly windowed, so CP-OFDM has sinc side lobes: it is
    *not* strictly band-limited (unlike the exact RRC); with no guard band ≈ 98.5 % of the energy
    lies within the N·Δf subcarrier band and the rest decays away from it."""
    tx = make_tx(oversampling=oversampling, n_frames=64, n_guard=0, cp_len=0)
    sig = tx.generate()
    psd = np.abs(np.fft.fft(sig.field)) ** 2
    af = np.abs(np.fft.fftfreq(sig.field.size, sig.dt_s))
    half = tx.bandwidth_hz / 2
    assert psd[af <= half].sum() / psd.sum() > 0.98
    assert psd[af > 1.5 * half].sum() < psd[(af > half) & (af <= 1.5 * half)].sum()


# ---- power ---------------------------------------------------------------------


@pytest.mark.parametrize("oversampling", OVERSAMPLING)
@pytest.mark.parametrize("pilots", [(), (3, 5, 7, 9)])
@pytest.mark.parametrize("power_dbm", [-5.0, 0.0, 4.0])
def test_field_power_is_the_launch_power(pilots, oversampling, power_dbm):
    tx = make_tx(
        n_subcarriers=128,
        pilots=pilots,
        oversampling=oversampling,
        n_frames=200,
        seed=42,
        modulation_order=16,
        launch_power_per_channel_dbm=power_dbm,
    )
    np.testing.assert_allclose(tx.generate().mean_power_w, dbm_to_w(power_dbm), rtol=0.02)


def test_dual_polarisation_splits_the_power():
    tx = make_tx(polarization="dual", n_frames=200, launch_power_per_channel_dbm=0.0)
    np.testing.assert_allclose(
        np.mean(np.abs(tx.generate().field) ** 2, axis=-1), 0.5e-3, rtol=0.03
    )


# ---- deterministic inputs ---------------------------------------------------------


def test_fed_bits_are_transmitted():
    tx = make_tx(n_frames=4)
    bits = np.random.default_rng(7).integers(0, 2, tx.n_bits_per_stream, dtype=np.uint8)
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)
    np.testing.assert_array_equal(sent.symbols, tx.resolved_constellation.map(bits))


def test_fed_symbols_are_transmitted_and_demapped():
    tx = make_tx(n_frames=4)
    cfg = tx.resolved_constellation
    idx = np.random.default_rng(8).integers(0, 4, tx.n_data_symbols_per_stream)
    sent = tx.generate(symbols=cfg.points[idx])
    np.testing.assert_array_equal(sent.symbols, cfg.points[idx])
    np.testing.assert_array_equal(sent.bits, cfg.bits_from_indices(idx))


@pytest.mark.parametrize(
    ("kwarg", "value", "match"),
    [
        ("bits", np.zeros(7, np.uint8), "`bits` must have shape"),
        ("symbols", np.zeros(3, complex), "`symbols` must have shape"),
    ],
)
def test_wrongly_shaped_input_is_rejected(kwarg, value, match):
    with pytest.raises(ValueError, match=match):
        make_tx(n_frames=4).generate(**{kwarg: value})


def test_bits_and_symbols_together_are_rejected():
    tx = make_tx(n_frames=2)
    with pytest.raises(ValueError, match="at most one"):
        tx.generate(bits=np.zeros(tx.n_bits_per_stream, np.uint8), symbols=np.zeros(112, complex))


# ---- reproducibility, DM ---------------------------------------------------------------


def test_same_seed_same_field_different_seed_different_field():
    np.testing.assert_array_equal(
        make_tx(seed=3).generate().field, make_tx(seed=3).generate().field
    )
    assert not np.allclose(make_tx(seed=3).generate().field, make_tx(seed=4).generate().field)


def test_dm_bit_count_and_exact_composition():
    base = ConstellationConfig.from_square_qam(16)
    pmf = maxwell_boltzmann(base.points, 0.3)
    cfg = base.with_probabilities(pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=56)
    tx = make_tx(constellation=cfg, dm=dm, modulation_order=None, n_frames=8)
    sent = tx.generate()
    assert sent.bits.shape == (8 * dm.bits_per_block,)
    counts = np.bincount(cfg.nearest(sent.symbols), minlength=16)
    np.testing.assert_array_equal(counts, 8 * dm.composition)
