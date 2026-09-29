"""Validation and edge branches not exercised by the behavioural tests."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import hpcom
from helpers import sc_pair
from hpcom import (
    AdditiveNoise,
    ConstantCompositionDM,
    ConstellationConfig,
    Link,
    OFDMConfig,
    OFDMReceiver,
    OFDMTransmitter,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)
from hpcom._model import FloatArrayField, HpcomModel, UInt8ArrayField, _hashable, _values_equal
from hpcom.backends.base import check_field_shape
from hpcom.backends.python import kernels
from hpcom.experimental import ExperimentalReceiver, ExperimentalTransmission
from hpcom.modulation import apsk, maxwell_boltzmann, maxwell_boltzmann_for_entropy


class _Arrays(HpcomModel):
    x: FloatArrayField | None = None
    b: UInt8ArrayField | None = None


# ---- model base ----------------------------------------------------------------


def test_array_fields_reject_complex_and_non_integer_values():
    with pytest.raises(ValidationError, match="real-valued"):
        _Arrays(x=[1 + 1j])
    with pytest.raises(ValidationError, match="integer values"):
        _Arrays(b=[0.5])
    with pytest.raises(ValidationError, match="integer values"):
        _Arrays(b=[np.nan])


def test_value_equality_and_hashing_of_containers():
    assert not _values_equal(np.zeros(2), [0.0, 0.0])
    assert _values_equal({"a": np.ones(2)}, {"a": np.ones(2)})
    assert not _values_equal({"a": np.ones(2)}, {"b": np.ones(2)})
    assert not _values_equal((1, 2), [1, 2])
    assert _hashable({"b": [np.zeros(1)], "a": 1}) == _hashable({"a": 1, "b": [np.zeros(1)]})


def test_plain_model_copy_is_equal():
    tx, _ = sc_pair()
    assert tx.model_copy() == tx and tx.model_copy(deep=True) == tx


# ---- package ---------------------------------------------------------------------


def test_legacy_modules_are_attributes_of_the_package_and_warn_on_first_use():
    """Checked in a fresh interpreter: the warning fires once per process."""
    import subprocess
    import sys

    pytest.importorskip("tensorflow")
    code = (
        "import warnings, hpcom\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    assert hpcom.decorators.__name__ == 'hpcom._legacy.decorators'\n"
        "assert any(issubclass(x.category, DeprecationWarning) for x in w), w\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_unknown_package_attribute():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = hpcom.nope


# ---- backends --------------------------------------------------------------------


def test_field_shape_checks():
    class F:
        polarization = "single"
        n_cores = 1

    with pytest.raises(ValueError, match=r"expects a field of shape \(\.\.\., n_samples\)"):
        check_field_shape(np.array(1.0 + 0j), F())
    F.polarization = "triple"
    with pytest.raises(ValueError, match="unknown fiber polarization"):
        check_field_shape(np.ones(4, complex), F())


@pytest.mark.parametrize(
    ("polarization", "good", "bad"),
    [
        ("single", (3, 8), [(8,), (2, 8), (3,)]),
        ("dual", (3, 2, 8), [(2, 8), (3, 8), (2, 3, 8)]),
    ],
)
def test_multicore_field_shape_checks(polarization, good, bad):
    class F:
        n_cores = 3

    F.polarization = polarization
    check_field_shape(np.ones(good, complex), F())
    check_field_shape(np.ones((5, *good), complex), F())  # batch axes are fine
    for shape in bad:
        with pytest.raises(ValueError, match="3-core"):
            check_field_shape(np.ones(shape, complex), F())


def test_ssfm_rejects_zero_steps():
    from dataclasses import dataclass

    from hpcom.backends import get_propagator

    @dataclass
    class F:
        length_km: float = 1.0
        alpha_db_per_km: float = 0.0
        beta2_s2_per_m: float = 0.0
        beta3_s3_per_m: float = 0.0
        gamma_per_w_per_km: float = 1.0
        n_steps: int = 0
        step_distribution: str = "uniform"
        step_log_factor: float = 1.0
        beta1_s_per_m: float = 0.0
        scheme: object = None
        n_cores: int = 1
        polarization: str = "single"

    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        get_propagator("numpy").propagate(np.ones(8, complex), F(), 1e-12)


def test_kernel_input_checks():
    with pytest.raises(ValueError, match="1-D"):
        kernels.nearest_indices(np.ones((2, 2), complex), np.ones(4, complex))
    assert kernels._bits_to_int(np.empty(0, np.uint8)) == 0
    with pytest.raises(ValueError, match="multiple of block_length"):
        kernels.ccdm_decode(np.zeros(5, np.intp), np.array([2, 2]), 2)


# ---- modulation -------------------------------------------------------------------


def test_constellation_validation():
    with pytest.raises(ValidationError, match="distinct"):
        ConstellationConfig(points=[1, 1, -1, -1j], bit_labels=[[0, 0], [0, 1], [1, 0], [1, 1]])
    with pytest.raises(ValueError, match="power-of-2"):
        ConstellationConfig.from_points([1, 2, 3])
    with pytest.raises(ValueError, match="probabilities must have shape"):
        ConstellationConfig.from_points([1, -1], probabilities=[1.0])
    with pytest.raises(ValueError, match="zero expected power"):
        ConstellationConfig.from_points([0, 0], probabilities=[1.0, 0.0])
    with pytest.raises(ValueError, match="0/1"):
        ConstellationConfig.from_square_qam(4).indices_from_bits([0, 2])
    with pytest.raises(ValueError, match="ring_sizes must be >= 1"):
        apsk((0, 4), (1.0, 2.0))


def test_entropy_of_a_shaped_constellation():
    base = ConstellationConfig.from_square_qam(16)
    assert base.entropy_bits == pytest.approx(4.0)
    assert base.with_probabilities(
        maxwell_boltzmann_for_entropy(base.points, 3.0)
    ).entropy_bits == pytest.approx(3.0)


def test_maxwell_boltzmann_for_entropy_limits():
    pts = ConstellationConfig.from_square_qam(16).points
    np.testing.assert_allclose(maxwell_boltzmann_for_entropy(pts, 4.0), 1 / 16)
    with pytest.raises(ValueError, match="entropy_bits must be in"):
        maxwell_boltzmann_for_entropy(pts, 4.5)
    with pytest.raises(ValueError, match="not reachable"):
        maxwell_boltzmann_for_entropy(pts, 1e-9)  # below the entropy of the 4 inner points
    with pytest.raises(ValueError, match="nu must be >= 0"):
        maxwell_boltzmann(pts, -1.0)


def test_distribution_matcher_validation():
    with pytest.raises(ValidationError, match="1-D with >= 2 entries"):
        ConstantCompositionDM(target_pmf=[1.0], block_length=4)
    with pytest.raises(ValidationError, match="single permutation"):
        ConstantCompositionDM(target_pmf=[1.0, 0.0], block_length=4)
    dm = ConstantCompositionDM(target_pmf=[0.5, 0.5], block_length=4)  # 6 perms, 2 bits
    bits, valid = dm.decode_with_errors(np.array([1, 1, 0, 0]))  # right composition, rank 5
    assert not valid[0] and not bits.any()
    with pytest.raises(ValueError, match="multiple of block_length"):
        dm.decode_with_errors(np.zeros(3, np.intp))


def test_distribution_matchers_must_match():
    base = ConstellationConfig.from_square_qam(4)
    pmf = [0.4, 0.1, 0.1, 0.4]
    cfg = base.with_probabilities(pmf)
    a = ConstantCompositionDM(target_pmf=pmf, block_length=16)
    tx = SingleCarrierTransmitter(constellation=cfg, dm=a, symbol_rate_hz=1e9, n_symbols=64)
    assert (
        tx.modulation_mismatch(tx.replace(dm=a.replace(block_length=32)))
        == "distribution matchers differ"
    )


def test_fed_symbols_through_a_distribution_matcher():
    base = ConstellationConfig.from_square_qam(16)
    pmf = maxwell_boltzmann(base.points, 0.2)
    cfg = base.with_probabilities(pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    tx = SingleCarrierTransmitter(
        constellation=cfg, dm=dm, symbol_rate_hz=1e9, n_symbols=128, seed=0
    )
    sent = tx.generate()
    again = tx.generate(symbols=sent.symbols)
    np.testing.assert_array_equal(again.bits, sent.bits)


# ---- transmitters / receivers ------------------------------------------------------


def test_signal_convenience_properties():
    tx, _ = sc_pair(pilots=hpcom.PilotConfig(period=4))
    sig = tx.generate()
    assert tx.sample_rate_hz == pytest.approx(1 / tx.dt_s)
    np.testing.assert_array_equal(sig.pilot_indices, tx.pilot_indices)
    wdm = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    np.testing.assert_array_equal(wdm.generate().channel_frequencies_hz, [-50e9, 0, 50e9])
    ofdm = OFDMTransmitter(
        modulation_order=4, subcarrier_spacing_hz=1e9, n_frames=2, ofdm=OFDMConfig(n_subcarriers=16)
    )
    np.testing.assert_array_equal(ofdm.generate().data_indices, np.arange(32))
    exp = ExperimentalTransmission(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8)
    np.testing.assert_array_equal(exp.generate().data_indices, np.arange(8))


def test_receiver_validation_and_compatibility_messages():
    with pytest.raises(ValidationError, match="finite"):
        hpcom.SingleCarrierReceiver(modulation_order=4, convolution="linear")
    base = ConstellationConfig.from_square_qam(4)
    cfg = base.with_probabilities([0.4, 0.1, 0.1, 0.4])
    dm = ConstantCompositionDM(target_pmf=cfg.probabilities, block_length=16)
    with pytest.raises(ValidationError, match="cannot be combined"):
        hpcom.SingleCarrierReceiver(constellation=cfg, dm=dm, pilots=hpcom.PilotConfig(period=4))
    with pytest.raises(ValidationError, match="cannot be combined"):
        SingleCarrierTransmitter(
            constellation=cfg,
            dm=dm,
            pilots=hpcom.PilotConfig(period=4),
            symbol_rate_hz=1e9,
            n_symbols=64,
        )
    ofdm_tx = OFDMTransmitter(
        modulation_order=4, subcarrier_spacing_hz=1e9, n_frames=2, ofdm=OFDMConfig(n_subcarriers=16)
    )
    exp_tx = ExperimentalTransmission(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8)
    assert (
        "cannot receive"
        in hpcom.WDMReceiver(
            modulation_order=4, wdm=WDMConfig(n_channels=2, spacing_hz=1e9)
        ).compatibility_issues(ofdm_tx)[0]
    )
    assert (
        "cannot receive"
        in OFDMReceiver(modulation_order=4, ofdm=OFDMConfig(n_subcarriers=16)).compatibility_issues(
            exp_tx
        )[0]
    )
    assert any(
        "ofdm" in i
        for i in OFDMReceiver(
            modulation_order=4, ofdm=OFDMConfig(n_subcarriers=32)
        ).compatibility_issues(ofdm_tx)
    )
    assert (
        "cannot receive"
        in ExperimentalReceiver(modulation_order=4).compatibility_issues(ofdm_tx)[0]
    )
    _, sc_rx = sc_pair()
    assert "cannot receive a ExperimentalTransmission" in sc_rx.compatibility_issues(exp_tx)[0]
    assert "PulseShapedReceiver" not in sc_rx.compatibility_issues(exp_tx)[0]


def test_pilot_equaliser_without_pilots_fails_at_receive_time():
    tx = ExperimentalTransmission(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8)
    rx = ExperimentalReceiver(modulation_order=4, equalize="pilot")
    sig = tx.generate()
    with pytest.raises(ValueError, match="needs pilots"):
        rx.receive(sig.field, sig.dt_s)


# ---- link components --------------------------------------------------------------


def test_link_element_with_non_callable_apply():
    class Bad:
        apply = 3

    with pytest.raises(ValidationError, match="does not implement"):
        Link(elements=(Bad(),))


def test_additive_noise_paths():
    x = np.ones(1 << 16, complex)
    noiseless = AdditiveNoise(psd_per_polarization_w_per_hz=0.0).apply(x, 1e-11)
    np.testing.assert_array_equal(noiseless, x)
    assert noiseless is not x
    noisy = AdditiveNoise(psd_per_polarization_w_per_hz=1e-15).apply(x, 1e-11)  # unseeded
    assert np.var(noisy) == pytest.approx(1e-4, rel=0.05)
    np.testing.assert_array_equal(
        AdditiveNoise(psd_per_polarization_w_per_hz=1.0).apply_inverse(x, 1e-11), x
    )


def test_unseeded_edfa_draws_fresh_noise():
    edfa = hpcom.EDFA(gain_db=10.0)
    x = np.zeros(64, complex)
    assert not np.array_equal(edfa.amplify(x, 1e-11), edfa.amplify(x, 1e-11))


def test_rates_are_the_entropy_without_any_noise():
    from hpcom.metrics import gmi_gaussian, mi_gaussian

    cfg = ConstellationConfig.from_square_qam(16)
    idx = np.arange(16).repeat(4)
    assert mi_gaussian(cfg.points[idx], idx, cfg.points) == 4.0
    assert gmi_gaussian(cfg.points[idx], idx, cfg.points, cfg.bit_labels) == 4.0


def test_rrc_needs_at_least_one_tap():
    from hpcom._pulse import root_raised_cosine

    with pytest.raises(ValueError, match="n_taps"):
        root_raised_cosine(0, 4, 0.1)
