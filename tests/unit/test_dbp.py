"""DigitalBackPropagation and the receiver compensation stage: API, exact identities."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import (
    EDFA,
    AdditiveNoise,
    Compensator,
    DigitalBackPropagation,
    Fiber,
    Link,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)

DT = 1 / (32e9 * 8)


def _link(polarization: str = "dual", n_spans: int = 2, beta3: float = 1.2e-40) -> Link:
    fiber = Fiber.ssmf(length_km=60, n_steps=30, polarization=polarization).replace(
        beta3_s3_per_m=beta3
    )
    return Link.from_spans(n_spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))


def _field(
    polarization: str, n: int = 2048, power_w: float = 5e-3, band: float = 0.2
) -> np.ndarray:
    """Random field with content only at |f| ≤ band·f_s."""
    rng = np.random.default_rng(0)
    shape = (2, n) if polarization == "dual" else (n,)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[..., np.abs(np.fft.fftfreq(n)) > band] = 0
    x = np.fft.ifft(spectrum, axis=-1)
    return x * np.sqrt(power_w / np.mean(np.abs(x) ** 2))


# ---------------------------------------------------------------------------
# DigitalBackPropagation.compensate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["single", "dual"])
@pytest.mark.parametrize(("steps", "scale"), [(None, 1.0), (3, 0.7), (1, 0.0)])
def test_full_rate_dbp_is_link_backpropagate(polarization, steps, scale):
    link = _link(polarization)
    x = _field(polarization)
    dbp = DigitalBackPropagation(link=link, steps_per_span=steps, nonlinearity_scale=scale)
    np.testing.assert_allclose(
        dbp.compensate(x, DT),
        link.backpropagate(x, DT, steps_per_span=steps, nonlinearity_scale=scale),
        atol=1e-12 * np.max(np.abs(x)),
    )


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_dbp_with_the_forward_steps_inverts_a_noiseless_link(polarization):
    link = _link(polarization)
    x = _field(polarization)
    y = DigitalBackPropagation(link=link, steps_per_span=None).compensate(link.propagate(x, DT), DT)
    np.testing.assert_allclose(y, x, atol=1e-10 * np.max(np.abs(x)))


@pytest.mark.parametrize(("scale", "band", "rtol"), [(0.0, 0.2, 1e-12), (1.0, 0.04, 1e-6)])
@pytest.mark.parametrize("offset_bins", [-150, 37, 200])
def test_frequency_offset_equals_backpropagating_the_full_field_then_shifting(
    scale, band, rtol, offset_bins
):
    """A channel shifted to 0 Hz and back-propagated with its offset ≡ full-field DBP, then shift.

    Exact while the field stays inside the sampled band. With nonlinearity
    the Kerr products spread the spectrum and the two grids wrap it around
    different edges; an oversampled field keeps that below ``rtol``.
    """
    n = 2048
    f_k = offset_bins / (n * DT)
    t = np.arange(n) * DT
    link = _link("dual")
    channel = _field("dual", n, band=band)
    full = channel * np.exp(2j * np.pi * f_k * t)
    dbp = DigitalBackPropagation(link=link, steps_per_span=4, nonlinearity_scale=scale)
    expected = dbp.compensate(full, DT) * np.exp(-2j * np.pi * f_k * t)
    np.testing.assert_allclose(
        dbp.compensate(channel, DT, frequency_offset_hz=f_k),
        expected,
        atol=rtol * np.max(np.abs(channel)),
    )


def test_linear_dbp_uses_the_exact_linear_backend_unless_told_otherwise():
    fiber = _link().fibers[0]
    assert DigitalBackPropagation(
        link=_link(), steps_per_span=1, nonlinearity_scale=0.0
    )._backward_fiber(fiber, 0.0).backend == ("numpy-linear")
    assert (
        DigitalBackPropagation(
            link=_link(), steps_per_span=1, nonlinearity_scale=0.0, backend="numpy"
        )
        ._backward_fiber(fiber, 0.0)
        .backend
        == "numpy"
    )
    assert DigitalBackPropagation(link=_link(), steps_per_span=1)._backward_fiber(
        fiber, 0.0
    ).backend == ("numpy")


def test_other_invertible_components_are_inverted_and_others_are_rejected():
    noise = AdditiveNoise(psd_per_polarization_w_per_hz=1e-20)  # apply_inverse = identity
    link = Link(elements=(*_link().elements, noise))
    x = _field("dual")
    dbp = DigitalBackPropagation(link=link, steps_per_span=None)
    np.testing.assert_allclose(dbp.compensate(x, DT), _link().backpropagate(x, DT), atol=1e-15)

    class Opaque:
        def apply(self, field, dt_s, *, rng=None):
            return field

    with pytest.raises(ValidationError, match="cannot be back-propagated"):
        DigitalBackPropagation(link=Link(elements=(Opaque(),)), steps_per_span=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nonlinearity_scale": -0.1, "steps_per_span": 1},
        {"steps_per_span": 0},
        {"steps_per_span": 1, "samples_per_symbol": 0},
        {"steps_per_span": 1, "bandwidth_hz": 0.0},
        {},  # steps_per_span is required (None must be explicit)
    ],
)
def test_invalid_dbp_parameters_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        DigitalBackPropagation(link=_link(), **kwargs)


def test_dbp_round_trips_through_json_inside_a_receiver():
    rx = SingleCarrierReceiver(
        modulation_order=16,
        polarization="dual",
        compensation=DigitalBackPropagation(
            link=_link(), steps_per_span=2, nonlinearity_scale=0.8, bandwidth_hz=40e9
        ),
    )
    assert SingleCarrierReceiver.model_validate_json(rx.model_dump_json()) == rx


# ---------------------------------------------------------------------------
# Receiver integration
# ---------------------------------------------------------------------------


def _sc(polarization: str = "dual", **rx_kwargs):
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=256,
        polarization=polarization,
        launch_power_per_channel_dbm=3.0,
    )
    return tx, tx.matched_receiver(**rx_kwargs)


def test_compensation_and_bulk_cd_compensation_are_exclusive():
    with pytest.raises(ValidationError, match="already compensates dispersion"):
        _sc(
            compensation=DigitalBackPropagation(link=_link(), steps_per_span=1),
            cd_compensation_s2=1e-24,
        )


def test_bandwidth_above_the_front_end_rate_is_rejected_at_receive():
    tx, rx = _sc(
        compensation=DigitalBackPropagation(link=_link(), steps_per_span=1, bandwidth_hz=70e9)
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="exceeds the front-end sampling rate"):
        rx.receive(sent.field, sent.dt_s)
    # The same bandwidth fits at 3 samples/symbol (96 GHz).
    rx.replace(
        compensation=rx.compensation.replace(samples_per_symbol=3)  # type: ignore[union-attr]
    ).receive(sent.field, sent.dt_s)


@pytest.mark.parametrize(("sps", "expected_len"), [(2, 512), (1, 256), (None, 2048)])
def test_front_end_shapes_and_sample_periods(sps, expected_len):
    link = _link()
    tx, rx = _sc(
        compensation=DigitalBackPropagation(link=link, steps_per_span=1, samples_per_symbol=sps)
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    samples, dt = rx.front_end(sent.field, sent.dt_s)
    assert samples.shape == (2, expected_len)
    assert dt == pytest.approx(1 / (32e9 * (sps or 8)))

    wdm_tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=32e9,
        n_symbols=256,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        polarization="dual",
    )
    wdm_rx = wdm_tx.matched_receiver(
        channels=(0, 2),
        compensation=DigitalBackPropagation(link=link, steps_per_span=1, samples_per_symbol=sps),
    )
    wdm_sent = wdm_tx.generate(rng=np.random.default_rng(0))
    samples, _ = wdm_rx.front_end(wdm_sent.field, wdm_sent.dt_s)
    if sps is None:  # identity front end → the stage sees the whole field (full-field DBP)
        np.testing.assert_array_equal(samples, wdm_sent.field)
    else:
        assert samples.shape == (2, 2, expected_len)


def test_front_end_without_compensation_is_the_baseband_field():
    tx, rx = _sc()
    sent = tx.generate(rng=np.random.default_rng(0))
    samples, dt = rx.front_end(sent.field, sent.dt_s)
    np.testing.assert_array_equal(samples, sent.field)
    assert dt == sent.dt_s


class _ExactCDC(Compensator):
    """A user-defined stage: exact CD compensation, aware of the channel frequency."""

    dispersion_s2: float
    beta3_s3: float = 0.0

    def compensate(self, field, dt_s, *, frequency_offset_hz=0.0):
        omega = 2 * np.pi * (np.fft.fftfreq(np.shape(field)[-1], d=dt_s) + frequency_offset_hz)
        phase = -self.dispersion_s2 / 2 * omega**2 + self.beta3_s3 / 6 * omega**3
        return np.fft.ifft(np.fft.fft(field, axis=-1) * np.exp(1j * phase), axis=-1)


def test_a_custom_compensator_plugs_into_the_same_chain():
    link = _link()
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=512,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        polarization="dual",
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    out = link.propagate(sent.field, sent.dt_s)
    bulk = tx.matched_receiver(
        equalize="none",
        cd_compensation_s2=link.total_dispersion_s2,
        cd_compensation_s3=link.total_beta3_s3,
    )
    custom = tx.matched_receiver(
        equalize="none",
        compensation=_ExactCDC(
            dispersion_s2=link.total_dispersion_s2, beta3_s3=link.total_beta3_s3
        ),
    )
    a, b = bulk.receive(out, sent.dt_s).symbols, custom.receive(out, sent.dt_s).symbols
    # The per-channel phase exp(iφ(ω_k)) differs by a constant only; compare up to it.
    a, b = np.asarray(a), np.asarray(b)
    ratio = np.mean(b / a, axis=-1, keepdims=True)
    np.testing.assert_allclose(b, a * ratio, atol=1e-10 * np.max(np.abs(a)))
    assert SingleCarrierReceiver  # custom stages serialise by qualified name like any model
    assert type(custom.model_validate_json(custom.model_dump_json()).compensation) is _ExactCDC


def test_a_compensator_returning_the_wrong_shape_is_reported():
    class Broken(Compensator):
        def compensate(self, field, dt_s, *, frequency_offset_hz=0.0):
            return np.asarray(field)[..., ::2]

    tx, rx = _sc(compensation=Broken())
    sent = tx.generate(rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"Broken.compensate returned shape"):
        rx.receive(sent.field, sent.dt_s)


# ---------------------------------------------------------------------------
# OFDM receivers
# ---------------------------------------------------------------------------


def _ofdm(**rx_kwargs):
    from hpcom import OFDMConfig, OFDMTransmitter

    cfg = OFDMConfig(n_subcarriers=64, cp_len=8, n_guard=2, oversampling=4)
    tx = OFDMTransmitter(
        modulation_order=16,
        subcarrier_spacing_hz=500e6,
        n_frames=8,
        ofdm=cfg,
        polarization="dual",
        seed=0,
    )
    return tx, tx.matched_receiver(**rx_kwargs)


@pytest.mark.parametrize(("sps", "per_frame"), [(2, 2 * 72), (1, 72), (4, 4 * 72)])
def test_ofdm_front_end_resamples_to_the_stage_rate(sps, per_frame):
    tx, rx = _ofdm(
        compensation=DigitalBackPropagation(link=_link(), steps_per_span=1, samples_per_symbol=sps)
    )
    sent = tx.generate()
    samples, dt = rx.front_end(sent.field, sent.dt_s)
    assert samples.shape == (2, 8 * per_frame)
    assert dt == pytest.approx(1 / (sps * 64 * 500e6))
    plain, dt_plain = tx.matched_receiver().front_end(sent.field, sent.dt_s)
    np.testing.assert_array_equal(plain, sent.field)
    assert dt_plain == sent.dt_s


def test_ofdm_compensation_validation_and_serialisation():
    dbp = DigitalBackPropagation(link=_link(), steps_per_span=2, bandwidth_hz=70e9)
    with pytest.raises(ValidationError, match="already compensates dispersion"):
        _ofdm(compensation=dbp, cd_compensation_s2=1e-24)
    tx, rx = _ofdm(compensation=dbp)
    assert type(rx).model_validate_json(rx.model_dump_json()) == rx
    sent = tx.generate()
    with pytest.raises(ValueError, match="exceeds the front-end sampling rate"):
        rx.receive(sent.field, sent.dt_s)  # 70 GHz > 2 × 32 GHz


def test_ofdm_identity_compensation_leaves_the_receiver_unchanged():
    class Identity(Compensator):
        def compensate(self, field, dt_s, *, frequency_offset_hz=0.0):
            return field

    tx, rx = _ofdm(compensation=Identity(samples_per_symbol=None))
    sent = tx.generate()
    np.testing.assert_allclose(
        rx.receive(sent.field, sent.dt_s).symbols,
        tx.matched_receiver().receive(sent.field, sent.dt_s).symbols,
        rtol=0,
        atol=1e-13,
    )
