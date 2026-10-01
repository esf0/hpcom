"""hpcom.laser: gain fibers, lumped cavity components and the ring cavity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import (
    EDFA,
    Fiber,
    GainFiber,
    Link,
    OutputCoupler,
    RamanResponse,
    RingCavity,
    SaturableAbsorber,
    Scheme,
    SpectralFilter,
)

GAIN = {
    "length_km": 2e-3,
    "alpha_db_per_km": 0.0,
    "beta2_s2_per_m": -2e-26,
    "gamma_per_w_per_km": 3.0,
    "gain_db_per_m": 5.0,
    "n_steps": 10,
}
CW = {"n": 64, "dt": 1e-12, "p_sat": 5e-3, "g0_db": 2.0, "length_m": 10.0}


@pytest.mark.parametrize(
    ("cls", "kwargs", "match"),
    [
        (GainFiber, {**GAIN, "gain_db_per_m": 0.0}, "greater than 0"),
        (GainFiber, {**GAIN, "gain_bandwidth_hz": -1.0}, "greater than 0"),
        (GainFiber, {**GAIN, "saturation_energy_j": 0.0}, "greater than 0"),
        (GainFiber, {**GAIN, "gain_shape": "gaussian"}, "lorentzian"),
        (
            GainFiber,
            {
                **GAIN,
                "gain_bandwidth_hz": 3e12,
                "gain_shape": "parabolic",
                "scheme": "ssfm4-yoshida",
            },
            "diffusive",
        ),
        (
            GainFiber,
            {
                **GAIN,
                "gain_bandwidth_hz": 3e12,
                "gain_shape": "parabolic",
                "scheme": "ssfm4-suzuki",
            },
            "diffusive",
        ),
        (GainFiber, {**GAIN, "saturation_energy_j": 1e-12, "scheme": "rk4ip"}, "split-step"),
        (
            GainFiber,
            {
                **GAIN,
                "saturation_energy_j": 1e-12,
                "scheme": Scheme(method="ssfm-local-error", local_error=1e-6),
            },
            "split-step",
        ),
        (
            Fiber,
            {
                "length_km": 1.0,
                "nonlinear_gain_per_w_per_km": 1.0,
                "raman": RamanResponse(),
                "scheme": "rk4ip",
            },
            "instantaneous power",
        ),
        (
            Fiber,
            {
                "length_km": 1.0,
                "nonlinear_gain_per_w_per_km": 1.0,
                "self_steepening": True,
                "scheme": "rk4ip",
            },
            "instantaneous power",
        ),
        (
            Fiber,
            {
                "length_km": 1.0,
                "nonlinear_gain_per_w_per_km": 1.0,
                "scheme": Scheme(kerr_filter_bandwidth_hz=1e11),
            },
            "instantaneous power",
        ),
        (
            SaturableAbsorber,
            {"modulation_depth": 0.6, "saturation_power_w": 1.0, "non_saturable_loss": 0.4},
            "< 1",
        ),
        (SaturableAbsorber, {"modulation_depth": 0.0, "saturation_power_w": 1.0}, "greater than 0"),
        (SpectralFilter, {"bandwidth_hz": 0.0}, "greater than 0"),
        (SpectralFilter, {"bandwidth_hz": 1e12, "order": 0}, "greater than or equal to 1"),
        (OutputCoupler, {"output_ratio": 1.0}, "less than 1"),
    ],
)
def test_invalid_models_are_rejected(cls, kwargs, match):
    with pytest.raises(ValidationError, match=match):
        cls(**kwargs)


def test_ring_cavity_needs_exactly_one_output_coupler():
    with pytest.raises(ValidationError, match="exactly one OutputCoupler, got 0"):
        RingCavity(elements=(GainFiber(**GAIN),))
    with pytest.raises(ValidationError, match="exactly one OutputCoupler, got 2"):
        RingCavity(elements=(OutputCoupler(output_ratio=0.1), OutputCoupler(output_ratio=0.2)))


def test_models_round_trip_through_json():
    cavity = RingCavity(
        elements=(
            GainFiber(**GAIN, gain_bandwidth_hz=3e12, saturation_energy_j=4e-11),
            Fiber(length_km=8e-3, nonlinear_gain_per_w_per_km=-0.5),
            SaturableAbsorber(modulation_depth=0.3, saturation_power_w=10.0),
            SpectralFilter(bandwidth_hz=2e12, order=2, center_hz=1e11),
            OutputCoupler(output_ratio=0.5),
        )
    )
    restored = RingCavity.model_validate_json(cavity.model_dump_json())
    assert restored == cavity
    assert type(restored.elements[0]) is GainFiber
    assert GainFiber(**GAIN).small_signal_gain_db == pytest.approx(10.0)


def test_gain_fiber_is_recentred_only_with_flat_unsaturated_gain():
    flat = GainFiber(**GAIN)
    assert flat.recentred(0.0) is flat
    assert isinstance(flat.recentred(50e9), GainFiber)
    for kwargs in ({"gain_bandwidth_hz": 3e12}, {"saturation_energy_j": 1e-12}):
        fiber = GainFiber(**GAIN, **kwargs)
        assert fiber.recentred(0.0) is fiber
        with pytest.raises(ValueError, match="full-field"):
            fiber.recentred(50e9)


def test_saturable_absorber_transmission():
    sa = SaturableAbsorber(modulation_depth=0.3, saturation_power_w=2.0, non_saturable_loss=0.1)
    np.testing.assert_allclose(sa.transmission([0.0, 2.0, 1e12]), [0.6, 0.75, 0.9])
    x = np.array([[1.0, 0.0], [1.0, 2.0]], complex)  # dual polarisation: total power
    out = sa.apply(x, 1e-12)
    expected_t = sa.transmission(np.array([2.0, 4.0]))
    np.testing.assert_allclose(np.abs(out) ** 2, np.abs(x) ** 2 * expected_t)
    np.testing.assert_allclose(np.angle(out[:, 1]), 0.0)


@pytest.mark.parametrize("order", [1, 3])
def test_spectral_filter_has_its_fwhm_and_centre(order):
    n, dt = 10_000, 1e-14  # 10 GHz bins: the edges fall on the grid
    filt = SpectralFilter(bandwidth_hz=2e12, order=order, center_hz=5e11)
    f = np.fft.fftfreq(n, dt)
    h = filt.response(n, dt)
    assert h[np.argmin(np.abs(f - 5e11))] == pytest.approx(1.0)
    for edge in (5e11 + 1e12, 5e11 - 1e12):  # half power at centre ± FWHM/2
        assert h[np.argmin(np.abs(f - edge))] ** 2 == pytest.approx(0.5, rel=1e-9)
    tone = np.exp(2j * np.pi * f[200] * np.arange(n) * dt)
    np.testing.assert_allclose(filt.apply(tone, dt), tone * h[200], atol=1e-12)


def test_output_coupler_splits_the_power():
    coupler = OutputCoupler(output_ratio=0.3)
    x = np.full(8, 2.0 + 0j)
    kept, tapped = coupler.apply(x, 1e-12), coupler.tap(x)
    np.testing.assert_allclose(np.abs(kept) ** 2 + np.abs(tapped) ** 2, np.abs(x) ** 2)
    np.testing.assert_allclose(np.abs(tapped) ** 2, 0.3 * 4.0)


def test_runaway_nonlinear_gain_raises(nonlinear_backend):
    """2εcP ≥ 1 within one Kerr step: the exact solution blows up; every backend says so."""
    fiber = Fiber(
        length_km=1.0,
        alpha_db_per_km=0.0,
        gamma_per_w_per_km=1.0,
        nonlinear_gain_per_w_per_km=5.0,
        n_steps=2,
        backend=nonlinear_backend,
    )
    with pytest.raises(FloatingPointError):
        fiber.propagate(np.full(16, 1.0 + 0j), 1e-12)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_ring_cavity_reaches_the_cw_steady_state(polarization):
    r"""Gain = loss: :math:`P_\mathrm{in} = P_\mathrm{sat}(g_0L - \ln G)/(G - 1)`, :math:`G = 1/(1-r)`.

    The amplifier input :math:`P_\mathrm{in}` is amplified by :math:`G` to
    compensate the coupler; the output is :math:`rGP_\mathrm{in}`.
    """
    n, dt, p_sat = CW["n"], CW["dt"], CW["p_sat"]
    amplifier = GainFiber(
        length_km=CW["length_m"] / 1e3,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=0.0,
        gamma_per_w_per_km=0.0,
        gain_db_per_m=CW["g0_db"],
        saturation_energy_j=p_sat * n * dt,
        n_steps=40,
        polarization=polarization,
    )
    r = 0.3
    cavity = RingCavity(elements=(amplifier, OutputCoupler(output_ratio=r)))
    shape = (2, n) if polarization == "dual" else (n,)
    result = cavity.run(np.full(shape, 1e-3 + 0j), dt, 400, tol=1e-12)
    g_rt = 1 / (1 - r)
    g0l = CW["g0_db"] * math.log(10) / 10 * CW["length_m"]
    p_in = p_sat * (g0l - math.log(g_rt)) / (g_rt - 1)
    assert result.converged and result.round_trips < 200
    assert math.isnan(result.change[0])
    p_out = np.sum(np.mean(np.abs(result.output) ** 2, axis=-1))
    assert p_out == pytest.approx(r * g_rt * p_in, rel=2e-4)  # split-step error at 40 steps
    assert result.output_energy_j[-1] == pytest.approx(p_out * n * dt, rel=1e-12)
    np.testing.assert_allclose(result.intracavity, result.output * math.sqrt((1 - r) / r))


def test_lumped_components_work_in_a_link():
    link = Link(
        elements=(
            Fiber(length_km=1.0, n_steps=4),
            SpectralFilter(bandwidth_hz=50e9),
            EDFA(gain_db=0.2, noise_figure_db=None),
            SaturableAbsorber(modulation_depth=0.1, saturation_power_w=1.0),
        )
    )
    x = np.random.default_rng(0).standard_normal(256) * 1e-2 + 0j
    assert link.propagate(x, 1e-11).shape == x.shape
