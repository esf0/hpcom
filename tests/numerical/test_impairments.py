"""Transceiver impairments (:mod:`hpcom.impairments`) against theory, in links, and in JAX.

* laser and LO phase noise: Wiener increments of variance 2πΔν·Δt, phase
  variance growing linearly, one phase for both polarisations, independent
  batch elements;
* an LO frequency offset is an exact spectral shift;
* IQ imbalance: y = I + jg(Q cos φ − I sin φ), image rejection
  (1 + g² + 2g cos φ)/(1 + g² − 2g cos φ);
* quantiser: SNR 6.02b + 4.77 − 20 log₁₀ k dB for a Gaussian signal,
  uniform-like error;
* Mach–Zehnder modulator: error ∝ drive², power preserved;
* links with impairments round-trip through JSON, run in Simulation and are
  back-propagated (impairments are transparent to DBP);
* :func:`hpcom.autodiff.propagate_link` equals NumPy for the deterministic
  elements, draws the random ones from the key and passes gradients
  (straight through the quantiser).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpcom import (
    EDFA,
    DigitalBackPropagation,
    Fiber,
    IQImbalance,
    LaserPhaseNoise,
    Link,
    LocalOscillator,
    MachZehnderModulator,
    Quantizer,
    Simulation,
    SingleCarrierTransmitter,
)

DT = 1 / 64e9


def _gaussian(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


# ---------------------------------------------------------------------------
# Phase noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("element", "sign"),
    [
        (LaserPhaseNoise(linewidth_hz=1e6), 1.0),
        (LocalOscillator(linewidth_hz=1e6), -1.0),
    ],
    ids=["laser", "local-oscillator"],
)
def test_phase_noise_is_a_wiener_process(element, sign):
    batch, n = 2000, 512
    out = element.apply(np.ones((batch, n), complex), DT, rng=np.random.default_rng(1))
    np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)
    phase = sign * np.unwrap(np.angle(out), axis=-1)
    assert np.all(phase[:, 0] == 0.0)
    sigma2 = 2 * math.pi * element.linewidth_hz * DT
    increments = np.diff(phase, axis=-1).ravel()
    tol = 5 * sigma2 * math.sqrt(2 / increments.size)
    assert np.var(increments) == pytest.approx(sigma2, abs=tol)
    assert abs(np.mean(increments)) < 5 * math.sqrt(sigma2 / increments.size)
    for k in (n // 4, n // 2, n - 1):  # Var φ_k = kσ² across the batch: independent rows
        assert np.var(phase[:, k]) == pytest.approx(k * sigma2, rel=5 * math.sqrt(2 / batch))


def test_phase_noise_is_common_to_both_polarisations():
    x = _gaussian((3, 2, 256))
    out = LaserPhaseNoise(linewidth_hz=1e7, polarization="dual").apply(
        x, DT, rng=np.random.default_rng(2)
    )
    np.testing.assert_allclose(out[:, 0] / x[:, 0], out[:, 1] / x[:, 1], atol=1e-12)
    with pytest.raises(ValueError, match=r"\(\.\.\., 2, n\)"):
        LaserPhaseNoise(linewidth_hz=1e7, polarization="dual").apply(_gaussian((3, 256)), DT)
    assert np.array_equal(LaserPhaseNoise(linewidth_hz=0.0).apply(x, DT), x)


@pytest.mark.parametrize("bins", [1, -7, 40])
def test_lo_frequency_offset_is_an_exact_spectral_shift(bins):
    n = 1024
    offset = bins / (n * DT)
    x = _gaussian((2, n))
    out = LocalOscillator(frequency_offset_hz=offset).apply(x, DT)
    np.testing.assert_allclose(np.fft.fft(out), np.roll(np.fft.fft(x), -bins, axis=-1), atol=1e-9)


# ---------------------------------------------------------------------------
# IQ imbalance, quantiser, modulator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("db", "deg"), [(0.5, 0.0), (0.0, 3.0), (1.0, -5.0), (-0.3, 10.0)])
def test_iq_imbalance_definition_and_image_rejection(db, deg):
    el = IQImbalance(amplitude_imbalance_db=db, phase_imbalance_deg=deg)
    g, phi = 10 ** (db / 20), math.radians(deg)
    x = _gaussian((512,))
    expected = x.real + 1j * g * (x.imag * math.cos(phi) - x.real * math.sin(phi))
    np.testing.assert_allclose(el.apply(x, DT), expected, atol=1e-12)
    n, k = 1024, 37  # a tone at bin k: its image appears at −k
    tone = np.exp(2j * np.pi * k * np.arange(n) / n)
    spec = np.abs(np.fft.fft(el.apply(tone, DT))) ** 2
    irr = (1 + g**2 + 2 * g * math.cos(phi)) / (1 + g**2 - 2 * g * math.cos(phi))
    assert spec[k] / spec[-k] == pytest.approx(irr, rel=1e-9)
    assert el.image_rejection_ratio_db == pytest.approx(10 * math.log10(irr), rel=1e-12)
    assert IQImbalance().image_rejection_ratio_db == math.inf


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_quantizer_snr_follows_uniform_quantiser_theory(bits):
    x = _gaussian((2, 1 << 17), seed=bits)
    q = Quantizer(bits=bits, clipping_ratio=4.0).apply(x, DT)
    error = q - x
    snr_db = 10 * math.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(error) ** 2))
    assert snr_db == pytest.approx(6.02 * bits + 4.77 - 20 * math.log10(4.0), abs=0.5)
    assert len(np.unique(np.round(q.real[0], 12))) <= 2**bits
    # granular region (no clipping): error uniform on ±Δ/2, variance Δ²/12, kurtosis 9/5
    sigma = np.sqrt(np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) / 2)
    step = 2 * 4.0 * sigma / 2**bits
    inside = np.abs(x.real) < 4.0 * sigma
    e = (error.real / step)[inside]
    assert np.max(np.abs(e)) <= 0.5 + 1e-9
    assert np.var(e) == pytest.approx(1 / 12, rel=0.02)
    assert np.mean(e**4) / np.mean(e**2) ** 2 == pytest.approx(1.8, abs=0.03)
    assert np.array_equal(Quantizer(bits=bits).apply(np.zeros(16, complex), DT), np.zeros(16))


def test_modulator_is_linear_for_small_drive_and_keeps_power():
    x = _gaussian((2, 4096))
    errors = []
    for drive in (0.1, 0.05):
        y = MachZehnderModulator(drive=drive).apply(x, DT)
        np.testing.assert_allclose(
            np.mean(np.abs(y) ** 2, axis=-1), np.mean(np.abs(x) ** 2, axis=-1), rtol=1e-12
        )
        errors.append(np.linalg.norm(y - x) / np.linalg.norm(x))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    strong = MachZehnderModulator(drive=1.0).apply(x, DT)
    assert np.linalg.norm(strong - x) / np.linalg.norm(x) > 0.1


@pytest.mark.parametrize("drive", [0.1, 0.3, 0.6])
def test_modulator_distortion_follows_bussgang(drive):
    x = _gaussian((1 << 20,), seed=6)
    y = MachZehnderModulator(drive=drive).apply(x, DT)
    gain = np.vdot(x, y) / np.vdot(x, x)
    sdr_db = 10 * math.log10(np.mean(np.abs(gain * x) ** 2) / np.mean(np.abs(y - gain * x) ** 2))
    s2 = (drive * math.pi / 2) ** 2
    linear = s2 * math.exp(-s2)
    assert sdr_db == pytest.approx(
        10 * math.log10(linear / ((1 - math.exp(-2 * s2)) / 2 - linear)), abs=0.1
    )


# ---------------------------------------------------------------------------
# In links and simulations
# ---------------------------------------------------------------------------

FIBER = Fiber.ssmf(80.0, n_steps=10, polarization="dual")


def _impaired(laser_hz: float = 1e5, **lo: float) -> Link:
    return Link(
        elements=(
            MachZehnderModulator(drive=0.1),
            IQImbalance(amplitude_imbalance_db=0.2, phase_imbalance_deg=1.0),
            Quantizer(bits=8),
            LaserPhaseNoise(linewidth_hz=laser_hz, polarization="dual"),
            FIBER,
            EDFA(gain_db=FIBER.loss_db, noise_figure_db=5.0),
            LocalOscillator(polarization="dual", **lo),
        )
    )


def test_links_with_impairments_round_trip_and_back_propagate():
    link = _impaired(frequency_offset_hz=1e8, linewidth_hz=1e5)
    restored = Link.model_validate_json(link.model_dump_json())
    assert restored == link and type(restored.elements[2]) is Quantizer
    x = _gaussian((2, 1024)) * 0.03
    y = link.propagate(x, DT, rng=np.random.default_rng(3))
    dbp = DigitalBackPropagation(link=link, steps_per_span=2)  # impairments are transparent to DBP
    assert np.all(np.isfinite(dbp.compensate(y, DT)))
    assert np.all(np.isfinite(link.backpropagate(y, DT)))


def test_simulation_runs_with_impairments_and_they_cost_snr():
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=32e9, n_symbols=2048, polarization="dual"
    )

    def snr(link: Link) -> float:
        rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
        return Simulation(transmitter=tx, link=link, receiver=rx, seed=4).run().metrics.snr_db

    clean = snr(Link(elements=(FIBER, EDFA(gain_db=FIBER.loss_db, noise_figure_db=5.0))))
    static = snr(_impaired(laser_hz=0.0))  # MZM, IQ imbalance (37 dB IRR), 8-bit DAC
    noisy = snr(_impaired())  # + 100 kHz laser, no carrier-phase recovery in this receiver
    offset = snr(_impaired(frequency_offset_hz=2e8))
    # the image ν x* of QAM is uncorrelated with x: it adds like noise at −IRR (the rest is < 0.3 dB)
    irr_db = _impaired().elements[1].image_rejection_ratio_db
    assert static == pytest.approx(
        -10 * math.log10(10 ** (-clean / 10) + 10 ** (-irr_db / 10)), abs=0.3
    )
    assert noisy < static - 3.0
    assert offset < noisy - 10.0  # an uncompensated 200 MHz offset destroys the SNR
    assert noisy == snr(_impaired())  # reproducible from the seed


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "element",
    [
        IQImbalance(amplitude_imbalance_db=0.7, phase_imbalance_deg=-4.0),
        Quantizer(bits=5, clipping_ratio=3.0),
        MachZehnderModulator(drive=0.6),
        LocalOscillator(frequency_offset_hz=1.3e9, polarization="dual"),
    ],
    ids=["iq", "quantizer", "mzm", "lo-offset"],
)
def test_jax_equals_numpy_for_deterministic_elements(element):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hpcom import autodiff as ad

    x = _gaussian((3, 2, 256), seed=5)
    expected = element.apply(x, DT)
    link = Link(elements=(element,))
    with jax.enable_x64(True):
        out = ad.propagate_link(jnp.asarray(x), link, DT)
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)
    out32 = ad.propagate_link(jnp.asarray(x, jnp.complex64), link, DT)
    if not isinstance(element, Quantizer):  # float32 may land on the other side of a level
        np.testing.assert_allclose(np.asarray(out32), expected, atol=2e-5)


def test_jax_random_impairments_and_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hpcom import autodiff as ad

    with jax.enable_x64(True):
        noise = LaserPhaseNoise(linewidth_hz=1e6)
        out = ad.propagate_link(
            jnp.ones((2000, 256), jnp.complex128),
            Link(elements=(noise,)),
            DT,
            key=jax.random.key(0),
        )
        increments = np.diff(np.unwrap(np.angle(np.asarray(out)), axis=-1), axis=-1).ravel()
        sigma2 = 2 * math.pi * noise.linewidth_hz * DT
        assert np.var(increments) == pytest.approx(
            sigma2, abs=5 * sigma2 * math.sqrt(2 / increments.size)
        )
        with pytest.raises(ValueError, match="key"):
            ad.propagate_link(jnp.ones(16, jnp.complex128), Link(elements=(noise,)), DT)

        link = _impaired(frequency_offset_hz=1e8, linewidth_hz=1e5)
        x = jnp.asarray(_gaussian((2, 512)) * 0.03)
        assert ad.link_parameters(link)[0] is None

        def loss(a):
            return jnp.sum(jnp.abs(ad.propagate_link(a, link, DT, key=jax.random.key(1))) ** 2)

        assert np.all(np.isfinite(np.asarray(jax.grad(loss)(x))))

        # straight through the quantiser: the gradient of Re⟨c, Q(x)⟩ is that of the identity
        q = Link(elements=(Quantizer(bits=3),))
        c = jnp.asarray(_gaussian((512,), seed=9))
        g = jax.grad(lambda a: jnp.sum(jnp.real(jnp.conj(c) * ad.propagate_link(a, q, DT))))(x[0])
        np.testing.assert_allclose(np.asarray(g), np.conj(np.asarray(c)), atol=1e-12)

        # links without impairments keep their key assignment: one split per noisy EDFA
        plain = Link.from_spans(2, FIBER, EDFA(gain_db=FIBER.loss_db, noise_figure_db=5.0))
        k = jax.random.key(7)
        manual = x
        for el, kk in zip(plain.elements[1::2], jax.random.split(k, 2), strict=True):
            manual = ad.amplify(ad.propagate(manual, FIBER, DT), el, DT, key=kk)
        np.testing.assert_allclose(
            np.asarray(ad.propagate_link(x, plain, DT, key=k)), np.asarray(manual), atol=1e-14
        )
