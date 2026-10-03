"""Realistic receiver DSP: blind phase search and adaptive (CMA/LMS) butterfly equalisers.

* blind phase search tracks Wiener laser phase noise and restores the
  phase-noise-free SNR, where one complex gain per block fails; its phase
  error follows the window/noise/quantisation estimate;
* CMA undoes polarisation rotation and PMD without training data, close to
  the least-squares (``"mimo"``) limit; LMS removes CMA's misadjustment for
  16-QAM;
* :func:`hpcom.autodiff.demodulate` equals the NumPy receiver, its gradients
  through the adaptive equalisers equal finite differences, and the
  straight-through phase-search gradient follows a common rotation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import (
    EDFA,
    ConstellationConfig,
    Fiber,
    Link,
    OFDMConfig,
    OFDMReceiver,
    Simulation,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)
from hpcom._pulse import matched_filter_samples
from hpcom.dsp import AdaptiveEqualizer, BlindPhaseSearch, adaptive_equalize, blind_phase_search
from hpcom.metrics import bit_error_rate, snr_db
from hpcom.pilots import PilotConfig

RATE = 32e9


def _wiener(rng: np.random.Generator, n: int, variance: float) -> np.ndarray:
    """Wiener (laser) phase: increments of variance ``2π Δν dt``."""
    return np.cumsum(rng.normal(0.0, math.sqrt(variance), n))


def _awgn(rng: np.random.Generator, x: np.ndarray, snr: float, sps: int) -> np.ndarray:
    """Add white noise for a per-symbol SNR (dB) after the matched filter."""
    sigma2 = np.mean(np.abs(x) ** 2) / 10 ** (snr / 10) * sps
    return x + math.sqrt(sigma2 / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_defaults_leave_the_receiver_chain_unchanged():
    rx = SingleCarrierReceiver(modulation_order=16)
    assert rx.cpr is None and rx.equalize == "auto" and rx.adaptive == AdaptiveEqualizer()
    grid = BlindPhaseSearch(test_phases=4).test_phase_grid()
    np.testing.assert_allclose(grid, [-math.pi / 4, -math.pi / 8, 0.0, math.pi / 8])


def test_adaptive_equalisers_need_a_pulse_shaped_receiver():
    ofdm = OFDMConfig(n_subcarriers=64, cp_len=8)
    for mode in ("cma", "lms"):
        with pytest.raises(ValidationError, match="pulse-shaped receivers"):
            OFDMReceiver(modulation_order=4, ofdm=ofdm, equalize=mode)


# ---------------------------------------------------------------------------
# Blind phase search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "linewidth_symbol", "snr", "cpr", "pilots"),
    [
        (4, 1e-3, 14.0, BlindPhaseSearch(half_window=4), None),
        (16, 1e-4, 20.0, BlindPhaseSearch(half_window=6), None),
        (16, 1e-4, 20.0, BlindPhaseSearch(half_window=6), PilotConfig(period=32)),
        (64, 1e-5, 26.0, BlindPhaseSearch(test_phases=64), None),
    ],
    ids=["qpsk-1e-3", "16qam-1e-4", "16qam-1e-4-pilots", "64qam-1e-5"],
)
def test_blind_phase_search_tracks_laser_phase_noise(order, linewidth_symbol, snr, cpr, pilots):
    """Δν·T_s up to 1e-3 (QPSK), 1e-4 (16-QAM): BPS restores the SNR without phase noise.

    One complex gain per block cannot follow the random walk (tens of dB
    lost); BPS stays within 0.5 dB, without cycle slips (BER ≤ 1e-3).
    64-QAM needs 64 test phases (Pfau et al.): with 32 the spacing alone
    costs ≈0.4 dB at 26 dB SNR.
    """
    sps = 4
    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=RATE,
        n_symbols=16384,
        samples_per_symbol=sps,
        pilots=pilots,
        seed=0,
    )
    sent = tx.generate()
    rng = np.random.default_rng(1)
    clean = _awgn(rng, np.asarray(sent.field), snr, sps)
    phase = _wiener(rng, clean.size, 2 * math.pi * linewidth_symbol / sps)
    noisy = clean * np.exp(1j * phase)
    ref = tx.matched_receiver().reference(sent)[1]
    bps = tx.matched_receiver(cpr=cpr)
    snr_clean = snr_db(tx.matched_receiver().receive(clean, sent.dt_s, reference=sent).symbols, ref)
    snr_gain = snr_db(tx.matched_receiver().receive(noisy, sent.dt_s, reference=sent).symbols, ref)
    out = bps.receive(noisy, sent.dt_s, reference=sent)
    assert snr_gain < snr_clean - 10.0
    assert abs(snr_db(out.symbols, ref) - snr_clean) < 0.5
    assert bit_error_rate(out.bits, sent.bits)[0] <= 1e-3
    assert out.equalizer == ("pilot" if pilots else "data_aided")


@pytest.mark.parametrize(
    ("order", "snr", "linewidth_symbol", "half_window"),
    [
        (16, 20.0, 0.0, 8),
        (16, 20.0, 1e-4, 8),
        (4, 14.0, 1e-3, 4),
        (64, 28.0, 1e-5, 16),
    ],
)
def test_blind_phase_search_error_follows_the_estimate(order, snr, linewidth_symbol, half_window):
    r"""RMS phase error vs :math:`1/(2\,\mathrm{SNR}(2N+1)) + \sigma^2N(N+1)/(3(2N+1)) + \delta^2/12`.

    The window average of :math:`2N+1` symbols (additive noise), the phase
    walk inside the window (:math:`\sigma^2 = 2\pi\Delta\nu T_s`) and the
    test-phase spacing :math:`\delta`; within a factor 1.5 (measured
    0.90–1.01).
    """
    n, cfg = 16384, BlindPhaseSearch(half_window=half_window)
    rng = np.random.default_rng(2)
    points = ConstellationConfig.from_square_qam(order).points
    theta = 0.3 + _wiener(rng, n, 2 * math.pi * linewidth_symbol)
    r = _awgn(rng, points[rng.integers(0, order, n)] * np.exp(1j * theta), snr, 1)
    _, estimate = blind_phase_search(r, points, cfg)
    err = estimate - theta
    period = 2 * math.pi / cfg.symmetry
    err -= period * np.round(np.mean(err) / period)  # the π/2 ambiguity
    two_n = 2 * half_window + 1
    sigma2 = 2 * math.pi * linewidth_symbol
    step = period / cfg.test_phases
    theory = math.sqrt(
        1 / (2 * 10 ** (snr / 10) * two_n)
        + sigma2 * half_window * (half_window + 1) / (3 * two_n)
        + step**2 / 12
    )
    rms = float(np.sqrt(np.mean(err**2)))
    assert 1 / 1.5 < rms / theory < 1.5
    assert np.max(np.abs(err)) < period / 2  # no cycle slip


# ---------------------------------------------------------------------------
# Adaptive equalisers
# ---------------------------------------------------------------------------

_PMD_FIBER = Fiber.ssmf(80.0, n_steps=20, polarization="dual").with_pmd(0.2, sections=50)
_PMD_LINK = Link.from_spans(10, _PMD_FIBER, EDFA(gain_db=_PMD_FIBER.loss_db, noise_figure_db=5.0))


def _rotated_back_to_back(order: int, pilots: PilotConfig | None, seed: int):
    """Dual-pol signal through a Jones matrix close to a polarisation swap, 5 ps DGD, 20 dB SNR."""
    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=RATE,
        n_symbols=8192,
        polarization="dual",
        samples_per_symbol=4,
        pilots=pilots,
        seed=seed,
    )
    sent = tx.generate()
    omega = 2 * math.pi * np.fft.fftfreq(tx.n_samples, d=sent.dt_s)
    dgd = np.exp(0.5j * 5e-12 * np.outer([1.0, -1.0], omega))
    c, s = math.cos(1.2), math.sin(1.2)  # |sin| > |cos|: CMA locks X onto Y
    u = np.array([[c, -s * np.exp(-0.4j)], [s * np.exp(0.4j), c]]) * np.exp(0.7j)
    spectrum = u @ (dgd * np.fft.fft(sent.field, axis=-1))
    field = _awgn(np.random.default_rng(seed + 1), np.fft.ifft(spectrum, axis=-1), 20.0, 4)
    return tx, sent, field, tx.matched_receiver()


@pytest.fixture(scope="module")
def pmd_runs() -> dict[int, tuple]:
    out = {}
    for order in (4, 16):
        tx = SingleCarrierTransmitter(
            modulation_order=order,
            symbol_rate_hz=RATE,
            n_symbols=8192,
            polarization="dual",
            launch_power_per_channel_dbm=-6.0,
            seed=1,
        )
        cd = _PMD_LINK.total_dispersion_s2
        sim = Simulation(
            transmitter=tx,
            link=_PMD_LINK,
            receiver=tx.matched_receiver(cd_compensation_s2=cd),
            seed=2,
        )
        out[order] = (tx, sim, sim.run(), cd)
    return out


@pytest.mark.parametrize(
    ("scenario", "order", "tolerance_db"),
    [
        ("pmd", 4, 0.5),
        ("pmd", 16, 1.5),
        ("swap", 4, 0.5),
        ("swap", 16, 2.5),
        ("swap-pilots", 16, 2.5),
    ],
)
def test_cma_undoes_polarisation_rotation_and_pmd_blindly(pmd_runs, scenario, order, tolerance_db):
    """CMA reaches the least-squares butterfly FIR within ``tolerance_db``, never seeing data.

    The 16-QAM gap is CMA's misadjustment: its error is not zero on a
    multi-ring constellation, so the gap grows with the SNR (≈1 dB at the
    PMD link's 17 dB, ≈2 dB at 20 dB; it shrinks with the step size). The
    transmitted data only order the outputs after convergence: for a
    rotation past 45° CMA delivers Y on output X.
    """
    if scenario == "pmd":
        tx, sim, plain, cd = pmd_runs[order]

        def score(**kw):
            return sim.evaluate(plain, tx.matched_receiver(cd_compensation_s2=cd, **kw)).metrics

        snr_plain = plain.metrics.snr_db
        snr_mimo, snr_cma = score(equalize="mimo").snr_db, score(equalize="cma").snr_db
    else:
        pilots = PilotConfig(period=16) if scenario.endswith("pilots") else None
        tx, sent, field, rx = _rotated_back_to_back(order, pilots, seed=3)
        ref = rx.reference(sent)[1]

        def symbols(**kw):
            out = tx.matched_receiver(**kw).receive(field, sent.dt_s, reference=sent)
            return out.symbols, out.equalizer

        snr_plain = snr_db(symbols()[0], ref)
        snr_mimo = snr_db(symbols(equalize="mimo")[0], ref)
        cma, equalizer = symbols(equalize="cma")
        snr_cma = snr_db(cma, ref)
        assert equalizer == "cma"
    assert snr_plain < snr_mimo - 5.0
    assert snr_mimo - tolerance_db < snr_cma <= snr_mimo + 0.05


def test_cma_without_data_equals_the_identified_output_up_to_the_mode_order():
    """The reference only permutes CMA's outputs (here X ↔ Y): the taps never see it."""
    tx, sent, field, rx = _rotated_back_to_back(16, None, seed=4)
    two = matched_filter_samples(field, 4, rx.roll_off, None, "circular")[..., ::2]
    points = tx.resolved_constellation.points
    cfg = AdaptiveEqualizer()
    blind = adaptive_equalize(two, 2, 15, (0,), "cma", cfg, points)
    known = adaptive_equalize(two, 2, 15, (0,), "cma", cfg, points, targets=sent.data_symbols())
    np.testing.assert_array_equal(blind[::-1], known)


@pytest.mark.parametrize("reference", ["decisions", "data"])
def test_lms_improves_on_cma_for_16qam(pmd_runs, reference):
    """CMA pre-convergence, then LMS: within 0.5 dB of least squares, ≥ 0.3 dB above CMA."""
    tx, sim, plain, cd = pmd_runs[16]
    lms_rx = tx.matched_receiver(
        cd_compensation_s2=cd, equalize="lms", adaptive=AdaptiveEqualizer(lms_reference=reference)
    )
    lms = sim.evaluate(plain, lms_rx)
    cma = sim.evaluate(plain, tx.matched_receiver(cd_compensation_s2=cd, equalize="cma"))
    mimo = sim.evaluate(plain, tx.matched_receiver(cd_compensation_s2=cd, equalize="mimo"))
    assert lms.received.equalizer == "lms"
    assert lms.metrics.snr_db > cma.metrics.snr_db + 0.3
    assert lms.metrics.snr_db > mimo.metrics.snr_db - 0.5


def test_data_aided_lms_needs_the_transmitted_data():
    tx, sent, field, _ = _rotated_back_to_back(4, None, seed=5)
    rx = tx.matched_receiver(equalize="lms", adaptive=AdaptiveEqualizer(lms_reference="data"))
    with pytest.raises(ValueError, match="needs the transmitted symbols"):
        rx.receive(field, sent.dt_s)


@pytest.mark.parametrize(
    ("equalize", "reference"), [("cma", "decisions"), ("lms", "decisions"), ("lms", "data")]
)
def test_carrier_recovery_after_the_adaptive_equaliser(equalize, reference):
    """Real-receiver chain on rotation + DGD + laser phase noise (Δν·T_s = 1e-4), 16-QAM.

    CMA is phase-blind; LMS keeps the BPS phase in its loop (its taps would
    otherwise chase the laser phase). With carrier recovery both come within
    0.5 dB of their phase-noise-free SNR; without it the SNR collapses.
    """
    tx, sent, field, _ = _rotated_back_to_back(16, None, seed=6)
    rng = np.random.default_rng(7)
    noisy = field * np.exp(1j * _wiener(rng, field.shape[-1], 2 * math.pi * 1e-4 / 4))
    ref = tx.matched_receiver().reference(sent)[1]
    adaptive = AdaptiveEqualizer(lms_reference=reference)

    def snr(field_, cpr=None):
        rx = tx.matched_receiver(equalize=equalize, adaptive=adaptive, cpr=cpr)
        return snr_db(rx.receive(field_, sent.dt_s, reference=sent).symbols, ref)

    without = snr(field)
    assert snr(noisy) < without - 5.0
    assert snr(noisy, BlindPhaseSearch(half_window=6)) > without - 0.5


# ---------------------------------------------------------------------------
# Differentiable versions
# ---------------------------------------------------------------------------


@pytest.fixture
def _x64() -> Iterator[None]:
    jax = pytest.importorskip("jax")
    with jax.enable_x64(True):
        yield


_DSP_RECEIVERS = {
    "cma": {"equalize": "cma"},
    "lms": {"equalize": "lms"},
    "lms-data": {"equalize": "lms", "adaptive": AdaptiveEqualizer(lms_reference="data")},
    "bps": {"cpr": BlindPhaseSearch(half_window=8)},
    "lms-bps": {"equalize": "lms", "cpr": BlindPhaseSearch()},
    "lms-data-bps": {
        "equalize": "lms",
        "adaptive": AdaptiveEqualizer(lms_reference="data"),
        "cpr": BlindPhaseSearch(),
    },
    "mimo-bps": {"equalize": "mimo", "cpr": BlindPhaseSearch()},
}


@pytest.mark.usefixtures("_x64")
@pytest.mark.parametrize("pilots", [None, PilotConfig(period=8)], ids=["", "pilots"])
@pytest.mark.parametrize("name", list(_DSP_RECEIVERS))
def test_demodulate_equals_the_receiver(name, pilots):
    """Same samples as the NumPy receiver: ≈1e-15 (adaptive, BPS); ≈1e-8 with "mimo" (lstsq)."""
    jnp = pytest.importorskip("jax.numpy")
    from hpcom import autodiff as ad

    fiber = Fiber.ssmf(80.0, n_steps=20, polarization="dual").with_pmd(0.2, sections=40)
    link = Link.from_spans(2, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=RATE,
        n_symbols=1024,
        polarization="dual",
        launch_power_per_channel_dbm=-4.0,
        pilots=pilots,
    )
    sig = tx.generate(rng=np.random.default_rng(7))
    y = link.propagate(sig.field, tx.dt_s, rng=np.random.default_rng(8))
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, **_DSP_RECEIVERS[name])
    ref = rx.reference(sig)[1]
    expected = rx.receive(y, tx.dt_s, reference=sig).symbols
    out = ad.demodulate(jnp.asarray(y), rx, tx.dt_s, reference=jnp.asarray(ref))
    atol = (1e-7 if name.startswith("mimo") else 1e-12) * np.max(np.abs(expected))
    np.testing.assert_allclose(np.asarray(out), expected, rtol=0, atol=atol)


@pytest.mark.usefixtures("_x64")
def test_demodulate_equals_the_wdm_receiver_per_channel_group():
    jnp = pytest.importorskip("jax.numpy")
    from hpcom import autodiff as ad

    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=RATE,
        n_symbols=512,
        polarization="dual",
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    sig = tx.generate(rng=np.random.default_rng(1))
    y = _awgn(np.random.default_rng(2), np.asarray(sig.field), 25.0, tx.samples_per_symbol)
    for kw in ({"equalize": "lms"}, {"equalize": "cma", "cpr": BlindPhaseSearch()}):
        rx = tx.matched_receiver(channels=(0, 2), **kw)
        ref = rx.reference(sig)[1]
        expected = rx.receive(y, tx.dt_s, reference=sig).symbols
        out = ad.demodulate(jnp.asarray(y), rx, tx.dt_s, reference=jnp.asarray(ref))
        np.testing.assert_allclose(np.asarray(out), expected, rtol=0, atol=1e-12)


def _gmi_through_pmd(name: str):
    """GMI(PMD coefficient, launch power) through a 2-span PMD link and the receiver ``name``."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hpcom import autodiff as ad

    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=RATE,
        n_symbols=512,
        samples_per_symbol=4,
        polarization="dual",
    )
    fiber = Fiber.ssmf(80.0, n_steps=10, polarization="dual").with_pmd(0.3, sections=20)
    link = Link.from_spans(2, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    rx = tx.matched_receiver(
        cd_compensation_s2=link.total_dispersion_s2, mimo_taps=7, **_DSP_RECEIVERS[name]
    )
    idx = jnp.asarray(np.random.default_rng(9).integers(0, 16, (2, 512)))
    labels = tx.resolved_constellation.bit_labels

    def gmi(pmd, power):
        points, pmf = ad.constellation(tx)
        lp = tuple(
            q._replace(pmd_ps_per_sqrt_km=pmd) if isinstance(q, ad.FiberParameters) else q
            for q in ad.link_parameters(link)
        )
        x = ad.modulate(points[idx], tx, launch_power_dbm=power)
        y = ad.propagate_link(x, link, tx.dt_s, key=jax.random.key(3), params=lp)
        z = ad.demodulate(y, rx, tx.dt_s, reference=points[idx])
        return ad.information_rates(z, idx, points, labels, pmf)[1]

    return jax.jit(gmi)


@pytest.mark.usefixtures("_x64")
@pytest.mark.parametrize("name", ["cma", "lms", "lms-data"])
def test_gradient_through_the_adaptive_equaliser_equals_finite_differences(name):
    """d GMI / d(PMD coefficient, launch power) through a PMD link and every tap update.

    Decisions and the mode order are piecewise constant, so the gradient
    is exact almost everywhere.
    """
    jax = pytest.importorskip("jax")
    f = _gmi_through_pmd(name)
    g_pmd, g_power = jax.grad(f, argnums=(0, 1))(0.3, 8.0)
    h = 1e-5
    fd_pmd = (float(f(0.3 + h, 8.0)) - float(f(0.3 - h, 8.0))) / (2 * h)
    fd_power = (float(f(0.3, 8.0 + h)) - float(f(0.3, 8.0 - h))) / (2 * h)
    assert float(g_pmd) == pytest.approx(fd_pmd, rel=1e-4, abs=1e-8)
    assert float(g_power) == pytest.approx(fd_power, rel=1e-4, abs=1e-8)


@pytest.mark.usefixtures("_x64")
@pytest.mark.parametrize("name", ["bps", "lms-bps", "mimo-bps"])
def test_gradient_through_phase_search_is_finite(name):
    """With BPS the gradient is the straight-through one: finite, not a finite difference."""
    jax = pytest.importorskip("jax")
    grads = jax.grad(_gmi_through_pmd(name), argnums=(0, 1))(0.3, 8.0)
    assert all(np.isfinite(float(g)) and float(g) != 0.0 for g in grads)


@pytest.mark.usefixtures("_x64")
@pytest.mark.parametrize("half_window", [4, 16])
def test_phase_search_gradient_follows_a_common_rotation(half_window):
    r"""Straight-through BPS: :math:`d\hat\theta/d\alpha \approx 1` for an input rotated by α.

    The hard pick alone is piecewise constant (its gradient is 0, as
    finite differences see it); the soft relaxation tracks the rotation, so
    the derotated output is (to first order) insensitive to it.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hpcom.autodiff._transceiver import _blind_phase_search

    rng = np.random.default_rng(1)
    points = ConstellationConfig.from_square_qam(16).points
    a = points[rng.integers(0, 16, 4096)]
    r = jnp.asarray(_awgn(rng, a * np.exp(0.2j), 20.0, 1))
    rx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=RATE, n_symbols=4096
    ).matched_receiver(cpr=BlindPhaseSearch(half_window=half_window))

    def quadrature(alpha):
        return jnp.sum(jnp.imag(jnp.conj(a) * _blind_phase_search(rx, r * jnp.exp(1j * alpha))))

    out = np.asarray(_blind_phase_search(rx, r))
    np.testing.assert_allclose(out, blind_phase_search(r, points, rx.cpr)[0], atol=1e-13)
    frozen = np.sum(np.real(np.conj(a) * out))  # d quadrature / dα at a fixed phase
    assert abs(float(jax.grad(quadrature)(0.0)) / frozen) < 0.01
