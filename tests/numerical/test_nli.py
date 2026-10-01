"""Nonlinear-interference models (:mod:`hpcom.nli`) against brute force, the literature and SSFM.

* the GN integral, reduced to one dimension in w = (f₁ − f)(f₂ − f), equals
  the brute-force double integral;
* it reproduces the published GN curve of Poggiolini et al. 2015 (Fig. 1a);
* the first-order perturbation of the actual signal equals the GN model for
  Gaussian symbols and the split-step simulation for QAM, independently of
  the launch power;
* the GN model overestimates the NLI of QAM, and the closed-form EGN
  correction removes most of that;
* the predicted SNR with ASE matches the simulation around the optimum.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpcom import (
    EDFA,
    Fiber,
    Link,
    Simulation,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
    nli,
)
from hpcom.metrics import snr_db

SSMF = Fiber.ssmf(80.0, n_steps=100, polarization="dual").with_steps(
    distribution="logarithmic", log_factor=1 / 3
)


def _link(spans: int, fiber: Fiber = SSMF, noise_figure_db: float | None = None) -> Link:
    return Link.from_spans(
        spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=noise_figure_db)
    )


def _wdm(order: int = 16, **kwargs) -> WDMTransmitter:
    fields = {
        "modulation_order": order,
        "symbol_rate_hz": 32e9,
        "n_symbols": 4096,
        "polarization": "dual",
        "samples_per_symbol": 5,
        "wdm": WDMConfig(n_channels=3, spacing_hz=50e9),
        **kwargs,
    }
    return WDMTransmitter(**fields)


def _db(x: float) -> float:
    return 10.0 * math.log10(x)


@pytest.mark.parametrize("spans", [1, 3])
@pytest.mark.parametrize("f_hz", [0.0, 11e9])
def test_gn_integral_equals_the_brute_force_double_integral(spans, f_hz):
    link, tx = _link(spans), _wdm()
    prof, channels = nli._profile(link), nli._channels(tx)
    u = np.linspace(-170e9, 170e9, 2500)
    du = u[1] - u[0]
    total = 0.0
    for chunk in np.array_split(np.arange(u.size), 10):
        uu, vv = np.meshgrid(u[chunk], u, indexing="ij")
        kernel = np.abs(nli._span_terms(prof, (uu * vv).ravel()).sum(axis=0)) ** 2
        g = channels.psd(f_hz + uu) * channels.psd(f_hz + vv) * channels.psd(f_hz + uu + vv)
        total += float(np.sum(g.ravel() * kernel)) * du * du
    brute = 16 / 27 * total
    assert nli.gn_psd(link, tx, [f_hz])[0] == pytest.approx(brute, rel=3e-3)


def test_gn_reproduces_the_published_curve():
    """Poggiolini et al. 2015, Fig. 1(a): 3 × 32 GBd PM-QPSK, roll-off 0.05, 33.6 GHz grid,
    100 km SMF spans (D = 16.7 ps/(nm·km), γ = 1.3 /W/km, 0.22 dB/km); values read off
    the plot: GN ≈ 26.8 dB (1 span), 45.3 dB (50 spans); approximate EGN ≈ 44.0 dB (50)."""
    fiber = Fiber.ssmf(100.0, dispersion_ps_per_nm_per_km=16.7, polarization="dual").replace(
        alpha_db_per_km=0.22, gamma_per_w_per_km=1.3
    )
    tx = _wdm(
        4,
        roll_off=0.05,
        n_symbols=1000,
        samples_per_symbol=6,
        wdm=WDMConfig(n_channels=3, spacing_hz=33.6e9),
    )
    assert _db(nli.nli_coefficient(_link(1, fiber), tx)) == pytest.approx(26.8, abs=0.4)
    fifty = _link(50, fiber)
    assert _db(nli.nli_coefficient(fifty, tx)) == pytest.approx(45.3, abs=0.4)
    assert _db(nli.nli_coefficient(fifty, tx, model="egn")) == pytest.approx(44.0, abs=0.4)


def test_one_span_is_coherent_and_incoherent_alike_and_spans_add_up():
    tx = _wdm()
    one = nli.nli_coefficient(_link(1), tx)
    assert nli.nli_coefficient(_link(1), tx, coherent=False) == pytest.approx(one, rel=1e-9)
    five = nli.nli_coefficient(_link(5), tx, coherent=False)
    assert five == pytest.approx(5 * one, rel=1e-3)  # identical spans: N× the NLI power
    assert nli.nli_coefficient(_link(5), tx) > 1.1 * five  # coherent accumulation


@pytest.mark.parametrize(
    ("tx", "spans"),
    [
        (
            SingleCarrierTransmitter(
                modulation_order=16,
                symbol_rate_hz=32e9,
                n_symbols=4096,
                polarization="dual",
                samples_per_symbol=4,
            ),
            3,
        ),
        (_wdm(), 2),
    ],
    ids=["single-channel-3-spans", "wdm-2-spans"],
)
def test_first_order_perturbation_of_gaussian_symbols_equals_gn(tx, spans):
    link = _link(spans)
    estimate = nli.first_order_nli(link, tx, n_blocks=6, gaussian=True, nodes_per_fiber=96)
    gn = nli.nli_coefficient(link, tx)
    assert _db(estimate.eta_per_w2) == pytest.approx(_db(gn), abs=0.2)
    assert estimate.eta_stderr_per_w2 < 0.05 * estimate.eta_per_w2


def test_first_order_estimate_does_not_depend_on_the_launch_power():
    """At 0 dBm the mean nonlinear rotation of 3 channels over 5 spans is ≈0.6 rad: scored at
    the launch power, the first-order field (1 + iφ instead of e^{iφ}) would read ≈1 dB low."""
    link = _link(5)
    etas = [
        nli.first_order_nli(link, _wdm(launch_power_per_channel_dbm=p), n_blocks=2).eta_per_w2
        for p in (-10.0, 0.0)
    ]
    assert etas[1] == pytest.approx(etas[0], rel=1e-6)


@pytest.mark.parametrize("order", [4, 16])
def test_first_order_estimate_equals_the_split_step_simulation_for_qam(order):
    """On the same data blocks (η itself varies by ~0.3 dB from block to block)."""
    link = _link(3)
    tx = _wdm(order, launch_power_per_channel_dbm=-3.0)
    estimate = nli.first_order_nli(link, tx, n_blocks=2, nodes_per_fiber=96)
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, channels=(1,))
    etas = []
    for b in range(2):  # the blocks of first_order_nli(seed=0)
        sent = tx.generate(rng=np.random.default_rng([0, b]))
        y = rx.receive(link.propagate(sent.field, sent.dt_s), sent.dt_s, reference=sent).symbols
        x = rx.reference(sent)[1]
        snr = snr_db(np.asarray(y).ravel(), np.asarray(x).ravel())
        etas.append(10 ** (-snr / 10) / tx.launch_power_per_channel_w**2)
    assert _db(estimate.eta_per_w2) == pytest.approx(_db(float(np.mean(etas))), abs=0.06)


@pytest.mark.parametrize(("order", "phi"), [(4, 1.0), (16, 17 / 25), (64, 13 / 21)])
def test_gn_overestimates_qam_and_the_egn_correction_removes_most_of_it(order, phi):
    link, tx = _link(5), _wdm(order)
    assert nli.modulation_phi(tx) == pytest.approx(phi, abs=1e-12)
    truth = _db(nli.first_order_nli(link, tx, n_blocks=4, nodes_per_fiber=96).eta_per_w2)
    gn = _db(nli.nli_coefficient(link, tx))
    egn = _db(nli.nli_coefficient(link, tx, model="egn"))
    assert gn - truth > 1.0
    assert abs(egn - truth) < 0.35


def test_predicted_snr_matches_the_simulation_with_ase():
    link = _link(5, noise_figure_db=5.0)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=4096,
        polarization="dual",
        samples_per_symbol=4,
    )
    eta = nli.first_order_nli(link, tx, n_blocks=4).eta_per_w2
    best = nli.predict_snr(link, tx, eta_per_w2=eta).optimal_launch_power_dbm
    for offset in (-3.0, 0.0, 3.0):
        txp = tx.replace(launch_power_per_channel_dbm=round(best + offset, 2))
        predicted = nli.predict_snr(link, txp, eta_per_w2=eta).snr_db
        rx = txp.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
        measured = np.mean(
            [
                Simulation(transmitter=txp, link=link, receiver=rx, seed=s).run().metrics.snr_db
                for s in range(3)
            ]
        )
        assert measured == pytest.approx(predicted, abs=0.3), offset
