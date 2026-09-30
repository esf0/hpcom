"""PMD at the system level: configuration, links, DBP and the MIMO equaliser.

Backend-level physics (Maxwellian DGD, principal states, invariance, orders)
is in ``tests/backends/test_pmd_conformance.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import (
    EDFA,
    DigitalBackPropagation,
    Fiber,
    Link,
    Scheme,
    Simulation,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)
from hpcom.pilots import PilotConfig
from hpcom.receiver import mimo_equalize

PMD_FIBER = Fiber.ssmf(80.0, n_steps=20, polarization="dual").with_pmd(0.2, sections=50)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"polarization": "single"}, "polarization='dual'"),
        ({"scheme": Scheme(method="ssfm-local-error", local_error=1e-5)}, "fixed-step"),
        ({"pmd_ps_per_sqrt_km": -0.1}, "greater than or equal"),
    ],
)
def test_invalid_pmd_fibers_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        Fiber(**{"length_km": 10.0, "polarization": "dual", "pmd_ps_per_sqrt_km": 0.1, **kwargs})


def test_mean_dgd_and_recentring():
    assert PMD_FIBER.mean_dgd_s == pytest.approx(0.2e-12 * np.sqrt(80.0))
    assert PMD_FIBER.recentred(0.0) is PMD_FIBER
    with pytest.raises(ValueError, match="cannot be recentred"):
        PMD_FIBER.recentred(50e9)


def test_link_spans_get_independent_reproducible_realisations():
    link = Link.from_spans(5, PMD_FIBER, EDFA(gain_db=16.0))
    seeds = [el.pmd_seed for el in link.elements if isinstance(el, Fiber)]
    assert len(set(seeds)) == 5
    assert seeds == [
        el.pmd_seed
        for el in Link.from_spans(5, PMD_FIBER, EDFA(gain_db=16.0)).elements
        if isinstance(el, Fiber)
    ]
    no_pmd = Link.from_spans(3, PMD_FIBER.replace(pmd_ps_per_sqrt_km=0.0), EDFA(gain_db=16.0))
    assert no_pmd.elements[0] is no_pmd.elements[2]


def test_dbp_ignores_pmd_unless_told_and_then_inverts_exactly():
    link = Link.from_spans(3, PMD_FIBER, EDFA(gain_db=PMD_FIBER.loss_db, noise_figure_db=None))
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 2048)) + 1j * rng.standard_normal((2, 2048))) * 0.05
    y = link.propagate(x, 1e-11)
    ideal = DigitalBackPropagation(
        link=link, steps_per_span=None, samples_per_symbol=None, include_pmd=True
    )
    np.testing.assert_allclose(ideal.compensate(y, 1e-11), x, atol=1e-10 * np.max(np.abs(x)))
    realistic = DigitalBackPropagation(link=link, steps_per_span=None, samples_per_symbol=None)
    assert np.max(np.abs(realistic.compensate(y, 1e-11) - x)) > 0.1 * np.max(np.abs(x))
    assert realistic._backward_fiber(PMD_FIBER, 0.0).pmd_ps_per_sqrt_km == 0.0


# ---------------------------------------------------------------------------
# MIMO equaliser
# ---------------------------------------------------------------------------


def test_least_squares_butterfly_inverts_a_known_2x2_fir():
    """A random 2×2, 3-tap channel at 2 samples/symbol is undone to rounding by 7 taps... of its inverse."""
    rng = np.random.default_rng(1)
    n = 512
    s = (rng.choice([-1, 1], (2, n)) + 1j * rng.choice([-1, 1], (2, n))) / np.sqrt(2)
    up = np.zeros((2, 2 * n), complex)
    up[:, ::2] = s
    up[:, 1::2] = (s + np.roll(s, -1, axis=-1)) / 2  # a smooth 2-samples/symbol signal
    # a unitary polarisation rotation and a half-symbol differential delay (circular)
    theta, phi = 0.7, 1.3
    rot = np.array(
        [
            [np.cos(theta), -np.exp(-1j * phi) * np.sin(theta)],
            [np.exp(1j * phi) * np.sin(theta), np.cos(theta)],
        ]
    )
    mixed = rot @ up
    mixed[1] = np.roll(mixed[1], 1)
    out = mimo_equalize(mixed, 2, 9, (0,), np.arange(n), s)
    np.testing.assert_allclose(out, s, atol=1e-9)


def _pmd_link(n_spans: int = 10, noise_figure_db: float | None = None) -> Link:
    return Link.from_spans(
        n_spans, PMD_FIBER, EDFA(gain_db=PMD_FIBER.loss_db, noise_figure_db=noise_figure_db)
    )


@pytest.mark.parametrize(
    "tx",
    [
        SingleCarrierTransmitter(
            modulation_order=16,
            symbol_rate_hz=32e9,
            n_symbols=4096,
            polarization="dual",
            launch_power_per_channel_dbm=-10.0,
            seed=1,
        ),
        WDMTransmitter(
            modulation_order=16,
            symbol_rate_hz=32e9,
            n_symbols=4096,
            polarization="dual",
            launch_power_per_channel_dbm=-10.0,
            wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
            seed=1,
        ),
    ],
    ids=["single-carrier", "wdm"],
)
def test_mimo_equaliser_undoes_pmd_that_per_stream_gains_cannot(tx):
    link = _pmd_link()
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
        seed=2,
    )
    plain = sim.run()
    mimo = sim.evaluate(
        plain, tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, equalize="mimo")
    )
    assert plain.metrics.snr_db < 15.0 and plain.metrics.ber > 1e-2
    assert mimo.metrics.snr_db > 40.0 and mimo.metrics.ber == 0.0
    assert mimo.received.equalizer == "mimo"


def test_mimo_after_dbp_at_two_samples_per_symbol():
    """Realistic chain: DBP without PMD at 2 sps, then the MIMO equaliser removes the PMD."""
    link = _pmd_link(5)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=4096,
        polarization="dual",
        launch_power_per_channel_dbm=4.0,
        seed=3,
    )
    dbp = DigitalBackPropagation(link=link, steps_per_span=8, samples_per_symbol=2)
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, equalize="mimo"),
        seed=4,
    )
    cdc = sim.run()
    with_dbp = sim.evaluate(cdc, tx.matched_receiver(compensation=dbp, equalize="mimo"))
    assert with_dbp.metrics.snr_db > cdc.metrics.snr_db + 3.0


def test_mimo_with_pilots_trains_on_the_data_and_equalises_the_pilots():
    link = _pmd_link(3)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=2048,
        polarization="dual",
        pilots=PilotConfig(period=4),
        launch_power_per_channel_dbm=-10.0,
        seed=5,
    )
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, equalize="mimo")
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=6).run()
    assert result.metrics.snr_db > 40.0
    np.testing.assert_allclose(result.received.pilot_symbols, 1.0, atol=0.02)
    with pytest.raises(ValueError, match="trained on the transmitted data"):
        rx.receive(link.propagate(tx.generate().field, tx.dt_s), tx.dt_s)
    with pytest.raises(ValidationError, match="odd"):
        tx.matched_receiver(equalize="mimo", mimo_taps=8)
