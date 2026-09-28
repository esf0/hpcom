"""Digital back-propagation in the receiver: exact limits, convergence, physics.

* γ = 0 DBP at any front-end rate is exactly the bulk CD compensation.
* Full-field DBP with the forward step count inverts a noiseless link.
* Practical DBP converges with steps per span to a floor set by the
  front-end rate (spectral broadening outside the sampled band is lost):
  ≈ 50 dB at 2 samples/symbol, far lower errors at 4, and at 1 sample/symbol
  the roll-off band itself is lost before DBP.
* WDM: the more bandwidth is back-propagated, the more nonlinearity is
  undone (CDC < single-channel < 3-channel < 5-channel < full field); wide
  DBP needs short steps (inter-channel walk-off).
* With ASE, DBP recovers the SNR up to — never beyond — the ASE limit.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from helpers import PULSE_MODES
from hpcom import (
    EDFA,
    DigitalBackPropagation,
    Fiber,
    Link,
    PilotConfig,
    Simulation,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)


def _snr_db(y, x) -> float:
    y, x = np.asarray(y), np.asarray(x)
    return float(10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(y - x) ** 2)))


def _link(n_spans: int = 5, noise_figure_db: float | None = None, beta3: float = 0.0) -> Link:
    fiber = Fiber.ssmf(length_km=80, n_steps=50, polarization="dual").replace(beta3_s3_per_m=beta3)
    return Link.from_spans(
        n_spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=noise_figure_db)
    )


def _dbp(link: Link, **kwargs) -> DigitalBackPropagation:
    kwargs.setdefault("steps_per_span", None)
    return DigitalBackPropagation(link=link, **kwargs)


# ---------------------------------------------------------------------------
# Exact limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "front_end",
    [
        pytest.param({"samples_per_symbol": 2}, id="2sps"),
        pytest.param({"samples_per_symbol": 3}, id="3sps"),
        pytest.param({"samples_per_symbol": 4}, id="4sps"),
        pytest.param({"samples_per_symbol": None}, id="full-rate"),
        pytest.param({"samples_per_symbol": 2, "bandwidth_hz": 1.1 * 32e9}, id="2sps-1ch-band"),
    ],
)
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_linear_dbp_is_exactly_cd_compensation(front_end, n_channels, polarization):
    fiber = Fiber.ssmf(length_km=80, n_steps=10, polarization=polarization).replace(
        beta3_s3_per_m=1.2e-40, gamma_per_w_per_km=1.27
    )
    link = Link.from_spans(3, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
    common = dict(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=512,
        polarization=polarization,
        launch_power_per_channel_dbm=0.0,
    )
    tx = (
        SingleCarrierTransmitter(**common)
        if n_channels == 1
        else WDMTransmitter(**common, wdm=WDMConfig(n_channels=3, spacing_hz=50e9))
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    out = link.propagate(sent.field, sent.dt_s)
    cdc = tx.matched_receiver(
        equalize="none",
        cd_compensation_s2=link.total_dispersion_s2,
        cd_compensation_s3=link.total_beta3_s3,
    )
    lin = tx.matched_receiver(
        equalize="none", compensation=_dbp(link, nonlinearity_scale=0.0, **front_end)
    )
    a, b = np.asarray(cdc.receive(out, sent.dt_s).symbols), lin.receive(out, sent.dt_s).symbols
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-11 * np.max(np.abs(a)))


@pytest.mark.parametrize("n_channels", [1, 3])
def test_full_field_dbp_with_the_forward_steps_inverts_a_nonlinear_link(n_channels):
    link = _link()
    common = dict(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=1024,
        polarization="dual",
        launch_power_per_channel_dbm=6.0,
    )
    tx = (
        SingleCarrierTransmitter(**common)
        if n_channels == 1
        else WDMTransmitter(**common, wdm=WDMConfig(n_channels=3, spacing_hz=50e9))
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    out = link.propagate(sent.field, sent.dt_s)
    cdc = tx.matched_receiver(equalize="data_aided", cd_compensation_s2=link.total_dispersion_s2)
    dbp = tx.matched_receiver(
        equalize="data_aided", compensation=_dbp(link, samples_per_symbol=None)
    )
    assert _snr_db(cdc.receive(out, sent.dt_s, reference=sent).symbols, sent.symbols) < 25
    assert _snr_db(dbp.receive(out, sent.dt_s, reference=sent).symbols, sent.symbols) > 200


# ---------------------------------------------------------------------------
# Practical DBP: steps per span and front-end rate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sc_nonlinear():
    """Noiseless 5×80 km, 64-QAM 32 GBd dual-pol at 7 dBm (CDC SNR ≈ 16 dB)."""
    link = _link()
    tx = SingleCarrierTransmitter(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=2048,
        polarization="dual",
        launch_power_per_channel_dbm=7.0,
    )
    sent = tx.generate(rng=np.random.default_rng(1))
    return tx, link, sent, link.propagate(sent.field, sent.dt_s)


def _dbp_snr(case, **dbp_kwargs) -> float:
    tx, link, sent, out = case
    rx = tx.matched_receiver(equalize="data_aided", compensation=_dbp(link, **dbp_kwargs))
    return _snr_db(rx.receive(out, sent.dt_s, reference=sent).symbols, sent.symbols)


STEPS = (1, 2, 4, 8, 16, 32)


def test_dbp_at_2_and_4_sps_converges_with_steps_to_a_rate_set_floor(sc_nonlinear):
    tx, link, sent, out = sc_nonlinear
    cdc = tx.matched_receiver(equalize="data_aided", cd_compensation_s2=link.total_dispersion_s2)
    snr_cdc = _snr_db(cdc.receive(out, sent.dt_s, reference=sent).symbols, sent.symbols)
    two = [_dbp_snr(sc_nonlinear, steps_per_span=s, samples_per_symbol=2) for s in STEPS]
    four = [_dbp_snr(sc_nonlinear, steps_per_span=s, samples_per_symbol=4) for s in STEPS]
    assert snr_cdc < two[0] < two[1] < two[2] < two[3]  # ≈ 16 → 18, 24, 37, 48 dB
    assert all(a < b for a, b in itertools.pairwise(four))  # keeps improving
    # Both rates agree while the step error dominates …
    np.testing.assert_allclose(two[:3], four[:3], atol=1.0)
    # … then 2 sps saturates (broadening beyond ±R_s is lost) and 4 sps does not.
    assert abs(two[-1] - two[-2]) < 1.0 and 45 < two[-1] < 56
    assert four[-1] > two[-1] + 15


def test_one_sample_per_symbol_dbp_saturates_at_the_lost_roll_off(sc_nonlinear):
    one = [_dbp_snr(sc_nonlinear, steps_per_span=s, samples_per_symbol=1) for s in STEPS]
    two = _dbp_snr(sc_nonlinear, steps_per_span=32, samples_per_symbol=2)
    assert one[2] > one[0] + 1.5  # still undoes part of the nonlinearity …
    assert abs(one[-1] - one[3]) < 0.2 and one[-1] < 20 < two  # … but saturates low


# ---------------------------------------------------------------------------
# Step placement within each back-propagated span
# ---------------------------------------------------------------------------

CLE = {"step_distribution": "logarithmic", "step_log_factor": 1 / 3}
NLP = {"step_distribution": "logarithmic", "step_log_factor": 1.0}


@pytest.mark.parametrize("steps", [2, 4, 8])
def test_constant_local_error_placement_beats_uniform_steps(sc_nonlinear, steps):
    """k = 1/3 (short steps at the high-power end of each span): +3–6 dB at 2–8 steps/span."""
    uniform = _dbp_snr(sc_nonlinear, steps_per_span=steps, samples_per_symbol=4)
    cle = _dbp_snr(sc_nonlinear, steps_per_span=steps, samples_per_symbol=4, **CLE)
    assert cle > uniform + 2


def test_equal_nonlinear_phase_placement_over_concentrates_the_steps(sc_nonlinear):
    """k = 1 (Bosco/NLP rule): ≈ 32 dB vs uniform 38 and k = 1/3 43 dB at 4 steps/span."""
    snr = {
        name: _dbp_snr(sc_nonlinear, steps_per_span=4, samples_per_symbol=4, **kw)
        for name, kw in (("uniform", {}), ("nlp", NLP), ("cle", CLE))
    }
    assert snr["nlp"] < snr["uniform"] < snr["cle"]


def test_dbp_inverts_a_link_simulated_with_logarithmic_steps():
    fiber = Fiber.ssmf(length_km=80, n_steps=30, polarization="dual").with_steps(
        distribution="logarithmic", log_factor=1 / 3
    )
    link = Link.from_spans(3, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
    tx = SingleCarrierTransmitter(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=512,
        polarization="dual",
        launch_power_per_channel_dbm=8.0,
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    out = link.propagate(sent.field, sent.dt_s)
    exact = tx.matched_receiver(compensation=_dbp(link, samples_per_symbol=None))
    mismatched = tx.matched_receiver(
        compensation=_dbp(link, samples_per_symbol=None, step_distribution="uniform")
    )
    x = sent.symbols
    assert _snr_db(exact.receive(out, sent.dt_s, reference=sent).symbols, x) > 200
    # Another placement is an accurate (≈ 77 dB) but not an exact inverse.
    assert _snr_db(mismatched.receive(out, sent.dt_s, reference=sent).symbols, x) < 120


# ---------------------------------------------------------------------------
# WDM: DBP bandwidth
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wdm_nonlinear():
    """Noiseless 5×80 km, 5 × 32 GBd channels on 40 GHz, 4 dBm/ch; centre channel scored."""
    link = _link()
    tx = WDMTransmitter(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=2048,
        wdm=WDMConfig(n_channels=5, spacing_hz=40e9),
        polarization="dual",
        launch_power_per_channel_dbm=4.0,
    )
    sent = tx.generate(rng=np.random.default_rng(1))
    return tx, link, sent, link.propagate(sent.field, sent.dt_s)


def _centre_snr(case, **rx_kwargs) -> float:
    tx, _, sent, out = case
    rx = tx.matched_receiver(channels=(2,), equalize="data_aided", **rx_kwargs)
    y = np.asarray(rx.receive(out, sent.dt_s, reference=sent).symbols)[:, 0]
    return _snr_db(y, np.asarray(sent.symbols)[:, 2])


def test_back_propagating_more_bandwidth_undoes_more_nonlinearity(wdm_nonlinear):
    link = wdm_nonlinear[1]
    snr = {
        "cdc": _centre_snr(wdm_nonlinear, cd_compensation_s2=link.total_dispersion_s2),
        "1ch": _centre_snr(
            wdm_nonlinear,
            compensation=_dbp(link, samples_per_symbol=2, bandwidth_hz=1.1 * 32e9),
        ),
        "3ch": _centre_snr(
            wdm_nonlinear, compensation=_dbp(link, samples_per_symbol=4, bandwidth_hz=3 * 40e9)
        ),
        "5ch": _centre_snr(
            wdm_nonlinear, compensation=_dbp(link, samples_per_symbol=7, bandwidth_hz=5 * 40e9)
        ),
        "full": _centre_snr(wdm_nonlinear, compensation=_dbp(link, samples_per_symbol=None)),
    }
    # ≈ 17.7 < 20.0 < 22.3 < 27.2 dB, and the full field is the exact inverse.
    assert snr["cdc"] + 1 < snr["1ch"] < snr["3ch"] - 1 < snr["5ch"] - 3
    assert snr["full"] > 200


def test_wide_band_dbp_needs_short_steps(wdm_nonlinear):
    """Inter-channel walk-off makes coarse multi-channel DBP worse than single-channel DBP."""
    link = wdm_nonlinear[1]

    def snr(steps: int, **kw) -> float:
        return _centre_snr(wdm_nonlinear, compensation=_dbp(link, steps_per_span=steps, **kw))

    one = dict(samples_per_symbol=2, bandwidth_hz=1.1 * 32e9)
    three = dict(samples_per_symbol=4, bandwidth_hz=3 * 40e9)
    assert snr(4, **three) < snr(4, **one) < snr(50, **three)


# ---------------------------------------------------------------------------
# With ASE: paired comparison through Simulation.evaluate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("launch_dbm", [4.0, 7.0])
def test_dbp_recovers_the_snr_up_to_the_ase_limit(launch_dbm):
    link = _link(noise_figure_db=5.0)
    tx = SingleCarrierTransmitter(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=4096,
        polarization="dual",
        launch_power_per_channel_dbm=launch_dbm,
    )
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
        seed=3,
    )
    cdc = sim.run()
    dbp = sim.evaluate(cdc, tx.matched_receiver(compensation=_dbp(link, steps_per_span=8)))
    np.testing.assert_array_equal(dbp.received_field, cdc.received_field)  # same noise
    edfa = link.elements[1]
    ase_limit = 10 * np.log10(
        tx.launch_power_per_polarization_w
        / (link.n_spans * edfa.ase_psd_per_polarization_w_per_hz * tx.symbol_rate_hz)
    )
    assert dbp.metrics.snr_db > cdc.metrics.snr_db + 5
    assert ase_limit - 3.5 < dbp.metrics.snr_db < ase_limit + 0.3


# ---------------------------------------------------------------------------
# Every pulse mode and pilots run through the reduced-rate chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("pilots", [None, PilotConfig(period=16)])
def test_reduced_rate_chain_works_for_every_pulse_mode_and_pilots(pulse, pilots):
    link = _link(n_spans=2)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1024,
        polarization="dual",
        launch_power_per_channel_dbm=3.0,
        pilots=pilots,
        **pulse,
    )
    sent = tx.generate(rng=np.random.default_rng(0))
    out = link.propagate(sent.field, sent.dt_s)
    rx = tx.matched_receiver(compensation=_dbp(link, steps_per_span=8, samples_per_symbol=2))
    received = rx.receive(out, sent.dt_s, reference=sent)
    cut = rx.transient_symbols
    data = sent.data_symbols()
    sl = slice(cut, data.shape[-1] - cut)
    assert received.equalizer == ("pilot" if pilots else "data_aided")
    assert _snr_db(np.asarray(received.symbols)[..., sl], np.asarray(data)[..., sl]) > 25
