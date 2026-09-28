"""Simulation end to end: every waveform, validation, reproducibility, scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helpers import b2b_link, sc_pair, wdm_pair
from hpcom import (
    EDFA,
    AdditiveNoise,
    ConstantCompositionDM,
    ConstellationConfig,
    Fiber,
    Link,
    OFDMConfig,
    OFDMReceiver,
    OFDMTransmitter,
    PilotConfig,
    Result,
    Simulation,
    SingleCarrierReceiver,
)
from hpcom.experimental import ExperimentalReceiver, ExperimentalTransmission
from hpcom.modulation import apsk, maxwell_boltzmann_for_entropy

_QAM16 = ConstellationConfig.from_square_qam(16)
_PMF = maxwell_boltzmann_for_entropy(_QAM16.points, 3.5)
_PS = _QAM16.with_probabilities(_PMF)
_DM = ConstantCompositionDM(target_pmf=_PMF, block_length=64)
_OFDM = OFDMConfig(
    n_subcarriers=64, cp_len=8, n_guard=4, pilot_subcarrier_indices=(4, 20, 44, 60), oversampling=2
)


def _ofdm_pair(polarization="single"):
    tx = OFDMTransmitter(
        modulation_order=16,
        subcarrier_spacing_hz=500e6,
        n_frames=32,
        ofdm=_OFDM,
        polarization=polarization,
    )
    return tx, OFDMReceiver(
        modulation_order=16, ofdm=_OFDM, polarization=polarization, equalize="pilot"
    )


def _pilot_pair():
    return sc_pair(pilots=PilotConfig(period=8), rx_kwargs={"equalize": "pilot"})


SCENARIOS = {
    "sc-single": lambda: sc_pair(),
    "sc-dual": lambda: sc_pair(polarization="dual"),
    "sc-pilots": _pilot_pair,
    "sc-linear-fir": lambda: sc_pair(pulse_span_symbols=32, convolution="linear"),
    "sc-apsk": lambda: sc_pair(modulation_order=None, constellation=apsk((4, 12), (1.0, 2.8))),
    "sc-dm": lambda: sc_pair(modulation_order=None, constellation=_PS, dm=_DM),
    "sc-iid-ps": lambda: sc_pair(modulation_order=None, constellation=_PS),
    "wdm-single": lambda: wdm_pair(),
    "wdm-dual": lambda: wdm_pair(polarization="dual"),
    "wdm-subset": lambda: wdm_pair(
        n_channels=5, samples_per_symbol=16, rx_kwargs={"channels": (1, 3)}
    ),
    "ofdm-single": lambda: _ofdm_pair(),
    "ofdm-dual": lambda: _ofdm_pair("dual"),
    "experimental": lambda: (
        ExperimentalTransmission(modulation_order=16, symbol_rate_hz=10e9, n_symbols=1024),
        ExperimentalReceiver(modulation_order=16),
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_waveform_is_error_free_back_to_back(name):
    tx, rx = SCENARIOS[name]()
    link = b2b_link()
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
    m = result.metrics
    assert isinstance(result, Result)
    assert m.n_bits > 0 and m.n_bit_errors == 0 and m.ber == 0.0 and m.ser == 0.0
    assert m.evm_rms < 0.03  # blind "unit_power" normalisation has a small finite-block error
    assert 0.0 < m.gmi_bits <= m.mi_bits + 1e-9
    assert m.mi_bits <= np.log2(rx.effective_modulation_order) + 1e-9
    assert sum(s.n_bits for s in result.stream_metrics) == m.n_bits


def test_pilot_scenario_scores_data_bits_only():
    """Regression: pilots used to be scored against zero-filled bits (8 % phantom BER)."""
    tx, rx = _pilot_pair()
    result = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, seed=0).run()
    assert result.metrics.ber == 0.0 and result.metrics.evm_rms < 1e-12
    assert result.metrics.n_bits == tx.n_bits_per_stream


def test_custom_constellation_evm_is_measured_against_the_sent_points():
    """Regression: EVM used square-QAM references for APSK (EVM 1.48 on a clean link)."""
    tx, rx = sc_pair(
        modulation_order=None,
        constellation=apsk((4, 12), (1.0, 2.8)),
        rx_kwargs={"equalize": "data_aided"},
    )
    result = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, seed=0).run()
    assert result.metrics.evm_rms < 1e-12


@pytest.mark.parametrize("with_dm", [False, True])
def test_shaped_rates_equal_the_source_entropy_without_noise(with_dm):
    tx, rx = sc_pair(
        modulation_order=None,
        constellation=_PS,
        dm=_DM if with_dm else None,
        rx_kwargs={"equalize": "data_aided"},
    )
    result = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, seed=0).run()
    entropy = _DM.empirical_pmf if with_dm else _PMF
    h = float(-np.sum(entropy[entropy > 0] * np.log2(entropy[entropy > 0])))
    assert result.metrics.gmi_bits == pytest.approx(h, abs=1e-9)


# ---------------------------------------------------------------------------
# Stream labelling and aggregation
# ---------------------------------------------------------------------------


def test_streams_are_labelled_by_polarisation_and_channel():
    tx, rx = wdm_pair(
        n_channels=5, samples_per_symbol=16, polarization="dual", rx_kwargs={"channels": (4, 0)}
    )
    result = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, seed=0).run()
    labels = [(s.polarization, s.channel) for s in result.stream_metrics]
    assert labels == [(0, 4), (0, 0), (1, 4), (1, 0)]
    assert result.stream(polarization=1, channel=0).n_bits == tx.n_bits_per_stream
    with pytest.raises(KeyError):
        result.stream(polarization=0, channel=2)


def test_aggregate_metrics_combine_the_streams():
    tx, rx = sc_pair(polarization="dual", n_symbols=4096, rx_kwargs={"equalize": "data_aided"})
    noise = AdditiveNoise.for_snr(
        11.0, power_per_polarization_w=tx.launch_power_per_polarization_w, symbol_rate_hz=32e9
    )
    r = Simulation(transmitter=tx, link=Link(elements=(noise,)), receiver=rx, seed=4).run()
    s = r.stream_metrics
    assert r.metrics.n_bit_errors == sum(x.n_bit_errors for x in s) > 0
    assert r.metrics.ber == pytest.approx(r.metrics.n_bit_errors / r.metrics.n_bits)
    assert r.metrics.gmi_bits == pytest.approx(np.mean([x.gmi_bits for x in s]))
    assert min(x.snr_db for x in s) <= r.metrics.snr_db <= max(x.snr_db for x in s)
    assert r.metrics.q_factor_db is not None


# ---------------------------------------------------------------------------
# Discarding edge symbols
# ---------------------------------------------------------------------------


def test_linear_convolution_discards_the_filter_transient_by_default():
    tx, rx = sc_pair(n_symbols=512, pulse_span_symbols=32, convolution="linear")
    r = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, seed=0).run()
    assert r.metadata.discarded_symbols == 32
    assert r.metrics.n_symbols == 512 - 64


def test_explicit_discard_and_dm_block_rounding():
    tx, rx = sc_pair()
    assert (
        Simulation(transmitter=tx, link=b2b_link(), receiver=rx, discard_symbols=10)
        .run()
        .metrics.n_symbols
        == 1004
    )
    tx, rx = sc_pair(modulation_order=None, constellation=_PS, dm=_DM, n_symbols=1024)
    r = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, discard_symbols=10, seed=0).run()
    assert r.metadata.discarded_symbols == 64  # rounded up to one DM block
    assert r.metrics.n_bits == (1024 // 64 - 2) * _DM.bits_per_block


def test_discarding_everything_is_rejected():
    tx, rx = sc_pair(n_symbols=64)
    with pytest.raises(ValueError, match="leaves nothing"):
        Simulation(transmitter=tx, link=b2b_link(), receiver=rx, discard_symbols=32)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rx_change", "word"),
    [
        ({"modulation_order": 64}, "constellations differ"),
        ({"polarization": "dual"}, "polarization"),
        ({"samples_per_symbol": 8}, "samples_per_symbol"),
    ],
)
def test_incompatible_receivers_are_rejected_with_the_reason(rx_change, word):
    tx, rx = sc_pair()
    with pytest.raises(ValueError, match=rf"incompatible simulation:(.|\n)*{word}"):
        Simulation(transmitter=tx, link=b2b_link(), receiver=rx.replace(**rx_change))


def test_waveform_mismatch_is_rejected():
    tx, _ = wdm_pair()
    with pytest.raises(ValueError, match="SingleCarrierReceiver cannot receive a WDMTransmitter"):
        Simulation(
            transmitter=tx, link=b2b_link(), receiver=SingleCarrierReceiver(modulation_order=16)
        )
    _, rx = wdm_pair(spacing_hz=75e9)
    with pytest.raises(ValueError, match="wdm"):
        Simulation(transmitter=tx, link=b2b_link(), receiver=rx)


def test_fiber_polarisation_must_match_the_transmitter():
    tx, rx = sc_pair(polarization="dual")
    with pytest.raises(ValueError, match="link fibers are"):
        Simulation(transmitter=tx, link=Link(elements=(Fiber.ssmf(1.0),)), receiver=rx)


def test_unknown_keyword_is_rejected():
    tx, rx = sc_pair()
    with pytest.raises(ValueError, match="launch_power_dbm"):
        Simulation(transmitter=tx, link=b2b_link(), receiver=rx, launch_power_dbm=0.0)


# ---------------------------------------------------------------------------
# Reproducibility and provenance
# ---------------------------------------------------------------------------


def _noisy_sim(seed: int | None, tx_seed: int | None = None) -> Simulation:
    tx, rx = sc_pair(seed=tx_seed, rx_kwargs={"equalize": "data_aided"})
    return Simulation(
        transmitter=tx,
        link=Link.from_spans(2, Fiber.ssmf(40.0, n_steps=8), EDFA(gain_db=8.0)),
        receiver=rx.replace(cd_compensation_s2=2 * Fiber.ssmf(40.0).accumulated_dispersion_s2),
        seed=seed,
    )


def test_same_seed_same_result_different_seed_different_result():
    a, b, c = _noisy_sim(3).run(), _noisy_sim(3).run(), _noisy_sim(4).run()
    np.testing.assert_array_equal(a.received_field, b.received_field)
    assert a.metrics == b.metrics
    assert not np.array_equal(a.received_field, c.received_field)


def test_transmitter_seed_fixes_the_data_but_not_the_noise():
    a, b = _noisy_sim(1, tx_seed=7).run(), _noisy_sim(2, tx_seed=7).run()
    np.testing.assert_array_equal(a.transmitted.bits, b.transmitted.bits)
    assert not np.array_equal(a.received_field, b.received_field)


def test_unseeded_run_can_be_replayed_exactly_from_its_metadata():
    first = _noisy_sim(None).run()
    replay = (
        Simulation.model_validate(first.metadata.config)
        .replace(seed=first.metadata.seed_entropy)
        .run()
    )
    np.testing.assert_array_equal(replay.received_field, first.received_field)
    assert replay.metrics == first.metrics


def test_result_summary_is_json_serialisable():
    r = _noisy_sim(0).run()
    summary = json.loads(json.dumps(r.to_dict()))
    assert summary["metadata"]["config"]["transmitter"]["type"].endswith("SingleCarrierTransmitter")
    assert summary["metrics"]["n_bits"] == r.metrics.n_bits
    assert "ber=" in repr(r)


# ---------------------------------------------------------------------------
# Nonlinear link
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_default_receiver_removes_the_mean_nonlinear_phase_rotation(nonlinear_backend):
    """Regression: at +2 dBm over 10×80 km the SPM phase rotation (~50°) must not be
    mistaken for noise. Simulation's default receiver ("auto" → data-aided, as pre-v1
    ``nonlinear_shift``) removes it; blind amplitude normalisation cannot."""
    fiber = Fiber.ssmf(80.0, n_steps=80, backend=nonlinear_backend)
    link = Link.from_spans(10, fiber, EDFA(gain_db=16.0, noise_figure_db=4.5))
    tx, _ = sc_pair(n_symbols=4096, launch_power_per_channel_dbm=2.0)
    rx = SingleCarrierReceiver(
        modulation_order=16, samples_per_symbol=4, cd_compensation_s2=link.total_dispersion_s2
    )
    default = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
    blind = Simulation(
        transmitter=tx, link=link, receiver=rx.replace(equalize="unit_power"), seed=0
    ).run()
    assert default.received.equalizer == "data_aided"
    assert default.metrics.ber < 1e-2 < blind.metrics.ber
    assert np.degrees(np.angle(default.received.gain)) > 20


def test_duck_typed_link_components_run_and_are_recorded_by_repr():
    """Regression: a non-HpcomModel link element used to crash run() while dumping provenance."""

    class HalfAttenuator:
        def apply(self, field, dt_s, *, rng=None):
            return field * 0.5

        def __repr__(self) -> str:
            return "HalfAttenuator()"

    tx, rx = sc_pair()
    sim = Simulation(transmitter=tx, link=Link(elements=(HalfAttenuator(),)), receiver=rx, seed=0)
    result = sim.run()
    assert result.metrics.ber == 0.0
    element = result.metadata.config["link"]["elements"][0]
    assert element["repr"] == "HalfAttenuator()" and element["type"].endswith("HalfAttenuator")
    with pytest.raises(ValueError, match="recorded only by its repr"):
        Simulation.model_validate(result.metadata.config)


@pytest.mark.parametrize("name", ["sc-dual", "wdm-subset", "ofdm-dual", "sc-dm", "sc-pilots"])
def test_results_round_trip_through_save_and_load(name, tmp_path):
    tx, rx = SCENARIOS[name]()
    result = Simulation(transmitter=tx, link=b2b_link(), receiver=rx, seed=0).run()
    path = result.save(tmp_path / "run.npz")
    loaded = Result.load(path)
    assert loaded == result
    assert type(loaded.received) is type(result.received)
    assert np.isinf(loaded.metrics.snr_db) == np.isinf(result.metrics.snr_db)


def test_load_rejects_foreign_npz(tmp_path):
    np.savez(tmp_path / "x.npz", _document=np.array('{"format": "other"}'))
    with pytest.raises(ValueError, match=r"not written by Result\.save"):
        Result.load(tmp_path / "x.npz")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matched_receiver_equals_the_hand_built_one(name):
    tx, rx = SCENARIOS[name]()
    auto = tx.matched_receiver(
        **{
            k: getattr(rx, k)
            for k in ("equalize",) + (("channels",) if hasattr(rx, "channels") else ())
        }
    )
    assert auto == rx
    assert rx.compatibility_issues(tx) == []


def test_matched_receiver_overrides_and_custom_waveforms():
    tx, _ = sc_pair()
    rx = tx.matched_receiver(equalize="none", cd_compensation_s2=1e-23)
    assert (rx.equalize, rx.cd_compensation_s2, rx.samples_per_symbol) == ("none", 1e-23, 4)

    class Bare(ExperimentalTransmission):
        def matched_receiver(self, **overrides):  # a subclass that opts out
            return super(ExperimentalTransmission, self).matched_receiver(**overrides)

    with pytest.raises(NotImplementedError, match="does not define matched_receiver"):
        Bare(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8).matched_receiver()
