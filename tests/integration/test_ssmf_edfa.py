"""End-to-end SSMF + EDFA scenario via the public hpcom API."""

from __future__ import annotations

import pytest

from hpcom import EDFA, Fiber, Link, Result, Simulation
from hpcom.receiver import SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter


@pytest.mark.slow
def test_short_ssmf_edfa_chain_runs_and_decodes_correctly():
    """A few-span low-power SSMF link should produce a valid Result with low BER.

    Power is intentionally low (~ -20 dBm equivalent) so nonlinearity is small;
    the only impairment is amplifier ASE, which still leaves BER well under 1e-2.
    """
    span_km = 80.0
    n_spans = 3
    fiber = Fiber.ssmf(length_km=span_km, n_steps=20, backend="tensorflow")
    edfa = EDFA(gain_db=fiber.alpha_db_per_km * span_km, noise_figure_db=5.0)
    link = Link.from_spans(n_spans=n_spans, fiber=fiber, amplifier=edfa)

    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=2048,
        samples_per_symbol=4,
        roll_off=0.1,
        pulse_span_symbols=16,
        seed=0,
    )
    rx = SingleCarrierReceiver(
        modulation_order=16,
        samples_per_symbol=4,
        roll_off=0.1,
        pulse_span_symbols=16,
    )
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()

    assert isinstance(result, Result)
    assert result.metrics.n_bits > 0
    assert result.metadata.config["n_spans"] == n_spans
    assert result.metadata.seed == 0
    # Sanity bounds — exact value is sensitive to power scaling, so a wide gate.
    assert 0.0 <= result.metrics.ber < 0.5
    assert result.metrics.evm_rms >= 0.0


def test_simulation_rejects_mismatched_modulation_order():
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=1e9, n_symbols=10, samples_per_symbol=4
    )
    rx = SingleCarrierReceiver(modulation_order=64, samples_per_symbol=4)
    fiber = Fiber.ssmf(10.0, backend="numpy-linear")
    link = Link(elements=(fiber,))
    with pytest.raises(ValueError, match="modulation_order"):
        Simulation(transmitter=tx, link=link, receiver=rx)
