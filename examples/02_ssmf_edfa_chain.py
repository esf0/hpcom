"""02 — 16-QAM over a 3-span SSMF + EDFA link.

The flagship example. Transmits 16-QAM, propagates through three 80 km spans
of standard single-mode fibre with EDFAs that exactly compensate span loss,
and decodes at the receiver. Prints a structured ``Result``.

Requires the TensorFlow extra (``pip install hpcom[tf]``).

Run::

    python examples/02_ssmf_edfa_chain.py
"""

from __future__ import annotations

from hpcom import EDFA, Fiber, Link, Simulation, SingleCarrierReceiver, SingleCarrierTransmitter


def main() -> None:
    span_km = 80.0
    n_spans = 3
    fiber = Fiber.ssmf(length_km=span_km, n_steps=20, backend="tensorflow")
    edfa = EDFA(gain_db=fiber.alpha_db_per_km * span_km, noise_figure_db=5.0)
    link = Link.from_spans(n_spans=n_spans, fiber=fiber, amplifier=edfa)

    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=4096,
        samples_per_symbol=4,
        roll_off=0.1,
        pulse_span_symbols=16,
        seed=0,
    )
    rx = SingleCarrierReceiver(
        modulation_order=tx.modulation_order,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        equalize="unit_power",
        cd_compensation_s2=link.total_dispersion_s2,
    )

    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=rx,
        seed=0,
        launch_power_dbm=-3.0,
    )
    result = sim.run()

    total_km = n_spans * span_km
    print(f"16-QAM over {n_spans} × {span_km:.0f} km SSMF (= {total_km:.0f} km), EDFA NF=5 dB")
    print(result)
    print(f"  config:         {result.metadata.config}")


if __name__ == "__main__":
    main()
