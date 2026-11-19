# Add laser phase noise and recover the carrier phase

**Task.** Put transmitter and receiver lasers with 100 kHz linewidth into
the link. Then remove the phase noise with blind phase search (BPS).

```python
from hpcom import (EDFA, BlindPhaseSearch, Fiber, LaserPhaseNoise, Link, LocalOscillator,
                   Simulation, SingleCarrierTransmitter)

fiber = Fiber.ssmf(80.0, n_steps=40, polarization="dual")
spans = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
link = Link(elements=(
    LaserPhaseNoise(linewidth_hz=100e3, polarization="dual"),     # transmitter laser
    *spans.elements,
    LocalOscillator(linewidth_hz=100e3, polarization="dual"),     # receiver laser
))
tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=1 << 13,
                              polarization="dual", launch_power_per_channel_dbm=0.0)
cd = link.total_dispersion_s2

sim = Simulation(transmitter=tx, link=link, receiver=tx.matched_receiver(cd_compensation_s2=cd),
                 seed=0)
no_cpr = sim.run()
bps = sim.evaluate(no_cpr, tx.matched_receiver(cd_compensation_s2=cd, cpr=BlindPhaseSearch()))
ideal = Simulation(transmitter=tx, link=spans, receiver=tx.matched_receiver(cd_compensation_s2=cd),
                   seed=0).run()
print(f"no phase noise     : SNR {ideal.metrics.snr_db:.2f} dB")
print(f"phase noise, no CPR: SNR {no_cpr.metrics.snr_db:.2f} dB")
print(f"phase noise, BPS   : SNR {bps.metrics.snr_db:.2f} dB")
```

Output (NumPy backend, 5 s):

```text
no phase noise     : SNR 21.21 dB
phase noise, no CPR: SNR 14.04 dB
phase noise, BPS   : SNR 21.17 dB
```

**Notes**

- Other link elements of the same kind: `LocalOscillator(frequency_offset_hz=...)`,
  `IQImbalance`, `Quantizer` (DAC, ADC), `MachZehnderModulator`.
  A test checks each one against its closed-form theory
  ([docs/impairments.rst](../docs/impairments.rst)).
- BPS removes the transmitter laser noise. The LO noise interacts with
  the dispersion of the link (equalisation-enhanced phase noise). BPS
  does not remove that part. At 10 × 80 km and 100 kHz it is small: BPS
  comes within 0.04 dB of the run without phase noise.
- Adaptive equalisers: `equalize="cma"` or `"lms"` on the receiver. They
  run before BPS.
