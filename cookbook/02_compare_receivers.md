# Compare receivers on one received field: CDC and DBP

**Task.** Score several receivers on the same received field. The data and
the noise are then identical for each receiver, so the differences are
paired and have a small variance.

```python
from hpcom import (EDFA, DigitalBackPropagation, Fiber, Link, Scheme, Simulation,
                   SingleCarrierTransmitter)

fiber = Fiber.ssmf(80.0, n_steps=100, polarization="dual")
link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=1 << 13,
                              polarization="dual", launch_power_per_channel_dbm=4.0)
sim = Simulation(transmitter=tx, link=link,
                 receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2), seed=0)
cdc = sim.run()                                   # this result holds the received field
print(f"CDC                     : {cdc.metrics.snr_db:.2f} dB")

tuned = Scheme(kerr_length="effective", nonlinear_position=0.15)
receivers = {
    "DBP 1 step/span         ": DigitalBackPropagation(link=link, steps_per_span=1),
    "DBP 1 step/span, tuned  ": DigitalBackPropagation(link=link, steps_per_span=1, scheme=tuned),
    "DBP 4 steps/span        ": DigitalBackPropagation(link=link, steps_per_span=4),
}
for name, dbp in receivers.items():
    result = sim.evaluate(cdc, tx.matched_receiver(compensation=dbp))   # same data, same noise
    print(f"{name}: {result.metrics.snr_db:.2f} dB")
```

Output (NumPy backend, 8 s):

```text
CDC                     : 17.43 dB
DBP 1 step/span         : 19.94 dB
DBP 1 step/span, tuned  : 23.62 dB
DBP 4 steps/span        : 26.64 dB
```

**Notes**

- DBP runs on what the ADC gives: the channel at baseband, band-limited and
  resampled to `samples_per_symbol` (default 2) of the DBP stage.
- With few steps per span, the Kerr step position and length set the
  result. The tuned scheme puts the Kerr step at 0.15 of the step and uses
  the effective length. Here this adds 3.7 dB at 1 step/span.
  Background: [docs/numerics.rst](../docs/numerics.rst).
- A custom compensation stage is a subclass of `hpcom.Compensator`.
