# Find the optimum launch power: EGN model and split step

**Task.** Get the optimum launch power and SNR of a link in milliseconds
from the EGN model. Then check the prediction with the split-step
simulation at four powers.

```python
from hpcom import EDFA, Fiber, Link, Simulation, SingleCarrierTransmitter, nli

fiber = Fiber.ssmf(80.0, n_steps=40, polarization="dual")
link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=1 << 13,
                              polarization="dual", roll_off=0.1)

egn = nli.predict_snr(link, tx, model="egn")
print(f"EGN: optimum {egn.optimal_launch_power_dbm:+.2f} dBm, SNR {egn.optimal_snr_db:.2f} dB")

for p_dbm in (-2.0, 0.0, 2.0, 4.0):
    tx_p = tx.with_launch_power(per_channel_dbm=p_dbm)
    rx = tx_p.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
    ssfm = Simulation(transmitter=tx_p, link=link, receiver=rx, seed=0).run()
    model = nli.predict_snr(link, tx_p, model="egn")
    print(f"{p_dbm:+.0f} dBm: split step {ssfm.metrics.snr_db:.2f} dB, EGN {model.snr_db:.2f} dB")
```

Output (NumPy backend, 20 s):

```text
EGN: optimum +0.03 dBm, SNR 21.18 dB
-2 dBm: split step 20.44 dB, EGN 20.40 dB
+0 dBm: split step 21.21 dB, EGN 21.18 dB
+2 dBm: split step 20.23 dB, EGN 20.21 dB
+4 dBm: split step 17.44 dB, EGN 17.48 dB
```

**Notes**

- `model="gn"` gives the GN model. For QAM, GN overestimates the
  nonlinear interference. For 16-QAM the error is 3.6 dB at 1 span and
  1.2 dB at 30 spans. EGN is within 0.23 dB of the simulation from 5 spans
  ([research/2026-10-01-gn-egn](../research/2026-10-01-gn-egn/README.md)).
- For WDM, give a `WDMTransmitter` and select the channel with `channel=`.
- `SNRPrediction` also gives the ASE part (`snr_ase_db`) and the NLI part
  (`snr_nli_db`). Formulas: [docs/physics.rst](../docs/physics.rst),
  section "Nonlinear interference".
