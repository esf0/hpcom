# Train a constellation end to end through the fiber

**Task.** Learn the 64 points of a dual-polarisation constellation by
gradient ascent on the information rate, through the split-step simulation
of 10 × 80 km. Then score it against 64-QAM on a finer simulation of the
same link.

```python
import jax
import numpy as np

from hpcom import EDFA, Fiber, Link, SingleCarrierTransmitter, learn

fiber = Fiber.ssmf(80.0, n_steps=20, polarization="dual")
edfa = EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0)
link = Link.from_spans(10, fiber, edfa)
tx = SingleCarrierTransmitter(modulation_order=64, symbol_rate_hz=32e9, n_symbols=1024,
                              polarization="dual", samples_per_symbol=4,
                              launch_power_per_channel_dbm=2.0)

qam = learn.EndToEnd(transmitter=tx, link=link)               # fixed 64-QAM, CD compensation
gs = qam.replace(encoder=learn.Constellation())               # 64 trainable points
trained = learn.Trainer(steps=300).fit(gs)                    # gradients through the split step

finer = Link.from_spans(10, fiber.replace(n_steps=80), edfa)  # score on a finer simulation
r_qam = learn.evaluate(qam, qam.init(jax.random.key(0)), link=finer)
r_gs = learn.evaluate(gs, trained.params, link=finer)         # the same 16 blocks: paired
gain = np.subtract(r_gs.rates, r_qam.rates)
print(f"64-QAM: {r_qam.mean:.3f} bit/symbol/pol")
print(f"GS-64 : {r_gs.mean:.3f} bit/symbol/pol")
print(f"gain  : {gain.mean():+.3f} ± {gain.std(ddof=1) / np.sqrt(gain.size):.3f}")
```

Output (JAX on CPU, 33 s):

```text
64-QAM: 5.780 bit/symbol/pol
GS-64 : 5.831 bit/symbol/pol
gain  : +0.051 ± 0.005
```

**Notes**

- Other blocks: `learn.LearnedDBP` (front end), `learn.NNEncoder`,
  `learn.CNNPredistorter`, `learn.NNDecoder`, `learn.CNNDecoder`. Each block
  is fixed or trainable with its own learning rate
  ([docs/learn.rst](../docs/learn.rst)).
- The trainer uses the reversible adjoint (constant memory), a cosine
  learning-rate decay and a Polyak average of 0.99 of the parameters.
  Without the average, parameters with noisy gradients drift.
- Score on a finer simulation than the training one. Else a method can
  learn the numerical errors of the training simulation.
- WDM: use a `WDMTransmitter`. Its sampling rate must be at least twice the
  occupied bandwidth, else the Kerr products alias into the band.
