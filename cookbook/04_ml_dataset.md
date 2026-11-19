# Make a dataset of sent and received symbols, and replay a run

**Task.** Make arrays of sent and received symbols for a set of launch
powers and seeds, for example to train a neural-network equaliser. Then
store one run and replay it from its stored configuration.

```python
import tempfile
from pathlib import Path

import numpy as np

from hpcom import EDFA, Fiber, Link, Result, Simulation, SingleCarrierTransmitter

fiber = Fiber.ssmf(80.0, n_steps=20, polarization="dual")
link = Link.from_spans(5, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=2048,
                              polarization="dual")
base = Simulation(transmitter=tx, link=link,
                  receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2))

sent, received, power = [], [], []
for p_dbm in (0.0, 3.0, 6.0):
    for seed in range(4):
        sim = base.replace(transmitter=tx.with_launch_power(per_channel_dbm=p_dbm), seed=seed)
        result = sim.run()
        sent.append(result.transmitted.symbols)        # (2, 2048): polarisation x symbol
        received.append(result.received.symbols)
        power.append(p_dbm)
x, y = np.stack(received), np.stack(sent)
print(x.shape, x.dtype, y.shape)

path = Path(tempfile.mkdtemp()) / "run.npz"
result.save(path)                                      # arrays and the full configuration
stored = Result.load(path)
replay = Simulation.model_validate(stored.metadata.config).run()          # same seed, same run
print("replay identical:", np.array_equal(replay.received.symbols, result.received.symbols))
```

Output (NumPy backend, 3 s):

```text
(12, 2, 2048) complex128 (12, 2, 2048)
replay identical: True
```

**Notes**

- `received.symbols` are the equalised symbols after the matched filter.
  The receiver default removes the mean nonlinear phase rotation with the
  transmitted data (`equalize="data_aided"`). Use `equalize="none"` for raw
  samples. `result.received_field` is the field at the end of the link.
- The seed sets the data and the noise. Each `Result` holds its full
  configuration (`metadata.config`), so each sample of a dataset can be
  made again, bit for bit.
- Files for large datasets: [examples/04_dataset_generation.py](../examples/04_dataset_generation.py).
  Speed: `Fiber.ssmf(..., backend="jax")` on a GPU is 27–35 times faster at
  2¹⁸ samples ([docs/backends.rst](../docs/backends.rst)).
