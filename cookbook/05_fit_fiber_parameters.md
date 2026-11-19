# Fit fiber parameters to a measured field with gradients (JAX)

**Task.** A field was measured at the end of 80 km of fiber. The datasheet
values of the nonlinear coefficient γ and the dispersion β₂ are not
correct. Find the values that make the model agree with the measurement.
The gradient comes from the split-step solver itself.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

from hpcom import Fiber, SingleCarrierTransmitter
from hpcom import autodiff as ad

jax.config.update("jax_enable_x64", True)

# The "measurement": gamma = 1.45 /W/km and D = 16.6 ps/(nm km), and a small noise.
truth = Fiber.ssmf(80.0, n_steps=200, dispersion_ps_per_nm_per_km=16.6).replace(
    gamma_per_w_per_km=1.45)
tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=1024,
                              launch_power_per_channel_dbm=8.0)
sent = tx.generate()
rng = np.random.default_rng(0)
measured = truth.propagate(sent.field, sent.dt_s)
measured = measured + 1e-3 * np.sqrt(np.mean(np.abs(measured) ** 2)) * (
    rng.standard_normal(measured.shape) + 1j * rng.standard_normal(measured.shape))

# The model: the datasheet fiber (gamma = 1.27 /W/km, D = 17 ps/(nm km)).
model = Fiber.ssmf(80.0, n_steps=200)
start = ad.FiberParameters.from_fiber(model)
x_in, y = jnp.asarray(sent.field), jnp.asarray(measured)

def loss(theta):
    params = start._replace(gamma_per_w_per_km=theta[0],
                            betas_psn_per_km=start.betas_psn_per_km.at[1].set(theta[1]))
    y_model = ad.propagate(x_in, model, sent.dt_s, params=params)
    return jnp.sum(jnp.abs(y_model - y) ** 2) / jnp.sum(jnp.abs(y) ** 2)

theta0 = jnp.array([start.gamma_per_w_per_km, start.betas_psn_per_km[1]])
fit = minimize(jax.jit(loss), theta0, method="BFGS")
beta2_true = ad.FiberParameters.from_fiber(truth).betas_psn_per_km[1]
print(f"start: gamma {theta0[0]:.4f} /W/km, beta2 {theta0[1]:.4f} ps^2/km")
print(f"fit  : gamma {fit.x[0]:.4f} /W/km, beta2 {fit.x[1]:.4f} ps^2/km")
print(f"truth: gamma {truth.gamma_per_w_per_km:.4f} /W/km, beta2 {beta2_true:.4f} ps^2/km")
```

Output (JAX on CPU, 4 s):

```text
start: gamma 1.2700 /W/km, beta2 -21.6826 ps^2/km
fit  : gamma 1.4500 /W/km, beta2 -21.1724 ps^2/km
truth: gamma 1.4500 /W/km, beta2 -21.1724 ps^2/km
```

**Notes**

- Each leaf of `FiberParameters` has a gradient: loss, β₁…βₙ, γ, Raman,
  shock time, loss spectrum, multicore coupling, PMD coefficient and laser
  gain. `to_fiber` puts the fitted values back into a normal `Fiber`.
- `ad.propagate_link` and `ad.backpropagate` do the same for a full link
  and for DBP. `gradient="reversible"` keeps the memory constant in the
  number of steps.
- The gradient is the exact derivative of the discretised solution. It
  converges to that of the continuous equation at the order of the scheme.
  Full list: [docs/autodiff.rst](../docs/autodiff.rst).
