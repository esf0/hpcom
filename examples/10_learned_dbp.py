"""10 — Learned digital back-propagation with ``hpcom.autodiff``.

One step per span is the cheapest DBP, and with the link's own parameters
and the default Kerr step (a phase over the whole step length at the
mid-step power) it is poor. Most of what the training finds here is the
span's effective length: the learned γ scale ≈ 1.6 is 21 km / 12.7 km.
``Scheme(kerr_length="effective")`` gives that without training (the
default of :class:`hpcom.learn.DBP`).
Here the receiver's link model is *trained* instead: per-span γ and β₂ of
the back-propagation, by L-BFGS on exact gradients of the mean-square error
between the back-propagated field and the transmitted one (the approach of
learned DBP, Häger & Pfister 2018, with physical parameters). Training uses
one data set; the learned model then goes back into an ordinary
:class:`~hpcom.DigitalBackPropagation` and is scored by the standard
receiver chain on fresh data and noise.

Needs JAX (``pip install "hpcom[jax]"``). Run::

    python examples/10_learned_dbp.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from hpcom import (
    EDFA,
    DigitalBackPropagation,
    Fiber,
    Link,
    Simulation,
    SingleCarrierTransmitter,
)
from hpcom import autodiff as ad
from hpcom.dsp.frontend import resample

jax.config.update("jax_enable_x64", True)

N_SPANS, SPS = 10, 2


def simulate(seed: int, link: Link) -> tuple[Simulation, SingleCarrierTransmitter]:
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 12,
        launch_power_per_channel_dbm=5.0,
        seed=seed,
    )
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
        seed=seed + 100,
    )
    return sim, tx


def main() -> None:
    fiber = Fiber.ssmf(80.0, n_steps=100)
    link = Link.from_spans(N_SPANS, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=4.0))
    dbp = DigitalBackPropagation(link=link, steps_per_span=1, samples_per_symbol=SPS)

    # ---- training data: the ADC samples and the transmitted field at 2 sps ----
    sim, tx = simulate(1, link)
    train = sim.run()
    n = tx.n_symbols * SPS
    y = jnp.asarray(resample(train.received_field, n))
    x = jnp.asarray(resample(train.transmitted.field, n))
    dt = 1 / (tx.symbol_rate_hz * SPS)

    start = ad.link_parameters(link)
    fibers = [i for i, el in enumerate(link.elements) if isinstance(el, Fiber)]

    def model(v: jax.Array) -> ad.LinkParameters:
        """v = per-span (γ scale, β₂ scale) of the back-propagation model."""
        params = list(start)
        for k, i in enumerate(fibers):
            p = start[i]
            params[i] = p._replace(
                gamma_per_w_per_km=p.gamma_per_w_per_km * v[2 * k],
                betas_psn_per_km=p.betas_psn_per_km.at[1].multiply(v[2 * k + 1]),
            )
        return tuple(params)

    @jax.jit
    def mse(v: jax.Array) -> jax.Array:
        z = ad.compensate(y, dbp, dt, params=model(v), gradient="reversible")
        return jnp.mean(jnp.abs(z - x) ** 2) / jnp.mean(jnp.abs(x) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(mse))
    result = minimize(
        lambda v: tuple(np.asarray(o, np.float64) for o in value_and_grad(jnp.asarray(v))),
        np.ones(2 * N_SPANS),
        jac=True,
        method="L-BFGS-B",
    )
    learned_params = model(jnp.asarray(result.x))
    learned_link = Link(
        elements=tuple(
            p.to_fiber(el) if isinstance(el, Fiber) and isinstance(p, ad.FiberParameters) else el
            for el, p in zip(link.elements, learned_params, strict=True)
        )
    )

    # ---- evaluation on new data and noise, by the standard receiver chain ----
    sim, tx = simulate(7, link)
    test = sim.run()
    learned = DigitalBackPropagation(link=learned_link, steps_per_span=1, samples_per_symbol=SPS)
    print("16-QAM, 32 GBd, 10 × 80 km, +5 dBm, DBP at 2 samples/symbol (new data)")
    print(f"  linear CD compensation          : SNR {test.metrics.snr_db:5.2f} dB")
    for label, stage in (
        ("DBP, 1 step/span, link model   ", dbp),
        ("DBP, 1 step/span, learned model", learned),
        ("DBP, 16 steps/span, link model ", dbp.replace(steps_per_span=16)),
    ):
        snr = sim.evaluate(test, tx.matched_receiver(compensation=stage)).metrics.snr_db
        print(f"  {label} : SNR {snr:5.2f} dB")
    gammas = result.x[0::2]
    print(f"  learned γ scale per span: {np.array2string(gammas, precision=2)}")


if __name__ == "__main__":
    main()
