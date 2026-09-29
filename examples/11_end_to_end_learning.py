"""11 — End-to-end learning through a nonlinear fiber link with ``hpcom.autodiff``.

The transmitter's constellation (64 free points) and launch power are
trained jointly to maximise the GMI after the whole chain — RRC pulse
shaping, 20 × 80 km of SSMF with EDFA noise (the split-step NLSE), CD
compensation, matched filter, data-aided phase recovery and a bit-wise
Gaussian demapper — by gradient ascent (Adam) on exact gradients, with a
fresh batch of data and noise at every step. The learned constellation and
power then go back into an ordinary :class:`~hpcom.Simulation` and are
compared with 64-QAM at its own best launch power, on new data.

Needs JAX (``pip install "hpcom[jax]"``). Run::

    python examples/11_end_to_end_learning.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hpcom import (
    EDFA,
    ConstellationConfig,
    Fiber,
    Link,
    Simulation,
    SingleCarrierTransmitter,
)
from hpcom import autodiff as ad

jax.config.update("jax_enable_x64", True)

N_SPANS, N_SYMBOLS, STEPS, LR = 20, 1 << 11, 150, 0.02


def run(
    tx: SingleCarrierTransmitter, link: Link, seed: int, repeats: int = 1
) -> tuple[float, float]:
    """Mean GMI and SNR of the standard pipeline over ``repeats`` data/noise realisations."""
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
    runs = [
        Simulation(
            transmitter=tx.replace(seed=seed + k), link=link, receiver=rx, seed=seed + 1000 + k
        )
        .run()
        .metrics
        for k in range(repeats)
    ]
    return float(np.mean([m.gmi_bits for m in runs])), float(np.mean([m.snr_db for m in runs]))


def main() -> None:
    fiber = Fiber.ssmf(80.0, n_steps=20)
    link = Link.from_spans(N_SPANS, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    qam = ConstellationConfig.from_square_qam(64)
    tx = SingleCarrierTransmitter(
        constellation=qam, symbol_rate_hz=32e9, n_symbols=N_SYMBOLS, samples_per_symbol=4
    )
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)

    # ---- 64-QAM reference: its best launch power --------------------------
    powers = np.arange(-6.0, 2.5, 0.5)
    qam_curve = [
        run(tx.replace(launch_power_per_channel_dbm=p, seed=1), link, 2)[0] for p in powers
    ]
    p_best = float(powers[int(np.argmax(qam_curve))])

    # ---- end-to-end training ----------------------------------------------
    def gmi(tp: ad.TransmitterParameters, key: jax.Array) -> jax.Array:
        k_data, k_noise = jax.random.split(key)
        points, pmf = ad.constellation(tx, tp)
        idx = jax.random.randint(k_data, (N_SYMBOLS,), 0, 64)
        x = ad.modulate(points[idx], tx, launch_power_dbm=tp.launch_power_dbm)
        y = ad.propagate_link(x, link, tx.dt_s, key=k_noise)
        z = ad.demodulate(y, rx, tx.dt_s, reference=points[idx])
        return ad.information_rates(z, idx, points, qam.bit_labels, pmf)[1]

    grad = jax.jit(jax.value_and_grad(gmi))
    params = ad.TransmitterParameters.from_transmitter(tx)._replace(
        launch_power_dbm=jnp.asarray(p_best)
    )
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(lambda p: jnp.zeros_like(jnp.real(p)), params)
    for t, key in enumerate(jax.random.split(jax.random.key(0), STEPS), start=1):
        value, g = grad(params, key)
        g = jax.tree_util.tree_map(jnp.conj, g)  # ascent direction for complex leaves
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * jnp.abs(b) ** 2, v, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b, t=t: (
                p + LR * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-9)
            ),
            params,
            m,
            v,
        )
        if t % 50 == 0:
            print(
                f"  step {t:3d}: batch GMI {float(value):.3f} b/sym, launch {float(params.launch_power_dbm):.2f} dBm"
            )

    # ---- evaluation on new data by the standard pipeline --------------------
    learned = params.to_constellation(qam)
    p_learned = float(params.launch_power_dbm)
    gmi_qam, snr_qam = run(tx.replace(launch_power_per_channel_dbm=p_best), link, 7, 8)
    gmi_l, snr_l = run(
        tx.replace(constellation=learned, launch_power_per_channel_dbm=p_learned), link, 7, 8
    )
    print("64 points, 32 GBd, 20 × 80 km SSMF, CD compensation (mean of 8 new data/noise draws)")
    print(
        f"  64-QAM at its best {p_best:+.1f} dBm   : GMI {gmi_qam:.3f} b/sym, SNR {snr_qam:5.2f} dB"
    )
    print(
        f"  learned at {p_learned:+.2f} dBm          : GMI {gmi_l:.3f} b/sym, SNR {snr_l:5.2f} dB"
    )


if __name__ == "__main__":
    main()
