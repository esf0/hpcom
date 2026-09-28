"""08 — Probabilistic shaping: Maxwell–Boltzmann 64-QAM with a CC-DM vs. uniform 64-QAM.

At the same SNR the shaped constellation carries more information (GMI) —
the shaping gain. The constant-composition distribution matcher maps uniform
information bits to symbol blocks with the target distribution.

Run::

    python examples/08_probabilistic_shaping.py
"""

from __future__ import annotations

from hpcom import (
    AdditiveNoise,
    ConstantCompositionDM,
    ConstellationConfig,
    Link,
    Simulation,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
)
from hpcom.modulation import maxwell_boltzmann_for_entropy


def main() -> None:
    uniform = ConstellationConfig.from_square_qam(64)
    pmf = maxwell_boltzmann_for_entropy(uniform.points, 5.0)
    shaped = uniform.with_probabilities(pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=512)
    print(
        f"CC-DM: N = {dm.block_length}, rate {dm.rate_bits_per_symbol:.3f} b/sym, "
        f"rate loss {dm.rate_loss_bits_per_symbol:.3f} b/sym"
    )

    print(f"{'SNR [dB]':>9} {'GMI uniform':>12} {'GMI shaped':>11}")
    for snr_db in (10.0, 14.0, 18.0):
        gmis = []
        for cfg, matcher in ((uniform, None), (shaped, dm)):
            tx = SingleCarrierTransmitter(
                constellation=cfg,
                dm=matcher,
                symbol_rate_hz=32e9,
                n_symbols=1 << 14,
                samples_per_symbol=2,
            )
            rx = SingleCarrierReceiver(
                constellation=cfg, dm=matcher, samples_per_symbol=2, equalize="data_aided"
            )
            noise = AdditiveNoise.for_snr(
                snr_db,
                power_per_polarization_w=tx.launch_power_per_polarization_w,
                symbol_rate_hz=32e9,
            )
            gmis.append(
                Simulation(transmitter=tx, link=Link(elements=(noise,)), receiver=rx, seed=0)
                .run()
                .metrics.gmi_bits
            )
        print(f"{snr_db:9.1f} {gmis[0]:12.3f} {gmis[1]:11.3f}")


if __name__ == "__main__":
    main()
