"""Do gradients explode or vanish through a fiber link? Jacobian singular values vs power and length.

The literature on end-to-end learning avoids gradients through the split-step
method partly because of "uncontrolled growth or vanishing of gradients"
(asserted, never measured). This measures it: the leading singular values of
the Jacobian of a noiseless link (EDFAs restore the launch power), at the
actual transmitted signal, with ``hpcom.autodiff.jacobian_singular_values``.
The link is symplectic, so σ_min = 1/σ_max: the measured σ_max gives the whole
range of gradient gains.

Systems: 16-QAM, RRC β = 0.1, dual polarisation, 32 GBd; one channel, or
three on a 50 GHz grid (centre channel's view is the whole field here).
Links: 1–20 spans of 80 km SSMF, 50 steps/span (logarithmic).

Run::

    python run.py --out results
    python plot.py results
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hpcom
from hpcom import EDFA, Fiber, Link, SingleCarrierTransmitter, WDMConfig, WDMTransmitter
from hpcom import autodiff as ad


def transmitter(system: str, power_dbm: float, n_symbols: int):
    common = {
        "modulation_order": 16,
        "symbol_rate_hz": 32e9,
        "n_symbols": n_symbols,
        "polarization": "dual",
        "launch_power_per_channel_dbm": power_dbm,
        "seed": 1,
    }
    if system == "1ch":
        return SingleCarrierTransmitter(samples_per_symbol=4, **common)
    return WDMTransmitter(
        samples_per_symbol=5, wdm=WDMConfig(n_channels=3, spacing_hz=50e9), **common
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=["1ch", "3ch"])
    parser.add_argument("--spans", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--powers", type=float, nargs="+", default=[-6.0, -3.0, 0.0, 3.0, 6.0, 9.0])
    parser.add_argument("--symbols", type=int, default=1024)
    parser.add_argument("--steps-per-span", type=int, default=50)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.spans, args.powers, args.symbols, args.iterations = [1, 2], [0.0, 6.0], 128, 5
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    fiber = Fiber.ssmf(80.0, n_steps=args.steps_per_span, polarization="dual").with_steps(
        distribution="logarithmic", log_factor=1 / 3
    )
    rows = []
    for system in args.systems:
        for spans in args.spans:
            link = Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
            for power in args.powers:
                tx = transmitter(system, power, args.symbols)
                sent = tx.generate()
                t0 = time.perf_counter()
                spectrum = ad.jacobian_singular_values(
                    lambda a, link=link, dt=sent.dt_s: ad.propagate_link(a, link, dt),
                    jnp.asarray(sent.field),
                    k=args.k,
                    iterations=args.iterations,
                )
                s = np.asarray(spectrum.singular_values)
                rows.append(
                    {
                        "system": system,
                        "spans": spans,
                        "power_dbm": power,
                        "singular_values": s.tolist(),
                        "seconds": time.perf_counter() - t0,
                    }
                )
                print(
                    f"{system} {spans:3d} spans {power:+5.1f} dBm  σ_max {s[0]:9.3f} "
                    f"({20 * math.log10(s[0]):6.2f} dB)  σ_k {s[-1]:8.3f}  ({rows[-1]['seconds']:.0f} s)",
                    flush=True,
                )
                payload = {
                    "config": {k: str(v) for k, v in vars(args).items()},
                    "hpcom": hpcom.__version__,
                    "jax": jax.__version__,
                    "devices": [str(d) for d in jax.devices()],
                    "rows": rows,
                }
                (args.out / "results.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
