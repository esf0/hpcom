"""Does the simulated block length explain GN − first order for Gaussian symbols at many spans?

The simulated signal is periodic in ``n_symbols`` (spectral lines R_s/n apart); the GN
integral is a continuum. First-order η of the centre channel, Gaussian symbols, for
4096 and 16384 symbols per block (the same total number of symbols).
"""

from __future__ import annotations

import math

from hpcom import EDFA, Fiber, Link, WDMConfig, WDMTransmitter, nli

fiber = Fiber.ssmf(80.0, polarization="dual")
for spans in (10, 30):
    link = Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
    for n_symbols, blocks in ((4096, 8), (16384, 2)):
        tx = WDMTransmitter(
            modulation_order=64,
            symbol_rate_hz=32e9,
            n_symbols=n_symbols,
            polarization="dual",
            samples_per_symbol=10,
            wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        )
        r = nli.first_order_nli(link, tx, channel=1, n_blocks=blocks, seed=0, gaussian=True)
        gn = nli.nli_coefficient(link, tx, model="gn", channel=1)
        se = 10 * math.log10(1 + r.eta_stderr_per_w2 / r.eta_per_w2)
        print(
            f"{spans} spans, {n_symbols} symbols × {blocks}: first order "
            f"{10 * math.log10(r.eta_per_w2):.2f} ± {se:.2f} dB, GN {10 * math.log10(gn):.2f} dB",
            flush=True,
        )
