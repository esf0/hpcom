"""GN, EGN and first-order NLI models against the split-step simulation, vs number of spans.

3 × 32 GBd dual-polarisation channels on a 50 GHz grid (RRC β = 0.1), 80 km
SSMF spans with ideal noiseless EDFAs; the centre channel is scored. For each
span count and format: η of the GN model (coherent and incoherent), of the
closed-form EGN correction, of the first-order perturbation of the actual
signal (``hpcom.nli``), and of the noiseless split-step simulation
(100 steps/span, logarithmic), η = 1/(SNR·P²). The simulation runs at the
power where the GN model predicts SNR_NLI = 30 dB (weak enough for first
order, strong enough to stay far above the split-step error) and 3 dB above
it: equal η at both confirms the first-order regime.

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

import numpy as np

import hpcom
from hpcom import EDFA, Fiber, Link, WDMConfig, WDMTransmitter, nli
from hpcom.metrics import snr_db

FORMATS = {"qpsk": 4, "16qam": 16, "64qam": 64, "gaussian": 64}


def transmitter(order: int, power_dbm: float, n_symbols: int) -> WDMTransmitter:
    # 10 samples/symbol: Kerr products of the 135 GHz band reach ±203 GHz; at 5 samples/symbol
    # (the first run) they aliased into the centre channel and raised η by 0.2-0.6 dB.
    return WDMTransmitter(
        modulation_order=order,
        symbol_rate_hz=32e9,
        n_symbols=n_symbols,
        polarization="dual",
        samples_per_symbol=10,
        launch_power_per_channel_dbm=power_dbm,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )


def ssfm_eta(link: Link, tx: WDMTransmitter, gaussian: bool, seeds: int) -> tuple[float, float]:
    """η from the noiseless simulation, mean and standard error over ``seeds`` blocks.

    The blocks are those of ``nli.first_order_nli(..., seed=0)`` (same generator
    seeds): η varies by up to 0.3 dB from one 4096-symbol block to the next, so
    the comparison is paired.
    """
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, channels=(1,))
    p = tx.launch_power_per_channel_w
    etas = []
    for s in range(seeds):
        rng = np.random.default_rng([0, s])
        sent = tx.generate(rng=rng)
        if gaussian:
            shape = np.shape(sent.data_symbols())
            sent = tx.generate(
                symbols=(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                / math.sqrt(2)
            )
        out = link.propagate(sent.field, sent.dt_s)
        y = rx.receive(out, sent.dt_s, reference=sent).symbols
        x = rx.reference(sent)[1]
        etas.append(10 ** (-snr_db(np.asarray(y).ravel(), np.asarray(x).ravel()) / 10) / p**2)
    return float(np.mean(etas)), float(np.std(etas, ddof=1) / math.sqrt(seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spans", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 30])
    parser.add_argument("--formats", nargs="+", default=list(FORMATS))
    parser.add_argument("--snr-nli-db", type=float, default=30.0)
    parser.add_argument("--symbols", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps-per-span", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fiber = Fiber.ssmf(80.0, n_steps=args.steps_per_span, polarization="dual").with_steps(
        distribution="logarithmic", log_factor=1 / 3
    )
    rows = []
    for spans in args.spans:
        link = Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
        for name in args.formats:
            gaussian = name == "gaussian"
            t0 = time.perf_counter()
            gn = nli.nli_coefficient(link, transmitter(FORMATS[name], 0.0, args.symbols))
            power_w = math.sqrt(10 ** (-args.snr_nli_db / 10) / gn)  # SNR_NLI = 1/(η P²)
            power_dbm = round(10 * math.log10(power_w * 1e3), 2)
            tx = transmitter(FORMATS[name], power_dbm, args.symbols)
            row = {
                "spans": spans,
                "format": name,
                "power_dbm": power_dbm,
                "gn": gn,
                "gn_incoherent": nli.nli_coefficient(link, tx, coherent=False),
                "egn": None if gaussian else nli.nli_coefficient(link, tx, model="egn"),
            }
            first = nli.first_order_nli(
                link, tx, n_blocks=args.seeds, gaussian=gaussian, nodes_per_fiber=96
            )
            row["first_order"], row["first_order_se"] = first.eta_per_w2, first.eta_stderr_per_w2
            row["ssfm"], row["ssfm_se"] = ssfm_eta(link, tx, gaussian, args.seeds)
            hot = tx.replace(launch_power_per_channel_dbm=power_dbm + 3.0)
            row["ssfm_plus3db"], row["ssfm_plus3db_se"] = ssfm_eta(link, hot, gaussian, args.seeds)
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)

            def db(v: float | None) -> str:
                return "   —  " if v is None else f"{10 * math.log10(v):6.2f}"

            print(
                f"{spans:3d} spans {name:9s} GN {db(row['gn'])} inc {db(row['gn_incoherent'])} "
                f"EGN {db(row['egn'])} 1st {db(row['first_order'])} SSFM {db(row['ssfm'])} "
                f"(+3 dB: {db(row['ssfm_plus3db'])}) dB at {power_dbm:+.1f} dBm ({row['seconds']:.0f} s)",
                flush=True,
            )
            payload = {
                "config": {k: str(v) for k, v in vars(args).items()},
                "hpcom": hpcom.__version__,
                "rows": rows,
            }
            (args.out / "results.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
