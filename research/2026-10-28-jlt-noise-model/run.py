"""Noise model of Alves et al., JLT-40902-2026, against the split-step simulation: paired runs.

The paper's pipeline (``~/triplet-fitting/legacy/data_collection_pilot_and_channel_noise.py``)
propagated every waveform twice — with ASE added at each EDFA inside the
split-step loop and without ASE — and took the Gaussian noise variance of its
decoders from the difference of the two received symbol streams. This script
repeats that with hpcom v1 and adds a third run that splits the difference
exactly: the *same* ASE draws propagated through a linear link (γ = 0) from a
zero input. Per symbol, after CD compensation, the matched filter, sampling and
the data-aided complex gain of the noiseless run:

* ``y_clean``  noiseless split-step (signal–signal nonlinearity only);
* ``y_noisy``  split-step with in-line ASE (its own data-aided gain, as a receiver would);
* ``n_tot``    ``y_noisy − y_clean`` before the gains (the paper's "channel noise");
* ``n_lin``    the linear ASE alone; ``n_tot − n_lin`` is the signal–ASE nonlinear interaction.

Link of the paper: single polarisation, 16-QAM, 34.4 GBd, RRC β = 0.1, 15 × 80 km,
D = 16.8 ps/(nm·km), γ = 1.2 /(W·km), α = 0.2 dB/km, 1 km steps, 16 samples/symbol.
The legacy scalar model injected 2hν(G−1)F per sample (6 dB above v1's
n_sp hν(G−1)), so the legacy NF of 4.5 dB is v1 ``noise_figure_db = 10.52``.

Run::

    python run.py --out data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import hpcom
from hpcom import EDFA, Fiber, Link, SingleCarrierTransmitter
from hpcom.modulation.mapping import nearest_symbol_indices
from hpcom.units import dispersion_to_beta2


def fiber(length_km: float, steps: int, gamma: float, backend: str) -> Fiber:
    return Fiber(
        length_km=length_km,
        alpha_db_per_km=0.2,
        beta2_s2_per_m=dispersion_to_beta2(16.8, 1550e-9),
        gamma_per_w_per_km=gamma,
        n_steps=steps,
        backend=backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--powers", type=float, nargs="+", default=[-3, -2, -1, 0, 1, 2, 4, 6, 8])
    parser.add_argument("--runs", type=int, default=64)
    parser.add_argument("--symbols", type=int, default=1 << 16)
    parser.add_argument("--sps", type=int, default=16)
    parser.add_argument("--spans", type=int, default=15)
    parser.add_argument("--span-km", type=float, default=80.0)
    parser.add_argument("--steps-per-span", type=int, default=80)
    parser.add_argument("--nf-db", type=float, default=10.52)
    parser.add_argument("--rate", type=float, default=34.4e9)
    parser.add_argument("--backend", default="numpy")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    span = fiber(args.span_km, args.steps_per_span, 1.2, args.backend)
    gain = span.loss_db
    noisy = Link.from_spans(args.spans, span, EDFA(gain_db=gain, noise_figure_db=args.nf_db))
    clean = Link.from_spans(args.spans, span, EDFA(gain_db=gain, noise_figure_db=None))
    # γ = 0: one step per span is exact; the fibre draws no random numbers, so the
    # same generator seed gives this link the same ASE samples as `noisy`.
    linear = Link.from_spans(
        args.spans,
        fiber(args.span_km, 1, 0.0, "numpy"),
        EDFA(gain_db=gain, noise_figure_db=args.nf_db),
    )
    config = {k: str(v) for k, v in vars(args).items()} | {"hpcom": hpcom.__version__}
    (args.out / "config.json").write_text(json.dumps(config, indent=1))

    for ip, power in enumerate(args.powers):
        tx = SingleCarrierTransmitter(
            modulation_order=16,
            symbol_rate_hz=args.rate,
            n_symbols=args.symbols,
            samples_per_symbol=args.sps,
            polarization="single",
            roll_off=0.1,
            launch_power_per_channel_dbm=power,
        )
        rx = tx.matched_receiver(cd_compensation_s2=noisy.total_dispersion_s2, equalize="none")
        points = tx.resolved_constellation.points
        fields: dict[str, list[np.ndarray]] = {
            k: [] for k in ("idx", "y_clean", "y_noisy", "n_tot", "n_lin")
        }
        t0 = time.perf_counter()
        for run in range(args.runs):
            # Data depends on the run only (paired across powers); ASE on power and run.
            sent = tx.generate(rng=np.random.default_rng([1, run]))
            dt = sent.dt_s
            ase = [2, ip, run]
            raw_n = rx.receive(
                noisy.propagate(sent.field, dt, rng=np.random.default_rng(ase)), dt
            ).symbols
            raw_c = rx.receive(clean.propagate(sent.field, dt), dt).symbols
            raw_l = rx.receive(
                linear.propagate(np.zeros_like(sent.field), dt, rng=np.random.default_rng(ase)), dt
            ).symbols
            x = np.asarray(sent.data_symbols())
            g_c = np.vdot(x, raw_c) / np.vdot(x, x)
            g_n = np.vdot(x, raw_n) / np.vdot(x, x)
            fields["idx"].append(nearest_symbol_indices(x, points).astype(np.uint8))
            fields["y_clean"].append((raw_c / g_c).astype(np.complex64))
            fields["y_noisy"].append((raw_n / g_n).astype(np.complex64))
            fields["n_tot"].append(((raw_n - raw_c) / g_c).astype(np.complex64))
            fields["n_lin"].append((raw_l / g_c).astype(np.complex64))
            if run == 0:
                n_tot, n_lin = fields["n_tot"][0], fields["n_lin"][0]
                rho = abs(np.vdot(n_lin, n_tot)) / np.sqrt(
                    np.vdot(n_lin, n_lin).real * np.vdot(n_tot, n_tot).real
                )
                print(
                    f"{power:+5.1f} dBm  NL phase {np.degrees(np.angle(g_c)):6.2f}°  "
                    f"SNR clean {-10 * np.log10(np.mean(abs(fields['y_clean'][0] - x) ** 2)):5.2f} dB  "
                    f"noisy {-10 * np.log10(np.mean(abs(fields['y_noisy'][0] - x) ** 2)):5.2f} dB  "
                    f"corr(n_lin, n_tot) {rho:.4f}",
                    flush=True,
                )
        np.savez(
            args.out / f"p{power:+.1f}.npz",
            points=points,
            power_dbm=power,
            **{k: np.stack(v) for k, v in fields.items()},
        )
        print(
            f"{power:+5.1f} dBm  {args.runs} runs in {time.perf_counter() - t0:.0f} s", flush=True
        )


if __name__ == "__main__":
    main()
