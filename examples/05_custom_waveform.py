"""05 — Custom waveform skeleton through a dual-pol Manakov link.

Demonstrates the :mod:`hpcom.experimental` scaffold:
:class:`ExperimentalTransmission` + :class:`ExperimentalReceiver` driving
:class:`hpcom.link.Link` directly (bypassing :class:`hpcom.simulation.Simulation`,
which is contracted to single-carrier/WDM today).

Use this as the starting point for any new waveform that doesn't fit
SingleCarrier/WDM/OFDM. Copy ``src/hpcom/experimental.py``, rename, and
replace the blocks marked ``# >>> REPLACE THIS:`` with your real DSP.

The default waveform is rectangular-pulse QAM (sample-and-hold) — useful as
a wiring check, not as a real link. Expect noticeable BER on a dispersive
fibre because rectangular pulses spread badly. That is *the point*: it
proves the pipeline runs end-to-end with custom waveforms.

Requires the TensorFlow extra (``pip install hpcom[tf]``).

Run::

    python examples/05_custom_waveform.py
"""

from __future__ import annotations

import numpy as np

from hpcom import EDFA, Fiber, Link
from hpcom.experimental import ExperimentalReceiver, ExperimentalTransmission
from hpcom.metrics import bit_error_rate


def main() -> None:
    span_km = 80.0
    n_spans = 3
    sps = 8
    n_symbols = 4096

    # Dual-pol fibre → Manakov SSFM in the backend. Single-pol fibre would
    # use scalar NLSE; either works with the experimental waveform.
    fiber = Fiber.ssmf(length_km=span_km, n_steps=20, backend="tensorflow").model_copy(
        update={"polarization": "dual"},
    )
    edfa = EDFA(gain_db=fiber.alpha_db_per_km * span_km, noise_figure_db=5.0)
    link = Link.from_spans(n_spans=n_spans, fiber=fiber, amplifier=edfa)

    tx = ExperimentalTransmission(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=n_symbols,
        samples_per_symbol=sps,
        polarization="dual",
        seed=0,
    )
    rx = ExperimentalReceiver(
        modulation_order=tx.modulation_order,
        samples_per_symbol=sps,
        polarization="dual",
    )

    # Drive the pipeline manually — no Simulation wrapper.
    sent = tx.generate(rng=np.random.default_rng(0))

    # Back-to-back: bypass the link to prove Tx/Rx are consistent.
    recv_b2b = rx.receive(sent.field, dt_s=sent.dt_s)

    # Through a real dual-pol Manakov link.
    link_rng = np.random.default_rng(1)
    propagated = link.propagate(sent.field, sent.dt_s, rng=link_rng)
    recv_link = rx.receive(propagated, dt_s=sent.dt_s)

    # Bits per pol have shape (n_symbols * k,); drop a few symbols' worth of
    # edge bits. The trivial waveform has no pulse memory so a small skip
    # suffices; real waveforms must skip enough to clear their RRC/FFT tails.
    skip_syms = 4
    k = int(np.log2(tx.modulation_order))

    def ber_of(recv_bits: np.ndarray) -> tuple[float, int, int]:
        a = sent.bits[:, skip_syms * k : -skip_syms * k].reshape(-1)
        b = recv_bits[:, skip_syms * k : -skip_syms * k].reshape(-1)
        return bit_error_rate(a, b)

    b2b_ber, b2b_err, b2b_total = ber_of(recv_b2b.bits)
    link_ber, link_err, link_total = ber_of(recv_link.bits)

    total_km = n_spans * span_km
    print(
        f"Custom rectangular-pulse 16-QAM (sps={sps}), dual-pol over "
        f"{n_spans} × {span_km:.0f} km SSMF (= {total_km:.0f} km Manakov), EDFA NF=5 dB"
    )
    print(f"  field shape:        {sent.field.shape}  (2 × n_samples)")
    print(f"  dt_s:               {sent.dt_s:.3e} s")
    print(f"  B2B  bit errors:    {b2b_err} / {b2b_total}    BER = {b2b_ber:.3e}")
    print(f"  link bit errors:    {link_err} / {link_total}    BER = {link_ber:.3e}")
    print(
        "  (B2B = 0 confirms the Tx/Rx contract is consistent; the link BER "
        "is dominated by uncompensated CD on the rectangular pulses — this "
        "is the wiring demo, not a competitive waveform.)"
    )


if __name__ == "__main__":
    main()
