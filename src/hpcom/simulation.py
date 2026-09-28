"""End-to-end runner: Transmitter → Link → Receiver → :class:`~hpcom.result.Result`."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import HpcomModel, Polymorphic
from hpcom._version import __version__
from hpcom.link import Link
from hpcom.metrics import (
    Metrics,
    StreamMetrics,
    bit_error_rate,
    evm_rms,
    gmi_gaussian,
    mi_gaussian,
    q_factor_db,
    snr_db,
)
from hpcom.receiver import ReceivedSymbols, Receiver
from hpcom.result import Result, SimulationMetadata
from hpcom.transmitter import Transmitter
from hpcom.wdm import WDMReceiver

__all__ = ["Simulation"]


class Simulation(HpcomModel):
    """One Transmitter → Link → Receiver run, scored against the transmitted data.

    Works with any compatible transmitter/receiver pair (single-carrier, WDM,
    OFDM, custom subclasses); compatibility is checked at construction via
    :meth:`Receiver.compatibility_issues`.

    Attributes
    ----------
    transmitter, link, receiver
        The three stages. The launch power is the transmitter's.
    seed
        Root seed. It is split with :class:`numpy.random.SeedSequence` into
        independent streams for the transmitter data and the link noise
        (ASE). A transmitter with its own ``seed`` keeps it for the data. With
        ``seed=None`` fresh entropy is drawn and recorded in
        ``result.metadata.seed_entropy``.
    discard_symbols
        Data symbols excluded from scoring at *each* end of every stream.
        ``None`` → the receiver's filter transient (0 for circular pulse
        shaping, ``pulse_span_symbols`` for linear). With a DM the cut is
        rounded up to whole DM blocks.
    """

    transmitter: Polymorphic[Transmitter]
    link: Link
    receiver: Polymorphic[Receiver]
    seed: int | None = None
    discard_symbols: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate(self) -> Simulation:
        issues = self.receiver.compatibility_issues(self.transmitter)
        pols = self.link.polarizations
        if pols and pols != {self.transmitter.polarization}:
            issues.append(
                f"link fibers are {sorted(pols)}-polarisation but the transmitter is "
                f"{self.transmitter.polarization!r}"
            )
        if issues:
            raise ValueError("incompatible simulation:\n  - " + "\n  - ".join(issues))
        n_data = self.transmitter.n_data_symbols_per_stream
        cut = self._discard_count()
        if n_data - 2 * cut < 1:
            raise ValueError(
                f"discarding {cut} symbols at each end leaves nothing of the "
                f"{n_data} data symbols per stream to score"
            )
        return self

    def _discard_count(self) -> int:
        """Data symbols cut at each end (rounded up to DM blocks)."""
        cut = (
            self.receiver.transient_symbols
            if self.discard_symbols is None
            else self.discard_symbols
        )
        dm = self.transmitter.dm
        if dm is not None and cut:
            cut = math.ceil(cut / dm.block_length) * dm.block_length
        return cut

    def run(self) -> Result:
        """Execute the simulation and score it."""
        t0 = time.perf_counter()
        root = np.random.SeedSequence(self.seed)
        tx_seq, link_seq = root.spawn(2)
        tx_rng = None if self.transmitter.seed is not None else np.random.default_rng(tx_seq)

        sent = self.transmitter.generate(rng=tx_rng)
        out = self.link.propagate(sent.field, sent.dt_s, rng=np.random.default_rng(link_seq))
        received = self.receiver.receive(out, sent.dt_s, reference=sent)
        overall, per_stream = self._score(
            sent_bits_symbols=self.receiver.reference(sent), rx=received
        )

        metadata = SimulationMetadata(
            seed=self.seed,
            seed_entropy=int(root.entropy),  # type: ignore[arg-type]
            discarded_symbols=self._discard_count(),
            runtime_s=time.perf_counter() - t0,
            hpcom_version=__version__,
            numpy_version=np.__version__,
            config=self.model_dump(mode="json"),
        )
        return Result(
            transmitted=sent,
            received_field=np.asarray(out, dtype=np.complex128),
            received=received,
            metrics=overall,
            stream_metrics=per_stream,
            metadata=metadata,
        )

    # ---- scoring ----------------------------------------------------------

    def _score(
        self,
        sent_bits_symbols: tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]],
        rx: ReceivedSymbols,
    ) -> tuple[Metrics, tuple[StreamMetrics, ...]]:
        ref_bits, ref_syms = sent_bits_symbols
        lead = ref_syms.shape[:-1]
        cfg = self.receiver.resolved_constellation
        pmf = self.transmitter.dm.empirical_pmf if self.transmitter.dm is not None else cfg.pmf
        cut = self._discard_count()
        n_sym = ref_syms.shape[-1]
        sym_sl = slice(cut, n_sym - cut)
        dm = self.transmitter.dm
        # Bits carried by the `cut` symbols at each end (whole DM blocks with a DM).
        bit_cut = (
            cut // dm.block_length * dm.bits_per_block
            if dm is not None
            else cut * cfg.bits_per_symbol
        )
        bit_sl = slice(bit_cut, ref_bits.shape[-1] - bit_cut)

        # Index of the transmitted point of every data symbol (for SER / MI / GMI).
        sent_idx = cfg.nearest(ref_syms)
        streams: list[StreamMetrics] = []
        totals = {"bits": 0, "bit_err": 0, "sym": 0, "sym_err": 0, "err_pow": 0.0, "ref_pow": 0.0}
        mi_sum = gmi_sum = 0.0
        for pos in itertools.product(*(range(n) for n in lead)):
            y = np.asarray(rx.symbols[pos])[sym_sl]
            x = np.asarray(ref_syms[pos])[sym_sl]
            tx_idx = sent_idx[pos][sym_sl]
            ber, n_err, n_bits = bit_error_rate(ref_bits[pos][bit_sl], rx.bits[pos][bit_sl])
            n_sym_err = int(np.count_nonzero(np.asarray(rx.decisions[pos])[sym_sl] != tx_idx))
            mi = mi_gaussian(y, tx_idx, cfg.points, pmf)
            gmi = gmi_gaussian(y, tx_idx, cfg.points, cfg.bit_labels, pmf)
            pol, ch = self._stream_labels(pos)
            streams.append(
                StreamMetrics(
                    n_bits=n_bits,
                    n_bit_errors=n_err,
                    ber=ber,
                    n_symbols=int(y.size),
                    n_symbol_errors=n_sym_err,
                    ser=n_sym_err / y.size,
                    evm_rms=evm_rms(y, x),
                    snr_db=snr_db(y, x),
                    q_factor_db=q_factor_db(ber),
                    mi_bits=mi,
                    gmi_bits=gmi,
                    polarization=pol,
                    channel=ch,
                )
            )
            totals["bits"] += n_bits
            totals["bit_err"] += n_err
            totals["sym"] += int(y.size)
            totals["sym_err"] += n_sym_err
            totals["err_pow"] += float(np.sum(np.abs(y - x) ** 2))
            totals["ref_pow"] += float(np.sum(np.abs(x) ** 2))
            mi_sum += mi
            gmi_sum += gmi

        n_streams = len(streams)
        ber = totals["bit_err"] / totals["bits"] if totals["bits"] else 0.0
        evm = float(np.sqrt(totals["err_pow"] / totals["ref_pow"])) if totals["ref_pow"] else 0.0
        overall = Metrics(
            n_bits=int(totals["bits"]),
            n_bit_errors=int(totals["bit_err"]),
            ber=ber,
            n_symbols=int(totals["sym"]),
            n_symbol_errors=int(totals["sym_err"]),
            ser=totals["sym_err"] / totals["sym"],
            evm_rms=evm,
            snr_db=float("inf") if evm == 0 else float(-20.0 * np.log10(evm)),
            q_factor_db=q_factor_db(ber),
            mi_bits=mi_sum / n_streams,
            gmi_bits=gmi_sum / n_streams,
        )
        return overall, tuple(streams)

    def _stream_labels(self, pos: tuple[int, ...]) -> tuple[int | None, int | None]:
        """Map a stream position to (polarisation index, channel index)."""
        i = 0
        pol: int | None = None
        ch: int | None = None
        if self.transmitter.polarization == "dual":
            pol = pos[0]
            i = 1
        if isinstance(self.receiver, WDMReceiver):
            ch = self.receiver.selected_channels[pos[i]]
        return pol, ch
