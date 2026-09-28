"""Output of a :class:`~hpcom.simulation.Simulation` run."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from hpcom._model import ComplexArrayView, HpcomModel, Polymorphic
from hpcom.metrics import Metrics, StreamMetrics
from hpcom.receiver import ReceivedSymbols
from hpcom.transmitter import TransmittedSignal

__all__ = ["Result", "SimulationMetadata"]


class SimulationMetadata(HpcomModel):
    """Provenance of one run — enough to reproduce it exactly.

    ``config`` is the full ``Simulation.model_dump(mode="json")``. To rerun::

        sim = Simulation.model_validate(meta.config).replace(seed=meta.seed_entropy)

    ``seed_entropy`` is the entropy of the root :class:`numpy.random.SeedSequence`
    (equal to ``seed`` when one was given, otherwise the OS entropy drawn).
    """

    seed: int | None
    seed_entropy: int
    discarded_symbols: int = Field(ge=0)
    runtime_s: float
    hpcom_version: str
    numpy_version: str
    config: dict[str, Any] = Field(default_factory=dict)


class Result(HpcomModel):
    """Signals, decoded symbols, metrics and provenance of a simulation run.

    Attributes
    ----------
    transmitted
        The generated :class:`~hpcom.transmitter.TransmittedSignal`.
    received_field
        Field at the link output (before the receiver).
    received
        Receiver output (:class:`~hpcom.receiver.ReceivedSymbols`).
    metrics
        Aggregate over all scored streams.
    stream_metrics
        One entry per stream (polarisation × channel).
    metadata
        :class:`SimulationMetadata`.
    """

    transmitted: Polymorphic[TransmittedSignal]
    received_field: ComplexArrayView
    received: Polymorphic[ReceivedSymbols]
    metrics: Metrics
    stream_metrics: tuple[StreamMetrics, ...]
    metadata: SimulationMetadata

    def stream(
        self, *, polarization: int | None = None, channel: int | None = None
    ) -> StreamMetrics:
        """Metrics of the stream with the given polarisation / channel index."""
        for m in self.stream_metrics:
            if m.polarization == polarization and m.channel == channel:
                return m
        raise KeyError(f"no stream with polarization={polarization}, channel={channel}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary: metrics and metadata (no arrays)."""
        return {
            "metrics": self.metrics.model_dump(mode="json"),
            "stream_metrics": [m.model_dump(mode="json") for m in self.stream_metrics],
            "metadata": self.metadata.model_dump(mode="json"),
        }

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"Result(ber={m.ber:.3e}, snr={m.snr_db:.2f} dB, evm={m.evm_rms:.3e}, "
            f"gmi={m.gmi_bits:.3f} b/sym, n_bits={m.n_bits}, runtime={self.metadata.runtime_s:.3f}s)"
        )

    __str__ = __repr__
