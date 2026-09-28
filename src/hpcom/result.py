"""Output of a :class:`~hpcom.simulation.Simulation` run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from hpcom._model import ComplexArrayView, HpcomModel, Polymorphic, _class_key
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

    def save(self, path: str | Path) -> Path:
        """Write the result to a compressed ``.npz`` file (arrays + JSON metadata).

        Reload with :meth:`Result.load`. The file is self-describing: besides
        the arrays (``transmitted.*``, ``received_field``, ``received.*``) it
        holds the full simulation configuration and the metrics as JSON.
        """
        path = Path(path)
        arrays: dict[str, Any] = {"received_field": self.received_field}
        parts: dict[str, Any] = {}
        for name, model in (("transmitted", self.transmitted), ("received", self.received)):
            config_field = "transmitter" if name == "transmitted" else "receiver"
            fields: dict[str, Any] = {}
            for field_name in type(model).model_fields:
                value = getattr(model, field_name)
                if field_name == config_field:
                    continue
                if isinstance(value, np.ndarray):
                    arrays[f"{name}.{field_name}"] = value
                else:
                    fields[field_name] = value
            parts[name] = {"type": _class_key(type(model)), "fields": fields}
        document = {
            "format": "hpcom.Result",
            "version": 1,
            "parts": parts,
            **self.to_dict(),
        }
        np.savez_compressed(path, **arrays, _document=np.array(json.dumps(document)))
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> Result:
        """Read a file written by :meth:`save` (classes of custom waveforms must be imported)."""
        from hpcom.simulation import Simulation

        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        document = json.loads(str(arrays.pop("_document")))
        if document.get("format") != "hpcom.Result":
            raise ValueError(f"{path} was not written by Result.save")
        sim = Simulation.model_validate(document["metadata"]["config"])
        rebuilt: dict[str, Any] = {}
        for name, config in (("transmitted", sim.transmitter), ("received", sim.receiver)):
            part = document["parts"][name]
            model_cls = HpcomModel._registry.get(part["type"])
            if model_cls is None:
                raise ValueError(f"unknown type {part['type']!r}; import its module first")
            own = {k.split(".", 1)[1]: v for k, v in arrays.items() if k.startswith(f"{name}.")}
            config_field = "transmitter" if name == "transmitted" else "receiver"
            rebuilt[name] = model_cls.model_validate(
                {config_field: config, **part["fields"], **own}
            )
        return cls(
            transmitted=rebuilt["transmitted"],
            received_field=arrays["received_field"],
            received=rebuilt["received"],
            metrics=Metrics.model_validate(document["metrics"]),
            stream_metrics=tuple(
                StreamMetrics.model_validate(m) for m in document["stream_metrics"]
            ),
            metadata=SimulationMetadata.model_validate(document["metadata"]),
        )

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
