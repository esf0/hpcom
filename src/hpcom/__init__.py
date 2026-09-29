"""hpcom — high-performance simulation of coherent optical communication links.

Building blocks (all immutable, validated pydantic models)::

    from hpcom import (
        SingleCarrierTransmitter, SingleCarrierReceiver,   # RRC-shaped QAM
        WDMConfig, WDMTransmitter, WDMReceiver,             # multi-channel
        OFDMConfig, OFDMTransmitter, OFDMReceiver,          # CP-OFDM
        PilotConfig, ConstellationConfig,                   # pilots, shaping
        Fiber, EDFA, AdditiveNoise, Link,                   # the channel
        DigitalBackPropagation,                             # receiver DSP
        Simulation, Result, Metrics,                        # run + score
    )

Physics conventions (NLSE/Manakov form, launch power, ASE, pulse shaping,
metric definitions) are documented in ``docs/physics.rst``. Importing
``hpcom`` loads neither TensorFlow nor the pre-v1 code; the TensorFlow
propagator is imported the first time a fiber uses ``backend="tensorflow"``.

Pre-v1 modules remain importable for one release — ``hpcom.channel``,
``hpcom.signal``, … resolve to :mod:`hpcom._legacy` on first use and emit a
:class:`DeprecationWarning` then.
"""

from __future__ import annotations

from hpcom import _legacy_alias
from hpcom._version import __version__
from hpcom.amplifier import EDFA, Amplifier
from hpcom.dsp import Compensator, DigitalBackPropagation
from hpcom.fiber import Fiber, LossSpectrum
from hpcom.link import Link
from hpcom.metrics import Metrics, StreamMetrics
from hpcom.modulation import ConstantCompositionDM, ConstellationConfig
from hpcom.multicore import MulticoreFiber
from hpcom.noise import AdditiveNoise
from hpcom.numerics import Scheme
from hpcom.ofdm import (
    OFDMConfig,
    OFDMReceivedSymbols,
    OFDMReceiver,
    OFDMSignal,
    OFDMTransmitter,
)
from hpcom.pilots import PilotConfig
from hpcom.raman import RamanResponse
from hpcom.receiver import (
    PulseShapedReceiver,
    ReceivedSymbols,
    Receiver,
    SingleCarrierReceivedSymbols,
    SingleCarrierReceiver,
)
from hpcom.result import Result, SimulationMetadata
from hpcom.simulation import Simulation
from hpcom.transmitter import (
    PulseShapedTransmitter,
    SingleCarrierSignal,
    SingleCarrierTransmitter,
    TransmittedSignal,
    Transmitter,
)
from hpcom.wdm import WDMConfig, WDMReceivedSymbols, WDMReceiver, WDMSignal, WDMTransmitter

_legacy_alias.install()

__all__ = [
    "EDFA",
    "AdditiveNoise",
    "Amplifier",
    "Compensator",
    "ConstantCompositionDM",
    "ConstellationConfig",
    "DigitalBackPropagation",
    "Fiber",
    "Link",
    "LossSpectrum",
    "Metrics",
    "MulticoreFiber",
    "OFDMConfig",
    "OFDMReceivedSymbols",
    "OFDMReceiver",
    "OFDMSignal",
    "OFDMTransmitter",
    "PilotConfig",
    "PulseShapedReceiver",
    "PulseShapedTransmitter",
    "RamanResponse",
    "ReceivedSymbols",
    "Receiver",
    "Result",
    "Scheme",
    "Simulation",
    "SimulationMetadata",
    "SingleCarrierReceivedSymbols",
    "SingleCarrierReceiver",
    "SingleCarrierSignal",
    "SingleCarrierTransmitter",
    "StreamMetrics",
    "TransmittedSignal",
    "Transmitter",
    "WDMConfig",
    "WDMReceivedSymbols",
    "WDMReceiver",
    "WDMSignal",
    "WDMTransmitter",
    "__version__",
]


def __getattr__(name: str) -> object:
    """``hpcom.channel`` etc. as attributes (pre-v1 style), loaded lazily."""
    if name in _legacy_alias.LEGACY_ALIASES:
        return _legacy_alias.load(name)
    raise AttributeError(f"module 'hpcom' has no attribute {name!r}")
