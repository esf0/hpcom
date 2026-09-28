"""hpcom — High-performance optical communication library.

The v1 public API::

    from hpcom import (
        # Abstract bases
        Transmitter, Receiver, TransmittedSignal, ReceivedSymbols,
        # Single-carrier concrete classes
        SingleCarrierTransmitter, SingleCarrierReceiver,
        SingleCarrierSignal, SingleCarrierReceivedSymbols,
        # WDM
        WDMConfig, WDMTransmitter, WDMReceiver,
        WDMSignal, WDMReceivedSymbols,
        # OFDM
        OFDMTransmitter, OFDMReceiver, OFDMSignal, OFDMReceivedSymbols,
        # Infrastructure
        Fiber, EDFA, Amplifier, Link,
        Simulation, Result, Metrics,
    )

See ``docs/spec/hpcom-v1.md`` for the design and ``examples/`` for runnable
scenarios. Pre-v1 modules (``channel``, ``signal``, …) remain reachable for
one release via :mod:`hpcom._legacy`; importing them emits a
``DeprecationWarning``.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

from hpcom.amplifier import EDFA, Amplifier
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.metrics import Metrics
from hpcom.ofdm import OFDMReceivedSymbols, OFDMReceiver, OFDMSignal, OFDMTransmitter
from hpcom.receiver import (
    ReceivedSymbols,
    Receiver,
    SingleCarrierReceivedSymbols,
    SingleCarrierReceiver,
)
from hpcom.result import Result, SimulationMetadata
from hpcom.simulation import Simulation
from hpcom.transmitter import (
    SingleCarrierSignal,
    SingleCarrierTransmitter,
    TransmittedSignal,
    Transmitter,
)
from hpcom.wdm import (
    WDMConfig,
    WDMReceivedSymbols,
    WDMReceiver,
    WDMSignal,
    WDMTransmitter,
)

from . import _legacy


def _install_legacy_shim() -> None:
    """Re-expose legacy submodules at the top level so pre-v1 code keeps importing.

    Skip any name where the v1 tree now owns a real submodule — the new
    implementation must win. Only swallow ``ModuleNotFoundError`` when the
    missing module is the v1 name itself; other missing-import errors
    (e.g. a typo *inside* ``hpcom.fiber``) must propagate.
    """
    for name in _legacy.__all__:
        v1_name = f"{__name__}.{name}"
        try:
            _importlib.import_module(v1_name)
        except ModuleNotFoundError as exc:
            if exc.name != v1_name:
                raise
            _sys.modules[v1_name] = getattr(_legacy, name)


_install_legacy_shim()

__version__ = "0.2.0.dev0"

__all__ = [
    "EDFA",
    "Amplifier",
    "Fiber",
    "Link",
    "Metrics",
    "OFDMReceivedSymbols",
    "OFDMReceiver",
    "OFDMSignal",
    "OFDMTransmitter",
    "ReceivedSymbols",
    "Receiver",
    "Result",
    "Simulation",
    "SimulationMetadata",
    "SingleCarrierReceivedSymbols",
    "SingleCarrierReceiver",
    "SingleCarrierSignal",
    "SingleCarrierTransmitter",
    "TransmittedSignal",
    "Transmitter",
    "WDMConfig",
    "WDMReceivedSymbols",
    "WDMReceiver",
    "WDMSignal",
    "WDMTransmitter",
    "__version__",
]
