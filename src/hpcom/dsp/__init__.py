"""Receiver-side digital signal processing.

* :class:`Compensator` — base of the compensation stage a pulse-shaped
  receiver runs on its front-end samples (``compensation=``); subclass it to
  plug in any algorithm.
* :class:`DigitalBackPropagation` — split-step DBP at any sampling rate and
  bandwidth (single-channel to full-field).
* :class:`BlindPhaseSearch`, :func:`blind_phase_search` — blind
  carrier-phase recovery (the receiver's ``cpr=``).
* :class:`AdaptiveEqualizer`, :func:`adaptive_equalize` — CMA and LMS
  butterfly equalisers (``equalize="cma"``, ``"lms"``).
* :func:`lowpass`, :func:`resample` — the ideal front-end filter and exact
  band-limited resampling used by the receivers, for custom chains.
"""

from hpcom.dsp.adaptive import AdaptiveEqualizer, adaptive_equalize
from hpcom.dsp.compensation import Compensator, DigitalBackPropagation
from hpcom.dsp.frontend import lowpass, resample
from hpcom.dsp.phase import BlindPhaseSearch, blind_phase_search

__all__ = [
    "AdaptiveEqualizer",
    "BlindPhaseSearch",
    "Compensator",
    "DigitalBackPropagation",
    "adaptive_equalize",
    "blind_phase_search",
    "lowpass",
    "resample",
]
