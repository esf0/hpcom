"""Receiver-side digital signal processing.

* :class:`Compensator` — base of the compensation stage a pulse-shaped
  receiver runs on its front-end samples (``compensation=``); subclass it to
  plug in any algorithm.
* :class:`DigitalBackPropagation` — split-step DBP at any sampling rate and
  bandwidth (single-channel to full-field).
* :func:`lowpass`, :func:`resample` — the ideal front-end filter and exact
  band-limited resampling used by the receivers, for custom chains.
"""

from hpcom.dsp.compensation import Compensator, DigitalBackPropagation
from hpcom.dsp.frontend import lowpass, resample

__all__ = ["Compensator", "DigitalBackPropagation", "lowpass", "resample"]
