Transceiver impairments
=======================

:mod:`hpcom.impairments` models the transmitter and receiver hardware as
link elements. Put the transmitter-side ones before the first fiber and the
receiver-side ones after the last amplifier; :class:`~hpcom.Simulation`,
JSON serialisation and :func:`hpcom.autodiff.propagate_link` treat them
like fibers and amplifiers.

.. code-block:: python

   from hpcom import (EDFA, Fiber, IQImbalance, LaserPhaseNoise, Link, LocalOscillator,
                      MachZehnderModulator, Quantizer)

   fiber = Fiber.ssmf(80.0, polarization="dual")
   spans = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
   link = Link(elements=(
       Quantizer(bits=6),                                        # DAC
       MachZehnderModulator(drive=0.3),
       IQImbalance(amplitude_imbalance_db=0.3, phase_imbalance_deg=2.0),
       LaserPhaseNoise(linewidth_hz=100e3, polarization="dual"),
       *spans.elements,
       LocalOscillator(frequency_offset_hz=200e6, linewidth_hz=100e3, polarization="dual"),
       Quantizer(bits=8),                                        # ADC
   ))

Fields are ``(..., n)`` or, for ``polarization="dual"`` elements,
``(..., 2, n)``; leading axes are a batch. Sample :math:`k` is at time
:math:`t_k = k\,\Delta t`. Random elements draw from the ``rng`` they are
given (a :class:`~hpcom.Simulation` passes its link generator); without one
they use a fresh generator, as :class:`~hpcom.EDFA` does. Digital
back-propagation does not model the impairments: their ``apply_inverse`` is
the identity.

Laser and local-oscillator phase noise
--------------------------------------

A laser of linewidth :math:`\Delta\nu` (FWHM of its Lorentzian line) has a
Wiener phase

.. math:: \phi_0 = 0,\qquad \phi_k - \phi_{k-1} \sim N(0,\, 2\pi\Delta\nu\,\Delta t),
   \qquad \operatorname{Var}\phi_k = 2\pi\Delta\nu\,t_k .

:class:`~hpcom.LaserPhaseNoise` multiplies the field by :math:`e^{i\phi}`,
one phase for both polarisations, independent across the batch.
:class:`~hpcom.LocalOscillator` multiplies by

.. math:: e^{-i(2\pi\Delta f\,t_k + \phi_k)},

with :math:`\Delta f = f_\mathrm{LO} - f_\mathrm{signal}` (the received
spectrum moves by :math:`-\Delta f`) and :math:`\phi` the LO's own Wiener
phase. Without dispersion a transmitter laser and an LO of linewidths
:math:`\Delta\nu_1` and :math:`\Delta\nu_2` act like one laser of
:math:`\Delta\nu_1 + \Delta\nu_2`. With dispersion they differ: the
transmitter's phase noise passes through the fiber and the receiver's CD
compensation, which cancel, while the LO's passes through the CD
compensation alone and becomes equalisation-enhanced phase noise. A link
ending in the LO, followed by a receiver with CD compensation, applies them
in this order.

If the receiver removes only the block's mean phase (a data-aided complex
gain, no carrier-phase recovery), the residual phase variance over a block
of duration :math:`T` is :math:`2\pi\Delta\nu\,T/6`: a 100 kHz laser
limits a 2048-symbol block at 32 GBd to about 22 dB SNR.

IQ imbalance
------------

With :math:`x = I + jQ`, the Q branch of an IQ modulator or receiver has
gain :math:`g = 10^{\mathrm{dB}/20}` (``amplitude_imbalance_db``) and a
quadrature error :math:`\varphi` (``phase_imbalance_deg``):

.. math:: y = I + jg(Q\cos\varphi - I\sin\varphi) = \mu x + \nu x^*,\qquad
   \mu = \frac{1 + ge^{-i\varphi}}{2},\quad \nu = \frac{1 - ge^{i\varphi}}{2}.

A tone at :math:`f` gets an image at :math:`-f`, :math:`\mathrm{IRR}`
below it:

.. math:: \mathrm{IRR} = \frac{|\mu|^2}{|\nu|^2}
   = \frac{1 + g^2 + 2g\cos\varphi}{1 + g^2 - 2g\cos\varphi}

(:attr:`~hpcom.IQImbalance.image_rejection_ratio_db`). For QAM,
:math:`x^*` is uncorrelated with :math:`x`, so a receiver without IQ
compensation sees the image as noise at :math:`-\mathrm{IRR}`.

Quantisation (DAC, ADC)
-----------------------

:class:`~hpcom.Quantizer` is a uniform mid-rise quantiser of the real and
imaginary parts. The full scale is :math:`\pm k\sigma`, with :math:`\sigma`
the RMS of one real dimension of each row (each polarisation and block) and
:math:`k` the ``clipping_ratio``; with :math:`b` ``bits``,

.. math:: \Delta = \frac{2k\sigma}{2^b},\qquad
   Q(x) = \Delta\left(\operatorname{clip}\left(\left\lfloor x/\Delta\right\rfloor,\,
   -2^{b-1},\, 2^{b-1}-1\right) + \tfrac12\right).

In the granular region the error is uniform on :math:`\pm\Delta/2`
(variance :math:`\Delta^2/12`), so for a Gaussian signal with negligible
clipping

.. math:: \mathrm{SNR} = 6.02\,b + 4.77 - 20\log_{10}k\ \mathrm{dB}.

The quantisation noise is white over the simulation bandwidth: with
oversampling, the matched filter keeps only the in-band part.

Mach–Zehnder modulator
----------------------

An IQ modulator of two nested Mach–Zehnder modulators biased at null has a
sinusoidal field transfer per quadrature. :class:`~hpcom.MachZehnderModulator`
maps each quadrature, normalised by :math:`\sigma` as above, to

.. math:: I \mapsto \sin\!\left(d\,\frac{\pi}{2}\,\frac{I}{\sigma}\right)

with the ``drive`` :math:`d \in (0, 1]` (an RMS-level sample reaches
:math:`d` of the way to the transfer maximum), and rescales the output to
the input's mean power (the modulator's loss belongs to the launch power).
Since :math:`\sin z = z - z^3/6 + \dots`, the distortion relative to the
signal grows as :math:`d^2`. For a Gaussian input (Bussgang) the
signal-to-distortion ratio is

.. math:: \mathrm{SDR} = \frac{s^2e^{-s^2}}{\tfrac12(1 - e^{-2s^2}) - s^2e^{-s^2}}
   \approx \frac{6}{s^4},\qquad s = \frac{d\pi}{2},

i.e. 40 dB at :math:`d = 0.1` and 21 dB at :math:`d = 0.3`, before the
matched filter removes the out-of-band part.

Differentiable mode
-------------------

:func:`hpcom.autodiff.propagate_link` applies every impairment in JAX.
:func:`hpcom.autodiff.link_parameters` gives ``None`` for them (they have
no trainable parameters). The phase noises are reparameterised —
:math:`\phi = \sqrt{2\pi\Delta\nu\,\Delta t}\,\operatorname{cumsum}\xi` with
:math:`\xi` from the JAX key, which is split once over the noisy amplifiers
and random impairments in link order (links without impairments keep their
keys). The quantiser passes the gradient straight through,
:math:`Q(x) = x + \operatorname{stop\_gradient}(Q(x) - x)`, so a
transmitter can be trained through a DAC; the deterministic elements are
the NumPy formulas in JAX.

[``tests/numerical/test_impairments.py``: phase increments have variance
:math:`2\pi\Delta\nu\,\Delta t` and the phase variance grows linearly in
time (NumPy and JAX); one phase for both polarisations; an LO offset of
whole frequency bins is an exact cyclic shift of the spectrum; IQ
imbalance equals its definition and the image of a tone is IRR below it;
the quantiser's SNR follows :math:`6.02b + 4.77 - 20\log_{10}k` within
0.5 dB for :math:`b = 4…8` with uniform granular error; the modulator keeps
the power, its error scales as :math:`d^2` and its SDR follows the Bussgang
formula (0.1 dB); links round-trip through
JSON and run DBP; in a simulation the IQ image adds like noise at
:math:`-\mathrm{IRR}` (0.3 dB), phase noise and an LO offset cost SNR;
JAX equals NumPy to :math:`10^{-12}` (x64) for the deterministic elements,
gradients through the whole impaired link are finite, the quantiser's
gradient is the identity's.]
