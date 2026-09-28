r"""Receiver-side compensation of deterministic link impairments.

:class:`Compensator` is the stage a pulse-shaped receiver runs on its
front-end samples when ``compensation=`` is set; :class:`DigitalBackPropagation`
is the built-in implementation. Any algorithm — learned DBP, perturbation or
Volterra compensation, … — plugs in by subclassing :class:`Compensator`, and
then sees the same samples and is scored by the same matched filter,
equaliser and metrics as the built-in one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray
from hpcom.fiber import Fiber
from hpcom.link import Link

__all__ = ["Compensator", "DigitalBackPropagation"]


class Compensator(HpcomModel, ABC):
    """Base of receiver compensation stages (see the processing chain below).

    A :class:`~hpcom.receiver.PulseShapedReceiver` with ``compensation=``
    set processes every selected channel as a coherent receiver does:

    1. shift the channel to baseband (local oscillator at its frequency);
    2. ideal low-pass to ``bandwidth_hz`` (``None``: none beyond step 3);
    3. resample to ``samples_per_symbol`` (``None``: keep the simulation
       rate) — the ADC, with ideal anti-aliasing;
    4. :meth:`compensate` — this stage;
    5. RRC matched filter at that rate, symbol-rate sampling, equaliser,
       decisions.

    If steps 2–3 change nothing (``samples_per_symbol`` ``None`` or equal to
    the receiver's, ``bandwidth_hz`` ``None``) the stage instead runs once on
    the whole received field (centred at 0 Hz) before the channels are
    separated: *full-field* processing — for WDM, full-field DBP.

    The receiver's ``cd_compensation_s2``/``_s3`` must then be 0: dispersion
    compensation is this stage's job.

    Attributes
    ----------
    samples_per_symbol
        Sampling rate of the front end in samples per symbol of the channel.
        Real receivers run at ≈ 2; ``1`` keeps only ``|f| < R_s/2`` (the
        roll-off band is lost before compensation). ``None`` processes the
        field at the simulation rate.
    bandwidth_hz
        Two-sided optical bandwidth kept around the channel centre, e.g.
        ``(1+β)R_s`` for single-channel processing or ``3·Δf`` to include
        both neighbours. Must not exceed the front-end sampling rate.
        ``None``: everything the sampling rate holds.
    """

    samples_per_symbol: int | None = Field(default=2, ge=1)
    bandwidth_hz: float | None = Field(default=None, gt=0)

    @abstractmethod
    def compensate(
        self, field: ComplexArray, dt_s: float, *, frequency_offset_hz: float = 0.0
    ) -> ComplexArray:
        """Compensate one channel's samples; returns the same shape.

        ``field`` is ``(n,)`` (single polarisation) or ``(2, n)`` (dual),
        sampled every ``dt_s``, centred on the channel. ``frequency_offset_hz``
        is that centre's baseband frequency in the simulated optical field
        (the WDM channel frequency; 0 for a single carrier), for algorithms
        whose model depends on absolute frequency (dispersion slope,
        walk-off).
        """


class DigitalBackPropagation(Compensator):
    r"""Split-step digital back-propagation (DBP) through a model of the link.

    The link is run backwards, last element first: every fiber by its
    backend's symmetric split-step scheme in the ``backward`` direction with
    ``steps_per_span`` steps and Kerr coefficient ``nonlinearity_scale·γ``;
    every amplifier as the noiseless gain ``1/√G`` (ASE cannot be undone);
    other components through their ``apply_inverse``.

    * With ``samples_per_symbol=None``, no ``bandwidth_hz`` and the forward
      step count, it is the exact inverse of a noiseless link
      (:meth:`hpcom.link.Link.backpropagate`).
    * With ``nonlinearity_scale=0`` it is exact chromatic-dispersion
      compensation, identical to the receiver's ``cd_compensation_s2`` path.
    * Practical DBP runs at ≈ 2 samples per symbol with few steps per span
      and ``nonlinearity_scale`` < 1 optimised against noise.

    **Channel frequency.** A channel centred at baseband frequency
    :math:`f_k` (:math:`\omega_k = 2\pi f_k`) and shifted to 0 Hz sees the
    fiber's linear operator at its own frequency,
    :math:`\hat L(\omega' + \omega_k)`. Expanded around :math:`\omega_k`
    this is the operator of a fiber with

    .. math::

        \beta_1' = \beta_1 - \beta_2\omega_k + \beta_3\omega_k^2/2,\qquad
        \beta_2' = \beta_2 - \beta_3\omega_k,\qquad
        \beta_3' = \beta_3,

    plus the constant phase
    :math:`\phi_0 = -\beta_1\omega_k + \beta_2\omega_k^2/2 - \beta_3\omega_k^3/6`
    per metre, which commutes with every step and is undone once at the end.
    The channel's walk-off :math:`\beta_1'` stays inside the split-step
    scheme (lab frame), so at the simulation rate the result equals the
    full-field DBP shifted down to the channel, except for nonlinear products
    that wrap around the edge of the sampled band (differently on the two
    grids; negligible for an oversampled field). A receiver whose front end
    changes nothing avoids even that by back-propagating the whole field
    once.

    Attributes
    ----------
    link
        The receiver's model of the link — normally the simulated link;
        pass a different one to study parameter mismatch.
    steps_per_span
        Split-step count for every fiber (``None``: each fiber's own
        ``n_steps``, i.e. the forward model's).
    nonlinearity_scale
        Multiplies every fiber's γ (the "optimised γ" of practical DBP);
        ``0`` gives linear dispersion compensation.
    backend
        Propagator for the backward fibers (``None``: each fiber's own, or
        the exact ``"numpy-linear"`` one when ``nonlinearity_scale=0``).
    """

    link: Link
    steps_per_span: int | None = Field(ge=1)
    nonlinearity_scale: float = Field(default=1.0, ge=0.0)
    backend: str | None = None

    @model_validator(mode="after")
    def _validate_invertible(self) -> DigitalBackPropagation:
        for el in self.link.elements:
            if not isinstance(el, Fiber) and not callable(getattr(el, "apply_inverse", None)):
                raise ValueError(
                    f"{type(el).__name__} has no apply_inverse(field, dt_s); "
                    "the link cannot be back-propagated"
                )
        return self

    def compensate(
        self, field: ComplexArray, dt_s: float, *, frequency_offset_hz: float = 0.0
    ) -> ComplexArray:
        x: ComplexArray = np.asarray(field, dtype=np.complex128)
        omega_k = 2.0 * np.pi * frequency_offset_hz
        phase = 0.0  # Σ φ₀·L over the fibers
        for el in reversed(self.link.elements):
            if isinstance(el, Fiber):
                x = self._backward_fiber(el, omega_k).apply_inverse(x, dt_s)
                b1, b2, b3 = el.beta1_s_per_m, el.beta2_s2_per_m, el.beta3_s3_per_m
                phase += (
                    -b1 * omega_k + b2 * omega_k**2 / 2.0 - b3 * omega_k**3 / 6.0
                ) * el.length_m
            else:
                x = el.apply_inverse(x, dt_s)  # type: ignore[attr-defined]
        if phase == 0.0:
            return x
        out: npt.NDArray[np.complex128] = np.asarray(x) * np.exp(-1j * phase)
        return out

    def _backward_fiber(self, fiber: Fiber, omega_k: float) -> Fiber:
        b1, b2, b3 = fiber.beta1_s_per_m, fiber.beta2_s2_per_m, fiber.beta3_s3_per_m
        changes: dict[str, object] = {
            "gamma_per_w_per_km": fiber.gamma_per_w_per_km * self.nonlinearity_scale,
            "beta1_s_per_m": b1 - b2 * omega_k + b3 * omega_k**2 / 2.0,
            "beta2_s2_per_m": b2 - b3 * omega_k,
        }
        if self.steps_per_span is not None:
            changes["n_steps"] = self.steps_per_span
        if self.backend is not None:
            changes["backend"] = self.backend
        elif self.nonlinearity_scale == 0.0:
            changes["backend"] = "numpy-linear"
        return fiber.replace(**changes)
