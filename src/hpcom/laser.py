r"""Fiber lasers and amplifiers: gain fibers, saturable absorbers, filters, ring cavities.

A :class:`GainFiber` is a :class:`~hpcom.Fiber` with distributed gain of
finite bandwidth that saturates on the energy in the simulation window:

.. math::

    \frac{\partial\tilde A}{\partial z} = \Big[\hat L(\omega)
       + \frac{g(E)}{2}\,G(\omega)\Big]\tilde A + \mathcal F[\hat N A],
    \qquad g(E) = \frac{g_0}{1 + E/E_\mathrm{sat}},\quad E = \int|A|^2dt,

with :math:`G(\omega)` Lorentzian or parabolic (:func:`hpcom.backends.base.gain_spectrum`).
With a parabolic profile, no saturation and a cubic gain
``nonlinear_gain_per_w_per_km`` this is the cubic complex Ginzburg–Landau
equation; with saturation, a saturable absorber and a cavity, the Haus
master-equation model of a mode-locked fiber laser. Unsaturated gain is part
of the exact linear step (every scheme); saturated gain is applied per
linear sub-step at the mid-step energy (split-step schemes).

Lumped components (:class:`SaturableAbsorber`, :class:`SpectralFilter`,
:class:`OutputCoupler`) and fibers form a :class:`RingCavity`, which
circulates a field until the output repeats itself.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import ComplexArrayView, FloatArrayField, HpcomModel, Polymorphic
from hpcom._typing import ComplexArray, LinkComponent
from hpcom.fiber import Fiber

__all__ = [
    "CavityResult",
    "GainFiber",
    "OutputCoupler",
    "RingCavity",
    "SaturableAbsorber",
    "SpectralFilter",
]


class GainFiber(Fiber):
    r"""A span of gain fiber (see the module docstring).

    Attributes
    ----------
    gain_db_per_m
        Small-signal power gain :math:`g_0` at the carrier, dB/m.
    gain_bandwidth_hz
        FWHM of the gain profile (``None``: flat).
    gain_shape
        ``"lorentzian"`` :math:`G = 1/(1 + (\omega/\Omega_g)^2)` or
        ``"parabolic"`` :math:`G = 1 - (\omega/\Omega_g)^2/2`,
        :math:`\Omega_g = \pi\,\mathrm{FWHM}`.
    saturation_energy_j
        :math:`E_\mathrm{sat}` (for a laser: saturation power × round-trip
        time); ``None``: unsaturated (a linear amplifier). Saturation needs a
        split-step scheme.

    A parabolic profile makes the equation diffusive (unbounded
    :math:`-\omega^2` gain, the complex Ginzburg–Landau equation): the
    negative sub-steps of the fourth-order compositions would amplify the
    highest grid frequencies without bound, so they are rejected; Strang
    splitting and RK4IP are stable. A Lorentzian profile is bounded and
    keeps every scheme, at its order.
    """

    gain_db_per_m: float = Field(gt=0)
    gain_bandwidth_hz: float | None = Field(default=None, gt=0)
    gain_shape: Literal["lorentzian", "parabolic"] = "lorentzian"
    saturation_energy_j: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_gain(self) -> GainFiber:
        fourth_order = self.scheme.method in ("ssfm4-yoshida", "ssfm4-suzuki")
        if self.gain_bandwidth_hz is not None and self.gain_shape == "parabolic" and fourth_order:
            raise ValueError(
                "a parabolic gain profile makes the equation diffusive, and the negative "
                "sub-steps of fourth-order compositions then amplify the highest frequencies "
                "without bound (splitting beyond second order with real coefficients is "
                "ill-posed for parabolic equations); use 'ssfm', 'rk4ip' or a Lorentzian profile"
            )
        if self.saturation_energy_j is not None and not self.scheme.is_split_step:
            raise ValueError(
                "saturated gain needs a split-step scheme (ssfm, ssfm4-yoshida, ssfm4-suzuki), "
                f"got {self.scheme.method!r}"
            )
        return self

    @property
    def small_signal_gain_db(self) -> float:
        """Total small-signal gain of the span, dB (without background loss)."""
        return self.gain_db_per_m * self.length_m

    def recentred(self, frequency_offset_hz: float) -> Fiber:
        """As :meth:`hpcom.Fiber.recentred`; only for flat, unsaturated gain at an offset.

        A gain profile stays centred on the carrier, and saturation needs the
        energy of the whole field, not of one channel.
        """
        if frequency_offset_hz != 0.0 and (
            self.gain_bandwidth_hz is not None or self.saturation_energy_j is not None
        ):
            raise ValueError(
                "a gain fiber with a gain bandwidth or saturation cannot be recentred on another "
                "frequency; back-propagate it full-field"
            )
        return super().recentred(frequency_offset_hz)


class SaturableAbsorber(HpcomModel):
    r"""Fast (instantaneous) saturable absorber: power transmission

    .. math:: T(P) = 1 - q_\mathrm{ns} - \frac{q_0}{1 + P/P_\mathrm{sat}},

    applied to the field as :math:`\sqrt{T(P)}`, :math:`P` the instantaneous
    power (summed over polarisations).

    Attributes
    ----------
    modulation_depth
        :math:`q_0`, the saturable loss.
    saturation_power_w
        :math:`P_\mathrm{sat}`.
    non_saturable_loss
        :math:`q_\mathrm{ns}`.
    """

    modulation_depth: float = Field(gt=0, lt=1)
    saturation_power_w: float = Field(gt=0)
    non_saturable_loss: float = Field(default=0.0, ge=0, lt=1)

    @model_validator(mode="after")
    def _validate(self) -> SaturableAbsorber:
        if self.modulation_depth + self.non_saturable_loss >= 1.0:
            raise ValueError("modulation_depth + non_saturable_loss must be < 1")
        return self

    def transmission(self, power_w: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Power transmission :math:`T(P)`."""
        p = np.asarray(power_w, dtype=np.float64)
        return np.asarray(
            1.0
            - self.non_saturable_loss
            - self.modulation_depth / (1.0 + p / self.saturation_power_w),
            dtype=np.float64,
        )

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        del dt_s, rng
        a = np.asarray(field, dtype=np.complex128)
        power = np.abs(a) ** 2
        if a.ndim >= 2 and a.shape[-2] == 2:  # dual polarisation: total power
            power = power.sum(axis=-2, keepdims=True)
        out: ComplexArray = a * np.sqrt(self.transmission(power))
        return out


class SpectralFilter(HpcomModel):
    r"""Band-pass filter: power transmission :math:`e^{-\ln 2\,(2|f - f_c|/B)^{2m}}` (FWHM :math:`B`).

    ``order`` :math:`m` = 1 is Gaussian, larger is super-Gaussian;
    ``center_hz`` is a baseband frequency.
    """

    bandwidth_hz: float = Field(gt=0)
    order: int = Field(default=1, ge=1)
    center_hz: float = 0.0

    def response(self, n_samples: int, dt_s: float) -> npt.NDArray[np.float64]:
        """Amplitude response on the FFT grid."""
        f = np.fft.fftfreq(n_samples, d=dt_s) - self.center_hz
        x = (2.0 * np.abs(f) / self.bandwidth_hz) ** (2 * self.order)
        return np.asarray(np.exp(-np.log(2.0) / 2.0 * x), dtype=np.float64)

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        del rng
        a = np.asarray(field, dtype=np.complex128)
        out: ComplexArray = np.fft.ifft(
            np.fft.fft(a, axis=-1) * self.response(a.shape[-1], dt_s), axis=-1
        )
        return out


class OutputCoupler(HpcomModel):
    """Taps ``output_ratio`` of the power out of a cavity; the rest circulates."""

    output_ratio: float = Field(gt=0, lt=1)

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        """The field that stays in the cavity."""
        del dt_s, rng
        out: ComplexArray = np.asarray(field, dtype=np.complex128) * np.sqrt(
            1.0 - self.output_ratio
        )
        return out

    def tap(self, field: ComplexArray) -> ComplexArray:
        """The output field."""
        out: ComplexArray = np.asarray(field, dtype=np.complex128) * np.sqrt(self.output_ratio)
        return out


class CavityResult(HpcomModel):
    """Output of :meth:`RingCavity.run`.

    Attributes
    ----------
    output
        Field leaving the output coupler on the last round trip.
    intracavity
        Field after the last element on the last round trip.
    output_energy_j
        Output energy in the window, per round trip.
    change
        Relative change of the output power profile from the previous round
        trip, :math:`\\||A_k|^2 - |A_{k-1}|^2\\|/\\||A_k|^2\\|` (``nan`` on the
        first) — blind to the constant phase a pulse picks up per round trip.
    converged
        ``True`` if ``change`` fell below the requested tolerance.
    """

    output: ComplexArrayView
    intracavity: ComplexArrayView
    output_energy_j: FloatArrayField
    change: FloatArrayField
    converged: bool

    @property
    def round_trips(self) -> int:
        return int(self.output_energy_j.size)


class RingCavity(HpcomModel):
    """A unidirectional ring: ``elements`` in order, one :class:`OutputCoupler` among them.

    Each round trip applies every element in turn (fibers propagate, lumped
    components act); the output is tapped at the coupler.
    """

    elements: tuple[Polymorphic[LinkComponent], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> RingCavity:
        couplers = [el for el in self.elements if isinstance(el, OutputCoupler)]
        if len(couplers) != 1:
            raise ValueError(f"a ring cavity needs exactly one OutputCoupler, got {len(couplers)}")
        return self

    def round_trip(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> tuple[ComplexArray, ComplexArray]:
        """``(intracavity field after one round trip, output field)``."""
        x: ComplexArray = np.asarray(field, dtype=np.complex128)
        out = None
        for el in self.elements:
            if isinstance(el, OutputCoupler):
                out = el.tap(x)
            x = el.apply(x, dt_s, rng=rng)
        assert out is not None
        return x, out

    def run(
        self,
        field: ComplexArray,
        dt_s: float,
        n_round_trips: int,
        *,
        tol: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> CavityResult:
        """Circulate ``field`` for up to ``n_round_trips``, stopping early once ``change < tol``.

        ``field`` is one field, ``(n,)`` or ``(2, n)``; the recorded energy
        sums both polarisations.
        """
        x: ComplexArray = np.asarray(field, dtype=np.complex128)
        energies, changes = [], []
        previous: npt.NDArray[np.float64] | None = None
        converged = False
        out: ComplexArray = x
        for _ in range(n_round_trips):
            x, out = self.round_trip(x, dt_s, rng=rng)
            power = np.abs(out) ** 2
            energies.append(float(np.sum(power) * dt_s))
            norm = float(np.linalg.norm(power))
            change = (
                float("nan")
                if previous is None or norm == 0
                else float(np.linalg.norm(power - previous) / norm)
            )
            changes.append(change)
            previous = power
            if tol is not None and change < tol:
                converged = True
                break
        return CavityResult(
            output=np.asarray(out, dtype=np.complex128),
            intracavity=np.asarray(x, dtype=np.complex128),
            output_energy_j=np.asarray(energies),
            change=np.asarray(changes),
            converged=converged,
        )
