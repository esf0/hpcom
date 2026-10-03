r"""Transceiver impairments as link components: lasers, IQ imbalance, quantisation, modulator.

Each element acts on the field like any :class:`~hpcom.link.Link` component
(``apply(field, dt_s, *, rng)``); put the transmitter-side ones first in the
link and the receiver-side ones last. Fields are ``(..., n)`` or, for
``polarization="dual"`` elements, ``(..., 2, n)``; leading axes are a batch.

* :class:`LaserPhaseNoise` — Wiener phase noise of the transmitter laser;
* :class:`LocalOscillator` — the receiver's LO: frequency offset and phase
  noise;
* :class:`IQImbalance` — gain and quadrature errors of an IQ modulator or
  receiver;
* :class:`Quantizer` — DAC or ADC resolution;
* :class:`MachZehnderModulator` — the sinusoidal transfer of an IQ modulator.

Digital back-propagation does not model them: their ``apply_inverse`` is the
identity. :func:`hpcom.autodiff.propagate_link` applies them in JAX
(random ones from the JAX key, the quantiser with a straight-through
gradient).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray

__all__ = [
    "IQImbalance",
    "LaserPhaseNoise",
    "LocalOscillator",
    "MachZehnderModulator",
    "Quantizer",
]


def _complex(field: ComplexArray) -> npt.NDArray[np.complex128]:
    return np.asarray(field, dtype=np.complex128)


def _check_polarization(a: np.ndarray, polarization: str, name: str) -> None:
    if polarization == "dual" and (a.ndim < 2 or a.shape[-2] != 2):
        raise ValueError(f"{name}(polarization='dual') expects a field (..., 2, n), got {a.shape}")


def wiener_phase(
    shape: tuple[int, ...], linewidth_hz: float, dt_s: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    r"""Wiener phase along the last axis: :math:`\phi_0 = 0`, increments :math:`N(0, 2\pi\Delta\nu\,\Delta t)`."""
    steps = rng.standard_normal(shape) * math.sqrt(2.0 * math.pi * linewidth_hz * dt_s)
    steps[..., 0] = 0.0
    out: npt.NDArray[np.float64] = np.cumsum(steps, axis=-1)
    return out


def _phase_shape(a: np.ndarray, polarization: str) -> tuple[int, ...]:
    """One phase per batch element, common to the polarisations."""
    return (*a.shape[:-2], 1, a.shape[-1]) if polarization == "dual" else a.shape


class _Transparent(HpcomModel):
    def apply_inverse(self, field: ComplexArray, dt_s: float) -> ComplexArray:
        """Identity: digital back-propagation does not model transceiver impairments."""
        del dt_s
        return _complex(field)


class LaserPhaseNoise(_Transparent):
    r"""Phase noise of the transmitter laser: :math:`A \mapsto Ae^{i\phi(t)}`.

    :math:`\phi` is a Wiener process (Lorentzian line of FWHM
    :math:`\Delta\nu`): :math:`\phi_0 = 0`,
    :math:`\phi_k - \phi_{k-1} \sim N(0, 2\pi\Delta\nu\,\Delta t)`, common to
    both polarisations, independent across a batch. Without ``rng`` a fresh
    generator is used (as :class:`~hpcom.EDFA` does).
    """

    linewidth_hz: float = Field(ge=0)
    polarization: Literal["single", "dual"] = "single"

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        a = _complex(field)
        _check_polarization(a, self.polarization, type(self).__name__)
        if self.linewidth_hz == 0.0:
            return a
        rng = np.random.default_rng() if rng is None else rng
        phase = wiener_phase(_phase_shape(a, self.polarization), self.linewidth_hz, dt_s, rng)
        out: ComplexArray = a * np.exp(1j * phase)
        return out


class LocalOscillator(_Transparent):
    r"""The receiver's local oscillator: :math:`A \mapsto Ae^{-i(2\pi\Delta f\,t + \phi(t))}`.

    ``frequency_offset_hz`` :math:`\Delta f = f_\mathrm{LO} - f_\mathrm{signal}`
    shifts the received spectrum by :math:`-\Delta f`;
    :math:`t_k = k\,\Delta t`. :math:`\phi` is the LO's Wiener phase noise
    (as :class:`LaserPhaseNoise`). Put it at the end of the link.
    """

    frequency_offset_hz: float = 0.0
    linewidth_hz: float = Field(default=0.0, ge=0)
    polarization: Literal["single", "dual"] = "single"

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        a = _complex(field)
        _check_polarization(a, self.polarization, type(self).__name__)
        phase = 2.0 * np.pi * self.frequency_offset_hz * dt_s * np.arange(a.shape[-1])
        if self.linewidth_hz > 0.0:
            rng = np.random.default_rng() if rng is None else rng
            phase = phase + wiener_phase(
                _phase_shape(a, self.polarization), self.linewidth_hz, dt_s, rng
            )
        out: ComplexArray = a * np.exp(-1j * phase)
        return out


class IQImbalance(_Transparent):
    r"""Gain and quadrature errors of an IQ modulator or receiver.

    With :math:`x = I + jQ`, the Q branch has gain :math:`g` (``amplitude_imbalance_db``,
    :math:`g = 10^{\mathrm{dB}/20}`) and a quadrature error :math:`\varphi`
    (``phase_imbalance_deg``):

    .. math:: y = I + jg(Q\cos\varphi - I\sin\varphi) = \mu x + \nu x^*,\qquad
       \mu = \frac{1 + ge^{-i\varphi}}{2},\quad \nu = \frac{1 - ge^{i\varphi}}{2},

    with image-rejection ratio
    :math:`|\mu|^2/|\nu|^2 = (1 + g^2 + 2g\cos\varphi)/(1 + g^2 - 2g\cos\varphi)`.
    Pointwise: any field layout.
    """

    amplitude_imbalance_db: float = 0.0
    phase_imbalance_deg: float = 0.0

    @property
    def coefficients(self) -> tuple[complex, complex]:
        """``(μ, ν)``."""
        g = 10.0 ** (self.amplitude_imbalance_db / 20.0)
        phi = math.radians(self.phase_imbalance_deg)
        return (1.0 + g * complex(math.cos(-phi), math.sin(-phi))) / 2.0, (
            1.0 - g * complex(math.cos(phi), math.sin(phi))
        ) / 2.0

    @property
    def image_rejection_ratio_db(self) -> float:
        mu, nu = self.coefficients
        return math.inf if nu == 0 else 10.0 * math.log10(abs(mu) ** 2 / abs(nu) ** 2)

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        del dt_s, rng
        a = _complex(field)
        mu, nu = self.coefficients
        out: ComplexArray = mu * a + nu * np.conj(a)
        return out


def _rms_per_dimension(a: np.ndarray) -> np.ndarray:
    """RMS of one real dimension per row (last axis): :math:`\\sqrt{\\langle|a|^2\\rangle/2}`."""
    out: np.ndarray = np.sqrt(np.mean(np.abs(a) ** 2, axis=-1, keepdims=True) / 2.0)
    return out


class Quantizer(_Transparent):
    r"""DAC or ADC resolution: a uniform mid-rise quantiser of Re and Im separately.

    ``bits`` :math:`b` give :math:`2^b` levels over the full scale
    :math:`\pm k\sigma` (``clipping_ratio`` :math:`k`, :math:`\sigma` the RMS of
    one real dimension of each row, i.e. per polarisation and block); step
    :math:`\Delta = 2k\sigma/2^b`, levels :math:`\Delta(m + 1/2)`, saturating at
    the outermost. For a Gaussian signal the SNR is
    :math:`6.02b + 4.77 - 20\log_{10}k` dB while clipping is negligible.
    """

    bits: int = Field(ge=1)
    clipping_ratio: float = Field(default=4.0, gt=0)

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        del dt_s, rng
        a = _complex(field)
        full_scale = self.clipping_ratio * _rms_per_dimension(a)
        step = np.where(full_scale > 0, 2.0 * full_scale / 2**self.bits, 1.0)  # zero rows stay zero
        top = 2 ** (self.bits - 1) - 1

        def quantise(x: np.ndarray) -> np.ndarray:
            index = np.clip(np.floor(x / step), -top - 1, top)
            return (index + 0.5) * step

        out: ComplexArray = np.where(full_scale > 0, quantise(a.real) + 1j * quantise(a.imag), 0)
        return out


class MachZehnderModulator(_Transparent):
    r"""IQ modulator with the sinusoidal transfer of two nested Mach–Zehnder modulators.

    Each quadrature, normalised by the RMS :math:`\sigma` of one real
    dimension of its row, drives a sine: :math:`I \mapsto
    \sin(d\,\tfrac{\pi}{2}\,I/\sigma)` (``drive`` :math:`d`: an RMS-level
    sample reaches :math:`d` of the way to the transfer maximum); the output
    is rescaled to the input's mean power. Small drive is linear, with an
    error :math:`\propto d^2`; larger drive trades loss (not modelled) for
    distortion.
    """

    drive: float = Field(gt=0, le=1)

    def apply(
        self, field: ComplexArray, dt_s: float, *, rng: np.random.Generator | None = None
    ) -> ComplexArray:
        del dt_s, rng
        a = _complex(field)
        sigma = _rms_per_dimension(a)
        scale = self.drive * math.pi / 2.0 / np.where(sigma > 0, sigma, 1.0)
        y = np.sin(scale * a.real) + 1j * np.sin(scale * a.imag)
        power_in = np.mean(np.abs(a) ** 2, axis=-1, keepdims=True)
        power_out = np.mean(np.abs(y) ** 2, axis=-1, keepdims=True)
        out: ComplexArray = y * np.sqrt(power_in / np.where(power_out > 0, power_out, 1.0))
        return out
