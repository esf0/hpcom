"""Optical amplifiers.

Currently provides :class:`EDFA`. New amplifier types (Raman, semiconductor,
…) plug in by subclassing :class:`Amplifier` and implementing
:meth:`Amplifier.amplify`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from hpcom._typing import ComplexArray

#: Speed of light in vacuum, m/s.
_C_LIGHT_M_PER_S = 299_792_458.0
#: Planck constant, J·s.
_H_PLANCK_J_S = 6.62607015e-34


@dataclass(frozen=True)
class Amplifier(ABC):
    """Common base for optical amplifiers.

    Implementations should accept either a 1-D scalar field of shape ``(N,)``
    or a 2-D dual-polarization field of shape ``(2, N)`` and return the same
    shape. Stochastic noise (if any) is injected via the ``rng`` argument so
    callers can pin reproducibility from above.
    """

    @abstractmethod
    def amplify(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """Return the amplified field. Implementations may add stochastic noise."""

    def apply(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """:class:`hpcom._typing.LinkComponent` adaptor — alias for :meth:`amplify`."""
        return self.amplify(field, dt_s, rng=rng)


@dataclass(frozen=True)
class EDFA(Amplifier):
    """Erbium-doped fibre amplifier with ASE noise.

    The amplifier scales the field amplitude by ``√G`` and adds zero-mean
    complex Gaussian ASE noise. Noise spectral density per polarization is

        N₀ = h · f_c · (G − 1) · NF

    in W/Hz, so the per-sample complex variance at sample rate ``f_s = 1/dt_s``
    is ``N₀ · f_s`` (split equally between real and imaginary parts).

    Accepts either a 1-D field ``(N,)`` (single pol) or a 2-D field ``(2, N)``
    (dual pol). For dual pol, each polarization independently receives ASE of
    variance ``N₀ · f_s`` — i.e. ``N₀`` is per-pol, not split across pols.
    """

    gain_db: float
    noise_figure_db: float = 5.0
    wavelength_m: float = 1550e-9

    def __post_init__(self) -> None:
        if self.gain_db < 0:
            raise ValueError(f"gain_db must be >= 0, got {self.gain_db}")
        if self.noise_figure_db < 0:
            raise ValueError(f"noise_figure_db must be >= 0, got {self.noise_figure_db}")
        if self.wavelength_m <= 0:
            raise ValueError(f"wavelength_m must be > 0, got {self.wavelength_m}")

    @property
    def gain_linear(self) -> float:
        return float(10.0 ** (self.gain_db / 10.0))

    @property
    def noise_figure_linear(self) -> float:
        return float(10.0 ** (self.noise_figure_db / 10.0))

    def ase_noise_density_w_per_hz(self) -> float:
        """One-sided ASE PSD per polarization, in W/Hz."""
        f_c = _C_LIGHT_M_PER_S / self.wavelength_m
        return _H_PLANCK_J_S * f_c * (self.gain_linear - 1.0) * self.noise_figure_linear

    def amplify(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        if field.ndim not in (1, 2):
            raise ValueError(f"EDFA expects 1-D or 2-D field, got {field.ndim}-D")
        if field.ndim == 2 and field.shape[0] != 2:
            raise ValueError(
                f"EDFA expects 2-D field with leading axis 2 (dual pol); got shape {field.shape}"
            )

        sqrt_gain = float(np.sqrt(self.gain_linear))
        amplified = field * sqrt_gain

        if self.gain_db == 0.0:
            # G = 1 → ASE PSD = 0, regardless of NF. Skip RNG entirely.
            return amplified.astype(np.complex128, copy=False)

        if rng is None:
            rng = np.random.default_rng()

        sample_rate_hz = 1.0 / dt_s
        sigma_per_quadrature = float(np.sqrt(self.ase_noise_density_w_per_hz() * sample_rate_hz / 2.0))
        # Independent ASE per polarization (and per sample). For dual-pol the
        # broadcast over field.shape produces a (2, N) noise array natively.
        noise = rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        return (amplified + sigma_per_quadrature * noise).astype(np.complex128, copy=False)
