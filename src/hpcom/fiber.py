"""Fiber span: parameters + propagation through a configured backend.

A :class:`Fiber` is the user-facing representation of a single optical fiber
span. It is a frozen dataclass — pure config, no signal state — so the same
instance can be reused safely across simulations and shared between threads.

Numerical work is delegated to a :class:`~hpcom.backends.base.Propagator`
selected by the ``backend`` field at ``propagate()`` time. This is the seam
that lets v1.1+ swap in JAX or Rust without touching scenario code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hpcom._typing import ComplexArray, Direction, Polarization
from hpcom.backends import get_propagator


def _dispersion_to_beta2_s2_per_m(
    dispersion_ps_per_nm_per_km: float, wavelength_m: float = 1550e-9
) -> float:
    """Convert chromatic dispersion D [ps/(nm·km)] to β₂ [s²/m] at given wavelength.

    β₂ = -λ² · D / (2π c). With D in ps/(nm·km) = 1e-6 s/m², the result is in s²/m.
    """
    c = 299_792_458.0
    d_si = dispersion_ps_per_nm_per_km * 1e-6  # s / m²
    return -(wavelength_m**2) * d_si / (2.0 * 3.141592653589793 * c)


@dataclass(frozen=True)
class Fiber:
    """A single span of optical fiber.

    All parameters are SI except where the variable name says otherwise
    (``length_km``, ``alpha_db_per_km``, ``gamma_per_w_per_km``).

    Defaults reflect a generic standard single-mode fiber (SSMF) at 1550 nm.
    For named presets use :meth:`Fiber.ssmf`.
    """

    length_km: float
    alpha_db_per_km: float = 0.2
    beta2_s2_per_m: float = -2.16e-26
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 100
    backend: str = "tensorflow"
    polarization: Polarization = "single"

    def __post_init__(self) -> None:
        if self.length_km <= 0:
            raise ValueError(f"length_km must be > 0, got {self.length_km}")
        if self.alpha_db_per_km < 0:
            raise ValueError(f"alpha_db_per_km must be >= 0, got {self.alpha_db_per_km}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.polarization not in ("single", "dual"):
            raise ValueError(
                f"polarization must be 'single' or 'dual', got {self.polarization!r}"
            )

    @classmethod
    def ssmf(
        cls,
        length_km: float,
        *,
        n_steps: int = 100,
        backend: str = "tensorflow",
    ) -> Fiber:
        """Standard single-mode fiber preset (D = 17 ps/nm/km, α = 0.2 dB/km, γ = 1.27 1/W/km)."""
        return cls(
            length_km=length_km,
            alpha_db_per_km=0.2,
            beta2_s2_per_m=_dispersion_to_beta2_s2_per_m(17.0),
            gamma_per_w_per_km=1.27,
            n_steps=n_steps,
            backend=backend,
        )

    def with_backend(self, backend: str) -> Fiber:
        """Return a copy of this fiber that propagates on a different backend."""
        return replace(self, backend=backend)

    def propagate(
        self,
        field: ComplexArray,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        """Propagate ``field`` (sample period ``dt_s``) through this span."""
        return get_propagator(self.backend).propagate(field, self, dt_s=dt_s, direction=direction)

    def apply(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """:class:`hpcom._typing.LinkComponent` adaptor — equivalent to forward :meth:`propagate`."""
        return self.propagate(field, dt_s, direction="forward")
