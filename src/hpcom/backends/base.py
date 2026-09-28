"""Abstract base class and shared helpers for fiber propagation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray, Direction, FiberLike


class BackendUnavailableError(RuntimeError):
    """A registered backend cannot be used because its dependencies are missing."""


class Propagator(ABC):
    """Backend-agnostic propagation of an optical field through one fiber span.

    **Contract** (checked for every registered backend by the conformance
    suite in ``tests/backends/``):

    * ``fiber.polarization == "single"``: ``field`` has shape
      ``(..., n_samples)`` and is propagated with the scalar NLSE.
      ``"dual"``: shape ``(..., 2, n_samples)`` (X, Y) and the Manakov
      equation. Leading axes are independent batch dimensions.
    * The output has the input's shape and dtype ``complex128``; the input is
      not modified.
    * Equation and sign conventions are those of
      :doc:`/physics` — in NumPy's FFT convention the linear operator is
      ``exp[(i β₂ω²/2 − i β₃ω³/6 − α/2) z]`` and the Kerr term is
      ``+i γ |A|² A`` (``+i (8/9) γ (|A_x|²+|A_y|²) A`` for Manakov).
    * ``direction="backward"`` applies the exact inverse step sequence (same
      steps, negated length), as used by digital back-propagation.
    """

    #: Stable identifier; conventionally equal to the registry name.
    name: str = ""

    @abstractmethod
    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        """Propagate ``field`` (sample period ``dt_s``) through one span of ``fiber``."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Helpers shared by the built-in backends
# ---------------------------------------------------------------------------


def signed_length_m(fiber: FiberLike, direction: Direction) -> float:
    """Span length in metres, negated for backward propagation."""
    if direction == "forward":
        return fiber.length_km * 1e3
    if direction == "backward":
        return -fiber.length_km * 1e3
    raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")


def check_field_shape(field: np.ndarray, fiber: FiberLike) -> None:
    """Validate ``field`` against ``fiber.polarization`` (see :class:`Propagator`)."""
    if fiber.polarization == "single":
        if field.ndim < 1:
            raise ValueError("field must have at least one (time) axis")
    elif fiber.polarization == "dual":
        if field.ndim < 2 or field.shape[-2] != 2:
            raise ValueError(
                "a dual-polarisation fiber (Manakov) expects a field of shape "
                f"(..., 2, n_samples); got {field.shape}"
            )
    else:
        raise ValueError(f"unknown fiber polarization {fiber.polarization!r}")


def linear_operator_exponent(
    n_samples: int, dt_s: float, fiber: FiberLike
) -> npt.NDArray[np.complex128]:
    """Per-metre exponent of the linear step, ``i β₂ω²/2 − i β₃ω³/6 − α/2``.

    ``ω`` is the angular-frequency grid in NumPy FFT order. The sign of the
    β₃ term follows from Agrawal's NLSE written in NumPy's FFT convention (see
    :doc:`/physics`); the β₂ term is sign-convention independent.
    """
    omega = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=dt_s)
    alpha_per_m = fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3
    beta3 = fiber.beta3_s3_per_m
    exponent: npt.NDArray[np.complex128] = (
        1j * (fiber.beta2_s2_per_m / 2.0) * omega**2
        - 1j * (beta3 / 6.0) * omega**3
        - alpha_per_m / 2.0
    ).astype(np.complex128)
    return exponent


def effective_gamma_per_m(fiber: FiberLike) -> float:
    """Kerr coefficient in 1/(W·m) as used by the nonlinear step.

    Includes the Manakov 8/9 factor (polarisation averaging over the
    Poincaré sphere; Marcuse, Menyuk & Wai, JLT 15(9) 1997) for
    dual-polarisation fibers.
    """
    gamma = fiber.gamma_per_w_per_km * 1e-3
    return gamma * 8.0 / 9.0 if fiber.polarization == "dual" else gamma
