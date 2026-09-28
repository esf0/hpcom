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
      ``exp[(−i β₁ω + i β₂ω²/2 − i β₃ω³/6 − α/2) z]`` and the Kerr term is
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


def step_boundaries_m(fiber: FiberLike) -> npt.NDArray[np.float64]:
    r"""Step boundaries :math:`0 = z_0 < z_1 < … < z_K = L` of a span, in metres.

    ``"uniform"``: :math:`z_n = nL/K`. ``"logarithmic"`` with factor
    :math:`k` (``fiber.step_log_factor``) and power attenuation :math:`\alpha`:

    .. math::

        z_n = -\frac{1}{k\alpha}\ln\!\left(1 - n\,\frac{1 - e^{-k\alpha L}}{K}\right),

    so every step carries the same share of :math:`\int e^{-k\alpha z}\,dz`
    (``k = 1``: equal nonlinear phase per step). Without loss it is uniform.
    Derivation and references: ``docs/numerics.rst``.
    """
    k_steps = int(fiber.n_steps)
    if k_steps < 1:
        raise ValueError(f"fiber.n_steps must be >= 1, got {k_steps}")
    length_m = fiber.length_km * 1e3
    n = np.arange(k_steps + 1, dtype=np.float64)
    if fiber.step_distribution == "uniform":
        return n * (length_m / k_steps)
    if fiber.step_distribution != "logarithmic":
        raise ValueError(f"unknown step_distribution {fiber.step_distribution!r}")
    rate = fiber.step_log_factor * fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3  # kα, 1/m
    if rate * length_m == 0.0:
        return n * (length_m / k_steps)
    delta = -np.expm1(-rate * length_m) / k_steps
    z: npt.NDArray[np.float64] = -np.log1p(-n * delta) / rate
    z[-1] = length_m  # exact end point despite rounding
    return z


def step_sizes_m(fiber: FiberLike, direction: Direction) -> npt.NDArray[np.float64]:
    """Signed step lengths in the order they are taken.

    Forward: :math:`h_1, …, h_K` from the span input. Backward: the same
    steps from the span output, negated (:math:`-h_K, …, -h_1`) — the exact
    reverse of the forward sequence.
    """
    sign = 1.0 if signed_length_m(fiber, direction) > 0 else -1.0
    if int(fiber.n_steps) < 1:
        raise ValueError(f"fiber.n_steps must be >= 1, got {fiber.n_steps}")
    if fiber.step_distribution == "uniform":
        h = np.full(int(fiber.n_steps), sign * fiber.length_km * 1e3 / int(fiber.n_steps))
        return h
    sizes = np.diff(step_boundaries_m(fiber))
    return sign * (sizes if sign > 0 else sizes[::-1])


def merged_linear_steps(sizes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lengths of the ``K+1`` linear sub-steps of the symmetric scheme.

    ``[h₁/2, (h₁+h₂)/2, …, (h_{K−1}+h_K)/2, h_K/2]`` — adjacent half steps merged.
    """
    merged = np.empty(sizes.size + 1)
    merged[0], merged[-1] = sizes[0] / 2.0, sizes[-1] / 2.0
    merged[1:-1] = (sizes[:-1] + sizes[1:]) / 2.0
    return merged


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
    """Per-metre exponent of the linear step, ``−i β₁ω + i β₂ω²/2 − i β₃ω³/6 − α/2``.

    ``ω`` is the angular-frequency grid in NumPy FFT order. The signs of the
    β₁ and β₃ terms follow from Agrawal's NLSE written in NumPy's FFT
    convention (see :doc:`/physics`): β₁ > 0 delays the field by β₁z. The β₂
    term is sign-convention independent.
    """
    omega = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=dt_s)
    alpha_per_m = fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3
    beta3 = fiber.beta3_s3_per_m
    exponent: npt.NDArray[np.complex128] = (
        -1j * fiber.beta1_s_per_m * omega
        + 1j * (fiber.beta2_s2_per_m / 2.0) * omega**2
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
