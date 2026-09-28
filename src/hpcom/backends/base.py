"""Abstract base class and shared helpers for fiber propagation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray, Direction, FiberLike


class BackendUnavailableError(RuntimeError):
    """A registered backend cannot be used because its dependencies are missing."""


class CPUFallbackWarning(RuntimeWarning):
    """A span ran on the NumPy (CPU) backend because the requested one lacks its scheme."""


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

    #: ``Scheme.method`` values this backend implements (with every Kerr-step
    #: option and step placement). Others run on NumPy with a
    #: :class:`CPUFallbackWarning`.
    supported_methods: frozenset[str] = frozenset({"ssfm"})

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


_CBRT2, _CBRT4 = 2.0 ** (1.0 / 3.0), 4.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)
_SUZUKI_P = 1.0 / (4.0 - _CBRT4)

#: Sub-step weights of the symmetric compositions of the Strang step, per method:
#: Yoshida's triple jump and Suzuki's fractal (5-stage) fourth-order compositions.
COMPOSITION_WEIGHTS: dict[str, tuple[float, ...]] = {
    "ssfm": (1.0,),
    "ssfm4-yoshida": (_YOSHIDA_W1, 1.0 - 2.0 * _YOSHIDA_W1, _YOSHIDA_W1),
    "ssfm4-suzuki": (_SUZUKI_P, _SUZUKI_P, 1.0 - 4.0 * _SUZUKI_P, _SUZUKI_P, _SUZUKI_P),
}

#: Every ``Scheme.method`` hpcom defines.
ALL_METHODS: frozenset[str] = frozenset(COMPOSITION_WEIGHTS) | {"rk4ip", "ssfm-local-error"}

#: Step-size factors of the local-error controller (Sinkin et al. 2003).
LOCAL_ERROR_GROWTH = 2.0 ** (1.0 / 3.0)


def attenuation_per_m(fiber: FiberLike) -> float:
    """Power attenuation α in 1/m (``P(z) = P(0)·e^{−αz}``)."""
    return float(fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3)


def uses_fast_path(fiber: FiberLike) -> bool:
    """Uniform steps with the plain symmetric split step: one precomputed ``D(h)``."""
    s = fiber.scheme
    return (
        fiber.step_distribution == "uniform"
        and s.method == "ssfm"
        and s.nonlinear_position == 0.5
        and s.kerr_length == "step"
        and s.kerr_filter_bandwidth_hz is None
    )


def split_step_sequence(
    fiber: FiberLike, direction: Direction
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Signed linear and Kerr sub-step lengths of a span, in the order they are applied.

    Returns ``(m, c)`` with ``m`` of length ``J+1`` and ``c`` of length ``J``:

    .. math:: A_\text{out} = D(m_J)\,N(c_J)\cdots D(m_1)\,N(c_1)\,D(m_0)\,A_\text{in}.

    Each step :math:`h` (split into sub-steps :math:`w_i h` for compositions)
    becomes linear :math:`r h`, Kerr, linear :math:`(1-r)h`, with :math:`r`
    the Kerr position in the fiber's coordinate — :math:`1-r` along the
    traversal when going backward, so that backward is the exact inverse.
    ``c`` is the step length, or the effective length
    :math:`(e^{\alpha r|h|} - e^{-\alpha(1-r)|h|})/\alpha` with the sign of
    :math:`h`. See ``docs/numerics.rst``.
    """
    scheme = fiber.scheme
    if scheme.method not in COMPOSITION_WEIGHTS:
        raise ValueError(f"{scheme.method!r} is not a split-step method")
    sizes = step_sizes_m(fiber, direction)
    weights = np.asarray(COMPOSITION_WEIGHTS[scheme.method], dtype=np.float64)
    sub = (sizes[:, None] * weights[None, :]).ravel()
    r = float(scheme.nonlinear_position)
    r_trav = r if direction == "forward" else 1.0 - r
    pre, post = r_trav * sub, (1.0 - r_trav) * sub
    linear = np.empty(sub.size + 1)
    linear[0], linear[-1] = pre[0], post[-1]
    linear[1:-1] = post[:-1] + pre[1:]
    alpha = attenuation_per_m(fiber)
    if scheme.kerr_length == "step" or alpha == 0.0:
        return linear, sub
    mag = np.abs(sub)
    kerr = np.sign(sub) * (np.exp(alpha * r * mag) - np.exp(-alpha * (1.0 - r) * mag)) / alpha
    return linear, kerr


def kerr_filter_response(
    n_samples: int, dt_s: float, fiber: FiberLike
) -> npt.NDArray[np.float64] | None:
    """Real, even transfer function of the Kerr-step intensity filter (``None``: no filter)."""
    scheme = fiber.scheme
    fc = scheme.kerr_filter_bandwidth_hz
    if fc is None:
        return None
    f = np.fft.fftfreq(n_samples, d=dt_s)
    if scheme.kerr_filter_shape == "gaussian":
        response: npt.NDArray[np.float64] = np.exp(-np.log(2.0) / 2.0 * (f / fc) ** 2)
        return response
    if scheme.kerr_filter_shape == "rectangular":
        window: npt.NDArray[np.float64] = (np.abs(f) <= fc).astype(np.float64)
        return window
    raise ValueError(f"unknown kerr_filter_shape {scheme.kerr_filter_shape!r}")


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
    alpha_per_m = attenuation_per_m(fiber)
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
