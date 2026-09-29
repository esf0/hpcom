"""Split-step Fourier propagation (scalar NLSE and Manakov) in NumPy.

This is hpcom's reference nonlinear propagator: dependency-free, deterministic,
and the implementation other backends (TensorFlow, future JAX/Rust) are
checked against by the conformance suite.

Scheme (second order in the step; loss is part of the linear operator
``D``), for steps ``h_1, …, h_K`` placed by ``fiber.step_distribution``
(:func:`hpcom.backends.base.step_boundaries_m`)::

    A(L) = D(h_K/2) · N(h_K) · D((h_{K−1}+h_K)/2) · … · D((h_1+h_2)/2) · N(h_1) · D(h_1/2) · A(0)

with ``D(h) = IFFT · exp(h·L̂(ω)) · FFT`` and
``N(h) = exp(i γ_eff h |A|²)``; ``L̂`` and ``γ_eff`` are defined in
:func:`hpcom.backends.base.linear_operator_exponent` and
:func:`~hpcom.backends.base.effective_gamma_per_m`. For uniform steps every
inner linear sub-step is ``D(h)`` and is computed once.

Every other split-step ``fiber.scheme`` (Kerr position, effective Kerr
length, filtered Kerr step, fourth-order compositions) and step placement
runs the general sub-step sequence of
:func:`hpcom.backends.base.split_step_sequence`, ``D(m_J) N(c_J) … N(c_1)
D(m_0)``, optionally with low-pass filtered power in ``N``. ``"rk4ip"`` is
the fourth-order Runge–Kutta in the interaction picture (Hult 2007).

Generalised-NLSE terms: any dispersion and loss spectrum are in ``L̂``; a
Raman response makes the power in ``N`` the convolution ``R * |A|²``
(:func:`hpcom.backends.base.kerr_power_response`), still a pure phase;
self-steepening (RK4IP only) multiplies the nonlinear term by
``1 − τω`` in the frequency domain (:func:`~hpcom.backends.base.shock_factor`).

``N`` does not change ``|A|``, so running the same sub-steps in reverse order
with negated lengths is the exact inverse: backward propagation undoes
forward propagation to rounding error. Derivation and order of accuracy:
``docs/numerics.rst``.
"""

from __future__ import annotations

import numpy as np

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import (
    ALL_METHODS,
    LOCAL_ERROR_GROWTH,
    Propagator,
    attenuation_per_m,
    check_field_shape,
    core_axis,
    effective_gamma_per_m,
    kerr_power_response,
    linear_operator_exponent,
    multicore_linear_system,
    shock_factor,
    signed_length_m,
    split_step_sequence,
    step_sizes_m,
    uses_fast_path,
)

#: Linear operators cached per propagation (the first this many distinct lengths).
_MAX_CACHED_OPERATORS = 16


def _kerr_power(a: np.ndarray, dual: bool, response: np.ndarray | None) -> np.ndarray:
    """|A|² (Manakov: |A_x|²+|A_y|²), through the Raman response and Kerr filter if set."""
    power: np.ndarray = a.real**2 + a.imag**2
    if dual:
        power = power.sum(axis=-2, keepdims=True)
    if response is not None:
        power = np.fft.ifft(np.fft.fft(power, axis=-1) * response, axis=-1).real
    return power


class _Disperser:
    """Exact linear step ``spectrum ↦ e^{m L̂} spectrum`` (single- or multicore)."""

    def __init__(self, fiber: FiberLike, n_samples: int, dt_s: float) -> None:
        self._cache: dict[float, np.ndarray] = {}
        if fiber.n_cores == 1:
            self._exponent = linear_operator_exponent(n_samples, dt_s, fiber)
            self._vectors = None
        else:
            vectors, self._exponent = multicore_linear_system(n_samples, dt_s, fiber)
            self._vectors = vectors
            self._adjoint = np.conj(np.swapaxes(vectors, -1, -2))
            self._axis = core_axis(fiber)

    def factor(self, m: float) -> np.ndarray:
        key = float(m)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value: np.ndarray = np.exp(self._exponent * m)
        if len(self._cache) < _MAX_CACHED_OPERATORS:
            self._cache[key] = value
        return value

    def __call__(self, spectrum: np.ndarray, m: float) -> np.ndarray:
        if self._vectors is None:
            out: np.ndarray = spectrum * self.factor(m)
            return out
        x = np.moveaxis(spectrum, self._axis, -1)  # (..., n, M)
        z = (self._adjoint @ x[..., None])[..., 0] * self.factor(m)
        y = (self._vectors @ z[..., None])[..., 0]
        moved: np.ndarray = np.moveaxis(y, -1, self._axis)
        return moved


class NumpySSFMPropagator(Propagator):
    """Scalar NLSE / Manakov / multicore propagation schemes (see module docstring)."""

    name = "numpy"
    supported_methods = ALL_METHODS

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        a = np.array(field, dtype=np.complex128)  # always a private copy
        check_field_shape(a, fiber)
        sizes = step_sizes_m(fiber, direction)  # validates n_steps
        gamma_eff = effective_gamma_per_m(fiber)
        dual = fiber.polarization == "dual"
        if uses_fast_path(fiber):
            exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
            return self._uniform(a, exponent, sizes, gamma_eff, dual)
        disperse = _Disperser(fiber, a.shape[-1], dt_s)
        response = kerr_power_response(a.shape[-1], dt_s, fiber)
        if fiber.scheme.method == "rk4ip":
            shock = shock_factor(a.shape[-1], dt_s, fiber)
            return self._rk4ip(a, disperse, sizes, gamma_eff, dual, response, shock)
        if fiber.scheme.method == "ssfm-local-error":
            return self._local_error(
                a,
                disperse,
                signed_length_m(fiber, direction),
                float(sizes[0]),
                float(fiber.scheme.local_error or 0.0),
                gamma_eff,
                dual,
                response,
                attenuation_per_m(fiber) if fiber.scheme.kerr_length == "effective" else 0.0,
            )
        linear, kerr = split_step_sequence(fiber, direction)
        return self._sequence(a, disperse, linear, kerr, gamma_eff, dual, response)

    @staticmethod
    def _uniform(
        a: np.ndarray, exponent: np.ndarray, sizes: np.ndarray, gamma_eff: float, dual: bool
    ) -> ComplexArray:
        """``D(h/2) [N(h) D(h)]^{K−1} N(h) D(h/2)`` with one precomputed ``D(h)``."""
        h = sizes[0]
        d_full = np.exp(exponent * h)
        d_half = np.exp(exponent * (h / 2.0))
        spectrum = np.fft.fft(a, axis=-1) * d_half
        n_steps = sizes.size
        for step in range(n_steps):
            a = np.fft.ifft(spectrum, axis=-1)
            a *= np.exp(1j * (gamma_eff * h) * _kerr_power(a, dual, None))
            spectrum = np.fft.fft(a, axis=-1)
            spectrum *= d_full if step < n_steps - 1 else d_half
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out

    @staticmethod
    def _rk4ip(
        a: np.ndarray,
        disperse: _Disperser,
        sizes: np.ndarray,
        gamma_eff: float,
        dual: bool,
        response: np.ndarray | None,
        shock: np.ndarray | None,
    ) -> ComplexArray:
        """Fourth-order Runge–Kutta in the interaction picture (Hult 2007), 8 FFTs per step.

        Each of the 4 nonlinear evaluations costs 2 more FFTs with a Raman
        response and 2 more with self-steepening.
        """

        def kerr(x: np.ndarray) -> np.ndarray:  # iγ_eff (1 + iτ∂t)[x · R*P(x)]
            out: np.ndarray = 1j * gamma_eff * _kerr_power(x, dual, response) * x
            if shock is not None:
                out = np.fft.ifft(np.fft.fft(out, axis=-1) * shock, axis=-1)
            return out

        for h in sizes:

            def half(x: np.ndarray, h: float = float(h)) -> np.ndarray:
                out: np.ndarray = np.fft.ifft(disperse(np.fft.fft(x, axis=-1), h / 2.0), axis=-1)
                return out

            a_ip = half(a)
            k1 = half(h * kerr(a))
            k2 = h * kerr(a_ip + k1 / 2.0)
            k3 = h * kerr(a_ip + k2 / 2.0)
            k4 = h * kerr(half(a_ip + k3))
            a = half(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0) + k4 / 6.0
        result: ComplexArray = a
        return result

    @staticmethod
    def _local_error(
        a: np.ndarray,
        disperse: _Disperser,
        length_m: float,
        first_step_m: float,
        goal: float,
        gamma_eff: float,
        dual: bool,
        response: np.ndarray | None,
        alpha_eff: float,
    ) -> ComplexArray:
        """Local-error method (Sinkin et al. 2003) in the spectral domain, 6 FFTs per 2h."""
        sign, total = np.sign(length_m), abs(length_m)

        def kerr_length(h: float) -> float:  # symmetric step: 2 sinh(αh/2)/α, or h
            return 2.0 * np.sinh(alpha_eff * h / 2.0) / alpha_eff if alpha_eff else h

        def kerr(spectrum: np.ndarray, h: float) -> np.ndarray:
            x = np.fft.ifft(spectrum, axis=-1)
            x *= np.exp(1j * (gamma_eff * kerr_length(h)) * _kerr_power(x, dual, response))
            out: np.ndarray = np.fft.fft(x, axis=-1)
            return out

        spectrum: np.ndarray = np.fft.fft(a, axis=-1)
        z, step = 0.0, abs(first_step_m)
        while z < total * (1.0 - 1e-14):
            big = min(step, total - z)
            h = sign * big / 2.0
            coarse = disperse(kerr(disperse(spectrum, h), 2.0 * h), h)
            fine = disperse(kerr(disperse(kerr(disperse(spectrum, h / 2.0), h), h), h), h / 2.0)
            delta = np.linalg.norm(fine - coarse) / np.linalg.norm(fine)
            if not np.isfinite(delta):
                raise FloatingPointError("local-error method: non-finite field")
            if delta > 2.0 * goal:
                step = big / 2.0
                if step < total * 1e-12:
                    raise RuntimeError(f"local-error method cannot reach local_error={goal}")
                continue
            spectrum = (4.0 / 3.0) * fine - (1.0 / 3.0) * coarse
            z += big
            if delta > goal:
                step = big / LOCAL_ERROR_GROWTH
            elif delta < goal / 2.0:
                step = big * LOCAL_ERROR_GROWTH
            else:
                step = big
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out

    @staticmethod
    def _sequence(
        a: np.ndarray,
        disperse: _Disperser,
        linear: np.ndarray,
        kerr: np.ndarray,
        gamma_eff: float,
        dual: bool,
        response: np.ndarray | None,
    ) -> ComplexArray:
        """``D(m_J) N(c_J) … D(m_1) N(c_1) D(m_0)`` for any sub-step sequence."""
        spectrum = disperse(np.fft.fft(a, axis=-1), linear[0])
        for j, c in enumerate(kerr):
            a = np.fft.ifft(spectrum, axis=-1)
            a *= np.exp(1j * (gamma_eff * c) * _kerr_power(a, dual, response))
            spectrum = disperse(np.fft.fft(a, axis=-1), linear[j + 1])
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out
