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
    Propagator,
    check_field_shape,
    effective_gamma_per_m,
    kerr_filter_response,
    linear_operator_exponent,
    split_step_sequence,
    step_sizes_m,
    uses_fast_path,
)

#: Precompute every distinct linear operator when there are at most this many.
_MAX_CACHED_OPERATORS = 16


def _kerr_power(a: np.ndarray, dual: bool, response: np.ndarray | None) -> np.ndarray:
    """|A|² (Manakov: |A_x|²+|A_y|²), low-pass filtered when a Kerr filter is set."""
    power: np.ndarray = a.real**2 + a.imag**2
    if dual:
        power = power.sum(axis=-2, keepdims=True)
    if response is not None:
        power = np.fft.ifft(np.fft.fft(power, axis=-1) * response, axis=-1).real
    return power


class NumpySSFMPropagator(Propagator):
    """Scalar NLSE / Manakov split-step schemes (see module docstring)."""

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
        exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
        gamma_eff = effective_gamma_per_m(fiber)
        dual = fiber.polarization == "dual"
        if uses_fast_path(fiber):
            return self._uniform(a, exponent, sizes, gamma_eff, dual)
        if fiber.scheme.method == "rk4ip":
            return self._rk4ip(a, exponent, sizes, gamma_eff, dual)
        linear, kerr = split_step_sequence(fiber, direction)
        response = kerr_filter_response(a.shape[-1], dt_s, fiber)
        return self._sequence(a, exponent, linear, kerr, gamma_eff, dual, response)

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
        a: np.ndarray, exponent: np.ndarray, sizes: np.ndarray, gamma_eff: float, dual: bool
    ) -> ComplexArray:
        """Fourth-order Runge–Kutta in the interaction picture (Hult 2007), 8 FFTs per step."""

        def kerr(x: np.ndarray) -> np.ndarray:  # N̂(x)·x = iγ_eff P(x) x
            out: np.ndarray = 1j * gamma_eff * _kerr_power(x, dual, None) * x
            return out

        for h in sizes:
            half = np.exp(exponent * (h / 2.0))

            def disperse(x: np.ndarray, half: np.ndarray = half) -> np.ndarray:
                out: np.ndarray = np.fft.ifft(np.fft.fft(x, axis=-1) * half, axis=-1)
                return out

            a_ip = disperse(a)
            k1 = disperse(h * kerr(a))
            k2 = h * kerr(a_ip + k1 / 2.0)
            k3 = h * kerr(a_ip + k2 / 2.0)
            k4 = h * kerr(disperse(a_ip + k3))
            a = disperse(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0) + k4 / 6.0
        result: ComplexArray = a
        return result

    @staticmethod
    def _sequence(
        a: np.ndarray,
        exponent: np.ndarray,
        linear: np.ndarray,
        kerr: np.ndarray,
        gamma_eff: float,
        dual: bool,
        response: np.ndarray | None,
    ) -> ComplexArray:
        """``D(m_J) N(c_J) … D(m_1) N(c_1) D(m_0)`` for any sub-step sequence."""
        distinct = np.unique(linear)
        cache = (
            {float(m): np.exp(exponent * m) for m in distinct}
            if distinct.size <= _MAX_CACHED_OPERATORS
            else None
        )

        def operator(m: float) -> np.ndarray:
            return cache[float(m)] if cache is not None else np.exp(exponent * m)

        spectrum = np.fft.fft(a, axis=-1) * operator(linear[0])
        for j, c in enumerate(kerr):
            a = np.fft.ifft(spectrum, axis=-1)
            a *= np.exp(1j * (gamma_eff * c) * _kerr_power(a, dual, response))
            spectrum = np.fft.fft(a, axis=-1) * operator(linear[j + 1])
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out
