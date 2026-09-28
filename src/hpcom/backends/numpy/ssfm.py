"""Symmetric split-step Fourier propagation (scalar NLSE and Manakov) in NumPy.

This is hpcom's reference nonlinear propagator: dependency-free, deterministic,
and the implementation other backends (TensorFlow, future JAX/Rust) are
checked against by the conformance suite.

Scheme (second order in the step ``h = L / n_steps``; loss is part of the
linear operator ``D``)::

    A(L) = D(h/2) · N(h) · D(h) · N(h) · … · D(h) · N(h) · D(h/2) · A(0)
           └────────────── n_steps nonlinear steps ──────────────┘

with ``D(h) = IFFT · exp(h·L̂(ω)) · FFT`` and
``N(h) = exp(i γ_eff h |A|²)``; ``L̂`` and ``γ_eff`` are defined in
:func:`hpcom.backends.base.linear_operator_exponent` and
:func:`~hpcom.backends.base.effective_gamma_per_m`. ``N`` does not change
``|A|``, so running the same sequence with ``h → −h`` is the exact inverse:
backward propagation undoes forward propagation to rounding error.
"""

from __future__ import annotations

import numpy as np

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import (
    Propagator,
    check_field_shape,
    effective_gamma_per_m,
    linear_operator_exponent,
    signed_length_m,
)


class NumpySSFMPropagator(Propagator):
    """Scalar NLSE / Manakov symmetric SSFM (see module docstring)."""

    name = "numpy"

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        a = np.array(field, dtype=np.complex128)  # always a private copy
        check_field_shape(a, fiber)
        length_m = signed_length_m(fiber, direction)
        n_steps = int(fiber.n_steps)
        if n_steps < 1:
            raise ValueError(f"fiber.n_steps must be >= 1, got {n_steps}")

        h = length_m / n_steps
        exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
        d_full = np.exp(exponent * h)
        d_half = np.exp(exponent * (h / 2.0))
        phase_per_watt = effective_gamma_per_m(fiber) * h
        dual = fiber.polarization == "dual"

        spectrum = np.fft.fft(a, axis=-1) * d_half
        for step in range(n_steps):
            a = np.fft.ifft(spectrum, axis=-1)
            power = a.real**2 + a.imag**2
            if dual:
                power = power.sum(axis=-2, keepdims=True)  # |A_x|² + |A_y|²
            a *= np.exp(1j * phase_per_watt * power)
            spectrum = np.fft.fft(a, axis=-1)
            spectrum *= d_full if step < n_steps - 1 else d_half
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out
