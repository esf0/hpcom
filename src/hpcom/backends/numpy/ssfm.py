"""Symmetric split-step Fourier propagation (scalar NLSE and Manakov) in NumPy.

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

``N`` does not change ``|A|``, so running the same steps in reverse order
with ``h → −h`` is the exact inverse: backward propagation undoes forward
propagation to rounding error. Derivation and order of accuracy:
``docs/numerics.rst``.
"""

from __future__ import annotations

import numpy as np

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import (
    Propagator,
    check_field_shape,
    effective_gamma_per_m,
    linear_operator_exponent,
    merged_linear_steps,
    step_sizes_m,
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
        sizes = step_sizes_m(fiber, direction)
        exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
        gamma_eff = effective_gamma_per_m(fiber)
        dual = fiber.polarization == "dual"
        uniform = fiber.step_distribution == "uniform"
        if uniform:
            d_full = np.exp(exponent * sizes[0])
            d_half = np.exp(exponent * (sizes[0] / 2.0))
        else:
            merged = merged_linear_steps(sizes)

        spectrum = np.fft.fft(a, axis=-1) * (d_half if uniform else np.exp(exponent * merged[0]))
        n_steps = sizes.size
        for step in range(n_steps):
            a = np.fft.ifft(spectrum, axis=-1)
            power = a.real**2 + a.imag**2
            if dual:
                power = power.sum(axis=-2, keepdims=True)  # |A_x|² + |A_y|²
            a *= np.exp(1j * (gamma_eff * sizes[step]) * power)
            spectrum = np.fft.fft(a, axis=-1)
            if uniform:
                spectrum *= d_full if step < n_steps - 1 else d_half
            else:
                spectrum *= np.exp(exponent * merged[step + 1])
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out
