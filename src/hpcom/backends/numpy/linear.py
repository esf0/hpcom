"""Exact linear fiber propagation (loss + dispersion) in NumPy.

Solves the NLSE with ``γ = 0`` in closed form — one FFT, one multiply, one
inverse FFT — using the same linear operator as the SSFM backends
(:func:`hpcom.backends.base.linear_operator_exponent`). ``fiber.n_steps`` and
``fiber.gamma_per_w_per_km`` are ignored. Useful as an analytic reference and
for fast linear-regime studies.
"""

from __future__ import annotations

import numpy as np

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import (
    Propagator,
    check_field_shape,
    linear_operator_exponent,
    signed_length_m,
)


class LinearNumpyPropagator(Propagator):
    """Linear (dispersion + loss) propagation; the Kerr term is ignored."""

    name = "numpy-linear"

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        a = np.asarray(field, dtype=np.complex128)
        check_field_shape(a, fiber)
        length_m = signed_length_m(fiber, direction)
        transfer = np.exp(linear_operator_exponent(a.shape[-1], dt_s, fiber) * length_m)
        out: ComplexArray = np.fft.ifft(np.fft.fft(a, axis=-1) * transfer, axis=-1)
        return out
