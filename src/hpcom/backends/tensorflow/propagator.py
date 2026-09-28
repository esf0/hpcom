"""Symmetric split-step Fourier propagation on TensorFlow (GPU when available).

Implements exactly the step sequence of the NumPy reference backend
(:mod:`hpcom.backends.numpy.ssfm`) — ``D(h/2) [N(h) D(h)]… N(h) D(h/2)`` —
with the linear operator precomputed in NumPy and the step loop compiled with
:func:`tf.function`. Results agree with the NumPy backend to FFT rounding
(≈1e-13 relative) and with the original ``ssfm-gpu`` kernels this backend
replaces (golden data in ``tests/numerical/_data/tf_parity_v0.npz``).

All computation is in ``complex128``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import (
    Propagator,
    check_field_shape,
    effective_gamma_per_m,
    linear_operator_exponent,
    signed_length_m,
)


def _kerr(a: Any, phase_per_watt: Any, dual: bool) -> Any:
    power = tf.math.real(a) ** 2 + tf.math.imag(a) ** 2
    if dual:
        power = tf.reduce_sum(power, axis=-2, keepdims=True)
    phase = phase_per_watt * power
    return a * tf.exp(tf.complex(tf.zeros_like(phase), phase))


@tf.function(reduce_retracing=True)
def _ssfm(a: Any, d_full: Any, d_half: Any, phase_per_watt: Any, n_steps: Any, dual: bool) -> Any:
    spectrum = tf.signal.fft(a) * d_half
    a = _kerr(tf.signal.ifft(spectrum), phase_per_watt, dual)
    spectrum = tf.signal.fft(a)
    for _ in tf.range(n_steps - 1):
        a = _kerr(tf.signal.ifft(spectrum * d_full), phase_per_watt, dual)
        spectrum = tf.signal.fft(a)
    return tf.signal.ifft(spectrum * d_half)


class TensorFlowPropagator(Propagator):
    """Scalar NLSE / Manakov SSFM on TensorFlow (see module docstring)."""

    name = "tensorflow"

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
        n_steps = int(fiber.n_steps)
        if n_steps < 1:
            raise ValueError(f"fiber.n_steps must be >= 1, got {n_steps}")

        h = length_m / n_steps
        exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
        out = _ssfm(
            tf.constant(a),
            tf.constant(np.exp(exponent * h)),
            tf.constant(np.exp(exponent * (h / 2.0))),
            tf.constant(effective_gamma_per_m(fiber) * h, dtype=tf.float64),
            tf.constant(n_steps, dtype=tf.int32),
            fiber.polarization == "dual",
        )
        result: ComplexArray = np.asarray(out.numpy(), dtype=np.complex128)
        return result
