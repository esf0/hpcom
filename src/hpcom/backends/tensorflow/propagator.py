"""Symmetric split-step Fourier propagation on TensorFlow (GPU when available).

Implements exactly the step sequence of the NumPy reference backend
(:mod:`hpcom.backends.numpy.ssfm`) — ``D(h₁/2) N(h₁) D((h₁+h₂)/2) … N(h_K)
D(h_K/2)`` for any step placement — with the step loop compiled with
:func:`tf.function`. Uniform steps reuse one precomputed ``D(h)``; other
placements evaluate each merged linear sub-step on the device. Results agree with the NumPy backend to FFT rounding
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
    merged_linear_steps,
    step_sizes_m,
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


@tf.function(reduce_retracing=True)
def _ssfm_variable(a: Any, exponent: Any, merged: Any, phases: Any, dual: bool) -> Any:
    """Symmetric scheme with per-step lengths: ``merged`` (K+1 linear sub-steps), ``phases`` (K)."""
    spectrum = tf.signal.fft(a) * tf.exp(exponent * tf.cast(merged[0], tf.complex128))
    for i in tf.range(tf.shape(phases)[0]):
        a = _kerr(tf.signal.ifft(spectrum), phases[i], dual)
        spectrum = tf.signal.fft(a) * tf.exp(exponent * tf.cast(merged[i + 1], tf.complex128))
    return tf.signal.ifft(spectrum)


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
        sizes = step_sizes_m(fiber, direction)
        exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
        gamma_eff = effective_gamma_per_m(fiber)
        dual = fiber.polarization == "dual"
        if fiber.step_distribution == "uniform":
            h = sizes[0]
            out = _ssfm(
                tf.constant(a),
                tf.constant(np.exp(exponent * h)),
                tf.constant(np.exp(exponent * (h / 2.0))),
                tf.constant(gamma_eff * h, dtype=tf.float64),
                tf.constant(sizes.size, dtype=tf.int32),
                dual,
            )
        else:
            out = _ssfm_variable(
                tf.constant(a),
                tf.constant(exponent),
                tf.constant(merged_linear_steps(sizes), dtype=tf.float64),
                tf.constant(gamma_eff * sizes, dtype=tf.float64),
                dual,
            )
        result: ComplexArray = np.asarray(out.numpy(), dtype=np.complex128)
        return result
