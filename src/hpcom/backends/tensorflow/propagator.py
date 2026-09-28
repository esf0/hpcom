"""Symmetric split-step Fourier propagation on TensorFlow (GPU when available).

Implements exactly the sub-step sequences of the NumPy reference backend
(:mod:`hpcom.backends.numpy.ssfm`) — ``D(m_J) N(c_J) … N(c_1) D(m_0)`` from
:func:`hpcom.backends.base.split_step_sequence`, for every step placement,
Kerr position, Kerr length and intensity filter — with the step loop
compiled with :func:`tf.function`. Uniform plain steps reuse one
precomputed ``D(h)``; other sequences evaluate each linear sub-step on the
device. Results agree with the NumPy backend to FFT rounding
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
    COMPOSITION_WEIGHTS,
    Propagator,
    check_field_shape,
    effective_gamma_per_m,
    kerr_filter_response,
    linear_operator_exponent,
    split_step_sequence,
    step_sizes_m,
    uses_fast_path,
)


def _kerr(a: Any, phase_per_watt: Any, dual: bool, response: Any = None) -> Any:
    power = tf.math.real(a) ** 2 + tf.math.imag(a) ** 2
    if dual:
        power = tf.reduce_sum(power, axis=-2, keepdims=True)
    if response is not None:
        spectrum = tf.signal.fft(tf.complex(power, tf.zeros_like(power))) * response
        power = tf.math.real(tf.signal.ifft(spectrum))
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
def _ssfm_sequence(
    a: Any, exponent: Any, linear: Any, phases: Any, response: Any, dual: bool, filtered: bool
) -> Any:
    """``D(m_J) N(c_J) … N(c_1) D(m_0)``: ``linear`` (J+1 lengths), ``phases`` (J, γ_eff·c)."""
    spectrum = tf.signal.fft(a) * tf.exp(exponent * tf.cast(linear[0], tf.complex128))
    for i in tf.range(tf.shape(phases)[0]):
        a = _kerr(tf.signal.ifft(spectrum), phases[i], dual, response if filtered else None)
        spectrum = tf.signal.fft(a) * tf.exp(exponent * tf.cast(linear[i + 1], tf.complex128))
    return tf.signal.ifft(spectrum)


class TensorFlowPropagator(Propagator):
    """Scalar NLSE / Manakov SSFM on TensorFlow (see module docstring)."""

    name = "tensorflow"
    supported_methods = frozenset(COMPOSITION_WEIGHTS)

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        a = np.asarray(field, dtype=np.complex128)
        check_field_shape(a, fiber)
        sizes = step_sizes_m(fiber, direction)  # validates n_steps
        exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
        gamma_eff = effective_gamma_per_m(fiber)
        dual = fiber.polarization == "dual"
        if uses_fast_path(fiber):
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
            linear, kerr = split_step_sequence(fiber, direction)
            response = kerr_filter_response(a.shape[-1], dt_s, fiber)
            out = _ssfm_sequence(
                tf.constant(a),
                tf.constant(exponent),
                tf.constant(linear, dtype=tf.float64),
                tf.constant(gamma_eff * kerr, dtype=tf.float64),
                tf.constant(
                    np.ones(a.shape[-1]) if response is None else response, dtype=tf.complex128
                ),
                dual,
                response is not None,
            )
        result: ComplexArray = np.asarray(out.numpy(), dtype=np.complex128)
        return result
