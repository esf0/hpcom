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
    ALL_METHODS,
    LOCAL_ERROR_GROWTH,
    Propagator,
    attenuation_per_m,
    check_field_shape,
    effective_gamma_per_m,
    kerr_filter_response,
    linear_operator_exponent,
    signed_length_m,
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


@tf.function(reduce_retracing=True)
def _rk4ip(a: Any, exponent: Any, sizes: Any, gamma_eff: Any, dual: bool) -> Any:
    """Fourth-order Runge–Kutta in the interaction picture (Hult 2007), per-step lengths."""

    def kerr(x: Any) -> Any:
        power = tf.math.real(x) ** 2 + tf.math.imag(x) ** 2
        if dual:
            power = tf.reduce_sum(power, axis=-2, keepdims=True)
        return tf.complex(tf.zeros_like(power), gamma_eff * power) * x

    for i in tf.range(tf.shape(sizes)[0]):
        h = tf.cast(sizes[i], tf.complex128)
        half = tf.exp(exponent * (h / 2.0))
        a_ip = tf.signal.ifft(tf.signal.fft(a) * half)
        k1 = tf.signal.ifft(tf.signal.fft(h * kerr(a)) * half)
        k2 = h * kerr(a_ip + k1 / 2.0)
        k3 = h * kerr(a_ip + k2 / 2.0)
        k4 = h * kerr(tf.signal.ifft(tf.signal.fft(a_ip + k3) * half))
        a = tf.signal.ifft(tf.signal.fft(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0) * half) + k4 / 6.0
    return a


@tf.function(reduce_retracing=True)
def _local_error(
    a: Any,
    exponent: Any,
    total: Any,
    sign: Any,
    first_step: Any,
    goal: Any,
    gamma_eff: Any,
    alpha_eff: Any,
    response: Any,
    dual: bool,
    filtered: bool,
    effective: bool,
) -> Any:
    """Local-error method (Sinkin et al. 2003) as a ``tf.while_loop``; same decisions as NumPy."""

    def kerr(spectrum: Any, h: Any) -> Any:
        length = 2.0 * tf.sinh(alpha_eff * h / 2.0) / alpha_eff if effective else h
        x = _kerr(
            tf.signal.ifft(spectrum), gamma_eff * length, dual, response if filtered else None
        )
        return tf.signal.fft(x)

    def keep_going(z: Any, step: Any, spectrum: Any) -> Any:
        return tf.logical_and(z < total * (1.0 - 1e-14), step >= total * 1e-12)

    def advance(z: Any, step: Any, spectrum: Any) -> Any:
        big = tf.minimum(step, total - z)
        h = sign * big / 2.0
        d_h = tf.exp(exponent * tf.cast(h, tf.complex128))
        d_half = tf.exp(exponent * tf.cast(h / 2.0, tf.complex128))
        coarse = kerr(spectrum * d_h, 2.0 * h) * d_h
        fine = kerr(kerr(spectrum * d_half, h) * d_h, h) * d_half
        delta = tf.norm(fine - coarse) / tf.norm(fine)
        delta = tf.math.real(delta)
        reject = delta > 2.0 * goal
        new_spectrum = tf.cond(
            reject, lambda: spectrum, lambda: (4.0 / 3.0) * fine - (1.0 / 3.0) * coarse
        )
        new_z = tf.where(reject, z, z + big)
        new_step = tf.where(
            reject,
            big / 2.0,
            tf.where(
                delta > goal,
                big / LOCAL_ERROR_GROWTH,
                tf.where(delta < goal / 2.0, big * LOCAL_ERROR_GROWTH, big),
            ),
        )
        return new_z, new_step, new_spectrum

    z, _step, spectrum = tf.while_loop(
        keep_going,
        advance,
        (tf.constant(0.0, tf.float64), first_step, tf.signal.fft(a)),
    )
    return tf.signal.ifft(spectrum), z


class TensorFlowPropagator(Propagator):
    """Scalar NLSE / Manakov SSFM on TensorFlow (see module docstring)."""

    name = "tensorflow"
    supported_methods = ALL_METHODS

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
        elif fiber.scheme.method == "ssfm-local-error":
            length_m = signed_length_m(fiber, direction)
            response = kerr_filter_response(a.shape[-1], dt_s, fiber)
            effective = fiber.scheme.kerr_length == "effective" and attenuation_per_m(fiber) > 0
            out, reached = _local_error(
                tf.constant(a),
                tf.constant(exponent),
                tf.constant(abs(length_m), tf.float64),
                tf.constant(float(np.sign(length_m)), tf.float64),
                tf.constant(abs(float(sizes[0])), tf.float64),
                tf.constant(float(fiber.scheme.local_error or 0.0), tf.float64),
                tf.constant(gamma_eff, tf.float64),
                tf.constant(attenuation_per_m(fiber), tf.float64),
                tf.constant(
                    np.ones(a.shape[-1]) if response is None else response, dtype=tf.complex128
                ),
                dual,
                response is not None,
                effective,
            )
            if float(reached) < abs(length_m) * (1.0 - 1e-12):
                raise RuntimeError(
                    f"local-error method cannot reach local_error={fiber.scheme.local_error}"
                )
        elif fiber.scheme.method == "rk4ip":
            out = _rk4ip(
                tf.constant(a),
                tf.constant(exponent),
                tf.constant(sizes, dtype=tf.float64),
                tf.constant(gamma_eff, dtype=tf.float64),
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
