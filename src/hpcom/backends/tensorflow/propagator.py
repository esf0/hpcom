"""Symmetric split-step Fourier propagation on TensorFlow (GPU when available).

Implements exactly the sub-step sequences of the NumPy reference backend
(:mod:`hpcom.backends.numpy.ssfm`) — ``D(m_J) N(c_J) … N(c_1) D(m_0)`` from
:func:`hpcom.backends.base.split_step_sequence`, for every step placement,
Kerr position, Kerr length and intensity filter — with the step loop
compiled with :func:`tf.function`, for single- and multicore fibers, and
RK4IP and the local-error method, with every generalised-NLSE term (Raman
response, self-steepening) and the gain terms of :mod:`hpcom.laser`
(nonlinear gain, saturated gain). Uniform
plain single-core steps reuse one
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
    apply_jones,
    attenuation_per_m,
    check_field_shape,
    core_axis,
    effective_gamma_per_m,
    gain_spectrum,
    kerr_power_response,
    linear_operator_exponent,
    multicore_linear_system,
    nonlinear_gain_per_m,
    pmd_pieces,
    pmd_realization,
    saturation_energy_j,
    shock_factor,
    signed_length_m,
    small_signal_gain_per_m,
    split_step_sequence,
    step_sizes_m,
    uses_fast_path,
)


def _kerr(
    a: Any, phase_per_watt: Any, dual: bool, response: Any = None, gain_per_watt: Any = None
) -> Any:
    power = tf.math.real(a) ** 2 + tf.math.imag(a) ** 2
    if dual:
        power = tf.reduce_sum(power, axis=-2, keepdims=True)
    if response is not None:
        spectrum = tf.signal.fft(tf.complex(power, tf.zeros_like(power))) * response
        power = tf.math.real(tf.signal.ifft(spectrum))
    if gain_per_watt is not None:
        return a * _kerr_gain_factor(power, phase_per_watt, gain_per_watt)
    phase = phase_per_watt * power
    return a * tf.exp(tf.complex(tf.zeros_like(phase), phase))


def _kerr_gain_factor(power: Any, phase_per_watt: Any, gain_per_watt: Any) -> Any:
    """:func:`hpcom.backends.base.kerr_gain_factor` (NaN where 2εcP ≥ 1: the step diverges)."""
    u = 2.0 * gain_per_watt * power
    small = tf.abs(u) < 1e-8
    safe = tf.where(small, tf.fill(tf.shape(u), tf.constant(0.5, u.dtype)), u)
    log_ratio = tf.where(small, 1.0 + u / 2.0, -tf.math.log1p(-safe) / safe)
    phase = phase_per_watt * power * log_ratio
    amplitude = tf.cast(1.0 / tf.sqrt(1.0 - u), tf.complex128)
    return amplitude * tf.exp(tf.complex(tf.zeros_like(phase), phase))


@tf.function(reduce_retracing=True)
def _ssfm(a: Any, d_full: Any, d_half: Any, phase_per_watt: Any, n_steps: Any, dual: bool) -> Any:
    spectrum = tf.signal.fft(a) * d_half
    a = _kerr(tf.signal.ifft(spectrum), phase_per_watt, dual)
    spectrum = tf.signal.fft(a)
    for _ in tf.range(n_steps - 1):
        a = _kerr(tf.signal.ifft(spectrum * d_full), phase_per_watt, dual)
        spectrum = tf.signal.fft(a)
    return tf.signal.ifft(spectrum * d_half)


def _moveaxis(x: Any, source: int, destination: int) -> Any:
    rank = x.shape.rank
    order = list(range(rank))
    order.insert(destination % rank, order.pop(source % rank))
    return tf.transpose(x, order)


def _pmd_move(spectrum: Any, lin: Any, move: Any) -> Any:
    """Birefringence pieces of linear move ``move``: ``lin = (exponent, axes, sections, lengths, ω·Δβ₁/2)``."""
    _, axes, sections, lengths, half_dgd = lin
    for p in tf.range(tf.shape(sections)[1]):  # the widest move (shapes relax on retracing)
        b = tf.cast(tf.gather(axes, sections[move, p]), tf.complex128)
        theta = half_dgd * lengths[move, p]
        c = tf.cast(tf.cos(theta), tf.complex128)
        s = tf.cast(tf.sin(theta), tf.complex128)
        x, y = spectrum[..., 0, :], spectrum[..., 1, :]
        spectrum = tf.stack(
            [
                (c - 1j * s * b[2]) * x - 1j * s * (b[0] - 1j * b[1]) * y,
                -1j * s * (b[0] + 1j * b[1]) * x + (c + 1j * s * b[2]) * y,
            ],
            axis=-2,
        )
    return spectrum


def _disperse(spectrum: Any, m: Any, lin: Any, axis: int | None, move: Any = None) -> Any:
    """Exact linear step of length ``m``.

    ``lin = (exponent,)``; multicore ``(exponent, V, V†)``; with PMD
    ``(exponent, axes, sections, lengths, ω·Δβ₁/2)`` and the move's index.
    """
    mc = tf.cast(m, tf.complex128)
    if axis is None:
        out = spectrum * tf.exp(lin[0] * mc)
        return _pmd_move(out, lin, move) if len(lin) == 5 else out
    exponent, vectors, adjoint = lin
    x = _moveaxis(spectrum, axis, -1)  # (..., n, M)
    z = tf.linalg.matmul(adjoint, x[..., None])[..., 0] * tf.exp(exponent * mc)
    return _moveaxis(tf.linalg.matmul(vectors, z[..., None])[..., 0], -1, axis)


def _saturated(spectrum: Any, m: Any, lin: Any, sat: Any, dual: bool, move: Any) -> Any:
    """Linear step with saturated gain at the mid-step energy (``_Disperser._saturated``).

    ``sat = (g₀G(ω)/2, 1/E_sat, Δt/n)``.
    """
    half_gain, inv_sat, scale = sat

    def energy(x: Any) -> Any:
        power = tf.math.real(x) ** 2 + tf.math.imag(x) ** 2
        e = tf.reduce_sum(power, axis=-1, keepdims=True) * scale
        return tf.reduce_sum(e, axis=-2, keepdims=True) if dual else e

    def gain(e: Any) -> Any:
        return tf.cast(tf.exp(m * half_gain / (1.0 + e * inv_sat)), tf.complex128)

    base = spectrum * tf.exp(lin[0] * tf.cast(m, tf.complex128))
    e0 = energy(spectrum)
    out = base * gain((e0 + energy(base * gain(e0))) / 2.0)
    return _pmd_move(out, lin, move) if len(lin) == 5 else out


@tf.function(reduce_retracing=True)
def _ssfm_sequence(
    a: Any,
    lin: Any,
    linear: Any,
    phases: Any,
    response: Any,
    dual: bool,
    filtered: bool,
    axis: int | None,
    gains: Any = None,
    sat: Any = None,
) -> Any:
    """``D(m_J) N(c_J) … N(c_1) D(m_0)``: ``linear`` (J+1 lengths), ``phases`` (J, γ_eff·c).

    ``gains`` (J, ε·c) adds nonlinear gain; ``sat`` saturated gain.
    """

    def disperse(spectrum: Any, j: Any) -> Any:
        if sat is not None:
            return _saturated(spectrum, linear[j], lin, sat, dual, j)
        return _disperse(spectrum, linear[j], lin, axis, j)

    spectrum = disperse(tf.signal.fft(a), 0)
    for i in tf.range(tf.shape(phases)[0]):
        a = _kerr(
            tf.signal.ifft(spectrum),
            phases[i],
            dual,
            response if filtered else None,
            None if gains is None else gains[i],
        )
        spectrum = disperse(tf.signal.fft(a), i + 1)
    return tf.signal.ifft(spectrum)


@tf.function(reduce_retracing=True)
def _rk4ip(
    a: Any,
    lin: Any,
    sizes: Any,
    gamma_eff: Any,
    response: Any,
    shock: Any,
    dual: bool,
    axis: int | None,
    filtered: bool,
    shocked: bool,
    eps: Any = None,
) -> Any:
    """Fourth-order Runge–Kutta in the interaction picture (Hult 2007), per-step lengths."""

    def kerr(x: Any) -> Any:  # (iγ_eff + ε)(1 + iτ∂t)[x · R*P(x)]
        power = tf.math.real(x) ** 2 + tf.math.imag(x) ** 2
        if dual:
            power = tf.reduce_sum(power, axis=-2, keepdims=True)
        if filtered:
            spectrum = tf.signal.fft(tf.complex(power, tf.zeros_like(power))) * response
            power = tf.math.real(tf.signal.ifft(spectrum))
        if eps is None:
            out = tf.complex(tf.zeros_like(power), gamma_eff * power) * x
        else:
            out = tf.complex(eps * power, gamma_eff * power) * x
        if shocked:
            out = tf.signal.ifft(tf.signal.fft(out) * shock)
        return out

    for i in tf.range(tf.shape(sizes)[0]):
        h = sizes[i]
        hc = tf.cast(h, tf.complex128)

        def half(x: Any, move: Any, h: Any = h) -> Any:  # 2i: z → z + h/2; 2i + 1: to z + h
            return tf.signal.ifft(_disperse(tf.signal.fft(x), h / 2.0, lin, axis, move))

        a_ip = half(a, 2 * i)
        k1 = half(hc * kerr(a), 2 * i)
        k2 = hc * kerr(a_ip + k1 / 2.0)
        k3 = hc * kerr(a_ip + k2 / 2.0)
        k4 = hc * kerr(half(a_ip + k3, 2 * i + 1))
        a = half(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0, 2 * i + 1) + k4 / 6.0
    return a


@tf.function(reduce_retracing=True)
def _local_error(
    a: Any,
    lin: Any,
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
    axis: int | None,
    eps: Any = None,
) -> Any:
    """Local-error method (Sinkin et al. 2003) as a ``tf.while_loop``; same decisions as NumPy."""

    def kerr(spectrum: Any, h: Any) -> Any:
        length = 2.0 * tf.sinh(alpha_eff * h / 2.0) / alpha_eff if effective else h
        x = _kerr(
            tf.signal.ifft(spectrum),
            gamma_eff * length,
            dual,
            response if filtered else None,
            None if eps is None else eps * length,
        )
        return tf.signal.fft(x)

    def disperse(spectrum: Any, m: Any) -> Any:
        return _disperse(spectrum, m, lin, axis)

    def _norm(x: Any) -> Any:  # a scalar for any rank (tf.norm loses it under relaxed shapes)
        return tf.sqrt(tf.reduce_sum(tf.math.real(x) ** 2 + tf.math.imag(x) ** 2))

    def keep_going(z: Any, step: Any, spectrum: Any) -> Any:
        return tf.logical_and(z < total * (1.0 - 1e-14), step >= total * 1e-12)

    def advance(z: Any, step: Any, spectrum: Any) -> Any:
        big = tf.minimum(step, total - z)
        h = sign * big / 2.0
        coarse = disperse(kerr(disperse(spectrum, h), 2.0 * h), h)
        fine = disperse(kerr(disperse(kerr(disperse(spectrum, h / 2.0), h), h), h), h / 2.0)
        delta = _norm(fine - coarse) / _norm(fine)
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
    """Every hpcom scheme on TensorFlow, single- and multicore (see module docstring)."""

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
        gamma_eff = effective_gamma_per_m(fiber)
        dual = fiber.polarization == "dual"
        if uses_fast_path(fiber):
            exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
            h = sizes[0]
            out = _ssfm(
                tf.constant(a),
                tf.constant(np.exp(exponent * h)),
                tf.constant(np.exp(exponent * (h / 2.0))),
                tf.constant(gamma_eff * h, dtype=tf.float64),
                tf.constant(sizes.size, dtype=tf.int32),
                dual,
            )
            return np.asarray(out.numpy(), dtype=np.complex128)
        pmd = pmd_realization(fiber)
        if pmd is not None and direction == "backward":
            a = apply_jones(a, pmd.output_rotation.conj().T)
        if fiber.n_cores == 1:
            lin: tuple[Any, ...] = (
                tf.constant(linear_operator_exponent(a.shape[-1], dt_s, fiber)),
            )
            axis = None
            if pmd is not None:
                start = 0.0 if direction == "forward" else fiber.length_km * 1e3
                moves = (
                    np.repeat(sizes / 2.0, 2)
                    if fiber.scheme.method == "rk4ip"
                    else split_step_sequence(fiber, direction)[0]
                )
                sections, lengths = pmd_pieces(pmd, start, moves)
                omega = 2.0 * np.pi * np.fft.fftfreq(a.shape[-1], d=dt_s)
                lin = (
                    *lin,
                    tf.constant(pmd.axes),
                    tf.constant(sections, tf.int64),
                    tf.constant(lengths),
                    tf.constant(omega * (pmd.dgd_per_m / 2.0)),
                )
        else:
            vectors, exponent = multicore_linear_system(a.shape[-1], dt_s, fiber)
            adjoint = np.conj(np.swapaxes(vectors, -1, -2))
            lin = (tf.constant(exponent), tf.constant(vectors), tf.constant(adjoint))
            axis = core_axis(fiber)
        response = kerr_power_response(a.shape[-1], dt_s, fiber)
        response_t = tf.constant(
            np.ones(a.shape[-1]) if response is None else response, dtype=tf.complex128
        )
        eps = nonlinear_gain_per_m(fiber)
        eps_t = tf.constant(eps, tf.float64) if eps != 0.0 else None
        if fiber.scheme.method == "ssfm-local-error":
            length_m = signed_length_m(fiber, direction)
            effective = fiber.scheme.kerr_length == "effective" and attenuation_per_m(fiber) > 0
            out, reached = _local_error(
                tf.constant(a),
                lin,
                tf.constant(abs(length_m), tf.float64),
                tf.constant(float(np.sign(length_m)), tf.float64),
                tf.constant(abs(float(sizes[0])), tf.float64),
                tf.constant(float(fiber.scheme.local_error or 0.0), tf.float64),
                tf.constant(gamma_eff, tf.float64),
                tf.constant(attenuation_per_m(fiber), tf.float64),
                response_t,
                dual,
                response is not None,
                effective,
                axis,
                eps_t,
            )
            if float(reached) < abs(length_m) * (1.0 - 1e-12):
                raise RuntimeError(
                    f"local-error method cannot reach local_error={fiber.scheme.local_error}"
                )
        elif fiber.scheme.method == "rk4ip":
            shock = shock_factor(a.shape[-1], dt_s, fiber)
            out = _rk4ip(
                tf.constant(a),
                lin,
                tf.constant(sizes, dtype=tf.float64),
                tf.constant(gamma_eff, dtype=tf.float64),
                response_t,
                tf.constant(np.ones(a.shape[-1]) if shock is None else shock, tf.complex128),
                dual,
                axis,
                response is not None,
                shock is not None,
                eps_t,
            )
        else:
            linear, kerr = split_step_sequence(fiber, direction)
            e_sat = saturation_energy_j(fiber)
            sat = None
            if e_sat is not None:
                g = gain_spectrum(a.shape[-1], dt_s, fiber)
                assert g is not None
                sat = (
                    tf.constant(small_signal_gain_per_m(fiber) * g / 2.0, tf.float64),
                    tf.constant(1.0 / e_sat, tf.float64),
                    tf.constant(dt_s / a.shape[-1], tf.float64),
                )
            out = _ssfm_sequence(
                tf.constant(a),
                lin,
                tf.constant(linear, dtype=tf.float64),
                tf.constant(gamma_eff * kerr, dtype=tf.float64),
                response_t,
                dual,
                response is not None,
                axis,
                None if eps_t is None else tf.constant(eps * kerr, dtype=tf.float64),
                sat,
            )
        result: ComplexArray = np.asarray(out.numpy(), dtype=np.complex128)
        if pmd is not None and direction == "forward":
            result = apply_jones(result, pmd.output_rotation)
        return result
