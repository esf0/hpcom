"""Split-step, RK4IP and local-error propagation on JAX (GPU/TPU when available).

Runs exactly the schemes of the NumPy reference backend
(:mod:`hpcom.backends.numpy.ssfm`) — the uniform fast path, the general
sub-step sequence ``D(m_J) N(c_J) … N(c_1) D(m_0)`` (every step placement,
Kerr position, Kerr length, intensity filter and fourth-order composition),
RK4IP and the adaptive local-error method, for single- and multicore fibers
and every generalised-NLSE term (Raman response, self-steepening) —
as XLA-compiled loops
(``lax.fori_loop`` / ``lax.while_loop``). The linear operators are built in
NumPy, so results agree with the NumPy backend to FFT rounding.

All computation is in ``complex128`` inside a scoped ``jax.enable_x64``
context: importing hpcom does not change JAX's global precision. JAX places
the computation on its default device (a GPU with ``jax[cuda12]``).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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


def _power(a: Any, dual: bool) -> Any:
    p = jnp.real(a) ** 2 + jnp.imag(a) ** 2
    return jnp.sum(p, axis=-2, keepdims=True) if dual else p


def _kerr(a: Any, phase_per_watt: Any, dual: bool, response: Any = None) -> Any:
    p = _power(a, dual)
    if response is not None:
        p = jnp.real(jnp.fft.ifft(jnp.fft.fft(p, axis=-1) * response, axis=-1))
    return a * jnp.exp(1j * phase_per_watt * p)


def _disperse(spectrum: Any, m: Any, lin: Any, axis: int | None) -> Any:
    """Exact linear step of length ``m``: ``lin = (exponent,)`` or, multicore, ``(exponent, V, V†)``."""
    if axis is None:
        return spectrum * jnp.exp(lin[0] * m)
    exponent, vectors, adjoint = lin
    x = jnp.moveaxis(spectrum, axis, -1)  # (..., n, M)
    z = jnp.matmul(adjoint, x[..., None])[..., 0] * jnp.exp(exponent * m)
    return jnp.moveaxis(jnp.matmul(vectors, z[..., None])[..., 0], -1, axis)


@partial(jax.jit, static_argnames=("dual",))
def _uniform(a: Any, d_full: Any, d_half: Any, phase: Any, n_steps: Any, dual: bool) -> Any:
    def step(_: Any, spectrum: Any) -> Any:
        return jnp.fft.fft(_kerr(jnp.fft.ifft(spectrum, axis=-1), phase, dual), axis=-1) * d_full

    spectrum = lax.fori_loop(0, n_steps - 1, step, jnp.fft.fft(a, axis=-1) * d_half)
    last = _kerr(jnp.fft.ifft(spectrum, axis=-1), phase, dual)
    return jnp.fft.ifft(jnp.fft.fft(last, axis=-1) * d_half, axis=-1)


@partial(jax.jit, static_argnames=("dual", "filtered", "axis"))
def _sequence(
    a: Any,
    lin: Any,
    linear: Any,
    phases: Any,
    response: Any,
    dual: bool,
    filtered: bool,
    axis: int | None,
) -> Any:
    def step(j: Any, spectrum: Any) -> Any:
        x = _kerr(jnp.fft.ifft(spectrum, axis=-1), phases[j], dual, response if filtered else None)
        return _disperse(jnp.fft.fft(x, axis=-1), linear[j + 1], lin, axis)

    spectrum = _disperse(jnp.fft.fft(a, axis=-1), linear[0], lin, axis)
    return jnp.fft.ifft(lax.fori_loop(0, phases.shape[0], step, spectrum), axis=-1)


@partial(jax.jit, static_argnames=("dual", "axis", "filtered", "shocked"))
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
) -> Any:
    def kerr(x: Any) -> Any:  # iγ_eff (1 + iτ∂t)[x · R*P(x)]
        p = _power(x, dual)
        if filtered:
            p = jnp.real(jnp.fft.ifft(jnp.fft.fft(p, axis=-1) * response, axis=-1))
        out = 1j * gamma_eff * p * x
        if shocked:
            out = jnp.fft.ifft(jnp.fft.fft(out, axis=-1) * shock, axis=-1)
        return out

    def step(i: Any, a: Any) -> Any:
        h = sizes[i]

        def half(x: Any) -> Any:
            return jnp.fft.ifft(_disperse(jnp.fft.fft(x, axis=-1), h / 2.0, lin, axis), axis=-1)

        a_ip = half(a)
        k1 = half(h * kerr(a))
        k2 = h * kerr(a_ip + k1 / 2.0)
        k3 = h * kerr(a_ip + k2 / 2.0)
        k4 = h * kerr(half(a_ip + k3))
        return half(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0) + k4 / 6.0

    return lax.fori_loop(0, sizes.shape[0], step, a)


@partial(jax.jit, static_argnames=("dual", "filtered", "effective", "axis"))
def _local_error(
    a: Any,
    lin: Any,
    total: Any,
    sign: Any,
    first_step: Any,
    goal: Any,
    gamma_eff: Any,
    alpha: Any,
    response: Any,
    dual: bool,
    filtered: bool,
    effective: bool,
    axis: int | None,
) -> Any:
    def kerr(spectrum: Any, h: Any) -> Any:
        length = 2.0 * jnp.sinh(alpha * h / 2.0) / alpha if effective else h
        x = jnp.fft.ifft(spectrum, axis=-1)
        x = _kerr(x, gamma_eff * length, dual, response if filtered else None)
        return jnp.fft.fft(x, axis=-1)

    def disperse(spectrum: Any, m: Any) -> Any:
        return _disperse(spectrum, m, lin, axis)

    def norm(x: Any) -> Any:
        return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))

    def keep_going(state: Any) -> Any:
        z, step, _ = state
        return (z < total * (1.0 - 1e-14)) & (step >= total * 1e-12)

    def advance(state: Any) -> Any:
        z, step, spectrum = state
        big = jnp.minimum(step, total - z)
        h = sign * big / 2.0
        coarse = disperse(kerr(disperse(spectrum, h), 2.0 * h), h)
        fine = disperse(kerr(disperse(kerr(disperse(spectrum, h / 2.0), h), h), h), h / 2.0)
        delta = norm(fine - coarse) / norm(fine)
        reject = delta > 2.0 * goal
        spectrum = jnp.where(reject, spectrum, (4.0 / 3.0) * fine - (1.0 / 3.0) * coarse)
        z = jnp.where(reject, z, z + big)
        step = jnp.where(
            reject,
            big / 2.0,
            jnp.where(
                delta > goal,
                big / LOCAL_ERROR_GROWTH,
                jnp.where(delta < goal / 2.0, big * LOCAL_ERROR_GROWTH, big),
            ),
        )
        return z, step, spectrum

    start = (jnp.asarray(0.0, jnp.float64), first_step, jnp.fft.fft(a, axis=-1))
    z, _, spectrum = lax.while_loop(keep_going, advance, start)
    return jnp.fft.ifft(spectrum, axis=-1), z


class JaxPropagator(Propagator):
    """Every hpcom scheme on JAX, single- and multicore (see module docstring)."""

    name = "jax"
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
        scheme = fiber.scheme
        with jax.enable_x64(True):
            if uses_fast_path(fiber):
                exponent = linear_operator_exponent(a.shape[-1], dt_s, fiber)
                h = sizes[0]
                out = _uniform(
                    jnp.asarray(a),
                    jnp.asarray(np.exp(exponent * h)),
                    jnp.asarray(np.exp(exponent * (h / 2.0))),
                    gamma_eff * h,
                    sizes.size,
                    dual,
                )
                return np.asarray(out, dtype=np.complex128)
            if fiber.n_cores == 1:
                lin: tuple[Any, ...] = (
                    jnp.asarray(linear_operator_exponent(a.shape[-1], dt_s, fiber)),
                )
                axis = None
            else:
                vectors, exponent = multicore_linear_system(a.shape[-1], dt_s, fiber)
                adjoint = np.conj(np.swapaxes(vectors, -1, -2))
                lin = (jnp.asarray(exponent), jnp.asarray(vectors), jnp.asarray(adjoint))
                axis = core_axis(fiber)
            response = kerr_power_response(a.shape[-1], dt_s, fiber)
            response_arr = jnp.asarray(
                np.ones(a.shape[-1], np.complex128) if response is None else response
            )
            if scheme.method == "rk4ip":
                shock = shock_factor(a.shape[-1], dt_s, fiber)
                out = _rk4ip(
                    jnp.asarray(a),
                    lin,
                    jnp.asarray(sizes),
                    gamma_eff,
                    response_arr,
                    jnp.asarray(np.ones(a.shape[-1]) if shock is None else shock),
                    dual,
                    axis,
                    response is not None,
                    shock is not None,
                )
            elif scheme.method == "ssfm-local-error":
                length_m = signed_length_m(fiber, direction)
                alpha = attenuation_per_m(fiber)
                out, reached = _local_error(
                    jnp.asarray(a),
                    lin,
                    abs(length_m),
                    float(np.sign(length_m)),
                    abs(float(sizes[0])),
                    float(scheme.local_error or 0.0),
                    gamma_eff,
                    alpha,
                    response_arr,
                    dual,
                    response is not None,
                    scheme.kerr_length == "effective" and alpha > 0,
                    axis,
                )
                if float(reached) < abs(length_m) * (1.0 - 1e-12):
                    raise RuntimeError(
                        f"local-error method cannot reach local_error={scheme.local_error}"
                    )
            else:
                linear, kerr = split_step_sequence(fiber, direction)
                out = _sequence(
                    jnp.asarray(a),
                    lin,
                    jnp.asarray(linear),
                    jnp.asarray(gamma_eff * kerr),
                    response_arr,
                    dual,
                    response is not None,
                    axis,
                )
            result: ComplexArray = np.asarray(out, dtype=np.complex128)
        return result
