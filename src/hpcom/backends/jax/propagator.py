"""Split-step, RK4IP and local-error propagation on JAX (GPU/TPU when available).

Runs exactly the schemes of the NumPy reference backend
(:mod:`hpcom.backends.numpy.ssfm`) — the uniform fast path, the general
sub-step sequence ``D(m_J) N(c_J) … N(c_1) D(m_0)`` (every step placement,
Kerr position, Kerr length, intensity filter and fourth-order composition),
RK4IP and the adaptive local-error method, for single- and multicore fibers,
every generalised-NLSE term (Raman response, self-steepening) and the gain
terms of :mod:`hpcom.laser` (nonlinear gain, saturated gain) —
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


def _power(a: Any, dual: bool) -> Any:
    p = jnp.real(a) ** 2 + jnp.imag(a) ** 2
    return jnp.sum(p, axis=-2, keepdims=True) if dual else p


def _kerr(
    a: Any, phase_per_watt: Any, dual: bool, response: Any = None, gain_per_watt: Any = None
) -> Any:
    p = _power(a, dual)
    if response is not None:
        p = jnp.real(jnp.fft.ifft(jnp.fft.fft(p, axis=-1) * response, axis=-1))
    if gain_per_watt is None:
        return a * jnp.exp(1j * phase_per_watt * p)
    return a * _kerr_gain_factor(p, phase_per_watt, gain_per_watt)


def _kerr_gain_factor(p: Any, phase_per_watt: Any, gain_per_watt: Any) -> Any:
    """:func:`hpcom.backends.base.kerr_gain_factor` (NaN where 2εcP ≥ 1: the step diverges)."""
    u = 2.0 * gain_per_watt * p
    small = jnp.abs(u) < 1e-8
    safe = jnp.where(small, 0.5, u)
    log_ratio = jnp.where(small, 1.0 + u / 2.0, -jnp.log1p(-safe) / safe)
    return jnp.exp(1j * phase_per_watt * p * log_ratio) / jnp.sqrt(1.0 - u)


def _saturated(spectrum: Any, m: Any, lin: Any, sat: Any, dual: bool, move: Any) -> Any:
    """Linear step with saturated gain at the mid-step energy (``_Disperser._saturated``).

    ``sat = (g₀G(ω)/2, 1/E_sat, Δt/n)``.
    """
    half_gain, inv_sat, scale = sat

    def energy(x: Any) -> Any:
        e = jnp.sum(jnp.real(x) ** 2 + jnp.imag(x) ** 2, axis=-1, keepdims=True) * scale
        return jnp.sum(e, axis=-2, keepdims=True) if dual else e

    base = spectrum * jnp.exp(lin[0] * m)
    e0 = energy(spectrum)
    e_mid = (e0 + energy(base * jnp.exp(m * half_gain / (1.0 + e0 * inv_sat)))) / 2.0
    out = base * jnp.exp(m * half_gain / (1.0 + e_mid * inv_sat))
    return _pmd_move(out, lin, move) if len(lin) == 5 else out


def _pmd_move(spectrum: Any, lin: Any, move: Any) -> Any:
    """Birefringence pieces of linear move ``move``: ``lin = (exponent, axes, sections, lengths, ω·Δβ₁/2)``."""
    _, axes, sections, lengths, half_dgd = lin
    for p in range(sections.shape[1]):  # static: the widest move
        b = axes[sections[move, p]]
        theta = half_dgd * lengths[move, p]
        c, s = jnp.cos(theta), jnp.sin(theta)
        x, y = spectrum[..., 0, :], spectrum[..., 1, :]
        spectrum = jnp.stack(
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
    if axis is None:
        out = spectrum * jnp.exp(lin[0] * m)
        return _pmd_move(out, lin, move) if len(lin) == 5 else out
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


@partial(jax.jit, static_argnames=("dual", "filtered", "axis", "gained", "saturated"))
def _sequence(
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
    gained: bool = False,
    saturated: bool = False,
) -> Any:
    def disperse(spectrum: Any, j: Any) -> Any:
        if saturated:
            return _saturated(spectrum, linear[j], lin, sat, dual, j)
        return _disperse(spectrum, linear[j], lin, axis, j)

    def step(j: Any, spectrum: Any) -> Any:
        x = _kerr(
            jnp.fft.ifft(spectrum, axis=-1),
            phases[j],
            dual,
            response if filtered else None,
            gains[j] if gained else None,
        )
        return disperse(jnp.fft.fft(x, axis=-1), j + 1)

    spectrum = disperse(jnp.fft.fft(a, axis=-1), 0)
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
    eps: Any = 0.0,
) -> Any:
    def kerr(x: Any) -> Any:  # (iγ_eff + ε)(1 + iτ∂t)[x · R*P(x)]
        p = _power(x, dual)
        if filtered:
            p = jnp.real(jnp.fft.ifft(jnp.fft.fft(p, axis=-1) * response, axis=-1))
        out = (1j * gamma_eff + eps) * p * x
        if shocked:
            out = jnp.fft.ifft(jnp.fft.fft(out, axis=-1) * shock, axis=-1)
        return out

    def step(i: Any, a: Any) -> Any:
        h = sizes[i]

        def half(x: Any, move: Any) -> Any:  # move 2i: z → z + h/2; 2i + 1: z + h/2 → z + h
            spectrum = _disperse(jnp.fft.fft(x, axis=-1), h / 2.0, lin, axis, move)
            return jnp.fft.ifft(spectrum, axis=-1)

        a_ip = half(a, 2 * i)
        k1 = half(h * kerr(a), 2 * i)
        k2 = h * kerr(a_ip + k1 / 2.0)
        k3 = h * kerr(a_ip + k2 / 2.0)
        k4 = h * kerr(half(a_ip + k3, 2 * i + 1))
        return half(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0, 2 * i + 1) + k4 / 6.0

    return lax.fori_loop(0, sizes.shape[0], step, a)


@partial(jax.jit, static_argnames=("dual", "filtered", "effective", "axis", "gained"))
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
    eps: Any = 0.0,
    gained: bool = False,
) -> Any:
    def kerr(spectrum: Any, h: Any) -> Any:
        length = 2.0 * jnp.sinh(alpha * h / 2.0) / alpha if effective else h
        x = jnp.fft.ifft(spectrum, axis=-1)
        gain = eps * length if gained else None
        x = _kerr(x, gamma_eff * length, dual, response if filtered else None, gain)
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
            pmd = pmd_realization(fiber)
            if pmd is not None and direction == "backward":
                a = apply_jones(a, pmd.output_rotation.conj().T)
            if fiber.n_cores == 1:
                lin: tuple[Any, ...] = (
                    jnp.asarray(linear_operator_exponent(a.shape[-1], dt_s, fiber)),
                )
                axis = None
                if pmd is not None:
                    start = 0.0 if direction == "forward" else fiber.length_km * 1e3
                    moves = (
                        np.repeat(sizes / 2.0, 2)
                        if scheme.method == "rk4ip"
                        else split_step_sequence(fiber, direction)[0]
                    )
                    sections, lengths = pmd_pieces(pmd, start, moves)
                    omega = 2.0 * np.pi * np.fft.fftfreq(a.shape[-1], d=dt_s)
                    lin = (
                        *lin,
                        jnp.asarray(pmd.axes),
                        jnp.asarray(sections),
                        jnp.asarray(lengths),
                        jnp.asarray(omega * (pmd.dgd_per_m / 2.0)),
                    )
            else:
                vectors, exponent = multicore_linear_system(a.shape[-1], dt_s, fiber)
                adjoint = np.conj(np.swapaxes(vectors, -1, -2))
                lin = (jnp.asarray(exponent), jnp.asarray(vectors), jnp.asarray(adjoint))
                axis = core_axis(fiber)
            response = kerr_power_response(a.shape[-1], dt_s, fiber)
            eps = nonlinear_gain_per_m(fiber)
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
                    eps,
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
                    eps,
                    eps != 0.0,
                )
                if float(reached) < abs(length_m) * (1.0 - 1e-12):
                    raise RuntimeError(
                        f"local-error method cannot reach local_error={scheme.local_error}"
                    )
            else:
                linear, kerr = split_step_sequence(fiber, direction)
                e_sat = saturation_energy_j(fiber)
                sat = None
                if e_sat is not None:
                    g = gain_spectrum(a.shape[-1], dt_s, fiber)
                    assert g is not None
                    half_gain = small_signal_gain_per_m(fiber) * g / 2.0
                    sat = (jnp.asarray(half_gain), 1.0 / e_sat, dt_s / a.shape[-1])
                out = _sequence(
                    jnp.asarray(a),
                    lin,
                    jnp.asarray(linear),
                    jnp.asarray(gamma_eff * kerr),
                    response_arr,
                    dual,
                    response is not None,
                    axis,
                    jnp.asarray(eps * kerr),
                    sat,
                    eps != 0.0,
                    sat is not None,
                )
            result: ComplexArray = np.asarray(out, dtype=np.complex128)
        if pmd is not None and direction == "forward":
            result = apply_jones(result, pmd.output_rotation)
        return result
