r"""Differentiable fiber propagation: parameters, operators and step loops.

Internal units are chosen so that every quantity is O(1) and float32-safe:
time in ps, angular frequency in rad/ps, length in km (β_n in psⁿ/km).
"""

from __future__ import annotations

import math
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from hpcom._typing import Direction
from hpcom.backends.base import (
    check_field_shape,
    core_axis,
    kerr_filter_response,
    pmd_pieces,
    pmd_realization,
    split_step_substeps,
    step_sizes_m,
)
from hpcom.fiber import Fiber
from hpcom.units import SPEED_OF_LIGHT_M_PER_S

__all__ = ["FiberParameters", "GradientMode", "propagate"]

#: How reverse-mode gradients get the forward states (see :func:`propagate`).
GradientMode = Literal["store", "checkpoint", "reversible"]

_PS, _FS, _KM = 1e-12, 1e-15, 1e3
_DB = math.log(10.0) / 10.0  # dB → neper (power)
#: Share of the anisotropic part in Lin & Agrawal's Raman model (hpcom.raman).
_LIN_AGRAWAL_FB = 0.21

Array = Any  # a JAX array (or anything jnp.asarray accepts)


class FiberParameters(NamedTuple):
    r"""The continuous parameters of a :class:`~hpcom.Fiber` as a JAX pytree.

    Differentiate with respect to any leaf; ``None`` marks a term the fiber
    does not have. Units keep every value O(1), so float32 works too:

    ``alpha_db_per_km``
        Loss (``None`` when the fiber has a loss spectrum: its values are the
        parameters then).
    ``betas_psn_per_km``
        :math:`(\beta_1, \beta_2, \beta_3, \beta_4, …)` in psⁿ/km
        (β₂ = −21.7 ps²/km for SSMF).
    ``gamma_per_w_per_km``
        Kerr coefficient γ (the Manakov 8/9 is applied on top).
    ``raman_fraction``, ``raman_tau1_fs``, ``raman_tau2_fs``, ``raman_tau_b_fs``
        :math:`f_R`, :math:`\tau_1`, :math:`\tau_2`, :math:`\tau_b` of the Raman
        response (``tau_b`` for Lin–Agrawal only).
    ``shock_time_fs``
        :math:`\tau_\mathrm{shock}` (self-steepening only).
    ``loss_spectrum_db_per_km``
        Values of the loss table (its wavelengths stay fixed).
    ``coupling_per_km``, ``core_beta0_offsets_per_km``, ``core_beta1_offsets_ps_per_km``
        Multicore coupling matrix, propagation-constant mismatch and skew.
    ``pmd_ps_per_sqrt_km``
        PMD coefficient (the birefringence axes of the realisation stay
        fixed by the fiber's ``pmd_seed``).
    """

    alpha_db_per_km: Array | None
    betas_psn_per_km: Array
    gamma_per_w_per_km: Array
    raman_fraction: Array | None = None
    raman_tau1_fs: Array | None = None
    raman_tau2_fs: Array | None = None
    raman_tau_b_fs: Array | None = None
    shock_time_fs: Array | None = None
    loss_spectrum_db_per_km: Array | None = None
    coupling_per_km: Array | None = None
    core_beta0_offsets_per_km: Array | None = None
    core_beta1_offsets_ps_per_km: Array | None = None
    pmd_ps_per_sqrt_km: Array | None = None

    @classmethod
    def from_fiber(cls, fiber: Fiber) -> FiberParameters:
        """The fiber's values, as JAX arrays of the default float dtype."""

        def arr(value: Any) -> Array:
            return jnp.asarray(value, dtype=jnp.result_type(float))

        betas = [b * _KM / _PS**n for n, b in enumerate(fiber.dispersion_betas, start=1)]
        raman = fiber.raman
        coupling = getattr(fiber, "coupling_per_m", None)
        beta0 = getattr(fiber, "core_beta0_offsets_per_m", None)
        skew = getattr(fiber, "core_beta1_offsets_s_per_m", None)
        spectrum = fiber.loss_spectrum
        return cls(
            alpha_db_per_km=None if spectrum is not None else arr(fiber.alpha_db_per_km),
            betas_psn_per_km=arr(betas),
            gamma_per_w_per_km=arr(fiber.gamma_per_w_per_km),
            raman_fraction=None if raman is None else arr(raman.f_r),
            raman_tau1_fs=None if raman is None else arr(raman.tau1_s / _FS),
            raman_tau2_fs=None if raman is None else arr(raman.tau2_s / _FS),
            raman_tau_b_fs=(
                arr(raman.tau_b_s / _FS)
                if raman is not None and raman.model == "lin-agrawal"
                else None
            ),
            shock_time_fs=(
                arr(fiber.effective_shock_time_s / _FS) if fiber.self_steepening else None
            ),
            loss_spectrum_db_per_km=None if spectrum is None else arr(spectrum.alpha_db_per_km),
            coupling_per_km=None if coupling is None else arr(np.asarray(coupling) * _KM),
            core_beta0_offsets_per_km=None if beta0 is None else arr(np.asarray(beta0) * _KM),
            core_beta1_offsets_ps_per_km=(
                None if skew is None else arr(np.asarray(skew) * _KM / _PS)
            ),
            pmd_ps_per_sqrt_km=(
                arr(fiber.pmd_ps_per_sqrt_km) if fiber.pmd_ps_per_sqrt_km > 0 else None
            ),
        )

    def to_fiber(self, fiber: Fiber) -> Fiber:
        """``fiber`` with these values (concrete arrays only, not inside ``jit``)."""

        def val(x: Array) -> float:
            return float(np.asarray(x))

        betas = [val(b) * _PS**n / _KM for n, b in enumerate(self.betas_psn_per_km, start=1)]
        changes: dict[str, object] = {
            "beta1_s_per_m": betas[0],
            "beta2_s2_per_m": betas[1],
            "beta3_s3_per_m": betas[2],
            "higher_order_betas": tuple(betas[3:]),
            "gamma_per_w_per_km": val(self.gamma_per_w_per_km),
        }
        if self.alpha_db_per_km is not None:
            changes["alpha_db_per_km"] = val(self.alpha_db_per_km)
        if fiber.raman is not None and self.raman_fraction is not None:
            raman = fiber.raman.replace(
                fraction=val(self.raman_fraction),
                tau1_s=val(self.raman_tau1_fs) * _FS,
                tau2_s=val(self.raman_tau2_fs) * _FS,
            )
            if self.raman_tau_b_fs is not None:
                raman = raman.replace(tau_b_s=val(self.raman_tau_b_fs) * _FS)
            changes["raman"] = raman
        if self.shock_time_fs is not None:
            changes["shock_time_s"] = val(self.shock_time_fs) * _FS
        if self.coupling_per_km is not None:
            changes["coupling_per_m"] = np.asarray(self.coupling_per_km, np.float64) / _KM
        if self.core_beta0_offsets_per_km is not None:
            changes["core_beta0_offsets_per_m"] = (
                np.asarray(self.core_beta0_offsets_per_km, np.float64) / _KM
            )
        if self.core_beta1_offsets_ps_per_km is not None:
            changes["core_beta1_offsets_s_per_m"] = (
                np.asarray(self.core_beta1_offsets_ps_per_km, np.float64) * _PS / _KM
            )
        if self.pmd_ps_per_sqrt_km is not None:
            changes["pmd_ps_per_sqrt_km"] = val(self.pmd_ps_per_sqrt_km)
        out = fiber.replace(**changes)
        if self.loss_spectrum_db_per_km is not None and fiber.loss_spectrum is not None:
            spectrum = fiber.loss_spectrum.replace(
                alpha_db_per_km=np.asarray(self.loss_spectrum_db_per_km, np.float64)
            )
            out = out.with_loss_spectrum(spectrum)
        return out


# ---------------------------------------------------------------------------
# Operators (jnp, differentiable in the parameters)
# ---------------------------------------------------------------------------


class _Coupling(NamedTuple):
    """Hermitian core generator K (per km) and its eigen-decomposition (gradient-free)."""

    matrix: Array  # (M, M) or (n, M, M), real symmetric
    vectors: Array  # same shape, K = V diag(λ) Vᵀ
    values: Array  # (M,) or (n, M)


class _Operators(NamedTuple):
    exponent: Array  # (n,) complex, per km: iD(ω) − α(ω)/2
    response: Array | None  # (n,) complex Kerr power response, or None
    shock: Array | None  # (n,) real 1 − τω, or None
    coupling: _Coupling | None
    gamma_eff: Array  # 1/(W·km), with the Manakov 8/9
    alpha_carrier: Array  # 1/km, for the effective Kerr length
    pmd_axes: Array | None = None  # (N, 3) birefringence axes of the sections
    pmd_half_dgd: Array | None = None  # (n,) ω·Δβ₁/2 per km


def _omega_ps(n: int, dt_s: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dt_s / _PS)  # rad/ps


def _loss_weights(fiber: Fiber, n: int, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Static linear-interpolation indices and weights of the loss table on the grid."""
    spectrum = fiber.loss_spectrum
    assert spectrum is not None
    nu = SPEED_OF_LIGHT_M_PER_S / fiber.wavelength_m - np.fft.fftfreq(n, d=dt_s)
    lam = np.full(n, np.inf)
    lam[nu > 0] = SPEED_OF_LIGHT_M_PER_S / nu[nu > 0]
    table = np.asarray(spectrum.wavelengths_m)
    lam = np.clip(lam, table[0], table[-1])
    idx = np.clip(np.searchsorted(table, lam, side="right") - 1, 0, table.size - 2)
    frac = (lam - table[idx]) / (table[idx + 1] - table[idx])
    return idx, frac


def _carrier_loss_weights(fiber: Fiber) -> tuple[int, float]:
    spectrum = fiber.loss_spectrum
    assert spectrum is not None
    table = np.asarray(spectrum.wavelengths_m)
    lam = float(np.clip(fiber.wavelength_m, table[0], table[-1]))
    idx = int(np.clip(np.searchsorted(table, lam, side="right") - 1, 0, table.size - 2))
    return idx, (lam - table[idx]) / (table[idx + 1] - table[idx])


def _raman_transfer(params: FiberParameters, omega: Array) -> Array:
    """H_R(ω) of :class:`hpcom.raman.RamanResponse` with τ in fs → ps (ω in rad/ps)."""
    assert params.raman_tau1_fs is not None and params.raman_tau2_fs is not None
    a = 1.0 / (params.raman_tau2_fs * 1e-3)
    b = 1.0 / (params.raman_tau1_fs * 1e-3)
    h_a = (a * a + b * b) / ((a + 1j * omega) ** 2 + b * b)
    if params.raman_tau_b_fs is None:
        return h_a
    u = 1.0 + 1j * omega * (params.raman_tau_b_fs * 1e-3)
    return (1.0 - _LIN_AGRAWAL_FB) * h_a + _LIN_AGRAWAL_FB * (2.0 * u - 1.0) / u**2


def _operators(
    params: FiberParameters, fiber: Fiber, n: int, dt_s: float, rdtype: Any
) -> _Operators:
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, rdtype), params)
    omega_np = _omega_ps(n, dt_s)
    omega = jnp.asarray(omega_np, rdtype)
    betas = jnp.asarray(params.betas_psn_per_km, rdtype)
    d = jnp.zeros(n, rdtype)
    for k in range(betas.shape[0]):
        order = k + 1
        d = d + betas[k] * jnp.asarray((-omega_np) ** order / math.factorial(order), rdtype)
    if params.loss_spectrum_db_per_km is not None:
        values = jnp.asarray(params.loss_spectrum_db_per_km, rdtype)
        idx, frac = _loss_weights(fiber, n, dt_s)
        f = jnp.asarray(frac, rdtype)
        alpha = (values[idx] * (1.0 - f) + values[idx + 1] * f) * _DB
        i0, f0 = _carrier_loss_weights(fiber)
        alpha_carrier = (values[i0] * (1.0 - f0) + values[i0 + 1] * f0) * _DB
    else:
        alpha_carrier = jnp.asarray(params.alpha_db_per_km, rdtype) * _DB
        alpha = alpha_carrier
    exponent = 1j * d - alpha / 2.0
    response = None
    if params.raman_fraction is not None:
        f_r = params.raman_fraction
        response = (1.0 - f_r) + f_r * _raman_transfer(params, omega)
    filt = kerr_filter_response(n, dt_s, fiber)
    if filt is not None:
        f_arr = jnp.asarray(filt, rdtype)
        response = f_arr if response is None else response * f_arr
    shock = None
    if params.shock_time_fs is not None:
        shock = 1.0 - (params.shock_time_fs * 1e-3) * omega
    coupling = None
    if params.coupling_per_km is not None:
        c = jnp.asarray(params.coupling_per_km, rdtype)
        c = (c + c.T) / 2.0  # the model is symmetric; so is the gradient's direction
        if params.core_beta0_offsets_per_km is not None:
            c = c + jnp.diag(jnp.asarray(params.core_beta0_offsets_per_km, rdtype))
        if params.core_beta1_offsets_ps_per_km is not None:
            skew = jnp.asarray(params.core_beta1_offsets_ps_per_km, rdtype)
            c = c[None, :, :] - jax.vmap(jnp.diag)(omega[:, None] * skew[None, :])
        values, vectors = jnp.linalg.eigh(lax.stop_gradient(c))
        coupling = _Coupling(c, vectors, values)
    gamma = jnp.asarray(params.gamma_per_w_per_km, rdtype)
    gamma_eff = gamma * 8.0 / 9.0 if fiber.polarization == "dual" else gamma
    pmd_axes = pmd_half_dgd = None
    realization = pmd_realization(fiber)
    if realization is not None and params.pmd_ps_per_sqrt_km is not None:
        n_sec = realization.axes.shape[0]
        tau_s_ps = params.pmd_ps_per_sqrt_km * math.sqrt(
            fiber.length_km * 3.0 * math.pi / (8.0 * n_sec)
        )
        dgd_per_km = tau_s_ps / (fiber.length_km / n_sec)  # ps/km
        pmd_axes = jnp.asarray(realization.axes, rdtype)
        pmd_half_dgd = omega * dgd_per_km / 2.0
    return _Operators(
        exponent, response, shock, coupling, gamma_eff, alpha_carrier, pmd_axes, pmd_half_dgd
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _power(a: Array, dual: bool, response: Array | None) -> Array:
    p = jnp.real(a) ** 2 + jnp.imag(a) ** 2
    if dual:
        p = jnp.sum(p, axis=-2, keepdims=True)
    if response is not None:
        p = jnp.real(jnp.fft.ifft(jnp.fft.fft(p, axis=-1) * response, axis=-1))
    return p


@jax.custom_jvp
def _coupled(x: Array, k: Array, m: Array, v: Array, lam: Array) -> Array:
    r""":math:`e^{imK}x = V(e^{im\lambda}\circ V^\top x)` for real symmetric ``k`` = :math:`VΛV^\top`.

    ``x`` is ``(..., n, M)``; ``k``, ``v`` are ``(M, M)`` or ``(n, M, M)``.
    ``v``, ``lam`` come from ``eigh`` and carry no gradient: the derivative
    with respect to ``k`` is the Daleckii–Krein formula (custom JVP).
    """
    vx = _rot(v, x, transpose=True)
    return _rot(v, jnp.exp(1j * m * lam) * vx, transpose=False)


def _rot(v: Array, x: Array, *, transpose: bool) -> Array:
    """``Vᵀx`` or ``Vx`` for ``x`` of shape ``(..., n, M)``."""
    if v.ndim == 2:
        return x @ (v if transpose else v.T)
    return jnp.einsum("fji,...fj->...fi" if transpose else "fij,...fj->...fi", v, x)


@_coupled.defjvp
def _coupled_jvp(primals: tuple[Array, ...], tangents: tuple[Array, ...]) -> tuple[Array, Array]:
    r"""Fréchet derivative of :math:`e^{imK}` at symmetric :math:`K = VΛV^\top`.

    .. math::

        d\,e^{imK} = V\,(F\circ V^\top dK\,V)\,V^\top + i\,dm\,K e^{imK},\qquad
        F_{ij} = \frac{e^{im\lambda_i} - e^{im\lambda_j}}{\lambda_i - \lambda_j}
               = im\,e^{im(\lambda_i+\lambda_j)/2}\operatorname{sinc}\frac{m(\lambda_i-\lambda_j)}{2},

    the sinc form being finite for degenerate eigenvalues (ring, hexagon).
    """
    x, k, m, v, lam = primals
    dx, dk, dm, _, _ = tangents
    out = _coupled(x, k, m, v, lam)
    vx = _rot(v, x, transpose=True)  # (..., n, M)
    phase = jnp.exp(1j * m * lam)
    diff = lam[..., :, None] - lam[..., None, :]
    mean = (lam[..., :, None] + lam[..., None, :]) / 2.0
    f = 1j * m * jnp.exp(1j * m * mean) * jnp.sinc(m * diff / (2.0 * jnp.pi))
    if v.ndim == 2:
        dk_eig = v.T @ dk @ v
        inner = vx @ (f * dk_eig).T
    else:
        dk_eig = jnp.einsum("fji,fjk,fkl->fil", v, dk, v)
        inner = jnp.einsum("fij,...fj->...fi", f * dk_eig, vx)
    d_out = _rot(v, inner, transpose=False)
    d_out = d_out + _coupled(dx, k, m, v, lam)
    d_out = d_out + dm * _rot(v, 1j * lam * phase * vx, transpose=False)
    return out, d_out


def _pmd(spectrum: Array, ops: _Operators, pieces: Any, *, inverse: bool = False) -> Array:
    """The birefringence pieces of one move, ``pieces = (sections (P,), lengths_km (P,))``.

    Each piece is :math:`e^{-i\\theta(\\hat b\\cdot\\vec\\sigma)}`,
    :math:`\\theta = \\omega\\Delta\\beta_1\\ell/2`; ``inverse`` undoes the move
    (pieces reversed, lengths negated).
    """
    if ops.pmd_axes is None or ops.pmd_half_dgd is None:
        return spectrum
    sections, lengths = pieces
    order = range(sections.shape[0] - 1, -1, -1) if inverse else range(sections.shape[0])
    for p in order:
        b = ops.pmd_axes[sections[p]]
        theta = ops.pmd_half_dgd * (-lengths[p] if inverse else lengths[p])
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


def _disperse(
    spectrum: Array, m: Array, ops: _Operators, axis: int | None, pieces: Any = None
) -> Array:
    """Exact linear step of ``m`` km in the spectral domain (coupling by eigen-decomposition).

    With PMD, ``pieces`` are the move's birefringence pieces (:func:`_pmd`).
    """
    out = spectrum * jnp.exp(ops.exponent * m)
    if ops.pmd_axes is not None:
        return _pmd(out, ops, pieces)
    if ops.coupling is None or axis is None:
        return out
    c = ops.coupling
    x = jnp.moveaxis(out, axis, -1)  # (..., n, M)
    y = _coupled(x, c.matrix.astype(x.dtype), m, c.vectors.astype(x.dtype), c.values)
    return jnp.moveaxis(y, -1, axis)


def _kerr_step(spectrum: Array, phase: Array, ops: _Operators, dual: bool) -> Array:
    """N(c) in the spectral domain: A ↦ A·exp(iγc·Q), a pure phase (exactly invertible)."""
    x = jnp.fft.ifft(spectrum, axis=-1)
    x = x * jnp.exp(1j * phase * _power(x, dual, ops.response))
    return jnp.fft.fft(x, axis=-1)


def _sequence_step(
    spectrum: Array,
    phase: Array,
    m: Array,
    ops: _Operators,
    dual: bool,
    axis: int | None,
    pieces: Any = None,
) -> Array:
    return _disperse(_kerr_step(spectrum, phase, ops, dual), m, ops, axis, pieces)


def _sequence_inverse_step(
    spectrum: Array,
    phase: Array,
    m: Array,
    ops: _Operators,
    dual: bool,
    axis: int | None,
    pieces: Any = None,
) -> Array:
    undone = _pmd(spectrum, ops, pieces, inverse=True) if ops.pmd_axes is not None else spectrum
    scalar = ops._replace(pmd_axes=None, pmd_half_dgd=None)
    return _kerr_step(_disperse(undone, -m, scalar, axis), -phase, ops, dual)


def _scan(step: Any, carry: Array, xs: Any, gradient: GradientMode) -> Array:
    """``lax.scan`` of ``step`` (carry-only), storing states or checkpointing √J blocks."""
    if gradient == "store":
        return lax.scan(lambda c, x: (step(c, x), None), carry, xs)[0]
    length = jax.tree_util.tree_leaves(xs)[0].shape[0]
    block = max(1, math.isqrt(length))
    n_blocks = -(-length // block)
    pad = n_blocks * block - length

    def padded(x: Array) -> Array:  # zero-length, zero-phase steps are the identity
        return jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), x.dtype)]).reshape(
            n_blocks, block, *x.shape[1:]
        )

    blocks = jax.tree_util.tree_map(padded, xs)

    @jax.checkpoint
    def run_block(c: Array, xb: Any) -> Array:
        return lax.scan(lambda c2, x: (step(c2, x), None), c, xb)[0]

    return lax.scan(lambda c, xb: (run_block(c, xb), None), carry, blocks)[0]


def _reversible_sequence(
    spectrum: Array,
    ops: _Operators,
    phases: Array,
    linear: Array,
    dual: bool,
    axis: int | None,
    pieces: tuple[Array, Array],
) -> Array:
    """The split-step loop with an O(1)-memory exact adjoint (see :func:`propagate`).

    ``pieces`` (the PMD pieces of each move, fixed) ride along the scan as
    constants; everything else gets its cotangent.
    """

    @jax.custom_vjp
    def run(s0: Array, ops: _Operators, phases: Array, linear: Array) -> Array:
        return lax.scan(
            lambda s, x: (_sequence_step(s, x[0], x[1], ops, dual, axis, x[2]), None),
            s0,
            (phases, linear, pieces),
        )[0]

    def fwd(s0: Array, ops: _Operators, phases: Array, linear: Array) -> tuple[Array, Any]:
        out = run(s0, ops, phases, linear)
        return out, (out, ops, phases, linear)

    def bwd(res: Any, g: Array) -> tuple[Array, Any, Array, Array]:
        out, ops, phases, linear = res

        def back(carry: Any, x: Any) -> tuple[Any, Any]:
            s_next, g_next, g_ops = carry
            phase, m, piece = x
            s = _sequence_inverse_step(s_next, phase, m, ops, dual, axis, piece)
            _, vjp = jax.vjp(
                lambda s_, o_, p_, m_: _sequence_step(s_, p_, m_, o_, dual, axis, piece),
                s,
                ops,
                phase,
                m,
            )
            g_s, g_o, g_p, g_m = vjp(g_next)
            g_ops = jax.tree_util.tree_map(jnp.add, g_ops, g_o)
            return (s, g_s, g_ops), (g_p, g_m)

        zeros = jax.tree_util.tree_map(jnp.zeros_like, ops)
        (_, g_s0, g_ops), (g_phases, g_linear) = lax.scan(
            back, (out, g, zeros), (phases, linear, pieces), reverse=True
        )
        return g_s0, g_ops, g_phases, g_linear

    run.defvjp(fwd, bwd)
    return run(spectrum, ops, phases, linear)


def _moves(
    fiber: Fiber, direction: Direction, lengths_m: np.ndarray, rdtype: Any
) -> tuple[np.ndarray, np.ndarray]:
    """PMD pieces of consecutive moves: ``(J, P)`` sections and lengths in km (NumPy constants).

    One zero-length dummy piece per move without PMD. Kept in NumPy so that
    the reversible adjoint can close over them.
    """
    realization = pmd_realization(fiber)
    dtype = np.dtype(rdtype)
    if realization is None:
        return np.zeros((lengths_m.size, 1), np.int32), np.zeros((lengths_m.size, 1), dtype)
    start = 0.0 if direction == "forward" else fiber.length_km * 1e3
    sections, lengths = pmd_pieces(realization, start, lengths_m)
    return sections.astype(np.int32), (lengths / _KM).astype(dtype)


def _kerr_lengths_km(fiber: Fiber, sub_m: np.ndarray, ops: _Operators, rdtype: Any) -> Array:
    """Kerr lengths c of the sub-steps: h, or (e^{αr|h|} − e^{−α(1−r)|h|})/α with α traced."""
    h = jnp.asarray(sub_m / _KM, rdtype)
    if fiber.scheme.kerr_length == "step":
        return h
    r = float(fiber.scheme.nonlinear_position)
    alpha = ops.alpha_carrier
    mag = jnp.abs(h)
    safe = jnp.where(alpha > 0, alpha, 1.0)
    eff = jnp.exp(safe * r * mag) * (-jnp.expm1(-safe * mag)) / safe
    return jnp.where(alpha > 0, jnp.sign(h) * eff, h)


@partial(jax.jit, static_argnames=("fiber", "dt_s", "direction", "gradient"))
def _propagate(
    field: Array,
    params: FiberParameters,
    fiber: Fiber,
    dt_s: float,
    direction: Direction,
    gradient: GradientMode,
) -> Array:
    a = jnp.asarray(field)
    if not jnp.issubdtype(a.dtype, jnp.complexfloating):
        a = a.astype(jnp.result_type(a.dtype, jnp.complex64))
    rdtype = jnp.real(a).dtype
    n = a.shape[-1]
    ops = _operators(params, fiber, n, dt_s, rdtype)
    dual = fiber.polarization == "dual"
    axis = core_axis(fiber) if fiber.n_cores > 1 else None
    realization = pmd_realization(fiber)
    rotation = None if realization is None else jnp.asarray(realization.output_rotation, a.dtype)
    if rotation is not None and direction == "backward":
        a = jnp.einsum("pq,...qn->...pn", jnp.conj(rotation).T, a)
    if fiber.scheme.method == "rk4ip":
        sizes_m = step_sizes_m(fiber, direction)
        sec, plen = _moves(fiber, direction, np.repeat(sizes_m / 2.0, 2), rdtype)
        halves = ((sec[0::2], plen[0::2]), (sec[1::2], plen[1::2]))
        out = _rk4ip(a, ops, jnp.asarray(sizes_m / _KM, rdtype), halves, dual, axis, gradient)
    else:
        linear_m, sub_m, _ = split_step_substeps(fiber, direction)
        linear = jnp.asarray(linear_m / _KM, rdtype)
        phases = ops.gamma_eff * _kerr_lengths_km(fiber, sub_m, ops, rdtype)
        sec, plen = _moves(fiber, direction, linear_m, rdtype)
        spectrum = _disperse(jnp.fft.fft(a, axis=-1), linear[0], ops, axis, (sec[0], plen[0]))
        pieces = (sec[1:], plen[1:])
        if gradient == "reversible":
            spectrum = _reversible_sequence(spectrum, ops, phases, linear[1:], dual, axis, pieces)
        else:
            spectrum = _scan(
                lambda s, x: _sequence_step(s, x[0], x[1], ops, dual, axis, x[2]),
                spectrum,
                (phases, linear[1:], pieces),
                gradient,
            )
        out = jnp.fft.ifft(spectrum, axis=-1)
    if rotation is not None and direction == "forward":
        out = jnp.einsum("pq,...qn->...pn", rotation, out)
    return out


def _rk4ip(
    a: Array,
    ops: _Operators,
    sizes: Array,
    halves: Any,
    dual: bool,
    axis: int | None,
    gradient: GradientMode,
) -> Array:
    """Fourth-order Runge–Kutta in the interaction picture (Hult 2007), as in the backends.

    ``halves`` are the PMD pieces of each step's first (z → z + h/2) and
    second (z + h/2 → z + h) half.
    """

    def nonlinear(x: Array) -> Array:  # iγ_eff (1 + iτ∂t)[x · R*P(x)]
        out = 1j * ops.gamma_eff * _power(x, dual, ops.response) * x
        if ops.shock is not None:
            out = jnp.fft.ifft(jnp.fft.fft(out, axis=-1) * ops.shock, axis=-1)
        return out

    def step(x: Array, xs: Any) -> Array:
        h, first, second = xs

        def half(y: Array, pieces: Any) -> Array:
            spectrum = _disperse(jnp.fft.fft(y, axis=-1), h / 2.0, ops, axis, pieces)
            return jnp.fft.ifft(spectrum, axis=-1)

        a_ip = half(x, first)
        k1 = half(h * nonlinear(x), first)
        k2 = h * nonlinear(a_ip + k1 / 2.0)
        k3 = h * nonlinear(a_ip + k2 / 2.0)
        k4 = h * nonlinear(half(a_ip + k3, second))
        return half(a_ip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0, second) + k4 / 6.0

    return _scan(step, a, (sizes, halves[0], halves[1]), gradient)


def propagate(
    field: Array,
    fiber: Fiber,
    dt_s: float,
    *,
    params: FiberParameters | None = None,
    direction: Direction = "forward",
    gradient: GradientMode = "store",
) -> Array:
    r"""Propagate ``field`` through ``fiber`` — a pure, differentiable JAX function.

    Same equations, schemes and step sequences as :meth:`hpcom.Fiber.propagate`
    (layouts ``(..., n)``, ``(..., 2, n)``, ``(..., M, n)``, ``(..., M, 2, n)``),
    with the continuous parameters taken from ``params``
    (:class:`FiberParameters`; default: the fiber's own). ``fiber`` fixes the
    structure — length, steps and their placement, scheme, polarisation,
    which terms exist — and must be a static (hashable) argument under
    :func:`jax.jit`; so must ``dt_s``. The step grid does not depend on
    ``params``; gradients are those of the discretised solution.

    The precision follows ``field``: complex128 needs
    ``jax.config.update("jax_enable_x64", True)`` (or a ``jax.enable_x64``
    context); complex64 works as well.

    ``gradient`` sets how reverse-mode AD recovers the forward states:

    * ``"store"`` — keep every step's field (memory ∝ steps);
    * ``"checkpoint"`` — keep √J block boundaries and recompute inside a
      block (memory ∝ √steps, about twice the forward work);
    * ``"reversible"`` — split-step schemes only: keep nothing and rebuild
      each state in the backward pass by the exact inverse step (the Kerr
      step is a pure phase and the linear step is invertible), memory
      independent of the step count, about three times the forward work.
      Exact up to the rounding of running the span backwards.

    The adaptive ``"ssfm-local-error"`` scheme is not differentiable (its
    step sequence depends on the field) and is rejected.
    """
    if fiber.scheme.method == "ssfm-local-error":
        raise NotImplementedError(
            'the adaptive "ssfm-local-error" scheme is not differentiable; use a '
            "fixed-step scheme (e.g. fiber.with_steps(...) and scheme 'ssfm4-suzuki')"
        )
    if gradient not in ("store", "checkpoint", "reversible"):
        raise ValueError(
            f"gradient must be 'store', 'checkpoint' or 'reversible', got {gradient!r}"
        )
    if gradient == "reversible" and fiber.scheme.method == "rk4ip":
        raise ValueError(
            'gradient="reversible" needs an exactly invertible split-step scheme; RK4IP is '
            'not — use "store" or "checkpoint"'
        )
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
    shape = tuple(np.shape(field))
    check_field_shape(SimpleNamespace(ndim=len(shape), shape=shape), fiber)  # type: ignore[arg-type]
    own = FiberParameters.from_fiber(fiber)
    if params is None:
        params = own
    elif jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(own):
        raise ValueError(
            "params must have the structure of FiberParameters.from_fiber(fiber) (the fiber "
            "decides which terms exist); got a different set of None fields"
        )
    return _propagate(field, params, fiber, float(dt_s), direction, gradient)
