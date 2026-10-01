r"""Learned digital back-propagation: every step of DBP trainable.

Digital back-propagation (DBP) runs the split-step solution of the link
backwards: :math:`L_J N_J \cdots L_1 N_1 L_0`, with linear steps
:math:`L_j = e^{E_j(\omega)}` (dispersion, loss, amplifier gains of the
receiver's link model) and Kerr steps :math:`N_j: A \mapsto Ae^{i\kappa_jP}`,
:math:`\kappa_j = \gamma_\mathrm{eff}c_j`. :func:`backpropagate` learns the
*physical* parameters of that model (per-span :math:`\beta_2`, :math:`\gamma`,
…). Here every step gets its own free parameters, in the spirit of the
learned DBP of Häger and Pfister [HagerPfister2018]_:

* each linear step a smooth phase and log-gain correction,
  :math:`L_j' = L_j\,e^{i\delta\phi_j(\omega) + \delta g_j(\omega)}`, piecewise
  linear in :math:`\omega` on ``knots`` points across the band;
* each Kerr step its own coefficient :math:`\kappa_j'` and a real FIR
  filter :math:`h_j` of ``2·intensity_taps + 1`` taps on the power,
  :math:`A \mapsto A\,e^{i\kappa_j'(h_j * P)}` — the filtered DBP of
  [DuLowery2010]_ and enhanced split step of [Secondini2016]_, which can
  follow walk-off and the low-pass nature of the nonlinearity at long
  steps.

:func:`learned_dbp_init` returns the structure (fixed arrays) and the
parameters that reproduce standard DBP exactly; training only moves away
from it. :func:`learned_dbp` is a pure JAX function, usable as the
``compensation`` stage of :func:`demodulate`.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from hpcom._typing import StepDistribution
from hpcom.amplifier import EDFA
from hpcom.autodiff._fiber import (
    _KM,
    Array,
    FiberParameters,
    _kerr_lengths_km,
    _operators,
)
from hpcom.autodiff._link import _recentred
from hpcom.backends.base import split_step_substeps
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.numerics import Scheme

__all__ = ["LearnedDBP", "LearnedDBPParameters", "learned_dbp", "learned_dbp_init"]


class LearnedDBPParameters(NamedTuple):
    """Trainable parameters of :func:`learned_dbp` (a JAX pytree).

    ``phase``, ``log_gain``: ``(J+1, knots)`` corrections of the linear
    steps at the knots; ``nonlinear``: ``(J,)`` Kerr coefficients
    :math:`\\kappa_j` (rad/W); ``intensity``: ``(J, 2M+1)`` taps of the
    power filters (a centred 1 is no filter).
    """

    phase: Array
    log_gain: Array
    nonlinear: Array
    intensity: Array


class LearnedDBP(NamedTuple):
    """The fixed structure of a learned DBP (from :func:`learned_dbp_init`).

    ``base``: ``(J+1, n)`` transfer functions of the standard DBP's linear
    steps (amplifier gains folded in); ``grid``: the FFT frequencies scaled
    to [-1, 1); ``knots``: the knot positions on that scale (NumPy
    constants); ``dual``: the Kerr power sums two polarisations.
    """

    base: Array
    grid: Array
    knots: Array
    dual: bool


def learned_dbp_init(
    link: Link,
    n_samples: int,
    dt_s: float,
    *,
    steps_per_span: int | None = None,
    step_distribution: StepDistribution | None = None,
    step_log_factor: float | None = None,
    scheme: Scheme | None = None,
    frequency_offset_hz: float = 0.0,
    knots: int = 33,
    intensity_taps: int = 0,
) -> tuple[LearnedDBP, LearnedDBPParameters]:
    """Structure and initial parameters: standard DBP of ``link`` at this sampling.

    Step settings as :class:`~hpcom.DigitalBackPropagation` (``None``: each
    fiber's own). A channel at ``frequency_offset_hz`` uses the recentred
    fibers and removes the constant phase, as :func:`backpropagate` does.
    Supports scalar and Manakov fibers with Kerr nonlinearity (PMD is left
    out of the model); amplifiers are noiseless gains.
    """
    exponents: list[Any] = []
    kappas: list[Any] = []
    phase = 0.0
    dual = None
    with jax.enable_x64(True):  # the structure in double precision, kept as NumPy constants
        pending = jnp.zeros(n_samples, jnp.complex128)
        for el in reversed(link.elements):
            if isinstance(el, EDFA):
                pending = pending - 0.5 * math.log(el.gain_linear)
                continue
            if type(el) is not Fiber:
                raise TypeError(
                    f"learned DBP supports Fiber and EDFA elements, got {type(el).__name__}"
                )
            if el.raman is not None or el.self_steepening or el.nonlinear_gain_per_w_per_km:
                raise ValueError("learned DBP models the Kerr nonlinearity only")
            if el.pmd_ps_per_sqrt_km > 0:
                el = el.replace(pmd_ps_per_sqrt_km=0.0)
            model = el.recentred(frequency_offset_hz).with_steps(
                steps_per_span, step_distribution, step_log_factor
            )
            if scheme is not None:
                model = model.replace(scheme=scheme)
            if not model.scheme.is_split_step:
                raise ValueError(
                    f"learned DBP needs a split-step scheme, got {model.scheme.method!r}"
                )
            params, phi = _recentred(FiberParameters.from_fiber(el), frequency_offset_hz)
            params = params._replace(pmd_ps_per_sqrt_km=None)
            ops = _operators(params, model, n_samples, dt_s, jnp.float64)
            linear_m, sub_m, _ = split_step_substeps(model, "backward")
            kerr = _kerr_lengths_km(model, sub_m, ops, jnp.float64)
            linear = linear_m / _KM
            pending = pending + ops.exponent * linear[0]
            for j in range(sub_m.size):
                exponents.append(pending)
                kappas.append(ops.gamma_eff * kerr[j])
                pending = ops.exponent * linear[j + 1]
            phase += float(phi) * el.length_km
            dual = model.polarization == "dual" if dual is None else dual
        if dual is None:
            raise ValueError("the link has no fiber")
        exponents.append(pending - 1j * phase)
        base = np.exp(np.asarray(jnp.stack(exponents)))
        kappa = np.asarray(jnp.stack(kappas), dtype=np.float64)
    n_kerr = kappa.size
    taps = np.zeros((n_kerr, 2 * intensity_taps + 1))
    taps[:, intensity_taps] = 1.0
    structure = LearnedDBP(
        base=base,
        grid=np.fft.fftfreq(n_samples) * 2.0,
        knots=np.linspace(-1.0, 1.0, knots),
        dual=bool(dual),
    )
    dtype = jnp.result_type(float)
    params0 = LearnedDBPParameters(
        phase=jnp.zeros((n_kerr + 1, knots), dtype),
        log_gain=jnp.zeros((n_kerr + 1, knots), dtype),
        nonlinear=jnp.asarray(kappa, dtype),
        intensity=jnp.asarray(taps, dtype),
    )
    return structure, params0


def _power_filter(taps: Array, n: int, dtype: Any) -> Array:
    """Frequency response of a centred FIR ``taps`` (length 2M+1) on an ``n``-point circle."""
    m = (taps.shape[-1] - 1) // 2
    circle = jnp.zeros(n, dtype).at[: m + 1].set(taps[m:].astype(dtype))
    if m:
        circle = circle.at[n - m :].set(taps[:m].astype(dtype))
    return jnp.fft.fft(circle)


def learned_dbp(field: Array, structure: LearnedDBP, params: LearnedDBPParameters) -> Array:
    r"""Back-propagate ``field`` (``(..., n)`` or ``(..., 2, n)``) through the learned DBP.

    :math:`L_J'N_J'\cdots L_1'N_1'L_0'`; with the parameters of
    :func:`learned_dbp_init` it equals :func:`backpropagate`. Differentiable
    in ``field`` and every leaf of ``params``; the precision follows
    ``field`` (complex64 works).
    """
    x = jnp.asarray(field)
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        x = x.astype(jnp.result_type(x.dtype, jnp.complex64))
    cdtype = x.dtype
    rdtype = jnp.real(x).dtype

    def correction(values: Array) -> Array:
        return jax.vmap(lambda v: jnp.interp(structure.grid, structure.knots, v))(values)

    filters = structure.base.astype(cdtype) * jnp.exp(
        (1j * correction(params.phase) + correction(params.log_gain)).astype(cdtype)
    )
    filtered = params.intensity.shape[-1] > 1
    if filtered:
        kappas = params.nonlinear
        n = x.shape[-1]
        responses = jax.vmap(lambda t: _power_filter(t, n, cdtype))(params.intensity)
    else:  # one tap is a scale of κ
        kappas = params.nonlinear * params.intensity[:, 0]
        responses = None

    def kerr(y: Array, kappa: Array, response: Array | None) -> Array:
        p = jnp.real(y) ** 2 + jnp.imag(y) ** 2
        if structure.dual:
            p = jnp.sum(p, axis=-2, keepdims=True)
        if response is not None:
            p = jnp.real(jnp.fft.ifft(jnp.fft.fft(p.astype(cdtype), axis=-1) * response, axis=-1))
        return y * jnp.exp(1j * (kappa.astype(rdtype) * p))

    def body(spectrum: Array, xs: Any) -> tuple[Array, None]:
        if filtered:
            kappa, response, next_filter = xs
        else:
            (kappa, next_filter), response = xs, None
        y = kerr(jnp.fft.ifft(spectrum, axis=-1), kappa, response)
        return jnp.fft.fft(y, axis=-1) * next_filter, None

    xs = (kappas, responses, filters[1:]) if filtered else (kappas, filters[1:])
    spectrum, _ = lax.scan(body, jnp.fft.fft(x, axis=-1) * filters[0], xs)
    return jnp.fft.ifft(spectrum, axis=-1)
