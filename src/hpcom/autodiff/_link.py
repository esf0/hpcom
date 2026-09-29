"""Differentiable amplifiers, links and digital back-propagation."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hpcom._typing import StepDistribution
from hpcom.amplifier import EDFA
from hpcom.autodiff._fiber import _PS, Array, FiberParameters, GradientMode, propagate
from hpcom.dsp.compensation import DigitalBackPropagation
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.numerics import Scheme
from hpcom.units import PLANCK_J_S, wavelength_to_frequency_hz

__all__ = [
    "AmplifierParameters",
    "LinkParameters",
    "amplify",
    "backpropagate",
    "compensate",
    "link_parameters",
    "propagate_link",
]


class AmplifierParameters(NamedTuple):
    """The continuous parameters of an :class:`~hpcom.EDFA` as a JAX pytree.

    ``noise_figure_db`` is ``None`` for a noiseless amplifier.
    """

    gain_db: Array
    noise_figure_db: Array | None = None

    @classmethod
    def from_amplifier(cls, amplifier: EDFA) -> AmplifierParameters:
        dtype = jnp.result_type(float)
        nf = None if amplifier.is_noiseless else jnp.asarray(amplifier.noise_figure_db, dtype)
        return cls(gain_db=jnp.asarray(amplifier.gain_db, dtype), noise_figure_db=nf)


#: One entry per link element: :class:`FiberParameters` or :class:`AmplifierParameters`.
LinkParameters = tuple[FiberParameters | AmplifierParameters, ...]


def link_parameters(link: Link) -> LinkParameters:
    """The parameters of every element of ``link``, in order."""
    out: list[FiberParameters | AmplifierParameters] = []
    for el in link.elements:
        if isinstance(el, Fiber):
            out.append(FiberParameters.from_fiber(el))
        elif isinstance(el, EDFA):
            out.append(AmplifierParameters.from_amplifier(el))
        else:
            raise TypeError(
                f"{type(el).__name__} has no differentiable model; links of Fiber and EDFA "
                "elements are supported"
            )
    return tuple(out)


def amplify(
    field: Array,
    amplifier: EDFA,
    dt_s: float,
    *,
    key: Array | None = None,
    params: AmplifierParameters | None = None,
) -> Array:
    r""":meth:`hpcom.EDFA.amplify` as a JAX function: :math:`\sqrt G A + n`.

    The ASE :math:`n` (per-polarisation PSD :math:`(GF-1)h\nu/2`, variance
    :math:`S_\mathrm{ASE}/\Delta t` per complex sample) is drawn from the
    JAX PRNG ``key`` — required unless the amplifier is noiseless — and is
    reparameterised, so gradients flow through the gain and noise figure.
    """
    p = AmplifierParameters.from_amplifier(amplifier) if params is None else params
    a = jnp.asarray(field)
    rdtype = jnp.real(a).dtype
    gain = 10.0 ** (jnp.asarray(p.gain_db, rdtype) / 10.0)
    out = a * jnp.sqrt(gain)
    if p.noise_figure_db is None:
        return out
    if key is None:
        raise ValueError("a noisy amplifier needs a JAX PRNG key")
    h_nu = PLANCK_J_S * wavelength_to_frequency_hz(amplifier.wavelength_m)
    nf = 10.0 ** (jnp.asarray(p.noise_figure_db, rdtype) / 10.0)
    variance = (gain * nf - 1.0) * (h_nu / 2.0 / dt_s)
    k_re, k_im = jax.random.split(key)
    noise = jax.random.normal(k_re, a.shape, rdtype) + 1j * jax.random.normal(k_im, a.shape, rdtype)
    return out + jnp.sqrt(variance / 2.0) * noise


def propagate_link(
    field: Array,
    link: Link,
    dt_s: float,
    *,
    key: Array | None = None,
    params: LinkParameters | None = None,
    gradient: GradientMode = "store",
) -> Array:
    """:meth:`hpcom.Link.propagate` as a JAX function (Fiber and EDFA elements).

    Every noisy amplifier draws from its own split of ``key``.
    """
    ps = link_parameters(link) if params is None else params
    n_noisy = sum(isinstance(el, EDFA) and not el.is_noiseless for el in link.elements)
    keys = list(jax.random.split(key, n_noisy)) if key is not None and n_noisy else []
    x = jnp.asarray(field)
    for el, p in zip(link.elements, ps, strict=True):
        if isinstance(el, Fiber):
            x = propagate(x, el, dt_s, params=p, gradient=gradient)  # type: ignore[arg-type]
        elif isinstance(el, EDFA):
            k = keys.pop(0) if not el.is_noiseless and keys else None
            x = amplify(x, el, dt_s, key=k, params=p)  # type: ignore[arg-type]
        else:
            raise TypeError(f"{type(el).__name__} has no differentiable model")
    return x


def _recentred(
    params: FiberParameters, frequency_offset_hz: float
) -> tuple[FiberParameters, Array]:
    """``params`` of ``fiber.recentred(f_k)`` and the phase D(ω_k) per km (see Fiber.recentred)."""
    w = 2.0 * math.pi * frequency_offset_hz * _PS  # rad/ps
    betas = params.betas_psn_per_km
    n = betas.shape[0]
    shifted = jnp.stack(
        [
            sum(betas[k - 1] * (-w) ** (k - m) / math.factorial(k - m) for k in range(m, n + 1))
            for m in range(1, n + 1)
        ]
    )
    phase = sum(betas[k - 1] * (-w) ** k / math.factorial(k) for k in range(1, n + 1))
    changes: dict[str, Any] = {"betas_psn_per_km": shifted}
    if params.shock_time_fs is not None:
        factor = 1.0 - params.shock_time_fs * 1e-3 * w
        changes["gamma_per_w_per_km"] = params.gamma_per_w_per_km * factor
        changes["shock_time_fs"] = params.shock_time_fs / factor
    if params.core_beta1_offsets_ps_per_km is not None:
        base = params.core_beta0_offsets_per_km
        skew = params.core_beta1_offsets_ps_per_km * w
        changes["core_beta0_offsets_per_km"] = -skew if base is None else base - skew
    return params._replace(**changes), phase


def backpropagate(
    field: Array,
    link: Link,
    dt_s: float,
    *,
    params: LinkParameters | None = None,
    steps_per_span: int | None = None,
    step_distribution: StepDistribution | None = None,
    step_log_factor: float | None = None,
    scheme: Scheme | None = None,
    nonlinearity_scale: Array = 1.0,
    frequency_offset_hz: float = 0.0,
    gradient: GradientMode = "store",
) -> Array:
    r"""Digital back-propagation through the link model — as :class:`~hpcom.DigitalBackPropagation`.

    Last element first: each fiber backward (``steps_per_span``, placement
    and ``scheme`` overriding its own), with :math:`\gamma \to
    \xi\gamma` (``nonlinearity_scale``, may be traced), each amplifier as
    the noiseless :math:`1/\sqrt G`. A channel at ``frequency_offset_hz``
    uses the recentred fiber and removes the constant phase
    :math:`D(\omega_k)L`, as the NumPy implementation does. Differentiable
    in ``field``, ``params`` (the receiver's link model — learned DBP) and
    ``nonlinearity_scale``.
    """
    ps = link_parameters(link) if params is None else params
    x = jnp.asarray(field)
    phase = 0.0
    for el, p in zip(reversed(link.elements), reversed(ps), strict=True):
        if isinstance(el, Fiber):
            model = el.recentred(frequency_offset_hz).with_steps(
                steps_per_span, step_distribution, step_log_factor
            )
            if scheme is not None:
                model = model.replace(scheme=scheme)
            q, phi = _recentred(p, frequency_offset_hz)  # type: ignore[arg-type]
            q = q._replace(gamma_per_w_per_km=q.gamma_per_w_per_km * nonlinearity_scale)
            x = propagate(x, model, dt_s, params=q, direction="backward", gradient=gradient)
            phase = phase + phi * el.length_km
        elif isinstance(el, EDFA):
            gain = 10.0 ** (jnp.asarray(p.gain_db) / 10.0)  # type: ignore[union-attr]
            x = x / jnp.sqrt(gain).astype(jnp.real(x).dtype)
        else:
            raise TypeError(f"{type(el).__name__} has no differentiable model")
    if frequency_offset_hz == 0.0:
        return x
    return x * jnp.exp(-1j * jnp.asarray(phase, jnp.real(x).dtype))


def compensate(
    field: Array,
    dbp: DigitalBackPropagation,
    dt_s: float,
    *,
    params: LinkParameters | None = None,
    nonlinearity_scale: Array | None = None,
    frequency_offset_hz: float = 0.0,
    gradient: GradientMode = "store",
) -> Array:
    """:meth:`hpcom.DigitalBackPropagation.compensate` as a JAX function.

    ``params`` (default: from ``dbp.link``) and ``nonlinearity_scale``
    (default: ``dbp.nonlinearity_scale``) are the trainable quantities.
    ``field`` must already be at the compensator's sampling rate.
    """
    return backpropagate(
        field,
        dbp.link,
        dt_s,
        params=params,
        steps_per_span=dbp.steps_per_span,
        step_distribution=dbp.step_distribution,
        step_log_factor=dbp.step_log_factor,
        scheme=dbp.scheme,
        nonlinearity_scale=dbp.nonlinearity_scale
        if nonlinearity_scale is None
        else nonlinearity_scale,
        frequency_offset_hz=frequency_offset_hz,
        gradient=gradient,
    )
