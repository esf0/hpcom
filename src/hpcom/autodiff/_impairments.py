"""Transceiver impairments (:mod:`hpcom.impairments`) as JAX functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from hpcom.autodiff._fiber import Array
from hpcom.impairments import (
    IQImbalance,
    LaserPhaseNoise,
    LocalOscillator,
    MachZehnderModulator,
    Quantizer,
)

__all__ = ["IMPAIRMENTS", "apply_impairment", "is_random"]

#: The impairment element types :func:`hpcom.autodiff.propagate_link` applies.
IMPAIRMENTS = (LaserPhaseNoise, LocalOscillator, IQImbalance, Quantizer, MachZehnderModulator)

Impairment = LaserPhaseNoise | LocalOscillator | IQImbalance | Quantizer | MachZehnderModulator


def is_random(element: object) -> bool:
    """``True`` if the element draws random numbers (needs a key)."""
    return isinstance(element, LaserPhaseNoise | LocalOscillator) and element.linewidth_hz > 0


def _wiener(
    key: Array, shape: tuple[int, ...], linewidth_hz: float, dt_s: float, dtype: object
) -> Array:
    steps = jax.random.normal(key, shape, dtype) * math.sqrt(2.0 * math.pi * linewidth_hz * dt_s)
    steps = steps.at[..., 0].set(0.0)
    return jnp.cumsum(steps, axis=-1)


def _phase_shape(shape: tuple[int, ...], polarization: str) -> tuple[int, ...]:
    return (*shape[:-2], 1, shape[-1]) if polarization == "dual" else shape


def _rms(a: Array) -> Array:
    return jnp.sqrt(jnp.mean(jnp.real(a) ** 2 + jnp.imag(a) ** 2, axis=-1, keepdims=True) / 2.0)


def apply_impairment(element: Impairment, field: Array, dt_s: float, key: Array | None) -> Array:
    """``element.apply`` in JAX: random ones draw from ``key``; the quantiser passes gradients straight through."""
    a = jnp.asarray(field)
    rdtype = jnp.real(a).dtype
    if isinstance(element, LaserPhaseNoise | LocalOscillator):
        phase = jnp.zeros((), rdtype)
        if isinstance(element, LocalOscillator):
            phase = (
                -2.0
                * math.pi
                * element.frequency_offset_hz
                * dt_s
                * jnp.arange(a.shape[-1], dtype=rdtype)
            )
        if element.linewidth_hz > 0:
            if key is None:
                raise ValueError(f"{type(element).__name__} with a linewidth needs a JAX PRNG key")
            wiener = _wiener(
                key, _phase_shape(a.shape, element.polarization), element.linewidth_hz, dt_s, rdtype
            )
            phase = phase + (wiener if isinstance(element, LaserPhaseNoise) else -wiener)
        return a * jnp.exp(1j * phase)
    if isinstance(element, IQImbalance):
        mu, nu = element.coefficients
        return mu * a + nu * jnp.conj(a)
    if isinstance(element, Quantizer):
        full_scale = element.clipping_ratio * _rms(a)
        step = jnp.where(full_scale > 0, 2.0 * full_scale / 2**element.bits, 1.0)
        top = 2 ** (element.bits - 1) - 1

        def quantise(x: Array) -> Array:
            return (jnp.clip(jnp.floor(x / step), -top - 1, top) + 0.5) * step

        q = jnp.where(full_scale > 0, quantise(jnp.real(a)) + 1j * quantise(jnp.imag(a)), 0)
        return a + lax.stop_gradient(q - a)  # straight-through: the gradient of the identity
    if isinstance(element, MachZehnderModulator):
        sigma = _rms(a)
        scale = element.drive * math.pi / 2.0 / jnp.where(sigma > 0, sigma, 1.0)
        y = jnp.sin(scale * jnp.real(a)) + 1j * jnp.sin(scale * jnp.imag(a))
        power_in = jnp.mean(jnp.abs(a) ** 2, axis=-1, keepdims=True)
        power_out = jnp.mean(jnp.abs(y) ** 2, axis=-1, keepdims=True)
        return y * jnp.sqrt(power_in / jnp.where(power_out > 0, power_out, 1.0))
    raise TypeError(f"{type(element).__name__} is not an impairment")
