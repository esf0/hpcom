"""Exact linear fiber propagation (loss + dispersion) in NumPy.

Solves the NLSE with ``γ = 0`` in closed form — one FFT, one multiply, one
inverse FFT — using the same linear operator as the SSFM backends
(:func:`hpcom.backends.base.linear_operator_exponent`). ``fiber.n_steps`` and
``fiber.gamma_per_w_per_km`` (and a nonlinear gain) are ignored; unsaturated
gain (:class:`hpcom.laser.GainFiber`) is linear and included, saturated gain
is rejected. Useful as an analytic reference and for fast linear-regime
studies.
"""

from __future__ import annotations

import numpy as np

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import (
    ALL_METHODS,
    Propagator,
    apply_jones,
    check_field_shape,
    core_axis,
    linear_operator_exponent,
    multicore_linear_system,
    pmd_pieces,
    pmd_realization,
    pmd_rotate,
    saturation_energy_j,
    signed_length_m,
)


class LinearNumpyPropagator(Propagator):
    """Linear (dispersion + loss) propagation; the Kerr term is ignored."""

    name = "numpy-linear"
    supported_methods = ALL_METHODS  # exact: the scheme and steps are irrelevant

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        a = np.asarray(field, dtype=np.complex128)
        check_field_shape(a, fiber)
        if saturation_energy_j(fiber) is not None:
            raise ValueError(
                "saturated gain depends on the field's energy (nonlinear); the numpy-linear "
                "backend cannot propagate it — use a nonlinear backend"
            )
        length_m = signed_length_m(fiber, direction)
        if fiber.n_cores == 1:
            transfer = np.exp(linear_operator_exponent(a.shape[-1], dt_s, fiber) * length_m)
            pmd = pmd_realization(fiber)
            if pmd is None:
                out: ComplexArray = np.fft.ifft(np.fft.fft(a, axis=-1) * transfer, axis=-1)
                return out
            # T = Q_N Π_k e^{−iωτ_s b_k·σ/2}; backward is its inverse (Q_N† first).
            if direction == "backward":
                a = apply_jones(a, pmd.output_rotation.conj().T)
            spectrum = np.fft.fft(a, axis=-1) * transfer
            start = 0.0 if direction == "forward" else fiber.length_km * 1e3
            sections, lengths = pmd_pieces(pmd, start, np.array([length_m]))
            half_dgd = np.pi * np.fft.fftfreq(a.shape[-1], d=dt_s) * pmd.dgd_per_m
            for k, ell in zip(sections[0], lengths[0], strict=True):
                spectrum = pmd_rotate(spectrum, pmd.axes[k], half_dgd * ell)
            field = np.fft.ifft(spectrum, axis=-1)
            if direction == "forward":
                field = apply_jones(field, pmd.output_rotation)
            return field
        vectors, exponent = multicore_linear_system(a.shape[-1], dt_s, fiber)
        axis = core_axis(fiber)
        x = np.moveaxis(np.fft.fft(a, axis=-1), axis, -1)  # (..., n, M)
        z = (np.conj(np.swapaxes(vectors, -1, -2)) @ x[..., None])[..., 0]
        y = (vectors @ (z * np.exp(exponent * length_m))[..., None])[..., 0]
        coupled: ComplexArray = np.fft.ifft(np.moveaxis(y, -1, axis), axis=-1)
        return coupled
