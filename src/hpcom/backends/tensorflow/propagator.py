"""Single-span fiber propagation on TensorFlow.

Thin adapter from hpcom's :class:`~hpcom.backends.base.Propagator` interface to
the vendored ``ssfm-gpu`` SSFM/Manakov kernels under
:mod:`hpcom.backends.tensorflow._vendor`. Pure fiber only — amplification and
ASE noise live on :class:`hpcom.amplifier.Amplifier`, applied between spans by
:class:`hpcom.link.Link`.

Polarization dispatch is driven by ``fiber.polarization``:

* ``"single"`` → scalar NLSE on a 1-D field via ``tf_fiber_propogate``.
* ``"dual"``   → Manakov on a 2-D field of shape ``(2, N)`` via
  ``tf_manakov_fiber_propogate``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import tensorflow as tf

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import Propagator
from hpcom.backends.tensorflow._vendor.propagation import (
    tf_fiber_propogate,
    tf_manakov_fiber_propogate,
)


def _alpha_db_per_km_to_np_per_km(alpha_db_per_km: float) -> float:
    """Convert α from dB/km to Np/km (the unit ``tf_fiber_propogate`` expects)."""
    return float(alpha_db_per_km * np.log(10.0) / 10.0)


def _signed_length_km(fiber: FiberLike, direction: Direction) -> float:
    if direction == "forward":
        return fiber.length_km
    if direction == "backward":
        return -fiber.length_km
    raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")


class TensorFlowPropagator(Propagator):
    """Backend that delegates to the vendored ssfm-gpu TF kernels.

    Stateless: each call rebuilds the per-step dispersion phase. Heavy lifting
    is in TensorFlow ops, which use GPU when one is available and fall back to
    CPU otherwise (a `WARNING` line from TF on CPU-only machines is harmless).
    """

    name = "tensorflow"

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        length_km = _signed_length_km(fiber, direction)
        n_steps = max(int(fiber.n_steps), 1)
        gamma = float(fiber.gamma_per_w_per_km)
        # FiberLike stores β₂ in s²/m; the vendored kernel uses s²/km.
        beta2_s2_per_km = float(fiber.beta2_s2_per_m) * 1e3
        alpha = _alpha_db_per_km_to_np_per_km(float(fiber.alpha_db_per_km))

        n_samples = field.shape[-1]
        t_span = float(dt_s) * n_samples
        field_c = np.ascontiguousarray(field, dtype=np.complex128)
        polarization = fiber.polarization

        if polarization == "single":
            if field_c.ndim != 1:
                raise ValueError(f"single-pol fiber expects 1-D field, got shape {field_c.shape}")
            sig = tf.constant(field_c, dtype=tf.complex128)
            out_tf = tf_fiber_propogate(  # type: ignore[no-untyped-call]
                sig,
                t_span,
                length_km,
                n_steps,
                gamma,
                beta2_s2_per_km,
                alpha=alpha,
                beta3=0.0,
            )
            return cast(ComplexArray, np.asarray(out_tf, dtype=np.complex128))

        if polarization == "dual":
            if field_c.ndim != 2 or field_c.shape[0] != 2:
                raise ValueError(
                    "dual-pol fiber expects 2-D field with leading axis 2; "
                    f"got shape {field_c.shape}"
                )
            sig_x = tf.constant(field_c[0], dtype=tf.complex128)
            sig_y = tf.constant(field_c[1], dtype=tf.complex128)
            x_out, y_out = tf_manakov_fiber_propogate(  # type: ignore[no-untyped-call]
                sig_x,
                sig_y,
                t_span,
                length_km,
                n_steps,
                gamma,
                beta2_s2_per_km,
                alpha=alpha,
                beta3=0.0,
            )
            stacked = np.stack(
                [np.asarray(x_out, dtype=np.complex128), np.asarray(y_out, dtype=np.complex128)],
                axis=0,
            )
            return cast(ComplexArray, stacked)

        raise ValueError(f"unknown polarization {polarization!r}")
