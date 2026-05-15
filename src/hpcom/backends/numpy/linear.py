"""Linear-only fiber propagation in NumPy.

Implements pure dispersion (β₂) and attenuation (α). Nonlinearity (γ) is
ignored. Useful for:

  * validating the :class:`~hpcom.backends.base.Propagator` interface without
    pulling in TensorFlow,
  * fast CPU sanity checks,
  * the closed-form linear-dispersion correctness test in ``tests/numerical/``.

The dispersion operator follows the optical-comm convention with NumPy's
FFT sign convention :math:`\\hat A(\\omega) = \\int A(t) e^{-i\\omega t} dt`,
so the frequency-domain phase shift over distance ``z`` is
:math:`\\exp(+i \\beta_2 \\omega^2 z / 2)`.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends.base import Propagator


class LinearNumpyPropagator(Propagator):
    """Linear (dispersion + loss) fiber propagation, no nonlinearity."""

    name = "numpy-linear"

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

        n = field.shape[-1]
        length_m = fiber.length_km * 1e3
        if direction == "backward":
            length_m = -length_m

        # Loss in field amplitude. α [dB/km] → Np/m.
        alpha_np_per_m = fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3
        loss_amp = np.exp(-alpha_np_per_m * length_m / 2.0)

        # Angular frequency grid.
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt_s)
        disp_phase = np.exp(1j * fiber.beta2_s2_per_m / 2.0 * omega**2 * length_m)

        spectrum = np.fft.fft(field, axis=-1) * disp_phase
        out = np.fft.ifft(spectrum, axis=-1) * loss_amp
        return cast(ComplexArray, out)
