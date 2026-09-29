r"""Delayed (Raman) response of silica for the generalised NLSE.

The Kerr term of the generalised NLSE (``docs/physics.rst``, *Generalised
NLSE*) is :math:`i\gamma A\,(R * |A|^2)` with the normalised response

.. math::

    R(t) = (1 - f_R)\,\delta(t) + f_R\,h_R(t), \qquad \int h_R\,dt = 1,

an instantaneous electronic part and a delayed nuclear (Raman) part.
:class:`RamanResponse` gives :math:`h_R` in closed form and, since hpcom
applies the convolution in the frequency domain, its exact transfer function

.. math:: H_R(\omega) = \int_0^\infty h_R(t)\,e^{-i\omega t}\,dt

in hpcom's (NumPy) Fourier convention: :math:`\omega` is the baseband
frequency of the grid, and :math:`\omega > 0` is the *lower* optical
frequency, so the Stokes side of a pump is at higher :math:`\omega` and
:math:`\operatorname{Im} H_R(\omega) < 0` there. Using the analytic
:math:`H_R` rather than a sampled :math:`h_R(t)` avoids the discretisation
error of sampling a response whose time constants (12–32 fs) are
comparable to the sample period of a telecom simulation.

Models
------

``"blow-wood"`` (Blow & Wood 1989), :math:`f_R = 0.18`:

.. math::

    h_a(t) = \frac{\tau_1^2+\tau_2^2}{\tau_1\tau_2^2}\,e^{-t/\tau_2}\sin(t/\tau_1),
    \qquad
    H_a(\omega) = \frac{a^2+b^2}{(a+i\omega)^2+b^2},\quad a = 1/\tau_2,\ b = 1/\tau_1,

with :math:`\tau_1 = 12.2` fs, :math:`\tau_2 = 32` fs (a single damped
oscillator at the 13.2 THz gain peak).

``"lin-agrawal"`` (Lin & Agrawal 2006), :math:`f_R = 0.245`: adds the
anisotropic (boson-peak) part that the single oscillator misses at low
frequency shifts,

.. math::

    h_R = (1 - f_b)\,h_a + f_b\,h_b,\qquad
    h_b(t) = \frac{2\tau_b - t}{\tau_b^2}\,e^{-t/\tau_b},\qquad
    H_b(\omega) = \frac{2u-1}{u^2},\quad u = 1 + i\omega\tau_b,

with :math:`f_b = 0.21` and :math:`\tau_b = 96` fs.

Both are for co-polarised light (the scalar equation).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import HpcomModel

__all__ = ["RamanModel", "RamanResponse"]

#: Closed-form Raman response models (see the module docstring).
RamanModel = Literal["blow-wood", "lin-agrawal"]

#: Raman fraction f_R of each model as published.
_DEFAULT_FRACTION: dict[str, float] = {"blow-wood": 0.18, "lin-agrawal": 0.245}
#: Share f_b of the anisotropic response in Lin & Agrawal's model.
_LIN_AGRAWAL_FB = 0.21


class RamanResponse(HpcomModel):
    r"""Normalised delayed response :math:`h_R` and fraction :math:`f_R` (module docstring).

    Attributes
    ----------
    model
        :data:`RamanModel`.
    fraction
        :math:`f_R`; ``None``: the model's published value
        (0.18 Blow–Wood, 0.245 Lin–Agrawal).
    tau1_s, tau2_s
        Oscillation period / 2π and damping time of :math:`h_a`.
    tau_b_s
        Decay time of Lin & Agrawal's anisotropic part :math:`h_b`.
    """

    model: RamanModel = "blow-wood"
    fraction: float | None = Field(default=None, ge=0.0, lt=1.0)
    tau1_s: float = Field(default=12.2e-15, gt=0)
    tau2_s: float = Field(default=32e-15, gt=0)
    tau_b_s: float = Field(default=96e-15, gt=0)

    @property
    def f_r(self) -> float:
        """Raman fraction :math:`f_R` in use."""
        return _DEFAULT_FRACTION[self.model] if self.fraction is None else float(self.fraction)

    @property
    def _anisotropic_share(self) -> float:
        return _LIN_AGRAWAL_FB if self.model == "lin-agrawal" else 0.0

    def transfer(self, omega_rad_per_s: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        r""":math:`H_R(\omega) = \int h_R(t)e^{-i\omega t}dt` (hpcom's convention); :math:`H_R(0) = 1`."""
        w = np.asarray(omega_rad_per_s, dtype=np.float64)
        a, b = 1.0 / self.tau2_s, 1.0 / self.tau1_s
        h_a = (a * a + b * b) / ((a + 1j * w) ** 2 + b * b)
        fb = self._anisotropic_share
        if fb == 0.0:
            return np.asarray(h_a, dtype=np.complex128)
        u = 1.0 + 1j * w * self.tau_b_s
        h_b = (2.0 * u - 1.0) / u**2
        return np.asarray((1.0 - fb) * h_a + fb * h_b, dtype=np.complex128)

    def impulse(self, t_s: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """:math:`h_R(t)` in 1/s (zero for ``t < 0``)."""
        t = np.asarray(t_s, dtype=np.float64)
        tp = np.maximum(t, 0.0)
        t1, t2 = self.tau1_s, self.tau2_s
        h = (t1 * t1 + t2 * t2) / (t1 * t2 * t2) * np.exp(-tp / t2) * np.sin(tp / t1)
        fb = self._anisotropic_share
        if fb:
            tb = self.tau_b_s
            h = (1.0 - fb) * h + fb * (2.0 * tb - tp) / tb**2 * np.exp(-tp / tb)
        return np.asarray(np.where(t >= 0.0, h, 0.0), dtype=np.float64)

    @property
    def response_time_s(self) -> float:
        r""":math:`T_R = f_R\int t\,h_R(t)\,dt` — the slope of the Raman gain at zero shift.

        Blow–Wood: :math:`2\tau_1^2\tau_2/(\tau_1^2+\tau_2^2)`; :math:`h_b` has
        zero first moment. Sets the soliton self-frequency shift of long
        pulses (Gordon 1986).
        """
        t1, t2 = self.tau1_s, self.tau2_s
        first_moment = (1.0 - self._anisotropic_share) * 2.0 * t1 * t1 * t2 / (t1 * t1 + t2 * t2)
        return self.f_r * first_moment

    def gain_per_w_per_m(
        self, gamma_per_w_per_km: float, frequency_shift_hz: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        r"""Raman gain :math:`g(\Delta\nu) = -2\gamma f_R\,\operatorname{Im}H_R(2\pi\Delta\nu)` in 1/(W·m).

        ``frequency_shift_hz`` is the pump optical frequency minus the
        signal's (positive: the signal is on the Stokes side and grows,
        :math:`dP_s/dz = g P_p P_s` for co-polarised CW waves). This is
        :math:`g_R/A_\mathrm{eff}` of the literature.
        """
        shift = 2.0 * np.pi * np.asarray(frequency_shift_hz, dtype=np.float64)
        gamma = gamma_per_w_per_km * 1e-3
        return np.asarray(-2.0 * gamma * self.f_r * self.transfer(shift).imag, dtype=np.float64)
