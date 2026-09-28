r"""Additive white Gaussian noise as a link component (noise loading).

:class:`AdditiveNoise` adds circular complex white Gaussian noise of one-sided
PSD :math:`N_0` **per polarisation** — the same noise model as the ASE of
:class:`~hpcom.amplifier.EDFA`, but without gain. Put it in a
:class:`~hpcom.link.Link` to sweep SNR or OSNR back-to-back.

Two constructors translate the usual figures of merit into :math:`N_0`:

* :meth:`AdditiveNoise.for_snr` — per-symbol SNR of one polarisation
  tributary, :math:`\mathrm{SNR} = E_s/N_0 = P_\text{pol}/(N_0 R_s)`, which is
  exactly the SNR at the output of hpcom's RRC matched filter.
* :meth:`AdditiveNoise.for_osnr` — OSNR in a reference bandwidth
  :math:`B_\text{ref}` (default 12.5 GHz ≈ 0.1 nm at 1550 nm), counting the
  noise in *both* polarisations:
  :math:`\mathrm{OSNR} = P_\text{ch} / (2 N_0 B_\text{ref})`. For a
  dual-polarisation channel this gives :math:`\mathrm{SNR} = \mathrm{OSNR}\,
  B_\text{ref}/R_s` (Essiambre et al., JLT 2010).
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray
from hpcom.units import db_to_linear

__all__ = ["AdditiveNoise"]


class AdditiveNoise(HpcomModel):
    """White complex Gaussian noise, PSD ``psd_per_polarization_w_per_hz`` per polarisation.

    Each sample of each polarisation receives variance ``N₀/Δt`` (half per
    quadrature). Accepts any field shape; the input is not modified.
    """

    psd_per_polarization_w_per_hz: float = Field(ge=0)

    @classmethod
    def for_snr(
        cls, snr_db: float, *, power_per_polarization_w: float, symbol_rate_hz: float
    ) -> AdditiveNoise:
        """Noise giving per-symbol SNR ``Es/N0 = P_pol/(N0·R_s)`` of ``snr_db``."""
        return cls(
            psd_per_polarization_w_per_hz=power_per_polarization_w
            / (db_to_linear(snr_db) * symbol_rate_hz)
        )

    @classmethod
    def for_osnr(
        cls, osnr_db: float, *, channel_power_w: float, reference_bandwidth_hz: float = 12.5e9
    ) -> AdditiveNoise:
        """Noise giving OSNR ``P_ch/(2·N0·B_ref)`` of ``osnr_db`` (noise counted in both polarisations)."""
        return cls(
            psd_per_polarization_w_per_hz=channel_power_w
            / (2.0 * db_to_linear(osnr_db) * reference_bandwidth_hz)
        )

    def apply(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """:class:`~hpcom._typing.LinkComponent` interface: return ``field + noise``."""
        field = np.asarray(field, dtype=np.complex128)
        if self.psd_per_polarization_w_per_hz == 0.0:
            clean: ComplexArray = field.copy()
            return clean
        if rng is None:
            rng = np.random.default_rng()
        sigma = np.sqrt(self.psd_per_polarization_w_per_hz / dt_s / 2.0)
        out: ComplexArray = field + sigma * (
            rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        )
        return out

    def apply_inverse(self, field: ComplexArray, dt_s: float) -> ComplexArray:
        """Noise cannot be undone; back-propagation passes the field through."""
        del dt_s
        return np.asarray(field, dtype=np.complex128)
