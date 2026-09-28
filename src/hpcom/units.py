"""Physical constants and unit conversions used throughout hpcom.

hpcom's public API is SI except where a field name states otherwise
(``length_km``, ``alpha_db_per_km``, ``gamma_per_w_per_km``, ``*_dbm``). The
helpers here convert between those and the units common in the literature
(ps/(nm·km), dBm, …) so user code never needs to hard-code a factor.
"""

from __future__ import annotations

import math

#: Speed of light in vacuum, m/s (exact).
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
#: Planck constant, J·s (exact).
PLANCK_J_S = 6.62607015e-34


def dbm_to_w(power_dbm: float) -> float:
    """Power in dBm → watts: ``P = 1 mW · 10^(dBm/10)``."""
    return 1e-3 * math.pow(10.0, power_dbm / 10.0)


def w_to_dbm(power_w: float) -> float:
    """Power in watts → dBm. Raises for non-positive power."""
    if power_w <= 0:
        raise ValueError(f"power must be > 0 W to express in dBm, got {power_w}")
    return 10.0 * math.log10(power_w / 1e-3)


def db_to_linear(value_db: float) -> float:
    """Power ratio in dB → linear."""
    return math.pow(10.0, value_db / 10.0)


def linear_to_db(value: float) -> float:
    """Linear power ratio → dB. Raises for non-positive ratios."""
    if value <= 0:
        raise ValueError(f"ratio must be > 0 to express in dB, got {value}")
    return 10.0 * math.log10(value)


def wavelength_to_frequency_hz(wavelength_m: float) -> float:
    """Vacuum wavelength → optical frequency, ``ν = c / λ``."""
    return SPEED_OF_LIGHT_M_PER_S / wavelength_m


def dispersion_to_beta2(dispersion_ps_per_nm_per_km: float, wavelength_m: float = 1550e-9) -> float:
    """Chromatic dispersion D [ps/(nm·km)] → β₂ [s²/m].

    ``β₂ = −λ² D / (2π c)``. Standard SSMF (D = 17 ps/(nm·km) at 1550 nm)
    gives β₂ ≈ −2.17e-26 s²/m (−21.7 ps²/km).
    """
    d_si = dispersion_ps_per_nm_per_km * 1e-6  # ps/(nm·km) → s/m²
    return -(wavelength_m**2) * d_si / (2.0 * math.pi * SPEED_OF_LIGHT_M_PER_S)


def beta2_to_dispersion(beta2_s2_per_m: float, wavelength_m: float = 1550e-9) -> float:
    """β₂ [s²/m] → chromatic dispersion D [ps/(nm·km)] (inverse of :func:`dispersion_to_beta2`)."""
    d_si = -2.0 * math.pi * SPEED_OF_LIGHT_M_PER_S * beta2_s2_per_m / wavelength_m**2
    return d_si * 1e6


def dispersion_slope_to_beta3(
    slope_ps_per_nm2_per_km: float,
    dispersion_ps_per_nm_per_km: float,
    wavelength_m: float = 1550e-9,
) -> float:
    """Dispersion slope S [ps/(nm²·km)] and D [ps/(nm·km)] → β₃ [s³/m].

    ``β₃ = (λ² / 2πc)² · (S + 2D/λ)`` (Agrawal, *Nonlinear Fiber Optics*,
    eq. 1.2.11 differentiated once more). SSMF (S = 0.057, D = 17) gives
    β₃ ≈ 1.2e-40 s³/m (0.12 ps³/km).
    """
    s_si = slope_ps_per_nm2_per_km * 1e3  # ps/(nm²·km) → s/m³
    d_si = dispersion_ps_per_nm_per_km * 1e-6
    factor = wavelength_m**2 / (2.0 * math.pi * SPEED_OF_LIGHT_M_PER_S)
    return factor**2 * (s_si + 2.0 * d_si / wavelength_m)


def alpha_db_per_km_to_per_m(alpha_db_per_km: float) -> float:
    """Power attenuation α [dB/km] → α [1/m] (``P(z) = P(0)·exp(−α z)``)."""
    return alpha_db_per_km * math.log(10.0) / 10.0 / 1e3


__all__ = [
    "PLANCK_J_S",
    "SPEED_OF_LIGHT_M_PER_S",
    "alpha_db_per_km_to_per_m",
    "beta2_to_dispersion",
    "db_to_linear",
    "dbm_to_w",
    "dispersion_slope_to_beta3",
    "dispersion_to_beta2",
    "linear_to_db",
    "w_to_dbm",
    "wavelength_to_frequency_hz",
]
