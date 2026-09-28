"""Unit conversions and constants."""

from __future__ import annotations

import math

import pytest

from hpcom import units


@pytest.mark.parametrize(
    ("dbm", "w"), [(0.0, 1e-3), (30.0, 1.0), (-30.0, 1e-6), (3.0, 1.9952623149688795e-3)]
)
def test_dbm_and_watts(dbm, w):
    assert units.dbm_to_w(dbm) == pytest.approx(w, rel=1e-15)
    assert units.w_to_dbm(w) == pytest.approx(dbm, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1e-3])
def test_w_to_dbm_rejects_non_positive_power(bad):
    with pytest.raises(ValueError, match="> 0"):
        units.w_to_dbm(bad)


def test_db_and_linear_round_trip():
    for db in (-20.0, 0.0, 3.0, 16.5):
        assert units.linear_to_db(units.db_to_linear(db)) == pytest.approx(db)
    with pytest.raises(ValueError):
        units.linear_to_db(0.0)


def test_ssmf_dispersion_to_beta2_matches_textbook_value():
    """D = 17 ps/(nm·km) at 1550 nm ↔ β₂ ≈ −21.7 ps²/km (Agrawal, NFO, §1.2.3)."""
    beta2 = units.dispersion_to_beta2(17.0)
    assert beta2 == pytest.approx(-2.1683e-26, rel=1e-4)
    assert units.beta2_to_dispersion(beta2) == pytest.approx(17.0, rel=1e-14)


def test_dispersion_slope_to_beta3_matches_the_derivative_of_beta2():
    """β₃ = dβ₂/dω: compare the closed form with a finite difference of D(λ) = D₀ + S(λ − λ₀)."""
    lam0, d0, s = 1550e-9, 17.0, 0.057
    beta3 = units.dispersion_slope_to_beta3(s, d0, lam0)
    c = units.SPEED_OF_LIGHT_M_PER_S

    def beta2_at(lam: float) -> float:
        return units.dispersion_to_beta2(d0 + s * (lam - lam0) * 1e9, lam)

    dlam = 1e-12
    dbeta2_dlam = (beta2_at(lam0 + dlam) - beta2_at(lam0 - dlam)) / (2 * dlam)
    domega_dlam = -2 * math.pi * c / lam0**2
    assert beta3 == pytest.approx(dbeta2_dlam / domega_dlam, rel=1e-6)
    assert beta3 == pytest.approx(1.28e-40, rel=0.01)  # ≈ 0.128 ps³/km


def test_alpha_conversion():
    assert units.alpha_db_per_km_to_per_m(0.2) == pytest.approx(0.2 * math.log(10) / 10 / 1e3)
    # 0.2 dB/km over 100 km = 20 dB = factor 100.
    assert math.exp(-units.alpha_db_per_km_to_per_m(0.2) * 100e3) == pytest.approx(0.01)


def test_wavelength_to_frequency():
    assert units.wavelength_to_frequency_hz(1550e-9) == pytest.approx(193.4145e12, rel=1e-6)
