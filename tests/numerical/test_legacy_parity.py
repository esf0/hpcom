"""Parity with the pre-v1 transmitter (``hpcom._legacy.signal.generate_wdm``).

Same bits in → the same WDM field out, for 1/3/5 channels and 1/2
polarisations, up to one *known* transform:

    legacy_field = c · conj(v1_field),   c = 1 (power conventions agree),

with WDM channel ``i`` of legacy carrying the bits of v1 channel ``N−1−i``
(conjugation mirrors the spectrum). The conjugation comes from the Q-axis
sign convention of the legacy constellation (legacy point = RMS·conj(v1
point), checked exactly below).

The only other difference is the pulse: legacy samples the RRC impulse
response over the signal window (truncation error ≈ 4e-4 of the spectrum);
v1's default is the exact band-limited RRC. With legacy's filter injected
into v1's pipeline the two fields agree to rounding error, which proves that
mapping, conjugation, power scaling, polarisation split, carrier placement
and summation are identical.

Legacy's ``get_constellation`` is broken under NumPy ≥ 1.25, so the legacy
config dict is built by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

for _dependency in ("tensorflow", "sklearn"):  # pre-v1 code: pip install "hpcom[legacy]"
    pytest.importorskip(_dependency)

import hpcom._pulse as pulse  # noqa: E402
from hpcom import SingleCarrierTransmitter, WDMConfig, WDMReceiver, WDMTransmitter  # noqa: E402
from hpcom._legacy.modulation import (  # noqa: E402
    generate_constellation_dict,
    get_constellation_point,
)
from hpcom._legacy.signal import generate_wdm, rrcosfilter  # noqa: E402
from hpcom.modulation import bits_to_symbols, square_qam  # noqa: E402

pytestmark = pytest.mark.legacy

_TYPE = {4: "qpsk", 16: "16qam", 64: "64qam"}
_K = {4: 2, 16: 4, 64: 6}
RS, SPS, BETA, SPACING, NSYM = 10e9, 32, 0.1, 50e9, 512


def _rms_unscaled(m: int) -> float:
    side = int(np.sqrt(m))
    levels = np.arange(side) * 2 - (side - 1)
    return float(np.sqrt(2 * np.mean(levels**2)))


def _legacy_dict(m: int, n_channels: int, n_pol: int, p_ave_w: float) -> dict:
    return {
        "n_channels": n_channels,
        "channel_spacing": SPACING,
        "n_polarisations": n_pol,
        "p_ave_dbm": 10 * np.log10(p_ave_w / 1e-3),
        "n_symbols": NSYM,
        "m_order": m,
        "modulation_type": _TYPE[m],
        "n_bits_symbol": _K[m],
        "roll_off": BETA,
        "upsampling": SPS,
        "downsampling_rate": 1,
        "symb_freq": RS,
        "sample_freq": int(RS * SPS),
        "np_filter": 0,
        "p_ave": p_ave_w,
        "scale_coef": _rms_unscaled(m) / np.sqrt(p_ave_w / n_pol),
        "seed": "fixed",
    }


def _both_fields(m: int, n_channels: int, n_pol: int, seed: int):
    """(legacy field, v1 field, v1 signal, bits); fields as (n_pol, n) arrays."""
    bits = np.random.default_rng(seed).integers(
        0, 2, (n_pol, n_channels, NSYM * _K[m]), dtype=np.uint8
    )
    pol = "dual" if n_pol == 2 else "single"
    common = dict(
        modulation_order=m,
        symbol_rate_hz=RS,
        n_symbols=NSYM,
        samples_per_symbol=SPS,
        roll_off=BETA,
        polarization=pol,
        launch_power_per_channel_dbm=0.0,
    )
    if n_channels == 1:
        sent = SingleCarrierTransmitter(**common).generate(
            bits=bits[:, 0] if n_pol == 2 else bits[0, 0]
        )
    else:
        sent = WDMTransmitter(
            **common, wdm=WDMConfig(n_channels=n_channels, spacing_hz=SPACING)
        ).generate(bits=bits if n_pol == 2 else bits[0])
    legacy_bits = tuple(
        ["".join(map(str, bits[p, n_channels - 1 - i])) for i in range(n_channels)]
        for p in range(n_pol)
    )
    out = generate_wdm(_legacy_dict(m, n_channels, n_pol, 1e-3), bits=legacy_bits)
    legacy = (
        np.stack([np.asarray(o) for o in out[:n_pol]]) if n_pol == 2 else np.asarray(out[0])[None]
    )
    return legacy, np.atleast_2d(sent.field), sent, bits


def _fit(legacy: np.ndarray, v1: np.ndarray) -> tuple[complex, float]:
    """LS scale c of legacy ≈ c·conj(v1) and the max residual relative to the legacy RMS."""
    target = np.conj(v1).ravel()
    c = np.vdot(target, legacy.ravel()) / np.vdot(target, target)
    resid = np.max(np.abs(legacy.ravel() - c * target)) / np.sqrt(np.mean(np.abs(legacy) ** 2))
    return complex(c), float(resid)


# ---------------------------------------------------------------------------
# Symbol mapping (exact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [4, 16, 64])
def test_every_legacy_point_is_rms_times_conj_of_the_v1_point(m):
    legacy_dict, _ = generate_constellation_dict(_K[m])
    points, labels = square_qam(m)
    for i in range(m):
        key = "".join(str(int(b)) for b in labels[i])
        np.testing.assert_allclose(
            legacy_dict[key], _rms_unscaled(m) * np.conj(points[i]), atol=1e-12
        )


@pytest.mark.parametrize("m", [4, 16, 64])
def test_legacy_bit_mapping_matches_v1_for_random_bits(m):
    bits = np.random.default_rng(0).integers(0, 2, 64 * _K[m], dtype=np.uint8)
    legacy = get_constellation_point("".join(map(str, bits)), type=_TYPE[m])
    np.testing.assert_allclose(
        legacy, _rms_unscaled(m) * np.conj(bits_to_symbols(bits, m)), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Field parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("n_channels", [1, 3, 5])
def test_same_bits_give_the_same_field_up_to_the_pulse_truncation(n_channels, n_pol):
    legacy, v1, _, _ = _both_fields(16, n_channels, n_pol, seed=10 * n_channels + n_pol)
    c, resid = _fit(legacy, v1)
    assert abs(c - 1.0) < 1e-3, f"power conventions differ: c = {c}"
    assert resid < 5e-4, f"max residual {resid:.2e} of the RMS (legacy RRC truncation ≈ 3e-4)"


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("n_channels", [1, 3, 5])
def test_with_the_legacy_filter_the_fields_agree_to_rounding(n_channels, n_pol, monkeypatch):
    def legacy_responses(n, sps, roll_off, span):
        h = np.fft.fft(np.fft.ifftshift(rrcosfilter(n, roll_off, 1 / RS, RS * sps)))
        h *= np.sqrt(sps / np.mean(np.abs(h) ** 2))  # v1's transmit normalisation
        return h, np.conj(h) / sps

    monkeypatch.setattr(pulse, "_circular_responses", legacy_responses)
    legacy, v1, _, _ = _both_fields(16, n_channels, n_pol, seed=10 * n_channels + n_pol)
    c, resid = _fit(legacy, v1)
    assert abs(c.imag) < 1e-12 * abs(c) and c.real > 0
    # Rounding only: WDM carrier phases 2πf·t reach ~1e4 rad (cf. 3e-4 with v1's own filter).
    assert resid < 1e-10, f"max residual {resid:.2e}"


@pytest.mark.parametrize("m", [4, 16, 64])
def test_v1_receiver_decodes_legacy_generated_wdm_signals(m):
    legacy, _, sent, bits = _both_fields(m, 3, 2, seed=m)
    rx = WDMReceiver(
        modulation_order=m,
        samples_per_symbol=SPS,
        roll_off=BETA,
        polarization="dual",
        wdm=WDMConfig(n_channels=3, spacing_hz=SPACING),
    )
    received = rx.receive(np.conj(legacy), sent.dt_s)
    np.testing.assert_array_equal(received.bits, bits)
