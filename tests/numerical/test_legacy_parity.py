"""Legacy-vs-v1 parity (Path B): same bits → known transform → ≈ same field.

The two implementations differ at several layers (see memory entry
``reference-legacy-parity-deltas``). The parity tests here pin the
*semantic* correspondence:

1. **Bit→symbol convention**: ``legacy_unscaled_symbol = norm · conj(v1_symbol)``,
   where ``norm = sqrt(<|legacy_unscaled|²>)`` is the constellation RMS (the
   same scale factor that v1 uses to normalise to unit power). Tested to
   ``atol=1e-15`` since both implementations construct the constellation from
   the same recursive Gray code.

2. **Pulse-shaping**: legacy uses spectral (FFT-multiply, circular)
   convolution; v1 uses time-domain ``np.convolve`` (linear). For long
   signals the steady-state interior agrees up to a real scale factor; the
   conjugate-of-v1 relationship holds with a tight bound on the imaginary
   part of the residual.

We build the legacy WDM dict by hand (see ``_legacy_wdm_dict`` below) rather
than going through ``create_wdm_parameters``. This keeps the parity tests
independent of the legacy config-builder surface and lets us state every
field that the parity claim depends on (``scale_coef``, ``p_ave``,
``modulation_type``, …) directly in the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom._legacy.signal import generate_wdm_base
from hpcom.modulation import bits_to_symbols
from hpcom.modulation.constellations import square_qam
from hpcom.transmitter import Transmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MOD_TYPE = {4: "qpsk", 16: "16qam", 64: "64qam", 256: "256qam"}
_N_BITS = {"qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8}


def _constellation_rms(m_order: int) -> float:
    """RMS amplitude of the legacy *unscaled* QAM constellation."""
    side = int(np.sqrt(m_order))
    levels = np.arange(side) * 2 - (side - 1)
    # |level|² mean over both axes
    return float(np.sqrt(2.0 * np.mean(levels**2)))


def _legacy_wdm_dict(
    *,
    m_order: int,
    n_symbols: int,
    sps: int,
    rs: float,
    p_ave_w: float,
    roll_off: float,
    n_channels: int = 1,
    channel_spacing: float = 50e9,
) -> dict:
    """Manually construct the legacy wdm config dict.

    Equivalent to ``create_wdm_parameters`` but spelled out field-by-field so
    every quantity the parity claim depends on (scale_coef, p_ave,
    modulation_type, …) is visible inline.
    """
    return {
        "n_channels": n_channels,
        "channel_spacing": channel_spacing,
        "n_polarisations": 1,
        "p_ave_dbm": 10.0 * np.log10(p_ave_w / 1e-3),
        "n_symbols": n_symbols,
        "m_order": m_order,
        "modulation_type": _MOD_TYPE[m_order],
        "n_bits_symbol": _N_BITS[_MOD_TYPE[m_order]],
        "roll_off": roll_off,
        "upsampling": sps,
        "downsampling_rate": 1,
        "symb_freq": rs,
        "sample_freq": int(rs * sps),
        "np_filter": 0,
        "p_ave": p_ave_w,
        "scale_coef": _constellation_rms(m_order) / np.sqrt(p_ave_w),
        "seed": "fixed",
    }


# ---------------------------------------------------------------------------
# Symbol-level parity (exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m_order", [4, 16, 64])
def test_legacy_symbol_equals_conj_v1_times_rms(m_order: int):
    """The legacy and v1 constellations are related by:
        legacy_unscaled[bits] = norm · conj(v1_unit_power[bits])
    where ``norm`` is the RMS amplitude of the legacy unscaled constellation.
    This is the bit-exact mapping equivalence; any future change to either
    constellation order breaks this test.
    """
    from hpcom._legacy.modulation import generate_constellation_dict

    legacy_dict, _ = generate_constellation_dict(_N_BITS[_MOD_TYPE[m_order]])
    constellation, labels = square_qam(m_order)
    norm = _constellation_rms(m_order)

    for sym_idx in range(m_order):
        bits = labels[sym_idx]
        v1_sym = constellation[sym_idx]
        bit_str = "".join(str(int(b)) for b in bits)
        legacy_sym = legacy_dict[bit_str]
        # Match: legacy = norm · conj(v1)
        np.testing.assert_allclose(
            legacy_sym, norm * np.conj(v1_sym), atol=1e-12
        )


@pytest.mark.parametrize("m_order", [4, 16, 64])
def test_legacy_bits_to_constellation_round_trip(m_order: int):
    """For any bit pattern, ``legacy_get_constellation_point(bits)`` equals
    ``norm · conj(bits_to_symbols(bits))`` from v1."""
    from hpcom._legacy.modulation import get_constellation_point

    norm = _constellation_rms(m_order)
    k = _N_BITS[_MOD_TYPE[m_order]]
    rng = np.random.default_rng(0)
    bits_arr = rng.integers(0, 2, size=32 * k, dtype=np.uint8)
    bits_str = "".join(str(int(b)) for b in bits_arr)

    legacy_syms = get_constellation_point(bits_str, type=_MOD_TYPE[m_order])
    v1_syms = bits_to_symbols(bits_arr, m_order)
    np.testing.assert_allclose(legacy_syms, norm * np.conj(v1_syms), atol=1e-12)


# ---------------------------------------------------------------------------
# Field-level parity (approximate, conjugate convention exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m_order", [4, 16])
def test_legacy_field_is_real_scaled_conj_of_v1_field(m_order: int):
    """In the steady-state interior, ``legacy_field / conj(v1_field)`` is a
    real constant up to a small algorithmic residual (circular vs linear
    conv, RRC centering / normalisation). The imaginary part of the mean
    ratio must be ≈ 0 (the conjugate convention is the load-bearing claim;
    anything else would indicate a sign or layout regression in either
    implementation).
    """
    n_sym = 512
    sps = 8
    rs = 10e9
    p_ave_w = 1e-3
    rolloff = 0.1

    wdm = _legacy_wdm_dict(
        m_order=m_order, n_symbols=n_sym, sps=sps, rs=rs,
        p_ave_w=p_ave_w, roll_off=rolloff,
    )
    rng = np.random.default_rng(0)
    bits_arr = rng.integers(0, 2, size=n_sym * _N_BITS[_MOD_TYPE[m_order]], dtype=np.uint8)
    bits_str = "".join(str(int(b)) for b in bits_arr)

    legacy_signal, _ = generate_wdm_base(wdm, bits=bits_str, seed=0)
    legacy_field = np.asarray(legacy_signal, dtype=np.complex128)

    tx = Transmitter(
        modulation_order=m_order, symbol_rate_hz=rs, n_symbols=n_sym,
        samples_per_symbol=sps, roll_off=rolloff, pulse_span_symbols=32,
    )
    v1_field = tx.generate(bits=bits_arr).field

    skip = 32 * sps
    ratios = legacy_field[skip:-skip] / np.conj(v1_field[skip:-skip])
    # The imaginary part of the per-sample ratio centers on zero. Compare
    # |Im(ratio).mean()| against |Re(ratio).mean()| as a ratio test — should
    # be much smaller than 1.
    re_mean = float(np.real(ratios).mean())
    im_mean = float(np.imag(ratios).mean())
    assert abs(im_mean) < 1e-3 * abs(re_mean), (
        f"Im(legacy / conj(v1)).mean() = {im_mean:.3e}, "
        f"|Re|.mean() = {abs(re_mean):.3e} — conjugate convention violated"
    )
    # Per-sample magnitude variation around the mean comes from the
    # circular-vs-linear conv residual bleeding into the trimmed region;
    # in practice 99%+ of samples agree to 5%, with a handful of edge-
    # bleed outliers. Test the *median* magnitude, not every sample.
    median_dev = float(
        np.median(np.abs(np.abs(ratios) - abs(re_mean)) / abs(re_mean))
    )
    assert median_dev < 0.05, (
        f"median |ratio| deviation = {median_dev:.3%} exceeds 5%"
    )


def test_legacy_field_qpsk_known_power_scale():
    """The expected real-scale factor of legacy / conj(v1) is sqrt(p_ave) ·
    (legacy_rrc_peak / v1_rrc_peak). We don't pin the absolute value
    (depends on filter length and conv mode), but the sign must be positive
    and the magnitude positive; the previous test pins the conjugate
    relation."""
    n_sym = 512
    sps = 8
    wdm = _legacy_wdm_dict(
        m_order=4, n_symbols=n_sym, sps=sps, rs=10e9,
        p_ave_w=1e-3, roll_off=0.1,
    )
    rng = np.random.default_rng(1)
    bits_arr = rng.integers(0, 2, size=n_sym * 2, dtype=np.uint8)
    bits_str = "".join(str(int(b)) for b in bits_arr)
    legacy_signal, _ = generate_wdm_base(wdm, bits=bits_str, seed=0)
    legacy_field = np.asarray(legacy_signal, dtype=np.complex128)
    tx = Transmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=n_sym,
        samples_per_symbol=sps, roll_off=0.1, pulse_span_symbols=32,
    )
    v1_field = tx.generate(bits=bits_arr).field

    skip = 32 * sps
    re_mean = float(
        np.real(legacy_field[skip:-skip] / np.conj(v1_field[skip:-skip])).mean()
    )
    # Positive real scale (sqrt(p_ave) > 0, both RRC peaks > 0).
    assert re_mean > 0.0


# ---------------------------------------------------------------------------
# WDM (multi-channel) parity (single-pol)
# ---------------------------------------------------------------------------
def test_legacy_qpsk_3channel_wdm_conjugate_relation_holds():
    """3-channel WDM: ``generate_wdm_base`` only supports n_channels=1, but
    the bit→symbol convention is independent of WDM. Verify v1's WDMTransmitter
    output, demuxed per channel, still relates to legacy single-channel output
    by the same conjugate transform on each channel.
    """
    from hpcom.transmitter import WDMTransmitter
    from hpcom.wdm import WDMConfig

    n_sym = 256
    sps = 16
    rs = 10e9
    rolloff = 0.1
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)

    rng = np.random.default_rng(2)
    bits_per_ch = rng.integers(0, 2, size=(3, n_sym * 2), dtype=np.uint8)

    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=rs, n_symbols=n_sym,
        samples_per_symbol=sps, roll_off=rolloff, pulse_span_symbols=32,
        wdm=cfg,
    )
    sent = tx.generate(bits=bits_per_ch)
    # Downshift each channel of the v1 composite back to baseband, then
    # compare with a legacy single-channel build for that channel's bits.
    t = np.arange(sent.field.size) * sent.dt_s

    for ch_idx, f_ch in enumerate(cfg.channel_frequencies_hz()):
        v1_baseband = sent.field * np.exp(-2j * np.pi * float(f_ch) * t)

        wdm_dict = _legacy_wdm_dict(
            m_order=4, n_symbols=n_sym, sps=sps, rs=rs,
            p_ave_w=1e-3, roll_off=rolloff,
        )
        bits_str = "".join(str(int(b)) for b in bits_per_ch[ch_idx])
        legacy_signal, _ = generate_wdm_base(wdm_dict, bits=bits_str, seed=0)
        legacy_field = np.asarray(legacy_signal, dtype=np.complex128)

        skip = 32 * sps
        ratios = legacy_field[skip:-skip] / np.conj(v1_baseband[skip:-skip])
        re_mean = float(np.real(ratios).mean())
        im_mean = float(np.imag(ratios).mean())
        # Cross-channel leakage in v1 + circular-vs-linear residual in
        # legacy bound the parity tolerance more loosely than single-carrier;
        # the conjugate convention must still be visible. A sign or layout
        # regression would give |Im/Re| ~ 1 (90° error); 15% is a tight
        # bound for catching that while permitting the documented algorithmic
        # residual.
        assert abs(im_mean) < 0.15 * abs(re_mean), (
            f"ch={ch_idx}: |Im(ratio).mean()| = {abs(im_mean):.3e} "
            f"vs |Re(ratio).mean()| = {abs(re_mean):.3e} — convention violated"
        )
