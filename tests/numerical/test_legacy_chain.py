"""End-to-end parity with pre-v1 hpcom's ``full_line_model_wdm`` (master).

Same bits → transmitter → nonlinear Manakov link (noiseless amplifiers) →
CD compensation → matched filter → data-aided phase/gain (pre-v1
``nonlinear_shift``). Pre-v1 and v1 agree

* exactly on the transmitted symbols (default labelling ``q_descending``);
* to ≈ 6e-6 RMS on the equalised received symbols, at every step size — the
  residual is the pre-v1 window-truncated RRC (``test_legacy_parity.py``);
* in how they converge when the step is refined: both change by the same
  amount, so they are the same numerical method, not two methods that happen
  to agree at one setting.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

for _dependency in ("tensorflow", "sklearn", "ssfm_gpu"):  # pip install "hpcom[legacy]"
    pytest.importorskip(_dependency)

from hpcom import EDFA, Fiber, Link, WDMConfig, WDMTransmitter  # noqa: E402
from hpcom._legacy.channel import (  # noqa: E402
    create_channel_parameters,
    full_line_model_wdm,
)

pytestmark = pytest.mark.legacy

M, NSYM, SPS, RS, BETA, DF, NCH, PDBM, NSPAN = 16, 1 << 10, 8, 32e9, 0.1, 50e9, 3, 6.0, 2
P_AVE = 1e-3 * 10 ** (PDBM / 10)
P_POL = P_AVE / 2


def _legacy_wdm() -> dict:
    levels = np.arange(4) * 2 - 3
    rms = float(np.sqrt(2 * np.mean(levels**2)))
    return {
        "n_channels": NCH,
        "channel_spacing": DF,
        "n_polarisations": 2,
        "p_ave_dbm": PDBM,
        "n_symbols": NSYM,
        "m_order": M,
        "modulation_type": "16qam",
        "n_bits_symbol": 4,
        "roll_off": BETA,
        "upsampling": SPS,
        "downsampling_rate": 1,
        "symb_freq": RS,
        "sample_freq": int(RS * SPS),
        "np_filter": 0,
        "p_ave": P_AVE,
        "scale_coef": rms / np.sqrt(P_POL),
        "seed": "fixed",
    }


@pytest.fixture(scope="module")
def chains():
    """dz (km) → (pre-v1 received, v1 received, pre-v1 sent, v1 sent), symbols in unit power."""
    bits = np.random.default_rng(0).integers(0, 2, (2, NCH, NSYM * 4), dtype=np.uint8)
    legacy_bits = tuple(["".join(map(str, bits[p, k])) for k in range(NCH)] for p in range(2))
    tx = WDMTransmitter(
        modulation_order=M,
        symbol_rate_hz=RS,
        n_symbols=NSYM,
        samples_per_symbol=SPS,
        roll_off=BETA,
        wdm=WDMConfig(n_channels=NCH, spacing_hz=DF),
        polarization="dual",
        launch_power_per_channel_dbm=PDBM,
    )
    sent = tx.generate(bits=bits)
    out = {}
    for dz in (2.0, 1.0, 0.5):
        channel = create_channel_parameters(
            n_spans=NSPAN,
            z_span=80,
            alpha_db=0.2,
            gamma=1.27,
            noise_figure_db=-200,
            dispersion_parameter=16.8,
            dz=dz,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = full_line_model_wdm(channel, _legacy_wdm(), bits=legacy_bits)
        fiber = Fiber(
            length_km=80,
            alpha_db_per_km=0.2,
            beta2_s2_per_m=channel["beta2"] * 1e-3,  # pre-v1 stores β₂ in s²/km
            gamma_per_w_per_km=1.27,
            n_steps=round(80 / dz),
            polarization="dual",
        )
        amp = EDFA(gain_db=10 * np.log10(channel["gain"]), noise_figure_db=None)
        link = Link.from_spans(NSPAN, fiber, amp)
        rx = tx.matched_receiver(equalize="data_aided", cd_compensation_s2=link.total_dispersion_s2)
        received = rx.receive(link.propagate(sent.field, sent.dt_s), sent.dt_s, reference=sent)
        out[dz] = (
            _stack(legacy, "shifted"),
            np.asarray(received.symbols),
            _stack(legacy, "orig"),
            sent.symbols,
        )
    return out


def _stack(legacy: dict, key: str) -> np.ndarray:
    """Pre-v1 per-channel point lists → (pol, channel, n) in unit symbol power."""
    points = [[np.asarray(legacy[f"points_{p}_{key}"][k]) for k in range(NCH)] for p in "xy"]
    return np.array(points) / np.sqrt(P_POL)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


@pytest.mark.parametrize("dz", [2.0, 1.0, 0.5])
def test_same_bits_give_the_same_received_symbols(chains, dz):
    legacy_rx, v1_rx, legacy_tx, v1_tx = chains[dz]
    np.testing.assert_allclose(legacy_tx, v1_tx, rtol=0, atol=1e-14)
    assert _rms(legacy_rx, v1_rx) < 2e-5
    # The comparison is not trivial: nonlinearity distorts the symbols by ~22 dB SNR.
    snr = 10 * np.log10(1 / np.mean(np.abs(v1_rx - v1_tx) ** 2))
    assert 18 < snr < 26


def test_both_converge_identically_as_the_step_is_refined(chains):
    for coarse, fine in itertools.pairwise([2.0, 1.0, 0.5]):
        change_legacy = _rms(chains[coarse][0], chains[fine][0])
        change_v1 = _rms(chains[coarse][1], chains[fine][1])
        assert change_v1 > 1e-4  # the step error is real and resolved…
        assert abs(change_legacy - change_v1) < 0.02 * change_v1  # …and the same in both
    assert _rms(chains[1.0][1], chains[0.5][1]) < _rms(chains[2.0][1], chains[1.0][1]) / 4
