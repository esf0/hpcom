"""WDM grid configuration and validation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import SingleCarrierTransmitter, WDMConfig, WDMReceiver, WDMTransmitter


def test_config_rejects_fewer_than_two_channels():
    with pytest.raises(ValidationError, match="n_channels"):
        WDMConfig(n_channels=1, spacing_hz=50e9)


def test_config_rejects_non_positive_spacing():
    with pytest.raises(ValidationError, match="spacing_hz"):
        WDMConfig(n_channels=3, spacing_hz=0.0)


@pytest.mark.parametrize(
    ("n", "spacing", "expected"),
    [
        (3, 50e9, [-50e9, 0.0, 50e9]),
        (4, 25e9, [-37.5e9, -12.5e9, 12.5e9, 37.5e9]),
        (2, 100e9, [-50e9, 50e9]),
    ],
)
def test_channel_frequencies_are_centred_on_the_carrier(n, spacing, expected):
    cfg = WDMConfig(n_channels=n, spacing_hz=spacing)
    np.testing.assert_allclose(cfg.channel_frequencies_hz(), expected)
    assert cfg.total_bandwidth_hz == pytest.approx((n - 1) * spacing)


def test_transmitter_rejects_a_sample_rate_that_cannot_hold_the_grid():
    # 5 × 50 GHz: band edge ±105.5 GHz, but R_s·sps = 160 GHz → need ≥ 211 GHz.
    with pytest.raises(ValidationError, match=r"cannot hold 5 channels.*at least 22"):
        WDMTransmitter(
            modulation_order=4,
            symbol_rate_hz=10e9,
            n_symbols=64,
            samples_per_symbol=16,
            wdm=WDMConfig(n_channels=5, spacing_hz=50e9),
        )


@pytest.mark.parametrize(
    ("n_channels", "n_symbols", "ok"),
    [(3, 1024, True), (3, 1000, False), (4, 1024, True), (4, 1008, False)],
)
def test_circular_signals_need_carriers_on_the_fft_grid(n_channels, n_symbols, ok):
    """Δf·n_symbols/R_s (/2 for even N) must be an integer for a periodic signal."""
    kw = dict(
        modulation_order=4,
        symbol_rate_hz=32e9,
        n_symbols=n_symbols,
        samples_per_symbol=8,
        wdm=WDMConfig(n_channels=n_channels, spacing_hz=50e9),
    )
    if ok:
        WDMTransmitter(**kw)
    else:
        with pytest.raises(ValidationError, match=r"not periodic.*multiple of"):
            WDMTransmitter(**kw)
    # Linear convolution does not require periodicity.
    WDMTransmitter(**kw, pulse_span_symbols=16, convolution="linear")


def test_wdm_transmitter_is_a_sibling_not_a_subclass_of_single_carrier():
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=32e9,
        n_symbols=64,
        samples_per_symbol=8,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    assert not isinstance(tx, SingleCarrierTransmitter)
    assert tx.n_channels == 3


def test_receiver_channel_selection_validation():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    assert WDMReceiver(modulation_order=4, wdm=cfg).selected_channels == (0, 1, 2)
    assert WDMReceiver(modulation_order=4, wdm=cfg, channels=(2, 0)).selected_channels == (2, 0)
    with pytest.raises(ValidationError, match=r"outside \[0, 3\)"):
        WDMReceiver(modulation_order=4, wdm=cfg, channels=(3,))
    with pytest.raises(ValidationError, match="unique"):
        WDMReceiver(modulation_order=4, wdm=cfg, channels=(1, 1))
    with pytest.raises(ValidationError, match="non-empty"):
        WDMReceiver(modulation_order=4, wdm=cfg, channels=())


def test_wdm_config_is_required():
    with pytest.raises(ValidationError, match="wdm"):
        WDMTransmitter(modulation_order=4, symbol_rate_hz=32e9, n_symbols=64)
    with pytest.raises(ValidationError, match="wdm"):
        WDMReceiver(modulation_order=4)


def test_compatibility_issues_name_every_mismatch():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=16, symbol_rate_hz=32e9, n_symbols=64, samples_per_symbol=8, wdm=cfg
    )
    rx = WDMReceiver(
        modulation_order=4, samples_per_symbol=4, wdm=WDMConfig(n_channels=3, spacing_hz=75e9)
    )
    issues = "\n".join(rx.compatibility_issues(tx))
    for word in ("constellations differ", "samples_per_symbol", "wdm"):
        assert word in issues
    sc = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=32e9, n_symbols=64)
    assert rx.compatibility_issues(sc) == ["WDMReceiver cannot receive a SingleCarrierTransmitter"]
