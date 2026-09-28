"""ASE-limited SNR of an amplified link vs. the closed form.

For N identical spans whose EDFA gain exactly compensates the span loss, each
amplifier adds white ASE of PSD S = n_sp·hν·(G−1) per polarisation
(Essiambre et al., JLT 2010) that reaches the receiver with net unit gain, so
after the matched filter

    SNR = P_pol / (N · S · R_s).

The fibers are linear (γ = 0) and the receiver removes the dispersion, so the
ASE is the only impairment. The reference is computed here from the formula,
independently of hpcom's EDFA helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import matched_receiver_kwargs
from hpcom import (
    EDFA,
    AdditiveNoise,
    Fiber,
    Link,
    Simulation,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMReceiver,
    WDMTransmitter,
)

pytestmark = pytest.mark.slow

H, C = 6.62607015e-34, 299_792_458.0


def _theory_snr_db(p_pol_w: float, n_spans: int, gain_db: float, nf_db: float, rs: float) -> float:
    g, f = 10 ** (gain_db / 10), 10 ** (nf_db / 10)
    s = (g * f - 1) / (2 * (g - 1)) * H * (C / 1550e-9) * (g - 1)
    return float(10 * np.log10(p_pol_w / (n_spans * s * rs)))


def _link(n_spans: int, polarization: str) -> Link:
    fiber = Fiber.ssmf(80.0, backend="numpy-linear", polarization=polarization)
    return Link.from_spans(n_spans, fiber, EDFA(gain_db=16.0, noise_figure_db=5.0))


@pytest.mark.parametrize("polarization", ["single", "dual"])
@pytest.mark.parametrize(("n_spans", "power_dbm"), [(1, -10.0), (5, -3.0), (20, 0.0)])
def test_single_carrier_snr_matches_closed_form(n_spans, power_dbm, polarization):
    link = _link(n_spans, polarization)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 15,
        samples_per_symbol=4,
        polarization=polarization,
        launch_power_per_channel_dbm=power_dbm,
    )
    rx = SingleCarrierReceiver(
        **matched_receiver_kwargs(
            tx, equalize="data_aided", cd_compensation_s2=link.total_dispersion_s2
        )
    )
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=21).run()
    expected = _theory_snr_db(tx.launch_power_per_polarization_w, n_spans, 16.0, 5.0, 32e9)
    for stream in result.stream_metrics:
        assert stream.snr_db == pytest.approx(expected, abs=0.1)


def test_every_wdm_channel_sees_the_same_ase_limited_snr():
    link = _link(4, "dual")
    wdm = WDMConfig(n_channels=5, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 13,
        samples_per_symbol=8,
        wdm=wdm,
        polarization="dual",
        launch_power_per_channel_dbm=-2.0,
    )
    rx = WDMReceiver(
        wdm=wdm,
        **matched_receiver_kwargs(
            tx, equalize="data_aided", cd_compensation_s2=link.total_dispersion_s2
        ),
    )
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=4).run()
    expected = _theory_snr_db(tx.launch_power_per_polarization_w, 4, 16.0, 5.0, 32e9)
    snrs = np.array([m.snr_db for m in result.stream_metrics])
    assert snrs.shape == (10,)
    np.testing.assert_allclose(snrs, expected, atol=0.2)


def test_osnr_noise_loading_gives_snr_equal_osnr_times_bref_over_rs():
    """Dual polarisation: SNR = OSNR·B_ref/R_s (Essiambre et al. 2010)."""
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 15,
        samples_per_symbol=2,
        polarization="dual",
    )
    noise = AdditiveNoise.for_osnr(18.0, channel_power_w=tx.launch_power_per_channel_w)
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx, equalize="data_aided"))
    result = Simulation(transmitter=tx, link=Link(elements=(noise,)), receiver=rx, seed=2).run()
    assert result.metrics.snr_db == pytest.approx(18.0 + 10 * np.log10(12.5e9 / 32e9), abs=0.05)
