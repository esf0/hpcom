"""The two square-QAM bit labellings are statistically equivalent — exactly.

``q_ascending`` counts the Q rows of the Gray code from the bottom,
``q_descending`` (default, pre-v1 hpcom) from the top. For a reflected Gray
code, reversing the row order flips the leading bit of the row label, so

    map_ascending(b) = map_descending(b ⊕ m),   m = leading Q bit of every symbol.

``b ↦ b ⊕ m`` is a bijection of the bit sequences that preserves Hamming
distances. Hence, for *any* channel (nonlinear fibre included), labelling
``q_ascending`` with bits ``b`` transmits the same field and scores the same
BER, SER, EVM, SNR, MI and GMI as ``q_descending`` with bits ``b ⊕ m`` — and
with uniformly random bits the two metric distributions are identical.

Tested here:

1. the bit-flip identity for every supported order;
2. identical fields and metrics for mapped bits on a noisy nonlinear WDM link;
3. with independent random data, the metric differences shrink as the block
   grows and stay within their statistical uncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom import (
    EDFA,
    ConstellationConfig,
    Fiber,
    Link,
    Simulation,
    WDMConfig,
    WDMTransmitter,
)
from hpcom.metrics import bit_error_rate, information_rates, snr_db
from hpcom.modulation import bits_per_symbol, bits_to_symbols
from hpcom.modulation.constellations import SUPPORTED_QAM_ORDERS

LABELINGS = ("q_descending", "q_ascending")


def _leading_q_bit_mask(order: int, n_symbols: int) -> np.ndarray:
    k = bits_per_symbol(order)
    mask = np.zeros((n_symbols, k), dtype=np.uint8)
    mask[:, 0] = 1
    return mask.reshape(-1)


@pytest.mark.parametrize("order", SUPPORTED_QAM_ORDERS)
def test_q_ascending_is_q_descending_with_the_leading_q_bit_flipped(order):
    n = 256
    bits = np.random.default_rng(order).integers(0, 2, n * bits_per_symbol(order), dtype=np.uint8)
    np.testing.assert_array_equal(
        bits_to_symbols(bits, order, "q_ascending"),
        bits_to_symbols(bits ^ _leading_q_bit_mask(order, n), order, "q_descending"),
    )


def _nonlinear_wdm(labeling: str, n_symbols: int, seed: int = 5) -> WDMTransmitter:
    return WDMTransmitter(
        constellation=ConstellationConfig.from_square_qam(64, labeling),
        symbol_rate_hz=32e9,
        n_symbols=n_symbols,
        samples_per_symbol=6,
        roll_off=0.1,
        polarization="dual",
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        launch_power_per_channel_dbm=4.0,
        seed=seed,
    )


def _link() -> Link:
    fiber = Fiber.ssmf(length_km=80, n_steps=50, polarization="dual")
    return Link.from_spans(3, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))


def test_mapped_bits_give_identical_fields_and_metrics_on_a_noisy_nonlinear_link():
    n, order = 2048, 64
    link = _link()
    tx = {lab: _nonlinear_wdm(lab, n) for lab in LABELINGS}
    bits_asc = np.random.default_rng(0).integers(0, 2, (2, 3, n * 6), dtype=np.uint8)
    bits = {"q_ascending": bits_asc, "q_descending": bits_asc ^ _leading_q_bit_mask(order, n)}
    sent = {lab: tx[lab].generate(bits=bits[lab]) for lab in LABELINGS}
    np.testing.assert_array_equal(sent["q_ascending"].field, sent["q_descending"].field)

    metrics = {}
    for lab in LABELINGS:
        out = link.propagate(sent[lab].field, sent[lab].dt_s, rng=np.random.default_rng(1))
        rx = tx[lab].matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
        received = rx.receive(out, sent[lab].dt_s, reference=sent[lab])
        cfg = rx.resolved_constellation
        y, x = np.asarray(received.symbols).ravel(), np.asarray(sent[lab].symbols).ravel()
        ber = bit_error_rate(bits[lab], received.bits)
        mi, gmi = information_rates(y, cfg.nearest(x), cfg.points, cfg.bit_labels, cfg.pmf)
        metrics[lab] = (ber, snr_db(y, x), mi, gmi, received.decisions)
    a, b = metrics["q_descending"], metrics["q_ascending"]
    assert a[0] == b[0] and a[0][1] > 0  # identical (non-zero) bit-error counts
    assert a[1] == b[1]
    np.testing.assert_allclose(a[2:4], b[2:4], rtol=1e-12)
    np.testing.assert_array_equal(a[4], b[4])


@pytest.mark.parametrize("n_symbols", [2**10, 2**12, 2**14])
def test_independent_data_metrics_agree_within_their_uncertainty(n_symbols):
    """Same seeds → independent data per labelling; differences shrink ∝ 1/√N."""
    link = _link()
    result = {}
    for lab in LABELINGS:
        tx = _nonlinear_wdm(lab, n_symbols)
        rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
        result[lab] = Simulation(transmitter=tx, link=link, receiver=rx, seed=11).run().metrics
    a, b = result["q_descending"], result["q_ascending"]
    n_samples = 6 * n_symbols  # 2 polarisations × 3 channels
    # Two independent estimates: σ(ΔBER) ≈ √(2·BER/N_bits) (Gray: ≈ 1 bit error per
    # symbol error); σ(ΔSNR_dB) ≈ 10/ln10 · √(2/N) for the noise-power estimates.
    assert abs(a.ber - b.ber) < 4 * np.sqrt(2 * a.ber / a.n_bits)
    assert abs(a.snr_db - b.snr_db) < 4 * 10 / np.log(10) * np.sqrt(2 / n_samples)
    assert abs(a.gmi_bits - b.gmi_bits) < 4 * np.sqrt(2 / n_samples)
    assert abs(a.mi_bits - b.mi_bits) < 4 * np.sqrt(2 / n_samples)
