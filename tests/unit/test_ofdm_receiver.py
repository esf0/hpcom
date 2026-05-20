"""T12.3: OFDMReceiver — noiseless B2B, pilot equalization, shape checks.

Verifies that ``OFDMReceiver`` correctly demodulates OFDM-transmitted fields:
CP removal, FFT, power de-normalisation, data extraction, optional one-tap
pilot-based channel equalisation, and hard-decision demap.

All noiseless B2B tests assert BER == 0: any non-zero BER indicates a
normalisation or indexing bug in the Tx → Rx round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import (
    ConstantCompositionDM,
    ConstellationConfig,
    maxwell_boltzmann,
    square_qam,
)
from hpcom.ofdm import OFDMConfig, OFDMReceivedSymbols, OFDMReceiver, OFDMTransmitter
from hpcom.pilots import PilotConfig
from hpcom.receiver import Receiver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tx_rx(
    *,
    n_subcarriers: int = 64,
    cp_len: int = 16,
    n_guard: int = 4,
    pilot_indices: tuple[int, ...] = (),
    n_frames: int = 32,
    modulation_order: int = 4,
    polarization: str = "single",
    seed: int = 0,
    constellation: ConstellationConfig | None = None,
    dm: ConstantCompositionDM | None = None,
) -> tuple[OFDMTransmitter, OFDMReceiver]:
    ofdm = OFDMConfig(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        n_guard=n_guard,
        pilot_subcarrier_indices=pilot_indices,
    )
    tx_kwargs: dict = dict(
        symbol_rate_hz=32e9,
        n_symbols=n_frames,
        ofdm=ofdm,
        polarization=polarization,
        seed=seed,
    )
    rx_kwargs: dict = dict(
        ofdm=ofdm,
        polarization=polarization,
    )
    if constellation is not None:
        tx_kwargs["constellation"] = constellation
        rx_kwargs["constellation"] = constellation
    else:
        tx_kwargs["modulation_order"] = modulation_order
        rx_kwargs["modulation_order"] = modulation_order
    if dm is not None:
        tx_kwargs["dm"] = dm
        rx_kwargs["dm"] = dm

    tx = OFDMTransmitter(**tx_kwargs)
    rx = OFDMReceiver(**rx_kwargs)
    return tx, rx


def _ber(sent_bits: np.ndarray, recv_bits: np.ndarray) -> float:
    """Bit error rate between two bit arrays."""
    return float(np.mean(sent_bits != recv_bits))


# ---------------------------------------------------------------------------
# Inheritance and construction
# ---------------------------------------------------------------------------


def test_ofdm_receiver_is_a_receiver():
    """Polymorphism: an OFDMReceiver satisfies the Receiver API."""
    _, rx = _make_tx_rx()
    assert isinstance(rx, Receiver)


def test_ofdm_received_symbols_carries_ofdm_config():
    tx, rx = _make_tx_rx()
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert isinstance(rcv, OFDMReceivedSymbols)
    assert rcv.ofdm == tx.ofdm


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_base_pilots():
    """Base-class PilotConfig is for time-domain SC pilots; OFDM uses
    frequency-domain pilots via OFDMConfig instead."""
    ofdm = OFDMConfig(n_subcarriers=64)
    with pytest.raises(ValueError, match="OFDMReceiver does not accept"):
        OFDMReceiver(
            modulation_order=4,
            ofdm=ofdm,
            pilots=PilotConfig(symbol=1.0 + 0j, period=8),
        )


def test_rejects_wrong_field_length():
    """Field whose length isn't a multiple of samples_per_ofdm_symbol."""
    _, rx = _make_tx_rx(n_subcarriers=64, cp_len=16, n_frames=4)
    bad_field = np.zeros(100, dtype=np.complex128)  # not a multiple of 80
    with pytest.raises(ValueError, match="not a multiple"):
        rx.receive(bad_field)


def test_rejects_wrong_pol_shape():
    """Single-pol receiver rejects 2-D field; dual-pol rejects 1-D."""
    _, rx_single = _make_tx_rx(polarization="single", n_frames=4)
    tx, _ = _make_tx_rx(polarization="single", n_frames=4)
    sent = tx.generate()
    with pytest.raises(ValueError, match="1-D field"):
        rx_single.receive(np.stack([sent.field, sent.field]))

    _, rx_dual = _make_tx_rx(polarization="dual", n_frames=4)
    with pytest.raises(ValueError, match="leading axis 2"):
        rx_dual.receive(sent.field)


# ---------------------------------------------------------------------------
# Noiseless B2B: BER == 0 (single-pol)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_order", [4, 16, 64])
def test_noiseless_b2b_single_pol(mod_order: int):
    """Noiseless back-to-back: IFFT→CP→field→stripCP→FFT→demap recovers
    all transmitted bits for QPSK, 16-QAM, 64-QAM."""
    tx, rx = _make_tx_rx(
        modulation_order=mod_order,
        n_subcarriers=64,
        cp_len=16,
        n_guard=4,
        n_frames=16,
    )
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits, rcv.bits) == 0.0


def test_noiseless_b2b_no_cp():
    """BER == 0 with cp_len=0."""
    tx, rx = _make_tx_rx(cp_len=0, n_guard=4, n_frames=16)
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits, rcv.bits) == 0.0


def test_noiseless_b2b_no_guard():
    """BER == 0 with n_guard=0 (all subcarriers are data or pilot)."""
    tx, rx = _make_tx_rx(n_guard=0, n_frames=16)
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits, rcv.bits) == 0.0


# ---------------------------------------------------------------------------
# Noiseless B2B: dual-pol
# ---------------------------------------------------------------------------


def test_noiseless_b2b_dual_pol():
    """Dual-pol noiseless B2B: BER == 0 on both polarisations."""
    tx, rx = _make_tx_rx(
        polarization="dual",
        modulation_order=16,
        n_frames=8,
    )
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits[0], rcv.bits[0]) == 0.0
    assert _ber(sent.bits[1], rcv.bits[1]) == 0.0


# ---------------------------------------------------------------------------
# Noiseless B2B: with frequency-domain pilots
# ---------------------------------------------------------------------------


def test_noiseless_b2b_with_pilots():
    """Pilots in OFDMConfig; noiseless B2B should still recover BER == 0.
    (One-tap eq on a noiseless identity channel is a no-op.)"""
    tx, rx = _make_tx_rx(
        n_subcarriers=64,
        cp_len=16,
        n_guard=4,
        pilot_indices=(5, 11, 25, 50),
        n_frames=16,
        modulation_order=16,
    )
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits, rcv.bits) == 0.0


# ---------------------------------------------------------------------------
# Noiseless B2B: with DM (probabilistic shaping)
# ---------------------------------------------------------------------------


def test_noiseless_b2b_with_dm():
    """Probabilistic shaping via DM: noiseless B2B recovers shaped bits."""
    points, _ = square_qam(16)
    pmf = maxwell_boltzmann(points, nu=0.5)
    cfg = ConstellationConfig.from_points(points, probabilities=pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    # N=64, n_guard=4 → n_data=56. n_frames=4 → total=224 = 7*32 ✓
    tx, rx = _make_tx_rx(
        n_subcarriers=64,
        n_guard=4,
        n_frames=4,
        constellation=cfg,
        dm=dm,
    )
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits, rcv.bits) == 0.0


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


def test_single_pol_output_shapes():
    tx, rx = _make_tx_rx(
        n_subcarriers=64, n_guard=4, n_frames=10, modulation_order=16,
    )
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    n_data = tx.ofdm.n_data  # 56
    assert rcv.symbols.shape == (10 * n_data,)
    assert rcv.bits.shape == (10 * n_data * 4,)  # 16-QAM → 4 bits/symbol
    assert rcv.decisions.shape == (10 * n_data,)


def test_dual_pol_output_shapes():
    tx, rx = _make_tx_rx(
        n_subcarriers=64, n_guard=4, n_frames=8,
        modulation_order=16, polarization="dual",
    )
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    n_data = tx.ofdm.n_data
    assert rcv.symbols.shape == (2, 8 * n_data)
    assert rcv.bits.shape == (2, 8 * n_data * 4)
    assert rcv.decisions.shape == (2, 8 * n_data)
    assert rcv.polarization == "dual"


# ---------------------------------------------------------------------------
# One-tap pilot equalization under known channel
# ---------------------------------------------------------------------------


def test_pilot_eq_recovers_flat_fading_channel():
    """Apply a known per-subcarrier flat-fading channel (constant amplitude,
    linear phase across subcarriers) to the Tx field. With pilots, the
    one-tap equalizer should recover BER == 0. Without pilots, BER > 0."""
    n_sub = 64
    cp_len = 16
    n_guard = 4
    n_frames = 32
    mod = 4  # QPSK — robust to small estimation errors

    # Place pilots at well-spaced positions for good interpolation.
    pilot_idx = (2, 10, 20, 26, 40, 50, 60)

    ofdm_with_pilots = OFDMConfig(
        n_subcarriers=n_sub, cp_len=cp_len, n_guard=n_guard,
        pilot_subcarrier_indices=pilot_idx,
    )
    ofdm_no_pilots = OFDMConfig(
        n_subcarriers=n_sub, cp_len=cp_len, n_guard=n_guard,
    )

    # Build Tx with pilots and generate.
    tx_p = OFDMTransmitter(
        modulation_order=mod, symbol_rate_hz=32e9, n_symbols=n_frames,
        ofdm=ofdm_with_pilots, seed=42,
    )
    sent_p = tx_p.generate()

    # Apply a flat-fading channel: H(k) = A * exp(j*θ*k) per subcarrier.
    # This simulates a simple delay + attenuation.
    sps = n_sub + cp_len
    faded_field = np.empty_like(sent_p.field)
    amplitude = 0.8
    phase_slope = 0.05  # radians per subcarrier
    for f in range(n_frames):
        block = sent_p.field[f * sps: (f + 1) * sps]
        useful = block[cp_len:]
        spectrum = np.fft.fft(useful)
        # Apply channel in frequency domain.
        h = amplitude * np.exp(1j * phase_slope * np.arange(n_sub))
        spectrum *= h
        useful_faded = np.fft.ifft(spectrum)
        # Reconstruct with CP.
        if cp_len > 0:
            cp = useful_faded[-cp_len:]
            faded_block = np.concatenate([cp, useful_faded])
        else:
            faded_block = useful_faded
        faded_field[f * sps: (f + 1) * sps] = faded_block

    # With pilots → equalizer should recover BER == 0.
    rx_p = OFDMReceiver(modulation_order=mod, ofdm=ofdm_with_pilots)
    rcv_p = rx_p.receive(faded_field)
    assert _ber(sent_p.bits, rcv_p.bits) == 0.0

    # Without pilots on the same (faded) field → BER > 0 (channel distorts).
    # We need a Tx without pilots to get the reference bits.
    tx_np = OFDMTransmitter(
        modulation_order=mod, symbol_rate_hz=32e9, n_symbols=n_frames,
        ofdm=ofdm_no_pilots, seed=42,
    )
    sent_np = tx_np.generate()

    # Apply the same channel.
    faded_field_np = np.empty_like(sent_np.field)
    sps_np = n_sub + cp_len
    for f in range(n_frames):
        block = sent_np.field[f * sps_np: (f + 1) * sps_np]
        useful = block[cp_len:]
        spectrum = np.fft.fft(useful)
        h = amplitude * np.exp(1j * phase_slope * np.arange(n_sub))
        spectrum *= h
        useful_faded = np.fft.ifft(spectrum)
        if cp_len > 0:
            cp = useful_faded[-cp_len:]
            faded_block = np.concatenate([cp, useful_faded])
        else:
            faded_block = useful_faded
        faded_field_np[f * sps_np: (f + 1) * sps_np] = faded_block

    rx_np = OFDMReceiver(modulation_order=mod, ofdm=ofdm_no_pilots)
    rcv_np = rx_np.receive(faded_field_np)
    assert _ber(sent_np.bits, rcv_np.bits) > 0.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_output():
    """Two Tx/Rx pairs with the same seed produce identical receive results."""
    tx1, rx1 = _make_tx_rx(seed=77)
    tx2, rx2 = _make_tx_rx(seed=77)
    sent1 = tx1.generate()
    sent2 = tx2.generate()
    rcv1 = rx1.receive(sent1.field)
    rcv2 = rx2.receive(sent2.field)
    np.testing.assert_array_equal(rcv1.bits, rcv2.bits)
    np.testing.assert_allclose(rcv1.symbols, rcv2.symbols, atol=1e-15)


# ---------------------------------------------------------------------------
# ConstellationConfig path
# ---------------------------------------------------------------------------


def test_custom_constellation_b2b():
    """OFDMReceiver with ConstellationConfig (not modulation_order)."""
    cfg = ConstellationConfig.from_square_qam(16)
    tx, rx = _make_tx_rx(constellation=cfg, n_frames=8)
    sent = tx.generate()
    rcv = rx.receive(sent.field)
    assert _ber(sent.bits, rcv.bits) == 0.0
