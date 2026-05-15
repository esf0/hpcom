"""Quality metrics for received signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray


@dataclass(frozen=True)
class Metrics:
    """Per-simulation quality summary."""

    ber: float
    evm_rms: float
    n_bits: int
    n_bit_errors: int


def bit_error_rate(
    sent_bits: npt.NDArray[np.uint8],
    received_bits: npt.NDArray[np.uint8],
) -> tuple[float, int, int]:
    """Return ``(ber, n_errors, n_bits)``."""
    if sent_bits.shape != received_bits.shape:
        raise ValueError(
            f"shape mismatch: sent_bits {sent_bits.shape} vs received_bits {received_bits.shape}"
        )
    n_bits = int(sent_bits.size)
    n_errors = int(np.sum(sent_bits != received_bits))
    ber = n_errors / n_bits if n_bits else 0.0
    return ber, n_errors, n_bits


def evm_rms(received: ComplexArray, reference: ComplexArray) -> float:
    """Root-mean-square error vector magnitude, normalized to reference RMS power."""
    if received.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: received {received.shape} vs reference {reference.shape}"
        )
    err_power = float(np.mean(np.abs(received - reference) ** 2))
    ref_power = float(np.mean(np.abs(reference) ** 2))
    if ref_power == 0:
        return float("inf") if err_power > 0 else 0.0
    return float(np.sqrt(err_power / ref_power))
